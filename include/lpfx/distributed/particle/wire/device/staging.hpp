#ifndef LPFX_DISTRIBUTED_PARTICLE_WIRE_DEVICE_STAGING_HPP
#define LPFX_DISTRIBUTED_PARTICLE_WIRE_DEVICE_STAGING_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/distributed/particle/wire/storage.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {

namespace detail {

enum class DeviceParticleMigrationWireStagingPhase : std::uint8_t {
  unprepared,
  idle,
  pending_outgoing_download,
  outgoing_host_ready,
  host_transport_ready,
  incoming_enqueued
};

template <class Backend>
concept DeviceParticleMigrationWireStagingBackend =
    device::PinnedHostAllocationBackend<Backend> &&
    device::DeviceEventBackend<Backend> && device::DeviceCopyBackend<Backend>;

} // namespace detail

// Persistent pinned bridge for the two device migration wire slots. The
// phase sequence is the slot-lifetime proof:
//
//   idle -> outgoing download -> outgoing host ready -> admit incoming
//        -> host transport -> incoming upload enqueued -> recycle -> idle
//
// Count exchange may overlap the outgoing D2H copy because the incoming count
// is admitted only after that copy completes. enqueue_incoming_upload() is the
// caller's proof that native host transport no longer references either
// pinned slot; the upload is ordered on the consumer's stream and completes
// with the consumer. recycle() is the proof that device processing no longer
// references slot B and has finished writing the next slot-A image.
template <detail::ParticleWireContractType Contract,
          detail::DeviceParticleMigrationWireStagingBackend Backend>
class DeviceParticleMigrationWireStaging final {
  using Phase = detail::DeviceParticleMigrationWireStagingPhase;

public:
  using contract_type = Contract;
  using backend_type = Backend;
  using wire_storage_type =
      DeviceParticleMigrationWireStorage<Contract, Backend>;
  using staging_allocation_type = device::AlignedPinnedHostAllocation<
      Backend, PreparedParticleMigrationWireStorage<Contract>::slot_alignment>;

  constexpr DeviceParticleMigrationWireStaging() noexcept = default;

  DeviceParticleMigrationWireStaging(
      DeviceParticleMigrationWireStaging &&other) noexcept
      : staging_(std::move(other.staging_)), event_(std::move(other.event_)),
        prepared_(std::exchange(other.prepared_, {})),
        outgoing_count_(std::exchange(other.outgoing_count_, 0)),
        incoming_count_(std::exchange(other.incoming_count_, 0)),
        phase_(std::exchange(other.phase_, Phase::unprepared)) {}

  DeviceParticleMigrationWireStaging &
  operator=(DeviceParticleMigrationWireStaging &&) = delete;

  ~DeviceParticleMigrationWireStaging() noexcept {
    if (phase_ != Phase::unprepared && phase_ != Phase::idle) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return phase_ != Phase::unprepared;
  }
  [[nodiscard]] constexpr bool idle() const noexcept {
    return phase_ == Phase::idle;
  }

  // Hard precondition: the outgoing download has completed.
  [[nodiscard]] BufferView<const std::byte, HostSpace>
  host_outgoing_image() const noexcept {
    const std::size_t bytes = active_bytes(outgoing_count_);
    return detail::make_buffer_view<const std::byte>(
        bytes == 0 ? nullptr : staging_.raw_base(), bytes);
  }

  // Hard precondition: admit_incoming() succeeded. The caller holds exclusive
  // host access until enqueue_incoming_upload().
  [[nodiscard]] BufferView<std::byte, HostSpace>
  host_incoming_image() noexcept {
    const std::size_t bytes = active_bytes(incoming_count_);
    return detail::make_buffer_view(
        bytes == 0 ? nullptr : staging_.raw_base() + prepared_.slot_b_offset(),
        bytes);
  }

  // Starts one contiguous active-byte D2H copy of wire slot A. A zero-byte
  // image is ready synchronously and needs no executor.
  [[nodiscard]] Status
  submit_outgoing_download(const wire_storage_type &wire,
                           const std::uint64_t outgoing_count,
                           const device::Executor<Backend> executor) noexcept {
    if (phase_ != Phase::idle) {
      return Status::failure(ErrorCode::invalid_state);
    }
    std::uint32_t count{};
    std::size_t bytes{};
    const ErrorCode admission =
        admit_count(outgoing_count, prepared_.slot_a_bytes(), count, bytes);
    if (admission != ErrorCode::success) {
      return Status::failure(admission);
    }
    if (bytes == 0) {
      outgoing_count_ = count;
      phase_ = Phase::outgoing_host_ready;
      return Status::success();
    }
    if (!executor.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    detail::copy_and_record<detail::DeviceCopyDirection::to_host>(
        staging_.raw_base(),
        wire.template image<ParticleMigrationWireSlot::a>(count).base(), bytes,
        executor, event_);
    outgoing_count_ = count;
    phase_ = Phase::pending_outgoing_download;
    return Status::accepted();
  }

  template <bool Wait> [[nodiscard]] Status advance_outgoing() noexcept {
    if (phase_ == Phase::outgoing_host_ready) {
      return Status::success();
    }
    if (phase_ != Phase::pending_outgoing_download) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const Status result = event_.template advance<Wait>();
    if (result.is_failure()) {
      std::abort();
    }
    if (result.is_pending()) {
      return result;
    }
    phase_ = Phase::outgoing_host_ready;
    return Status::success();
  }

  // Admits the exchanged incoming count after the outgoing copy completes.
  // Failure keeps the outgoing image so corrected counts can be retried.
  [[nodiscard]] Status
  admit_incoming(const std::uint64_t incoming_count) noexcept {
    if (phase_ != Phase::outgoing_host_ready) {
      return Status::failure(ErrorCode::invalid_state);
    }
    std::uint32_t count{};
    std::size_t bytes{};
    const ErrorCode admission =
        admit_count(incoming_count, prepared_.slot_b_bytes(), count, bytes);
    if (admission != ErrorCode::success) {
      return Status::failure(admission);
    }
    incoming_count_ = count;
    phase_ = Phase::host_transport_ready;
    return Status::success();
  }

  // Releases the host slots and enqueues one active-byte H2D copy into wire
  // slot B on the consumer's executor, which orders its consumer and
  // completion event after the upload. A zero-byte image needs no executor.
  [[nodiscard]] Status
  enqueue_incoming_upload(wire_storage_type &wire,
                          const device::Executor<Backend> executor) noexcept {
    using Traits = device::BackendTraits<Backend>;
    if (phase_ != Phase::host_transport_ready) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const std::size_t bytes = active_bytes(incoming_count_);
    if (bytes == 0) {
      phase_ = Phase::incoming_enqueued;
      return Status::success();
    }
    if (!executor.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if (!Traits::succeeded(Traits::copy_host_to_device_async(
            wire.template image<ParticleMigrationWireSlot::b>(incoming_count_)
                .base(),
            staging_.raw_base() + prepared_.slot_b_offset(), bytes,
            executor.stream()))) {
      std::abort();
    }
    phase_ = Phase::incoming_enqueued;
    return Status::accepted();
  }

  // Only this transition permits the next outgoing D2H submission.
  [[nodiscard]] Status recycle() noexcept {
    if (phase_ != Phase::incoming_enqueued) {
      return Status::failure(ErrorCode::invalid_state);
    }
    outgoing_count_ = 0;
    incoming_count_ = 0;
    phase_ = Phase::idle;
    return Status::success();
  }

private:
  DeviceParticleMigrationWireStaging(
      staging_allocation_type &&staging, device::DeviceEvent<Backend> &&event,
      const PreparedParticleMigrationWireStorage<Contract> &prepared) noexcept
      : staging_(std::move(staging)), event_(std::move(event)),
        prepared_(prepared), phase_(Phase::idle) {}

  // Covers every packet of an image of count records (image.hpp).
  [[nodiscard]] static constexpr std::size_t
  active_bytes(const std::uint32_t count) noexcept {
    return static_cast<std::size_t>(
        detail::particle_wire_extent_bytes<Contract>(count));
  }

  [[nodiscard]] static constexpr ErrorCode
  admit_count(const std::uint64_t requested, const std::size_t capacity_bytes,
              std::uint32_t &count, std::size_t &bytes) noexcept {
    std::uint32_t candidate{};
    if (!detail::checked_integral_cast(requested, candidate)) {
      return ErrorCode::overflow;
    }
    std::size_t candidate_bytes{};
    if (!detail::try_particle_wire_extent_bytes<Contract>(
            static_cast<std::size_t>(candidate), candidate_bytes)) {
      return ErrorCode::overflow;
    }
    if (candidate_bytes > capacity_bytes) {
      return ErrorCode::capacity_exceeded;
    }
    count = candidate;
    bytes = candidate_bytes;
    return ErrorCode::success;
  }

  // The allocation returns its budget lease after the native free.
  staging_allocation_type staging_{};
  device::DeviceEvent<Backend> event_{};
  PreparedParticleMigrationWireStorage<Contract> prepared_{};
  std::uint32_t outgoing_count_{};
  std::uint32_t incoming_count_{};
  Phase phase_ = Phase::unprepared;

  friend struct detail::PrivateConstructionAccess;
};

// Pinned staging for a prepared wire geometry; the device wire storage is
// passed to each transfer.
template <detail::ParticleWireContractType Contract,
          detail::DeviceParticleMigrationWireStagingBackend Backend>
[[nodiscard]] Result<DeviceParticleMigrationWireStaging<Contract, Backend>>
make_device_particle_migration_wire_staging(
    HostMemoryBudget &budget,
    const PreparedParticleMigrationWireStorage<Contract> &prepared) noexcept {
  using Staging = DeviceParticleMigrationWireStaging<Contract, Backend>;
  using Output = Result<Staging>;
  using Allocation = typename Staging::staging_allocation_type;
  if (!prepared.valid()) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  HostMemoryRequirement requirement{};
  std::size_t allocation_bytes = 0;
  if (!Allocation::try_allocation_bytes(prepared.total_bytes(),
                                        allocation_bytes) ||
      !requirement.try_add(allocation_bytes, Allocation::alignment())) {
    return Output::from_error(ErrorCode::overflow);
  }
  auto admission = budget.try_admit(requirement);
  LPFX_TRY_RESULT(admission, Output);
  auto staging =
      Allocation::allocate(std::move(admission).value(), prepared.total_bytes());
  LPFX_TRY_RESULT(staging, Output);

  device::DeviceEvent<Backend> event{};
  if (prepared.slot_a_bytes() != 0 &&
      !device::BackendTraits<Backend>::succeeded(
          device::DeviceEvent<Backend>::create(event))) {
    return Output::from_error(ErrorCode::backend_failure);
  }
  return Output::from_value(detail::PrivateConstructionAccess::create<Staging>(
      std::move(staging).value(), std::move(event), prepared));
}

} // namespace lpfx

#endif
