#ifndef LPFX_DATA_PARTICLE_DEVICE_TRANSFER_HPP
#define LPFX_DATA_PARTICLE_DEVICE_TRANSFER_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/fixed_array.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/data/particle/host/storage.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {

namespace detail {

template <class Backend, DeviceCopyDirection Direction>
concept DeviceParticleTransferBackend =
    device::PinnedHostAllocationBackend<Backend> &&
    device::DeviceEventBackend<Backend> && device::DeviceCopyBackend<Backend> &&
    (Direction == DeviceCopyDirection::to_host ||
     device::DeviceAllocationBackend<Backend>);

} // namespace detail

// Persistent pinned staging between one bound owner and the other memory
// space. to_device uploads a host view into a device owner; to_host
// downloads a device view into a host owner. Each epoch copies only the active
// prefix of every SoA attribute and publishes the destination count after the
// reusable event completes; a download does not touch the host destination
// before then. The bound owner must remain alive at its original address and
// must not be moved, reset, or structurally mutated outside this transfer
// while it exists; an upload rejects a destination whose allocation changed.
// A submitted device view's allocation must stay alive and unmodified by
// unordered work until its epoch is concluded. Runtime calls and executor
// streams must remain on the device allocation's owning device/context.
template <detail::ParticleSchemaType Schema, class Backend,
          detail::DeviceCopyDirection Direction>
  requires detail::DeviceParticleTransferBackend<Backend, Direction>
class DeviceParticleTransfer final {
public:
  static constexpr bool upload =
      Direction == detail::DeviceCopyDirection::to_device;

  using schema_type = Schema;
  using backend_type = Backend;
  using traits_type = device::BackendTraits<Backend>;
  using source_view_type =
      std::conditional_t<upload, BasicParticleView<Schema, HostSpace, true>,
                         BasicParticleView<Schema, DeviceSpace<Backend>, true>>;
  using destination_type =
      std::conditional_t<upload, Particles<Schema, DeviceSpace<Backend>>,
                         Particles<Schema, HostSpace>>;
  static constexpr std::size_t attribute_count = Schema::attribute_count;

  constexpr DeviceParticleTransfer() noexcept = default;

  DeviceParticleTransfer(DeviceParticleTransfer &&other) noexcept
      : admission_(std::move(other.admission_)),
        staging_(std::move(other.staging_)), event_(std::move(other.event_)),
        state_(std::exchange(other.state_, {})) {}

  DeviceParticleTransfer &operator=(DeviceParticleTransfer &&) = delete;

  ~DeviceParticleTransfer() noexcept {
    if (pending()) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return state_.phase != Phase::unprepared;
  }

  [[nodiscard]] constexpr bool pending() const noexcept {
    return state_.phase == Phase::pending;
  }

  [[nodiscard]] std::size_t staging_capacity_bytes() const noexcept {
    return staging_.capacity_bytes();
  }

  [[nodiscard]] Status
  submit(const source_view_type source,
         const device::Executor<Backend> executor) noexcept {
    if (state_.phase != Phase::idle) {
      return Status::failure(ErrorCode::invalid_state);
    }
    destination_type &destination = *state_.destination;
    if constexpr (upload) {
      if (!state_.owner.matches(destination)) {
        return Status::failure(ErrorCode::invalid_state);
      }
    }
    if (source.size() > destination.capacity()) {
      return Status::failure(ErrorCode::capacity_exceeded);
    }

    const auto count = static_cast<std::uint32_t>(source.size());
    if (count == 0) {
      if constexpr (upload) {
        detail::DeviceParticlesAccess::publish_size(destination, 0);
      } else {
        destination.clear();
      }
      return Status::success();
    }
    if (!executor.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }

    const auto &staging = state_.staging_bases;
    const auto stream = executor.stream();
    ErrorCode copied = ErrorCode::success;
    if constexpr (upload) {
      // Slot-aware sorting can publish different attribute bases between
      // epochs. Capture the current mapping once, after allocation admission.
      const auto device_view =
          detail::DeviceParticlesAccess::capacity_view(destination);
      for_each_attribute<Schema>([&](const auto attribute) noexcept {
        constexpr std::size_t I = decltype(attribute)::value;
        static_cast<void>(
            std::memcpy(staging[I], source.template get<I>().raw_base(),
                        static_cast<std::size_t>(count) * Schema::widths[I]));
      });
      copied = for_each_attribute<Schema>([&](const auto attribute) noexcept {
        constexpr std::size_t I = decltype(attribute)::value;
        return device::to_error_code<Backend>(
            traits_type::copy_host_to_device_async(
                device_view.template get<I>(), staging[I],
                static_cast<std::size_t>(count) * Schema::widths[I], stream));
      });
    } else {
      copied = for_each_attribute<Schema>([&](const auto attribute) noexcept {
        constexpr std::size_t I = decltype(attribute)::value;
        return device::to_error_code<Backend>(
            traits_type::copy_device_to_host_async(
                staging[I], source.template get<I>(),
                static_cast<std::size_t>(count) * Schema::widths[I], stream));
      });
    }
    if (copied != ErrorCode::success ||
        !traits_type::succeeded(event_.record(executor))) {
      std::abort();
    }
    state_.pending_size = count;
    state_.phase = Phase::pending;
    return Status::accepted();
  }

  [[nodiscard]] Status conclude() noexcept { return advance<false>(); }

  // Explicit blocking conclusion; no submission path calls it implicitly.
  [[nodiscard]] Status wait() noexcept { return advance<true>(); }

private:
  enum class Phase : std::uint8_t { unprepared, idle, pending };

  using staging_bases_type =
      detail::DeviceFixedArray<std::byte *, attribute_count>;

  struct State {
    staging_bases_type staging_bases{};
    destination_type *destination = nullptr;
    // Allocation identity of an upload destination at preparation.
    DeviceParticleOwnerStamp owner{};
    std::uint32_t pending_size = 0;
    Phase phase = Phase::unprepared;
  };

  DeviceParticleTransfer(HostMemoryLease &&admission,
                         device::PinnedHostAllocation<Backend> &&staging,
                         device::DeviceEvent<Backend> &&event,
                         const staging_bases_type &staging_bases,
                         destination_type &destination) noexcept
      : admission_(std::move(admission)), staging_(std::move(staging)),
        event_(std::move(event)),
        state_{staging_bases, &destination, {}, 0, Phase::idle} {
    if constexpr (upload) {
      state_.owner = DeviceParticleOwnerStamp::capture(destination);
    }
  }

  template <bool Wait> [[nodiscard]] Status advance() noexcept {
    if (state_.phase != Phase::pending) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const Status completed = event_.template advance<Wait>();
    if constexpr (!Wait) {
      if (completed.is_pending()) {
        return completed;
      }
    }
    if (completed.is_failure()) {
      std::abort();
    }
    destination_type &destination = *state_.destination;
    const std::uint32_t count = state_.pending_size;
    if constexpr (upload) {
      detail::DeviceParticlesAccess::publish_size(destination, count);
    } else {
      for_each_attribute<Schema>([&](const auto attribute) noexcept {
        constexpr std::size_t I = decltype(attribute)::value;
        static_cast<void>(std::memcpy(
            detail::HostParticlesAccess::raw_attribute_base<I>(destination),
            state_.staging_bases[I],
            static_cast<std::size_t>(count) * Schema::widths[I]));
      });
      detail::HostParticlesAccess::publish_initialized_size(destination,
                                                            count);
    }
    state_.pending_size = 0;
    state_.phase = Phase::idle;
    return Status::success();
  }

  // Declaration order releases the native staging allocation before its
  // accounting lease.
  HostMemoryLease admission_{};
  device::PinnedHostAllocation<Backend> staging_{};
  device::DeviceEvent<Backend> event_{};
  State state_{};

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

// Stages the destination capacity with the destination space's block layout.
// A non-null budget admits the staging bytes and must outlive the transfer.
template <DeviceCopyDirection Direction, ParticleSchemaType Schema,
          class Backend, class Destination>
[[nodiscard]] Result<DeviceParticleTransfer<Schema, Backend, Direction>>
make_device_particle_transfer(HostMemoryBudget *const budget,
                              Destination &destination) noexcept {
  using Transfer = DeviceParticleTransfer<Schema, Backend, Direction>;
  using Traits = device::BackendTraits<Backend>;
  using Space = std::conditional_t<Transfer::upload, DeviceSpace<Backend>,
                                   HostSpace>;
  if (!destination.valid()) {
    return Result<Transfer>::from_error(ErrorCode::invalid_argument);
  }
  PreparedParticleLayout<Schema, Space> layout{};
  LPFX_TRY_RESULT(try_prepare_particle_layout(destination.capacity(), layout),
                  Result<Transfer>);
  const std::size_t bytes = layout.total_bytes();

  HostMemoryLease admission{};
  if (budget != nullptr) {
    HostMemoryRequirement requirement{};
    if (!requirement.try_add(bytes, alignof(std::max_align_t))) {
      return Result<Transfer>::from_error(ErrorCode::overflow);
    }
    auto admitted = budget->try_admit(requirement);
    LPFX_TRY_RESULT(admitted, Result<Transfer>);
    admission = std::move(admitted).value();
  }
  device::PinnedHostAllocation<Backend> staging{};
  if (!Traits::succeeded(
          device::PinnedHostAllocation<Backend>::create(bytes, staging))) {
    return Result<Transfer>::from_error(ErrorCode::allocation_failed);
  }
  device::DeviceEvent<Backend> event{};
  if (bytes != 0 &&
      !Traits::succeeded(device::DeviceEvent<Backend>::create(event))) {
    return Result<Transfer>::from_error(ErrorCode::backend_failure);
  }

  const DeviceFixedArray<std::byte *, Schema::attribute_count> staging_bases{
      layout.bases(staging.raw_base())};
  return Result<Transfer>::from_value(PrivateConstructionAccess::create<Transfer>(
      std::move(admission), std::move(staging), std::move(event),
      staging_bases, destination));
}

} // namespace detail

template <detail::ParticleSchemaType Schema, class Backend>
  requires detail::DeviceParticleTransferBackend<
      Backend, detail::DeviceCopyDirection::to_device>
[[nodiscard]] Result<DeviceParticleTransfer<
    Schema, Backend, detail::DeviceCopyDirection::to_device>>
make_device_particle_upload(
    HostMemoryBudget *const budget,
    Particles<Schema, DeviceSpace<Backend>> &destination) noexcept {
  return detail::make_device_particle_transfer<
      detail::DeviceCopyDirection::to_device, Schema, Backend>(budget,
                                                               destination);
}

template <detail::ParticleSchemaType Schema, class Backend>
  requires detail::DeviceParticleTransferBackend<
      Backend, detail::DeviceCopyDirection::to_host>
[[nodiscard]] Result<
    DeviceParticleTransfer<Schema, Backend, detail::DeviceCopyDirection::to_host>>
make_device_particle_download(
    HostMemoryBudget *const budget,
    Particles<Schema, HostSpace> &destination) noexcept {
  return detail::make_device_particle_transfer<
      detail::DeviceCopyDirection::to_host, Schema, Backend>(budget,
                                                             destination);
}

} // namespace lpfx

#endif
