#ifndef LPFX_ALGORITHM_PARTICLE_DEVICE_COMPACTION_HPP
#define LPFX_ALGORITHM_PARTICLE_DEVICE_COMPACTION_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include <lpfx/algorithm/particle/device/launch.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {

// Only capacity-dependent block metadata is prepared here. Hole/filler lists
// are sized by the compactor and reused thereafter.
template <class LaunchPolicy = DefaultDeviceUnstableHoleFillLaunchPolicy>
class PreparedDeviceUnstableHoleFillWorkspace final {
public:
  using launch_policy = LaunchPolicy;
  static constexpr std::size_t allocation_alignment = 64;

  constexpr PreparedDeviceUnstableHoleFillWorkspace() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::uint32_t particle_capacity() const noexcept {
    return particle_capacity_;
  }
  [[nodiscard]] constexpr std::uint32_t block_capacity() const noexcept {
    return block_capacity_;
  }
  // Block prefixes start the workspace; unordered lists need none.
  [[nodiscard]] constexpr std::size_t block_prefix_bytes() const noexcept {
    return LaunchPolicy::unordered_lists
               ? 0
               : static_cast<std::size_t>(block_capacity_) *
                     sizeof(DeviceUnstableHoleFillBlockPrefix);
  }
  [[nodiscard]] constexpr std::size_t summary_byte_offset() const noexcept {
    return block_prefix_bytes();
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return particle_capacity_ == 0
               ? 0
               : block_prefix_bytes() + sizeof(DeviceUnstableHoleFillSummary);
  }

private:
  constexpr PreparedDeviceUnstableHoleFillWorkspace(
      const std::uint32_t particle_capacity,
      const std::uint32_t block_capacity) noexcept
      : particle_capacity_(particle_capacity), block_capacity_(block_capacity),
        valid_(true) {}

  std::uint32_t particle_capacity_{};
  std::uint32_t block_capacity_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <class LaunchPolicy = DefaultDeviceUnstableHoleFillLaunchPolicy,
          detail::checked_integer Capacity>
[[nodiscard]] constexpr ErrorCode
try_prepare_device_unstable_hole_fill_workspace(
    const Capacity requested_capacity,
    const device::DeviceProperties &properties,
    PreparedDeviceUnstableHoleFillWorkspace<LaunchPolicy> &output) noexcept {
  using Prepared = PreparedDeviceUnstableHoleFillWorkspace<LaunchPolicy>;
  std::uint32_t device_capacity{};
  if (!detail::checked_integral_cast(requested_capacity, device_capacity)) {
    return ErrorCode::overflow;
  }
  if (device_capacity == 0) {
    output = detail::PrivateConstructionAccess::create<Prepared>(
        std::uint32_t{0}, std::uint32_t{0});
    return ErrorCode::success;
  }
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  const std::uint32_t block_capacity =
      device_capacity / block_threads +
      static_cast<std::uint32_t>(device_capacity % block_threads != 0);
  if (properties.maximum_threads_per_block < block_threads ||
      properties.maximum_grid_size_x < block_capacity) {
    return ErrorCode::invalid_argument;
  }
  output = detail::PrivateConstructionAccess::create<Prepared>(device_capacity,
                                                               block_capacity);
  return ErrorCode::success;
}

// Prepared exact-count unstable compaction for one stable device owner. The
// owner object and allocation must remain alive at their original addresses
// and must not be moved, reset, or structurally mutated while bound to this
// object. While an epoch is pending, classification[0, size()) must
// remain live and unchanged on the executor's current device and no other
// operation may publish a particle size. Stream ordering and current-device
// discipline are caller preconditions. Classification scratch is never moved
// with records; rewrite it for the new particle order before reuse. The owner
// retains its old structural size until the event and device-written summary
// both report success. Hole/filler lists hold min(retained, removed) indices
// and keep their high-water capacity; a submission that does not fit grows
// them under the particle capacity policy before any kernel is accepted
// (native allocation/free may synchronize). Fitting submissions reuse all
// resources.
template <detail::ParticleSchemaType Schema,
          detail::DeviceParticleCompactionBackend Backend,
          class LaunchPolicy = DefaultDeviceUnstableHoleFillLaunchPolicy,
          class RetainedClassification =
              CanonicalDeviceRetainedClassification>
class DeviceExactUnstableParticleCompaction final {
public:
  using schema_type = Schema;
  using backend_type = Backend;
  using traits_type = device::BackendTraits<Backend>;
  using owner_type = Particles<Schema, DeviceSpace<Backend>>;
  using workspace_type = PreparedDeviceUnstableHoleFillWorkspace<LaunchPolicy>;

  constexpr DeviceExactUnstableParticleCompaction() noexcept = default;

  DeviceExactUnstableParticleCompaction(
      DeviceExactUnstableParticleCompaction &&other) noexcept
      : scratch_(std::move(other.scratch_)), lists_(std::move(other.lists_)),
        readback_(std::move(other.readback_)),
        state_(std::exchange(other.state_, {})) {}

  DeviceExactUnstableParticleCompaction &
  operator=(DeviceExactUnstableParticleCompaction &&) = delete;

  ~DeviceExactUnstableParticleCompaction() noexcept {
    if (state_.pending) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return state_.particles != nullptr;
  }
  [[nodiscard]] constexpr bool pending() const noexcept {
    return state_.pending;
  }
  [[nodiscard]] constexpr std::size_t scratch_capacity_bytes() const noexcept {
    return scratch_.capacity_bytes() + lists_.capacity_bytes();
  }
  [[nodiscard]] constexpr std::size_t
  summary_readback_capacity_bytes() const noexcept {
    return state_.device_summary == nullptr
               ? 0
               : sizeof(DeviceUnstableHoleFillSummary);
  }
  // Device-written failure word of the last accepted epoch; stable from its
  // move kernel until the next submission.
  [[nodiscard]] const std::uint32_t *failure_word() const noexcept {
    return state_.device_summary == nullptr ? nullptr
                                            : &state_.device_summary->failure;
  }

  // classification is a device pointer or trivially copyable reader indexed
  // by record, with RetainedClassification::retained(classification[i]).
  template <class ClassificationReader>
  [[nodiscard]] Status
  submit(const ClassificationReader classification,
         const std::uint32_t expected_retained_count,
         const device::Executor<Backend> executor) noexcept {
    return submit_impl<false>(classification, expected_retained_count,
                              executor, nullptr);
  }

  // As submit, and records ready once the move kernel has finalized the
  // records and the failure word, before the summary readback.
  template <class ClassificationReader>
  [[nodiscard]] Status
  submit(const ClassificationReader classification,
         const std::uint32_t expected_retained_count,
         const device::Executor<Backend> executor,
         device::DeviceEvent<Backend> &ready) noexcept {
    return submit_impl<true>(classification, expected_retained_count,
                             executor, &ready);
  }

  [[nodiscard]] Status conclude() noexcept { return finish<false>(); }

  // Waits for accepted kernels. Cold list allocation/free in submit can also
  // synchronize through the native allocator; conclude only polls the event.
  [[nodiscard]] Status wait() noexcept { return finish<true>(); }

  // Hard precondition: neither compaction is pending and every reader of the
  // source lists has completed. Transfers only spare list capacity.
  void reuse_lists_from(DeviceExactUnstableParticleCompaction &source) noexcept {
    if (!valid() || !source.valid() || pending() || source.pending())
      std::abort();
    if (source.state_.list_capacity > state_.list_capacity) {
      lists_ = std::move(source.lists_);
      state_.list_capacity = std::exchange(source.state_.list_capacity, 0);
    }
  }

  // Hard precondition: owner has this compaction's prepared capacity.
  void rebind_owner(owner_type &owner) noexcept {
    if (!valid() || pending())
      std::abort();
    state_.particles = &owner;
  }

private:
  struct State {
    DeviceUnstableHoleFillBlockPrefix *block_prefixes = nullptr;
    DeviceUnstableHoleFillSummary *device_summary = nullptr;
    owner_type *particles = nullptr;
    std::uint32_t particle_capacity = 0;
    std::size_t list_capacity = 0;
    std::uint32_t pending_size = 0;
    bool pending = false;
  };

  DeviceExactUnstableParticleCompaction(
      device::DeviceAllocation<Backend> &&scratch,
      detail::PinnedReadback<Backend> &&readback,
      const std::size_t summary_byte_offset, owner_type &particles,
      const std::uint32_t particle_capacity) noexcept
      : scratch_(std::move(scratch)), readback_(std::move(readback)) {
    std::byte *const base = scratch_.raw_base();
    if (base != nullptr) {
      state_.block_prefixes =
          reinterpret_cast<DeviceUnstableHoleFillBlockPrefix *>(base);
      state_.device_summary = reinterpret_cast<DeviceUnstableHoleFillSummary *>(
          base + summary_byte_offset);
    }
    state_.particles = &particles;
    state_.particle_capacity = particle_capacity;
  }

  template <bool PublishReady, class ClassificationReader>
  [[nodiscard]] Status
  submit_impl(const ClassificationReader classification,
              const std::uint32_t expected_retained_count,
              const device::Executor<Backend> executor,
              device::DeviceEvent<Backend> *const ready) noexcept {
    if (!valid() || state_.pending) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const std::uint32_t particle_count = state_.particles->size();
    if (particle_count > state_.particle_capacity) {
      return Status::failure(ErrorCode::capacity_exceeded);
    }
    if (expected_retained_count > particle_count) {
      return Status::failure(ErrorCode::invalid_argument);
    }
    if (particle_count == 0) {
      detail::DeviceParticlesAccess::publish_size(*state_.particles, 0);
      return Status::success();
    }
    if (!executor.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if constexpr (std::is_pointer_v<ClassificationReader>) {
      if (classification == nullptr)
        return Status::failure(ErrorCode::invalid_argument);
    }

    const std::uint32_t removed = particle_count - expected_retained_count;
    const std::uint32_t required =
        expected_retained_count < removed ? expected_retained_count : removed;
    if (required > state_.list_capacity) {
      const ErrorCode error = grow_lists(required);
      if (error != ErrorCode::success)
        return Status::failure(error);
    }
    auto *const holes = reinterpret_cast<std::uint32_t *>(lists_.raw_base());
    auto *const fillers =
        holes == nullptr ? nullptr : holes + state_.list_capacity;

    bool accepted = false;
    const typename traits_type::error_type launch_error =
        detail::DeviceUnstableParticleCompactionLaunch<Backend>::
            template enqueue<LaunchPolicy, RetainedClassification, Schema>(
                particle_count, expected_retained_count, classification,
                holes, fillers, state_.block_prefixes, state_.device_summary,
                detail::DeviceParticlesAccess::capacity_view(
                    *state_.particles),
                executor.stream(), accepted);
    // A failure after acceptance leaves neither quiescence nor the particle
    // image provable; it aborts on this cold boundary.
    const ErrorCode launched =
        device::finish_device_enqueue<Backend>(launch_error, accepted);
    if (launched != ErrorCode::success) {
      return Status::failure(launched);
    }

    // The move kernel has finalized both records and its failure word. A
    // continuation may consume those on another stream before the readback.
    if constexpr (PublishReady) {
      if (!traits_type::succeeded(ready->record(executor))) {
        std::abort();
      }
    }
    readback_.enqueue(state_.device_summary,
                      sizeof(DeviceUnstableHoleFillSummary), executor);
    state_.pending_size = expected_retained_count;
    state_.pending = true;
    return Status::accepted();
  }

  template <bool Wait> [[nodiscard]] Status finish() noexcept {
    if (!state_.pending) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const Status completed = readback_.template advance<Wait>();
    if (completed.is_pending()) {
      return completed;
    }
    if (completed.is_failure()) {
      std::abort();
    }
    DeviceUnstableHoleFillSummary summary{};
    std::memcpy(&summary, readback_.data(), sizeof(summary));
    const std::uint32_t completed_size = state_.pending_size;
    state_.pending_size = 0;
    state_.pending = false;
    if (summary.failure == static_cast<std::uint32_t>(
                               detail::DeviceUnstableHoleFillFailureWord::none)) {
      detail::DeviceParticlesAccess::publish_size(*state_.particles,
                                                  completed_size);
      return Status::success();
    }
    if (summary.failure ==
        static_cast<std::uint32_t>(
            detail::DeviceUnstableHoleFillFailureWord::invariant_mismatch)) {
      return Status::failure(ErrorCode::invalid_argument);
    }
    // The event proved quiescence, but an unknown device-written value means
    // the backing particle image cannot be trusted or safely republished.
    std::abort();
  }

  [[nodiscard]] LPFX_NOINLINE ErrorCode
  grow_lists(const std::uint32_t required) noexcept {
    // Capacity is host-side allocation metadata, independent of uint32 record
    // indices. Widen so exact UINT32_MAX owners can round spare list slots up.
    std::size_t capacity{};
    LPFX_TRY(try_prepare_particle_capacity(static_cast<std::size_t>(required),
                                           capacity));
    std::size_t list_bytes{}, bytes{};
    if (!detail::checked_byte_count<std::uint32_t>(capacity, list_bytes) ||
        !detail::checked_add(list_bytes, list_bytes, bytes))
      return ErrorCode::overflow;
    device::DeviceAllocation<Backend> candidate;
    if (!traits_type::succeeded(
            device::DeviceAllocation<Backend>::create(bytes, candidate)))
      return ErrorCode::allocation_failed;
    // Lists from the previous completed attempt are dead. Commit only after
    // allocation succeeds, with no D2D copy or copy-completion dependency.
    lists_ = std::move(candidate);
    state_.list_capacity = capacity;
    return ErrorCode::success;
  }

  device::DeviceAllocation<Backend> scratch_{};
  device::DeviceAllocation<Backend> lists_{};
  detail::PinnedReadback<Backend> readback_{};
  State state_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSchemaType Schema,
          detail::DeviceParticleCompactionBackend Backend,
          class LaunchPolicy = DefaultDeviceUnstableHoleFillLaunchPolicy,
          class RetainedClassification =
              CanonicalDeviceRetainedClassification>
[[nodiscard]] Result<DeviceExactUnstableParticleCompaction<
    Schema, Backend, LaunchPolicy, RetainedClassification>>
make_device_exact_unstable_particle_compaction(
    HostMemoryBudget &host_budget,
    const PreparedDeviceUnstableHoleFillWorkspace<LaunchPolicy> &prepared,
    Particles<Schema, DeviceSpace<Backend>> &particles) noexcept {
  using Compaction =
      DeviceExactUnstableParticleCompaction<Schema, Backend, LaunchPolicy,
                                            RetainedClassification>;
  using Output = Result<Compaction>;
  using Traits = device::BackendTraits<Backend>;

  if (!prepared.valid() || !particles.valid() ||
      prepared.particle_capacity() != particles.capacity() ||
      prepared.allocation_alignment >
          Traits::minimum_device_allocation_alignment) {
    return Output::from_error(ErrorCode::invalid_argument);
  }

  const bool has_device_work = prepared.total_bytes() != 0;
  auto readback = detail::PinnedReadback<Backend>::create(
      &host_budget, has_device_work ? sizeof(DeviceUnstableHoleFillSummary) : 0);
  LPFX_TRY_RESULT(readback, Output);

  device::DeviceAllocation<Backend> scratch{};
  if (!Traits::succeeded(device::DeviceAllocation<Backend>::create(
          prepared.total_bytes(), scratch))) {
    return Output::from_error(ErrorCode::allocation_failed);
  }

  return Output::from_value(detail::PrivateConstructionAccess::create<
                            Compaction>(std::move(scratch),
                                        std::move(readback).value(),
                                        prepared.summary_byte_offset(),
                                        particles,
                                        prepared.particle_capacity()));
}

} // namespace lpfx

#if LPFX_NATIVE_DEVICE_OPERATIONS
#include <lpfx/algorithm/particle/device/cuda_hip/compaction.hpp>
#endif

#endif
