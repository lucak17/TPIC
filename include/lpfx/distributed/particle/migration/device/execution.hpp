#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_EXECUTION_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_EXECUTION_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include <lpfx/algorithm/particle/device/compaction.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/device/classification.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/distributed/particle/migration/capacity.hpp>
#include <lpfx/distributed/particle/migration/device/arrivals.hpp>
#include <lpfx/distributed/particle/migration/device/kernel_suite.hpp>
#include <lpfx/distributed/particle/migration/resident_reorder.hpp>
#include <lpfx/distributed/particle/migration/routing.hpp>
#include <lpfx/distributed/particle/migration/stage.hpp>
#include <lpfx/distributed/particle/wire/storage.hpp>
#include <lpfx/distributed/plan/particle_semantics.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

#if LPFX_NATIVE_DEVICE_OPERATIONS
#include <lpfx/distributed/particle/migration/device/cuda_hip/kernel_suite.hpp>
#endif

namespace lpfx::detail {

struct DeviceParticleRegisteredRetainedClassification final {
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr bool
  retained(const std::uint64_t word) noexcept {
    return PackedDeviceParticleClassificationSlot::retained(word);
  }
};

} // namespace lpfx::detail

namespace lpfx::distributed::detail {

template <class Backend>
concept DeviceParticleMigrationBackend =
    device::DeviceAllocationBackend<Backend> &&
    device::PinnedHostAllocationBackend<Backend> &&
    device::DeviceEventBackend<Backend> && device::DeviceCopyBackend<Backend> &&
    device::DeviceStreamBackend<Backend>;

struct EmptyParticleResidentReady final {};

// Exact resident compaction of the main owner. It survives storage growth;
// only finalization rebinds the owner after resident-only growth. The ready
// event is published for external resident access.
template <class Schema, class Backend, bool PublishReady>
struct DeviceParticleResidentState final {
  using owner_type = Particles<Schema, DeviceSpace<Backend>>;
  using compactor_type = DeviceExactUnstableParticleCompaction<
      Schema, Backend, DeviceMigrationHoleFillLaunchPolicy,
      lpfx::detail::DeviceParticleRegisteredRetainedClassification>;

  compactor_type compactor{};
  [[no_unique_address]] std::conditional_t<
      PublishReady, device::DeviceEvent<Backend>, EmptyParticleResidentReady>
      ready{};
  owner_type *owner = nullptr;
  const std::uint32_t *failure = nullptr;
  std::uint32_t capacity = 0, count = 0;
  bool pending = false;

  [[nodiscard]] ErrorCode
  prepare(HostMemoryBudget &budget, owner_type &particles,
          const device::DeviceProperties &properties) noexcept {
    if (pending || !particles.valid())
      return ErrorCode::invalid_state;
    if (owner == &particles && capacity == particles.capacity()) {
      compactor.rebind_owner(particles);
      return ErrorCode::success;
    }
    PreparedDeviceUnstableHoleFillWorkspace<DeviceMigrationHoleFillLaunchPolicy>
        workspace{};
    LPFX_TRY(try_prepare_device_unstable_hole_fill_workspace(
        particles.capacity(), properties, workspace));
    auto result = make_device_exact_unstable_particle_compaction<
        Schema, Backend, DeviceMigrationHoleFillLaunchPolicy,
        lpfx::detail::DeviceParticleRegisteredRetainedClassification>(
        budget, workspace, particles);
    LPFX_TRY(result);
    if constexpr (PublishReady) {
      if (ready.empty() && !device::BackendTraits<Backend>::succeeded(
                               device::DeviceEvent<Backend>::create(ready)))
        return ErrorCode::backend_failure;
    }
    // Transfer only dead list capacity, after every fallible step.
    if (compactor.valid())
      result.value().reuse_lists_from(compactor);
    std::destroy_at(&compactor);
    std::construct_at(&compactor, std::move(result).value());
    owner = &particles;
    capacity = particles.capacity();
    return ErrorCode::success;
  }

  [[nodiscard]] Status
  submit(const std::uint64_t *const decisions, const std::uint32_t retained,
         const std::uint32_t source,
         const device::Executor<Backend> executor) noexcept {
    if (pending || owner == nullptr)
      return Status::failure(ErrorCode::invalid_state);
    count = retained;
    failure = nullptr;
    if (retained == 0 || retained == source) {
      if (retained == 0)
        lpfx::detail::DeviceParticlesAccess::publish_size(*owner, 0);
      if constexpr (PublishReady)
        device::abort_on_failure<Backend>(ready.record(executor));
      return Status::success();
    }
    const Status submitted = [&] {
      if constexpr (PublishReady)
        return compactor.submit(decisions, retained, executor, ready);
      else
        return compactor.submit(decisions, retained, executor);
    }();
    if (submitted.is_failure())
      return submitted;
    if (!submitted.requires_conclusion())
      std::abort();
    failure = compactor.failure_word();
    pending = true;
    return submitted;
  }

  template <bool Wait> [[nodiscard]] Status advance() noexcept {
    if (!pending)
      return Status::success();
    const Status completed = Wait ? compactor.wait() : compactor.conclude();
    if (completed.is_pending())
      return completed;
    if (!completed.is_success() || completed.requires_conclusion() ||
        owner->size() != count)
      std::abort();
    pending = false;
    return Status::success();
  }
};

// Routing storage and kernels of one migration. Initial packing reads the
// main owner on the preparation stream; afterwards routing reads only its
// wire images and received-classification slots, stages final arrivals in
// auxiliary chunks and repacks transit records. The resident compaction is
// owned by the program and never touched here. Growth replaces only the wire
// storage, in place.
template <lpfx::detail::ParticleWireContractType Contract,
          lpfx::detail::CartesianRank Rank, class Owner, class Executor,
          class Semantics>
  requires DeviceParticleMigrationBackend<typename Executor::backend_type>
class DeviceParticleExecution final {
public:
  using contract_type = Contract;
  using schema_type = typename Contract::schema_type;
  using owner_type = Owner;
  using executor_type = Executor;
  using semantics_type = Semantics;
  using backend_type = typename Executor::backend_type;
  using traits_type = device::BackendTraits<backend_type>;
  using routing_type =
      PreparedParticleRoutingPattern<Rank, typename Contract::hop_policy>;
  using capacity_type = PreparedParticleMigrationCapacity<Contract>;
  using epoch_semantics_type = std::remove_cvref_t<
      decltype(std::declval<const Semantics &>().device_view())>;
  using protocol_type = ParticleMigrationProtocol<
      routing_type, typename epoch_semantics_type::initial_owner_type,
      typename epoch_semantics_type::received_owner_type>;
  using wire_storage_type =
      DeviceParticleMigrationWireStorage<Contract, backend_type>;
  struct storage_type final {
    wire_storage_type wire;
  };
  using outgoing_prefixes_type = PreparedParticleOutgoingPrefixes<Contract>;
  using incoming_prefixes_type = PreparedParticleIncomingPrefixes<Contract>;
  using arrival_storage_type =
      lpfx::detail::DeviceParticleArrivals<schema_type, backend_type>;
  using resident_state_type =
      DeviceParticleResidentState<schema_type, backend_type,
                                  has_external_resident_access<Semantics>>;
  static constexpr bool registered_initial_classification =
      particle_classification_pipeline_v<Semantics> ==
      ParticleClassificationPipeline::user_flags;

  DeviceParticleExecution() noexcept = default;
  LPFX_NONMOVABLE(DeviceParticleExecution);
  ~DeviceParticleExecution() noexcept {
    if (pending()) {
      std::abort();
    }
  }

  // Wire storage for capacity. Every classification grid (the owner or one
  // incoming batch) must fit the device launch limits.
  [[nodiscard]] static ErrorCode
  try_prepare_storage(HostMemoryBudget &, const capacity_type &capacity,
                      const std::size_t owner_capacity,
                      const device::DeviceProperties &properties,
                      storage_type &output) noexcept {
    std::uint32_t owner_extent{};
    std::uint32_t incoming_extent{};
    if (!lpfx::detail::checked_integral_cast(owner_capacity, owner_extent) ||
        !lpfx::detail::checked_integral_cast(capacity.maximum_incoming(),
                                             incoming_extent)) {
      return ErrorCode::overflow;
    }
    const std::uint32_t extent =
        owner_extent > incoming_extent ? owner_extent : incoming_extent;
    constexpr std::uint32_t block_threads =
        device_particle_migration_block_threads;
    const std::uint32_t block_count =
        extent / block_threads +
        static_cast<std::uint32_t>(extent % block_threads != 0);
    if (properties.maximum_threads_per_block < block_threads ||
        properties.maximum_grid_size_x < block_count) {
      return ErrorCode::invalid_argument;
    }
    PreparedParticleMigrationWireStorage<Contract> prepared{};
    LPFX_TRY(try_prepare_particle_migration_wire_storage(capacity, prepared));
    auto wire = make_device_particle_migration_wire_storage<backend_type>(
        prepared);
    LPFX_TRY(wire);
    output.wire = std::move(wire).value();
    return ErrorCode::success;
  }

  template <class Pattern>
  [[nodiscard]] ErrorCode prepare(HostMemoryBudget &budget,
                                  const Pattern &pattern, owner_type &owner,
                                  storage_type &&storage) noexcept {
    HostMemoryRequirement requirement{};
    if (!requirement.try_add(sizeof(DeviceParticleMigrationSummary),
                             alignof(DeviceParticleMigrationSummary))) {
      return ErrorCode::overflow;
    }
    auto admission = budget.try_admit(requirement);
    LPFX_TRY(admission);
    device::DeviceAllocation<backend_type> scratch{};
    if (!traits_type::succeeded(device::DeviceAllocation<backend_type>::create(
            sizeof(DeviceParticleMigrationSummary), scratch))) {
      return ErrorCode::allocation_failed;
    }
    device::PinnedHostAllocation<backend_type> readback{};
    if (!traits_type::succeeded(
            device::PinnedHostAllocation<backend_type>::create(
                sizeof(DeviceParticleMigrationSummary), readback))) {
      return ErrorCode::allocation_failed;
    }
    device::DeviceEvent<backend_type> event{};
    if (!traits_type::succeeded(
            device::DeviceEvent<backend_type>::create(event))) {
      return ErrorCode::backend_failure;
    }
    admission_ = std::move(admission).value();
    scratch_ = std::move(scratch);
    readback_ = std::move(readback);
    event_ = std::move(event);
    device_summary_ =
        reinterpret_cast<DeviceParticleMigrationSummary *>(scratch_.raw_base());
    host_summary_ = std::construct_at(
        reinterpret_cast<DeviceParticleMigrationSummary *>(
            readback_.raw_base()),
        DeviceParticleMigrationSummary{});
    routing_ = &pattern.routing();
    capacity_ = &pattern.capacity();
    particles_ = &owner;
    host_budget_ = &budget;
    bound_particle_capacity_ = owner.capacity();
    wire_ = std::move(storage.wire);
    return ErrorCode::success;
  }

  // Replaces quiescent wire storage between epochs.
  void adopt(storage_type &&storage) noexcept {
    if (pending())
      std::abort();
    wire_ = std::move(storage.wire);
    bound_particle_capacity_ = particles_->capacity();
  }

  // Count-admission growth. Counts have completed and no receive kernel is
  // active. A stream-ordered payload may still run: the copy is ordered after
  // it on executor, and the native free of the old slots synchronizes the
  // device. Slot A holds the outgoing image once packed.
  [[nodiscard]] ErrorCode grow(storage_type &&storage, const bool packed,
                               const executor_type &executor) noexcept {
    if (pending())
      return ErrorCode::invalid_state;
    const auto bytes = static_cast<std::size_t>(outgoing_.total_bytes());
    if (packed && bytes != 0) {
      if (bytes > storage.wire.slot_a_capacity().size())
        return ErrorCode::capacity_exceeded;
      const auto copied = traits_type::copy_device_to_device_async(
          storage.wire.slot_a_capacity().data(),
          std::as_const(wire_).slot_a_capacity().data(), bytes,
          executor.stream());
      // The wait also drains a successful copy before the old slot is freed.
      if (!traits_type::succeeded(event_.record(executor)) ||
          !traits_type::succeeded(event_.synchronize()))
        std::abort();
      if (!traits_type::succeeded(copied))
        return ErrorCode::backend_failure;
    }
    wire_ = std::move(storage.wire);
    return ErrorCode::success;
  }

  // The slot lease owns capacity, producer ordering and lifetime; binding
  // changes only at a quiescent phase boundary.
  void bind_classification(
      const DeviceParticleRegisteredClassificationView<backend_type>
          view) noexcept {
    if (pending())
      std::abort();
    classification_ = view;
  }

  // Positions classify each source once and register its final route;
  // user flags were registered by the producer. Both admit only metadata.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_initial_classification(
      const protocol_type &protocol, const executor_type executor) noexcept {
    const std::uint32_t source_count = particles_->size();
    if (!classification_.valid() || classification_.size() != source_count ||
        classification_.peer_count() != routing_->peer_count())
      return Status::failure(ErrorCode::invalid_argument);
    if (!arrivals_.empty() || append_pending_)
      return Status::failure(ErrorCode::invalid_state);
    arrivals_appended_ = false;
    clear_published_classification();
    pending_source_count_ = source_count;
    if constexpr (!registered_initial_classification) {
      device::abort_on_failure<backend_type>(
          suite::template enqueue_initial_classification<schema_type>(
              source_count,
              lpfx::detail::DeviceParticlesAccess::const_prefix_view(
                  *particles_, source_count),
              classification_, protocol, executor.stream()));
    }
    device::abort_on_failure<backend_type>(
        suite::enqueue_registered_initial_admission(
            source_count, routing_->peer_count(), classification_.summary(),
            classification_.offsets(), device_summary_, executor.stream()));
    lpfx::detail::copy_and_record<lpfx::detail::DeviceCopyDirection::to_host>(
        host_summary_, device_summary_, sizeof(DeviceParticleMigrationSummary),
        executor, event_);
    pending_ = true;
    return Status::accepted();
  }

  // Direct writes to the unique sub-bucket ordinals of the classification.
  // Only reached after publish_classification read back failure = none from
  // the admission summary on this stream; nothing writes the device word in
  // between (growth replaces wire storage only), so it needs no reset.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_initial_pack(const epoch_semantics_type &semantics,
                     const executor_type executor) noexcept {
    const auto outgoing = static_cast<std::uint32_t>(outgoing_.total_particles());
    if (outgoing == 0)
      return Status::success();
    device::abort_on_failure<backend_type>(
        suite::template enqueue_registered_initial_pack<Contract>(
            published_source_count_, classification_.data(),
            classification_.offsets(),
            lpfx::detail::DeviceParticlesAccess::const_prefix_view(
                *particles_, published_source_count_),
            DeviceParticlePeerWireImageView<Contract>{
                wire_.template image<ParticleMigrationWireSlot::a>(outgoing)
                    .base(),
                outgoing_},
            &device_summary_->failure, semantics.codec_context(),
            executor.stream()));
    read_failure(executor);
    return Status::accepted();
  }

  // Consumes the round after the slot-A send and the incoming slot-B prefix
  // completed or were ordered before it on executor.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_received_classification(
      const std::uint64_t round, const incoming_prefixes_type &incoming,
      const protocol_type &protocol, const executor_type executor) noexcept {
    const auto incoming_count =
        static_cast<std::uint32_t>(incoming.total_particles());
    if (!classification_.valid() || classification_.size() != incoming_count ||
        classification_.peer_count() != routing_->peer_count())
      return Status::failure(ErrorCode::invalid_argument);
    incoming_count_ = incoming_count;
    incoming_ = ConstDeviceParticlePeerWireImageView<Contract>{
        std::as_const(wire_)
            .template image<ParticleMigrationWireSlot::b>(incoming_count)
            .base(),
        incoming};
    if (incoming_count == 0)
      return publish_received_classification(DeviceParticleMigrationSummary{});
    device::abort_on_failure<backend_type>(
        suite::template enqueue_received_classification<Contract>(
            round, incoming_count, routing_->peer_count(), incoming_,
            classification_, device_summary_, protocol, executor.stream()));
    lpfx::detail::copy_and_record<lpfx::detail::DeviceCopyDirection::to_host>(
        host_summary_, device_summary_, sizeof(DeviceParticleMigrationSummary),
        executor, event_);
    pending_ = true;
    return Status::accepted();
  }

  // Decodes final records into a retained arrival chunk and copies transit
  // records from slot B into slot A after the slot-A send, which completed
  // or is ordered before it on executor.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_received_processing(const incoming_prefixes_type &,
                            const epoch_semantics_type &semantics,
                            const executor_type executor) noexcept {
    const auto transit = static_cast<std::uint32_t>(outgoing_.total_particles());
    if (received_final_count_ == 0 && transit == 0) {
      complete_received_processing();
      return Status::success();
    }
    typename owner_type::view_type final_destination;
    const ErrorCode allocation = arrivals_.allocate(
        *host_budget_, received_final_count_, final_destination);
    if (allocation != ErrorCode::success)
      return Status::failure(allocation);
    device::abort_on_failure<backend_type>(
        suite::template enqueue_received_processing<Contract>(
            incoming_count_, classification_.data(), classification_.offsets(),
            incoming_, final_destination,
            DeviceParticlePeerWireImageView<Contract>{
                wire_.template image<ParticleMigrationWireSlot::a>(transit)
                    .base(),
                outgoing_},
            &device_summary_->failure, semantics.codec_context(),
            executor.stream()));
    read_failure(executor);
    return Status::accepted();
  }

  template <ParticleExecutionStage Stage, bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() noexcept {
    if (!pending_)
      return Status::success();
    const Status completed = event_.template advance<Wait>();
    if (completed.is_pending())
      return completed;
    if (completed.is_failure())
      std::abort();
    pending_ = false;
    if constexpr (Stage == ParticleExecutionStage::initial_classification) {
      return publish_classification();
    } else if constexpr (Stage == ParticleExecutionStage::initial_pack) {
      return failure_status(
          lpfx::detail::DeviceParticleMigrationFailureWord::write_failure);
    } else if constexpr (Stage ==
                         ParticleExecutionStage::received_classification) {
      return publish_received_classification(*host_summary_);
    } else {
      static_assert(Stage == ParticleExecutionStage::received_processing);
      const Status status = failure_status(
          lpfx::detail::DeviceParticleMigrationFailureWord::write_failure);
      if (status.is_success())
        complete_received_processing();
      return status;
    }
  }

  // Main storage is frozen through communication. destination is the owner's
  // capacity view after any resident-only growth.
  [[nodiscard]] Status
  submit_append_arrivals(const typename owner_type::view_type destination,
                         const executor_type executor) noexcept {
    std::uint32_t final_size{};
    if (pending() || arrivals_appended_ ||
        !lpfx::detail::checked_add(resident_count_, cumulative_final_count_,
                                   final_size) ||
        final_size > destination.size())
      return Status::failure(ErrorCode::invalid_state);
    if (arrivals_.empty()) {
      arrivals_appended_ = true;
      return Status::success();
    }
    const auto copied = arrivals_.append(destination, resident_count_, executor);
    device::abort_on_failure<backend_type>(event_.record(executor));
    if (!traits_type::succeeded(copied)) {
      device::abort_on_failure<backend_type>(event_.synchronize());
      return Status::failure(ErrorCode::backend_failure);
    }
    append_pending_ = true;
    return Status::accepted();
  }

  template <bool Wait> [[nodiscard]] Status advance_append_arrivals() noexcept {
    if (!append_pending_)
      return arrivals_appended_ ? Status::success()
                                : Status::failure(ErrorCode::invalid_state);
    const Status completed = event_.template advance<Wait>();
    if (completed.is_pending())
      return completed;
    if (completed.is_failure())
      std::abort();
    append_pending_ = false;
    arrivals_appended_ = true;
    arrivals_.clear();
    return Status::success();
  }

  // Publishes residents plus every decoded final exactly once. The program
  // proved the final transit image empty.
  void publish_completed_migration() noexcept {
    std::uint32_t completed_size{};
    if (!lpfx::detail::checked_add(resident_count_, cumulative_final_count_,
                                   completed_size) ||
        completed_size > bound_particle_capacity_ || !arrivals_appended_ ||
        append_pending_ || !arrivals_.empty()) {
      std::abort();
    }
    lpfx::detail::DeviceParticlesAccess::publish_size(*particles_,
                                                      completed_size);
    clear_published_classification();
    pending_source_count_ = 0;
    incoming_count_ = 0;
    received_final_count_ = 0;
    next_final_prefix_ = 0;
    cumulative_final_count_ = 0;
  }

  // The owner grew in place at finalization.
  void refresh_owner_capacity() noexcept {
    if (pending())
      std::abort();
    bound_particle_capacity_ = particles_->capacity();
  }

  [[nodiscard]] bool pending() const noexcept {
    return pending_ || append_pending_;
  }
  [[nodiscard]] const std::uint64_t *initial_decisions() const noexcept {
    return classification_.data();
  }
  [[nodiscard]] std::uint32_t source_count() const noexcept {
    return published_source_count_;
  }
  [[nodiscard]] std::uint32_t resident_count() const noexcept {
    return resident_count_;
  }
  [[nodiscard]] std::uint32_t cumulative_final_count() const noexcept {
    return cumulative_final_count_;
  }
  [[nodiscard]] const outgoing_prefixes_type &
  outgoing_prefixes() const noexcept {
    return outgoing_;
  }
  [[nodiscard]] wire_storage_type &wire_storage() noexcept { return wire_; }
  [[nodiscard]] arrival_storage_type &arrival_storage() noexcept {
    return arrivals_;
  }

private:
  using suite = lpfx::detail::DeviceParticleKernelSuite<backend_type>;

  LPFX_ALWAYS_INLINE void read_failure(const executor_type executor) noexcept {
    lpfx::detail::copy_and_record<lpfx::detail::DeviceCopyDirection::to_host>(
        &host_summary_->failure, &device_summary_->failure,
        sizeof(std::uint32_t), executor, event_);
    pending_ = true;
  }

  // A zero word succeeds; the stage's recoverable word is a rejected record;
  // any other word is a broken kernel contract.
  [[nodiscard]] LPFX_ALWAYS_INLINE static Status
  failure_status(const std::uint32_t word,
                 const lpfx::detail::DeviceParticleMigrationFailureWord
                     recoverable) noexcept {
    if (word ==
        static_cast<std::uint32_t>(
            lpfx::detail::DeviceParticleMigrationFailureWord::none))
      return Status::success();
    if (word != static_cast<std::uint32_t>(recoverable))
      std::abort();
    return Status::failure(ErrorCode::invalid_argument);
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE Status failure_status(
      const lpfx::detail::DeviceParticleMigrationFailureWord recoverable)
      const noexcept {
    return failure_status(host_summary_->failure, recoverable);
  }

  // Per-peer counts of summary and their total; categories of unprepared
  // peers must be empty.
  [[nodiscard]] LPFX_ALWAYS_INLINE std::uint64_t read_peer_counts(
      const DeviceParticleMigrationSummary &summary,
      std::array<std::uint64_t, particle_migration_maximum_peer_count> &counts)
      const noexcept {
    const std::uint8_t peers = routing_->peer_count();
    std::uint64_t total{};
    for (std::uint8_t peer = 0; peer != peers; ++peer) {
      counts[peer] = summary.counts[peer];
      total += counts[peer];
    }
    for (std::size_t category = peers;
         category != particle_migration_maximum_peer_count; ++category) {
      if (summary.counts[category] != 0)
        std::abort();
    }
    return total;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status publish_classification() noexcept {
    const std::uint32_t source_count = pending_source_count_;
    pending_source_count_ = 0;
    const Status failed = failure_status(
        lpfx::detail::DeviceParticleMigrationFailureWord::invalid_record);
    if (!failed.is_success())
      return failed;
    std::array<std::uint64_t, particle_migration_maximum_peer_count> counts{};
    const std::uint32_t resident =
        host_summary_->counts[device_particle_migration_resident_bucket];
    if (read_peer_counts(*host_summary_, counts) + resident > source_count)
      std::abort();
    outgoing_prefixes_type prefixes{};
    const ErrorCode admission = try_prepare_initial_particle_outgoing_counts(
        *capacity_,
        std::span<const std::uint64_t>{counts.data(), routing_->peer_count()},
        prefixes);
    if (admission == ErrorCode::capacity_exceeded)
      return Status::failure(admission);
    if (admission != ErrorCode::success)
      std::abort();
    if (routing_->round_count() == 0 && prefixes.total_particles() != 0)
      return Status::failure(ErrorCode::invalid_argument);
    outgoing_ = prefixes;
    published_source_count_ = source_count;
    resident_count_ = resident;
    return Status::success();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status publish_received_classification(
      const DeviceParticleMigrationSummary &summary) noexcept {
    const Status failed = failure_status(
        summary.failure,
        lpfx::detail::DeviceParticleMigrationFailureWord::invalid_record);
    if (!failed.is_success())
      return failed;
    std::array<std::uint64_t, particle_migration_maximum_peer_count>
        transit_counts{};
    const std::uint64_t transit = read_peer_counts(summary, transit_counts);
    const std::uint64_t final_count =
        summary.counts[device_particle_migration_resident_bucket];
    if (transit + final_count > incoming_count_)
      std::abort();
    std::uint64_t next_final_prefix{};
    ErrorCode error = lpfx::detail::try_admit_particle_received_split(
        *capacity_, incoming_count_, final_count, transit,
        incoming_count_ - transit - final_count, cumulative_final_count_, 0,
        UINT32_MAX, next_final_prefix);
    outgoing_prefixes_type transit_prefixes{};
    if (error == ErrorCode::success)
      error = try_admit_transit_particle_outgoing(
          *capacity_,
          std::span<const std::uint64_t>{transit_counts.data(),
                                         routing_->peer_count()},
          transit, transit_prefixes);
    if (error != ErrorCode::success)
      return Status::failure(error);
    outgoing_ = transit_prefixes;
    received_final_count_ = static_cast<std::uint32_t>(final_count);
    next_final_prefix_ = static_cast<std::uint32_t>(next_final_prefix);
    return Status::success();
  }

  // Routing may finish before resident compaction publishes the owner size;
  // only finalization inspects the owner.
  LPFX_ALWAYS_INLINE void complete_received_processing() noexcept {
    if (next_final_prefix_ < cumulative_final_count_)
      std::abort();
    cumulative_final_count_ = next_final_prefix_;
    incoming_count_ = 0;
    received_final_count_ = 0;
  }

  LPFX_ALWAYS_INLINE void clear_published_classification() noexcept {
    outgoing_ = {};
    published_source_count_ = 0;
    resident_count_ = 0;
  }

  HostMemoryLease admission_{};
  device::DeviceAllocation<backend_type> scratch_{};
  device::PinnedHostAllocation<backend_type> readback_{};
  device::DeviceEvent<backend_type> event_{};
  wire_storage_type wire_{};
  arrival_storage_type arrivals_{};
  DeviceParticleRegisteredClassificationView<backend_type> classification_{};
  outgoing_prefixes_type outgoing_{};
  ConstDeviceParticlePeerWireImageView<Contract> incoming_{};
  DeviceParticleMigrationSummary *device_summary_ = nullptr;
  DeviceParticleMigrationSummary *host_summary_ = nullptr;
  const routing_type *routing_ = nullptr;
  const capacity_type *capacity_ = nullptr;
  owner_type *particles_ = nullptr;
  HostMemoryBudget *host_budget_ = nullptr;
  std::uint32_t bound_particle_capacity_ = 0;
  std::uint32_t pending_source_count_ = 0;
  std::uint32_t published_source_count_ = 0;
  std::uint32_t resident_count_ = 0;
  std::uint32_t incoming_count_ = 0;
  std::uint32_t received_final_count_ = 0;
  std::uint32_t next_final_prefix_ = 0;
  std::uint32_t cumulative_final_count_ = 0;
  bool pending_ = false;
  bool append_pending_ = false;
  bool arrivals_appended_ = false;
};

} // namespace lpfx::distributed::detail

#endif
