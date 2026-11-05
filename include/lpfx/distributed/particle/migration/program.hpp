#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_PROGRAM_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_PROGRAM_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <lpfx/algorithm/particle/device/sort.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/device/classification.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/data/particle/host/storage.hpp>
#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/distributed/particle/classification_producer.hpp>
#include <lpfx/distributed/particle/migration/capacity.hpp>
#include <lpfx/distributed/particle/migration/pattern.hpp>
#include <lpfx/distributed/particle/migration/resident_reorder.hpp>
#include <lpfx/distributed/particle/migration/stage.hpp>
#include <lpfx/distributed/plan/particle_semantics.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed::detail {

template <class Semantics>
concept ParticleFlagDecoding =
    requires { typename Semantics::flag_decoder_type; } &&
    (!std::is_void_v<typename Semantics::flag_decoder_type>);

// Program state that exists only for device owners.
template <class Execution, class Space> struct DeviceOnlyState final {
  struct reservation_type final {};
};

template <class Execution, class Backend>
struct DeviceOnlyState<Execution, DeviceSpace<Backend>> final {
  using slots_type = DeviceParticleClassificationSlots<Backend>;
  // Receive slots and arrival chunks staged by reserve until rank agreement.
  struct reservation_type final {
    slots_type receive_slots;
    typename Execution::arrival_storage_type arrivals;
  };

  typename Execution::resident_state_type resident{};
  // Classification of the position pipeline; user flags borrow the
  // producer's slots.
  slots_type owned_slots{};
  slots_type receive_slots{};
  std::optional<typename Execution::executor_type> preparation_executor{};
  // The device projection of the semantics, for exactly the accepted epoch.
  std::optional<typename Execution::epoch_semantics_type> epoch_view{};
  device::DeviceProperties properties{};
  std::optional<CompletedParticlePartition> completed{};
  DeviceParticleOwnerStamp completed_owner{};
  std::uint64_t candidate_generation = 0;
  std::uint32_t resident_prefix = 0;
  bool submitted = false;
  bool complete = false;
  bool packed = false;
  bool lease_active = false;
  bool acquired = false;
  bool released = false;
  bool storage_changed = false;
};

enum class ParticleProgramPhase : std::uint8_t {
  empty,
  ready,
  initial_classification,
  initial_counts,
  initial_pack,
  round_counts,
  round_outgoing,
  round_payload,
  received_classification,
  received_processing,
  initial_admission,
  prepared,
  routing_complete,
  finalizing,
  appending,
  complete
};

// The migration state machine. Counts and payload are separate ports whose
// request batches borrow one backing, since their live intervals never
// overlap. The pattern, requests and count port keep their addresses for the
// program lifetime; growth replaces only the execution storage and the
// payload port.
template <class Execution, class CountPort, class PayloadPort>
class ParticleProgram final {
public:
  using execution_type = Execution;
  using pattern_type = typename CountPort::pattern_type;
  using contract_type = typename pattern_type::contract_type;
  using capacity_type = typename pattern_type::capacity_type;
  using owner_type = typename Execution::owner_type;
  using executor_type = typename Execution::executor_type;
  using semantics_type = typename Execution::semantics_type;
  using space_type = typename owner_type::space_type;
  using count_port_type = CountPort;
  using payload_port_type = PayloadPort;
  using request_storage_type = typename CountPort::request_storage_type;
  using phase_type = ParticleProgramPhase;
  static constexpr bool host = std::same_as<space_type, HostSpace>;
  static constexpr bool staged_migration = !host;
  static constexpr bool application_classification =
      particle_classification_pipeline_v<semantics_type> ==
      ParticleClassificationPipeline::user_flags;
  static constexpr bool external_resident_access =
      !host && has_external_resident_access<semantics_type>;
  static constexpr bool flag_decoding =
      !host && application_classification &&
      ParticleFlagDecoding<semantics_type>;
  static constexpr std::uint64_t maximum_peer_particles =
      PayloadPort::maximum_peer_particles;

  ParticleProgram() noexcept = default;
  LPFX_NONMOVABLE(ParticleProgram);

  [[nodiscard]] bool empty() const noexcept {
    return phase_ == phase_type::empty;
  }
  [[nodiscard]] bool is_valid() const noexcept {
    return phase_ >= phase_type::ready;
  }

  template <class Setup, class Binding>
  [[nodiscard]] static ErrorCode try_validate(const Setup &setup,
                                              const Binding &binding) noexcept {
    if (!binding.valid())
      return ErrorCode::invalid_argument;
    LPFX_TRY(validate_rank(setup, binding.plan().pattern()));
    return validate_affinity(setup, binding.resource(), binding.executor());
  }

  template <class Binding>
  [[nodiscard]] static bool needs_native_work(const Binding &binding) noexcept {
    const auto &routing = binding.plan().pattern().routing();
    return routing.round_count() != 0 && routing.peer_count() != 0;
  }

  template <class Session, class Setup, class Binding>
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &budget,
                                      Session &session, const Setup &setup,
                                      Binding &&binding,
                                      const int tag_base) noexcept {
    if (!empty())
      return ErrorCode::invalid_state;
    pattern_ = binding.plan().pattern();
    owner_ = &binding.resource();
    executor_.emplace(std::move(binding).take_executor());
    semantics_.emplace(std::move(binding).take_semantics());
    if constexpr (requires { semantics_->bind_classification_context(this); })
      if (semantics_->classification_context() == nullptr)
        semantics_->bind_classification_context(this);
    if constexpr (!host) {
      if (!setup.has_device_properties())
        return ErrorCode::invalid_argument;
      device_.properties = setup.device_properties();
      LPFX_TRY(device_.resident.prepare(budget, *owner_, device_.properties));
    }
    typename Execution::storage_type storage{};
    LPFX_TRY(Execution::try_prepare_storage(budget, pattern_.capacity(),
                                            owner_->capacity(), properties(),
                                            storage));
    if constexpr (host) {
      execution_.prepare(pattern_, *owner_, std::move(storage));
    } else {
      LPFX_TRY(execution_.prepare(budget, pattern_, *owner_,
                                  std::move(storage)));
    }
    std::size_t payload_requests = 0;
    LPFX_TRY(PayloadPort::request_capacity(pattern_, payload_requests));
    // The count exchange addresses every peer twice, which bounds the
    // payload requests of any later capacity as well.
    const std::size_t count_requests = CountPort::request_capacity(pattern_);
    LPFX_TRY(requests_.try_prepare_resources(
        budget, count_requests > payload_requests ? count_requests
                                                  : payload_requests));
    LPFX_TRY(count_port_.try_prepare_resources(budget, session, setup, plan(),
                                               tag_base, requests_));
    LPFX_TRY(payload_->try_prepare_resources(
        budget, session, setup, plan(), execution_.wire_storage().prepared(),
        execution_.wire_storage(), executor(), count_port_, requests_));
    owner_capacity_ = owner_->capacity();
    if constexpr (!host) {
      if constexpr (!application_classification) {
        LPFX_TRY(lpfx::detail::DeviceParticleClassificationSlotsAccess::
                     reserve_internal(device_.owned_slots, owner_->capacity()));
      }
      if (pattern_.routing().round_count() != 0) {
        LPFX_TRY(lpfx::detail::DeviceParticleClassificationSlotsAccess::
                     reserve_internal<true>(
                         device_.receive_slots,
                         static_cast<std::uint32_t>(
                             pattern_.capacity().maximum_incoming())));
      }
    }
    budget_ = &budget;
    phase_ = phase_type::ready;
    return ErrorCode::success;
  }

  template <class Session, class Setup>
  [[nodiscard]] ErrorCode reserve(HostMemoryBudget &budget, Session &session,
                                  const Setup &setup,
                                  const capacity_type &requested) noexcept {
    return reserve(budget, session, setup, requested, requested.maximum_final());
  }

  // All ranks enter in the same operation order with no accepted epoch.
  // Larger limits or a changed owner capacity replace the storage; the
  // decoded-arrival reservation is independent of the logical final bound.
  template <class Session, class Setup>
  [[nodiscard]] ErrorCode reserve(HostMemoryBudget &budget, Session &session,
                                  const Setup &setup,
                                  const capacity_type &requested,
                                  const std::uint64_t arrival_records) noexcept {
    assert(phase_ == phase_type::ready);
    capacity_type capacity{};
    bool changed = false;
    ErrorCode local = budget.sealed() && owner_->valid()
                          ? validate_rank(setup, pattern_)
                          : ErrorCode::invalid_argument;
    if (local == ErrorCode::success &&
        arrival_records > requested.maximum_final())
      local = ErrorCode::invalid_argument;
    if constexpr (host) {
      if (local == ErrorCode::success && owner_->capacity() > UINT32_MAX)
        local = ErrorCode::capacity_exceeded;
    }
    if (local == ErrorCode::success)
      local = lpfx::detail::try_merge_particle_migration_capacity(
          pattern_.capacity(), requested, capacity);
    if (local == ErrorCode::success) {
      changed = !(capacity == pattern_.capacity()) ||
                owner_->capacity() != owner_capacity_;
      local = validate_affinity(setup, *owner_, executor());
    }
    LPFX_TRY(count_port_.agree_error(local));

    // Revalidate the caller's reciprocal reservation. Automatic admission may
    // have raised local transit bounds independently of peers; those retained
    // hints need not be reciprocal.
    if (pattern_.routing().round_count() != 0) {
      const auto peers = requested.peer_count();
      for (std::uint8_t peer = 0; peer != peers; ++peer)
        outgoing_counts_[peer] = requested.maximum_peer_outgoing(peer);
      auto status = count_port_.start({outgoing_counts_.data(), peers},
                                      {incoming_counts_.data(), peers});
      while (status.is_pending())
        status = count_port_.template advance<true>();
      require_complete(status);
      for (std::uint8_t peer = 0; peer != peers; ++peer)
        if (incoming_counts_[peer] != requested.maximum_peer_incoming(peer))
          local = ErrorCode::invalid_argument;
    }
    if constexpr (!host && !application_classification) {
      if (local == ErrorCode::success)
        local = lpfx::detail::DeviceParticleClassificationSlotsAccess::
            reserve_internal(device_.owned_slots, owner_->capacity());
    }
    LPFX_TRY(count_port_.agree_error(local));

    const pattern_type previous = pattern_;
    typename Execution::storage_type storage{};
    [[maybe_unused]] typename device_state_type::reservation_type reservation{};
    PayloadPort &candidate = next_payload();
    const device::DeviceProperties device_properties = properties_of(setup);
    if (changed) {
      pattern_type grown{};
      local = try_prepare_particle_migration_pattern(pattern_.routing(),
                                                     capacity, grown);
      if (local == ErrorCode::success)
        local = Execution::try_prepare_storage(
            budget, capacity, owner_->capacity(), device_properties, storage);
      if constexpr (!host) {
        if (local == ErrorCode::success)
          local = device_.resident.prepare(budget, *owner_, device_properties);
      }
      if (local == ErrorCode::success) {
        pattern_ = grown;
        // A full preparation, as for the first port: the current port need
        // not be prepared as a payload source (zero-round patterns never
        // prepare their staging decorator).
        local = candidate.try_prepare_resources(
            budget, session, setup, plan(), storage.wire.prepared(),
            storage.wire, executor(), count_port_, requests_);
      }
    }
    if constexpr (!host) {
      if (local == ErrorCode::success &&
          pattern_.routing().round_count() != 0) {
        // Provision the caller's incoming bound and arrival working set, not
        // a larger logical final bound or speculative transit admission.
        const auto incoming =
            static_cast<std::uint32_t>(requested.maximum_incoming());
        const auto finals = static_cast<std::uint32_t>(arrival_records);
        if (incoming > device_.receive_slots.capacity())
          local = lpfx::detail::DeviceParticleClassificationSlotsAccess::
              reserve_internal<true>(reservation.receive_slots, incoming);
        if (local == ErrorCode::success &&
            finals > execution_.arrival_storage().maximum_capacity())
          local = reservation.arrivals.reserve(budget, finals);
      }
    }
    const ErrorCode agreed = count_port_.agree_error(local);
    if (agreed != ErrorCode::success) {
      pattern_ = previous;
      candidate.reset_resources();
      return agreed;
    }
    if (changed) {
      execution_.adopt(std::move(storage));
      payload_->reset_resources();
      payload_ = &candidate;
      owner_capacity_ = owner_->capacity();
    }
    if constexpr (!host) {
      if (reservation.receive_slots.capacity() != 0)
        lpfx::detail::DeviceParticleClassificationSlotsAccess::
            swap_received_reservation(device_.receive_slots,
                                      reservation.receive_slots);
      if (pattern_.routing().round_count() != 0) {
        auto &arrivals = execution_.arrival_storage();
        arrivals.prepend_reservation(std::move(reservation.arrivals));
        // Position an existing fitting chunk first even when capacities were
        // unchanged; the staged reservation proves this cannot allocate.
        if (arrivals.reserve(budget, static_cast<std::uint32_t>(
                                         arrival_records)) != ErrorCode::success)
          std::abort();
      }
      device_.completed.reset();
      device_.properties = device_properties;
    }
    budget_ = &budget;
    return ErrorCode::success;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start() noexcept
    requires host
  {
    return start_epoch();
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start(const executor_type local_executor) noexcept
    requires(!host)
  {
    if (!local_executor.bound())
      return Status::failure(ErrorCode::invalid_argument);
    device_.preparation_executor.emplace(local_executor);
    return start_epoch();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status release() noexcept {
    assert(phase_ == phase_type::complete);
    if constexpr (!host) {
      lpfx::detail::DeviceParticleClassificationSlotsAccess::release(
          classification_slots());
      device_.epoch_view.reset();
      device_.submitted = false;
      device_.complete = false;
      device_.packed = false;
    }
    rounds_completed_ = 0;
    phase_ = phase_type::ready;
    return Status::success();
  }

  void release_resources_for_close() noexcept {
    if (phase_ != phase_type::ready)
      std::abort();
    reset();
  }

  void reset() noexcept {
    if (phase_ > phase_type::ready)
      std::abort();
    for (auto &port : payload_ports_)
      port.reset_resources();
    payload_ = &payload_ports_[0];
    count_port_.reset_resources();
    requests_.reset_resources();
    std::destroy_at(&execution_);
    std::construct_at(&execution_);
    if constexpr (!host) {
      std::destroy_at(&device_);
      std::construct_at(&device_);
    }
    pattern_ = {};
    owner_ = nullptr;
    semantics_.reset();
    executor_.reset();
    budget_ = nullptr;
    owner_capacity_ = 0;
    phase_ = phase_type::empty;
  }

  // Admits one producer batch of application flags on producer. A previous
  // numerical receipt may still be read by its consumer; admission checks
  // only the communication epoch and the slots.
  [[nodiscard]] auto prepare_classification(const executor_type producer) noexcept
    requires flag_decoding
  {
    using Backend = typename executor_type::backend_type;
    using Decoder = typename semantics_type::flag_decoder_type;
    using Producer =
        ParticleClassificationProducer<typename pattern_type::routing_type,
                                       Backend, Decoder>;
    using Batch = ParticleClassificationBatch<Producer, Backend>;
    using Output = Result<Batch>;
    if (phase_ != phase_type::ready)
      return Output::from_error(ErrorCode::invalid_state);
    auto &slots = classification_slots();
    const auto &routing = pattern_.routing();
    auto prepared =
        slots.prepare_registered(*owner_, routing.peer_count(), producer);
    LPFX_TRY_RESULT(prepared, Output);
    const ErrorCode error =
        lpfx::detail::DeviceParticleClassificationSlotsAccess::
            set_registration_context(slots,
                                     semantics_->classification_context());
    if (error != ErrorCode::success) {
      if (slots.cancel(producer) != ErrorCode::success)
        std::abort();
      return Output::from_error(error);
    }
    if constexpr (ParticleActionFlagDecoder<Decoder>)
      return Output::from_value(
          Batch{Producer{prepared.value()}, slots, producer});
    else
      return Output::from_value(
          Batch{Producer{routing, prepared.value()}, slots, producer});
  }

  [[nodiscard]] bool resident_access_ready() const noexcept
    requires external_resident_access
  {
    return device_.submitted && device_.lease_active && !device_.acquired;
  }

  // The resident prefix [0, count) of the owner, reorderable while the lease
  // of this epoch is open.
  [[nodiscard]] auto acquire_resident_access() noexcept
    requires external_resident_access
  {
    using Range = ResidentReorderRange<typename owner_type::schema_type,
                                       typename executor_type::backend_type>;
    if (!resident_access_ready())
      return Result<Range>::from_error(ErrorCode::invalid_state);
    device_.acquired = true;
    return Result<Range>::from_value(
        Range{DeviceParticleOwnerStamp::capture(*owner_),
              device_.resident.count, device_.candidate_generation,
              &device_.candidate_generation, &device_.lease_active,
              &device_.resident.ready, device_.resident.failure});
  }

  // completion is recorded after the last writer and reader of the resident
  // range; it is polled once and not retained.
  template <class Event>
  [[nodiscard]] Status release_resident_access(const Event &completion) noexcept
    requires external_resident_access
  {
    if (!device_.acquired || !device_.lease_active || !device_.complete ||
        completion.empty())
      return Status::failure(ErrorCode::invalid_state);
    const Status completed = completion.template advance<false>();
    if (completed.is_pending())
      return completed;
    if (completed.is_failure())
      std::abort();
    device_.released = true;
    device_.lease_active = false;
    return Status::success();
  }

  // Side-effect free: no event wait, mutation, allocation or admission.
  [[nodiscard]] Result<ParticleFinalizationRequirements>
  finalization_requirements() const noexcept
    requires external_resident_access
  {
    using Output = Result<ParticleFinalizationRequirements>;
    if (!device_.complete || (phase_ != phase_type::routing_complete &&
                              phase_ != phase_type::finalizing))
      return Output::from_error(ErrorCode::invalid_state);
    const auto residents = execution_.resident_count();
    const auto final =
        std::uint64_t{residents} + execution_.cumulative_final_count();
    if (final > UINT32_MAX)
      return Output::from_error(ErrorCode::overflow);
    return Output::from_value({residents, static_cast<std::uint32_t>(final),
                               final > owner_->capacity()});
  }

  [[nodiscard]] bool storage_changed() const noexcept
    requires external_resident_access
  {
    return device_.completed.has_value() && device_.storage_changed;
  }

  [[nodiscard]] bool auxiliary_reusable() const noexcept
    requires external_resident_access
  {
    return !device_.lease_active;
  }

  [[nodiscard]] bool auxiliary_quiescent_for_close() const noexcept
    requires(external_resident_access || flag_decoding)
  {
    if constexpr (external_resident_access)
      return !device_.lease_active && classification_slots().reusable();
    else
      return classification_slots().reusable();
  }

  [[nodiscard]] Result<CompletedParticlePartition>
  completed_partition() const noexcept
    requires external_resident_access
  {
    if (!device_.completed || owner_ == nullptr ||
        !device_.completed_owner.matches_epoch(*owner_) ||
        owner_->size() != device_.completed->total_size)
      return Result<CompletedParticlePartition>::from_error(
          ErrorCode::invalid_state);
    return Result<CompletedParticlePartition>::from_value(*device_.completed);
  }

private:
  using device_state_type = DeviceOnlyState<Execution, space_type>;

  // The pattern as the ports see a plan: the count port binds its address.
  struct PatternPlan final {
    using pattern_type = ParticleProgram::pattern_type;
    using contract_type = ParticleProgram::contract_type;
    const pattern_type *value;

    [[nodiscard]] bool is_valid() const noexcept { return value->valid(); }
    [[nodiscard]] const pattern_type &pattern() const noexcept {
      return *value;
    }
  };

  friend struct PreparedOperationProgramAccess;

  template <class Setup>
  [[nodiscard]] static ErrorCode
  validate_rank(const Setup &setup, const pattern_type &pattern) noexcept {
    std::uint64_t ranks = 0;
    if (!setup.is_valid() ||
        !lpfx::detail::checked_volume(pattern.routing().process_shape(),
                                      ranks) ||
        !std::cmp_equal(pattern.process_rank(), setup.rank()) ||
        !std::cmp_equal(ranks, setup.size()))
      return ErrorCode::invalid_argument;
    return ErrorCode::success;
  }

  template <class Setup>
  [[nodiscard]] static ErrorCode
  validate_affinity([[maybe_unused]] const Setup &setup,
                    [[maybe_unused]] const owner_type &owner,
                    [[maybe_unused]] const executor_type &executor) noexcept {
    if constexpr (!host) {
      using Backend = typename executor_type::backend_type;
      LPFX_TRY(setup.template try_verify_device_affinity<Backend>(executor));
      const std::byte *const allocation =
          DeviceParticleOwnerStamp::capture(owner).allocation();
      if (allocation != nullptr)
        return setup.template try_verify_device_pointer_affinity<Backend>(
            allocation);
    }
    return ErrorCode::success;
  }

  template <class Setup>
  [[nodiscard]] static device::DeviceProperties
  properties_of([[maybe_unused]] const Setup &setup) noexcept {
    if constexpr (host)
      return {};
    else
      return setup.device_properties();
  }

  [[nodiscard]] device::DeviceProperties properties() const noexcept {
    if constexpr (host)
      return {};
    else
      return device_.properties;
  }

  [[nodiscard]] PatternPlan plan() const noexcept { return {&pattern_}; }

  [[nodiscard]] PayloadPort &next_payload() noexcept {
    return payload_ == &payload_ports_[0] ? payload_ports_[1]
                                          : payload_ports_[0];
  }

  [[nodiscard]] decltype(auto) classification_slots() noexcept
    requires(!host)
  {
    if constexpr (application_classification)
      return semantics_->classification_slots();
    else
      return (device_.owned_slots);
  }
  [[nodiscard]] decltype(auto) classification_slots() const noexcept
    requires(!host)
  {
    if constexpr (application_classification)
      return std::as_const(semantics_->classification_slots());
    else
      return std::as_const(device_.owned_slots);
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE const executor_type &
  executor() const noexcept {
    return executor_.get();
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE const executor_type &
  preparation_executor() const noexcept {
    if constexpr (host)
      return executor();
    else
      return *device_.preparation_executor;
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE const auto &
  epoch_semantics() const noexcept {
    if constexpr (host)
      return *semantics_;
    else
      return *device_.epoch_view;
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE typename Execution::protocol_type
  protocol() const noexcept {
    return {pattern_.routing(), epoch_semantics().initial_owner(),
            epoch_semantics().received_owner()};
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE bool last_round() const noexcept {
    return rounds_completed_ + UINT64_C(1) == pattern_.routing().round_count();
  }

  void set_candidate_generation(const std::uint64_t generation) noexcept
    requires external_resident_access
  {
    device_.acquired = false;
    device_.released = false;
    device_.storage_changed = false;
    device_.candidate_generation = generation;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_epoch() noexcept {
    assert(phase_ == phase_type::ready);
    // Owner replacement is legal between epochs, but a changed capacity needs
    // reserve before any classification can access the old scratch.
    if (!owner_->valid() || owner_->capacity() != owner_capacity_) {
      if constexpr (!host)
        std::abort();
      else
        return Status::failure(
            count_port_.agree_error(ErrorCode::invalid_state));
    }
    if constexpr (!host) {
      using SlotsAccess = lpfx::detail::DeviceParticleClassificationSlotsAccess;
      auto &slots = classification_slots();
      const auto peers = pattern_.routing().peer_count();
      if constexpr (application_classification) {
        if (!SlotsAccess::registration_matches(
                slots, semantics_->classification_context(), peers) ||
            SlotsAccess::acquire(slots, *owner_, preparation_executor()) !=
                ErrorCode::success)
          std::abort();
      } else if (SlotsAccess::prepare_internal(slots, *owner_, peers,
                                               preparation_executor()) !=
                 ErrorCode::success) {
        std::abort();
      }
      execution_.bind_classification(SlotsAccess::registered_view(slots));
      device_.epoch_view.emplace(semantics_->device_view());
    }
    const Status submitted = execution_.start_initial_classification(
        protocol(), preparation_executor());
    if constexpr (host) {
      const ErrorCode agreed = count_port_.agree_error(
          submitted.is_failure() ? submitted.code() : ErrorCode::success);
      if (agreed != ErrorCode::success) {
        execution_.discard_initial_classification();
        return Status::failure(agreed);
      }
    } else if (submitted.is_failure()) {
      std::abort();
    }
    if constexpr (external_resident_access)
      device_.completed.reset();
    phase_ = phase_type::initial_classification;
    if constexpr (!host)
      return Status::pending();
    else
      return drive<false>();
  }

  static LPFX_ALWAYS_INLINE void
  require_complete(const Status status) noexcept {
    if (!status.is_success() || status.requires_conclusion())
      std::abort();
  }
  static LPFX_ALWAYS_INLINE void
  require_submitted(const Status status) noexcept {
    if (status.is_failure() || status.is_pending())
      std::abort();
  }
  template <ParticleExecutionStage Stage, bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status complete_execution() noexcept {
    const Status status = execution_.template advance<Stage, Wait>();
    if (status.is_failure() || status.requires_conclusion())
      std::abort();
    return status;
  }
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status complete_count() noexcept {
    const Status status = count_port_.template advance<Wait>();
    if (status.is_failure() || status.requires_conclusion())
      std::abort();
    return status;
  }
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status complete_payload() noexcept {
    const Status status = payload_->template advance_payload<Wait>();
    if (status.is_failure() || status.requires_conclusion())
      std::abort();
    return status;
  }

  LPFX_ALWAYS_INLINE void prepare_outgoing_metadata() noexcept {
    const auto &prefixes = execution_.outgoing_prefixes();
    const std::uint8_t peers = pattern_.routing().peer_count();
    for (std::uint8_t peer = 0; peer != peers; ++peer) {
      outgoing_counts_[peer] = prefixes.particle_count(peer);
      incoming_counts_[peer] = 0;
    }
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_count_exchange() noexcept {
    const std::uint8_t peers = pattern_.routing().peer_count();
    // Native execution keeps the peer induction compact across its larger
    // inlined migration driver; the host count leaf retains its native width.
    using PeerIndex = std::conditional_t<host, std::size_t, std::uint8_t>;
    const Status started =
        count_port_.template start_with_peer_index<PeerIndex>(
            std::span<const std::uint64_t>{outgoing_counts_.data(), peers},
            std::span<std::uint64_t>{incoming_counts_.data(), peers});
    if (started.is_failure() || started.requires_conclusion())
      std::abort();
    return started;
  }

  LPFX_ALWAYS_INLINE void start_outgoing() noexcept {
    require_submitted(payload_->submit_outgoing(
        executor(), execution_.wire_storage(), execution_.outgoing_prefixes()));
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status admit_and_start_payload() noexcept {
    const std::uint8_t peers = pattern_.routing().peer_count();
    if (try_admit_particle_incoming(
            pattern_.capacity(),
            std::span<const std::uint64_t>{incoming_counts_.data(), peers},
            incoming_prefixes_) != ErrorCode::success)
      std::abort();
    // The completed pack/received-round event covers slot-A production and
    // any previous slot-B reads. Resident compaction touches neither buffer.
    const Status submitted = payload_->start_payload(
        executor(), execution_.wire_storage(), execution_.outgoing_prefixes(),
        incoming_prefixes_, count_port_);
    if (submitted.is_failure() || submitted.requires_conclusion())
      std::abort();
    return submitted;
  }

  // Slot B becomes available to the received classification enqueued next
  // on executor; staging stays leased until recycle.
  LPFX_ALWAYS_INLINE void start_received_classification() noexcept {
    assert(rounds_completed_ < pattern_.routing().round_count());
    require_submitted(payload_->finish_incoming(
        executor(), execution_.wire_storage(), incoming_prefixes_));
    if constexpr (!host) {
      using SlotsAccess = lpfx::detail::DeviceParticleClassificationSlotsAccess;
      if (SlotsAccess::prepare_received(
              device_.receive_slots,
              static_cast<std::uint32_t>(incoming_prefixes_.total_particles()),
              pattern_.routing().peer_count(),
              executor()) != ErrorCode::success)
        std::abort();
      execution_.bind_classification(
          SlotsAccess::registered_view(device_.receive_slots));
    }
    require_submitted(execution_.start_received_classification(
        rounds_completed_, incoming_prefixes_, protocol(), executor()));
    phase_ = phase_type::received_classification;
  }

  LPFX_ALWAYS_INLINE void submit_initial_pack() noexcept {
    require_submitted(execution_.start_initial_pack(epoch_semantics(),
                                                    preparation_executor()));
    phase_ = phase_type::initial_pack;
  }

  LPFX_ALWAYS_INLINE void submit_received_processing() noexcept {
    require_submitted(execution_.start_received_processing(
        incoming_prefixes_, epoch_semantics(), executor()));
    phase_ = phase_type::received_processing;
  }

  LPFX_ALWAYS_INLINE void submit_resident_compaction() noexcept {
    if constexpr (host) {
      require_submitted(execution_.compact_residents());
    } else {
      require_submitted(device_.resident.submit(
          execution_.initial_decisions(), execution_.resident_count(),
          execution_.source_count(), preparation_executor()));
      device_.submitted = true;
      if constexpr (external_resident_access) {
        device_.lease_active = true;
        device_.resident_prefix = device_.resident.count;
        device_.completed.reset();
      }
    }
  }

  // The steady state examines only the exchanged scalar counts; allocation,
  // copies and native synchronization are cold callees.
  template <bool FirstRound>
  [[nodiscard]] LPFX_ALWAYS_INLINE bool admit_round_capacity() noexcept {
    const auto &capacity = pattern_.capacity();
    std::uint64_t outgoing = 0, incoming = 0;
    bool growth = false;
    for (std::uint8_t peer = 0; peer != capacity.peer_count(); ++peer) {
      // A count beyond the prepared capacity enters the cold growth
      // validator, which applies the transport maximum before allocating.
      if (!lpfx::detail::checked_add(outgoing, outgoing_counts_[peer],
                                     outgoing) ||
          !lpfx::detail::checked_add(incoming, incoming_counts_[peer],
                                     incoming))
        std::abort();
      growth = growth ||
               outgoing_counts_[peer] > capacity.maximum_peer_outgoing(peer) ||
               incoming_counts_[peer] > capacity.maximum_peer_incoming(peer);
    }
    // Device incoming records can be transit or discarded: the main owner and
    // exact cumulative finals are admitted after classification.
    std::uint64_t finals = 0, owner_extent = 0;
    if (lpfx::detail::try_particle_round_bounds<!host, maximum_peer_particles>(
            outgoing, incoming, execution_.resident_count(),
            execution_.cumulative_final_count(), finals,
            owner_extent) != ErrorCode::success)
      std::abort();
    growth = growth ||
             outgoing > (FirstRound ? capacity.maximum_outgoing()
                                    : capacity.maximum_transit()) ||
             incoming > capacity.maximum_incoming() ||
             finals > capacity.maximum_final();
    if constexpr (host)
      growth = growth || owner_extent > owner_->capacity();
    if (!last_round()) {
      growth = growth || incoming > capacity.maximum_transit();
      const auto peer_transit =
          incoming < maximum_peer_particles ? incoming : maximum_peer_particles;
      for (std::uint8_t peer = 0; peer != capacity.peer_count(); ++peer)
        growth = growth || peer_transit > capacity.maximum_peer_outgoing(peer);
    }
    return growth && grow<FirstRound>(owner_extent);
  }

  // Replaces the storage and payload port for grown capacity; growth failure
  // is fatal. adopt moves the epoch into the new storage.
  template <class Adopt>
  LPFX_NOINLINE void replace_storage(const capacity_type &capacity,
                                     const std::size_t owner_capacity,
                                     Adopt &&adopt) noexcept {
    pattern_type grown{};
    typename Execution::storage_type storage{};
    if (try_prepare_particle_migration_pattern(pattern_.routing(), capacity,
                                               grown) != ErrorCode::success ||
        Execution::try_prepare_storage(*budget_, capacity, owner_capacity,
                                       properties(),
                                       storage) != ErrorCode::success)
      std::abort();
    pattern_ = grown;
    PayloadPort &candidate = next_payload();
    if (candidate.prepare_growth(*payload_, *budget_, plan(),
                                 storage.wire.prepared(), storage.wire,
                                 executor(), count_port_,
                                 requests_) != ErrorCode::success ||
        adopt(std::move(storage)) != ErrorCode::success)
      std::abort();
    payload_->reset_resources();
    payload_ = &candidate;
  }

  // Returns true when the host initial pack was completed by owner growth.
  template <bool FirstRound>
  [[nodiscard]] LPFX_NOINLINE bool
  grow(const std::uint64_t required_owner) noexcept {
    const auto peers = pattern_.routing().peer_count();
    capacity_type capacity{};
    std::uint64_t checked_owner = 0;
    bool changed = false;
    if (lpfx::detail::try_prepare_particle_round_capacity<!host>(
            pattern_.capacity(), {outgoing_counts_.data(), peers},
            {incoming_counts_.data(), peers}, execution_.resident_count(),
            execution_.cumulative_final_count(), !last_round(), capacity,
            checked_owner, changed, maximum_peer_particles,
            FirstRound) != ErrorCode::success ||
        checked_owner != required_owner)
      std::abort();
    if constexpr (!host) {
      // Wire admission never relocates the resident owner. A packed slot A
      // was produced on the stream of its stage.
      const bool packed = device_.packed;
      replace_storage(
          capacity, owner_->capacity(), [&](auto &&storage) noexcept {
            return execution_.grow(
                std::move(storage), packed,
                FirstRound ? preparation_executor() : executor());
          });
      return false;
    } else {
      if (required_owner <= owner_->capacity()) {
        replace_storage(capacity, owner_->capacity(),
                        [&](auto &&storage) noexcept {
                          return execution_.grow(std::move(storage),
                                                 FirstRound);
                        });
        return false;
      }
      std::uint32_t next_capacity = 0;
      if (lpfx::detail::try_grow_particle_capacity(
              static_cast<std::uint32_t>(owner_->capacity()), required_owner,
              next_capacity) != ErrorCode::success)
        std::abort();
      auto grown = [&] {
        if constexpr (FirstRound)
          return make_grown_host_particles<ParticleResizeMode::discard>(
              *owner_, next_capacity, 0);
        else
          return make_grown_host_particles(*owner_, next_capacity,
                                           execution_.preserved_count());
      }();
      if (!grown.has_value())
        std::abort();
      auto next_owner = std::move(grown).value();
      replace_storage(capacity, next_owner.capacity(),
                      [&](auto &&storage) noexcept {
                        if constexpr (FirstRound)
                          return execution_.grow_initial_owner(
                              std::move(storage), next_owner, *semantics_);
                        else
                          return execution_.grow(std::move(storage), false);
                      });
      *owner_ = std::move(next_owner);
      owner_capacity_ = owner_->capacity();
      return FirstRound;
    }
  }

  LPFX_ALWAYS_INLINE void publish_migration() noexcept {
    execution_.publish_completed_migration();
    if constexpr (external_resident_access) {
      device_.completed_owner = DeviceParticleOwnerStamp::capture(*owner_);
      device_.completed.emplace(CompletedParticlePartition{
          device_.resident_prefix, owner_->size(),
          device_.candidate_generation});
    }
    phase_ = phase_type::complete;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status finish_migration() noexcept {
    if constexpr (host) {
      publish_migration();
      return Status::success();
    } else {
      phase_ = phase_type::finalizing;
      if constexpr (external_resident_access) {
        if (device_.lease_active || !device_.released)
          return Status::pending();
      }
      const auto residents = execution_.resident_count();
      // Resident and routing stages have completed. Check main ownership here
      // before growth or append, never from routing while compaction may still
      // publish the retained size.
      if (owner_->size() != residents)
        std::abort();
      const auto required = static_cast<std::uint64_t>(residents) +
                            execution_.cumulative_final_count();
      if (required > UINT32_MAX)
        std::abort();
      if (required > owner_->capacity()) {
        if constexpr (external_resident_access)
          device_.storage_changed = true;
        std::uint32_t capacity = 0;
        if (lpfx::detail::try_grow_particle_capacity(owner_->capacity(),
                                                     required, capacity) !=
                ErrorCode::success ||
            owner_->grow_in_place(capacity, residents,
                                  preparation_executor()) !=
                ErrorCode::success ||
            device_.resident.prepare(*budget_, *owner_, device_.properties) !=
                ErrorCode::success)
          std::abort();
        execution_.refresh_owner_capacity();
        owner_capacity_ = owner_->capacity();
      }
      require_submitted(execution_.submit_append_arrivals(
          lpfx::detail::DeviceParticlesAccess::capacity_view(*owner_),
          preparation_executor()));
      phase_ = phase_type::appending;
      return Status::pending();
    }
  }

  template <bool Wait,
            ParticleMigrationStage Stage = ParticleMigrationStage::finalize>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status drive() noexcept {
    if constexpr (!host && Stage == ParticleMigrationStage::resident) {
      if (!device_.submitted)
        return Status::failure(ErrorCode::invalid_state);
      if (device_.complete)
        return Status::success();
      const Status compacted = device_.resident.template advance<Wait>();
      if (compacted.is_pending())
        return compacted;
      require_complete(compacted);
      device_.complete = true;
      return Status::success();
    } else {
      if constexpr (!host) {
        if constexpr (Stage == ParticleMigrationStage::finalize) {
          if (!device_.complete || (phase_ != phase_type::routing_complete &&
                                    phase_ != phase_type::finalizing &&
                                    phase_ != phase_type::appending &&
                                    phase_ != phase_type::complete))
            return Status::failure(ErrorCode::invalid_state);
        } else if constexpr (Stage == ParticleMigrationStage::route) {
          if (phase_ != phase_type::prepared &&
              phase_ != phase_type::routing_complete &&
              (phase_ < phase_type::round_counts ||
               phase_ > phase_type::received_processing))
            return Status::failure(ErrorCode::invalid_state);
        } else {
          if (phase_ != phase_type::initial_classification &&
              phase_ != phase_type::initial_counts &&
              phase_ != phase_type::initial_pack &&
              phase_ != phase_type::initial_admission &&
              phase_ != phase_type::prepared)
            return Status::failure(ErrorCode::invalid_state);
        }
      }
      const std::uint64_t rounds = pattern_.routing().round_count();
      for (;;) {
        switch (phase_) {
        case phase_type::initial_classification: {
          if (complete_execution<ParticleExecutionStage::initial_classification,
                                 Wait>()
                  .is_pending())
            return Status::pending();
          prepare_outgoing_metadata();
          if (rounds != 0) {
            if constexpr (!host) {
              // Admit the known outgoing image before starting counts;
              // incoming bounds are admitted after packing, without delaying
              // resident compaction behind the count exchange.
              static_cast<void>(admit_round_capacity<true>());
              static_cast<void>(start_count_exchange());
            } else {
              static_cast<void>(start_count_exchange());
              phase_ = phase_type::initial_counts;
              break;
            }
          }
          submit_initial_pack();
          break;
        }
        case phase_type::initial_counts: {
          if (complete_count<Wait>().is_pending())
            return Status::pending();
          if (admit_round_capacity<true>())
            phase_ = phase_type::initial_pack;
          else
            submit_initial_pack();
          break;
        }
        case phase_type::initial_pack: {
          Status counts = Status::success();
          if constexpr (!host) {
            if (rounds != 0)
              counts = complete_count<false>();
          }
          const Status packed =
              complete_execution<ParticleExecutionStage::initial_pack, Wait>();
          if (packed.is_pending())
            return packed;
          submit_resident_compaction();
          if constexpr (!host) {
            device_.packed = true;
            phase_ = phase_type::initial_admission;
            // Return at this milestone so the application can enqueue
            // resident work before another call enters count admission.
            if constexpr (external_resident_access)
              return Status::pending();
          } else {
            if (rounds == 0)
              return finish_migration();
            start_outgoing();
            phase_ = phase_type::round_outgoing;
          }
          if (counts.is_pending())
            return counts;
          break;
        }
        case phase_type::round_counts: {
          if (complete_count<Wait>().is_pending())
            return Status::pending();
          static_cast<void>(admit_round_capacity<false>());
          start_outgoing();
          phase_ = phase_type::round_outgoing;
          break;
        }
        case phase_type::round_outgoing: {
          const Status outgoing =
              payload_->template advance_outgoing<Wait>();
          if (outgoing.is_pending() || outgoing.requires_conclusion())
            return Status::pending();
          require_complete(outgoing);
          const Status payload = admit_and_start_payload();
          phase_ = phase_type::round_payload;
          // A newly issued batch belongs to the next driving call: counts
          // and payload never both reach native completion in one advance.
          if (payload.is_pending())
            return Status::pending();
          break;
        }
        case phase_type::round_payload: {
          // A stream-ordered payload precedes the classification on executor.
          if constexpr (!PayloadPort::stream_ordered) {
            if (complete_payload<Wait>().is_pending())
              return Status::pending();
          }
          start_received_classification();
          break;
        }
        case phase_type::received_classification: {
          if (complete_execution<
                  ParticleExecutionStage::received_classification, Wait>()
                  .is_pending())
            return Status::pending();
          if (last_round() &&
              execution_.outgoing_prefixes().total_particles() != 0)
            std::abort();
          submit_received_processing();
          // The next counts read only the published transit prefixes; the
          // payload holds no request of the shared storage.
          if (!last_round()) {
            prepare_outgoing_metadata();
            if (start_count_exchange().is_pending())
              return Status::pending();
          }
          break;
        }
        case phase_type::received_processing: {
          if (complete_execution<ParticleExecutionStage::received_processing,
                                 Wait>()
                  .is_pending())
            return Status::pending();
          require_complete(payload_->recycle());
          if constexpr (!host)
            lpfx::detail::DeviceParticleClassificationSlotsAccess::release(
                device_.receive_slots);
          ++rounds_completed_;
          if (rounds_completed_ == rounds) {
            if constexpr (host) {
              return finish_migration();
            } else {
              // A final round without arrivals observed no event after its
              // stream-ordered sends, and the next epoch packs slot A on the
              // preparation stream: the host blocks until the sends complete.
              if constexpr (PayloadPort::stream_ordered) {
                using Backend = typename executor_type::backend_type;
                if (incoming_prefixes_.total_particles() == 0)
                  device::abort_on_failure<Backend>(
                      device::BackendTraits<Backend>::synchronize_stream(
                          executor().stream()));
              }
              phase_ = phase_type::routing_complete;
              return Status::success();
            }
          }
          phase_ = phase_type::round_counts;
          break;
        }
        case phase_type::initial_admission: {
          if constexpr (!host) {
            if (rounds != 0) {
              const Status counts = complete_count<Wait>();
              if (counts.is_pending())
                return counts;
              static_cast<void>(admit_round_capacity<true>());
            }
            phase_ = phase_type::prepared;
            return Status::success();
          } else {
            return Status::failure(ErrorCode::invalid_state);
          }
        }
        case phase_type::prepared: {
          if constexpr (!host) {
            if constexpr (Stage == ParticleMigrationStage::prepare)
              return Status::success();
            if (rounds == 0) {
              phase_ = phase_type::routing_complete;
              return Status::success();
            }
            start_outgoing();
            phase_ = phase_type::round_outgoing;
            break;
          } else {
            return Status::failure(ErrorCode::invalid_state);
          }
        }
        case phase_type::routing_complete: {
          if constexpr (Stage == ParticleMigrationStage::route)
            return Status::success();
          return finish_migration();
        }
        case phase_type::finalizing:
          return finish_migration();
        case phase_type::appending: {
          if constexpr (!host) {
            const Status appended =
                execution_.template advance_append_arrivals<Wait>();
            if (appended.is_pending())
              return appended;
            require_complete(appended);
            publish_migration();
            return Status::success();
          } else {
            return Status::failure(ErrorCode::invalid_state);
          }
        }
        case phase_type::complete:
          return Status::success();
        case phase_type::empty:
        case phase_type::ready:
          return Status::failure(ErrorCode::invalid_state);
        }
      }
    }
  }

  pattern_type pattern_{};
  request_storage_type requests_{};
  CountPort count_port_{};
  std::array<PayloadPort, 2> payload_ports_{};
  PayloadPort *payload_ = &payload_ports_[0];
  Execution execution_{};
  [[no_unique_address]] device_state_type device_{};
  [[no_unique_address]] ExecutorSlot<space_type, executor_type> executor_{};
  std::optional<semantics_type> semantics_{};
  owner_type *owner_ = nullptr;
  HostMemoryBudget *budget_ = nullptr;
  std::array<std::uint64_t, particle_migration_maximum_peer_count>
      outgoing_counts_{};
  std::array<std::uint64_t, particle_migration_maximum_peer_count>
      incoming_counts_{};
  PreparedParticleIncomingPrefixes<contract_type> incoming_prefixes_{};
  std::uint64_t rounds_completed_ = 0;
  std::size_t owner_capacity_ = 0;
  phase_type phase_ = phase_type::empty;
};

} // namespace lpfx::distributed::detail

#endif
