#ifndef LPFX_DISTRIBUTED_TRANSPORT_PARTICLE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_PARTICLE_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/particle/migration/payload_slices.hpp>
#include <lpfx/distributed/particle/wire/device/staging.hpp>
#include <lpfx/distributed/particle/wire/storage.hpp>
#include <lpfx/distributed/transport/stage.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

// Payload decorators adapt the wire storage to native access. Hooks:
// admit_payload before native access, submit_outgoing/advance_outgoing before
// it, finish_incoming after it (completing with the incoming consumer).
template <class Contract, class Executor, class Storage, class PayloadSpace>
class StatelessParticleDecorator {
public:
  using contract_type = Contract;
  using executor_type = Executor;
  using space_type = PayloadSpace;
  using payload_space_type = PayloadSpace;
  using storage_type = Storage;

  [[nodiscard]] constexpr bool empty() const noexcept { return true; }
  [[nodiscard]] constexpr bool is_valid() const noexcept { return true; }
  [[nodiscard]] constexpr bool reusable() const noexcept { return true; }
  [[nodiscard]] constexpr ErrorCode
  prepare(HostMemoryBudget &,
          const PreparedParticleMigrationWireStorage<contract_type> &,
          storage_type &) noexcept {
    return ErrorCode::success;
  }
  template <class Outgoing, class Incoming>
  [[nodiscard]] constexpr Status admit_payload(const executor_type,
                                               storage_type &, const Outgoing &,
                                               const Incoming &) noexcept {
    return {};
  }
  template <class Prefixes>
  [[nodiscard]] constexpr Status submit_outgoing(const executor_type,
                                                 const storage_type &,
                                                 const Prefixes &) noexcept {
    return {};
  }
  template <bool Wait>
  [[nodiscard]] constexpr Status advance_outgoing() const noexcept {
    return {};
  }
  template <class Prefixes>
  [[nodiscard]] constexpr Status finish_incoming(const executor_type,
                                                 storage_type &,
                                                 const Prefixes &) noexcept {
    return {};
  }
  template <class StorageReference, class Prefixes>
  [[nodiscard]] constexpr auto
  payload(StorageReference &storage, const Prefixes &prefixes) const noexcept {
    static_assert(
        (Prefixes::kind == ParticlePeerPrefixKind::outgoing &&
         std::is_const_v<std::remove_reference_t<StorageReference>>) ||
        (Prefixes::kind == ParticlePeerPrefixKind::incoming &&
         !std::is_const_v<std::remove_reference_t<StorageReference>>));
    return storage.bytes(prefixes);
  }
  [[nodiscard]] constexpr Status recycle() noexcept { return {}; }
  constexpr void reset() noexcept {}
};

template <class Contract, class Executor>
using HostParticleDecorator = StatelessParticleDecorator<
    Contract, Executor, HostParticleMigrationWireStorage<Contract>, HostSpace>;

// Device wire storage accessed in place by the native library (XCCL).
template <class Contract, class Executor, class DeviceBackend>
using DeviceParticleDecorator = StatelessParticleDecorator<
    Contract, Executor,
    DeviceParticleMigrationWireStorage<Contract, DeviceBackend>,
    DeviceSpace<DeviceBackend>>;

// Device wire storage staged through pinned host slots.
template <class Contract, class Executor, class DeviceBackend>
class StagedParticleDecorator {
public:
  using contract_type = Contract;
  using executor_type = Executor;
  using device_backend_type = DeviceBackend;
  using space_type = DeviceSpace<device_backend_type>;
  using payload_space_type = HostSpace;
  using storage_type =
      DeviceParticleMigrationWireStorage<contract_type, device_backend_type>;
  using staging_type =
      DeviceParticleMigrationWireStaging<contract_type, device_backend_type>;

  [[nodiscard]] bool empty() const noexcept { return !staging_.valid(); }
  [[nodiscard]] bool is_valid() const noexcept { return staging_.valid(); }
  [[nodiscard]] bool reusable() const noexcept { return staging_.idle(); }
  [[nodiscard]] ErrorCode
  prepare(HostMemoryBudget &budget,
          const PreparedParticleMigrationWireStorage<contract_type> &prepared,
          storage_type &) noexcept {
    if (!empty()) {
      return ErrorCode::invalid_state;
    }
    auto staging = make_device_particle_migration_wire_staging<
        contract_type, device_backend_type>(budget, prepared);
    LPFX_TRY(staging);
    std::destroy_at(&staging_);
    std::construct_at(&staging_, std::move(staging).value());
    return ErrorCode::success;
  }
  template <class Outgoing, class Incoming>
  [[nodiscard]] Status admit_payload(const executor_type, storage_type &,
                                     const Outgoing &,
                                     const Incoming &incoming) noexcept {
    static_assert(Incoming::kind == ParticlePeerPrefixKind::incoming);
    return staging_.admit_incoming(incoming.total_particles());
  }
  template <class Prefixes>
  [[nodiscard]] Status submit_outgoing(const executor_type executor,
                                       const storage_type &storage,
                                       const Prefixes &prefixes) noexcept {
    static_assert(Prefixes::kind == ParticlePeerPrefixKind::outgoing);
    return staging_.submit_outgoing_download(
        storage, prefixes.total_particles(), executor);
  }
  template <bool Wait> [[nodiscard]] Status advance_outgoing() noexcept {
    return staging_.template advance_outgoing<Wait>();
  }
  // The upload is ordered before the incoming consumer on executor; recycle
  // follows only after that consumer has completed.
  template <class Prefixes>
  [[nodiscard]] Status finish_incoming(const executor_type executor,
                                       storage_type &storage,
                                       const Prefixes &) noexcept {
    static_assert(Prefixes::kind == ParticlePeerPrefixKind::incoming);
    return staging_.enqueue_incoming_upload(storage, executor);
  }
  template <class StorageReference, class Prefixes>
  [[nodiscard]] auto payload(StorageReference &, const Prefixes &) noexcept {
    if constexpr (Prefixes::kind == ParticlePeerPrefixKind::outgoing) {
      return staging_.host_outgoing_image();
    } else {
      static_assert(Prefixes::kind == ParticlePeerPrefixKind::incoming);
      return staging_.host_incoming_image();
    }
  }
  [[nodiscard]] Status recycle() noexcept { return staging_.recycle(); }
  // Destroying a non-idle staging aborts.
  void reset() noexcept {
    std::destroy_at(&staging_);
    std::construct_at(&staging_);
  }

private:
  staging_type staging_{};
};

// Device wire storage handed to MPI under an ordering policy.
template <class Ordering, class Contract, class Executor, class DeviceBackend>
class DirectParticleDecorator
    : public StatelessParticleDecorator<
          Contract, Executor,
          DeviceParticleMigrationWireStorage<Contract, DeviceBackend>,
          DeviceSpace<DeviceBackend>> {
public:
  using executor_type = Executor;
  using storage_type = DeviceParticleMigrationWireStorage<Contract, DeviceBackend>;
  static_assert(std::is_empty_v<Ordering>);

  // The migration driver has completed outgoing production and all prior
  // incoming consumption before admitting these private wire slots to MPI.
  template <class Outgoing, class Incoming>
  [[nodiscard]] Status admit_payload(const executor_type executor,
                                     storage_type &storage,
                                     const Outgoing &outgoing,
                                     const Incoming &incoming) noexcept {
    const ErrorCode error = Ordering::before_mpi_access(
        executor, std::as_const(storage).bytes(outgoing), storage.bytes(incoming));
    return error == ErrorCode::success ? Status::success()
                                       : Status::failure(error);
  }
  template <class Prefixes>
  [[nodiscard]] Status finish_incoming(const executor_type executor,
                                       storage_type &storage,
                                       const Prefixes &prefixes) noexcept {
    if constexpr (MpiPostWriteOrdering<Ordering, executor_type,
                                       decltype(storage.bytes(prefixes))>) {
      const ErrorCode error =
          Ordering::after_mpi_write(executor, storage.bytes(prefixes));
      return error == ErrorCode::success ? Status::success()
                                         : Status::failure(error);
    } else {
      return Status::success();
    }
  }
};

// Particle payload port: Decorator owns wire access and staging, P2p the
// native exchange (MpiParticleP2p or XcclParticleP2p). A stream-ordered
// exchange is ordered on executor before any later work there; the driver
// never advances it.
template <class Decorator, class P2p> class ParticlePayloadPort final {
public:
  using decorator_type = Decorator;
  using p2p_type = P2p;
  static constexpr bool stream_ordered = P2p::stream_ordered;
  static constexpr std::uint64_t maximum_peer_particles =
      P2p::maximum_peer_particles;
  using contract_type = typename Decorator::contract_type;
  using space_type = typename Decorator::space_type;
  using executor_type = typename Decorator::executor_type;
  using storage_type = typename Decorator::storage_type;
  using payload_space_type = typename Decorator::payload_space_type;
  using outgoing_prefixes_type = PreparedParticleOutgoingPrefixes<contract_type>;
  using incoming_prefixes_type = PreparedParticleIncomingPrefixes<contract_type>;
  using prepared_storage_type =
      PreparedParticleMigrationWireStorage<contract_type>;

  [[nodiscard]] bool empty() const noexcept {
    return p2p_.empty() && decorator_.empty();
  }
  [[nodiscard]] bool is_valid() const noexcept {
    return p2p_.is_valid() &&
           (p2p_.maximum_work_count() == 0 || decorator_.is_valid());
  }
  [[nodiscard]] bool reusable() const noexcept {
    return is_valid() && p2p_.reusable() && decorator_.reusable();
  }
  template <class Pattern>
    requires std::same_as<typename Pattern::contract_type, contract_type>
  [[nodiscard]] static ErrorCode request_capacity(const Pattern &pattern,
                                                  std::size_t &output) noexcept {
    return P2p::request_capacity(pattern, output);
  }

  template <class Plan, class Session, class Setup, class CountPort,
            class RequestStorage>
    requires std::same_as<typename Plan::contract_type, contract_type>
  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &budget, Session &session, const Setup &setup,
      const Plan &plan, const prepared_storage_type &prepared,
      storage_type &storage, const executor_type executor,
      const CountPort &count, RequestStorage &backing) noexcept {
    if (!empty() || !count.is_valid() || !setup.is_valid() || !prepared.valid() ||
        storage.prepared() != prepared || &count.pattern() != &plan.pattern()) {
      return ErrorCode::invalid_state;
    }
    if constexpr (requires { typename space_type::backend_type; }) {
      LPFX_TRY(setup.template try_verify_device_affinity<
               typename space_type::backend_type>(executor));
    }
    LPFX_TRY(prepare_decorator(budget, plan, prepared, storage));
    const ErrorCode error =
        p2p_.try_prepare(budget, session, setup, plan, count, backing);
    if (error != ErrorCode::success) {
      decorator_.reset();
    }
    return error;
  }

  template <class Plan, class CountPort, class RequestStorage>
    requires std::same_as<typename Plan::contract_type, contract_type>
  [[nodiscard]] ErrorCode prepare_growth(
      const ParticlePayloadPort &previous, HostMemoryBudget &budget,
      const Plan &plan, const prepared_storage_type &prepared,
      storage_type &storage, const executor_type, const CountPort &count,
      RequestStorage &backing) noexcept {
    if (!empty() || !previous.reusable() || !count.is_valid() ||
        !plan.is_valid() || !prepared.valid() || storage.prepared() != prepared ||
        &count.pattern() != &plan.pattern()) {
      return ErrorCode::invalid_state;
    }
    LPFX_TRY(prepare_decorator(budget, plan, prepared, storage));
    const ErrorCode error =
        p2p_.prepare_growth(previous.p2p_, budget, plan, count, backing);
    if (error != ErrorCode::success) {
      decorator_.reset();
    }
    return error;
  }

  // Outgoing slot-A production has completed on executor.
  [[nodiscard]] Status
  submit_outgoing(const executor_type executor, const storage_type &storage,
                  const outgoing_prefixes_type &prefixes) noexcept {
    return decorator_.submit_outgoing(executor, storage, prefixes);
  }
  template <bool Wait> [[nodiscard]] Status advance_outgoing() noexcept {
    return decorator_.template advance_outgoing<Wait>();
  }
  // After payload completion, or right after start when stream-ordered:
  // makes slot B available to the incoming consumer that the caller
  // enqueues next on executor. Never pending.
  [[nodiscard]] Status
  finish_incoming(const executor_type executor, storage_type &storage,
                  const incoming_prefixes_type &prefixes) noexcept {
    return decorator_.finish_incoming(executor, storage, prefixes);
  }
  // Requires completed slot-A production and slot-B consumption. The migration
  // driver obtains this proof from its pack/received-round completion before
  // launching independent resident compaction or starting the next round.
  template <class CountPort>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_payload(
      const executor_type executor, storage_type &storage,
      const outgoing_prefixes_type &outgoing_prefixes,
      const incoming_prefixes_type &incoming_prefixes,
      const CountPort &count) noexcept {
    if (!p2p_.reusable()) { return Status::failure(ErrorCode::invalid_state); }
    const Status admitted = decorator_.admit_payload(
        executor, storage, outgoing_prefixes, incoming_prefixes);
    if (!admitted.is_success()) { return admitted; }
    return p2p_.start(
        executor, outgoing_prefixes,
        decorator_.payload(std::as_const(storage), outgoing_prefixes),
        incoming_prefixes, decorator_.payload(storage, incoming_prefixes),
        count);
  }
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance_payload() noexcept {
    return p2p_.template advance<Wait>();
  }
  [[nodiscard]] Status recycle() noexcept { return decorator_.recycle(); }
  void reset_resources() noexcept {
    p2p_.reset();
    decorator_.reset();
  }
  void reset() noexcept { reset_resources(); }

private:
  template <class Plan>
  [[nodiscard]] ErrorCode prepare_decorator(HostMemoryBudget &budget,
                                            const Plan &plan,
                                            const prepared_storage_type &prepared,
                                            storage_type &storage) noexcept {
    return plan.pattern().routing().round_count() != 0
               ? decorator_.prepare(budget, prepared, storage)
               : ErrorCode::success;
  }

  [[no_unique_address]] Decorator decorator_{};
  P2p p2p_{};
};

} // namespace lpfx::distributed::detail

#endif
