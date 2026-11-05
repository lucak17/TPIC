#ifndef LPFX_DISTRIBUTED_COMPOSITION_MPI_HPP
#define LPFX_DISTRIBUTED_COMPOSITION_MPI_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/error.hpp>
#include <lpfx/distributed/binding/collective.hpp>
#include <lpfx/distributed/binding/field.hpp>
#include <lpfx/distributed/binding/message.hpp>
#include <lpfx/distributed/binding/particle.hpp>
#include <lpfx/distributed/collective/program.hpp>
#include <lpfx/distributed/communication.hpp>
#include <lpfx/distributed/composition/entry.hpp>
#include <lpfx/distributed/detail/fixed_exchange_program.hpp>
#include <lpfx/distributed/field/device.hpp>
#include <lpfx/distributed/field/host.hpp>
#include <lpfx/distributed/field/program.hpp>
#include <lpfx/distributed/message/program.hpp>
#include <lpfx/distributed/particle/migration/device/execution.hpp>
#include <lpfx/distributed/particle/migration/program.hpp>
#include <lpfx/distributed/particle/migration/host/execution.hpp>
#include <lpfx/distributed/transport/collective.hpp>
#include <lpfx/distributed/transport/fixed.hpp>
#include <lpfx/distributed/transport/mpi/agreement.hpp>
#include <lpfx/distributed/transport/mpi/collective.hpp>
#include <lpfx/distributed/transport/mpi/communicator.hpp>
#include <lpfx/distributed/transport/mpi/p2p.hpp>
#include <lpfx/distributed/transport/mpi/particle.hpp>
#include <lpfx/distributed/transport/mpi/registration.hpp>
#include <lpfx/distributed/transport/particle.hpp>
#include <lpfx/distributed/transport/routes.hpp>
#include <lpfx/distributed/transport/stage.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed::detail {

// Every context of the session must address the whole tag range.
template <bool RetainMpiPayload>
[[nodiscard]] ErrorCode
agree_communication_tag_capacity(MpiSession<RetainMpiPayload> &session,
                                 const std::size_t tag_extent) noexcept {
  const auto fits = [tag_extent](const auto &context) noexcept {
    return tag_extent == 0 ||
           tag_extent - 1 <= static_cast<std::size_t>(context.tag_upper_bound());
  };
  bool local = fits(session.control());
  if constexpr (RetainMpiPayload) {
    local = local && (!session.has_payload_context() || fits(session.payload()));
  }
  return agree_or_abort(session.control(), local ? ErrorCode::success
                                                 : ErrorCode::capacity_exceeded);
}

// One agreement of the local validation error, the communication identity
// {h, ~h} and the rank-global native work, then registration and activation
// of the MPI session. requirements receives the agreed native work.
template <bool RetainMpiPayload, class... Plans>
[[nodiscard]] ErrorCode try_open_mpi_session(
    MpiSession<RetainMpiPayload> &session, const CommunicationSetup &setup,
    HostMemoryBudget &budget, const ErrorCode local,
    const std::uint64_t identity, const std::size_t tag_extent,
    CommunicationRequirements &requirements,
    const std::tuple<const Plans *...> &plans) noexcept {
  if (!setup.is_valid() || setup.domain().registration_sealed()) {
    return ErrorCode::invalid_state;
  }
  std::array<std::uint64_t, 2> work{requirements.mpi_payload,
                                    requirements.xccl};
  LPFX_TRY(agree_contract(setup, local, std::array<std::uint64_t, 1>{identity},
                          work));
  requirements = {work[0] != 0, work[1] != 0};
  MpiRegistrationSeal seal{};
  LPFX_TRY(try_seal_mpi_registration(setup, plans, budget, seal));
  LPFX_TRY(session.try_activate(setup, std::move(seal),
                                requirements.mpi_payload));
  const ErrorCode tags = agree_communication_tag_capacity(session, tag_extent);
  if (tags != ErrorCode::success) {
    session.deactivate();
  }
  return tags;
}

// MPI-only session: the control context and, when any lowering needs it, the
// shared payload context.
template <bool RetainMpiPayload>
class CommunicationSession<RetainMpiPayload, void> final {
public:
  constexpr CommunicationSession() noexcept = default;
  LPFX_NONMOVABLE(CommunicationSession);

  [[nodiscard]] bool empty() const noexcept { return mpi_.empty(); }
  [[nodiscard]] bool is_valid() const noexcept { return mpi_.is_valid(); }
  [[nodiscard]] MpiSession<RetainMpiPayload> &mpi() noexcept { return mpi_; }
  [[nodiscard]] const MpiSession<RetainMpiPayload> &mpi() const noexcept {
    return mpi_;
  }

  template <class... Plans>
  [[nodiscard]] ErrorCode
  try_open(const CommunicationSetup &setup, HostMemoryBudget &budget,
           const ErrorCode local, const std::uint64_t identity,
           const std::size_t tag_extent, CommunicationRequirements requirements,
           const std::tuple<const Plans *...> &plans) noexcept {
    return try_open_mpi_session(mpi_, setup, budget, local, identity,
                                tag_extent, requirements, plans);
  }
  [[nodiscard]] ErrorCode agree(const ErrorCode local) noexcept {
    return agree_or_abort(mpi_.control(), local);
  }
  void rollback() noexcept { mpi_.deactivate(); }
  void close() noexcept { mpi_.deactivate(); }

private:
  MpiSession<RetainMpiPayload> mpi_{};
};

// Family program over a native port; shared by every route.
template <class Binding, class Port,
          operation_family Family = Binding::family>
struct FamilyProgram;

template <class Binding, class Port>
struct FamilyProgram<Binding, Port, operation_family::collective> final {
  using type = CollectiveProgram<typename Binding::plan_type::descriptor_type,
                                 typename Binding::resource_type,
                                 typename Binding::executor_type, Port>;
};

template <class Binding, class Space = typename Binding::space_type>
struct FieldExecutionFor;
template <class Binding> struct FieldExecutionFor<Binding, HostSpace> final {
  using type = HostFieldExecution<typename Binding::plan_type::pattern_type,
                                  typename Binding::resource_type>;
};
template <class Binding, class Backend>
struct FieldExecutionFor<Binding, DeviceSpace<Backend>> final {
  using type = DeviceFieldExecution<typename Binding::plan_type::pattern_type,
                                    typename Binding::resource_type,
                                    typename Binding::executor_type>;
};

template <class Binding, class Port>
struct FamilyProgram<Binding, Port, operation_family::field> final {
  using type =
      FixedExchangeProgram<Port,
                           FieldExchange<typename FieldExecutionFor<Binding>::type>,
                           typename Binding::executor_type>;
};

template <class Binding, class Port>
struct FamilyProgram<Binding, Port, operation_family::message> final {
  using type =
      FixedExchangeProgram<Port, MessageExchange<typename Binding::resource_type>,
                           typename Binding::executor_type>;
};

template <class Binding, class Space = typename Binding::space_type>
struct ParticleExecutionFor;
// Executions key on the wire contract and rank, so plans that differ only in
// channel or descriptor identity share one execution and its kernels.
template <class Binding> struct ParticleExecutionFor<Binding, HostSpace> final {
  using type = HostParticleExecution<typename Binding::plan_type::contract_type,
                                     typename Binding::plan_type::rank_type,
                                     typename Binding::resource_type,
                                     typename Binding::executor_type,
                                     typename Binding::semantics_type>;
};
template <class Binding, class Backend>
struct ParticleExecutionFor<Binding, DeviceSpace<Backend>> final {
  using type =
      DeviceParticleExecution<typename Binding::plan_type::contract_type,
                              typename Binding::plan_type::rank_type,
                              typename Binding::resource_type,
                              typename Binding::executor_type,
                              typename Binding::semantics_type>;
};

// Counts travel on the MPI control context; Port carries the payload.
template <class Binding, class Port>
struct FamilyProgram<Binding, Port, operation_family::particle> final {
  using type = ParticleProgram<
      typename ParticleExecutionFor<Binding>::type,
      MpiParticleCountPort<typename Binding::plan_type::pattern_type>, Port>;
};

template <class Binding, class Port>
using family_program_t = typename FamilyProgram<Binding, Port>::type;

template <class Binding>
using collective_value_t = typename Binding::resource_type::value_type;
template <class Binding>
using collective_descriptor_t = typename Binding::plan_type::descriptor_type;
template <class Binding>
using particle_contract_t = typename Binding::plan_type::contract_type;

// Native ports of one MPI route and memory space. Unsupported pairs stay
// incomplete; only collectives accept the nonblocking wrapper.
template <class Route, class Space> struct MpiRouteTraits;

template <> struct MpiRouteTraits<transport::MpiHost, HostSpace> {
  static constexpr bool nonblocking = false;
  static constexpr auto identity = CommunicationLoweringIdentity::mpi_host;
  static constexpr auto collective_identity =
      CommunicationLoweringIdentity::mpi_host_blocking_collective;
  static constexpr auto nonblocking_collective_identity = identity;

  template <class B>
  using fixed_port =
      FixedBytePort<HostFixedStage<typename B::executor_type>, MpiP2p>;
  template <class B>
  using particle_port = ParticlePayloadPort<
      HostParticleDecorator<particle_contract_t<B>, typename B::executor_type>,
      MpiParticleP2p<particle_contract_t<B>>>;
  template <class B, bool Nonblocking>
  using collective_port = CollectivePort<
      HostCollectiveStage<collective_value_t<B>, typename B::executor_type>,
      MpiCollectiveCall<collective_descriptor_t<B>::kind,
                        collective_descriptor_t<B>::reduction,
                        collective_value_t<B>,
                        B::resource_type::exact_in_place, Nonblocking>>;
};

template <class Backend>
struct MpiRouteTraits<transport::MpiStaged, DeviceSpace<Backend>> {
  static constexpr bool nonblocking = false;
  static constexpr auto identity = CommunicationLoweringIdentity::mpi_staged;
  static constexpr auto collective_identity =
      CommunicationLoweringIdentity::mpi_staged_blocking_collective;
  static constexpr auto nonblocking_collective_identity = identity;

  template <class B>
  using fixed_port = FixedBytePort<
      StagedFixedStage<Backend, typename B::executor_type,
                       B::family == operation_family::message>,
      MpiP2p>;
  template <class B>
  using particle_port = ParticlePayloadPort<
      StagedParticleDecorator<particle_contract_t<B>,
                              typename B::executor_type, Backend>,
      MpiParticleP2p<particle_contract_t<B>>>;
  template <class B, bool Nonblocking>
  using collective_port = CollectivePort<
      StagedCollectiveStage<collective_descriptor_t<B>::kind,
                            collective_value_t<B>, Backend,
                            typename B::executor_type>,
      MpiCollectiveCall<collective_descriptor_t<B>::kind,
                        collective_descriptor_t<B>::reduction,
                        collective_value_t<B>, false, Nonblocking>>;
};

template <class Ordering, class Backend>
struct MpiRouteTraits<transport::MpiDirect<Ordering>, DeviceSpace<Backend>> {
  static constexpr bool nonblocking = false;
  static constexpr auto identity = CommunicationLoweringIdentity::mpi_direct;
  static constexpr auto collective_identity = identity;
  static constexpr auto nonblocking_collective_identity =
      CommunicationLoweringIdentity::mpi_direct_nonblocking_collective;

  template <class B>
  using fixed_port = FixedBytePort<
      DirectFixedStage<Ordering, Backend, typename B::executor_type,
                       B::family == operation_family::message>,
      MpiP2p>;
  template <class B>
  using particle_port = ParticlePayloadPort<
      DirectParticleDecorator<Ordering, particle_contract_t<B>,
                              typename B::executor_type, Backend>,
      MpiParticleP2p<particle_contract_t<B>>>;
  template <class B, bool Nonblocking>
  using collective_port = CollectivePort<
      DirectCollectiveStage<Ordering, collective_value_t<B>, Backend,
                            typename B::executor_type>,
      MpiCollectiveCall<collective_descriptor_t<B>::kind,
                        collective_descriptor_t<B>::reduction,
                        collective_value_t<B>,
                        B::resource_type::exact_in_place, Nonblocking>>;
};

template <class Route, class Space>
  requires requires { MpiRouteTraits<Route, Space>::identity; } &&
           (!MpiRouteTraits<Route, Space>::nonblocking)
struct MpiRouteTraits<transport::NonblockingMpiCollectives<Route>, Space>
    : MpiRouteTraits<Route, Space> {
  static constexpr bool nonblocking = true;
};

template <class Binding>
using MpiRouteFor = MpiRouteTraits<typename Binding::transport_type,
                                   typename Binding::space_type>;

template <class Route, class Binding, operation_family = Binding::family>
struct MpiLowering {
  using port_type = typename Route::template fixed_port<Binding>;
  static constexpr auto identity = Route::identity;
  static constexpr std::size_t tag_extent = 1;
};
template <class Route, class Binding>
struct MpiLowering<Route, Binding, operation_family::collective> {
  using port_type =
      typename Route::template collective_port<Binding, Route::nonblocking>;
  static constexpr auto identity =
      Route::nonblocking ? Route::nonblocking_collective_identity
                         : Route::collective_identity;
  static constexpr std::size_t tag_extent = 0;
};
template <class Route, class Binding>
struct MpiLowering<Route, Binding, operation_family::particle> {
  using port_type = typename Route::template particle_port<Binding>;
  static constexpr auto identity = Route::identity;
  static constexpr std::size_t tag_extent = 2;
};

template <class Binding>
  requires is_bound_operation<Binding> &&
           requires { MpiRouteFor<Binding>::identity; } &&
           (Binding::family == operation_family::collective ||
            !MpiRouteFor<Binding>::nonblocking)
struct CommunicationEntryLowering<Binding> final
    : LoweredProgram<
          family_program_t<
              Binding,
              typename MpiLowering<MpiRouteFor<Binding>, Binding>::port_type>,
          MpiLowering<MpiRouteFor<Binding>, Binding>::identity,
          MpiLowering<MpiRouteFor<Binding>, Binding>::tag_extent> {};

} // namespace lpfx::distributed::detail

#endif
