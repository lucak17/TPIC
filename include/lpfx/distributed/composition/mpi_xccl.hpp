#ifndef LPFX_DISTRIBUTED_COMPOSITION_MPI_XCCL_HPP
#define LPFX_DISTRIBUTED_COMPOSITION_MPI_XCCL_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <tuple>
#include <type_traits>

#include <mpi.h>

#include <lpfx/backend/properties.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/composition/mpi.hpp>
#include <lpfx/distributed/transport/workspace.hpp>
#include <lpfx/distributed/transport/xccl/collective.hpp>
#include <lpfx/distributed/transport/xccl/p2p.hpp>
#include <lpfx/distributed/transport/xccl/session.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

// The NCCL or RCCL opt-in defines the one configured route mapping.
template <class Transport> struct XcclTransportConfiguration;

template <class Transport>
concept ConfiguredXcclTransport =
    std::same_as<Transport, std::remove_cv_t<Transport>> && requires {
      typename XcclTransportConfiguration<Transport>::native_type;
      typename std::integral_constant<
          CommunicationLoweringIdentity,
          XcclTransportConfiguration<Transport>::lowering_identity>;
    } &&
    XcclNativeAdapter<
        typename XcclTransportConfiguration<Transport>::native_type>;

// Every node-local rank must drive a distinct physical device.
template <device::DeviceDiscoveryBackend Backend>
[[nodiscard]] ErrorCode
preflight_mpi_xccl_device_mapping(const CommunicationSetup &setup,
                                  MpiControlContext &control,
                                  HostMemoryBudget &budget) noexcept {
  ErrorCode local =
      setup.template try_verify_current_device_affinity<Backend>();
  const device::DeviceIdentity identity = setup.has_device_affinity()
                                              ? setup.device_identity()
                                              : device::DeviceIdentity{};

  {
    MpiCallLease lease{};
    if (try_acquire_shared_mpi_call(control.domain(), lease) !=
        ErrorCode::success) {
      std::abort();
    }

  MPI_Comm shared = MPI_COMM_NULL;
  if (MPI_Comm_split_type(control.native_handle(), MPI_COMM_TYPE_SHARED,
                          control.rank(), MPI_INFO_NULL,
                          &shared) != MPI_SUCCESS ||
      shared == MPI_COMM_NULL ||
      MPI_Comm_set_errhandler(shared, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
    std::abort();
  }

  int shared_size = 0;
  if (MPI_Comm_size(shared, &shared_size) != MPI_SUCCESS || shared_size <= 0) {
    std::abort();
  }
  local = agree_or_abort(shared, local);

  HostWorkspace<device::DeviceIdentity> workspace{};
  if (local == ErrorCode::success) {
    local = workspace.try_prepare(budget,
                                  {static_cast<std::size_t>(shared_size)});
  }
  device::DeviceIdentity *const identities = workspace.data<0>();
  const std::size_t identity_count = workspace.size<0>();

  local = agree_or_abort(shared, local);
  if (local == ErrorCode::success) {
    if (MPI_Allgather(&identity, static_cast<int>(sizeof(identity)), MPI_BYTE,
                      identities, static_cast<int>(sizeof(identity)), MPI_BYTE,
                      shared) != MPI_SUCCESS) {
      std::abort();
    }
    for (std::size_t left = 0; left != identity_count; ++left) {
      if (!identities[left].valid()) {
        local = ErrorCode::invalid_argument;
        break;
      }
      for (std::size_t right = left + 1; right != identity_count; ++right) {
        if (identities[left] == identities[right]) {
          local = ErrorCode::invalid_argument;
          break;
        }
      }
      if (local != ErrorCode::success) {
        break;
      }
    }
  }
  local = agree_or_abort(shared, local);

  workspace.reset();
    if (MPI_Comm_free(&shared) != MPI_SUCCESS || shared != MPI_COMM_NULL) {
      std::abort();
    }
  }
  return agree_or_abort(control, local);
}

enum class XcclLifecycleState : std::uint32_t {
  complete = 0,
  pending = 1,
  failed = 2,
};

template <XcclNativeAdapter Native>
[[nodiscard]] ErrorCode agree_mpi_xccl_abort(XcclSession<Native> &xccl,
                                             MpiControlContext &control) noexcept {
  const auto aborted = xccl.abort();
  return agree_or_abort(control, Native::succeeded(aborted)
                                     ? ErrorCode::success
                                     : ErrorCode::backend_failure);
}

// Polls a nonblocking initialization (parent and split) or finalization until
// every rank has completed, or aborts every handle once any rank has failed.
template <bool Finalizing, XcclNativeAdapter Native>
[[nodiscard]] ErrorCode
complete_mpi_xccl_lifecycle(XcclSession<Native> &xccl,
                            MpiControlContext &control,
                            typename Native::result_type native) noexcept {
  for (;;) {
    const XcclLifecycleState local =
        Native::succeeded(native)
            ? XcclLifecycleState::complete
            : (Native::in_progress(native) ? XcclLifecycleState::pending
                                           : XcclLifecycleState::failed);
    std::array<std::uint64_t, 1> state{static_cast<std::uint64_t>(local)};
    if (agree_contract(control, ErrorCode::success,
                       std::array<std::uint64_t, 0>{}, state) !=
            ErrorCode::success ||
        state[0] > static_cast<std::uint64_t>(XcclLifecycleState::failed)) {
      std::abort();
    }
    const auto agreed = static_cast<XcclLifecycleState>(state[0]);
    if (agreed == XcclLifecycleState::failed) {
      if (agree_mpi_xccl_abort(xccl, control) != ErrorCode::success) {
        std::abort();
      }
      if constexpr (Finalizing) {
        std::abort();
      }
      return ErrorCode::backend_failure;
    }
    if (agreed == XcclLifecycleState::complete) {
      return ErrorCode::success;
    }
    if (local == XcclLifecycleState::pending) {
      if constexpr (Finalizing) {
        native = xccl.progress_finalization();
      } else {
        native = xccl.progress_initialization();
      }
    }
  }
}

template <XcclNativeAdapter Native>
[[nodiscard]] ErrorCode try_prepare_mpi_xccl_native_session(
    XcclSession<Native> &xccl, const CommunicationSetup &setup,
    MpiControlContext &control, HostMemoryBudget &budget) noexcept {
  LPFX_TRY(preflight_mpi_xccl_device_mapping<
           typename Native::device_backend_type>(setup, control, budget));

  int version = 0;
  const bool compatible =
      Native::succeeded(Native::query_runtime_version(version)) &&
      version >= Native::minimum_runtime_version;
  LPFX_TRY(agree_or_abort(control, compatible ? ErrorCode::success
                                              : ErrorCode::backend_failure));

  typename Native::unique_id_type unique_id{};
  ErrorCode local = ErrorCode::success;
  if (control.rank() == 0 &&
      !Native::succeeded(Native::get_unique_id(unique_id))) {
    local = ErrorCode::backend_failure;
  }
  LPFX_TRY(agree_or_abort(control, local));
  if (broadcast_bytes(control,
                      std::as_writable_bytes(
                          std::span{&unique_id, std::size_t{1}}),
                      0) != ErrorCode::success) {
    std::abort();
  }

  return complete_mpi_xccl_lifecycle<false>(
      xccl, control,
      xccl.try_initialize(control.size(), unique_id, control.rank()));
}

// MPI control (and optional payload) plus the XCCL communicator, created only
// when some rank has XCCL work.
template <bool RetainMpiPayload, XcclNativeAdapter Native>
class CommunicationSession<RetainMpiPayload, Native> final {
public:
  constexpr CommunicationSession() noexcept = default;
  LPFX_NONMOVABLE(CommunicationSession);

  [[nodiscard]] bool empty() const noexcept {
    return mpi_.empty() && xccl_.empty();
  }
  [[nodiscard]] bool is_valid() const noexcept {
    return mpi_.is_valid() && (xccl_.empty() || xccl_.ready());
  }
  [[nodiscard]] MpiSession<RetainMpiPayload> &mpi() noexcept { return mpi_; }
  [[nodiscard]] const MpiSession<RetainMpiPayload> &mpi() const noexcept {
    return mpi_;
  }
  [[nodiscard]] XcclSession<Native> &xccl() noexcept { return xccl_; }
  [[nodiscard]] const XcclSession<Native> &xccl() const noexcept {
    return xccl_;
  }

  template <class... Plans>
  [[nodiscard]] ErrorCode
  try_open(const CommunicationSetup &setup, HostMemoryBudget &budget,
           const ErrorCode local, const std::uint64_t identity,
           const std::size_t tag_extent, CommunicationRequirements requirements,
           const std::tuple<const Plans *...> &plans) noexcept {
    LPFX_TRY(try_open_mpi_session(mpi_, setup, budget, local, identity,
                                  tag_extent, requirements, plans));
    if (!requirements.xccl) {
      return ErrorCode::success;
    }
    const ErrorCode prepared =
        try_prepare_mpi_xccl_native_session(xccl_, setup, mpi_.control(),
                                            budget);
    if (prepared != ErrorCode::success) {
      mpi_.deactivate();
    }
    return prepared;
  }

  [[nodiscard]] ErrorCode agree(const ErrorCode local) noexcept {
    return agree_or_abort(mpi_.control(), local);
  }

  void rollback() noexcept {
    const bool teardown_failed =
        !xccl_.empty() &&
        agree_mpi_xccl_abort(xccl_, mpi_.control()) != ErrorCode::success;
    mpi_.deactivate();
    if (teardown_failed) {
      std::abort();
    }
  }

  void close() noexcept {
    if (!xccl_.empty()) {
      static_cast<void>(complete_mpi_xccl_lifecycle<true>(
          xccl_, mpi_.control(), xccl_.try_finalize()));
      const auto destroyed = xccl_.destroy_finalized();
      if (agree_or_abort(mpi_.control(), Native::succeeded(destroyed)
                                             ? ErrorCode::success
                                             : ErrorCode::backend_failure) !=
          ErrorCode::success) {
        std::abort();
      }
    }
    mpi_.deactivate();
  }

private:
  MpiSession<RetainMpiPayload> mpi_{};
  XcclSession<Native> xccl_{};
};

template <class Binding>
using xccl_configuration_t =
    XcclTransportConfiguration<typename Binding::transport_type>;
template <class Binding>
using xccl_native_t = typename xccl_configuration_t<Binding>::native_type;

// XCCL ports address device operands directly. A stream-ordered route
// completes fixed exchanges and collectives at enqueue (host stage); otherwise
// an event recorded after the native call carries completion.
template <class Binding, operation_family = Binding::family>
struct XcclLowering {
  using port_type = FixedBytePort<
      std::conditional_t<
          xccl_configuration_t<Binding>::completes_on_stream,
          HostFixedStage<typename Binding::executor_type,
                         typename xccl_native_t<Binding>::device_backend_type>,
          EventFixedStage<
              typename Binding::executor_type,
              typename xccl_native_t<Binding>::device_backend_type>>,
      XcclP2p<xccl_native_t<Binding>>>;
  static constexpr std::size_t tag_extent = 0;
};
template <class Binding>
struct XcclLowering<Binding, operation_family::collective> {
  using port_type = CollectivePort<
      std::conditional_t<
          xccl_configuration_t<Binding>::completes_on_stream,
          HostCollectiveStage<
              collective_value_t<Binding>, typename Binding::executor_type,
              typename xccl_native_t<Binding>::device_backend_type>,
          EventCollectiveStage<
              collective_value_t<Binding>, typename Binding::executor_type,
              typename xccl_native_t<Binding>::device_backend_type>>,
      XcclCollectiveCall<xccl_native_t<Binding>,
                         collective_descriptor_t<Binding>::kind,
                         collective_descriptor_t<Binding>::reduction,
                         collective_value_t<Binding>,
                         Binding::resource_type::exact_in_place>>;
  static constexpr std::size_t tag_extent = 0;
};
template <class Binding>
struct XcclLowering<Binding, operation_family::particle> {
  using port_type = ParticlePayloadPort<
      DeviceParticleDecorator<
          particle_contract_t<Binding>, typename Binding::executor_type,
          typename xccl_native_t<Binding>::device_backend_type>,
      XcclParticleP2p<xccl_native_t<Binding>, particle_contract_t<Binding>>>;
  // The count exchange still uses one MPI tag.
  static constexpr std::size_t tag_extent = 1;
};

template <class Binding>
  requires is_bound_operation<Binding> &&
           ConfiguredXcclTransport<typename Binding::transport_type> &&
           std::same_as<
               typename Binding::space_type,
               DeviceSpace<typename xccl_native_t<Binding>::device_backend_type>>
struct CommunicationEntryLowering<Binding> final
    : LoweredProgram<
          family_program_t<Binding,
                           typename XcclLowering<Binding>::port_type>,
          xccl_configuration_t<Binding>::lowering_identity,
          XcclLowering<Binding>::tag_extent, xccl_native_t<Binding>> {};

} // namespace lpfx::distributed::detail

#endif
