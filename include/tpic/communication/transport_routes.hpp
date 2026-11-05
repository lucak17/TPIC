#ifndef TPIC_COMMUNICATION_TRANSPORT_ROUTES_HPP
#define TPIC_COMMUNICATION_TRANSPORT_ROUTES_HPP

// The build selects available native support; each registered binding still
// names its exact route explicitly. No route value is stored at runtime.
#if defined(TPIC_CONFIG_COMMUNICATION_NCCL) && TPIC_CONFIG_COMMUNICATION_NCCL
#include <lpfx/distributed/nccl.hpp>
#elif defined(TPIC_CONFIG_COMMUNICATION_RCCL) && TPIC_CONFIG_COMMUNICATION_RCCL
#include <lpfx/distributed/rccl.hpp>
#else
#include <lpfx/distributed/mpi.hpp>
#endif

#include <string_view>

namespace tpic::communication {

#if defined(TPIC_CONFIG_COMMUNICATION_NCCL) && TPIC_CONFIG_COMMUNICATION_NCCL
using DeviceRoute =
    lpfx::distributed::transport::StreamOrderedCompletion<lpfx::distributed::transport::Nccl>;
using GmresRoute = DeviceRoute;
inline constexpr std::string_view payload_backend_name = "NCCL";
inline constexpr int payload_backend_id = 1;
#elif defined(TPIC_CONFIG_COMMUNICATION_RCCL) && TPIC_CONFIG_COMMUNICATION_RCCL
using DeviceRoute =
    lpfx::distributed::transport::StreamOrderedCompletion<lpfx::distributed::transport::Rccl>;
using GmresRoute = DeviceRoute;
inline constexpr std::string_view payload_backend_name = "RCCL";
inline constexpr int payload_backend_id = 2;
#else
using DeviceRoute = lpfx::distributed::transport::MpiDirect<
    lpfx::distributed::StreamOrderedMpiOrdering>;
// Only the iterative-solver reductions use pinned host staging in the
// device-aware MPI build. Fields, particles, mass and other reductions remain
// device-direct.
using GmresRoute = lpfx::distributed::transport::MpiStaged;
inline constexpr std::string_view payload_backend_name = "device-aware MPI";
inline constexpr int payload_backend_id = 0;
#endif

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_TRANSPORT_ROUTES_HPP
