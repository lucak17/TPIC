#ifndef LPFX_DISTRIBUTED_RCCL_HPP
#define LPFX_DISTRIBUTED_RCCL_HPP

#include <lpfx/config.hpp>

#if !LPFX_CONFIG_ENABLE_RCCL || !LPFX_CONFIG_DEVICE_BACKEND_HIP ||            \
    !defined(LPFX_DISTRIBUTED_ENABLE_RCCL_BACKEND) ||                          \
    !LPFX_DISTRIBUTED_ENABLE_RCCL_BACKEND
#error LPFX_RCCL_OPT_IN_REQUIRES_CONFIGURED_RCCL_COMPONENT
#elif !defined(__HIPCC__)
#error LPFX_RCCL_OPT_IN_REQUIRES_HIP_COMPILATION
#else

#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/distributed/mpi.hpp>
#include <lpfx/distributed/transport/xccl/native.hpp>
#include <lpfx/distributed/composition/mpi_xccl.hpp>

static_assert(lpfx::distributed::detail::ConfiguredXcclTransport<
              lpfx::distributed::transport::Rccl>);

#endif
#endif
