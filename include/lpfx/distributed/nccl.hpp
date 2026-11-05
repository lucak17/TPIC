#ifndef LPFX_DISTRIBUTED_NCCL_HPP
#define LPFX_DISTRIBUTED_NCCL_HPP

#include <lpfx/config.hpp>

#if !LPFX_CONFIG_ENABLE_NCCL || !LPFX_CONFIG_DEVICE_BACKEND_CUDA ||            \
    !defined(LPFX_DISTRIBUTED_ENABLE_NCCL_BACKEND) ||                          \
    !LPFX_DISTRIBUTED_ENABLE_NCCL_BACKEND
#error LPFX_NCCL_OPT_IN_REQUIRES_CONFIGURED_NCCL_COMPONENT
#elif !defined(__CUDACC__) || defined(__HIPCC__)
#error LPFX_NCCL_OPT_IN_REQUIRES_CUDA_COMPILATION
#else

#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/distributed/mpi.hpp>
#include <lpfx/distributed/transport/xccl/native.hpp>
#include <lpfx/distributed/composition/mpi_xccl.hpp>

static_assert(lpfx::distributed::detail::ConfiguredXcclTransport<
              lpfx::distributed::transport::Nccl>);

#endif
#endif
