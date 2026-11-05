#ifndef LPFX_BACKEND_BACKEND_HPP
#define LPFX_BACKEND_BACKEND_HPP

#include <lpfx/config.hpp>

#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#error LPFX_CUDA_BACKEND_CANNOT_BE_COMPILED_AS_HIP
#endif
#include <lpfx/backend/cuda/backend.hpp>
#define LPFX_DETAIL_ACTIVE_BACKEND CudaBackend
#elif LPFX_CONFIG_DEVICE_BACKEND_HIP
#if defined(__CUDACC__) && !defined(__HIPCC__)
#error LPFX_HIP_BACKEND_CANNOT_BE_COMPILED_AS_CUDA
#endif
#include <lpfx/backend/hip/backend.hpp>
#define LPFX_DETAIL_ACTIVE_BACKEND HipBackend
#else
#error LPFX_DEVICE_BACKEND_REQUIRES_CUDA_OR_HIP
#endif

// A translation unit compiled by hipcc or nvcc also receives the vendor kernel
// runtime (built-in indices, atomics, launch syntax), so application kernels
// need no vendor include of their own. Host compilers see no kernel runtime.
#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/handles.hpp>

namespace lpfx::device {

using ActiveBackend = LPFX_DETAIL_ACTIVE_BACKEND;
using ActiveBackendTraits = BackendTraits<ActiveBackend>;
using ActiveExecutor = Executor<ActiveBackend>;

static_assert(sizeof(ActiveExecutor) ==
                  sizeof(ActiveBackendTraits::stream_type) &&
              sizeof(DeviceStream<ActiveBackend>) ==
                  sizeof(ActiveBackendTraits::stream_type) &&
              sizeof(DeviceEvent<ActiveBackend>) ==
                  sizeof(ActiveBackendTraits::event_type));

} // namespace lpfx::device

#undef LPFX_DETAIL_ACTIVE_BACKEND

#endif
