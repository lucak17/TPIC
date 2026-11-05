#ifndef LPFX_BACKEND_CUDA_BACKEND_HPP
#define LPFX_BACKEND_CUDA_BACKEND_HPP

#define LPFX_NATIVE_HIP 0
#include <lpfx/backend/cuda_hip/native_traits.inc>

static_assert(lpfx::device::DeviceRuntimeBackend<lpfx::device::CudaBackend>);

#endif
