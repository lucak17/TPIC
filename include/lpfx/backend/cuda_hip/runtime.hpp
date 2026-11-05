#ifndef LPFX_BACKEND_CUDA_HIP_RUNTIME_HPP
#define LPFX_BACKEND_CUDA_HIP_RUNTIME_HPP

// The vendor kernel runtime (built-in indices, barriers, launch syntax) and
// the matching LPFx backend traits, for translation units compiled by hipcc or
// nvcc. Launch leaves specialize on the backend, so they need its traits.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <lpfx/backend/hip/backend.hpp>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#include <lpfx/backend/cuda/backend.hpp>
#endif

#endif
