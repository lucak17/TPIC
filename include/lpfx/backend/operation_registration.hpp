#ifndef LPFX_BACKEND_OPERATION_REGISTRATION_HPP
#define LPFX_BACKEND_OPERATION_REGISTRATION_HPP

// Custom backends provide their trait specializations through this optional
// hook. Native algorithms include only their own launch implementation after
// their neutral declarations.
#if defined(LPFX_DEVICE_OPERATION_REGISTRATION_HEADER)
#define LPFX_NATIVE_DEVICE_OPERATIONS 0
#include LPFX_DEVICE_OPERATION_REGISTRATION_HEADER
#elif !defined(LPFX_TEST_DEVICE_HOST_EMULATION) && \
    (defined(__CUDACC__) || defined(__HIPCC__))
#define LPFX_NATIVE_DEVICE_OPERATIONS 1
#if defined(__HIPCC__)
#include <lpfx/backend/hip/backend.hpp>
#else
#include <lpfx/backend/cuda/backend.hpp>
#endif
#else
#define LPFX_NATIVE_DEVICE_OPERATIONS 0
#endif

#endif
