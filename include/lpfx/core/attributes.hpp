#ifndef LPFX_CORE_ATTRIBUTES_HPP
#define LPFX_CORE_ATTRIBUTES_HPP

// Keep compiler-specific spelling at this boundary.  Public headers use these
// macros directly instead of defining subsystem-local variants.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LPFX_HOST_DEVICE __host__ __device__
// Native MockBackend tests execute device-only callables on the host even
// when a cluster C++ wrapper parses the translation unit in accelerator mode.
// Keep the complete emulated call graph host-only in every compiler pass.
// Native device compile tests do not define this test-only switch.
#if defined(LPFX_TEST_DEVICE_HOST_EMULATION)
#define LPFX_DEVICE
#else
#define LPFX_DEVICE __device__
#endif
#else
#define LPFX_HOST_DEVICE
#define LPFX_DEVICE
#endif

// Precedes an LPFX_HOST_DEVICE template that invokes a caller-supplied
// callable or constructor, which may be host-only or device-only.
#if defined(__CUDACC__) && !defined(__clang__)
#define LPFX_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#else
#define LPFX_EXEC_CHECK_DISABLE
#endif

// 1 while CUDA/HIP compiles the translation unit for a device architecture.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define LPFX_DEVICE_PASS 1
#else
#define LPFX_DEVICE_PASS 0
#endif

// Device passes parse host-only exception adapters without generating them.
#if defined(__cpp_exceptions) || LPFX_DEVICE_PASS
#define LPFX_HAS_EXCEPTIONS 1
#else
#define LPFX_HAS_EXCEPTIONS 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LPFX_ALWAYS_INLINE inline __attribute__((always_inline))
#define LPFX_NOINLINE __attribute__((noinline))
#define LPFX_RESTRICT __restrict__
#else
#define LPFX_ALWAYS_INLINE inline
#define LPFX_NOINLINE
#define LPFX_RESTRICT
#endif

// Declares Type neither copyable nor movable.
#define LPFX_NONMOVABLE(Type)                                                  \
  Type(const Type &) = delete;                                                 \
  Type &operator=(const Type &) = delete;                                      \
  Type(Type &&) = delete;                                                      \
  Type &operator=(Type &&) = delete

// HIP-Clang compiles each translation unit once for the host and once per
// device architecture. Preserve the host independence contract under hipcc,
// but do not attach a host-vectorization pragma to a HIP device pass. Keep
// Clang-CUDA suppressed because its host and device pragma support differs.
//
// Clang reports a failed explicitly requested vectorization as -Wpass-failed.
// Address/thread/memory sanitizer instrumentation can make otherwise valid
// independent loops deliberately non-vectorizable, so do not issue the hint in
// those diagnostic builds. Optimized nonsanitized builds retain the hint and
// the dedicated code-generation test verifies its generated-loop contract.
#if defined(__clang__)
#if __has_feature(address_sanitizer) ||                                        \
    __has_feature(undefined_behavior_sanitizer) ||                             \
    __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer) ||      \
    __has_feature(hwaddress_sanitizer)
#define LPFX_DETAIL_CLANG_SANITIZER_ACTIVE 1
#else
#define LPFX_DETAIL_CLANG_SANITIZER_ACTIVE 0
#endif
#else
#define LPFX_DETAIL_CLANG_SANITIZER_ACTIVE 0
#endif

#if defined(__clang__) && !defined(__CUDACC__) &&                              \
    (!defined(__HIPCC__) || !defined(__HIP_DEVICE_COMPILE__)) &&               \
    !LPFX_DETAIL_CLANG_SANITIZER_ACTIVE
#define LPFX_INDEPENDENT_LOOP                                                  \
  _Pragma("clang loop vectorize(assume_safety) interleave(enable)")
#elif defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__) &&      \
    !defined(__HIPCC__)
#define LPFX_INDEPENDENT_LOOP _Pragma("GCC ivdep")
#else
#define LPFX_INDEPENDENT_LOOP
#endif

#undef LPFX_DETAIL_CLANG_SANITIZER_ACTIVE

#endif
