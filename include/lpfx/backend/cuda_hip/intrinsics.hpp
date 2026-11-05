#ifndef LPFX_BACKEND_CUDA_HIP_INTRINSICS_HPP
#define LPFX_BACKEND_CUDA_HIP_INTRINSICS_HPP

#include <cstdint>
#include <type_traits>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/core/attributes.hpp>

namespace lpfx::device {

// This is the scalar intersection implemented by CUDA and HIP's __ldg
// overload sets. In particular, not every C++ arithmetic type is accepted:
// bool, wchar_t, and the charN_t types have no portable overload.
template <class T>
concept read_only_load_scalar =
    std::is_same_v<T, char> || std::is_same_v<T, signed char> ||
    std::is_same_v<T, unsigned char> || std::is_same_v<T, short> ||
    std::is_same_v<T, unsigned short> || std::is_same_v<T, int> ||
    std::is_same_v<T, unsigned int> || std::is_same_v<T, long> ||
    std::is_same_v<T, unsigned long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, unsigned long long> || std::is_same_v<T, float> ||
    std::is_same_v<T, double>;

// Read data which is immutable for the lifetime of the kernel. CUDA can route
// this through its read-only path; HIP provides the same source-level
// intrinsic and selects the appropriate native load for its target.
template <read_only_load_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
read_only_load(const T *const source) noexcept {
#if LPFX_DEVICE_PASS
  return __ldg(source);
#else
  return *source;
#endif
}

// Return the upper half of the full 32 x 32-bit unsigned product. Both native
// backends lower this intrinsic directly to multiply-high instructions.
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE std::uint32_t
unsigned_multiply_high(const std::uint32_t left,
                       const std::uint32_t right) noexcept {
#if LPFX_DEVICE_PASS
  static_assert(sizeof(unsigned int) == sizeof(std::uint32_t));
  return static_cast<std::uint32_t>(
      __umulhi(static_cast<unsigned int>(left),
               static_cast<unsigned int>(right)));
#else
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(left) *
       static_cast<std::uint64_t>(right)) >>
      32U);
#endif
}

} // namespace lpfx::device

#endif
