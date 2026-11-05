#ifndef LPFX_CORE_FIXED_ARRAY_HPP
#define LPFX_CORE_FIXED_ARRAY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>

namespace lpfx::detail {

// Minimal fixed storage for descriptors consumed by both host and device code.
// libstdc++'s std::array accessors are not device-callable under the supported
// strict NVCC configuration, even though its object representation is suitable.
template <class T, std::size_t Extent>
  requires(Extent != 0) && std::is_trivially_copyable_v<T>
class DeviceFixedArray final {
public:
  LPFX_HOST_DEVICE constexpr DeviceFixedArray() noexcept {}

  constexpr explicit DeviceFixedArray(
      const std::array<T, Extent> &source) noexcept {
    for (std::size_t index = 0; index != Extent; ++index) {
      values_[index] = source[index];
    }
  }

  // Hard precondition: index < Extent.
  [[nodiscard]] LPFX_HOST_DEVICE constexpr T &
  operator[](const std::size_t index) noexcept {
    return values_[index];
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr const T &
  operator[](const std::size_t index) const noexcept {
    return values_[index];
  }

private:
  T values_[Extent]{};
};

static_assert(sizeof(DeviceFixedArray<std::uint8_t, 26>) == 26);
static_assert(std::is_standard_layout_v<DeviceFixedArray<std::uint64_t, 27>>);
static_assert(
    std::is_trivially_copyable_v<DeviceFixedArray<std::uint64_t, 27>>);

} // namespace lpfx::detail

#endif
