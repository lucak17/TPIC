#ifndef LPFX_DISTRIBUTED_DETAIL_BYTE_INTERVAL_HPP
#define LPFX_DISTRIBUTED_DETAIL_BYTE_INTERVAL_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <lpfx/core/error.hpp>

namespace lpfx::detail {

struct ByteInterval final {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

[[nodiscard]] constexpr bool disjoint(const ByteInterval left,
                                      const ByteInterval right) noexcept {
  return left.begin == left.end || right.begin == right.end ||
         left.end <= right.begin || right.end <= left.begin;
}

template <class T>
  requires std::is_object_v<T>
[[nodiscard]] constexpr ErrorCode
try_make_byte_interval(const T *const base, const std::size_t count,
                       ByteInterval &output) noexcept {
  if (count != 0 && base == nullptr) {
    return ErrorCode::invalid_argument;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    return ErrorCode::overflow;
  }
  const std::size_t byte_count = count * sizeof(T);
  if constexpr (sizeof(std::uintptr_t) < sizeof(std::size_t)) {
    if (byte_count >
        static_cast<std::size_t>(std::numeric_limits<std::uintptr_t>::max())) {
      return ErrorCode::overflow;
    }
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(base);
  const auto bytes = static_cast<std::uintptr_t>(byte_count);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return ErrorCode::overflow;
  }
  output = ByteInterval{begin, begin + bytes};
  return ErrorCode::success;
}

} // namespace lpfx::detail

#endif
