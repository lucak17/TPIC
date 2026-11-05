#ifndef LPFX_CORE_CHECKED_ARITHMETIC_HPP
#define LPFX_CORE_CHECKED_ARITHMETIC_HPP

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace lpfx::detail {

template <class T>
concept checked_integer =
    std::same_as<T, std::remove_cv_t<T>> &&
    (std::same_as<T, signed char> || std::same_as<T, unsigned char> ||
     std::same_as<T, short> || std::same_as<T, unsigned short> ||
     std::same_as<T, int> || std::same_as<T, unsigned int> ||
     std::same_as<T, long> || std::same_as<T, unsigned long> ||
     std::same_as<T, long long> || std::same_as<T, unsigned long long>);

// Each checked operation writes output only on success; output may alias an
// operand.
template <checked_integer T>
[[nodiscard]] constexpr bool checked_add(T lhs, T rhs, T &output) noexcept {
  T result{};
  if (__builtin_add_overflow(lhs, rhs, &result)) {
    return false;
  }
  output = result;
  return true;
}

template <checked_integer T>
[[nodiscard]] constexpr bool checked_subtract(T lhs, T rhs,
                                              T &output) noexcept {
  T result{};
  if (__builtin_sub_overflow(lhs, rhs, &result)) {
    return false;
  }
  output = result;
  return true;
}

template <checked_integer T>
[[nodiscard]] constexpr bool checked_multiply(T lhs, T rhs,
                                              T &output) noexcept {
  T result{};
  if (__builtin_mul_overflow(lhs, rhs, &result)) {
    return false;
  }
  output = result;
  return true;
}

template <checked_integer To, checked_integer From>
[[nodiscard]] constexpr bool checked_integral_cast(From input,
                                                   To &output) noexcept {
  if (!std::in_range<To>(input)) {
    return false;
  }
  output = static_cast<To>(input);
  return true;
}

[[nodiscard]] constexpr bool checked_align_up(std::size_t value,
                                              std::size_t alignment,
                                              std::size_t &output) noexcept {
  if (!std::has_single_bit(alignment)) {
    return false;
  }
  const std::size_t remainder = value & (alignment - 1);
  if (remainder == 0) {
    output = value;
    return true;
  }
  return checked_add(value, alignment - remainder, output);
}

template <class T>
  requires std::is_object_v<T> && requires { sizeof(T); }
[[nodiscard]] constexpr bool checked_byte_count(std::size_t count,
                                                std::size_t &output) noexcept {
  return checked_multiply(count, sizeof(T), output);
}

} // namespace lpfx::detail

#endif
