#ifndef LPFX_CORE_META_HPP
#define LPFX_CORE_META_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx::meta {

template <class... Types> struct type_list final {};

template <std::size_t Index>
using index_constant = std::integral_constant<std::size_t, Index>;

template <class T, class... Types>
inline constexpr bool contains = (std::same_as<T, Types> || ...);

template <class... Types> inline constexpr bool all_distinct = true;

template <class First, class... Rest>
inline constexpr bool all_distinct<First, Rest...> =
    !contains<First, Rest...> && all_distinct<Rest...>;

namespace detail {

template <class T, class... Types>
[[nodiscard]] consteval std::size_t first_index_of() noexcept {
  constexpr bool matches[]{std::same_as<T, Types>...};
  std::size_t index = 0;
  while (!matches[index]) {
    ++index;
  }
  return index;
}

LPFX_EXEC_CHECK_DISABLE
template <class Function, std::size_t... Index>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr void
for_each_index(Function &function, std::index_sequence<Index...>) {
  (static_cast<void>(function(index_constant<Index>{})), ...);
}

LPFX_EXEC_CHECK_DISABLE
template <class Function, std::size_t... Index>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr void
reverse_for_each(Function &function, std::index_sequence<Index...>) {
  (static_cast<void>(
       function(index_constant<sizeof...(Index) - 1 - Index>{})),
   ...);
}

LPFX_EXEC_CHECK_DISABLE
template <class Function, std::size_t... Index>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr ErrorCode
for_each_until_error(Function &function, std::index_sequence<Index...>) {
  ErrorCode error = ErrorCode::success;
  static_cast<void>(
      (((error = function(index_constant<Index>{})) == ErrorCode::success) &&
       ...));
  return error;
}

} // namespace detail

// Position of the first T in Types.
template <class T, class... Types>
  requires contains<T, Types...>
inline constexpr std::size_t index_of = detail::first_index_of<T, Types...>();

// Calls function(index_constant<I>{}) for I = 0, ..., Count - 1.
LPFX_EXEC_CHECK_DISABLE
template <std::size_t Count, class Function>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr void
for_each_index(Function &&function) {
  detail::for_each_index(function, std::make_index_sequence<Count>{});
}

// Calls function(index_constant<I>{}) for I = Count - 1, ..., 0.
LPFX_EXEC_CHECK_DISABLE
template <std::size_t Count, class Function>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr void
reverse_for_each(Function &&function) {
  detail::reverse_for_each(function, std::make_index_sequence<Count>{});
}

// Calls function(index_constant<I>{}) -> ErrorCode in index order and returns
// the first failure without calling the remaining indices.
LPFX_EXEC_CHECK_DISABLE
template <std::size_t Count, class Function>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr ErrorCode
for_each_until_error(Function &&function) {
  return detail::for_each_until_error(function,
                                      std::make_index_sequence<Count>{});
}

} // namespace lpfx::meta

#endif
