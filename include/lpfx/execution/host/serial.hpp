#ifndef LPFX_EXECUTION_HOST_SERIAL_HPP
#define LPFX_EXECUTION_HOST_SERIAL_HPP

#include <concepts>

#include <lpfx/core/attributes.hpp>
#include <lpfx/domain/cartesian/index_policy.hpp>

namespace lpfx {

namespace detail {

template <class Function, class Index>
concept NothrowHostIndexFunction =
    requires(Function &function, const Index index) {
      { function(Index{index}) } noexcept -> std::same_as<void>;
    };

} // namespace detail

class SerialExecutor final {};

// Iterations must be independent: no loop-carried data dependencies and no
// reliance on callback order. This lets host compilers vectorize through the
// same zero-state executor interface as the serial ordered traversal.
class UnsequencedExecutor final {};

template <detail::CartesianIndexPolicy IndexPolicy, class Count, class Function>
  requires std::same_as<Count, typename IndexPolicy::count_type> &&
           detail::NothrowHostIndexFunction<Function, Count>
LPFX_ALWAYS_INLINE constexpr void for_each_index(const SerialExecutor &,
                                                 const Count count,
                                                 Function &&function) noexcept {
  for (Count index = Count{0}; index < count; ++index) {
    function(Count{index});
  }
}

template <detail::CartesianIndexPolicy IndexPolicy, class Count, class Function>
  requires std::same_as<Count, typename IndexPolicy::count_type> &&
           detail::NothrowHostIndexFunction<Function, Count>
LPFX_ALWAYS_INLINE constexpr void
for_each_index(const UnsequencedExecutor &, const Count count,
               Function &&function) noexcept {
  LPFX_INDEPENDENT_LOOP
  for (Count index = Count{0}; index < count; ++index) {
    function(Count{index});
  }
}

} // namespace lpfx

#endif
