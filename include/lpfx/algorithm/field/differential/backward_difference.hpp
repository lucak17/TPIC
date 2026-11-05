#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_BACKWARD_DIFFERENCE_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_BACKWARD_DIFFERENCE_HPP

#include <concepts>
#include <cstddef>

#include <lpfx/core/attributes.hpp>
#include <lpfx/algorithm/field/differential/staggered_difference.hpp>

namespace lpfx {

// Negative-axis halo depth read by the selected formula.
template <std::size_t AccuracyOrder>
  requires(AccuracyOrder >= 1 && AccuracyOrder <= 3)
inline constexpr std::size_t backward_difference_required_negative_halo =
    AccuracyOrder;

// Collocated backward derivative at one sample from AccuracyOrder + 1
// consecutive samples ending at (i,j,k). Hard preconditions: inverse_spacing
// stays finite and positive as Accumulator, every source sample is
// addressable and the negative shifts are representable.
template <std::floating_point Accumulator, std::size_t AccuracyOrder,
          std::size_t Axis, class Component,
          std::floating_point InverseSpacing>
  requires(AccuracyOrder >= 1 && AccuracyOrder <= 3 && Axis < 3) &&
          detail::FloatingDifferentialComponent<Component, Accumulator>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
backward_axis_difference(
    const Component component,
    const typename Component::index_policy::coordinate_type i,
    const typename Component::index_policy::coordinate_type j,
    const typename Component::index_policy::coordinate_type k,
    const InverseSpacing inverse_spacing) noexcept {
  using Coordinate = typename Component::index_policy::coordinate_type;
  const Accumulator inverse = static_cast<Accumulator>(inverse_spacing);
  const Accumulator value_0 = detail::shifted_staggered_value<Axis, Accumulator>(
      component, i, j, k, Coordinate{0});
  const Accumulator value_minus_1 =
      detail::shifted_staggered_value<Axis, Accumulator>(
          component, i, j, k, Coordinate{-1});

  if constexpr (AccuracyOrder == 1) {
    return (value_0 - value_minus_1) * inverse;
  } else {
    const Accumulator value_minus_2 =
        detail::shifted_staggered_value<Axis, Accumulator>(
            component, i, j, k, Coordinate{-2});
    if constexpr (AccuracyOrder == 2) {
      return (Accumulator{3} * value_0 -
              Accumulator{4} * value_minus_1 + value_minus_2) *
             (inverse / Accumulator{2});
    } else {
      const Accumulator value_minus_3 =
          detail::shifted_staggered_value<Axis, Accumulator>(
              component, i, j, k, Coordinate{-3});
      return (Accumulator{11} * value_0 -
              Accumulator{18} * value_minus_1 +
              Accumulator{9} * value_minus_2 -
              Accumulator{2} * value_minus_3) *
             (inverse / Accumulator{6});
    }
  }
}

} // namespace lpfx

#endif
