#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_STAGGERED_DIFFERENCE_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_STAGGERED_DIFFERENCE_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx {

namespace detail {

template <std::size_t Axis, CartesianLocation SourceLocation,
          CartesianLocation TargetLocation>
  requires(Axis < 3)
inline constexpr bool locations_match_transverse_to_axis =
    (Axis == 0 || SourceLocation::x == TargetLocation::x) &&
    (Axis == 1 || SourceLocation::y == TargetLocation::y) &&
    (Axis == 2 || SourceLocation::z == TargetLocation::z);

template <class Component, class Accumulator>
concept FloatingDifferentialComponent =
    std::floating_point<Accumulator> &&
    requires(const Component component,
             const typename Component::index_policy::coordinate_type i,
             const typename Component::index_policy::coordinate_type j,
             const typename Component::index_policy::coordinate_type k) {
      typename Component::element_type;
      typename Component::value_type;
      typename Component::space_type;
      typename Component::location_type;
      typename Component::index_policy;
      requires std::floating_point<typename Component::value_type>;
      requires std::same_as<
          std::remove_const_t<typename Component::element_type>,
          typename Component::value_type>;
      requires CartesianLocation<typename Component::location_type>;
      requires CartesianIndexPolicy<typename Component::index_policy>;
      {
        component(i, j, k)
      } -> std::same_as<typename Component::element_type &>;
      { static_cast<Accumulator>(component(i, j, k)) };
    };

template <class Component, class Location, class Space, class IndexPolicy,
          class Accumulator>
concept LocatedDifferentialComponent =
    FloatingDifferentialComponent<Component, Accumulator> &&
    std::same_as<typename Component::location_type, Location> &&
    std::same_as<typename Component::space_type, Space> &&
    std::same_as<typename Component::index_policy, IndexPolicy>;

// Host targets additionally expose their interior extents for traversal.
template <class Component, class Location, class Space, class IndexPolicy,
          class Accumulator>
concept MutableDifferentialComponent =
    LocatedDifferentialComponent<Component, Location, Space, IndexPolicy,
                                 Accumulator> &&
    (!std::is_const_v<typename Component::element_type>) &&
    requires(const Component component,
             const typename IndexPolicy::coordinate_type i,
             const typename IndexPolicy::coordinate_type j,
             const typename IndexPolicy::coordinate_type k,
             const Accumulator value) {
      {
        component(i, j, k) = static_cast<typename Component::value_type>(value)
      } -> std::same_as<typename Component::element_type &>;
    } &&
    (!std::same_as<Space, HostSpace> || requires(const Component component) {
      {
        component.extents()
      } -> std::same_as<CartesianExtent3<typename IndexPolicy::count_type>>;
    });

template <std::size_t Axis, std::floating_point Accumulator, class Component>
  requires(Axis < 3) && FloatingDifferentialComponent<Component, Accumulator>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
shifted_staggered_value(
    const Component component,
    const typename Component::index_policy::coordinate_type i,
    const typename Component::index_policy::coordinate_type j,
    const typename Component::index_policy::coordinate_type k,
    const typename Component::index_policy::coordinate_type shift) noexcept {
  if constexpr (Axis == 0) {
    return static_cast<Accumulator>(component(i + shift, j, k));
  } else if constexpr (Axis == 1) {
    return static_cast<Accumulator>(component(i, j + shift, k));
  } else {
    return static_cast<Accumulator>(component(i, j, k + shift));
  }
}

} // namespace detail

// Compatible-grid derivative at one target sample. Source and target differ
// on Axis and match on both transverse axes. A node source uses the forward
// pair [i,i+1]; a cell source uses the backward pair [i-1,i]. Hard
// preconditions: inverse_spacing stays finite and positive as Accumulator,
// both source samples are addressable and coordinate shifts are
// representable.
template <std::floating_point Accumulator, std::size_t Axis,
          detail::CartesianLocation TargetLocation, class Component,
          std::floating_point InverseSpacing>
  requires(Axis < 3) &&
          detail::FloatingDifferentialComponent<Component, Accumulator> &&
          (Component::location_type::template is_node<Axis> !=
           TargetLocation::template is_node<Axis>) &&
          detail::locations_match_transverse_to_axis<
              Axis, typename Component::location_type, TargetLocation>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
staggered_axis_difference(
    const Component component,
    const typename Component::index_policy::coordinate_type i,
    const typename Component::index_policy::coordinate_type j,
    const typename Component::index_policy::coordinate_type k,
    const InverseSpacing inverse_spacing) noexcept {
  using Coordinate = typename Component::index_policy::coordinate_type;
  const Accumulator inverse = static_cast<Accumulator>(inverse_spacing);
  if constexpr (Component::location_type::template is_node<Axis>) {
    return (detail::shifted_staggered_value<Axis, Accumulator>(
                component, i, j, k, Coordinate{1}) -
            detail::shifted_staggered_value<Axis, Accumulator>(
                component, i, j, k, Coordinate{0})) *
           inverse;
  } else {
    return (detail::shifted_staggered_value<Axis, Accumulator>(
                component, i, j, k, Coordinate{0}) -
            detail::shifted_staggered_value<Axis, Accumulator>(
                component, i, j, k, Coordinate{-1})) *
           inverse;
  }
}

} // namespace lpfx

#endif
