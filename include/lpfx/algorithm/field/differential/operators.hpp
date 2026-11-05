#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_OPERATORS_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_OPERATORS_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/algorithm/field/differential/spacing.hpp>
#include <lpfx/algorithm/field/differential/staggered_difference.hpp>

namespace lpfx {

namespace detail {

template <CartesianLocation X, CartesianLocation Y, CartesianLocation Z>
struct VectorLocation3 final {
  using x = X;
  using y = Y;
  using z = Z;

  template <std::size_t Axis>
  using axis =
      std::conditional_t<Axis == 0, X, std::conditional_t<Axis == 1, Y, Z>>;
};

using EdgeLocations3 = VectorLocation3<EdgeX, EdgeY, EdgeZ>;
using FaceLocations3 = VectorLocation3<FaceX, FaceY, FaceZ>;

// Stencil operators: a source vector location triple, target_count target
// locations and one point formula per target. Every location is a
// compile-time fact, so each formula keeps only its loads and arithmetic.

// x = d_y(z) - d_z(y), y = d_z(x) - d_x(z), z = d_x(y) - d_y(x).
template <class Source, class Target> struct CompatibleCurl final {
  using source_locations = Source;
  static constexpr std::size_t target_count = 3;
  template <std::size_t Index>
  using target_location = typename Target::template axis<Index>;

  template <std::floating_point Accumulator, std::size_t Index, class XComponent,
            class YComponent, class ZComponent,
            std::floating_point SpacingScalar>
  [[nodiscard]] LPFX_HOST_DEVICE static LPFX_ALWAYS_INLINE constexpr Accumulator
  apply(const XComponent source_x, const YComponent source_y,
        const ZComponent source_z,
        const typename XComponent::index_policy::coordinate_type i,
        const typename XComponent::index_policy::coordinate_type j,
        const typename XComponent::index_policy::coordinate_type k,
        const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
    using Location = target_location<Index>;
    if constexpr (Index == 0) {
      return staggered_axis_difference<Accumulator, 1, Location>(
                 source_z, i, j, k, spacing.inverse_spacing_y()) -
             staggered_axis_difference<Accumulator, 2, Location>(
                 source_y, i, j, k, spacing.inverse_spacing_z());
    } else if constexpr (Index == 1) {
      return staggered_axis_difference<Accumulator, 2, Location>(
                 source_x, i, j, k, spacing.inverse_spacing_z()) -
             staggered_axis_difference<Accumulator, 0, Location>(
                 source_z, i, j, k, spacing.inverse_spacing_x());
    } else {
      return staggered_axis_difference<Accumulator, 0, Location>(
                 source_y, i, j, k, spacing.inverse_spacing_x()) -
             staggered_axis_difference<Accumulator, 1, Location>(
                 source_x, i, j, k, spacing.inverse_spacing_y());
    }
  }
};

// d_x(x) + d_y(y) + d_z(z).
template <class Source, CartesianLocation Target>
struct CompatibleDivergence final {
  using source_locations = Source;
  static constexpr std::size_t target_count = 1;
  template <std::size_t> using target_location = Target;

  template <std::floating_point Accumulator, std::size_t Index, class XComponent,
            class YComponent, class ZComponent,
            std::floating_point SpacingScalar>
  [[nodiscard]] LPFX_HOST_DEVICE static LPFX_ALWAYS_INLINE constexpr Accumulator
  apply(const XComponent source_x, const YComponent source_y,
        const ZComponent source_z,
        const typename XComponent::index_policy::coordinate_type i,
        const typename XComponent::index_policy::coordinate_type j,
        const typename XComponent::index_policy::coordinate_type k,
        const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
    return staggered_axis_difference<Accumulator, 0, Target>(
               source_x, i, j, k, spacing.inverse_spacing_x()) +
           staggered_axis_difference<Accumulator, 1, Target>(
               source_y, i, j, k, spacing.inverse_spacing_y()) +
           staggered_axis_difference<Accumulator, 2, Target>(
               source_z, i, j, k, spacing.inverse_spacing_z());
  }
};

using CurlEdgeToFace = CompatibleCurl<EdgeLocations3, FaceLocations3>;
using CurlFaceToEdge = CompatibleCurl<FaceLocations3, EdgeLocations3>;
using DivergenceEdgeToNode = CompatibleDivergence<EdgeLocations3, Node>;

// Stencil operands by value, one member per component.
template <class... Components> struct StencilComponents;

template <class X> struct StencilComponents<X> final {
  X x{};
};

template <class X, class Y, class Z> struct StencilComponents<X, Y, Z> final {
  X x{};
  Y y{};
  Z z{};
};

template <class... Components>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr StencilComponents<
    Components...>
make_stencil_components(const Components... components) noexcept {
  return {components...};
}

template <std::size_t Index, class Components>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr const auto &
stencil_component(const Components &components) noexcept {
  if constexpr (Index == 0) {
    return components.x;
  } else if constexpr (Index == 1) {
    return components.y;
  } else {
    return components.z;
  }
}

template <class Locations, class XComponent, class YComponent,
          class ZComponent, class Accumulator>
concept DifferentialSources =
    FloatingDifferentialComponent<XComponent, Accumulator> &&
    std::same_as<typename XComponent::location_type,
                 typename Locations::x> &&
    LocatedDifferentialComponent<YComponent, typename Locations::y,
                                 typename XComponent::space_type,
                                 typename XComponent::index_policy,
                                 Accumulator> &&
    LocatedDifferentialComponent<ZComponent, typename Locations::z,
                                 typename XComponent::space_type,
                                 typename XComponent::index_policy,
                                 Accumulator>;

template <class Op, class Accumulator, class Source, class... Targets,
          std::size_t... Index>
[[nodiscard]] consteval bool
stencil_targets_writable(std::index_sequence<Index...>) noexcept {
  return (MutableDifferentialComponent<
              Targets, typename Op::template target_location<Index>,
              typename Source::space_type, typename Source::index_policy,
              Accumulator> &&
          ...);
}

// Sources match Op's locations in Space, and each target is a writable
// component at its target location with the sources' index policy.
template <class Op, class Accumulator, class Space, class XComponent,
          class YComponent, class ZComponent, class... Targets>
concept StencilOperands =
    sizeof...(Targets) == Op::target_count &&
    DifferentialSources<typename Op::source_locations, XComponent, YComponent,
                        ZComponent, Accumulator> &&
    std::same_as<typename XComponent::space_type, Space> &&
    stencil_targets_writable<Op, Accumulator, XComponent, Targets...>(
        std::index_sequence_for<Targets...>{});

} // namespace detail

// Point formulas. Each loads only its source samples and performs no bounds
// check. Hard preconditions: spacing was prepared, each inverse stays finite
// and positive as Accumulator, and every source sample is addressable.

template <std::floating_point Accumulator, std::size_t Component,
          class EdgeXComponent, class EdgeYComponent, class EdgeZComponent,
          std::floating_point SpacingScalar>
  requires(Component < 3) &&
          detail::DifferentialSources<detail::EdgeLocations3, EdgeXComponent,
                                      EdgeYComponent, EdgeZComponent,
                                      Accumulator>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
curl_edge_to_face_component(
    const EdgeXComponent edge_x, const EdgeYComponent edge_y,
    const EdgeZComponent edge_z,
    const typename EdgeXComponent::index_policy::coordinate_type i,
    const typename EdgeXComponent::index_policy::coordinate_type j,
    const typename EdgeXComponent::index_policy::coordinate_type k,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  return detail::CurlEdgeToFace::apply<Accumulator, Component>(
      edge_x, edge_y, edge_z, i, j, k, spacing);
}

template <std::floating_point Accumulator, std::size_t Component,
          class FaceXComponent, class FaceYComponent, class FaceZComponent,
          std::floating_point SpacingScalar>
  requires(Component < 3) &&
          detail::DifferentialSources<detail::FaceLocations3, FaceXComponent,
                                      FaceYComponent, FaceZComponent,
                                      Accumulator>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
curl_face_to_edge_component(
    const FaceXComponent face_x, const FaceYComponent face_y,
    const FaceZComponent face_z,
    const typename FaceXComponent::index_policy::coordinate_type i,
    const typename FaceXComponent::index_policy::coordinate_type j,
    const typename FaceXComponent::index_policy::coordinate_type k,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  return detail::CurlFaceToEdge::apply<Accumulator, Component>(
      face_x, face_y, face_z, i, j, k, spacing);
}

template <std::floating_point Accumulator, class EdgeXComponent,
          class EdgeYComponent, class EdgeZComponent,
          std::floating_point SpacingScalar>
  requires detail::DifferentialSources<detail::EdgeLocations3, EdgeXComponent,
                                       EdgeYComponent, EdgeZComponent,
                                       Accumulator>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
divergence_edge_to_node(
    const EdgeXComponent edge_x, const EdgeYComponent edge_y,
    const EdgeZComponent edge_z,
    const typename EdgeXComponent::index_policy::coordinate_type i,
    const typename EdgeXComponent::index_policy::coordinate_type j,
    const typename EdgeXComponent::index_policy::coordinate_type k,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  return detail::DivergenceEdgeToNode::apply<Accumulator, 0>(
      edge_x, edge_y, edge_z, i, j, k, spacing);
}

} // namespace lpfx

#endif
