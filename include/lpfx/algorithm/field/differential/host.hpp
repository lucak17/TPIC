#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_HOST_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_HOST_HPP

#include <concepts>
#include <cstddef>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/memory/space.hpp>
#include <lpfx/algorithm/field/differential/operators.hpp>

namespace lpfx {

namespace detail {

template <class Op, std::floating_point Accumulator, std::size_t Index,
          class XComponent, class YComponent, class ZComponent,
          class TargetComponent, std::floating_point SpacingScalar>
LPFX_ALWAYS_INLINE constexpr void run_serial_stencil_target(
    const XComponent source_x, const YComponent source_y,
    const ZComponent source_z, const TargetComponent target,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  using Coordinate = typename XComponent::index_policy::coordinate_type;
  using Target = typename TargetComponent::value_type;
  const auto extents = target.extents();
  for (auto k = decltype(extents.z()){}; k != extents.z(); ++k) {
    const Coordinate coordinate_k = static_cast<Coordinate>(k);
    for (auto j = decltype(extents.y()){}; j != extents.y(); ++j) {
      const Coordinate coordinate_j = static_cast<Coordinate>(j);
      LPFX_INDEPENDENT_LOOP
      for (Coordinate coordinate_i = 0;
           coordinate_i != static_cast<Coordinate>(extents.x());
           ++coordinate_i) {
        target(coordinate_i, coordinate_j, coordinate_k) =
            static_cast<Target>(Op::template apply<Accumulator, Index>(
                source_x, source_y, source_z, coordinate_i, coordinate_j,
                coordinate_k, spacing));
      }
    }
  }
}

// Traverses each target interior in target order.
template <class Op, std::floating_point Accumulator, class XComponent,
          class YComponent, class ZComponent, class... Targets,
          std::floating_point SpacingScalar>
LPFX_ALWAYS_INLINE constexpr void run_serial_stencil(
    const StencilComponents<XComponent, YComponent, ZComponent> sources,
    const StencilComponents<Targets...> targets,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  meta::for_each_index<Op::target_count>([&](auto index) {
    run_serial_stencil_target<Op, Accumulator, decltype(index)::value>(
        sources.x, sources.y, sources.z,
        stencil_component<decltype(index)::value>(targets), spacing);
  });
}

} // namespace detail

// Stateless serial traversal of every target interior. Hard preconditions:
// all layouts describe the location samples of one cell geometry, sources
// do not overlap targets, target interiors are pairwise disjoint, spacing
// was prepared and every source sample is addressable. Target halos are
// left unchanged.

template <std::floating_point Accumulator, class EdgeXComponent,
          class EdgeYComponent, class EdgeZComponent, class FaceXComponent,
          class FaceYComponent, class FaceZComponent,
          std::floating_point SpacingScalar>
  requires detail::StencilOperands<detail::CurlEdgeToFace, Accumulator,
                                   HostSpace, EdgeXComponent, EdgeYComponent,
                                   EdgeZComponent, FaceXComponent,
                                   FaceYComponent, FaceZComponent>
LPFX_ALWAYS_INLINE constexpr void run_serial_curl_edge_to_face(
    const EdgeXComponent edge_x, const EdgeYComponent edge_y,
    const EdgeZComponent edge_z, const FaceXComponent face_x,
    const FaceYComponent face_y, const FaceZComponent face_z,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  detail::run_serial_stencil<detail::CurlEdgeToFace, Accumulator>(
      detail::make_stencil_components(edge_x, edge_y, edge_z),
      detail::make_stencil_components(face_x, face_y, face_z), spacing);
}

template <std::floating_point Accumulator, class FaceXComponent,
          class FaceYComponent, class FaceZComponent, class EdgeXComponent,
          class EdgeYComponent, class EdgeZComponent,
          std::floating_point SpacingScalar>
  requires detail::StencilOperands<detail::CurlFaceToEdge, Accumulator,
                                   HostSpace, FaceXComponent, FaceYComponent,
                                   FaceZComponent, EdgeXComponent,
                                   EdgeYComponent, EdgeZComponent>
LPFX_ALWAYS_INLINE constexpr void run_serial_curl_face_to_edge(
    const FaceXComponent face_x, const FaceYComponent face_y,
    const FaceZComponent face_z, const EdgeXComponent edge_x,
    const EdgeYComponent edge_y, const EdgeZComponent edge_z,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  detail::run_serial_stencil<detail::CurlFaceToEdge, Accumulator>(
      detail::make_stencil_components(face_x, face_y, face_z),
      detail::make_stencil_components(edge_x, edge_y, edge_z), spacing);
}

template <std::floating_point Accumulator, class EdgeXComponent,
          class EdgeYComponent, class EdgeZComponent, class NodeComponent,
          std::floating_point SpacingScalar>
  requires detail::StencilOperands<detail::DivergenceEdgeToNode, Accumulator,
                                   HostSpace, EdgeXComponent, EdgeYComponent,
                                   EdgeZComponent, NodeComponent>
LPFX_ALWAYS_INLINE constexpr void run_serial_divergence_edge_to_node(
    const EdgeXComponent edge_x, const EdgeYComponent edge_y,
    const EdgeZComponent edge_z, const NodeComponent node,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  detail::run_serial_stencil<detail::DivergenceEdgeToNode, Accumulator>(
      detail::make_stencil_components(edge_x, edge_y, edge_z),
      detail::make_stencil_components(node), spacing);
}

} // namespace lpfx

#endif
