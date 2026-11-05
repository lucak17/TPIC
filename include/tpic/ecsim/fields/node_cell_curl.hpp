#ifndef TPIC_ECSIM_FIELDS_NODE_CELL_CURL_HPP
#define TPIC_ECSIM_FIELDS_NODE_CELL_CURL_HPP

#include <tpic/config/precision.hpp>
#include <tpic/ecsim/fields/cell_to_node.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/fields/stencil_region.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace tpic::ecsim {

template <class Compute> struct NodeCellSpacing {
  Compute inverse_x{}, inverse_y{}, inverse_z{};
};

template <class Compute> struct StencilVector {
  Compute x{}, y{}, z{};
};

// Match the solver's reciprocal in its arithmetic precision, independently of
// the source and destination storage types. Prepare once per geometry.
template <class Compute>
[[nodiscard]] lpfx::ErrorCode try_make_node_cell_spacing(
    const std::array<double, 3>& spacing, NodeCellSpacing<Compute>& result) noexcept {
  static_assert(std::is_floating_point_v<Compute>);
  result = {};
  Compute inverse[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(spacing[axis]) || spacing[axis] <= 0.0)
      return lpfx::ErrorCode::invalid_argument;
    if (spacing[axis] > config::convert<double>(std::numeric_limits<Compute>::max()))
      return lpfx::ErrorCode::overflow;
    const Compute delta = config::convert<Compute>(spacing[axis]);
    if (delta == Compute(0)) return lpfx::ErrorCode::overflow;
    inverse[axis] = Compute(1) / delta;
    if (!std::isfinite(inverse[axis]) || inverse[axis] <= Compute(0))
      return lpfx::ErrorCode::overflow;
  }
  result = {inverse[0], inverse[1], inverse[2]};
  return lpfx::ErrorCode::success;
}

namespace detail {

// Eight loads per component, reused across its derivatives. The anchor is the
// target's logical coordinate: node sources extend forwards, cells backwards.
template <class Compute, class Location> struct NodeCellSamples {
  static_assert(std::is_floating_point_v<Compute>);
  static_assert(std::is_same_v<Location, lpfx::Node> ||
                std::is_same_v<Location, lpfx::Cell>);
  Compute base, x, y, z, xy, xz, yz, xyz;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static Compute difference(
      Compute anchor, Compute neighbor) noexcept {
    if constexpr (std::is_same_v<Location, lpfx::Node>) return neighbor - anchor;
    else return anchor - neighbor;
  }

  // The four edge differences are summed in the order written; rearranging
  // them into a difference of face sums changes FP rounding.
  template <int Axis>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Compute derivative(Compute inverse) const noexcept {
    static_assert(Axis >= 0 && Axis < 3);
    if constexpr (Axis == 0)
      return Compute(0.25) * inverse *
          (difference(base, x) + difference(z, xz) +
           difference(y, xy) + difference(yz, xyz));
    else if constexpr (Axis == 1)
      return Compute(0.25) * inverse *
          (difference(base, y) + difference(z, yz) +
           difference(x, xy) + difference(xz, xyz));
    else
      return Compute(0.25) * inverse *
          (difference(base, z) + difference(x, xz) +
           difference(y, yz) + difference(xy, xyz));
  }
};

template <class Compute, class View>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE auto load_node_cell_samples(
    View view, int i, int j, int k) noexcept {
  using Location = typename View::location_type;
  constexpr int step = std::is_same_v<Location, lpfx::Node> ? 1 : -1;
  return NodeCellSamples<Compute, Location>{
      config::convert<Compute>(view(i, j, k)),
      config::convert<Compute>(view(i + step, j, k)),
      config::convert<Compute>(view(i, j + step, k)),
      config::convert<Compute>(view(i, j, k + step)),
      config::convert<Compute>(view(i + step, j + step, k)),
      config::convert<Compute>(view(i + step, j, k + step)),
      config::convert<Compute>(view(i, j + step, k + step)),
      config::convert<Compute>(view(i + step, j + step, k + step))};
}

template <class VectorView>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr void require_collocated_vector() noexcept {
  static_assert(VectorView::component_count == 3);
  using Location = typename decltype(std::declval<VectorView>().template component<0>())::location_type;
  static_assert(std::is_same_v<Location,
      typename decltype(std::declval<VectorView>().template component<1>())::location_type> &&
      std::is_same_v<Location,
      typename decltype(std::declval<VectorView>().template component<2>())::location_type>);
}

} // namespace detail

// Pointwise ECSIM curl, not an edge/face Yee operator. Coordinates address an
// interior target sample at the opposite (node/cell) location. Node sources
// need coherent shared interior nodes; cell sources need completed halos and
// physical closure. This primitive performs no communication or boundary work.
template <class Compute, class VectorView>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE StencilVector<Compute> node_cell_curl(
    VectorView source, int i, int j, int k, NodeCellSpacing<Compute> spacing) noexcept {
  detail::require_collocated_vector<VectorView>();
  const auto x = detail::load_node_cell_samples<Compute>(source.template component<0>(), i, j, k);
  const auto y = detail::load_node_cell_samples<Compute>(source.template component<1>(), i, j, k);
  const auto z = detail::load_node_cell_samples<Compute>(source.template component<2>(), i, j, k);
  return {z.template derivative<1>(spacing.inverse_y) - y.template derivative<2>(spacing.inverse_z),
          x.template derivative<2>(spacing.inverse_z) - z.template derivative<0>(spacing.inverse_x),
          y.template derivative<0>(spacing.inverse_x) - x.template derivative<1>(spacing.inverse_y)};
}

// The divergence of the same stencil. It annihilates the curl of the opposite
// direction exactly: N2C(divergence) of a C2N curl and C2N(divergence) of an
// N2C curl are zero in exact arithmetic.
template <class Compute, class VectorView>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Compute node_cell_divergence(
    VectorView source, int i, int j, int k, NodeCellSpacing<Compute> spacing) noexcept {
  detail::require_collocated_vector<VectorView>();
  return detail::load_node_cell_samples<Compute>(source.template component<0>(), i, j, k)
             .template derivative<0>(spacing.inverse_x) +
         detail::load_node_cell_samples<Compute>(source.template component<1>(), i, j, k)
             .template derivative<1>(spacing.inverse_y) +
         detail::load_node_cell_samples<Compute>(source.template component<2>(), i, j, k)
             .template derivative<2>(spacing.inverse_z);
}

// Check all components and the actual owners before creating public views,
// and that the prepared region geometry describes the target's samples.
// Halo freshness and source/destination non-aliasing remain caller contracts.
template <class SourceOwner, class TargetOwner>
[[nodiscard]] lpfx::ErrorCode try_prepare_node_cell_curl(
    const SourceOwner& source, const TargetOwner& target,
    const StencilRegionGeometry& geometry) noexcept {
  using SourceLocation = field_location_t<SourceOwner>;
  using TargetLocation = field_location_t<TargetOwner>;
  static_assert((std::is_same_v<SourceLocation, lpfx::Node> && std::is_same_v<TargetLocation, lpfx::Cell>) ||
                (std::is_same_v<SourceLocation, lpfx::Cell> && std::is_same_v<TargetLocation, lpfx::Node>));
  static_assert(VectorFieldAt<SourceOwner, SourceLocation> &&
                VectorFieldAt<TargetOwner, TargetLocation>);
  if (!source.initialized() || !target.initialized()) return lpfx::ErrorCode::invalid_state;
  if (source.cells() != target.cells()) return lpfx::ErrorCode::invalid_argument;
  if constexpr (std::is_same_v<SourceLocation, lpfx::Cell>)
    if (!has_halo(source, 1)) return lpfx::ErrorCode::invalid_argument;
  // The node stencil addresses n on the upper cell: node targets have cells + 1 samples.
  constexpr int node = std::is_same_v<TargetLocation, lpfx::Node> ? 1 : 0;
  if (!field_cells_equal(target, geometry.nx - node, geometry.ny - node, geometry.nz - node))
    return lpfx::ErrorCode::invalid_argument;
  return lpfx::ErrorCode::success;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_NODE_CELL_CURL_HPP
