#ifndef TPIC_ECSIM_FIELDS_CELL_TO_NODE_HPP
#define TPIC_ECSIM_FIELDS_CELL_TO_NODE_HPP

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/grid_shape.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>

#include <type_traits>

namespace tpic::ecsim {

// ECSIM's collocated vector stencil, not an edge/face interpolation. The
// caller supplies a halo-consistent cell view and an interior node coordinate.
// Promote each load before addition and sum the eight cells x-fastest, left to
// right as written; this order is part of the cell-to-node contract.
template <class Compute = double, class CellView>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Compute cell_to_node_average(
    CellView cell, int i, int j, int k) noexcept {
  static_assert(std::is_floating_point_v<Compute>);
  static_assert(std::is_same_v<typename CellView::location_type, lpfx::Cell>);
  return (config::convert<Compute>(cell(i - 1, j - 1, k - 1)) +
          config::convert<Compute>(cell(i,     j - 1, k - 1)) +
          config::convert<Compute>(cell(i - 1, j,     k - 1)) +
          config::convert<Compute>(cell(i,     j,     k - 1)) +
          config::convert<Compute>(cell(i - 1, j - 1, k)) +
          config::convert<Compute>(cell(i,     j - 1, k)) +
          config::convert<Compute>(cell(i - 1, j,     k)) +
          config::convert<Compute>(cell(i,     j,     k))) * Compute(0.125);
}

// Validate owners before taking a public view or launching signed-index work.
// This checks storage, not halo freshness; communication supplies that epoch.
template <class CellOwner, class NodeOwner>
[[nodiscard]] lpfx::ErrorCode try_prepare_cell_to_node(
    const CellOwner& cell, const NodeOwner& node, data::GridShape& shape) noexcept {
  static_assert(VectorFieldAt<CellOwner, lpfx::Cell> && VectorFieldAt<NodeOwner, lpfx::Node>);
  shape = {};
  if (!cell.initialized() || !node.initialized())
    return lpfx::ErrorCode::invalid_state;
  if (cell.cells() != node.cells()) return lpfx::ErrorCode::invalid_argument;
  if (!has_halo(cell, 1)) return lpfx::ErrorCode::invalid_argument;
  return data::try_make_grid_shape(cell.cells(), 1, {}, shape);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_CELL_TO_NODE_HPP
