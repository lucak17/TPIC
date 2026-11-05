#ifndef TPIC_ECSIM_FIELDS_CELL_TO_NODE_DEVICE_HPP
#define TPIC_ECSIM_FIELDS_CELL_TO_NODE_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/ecsim/fields/cell_to_node.hpp>

#include <cmath>
#include <type_traits>

namespace tpic::ecsim {
namespace kernels {

// Grid-stride over the interior nodes with KernelLaunch<> blocks; *invalid,
// when given, flags a non-finite node value.
template <class Precision, class CellView, class NodeView>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void magnetic_cell_to_node_kernel(
    CellView cell, NodeView node, data::GridShape shape, unsigned* invalid) {
  using Fields = data::FieldSchema<Precision>;
  using Compute = typename Precision::Compute::Magnetic;
  // Unsigned loop arithmetic avoids overflow at the last signed-index block.
  lpfx::device::for_each_grid_index(static_cast<unsigned>(shape.count), [&](const unsigned index) {
    const auto [i, j, k] = device::decode_x_fastest(index, shape.nx, shape.ny);
    const auto bx = static_cast<typename Precision::Storage::Magnetic>(cell_to_node_average<Compute>(
        cell.template get<typename Fields::Bx>(), i, j, k));
    const auto by = static_cast<typename Precision::Storage::Magnetic>(cell_to_node_average<Compute>(
        cell.template get<typename Fields::By>(), i, j, k));
    const auto bz = static_cast<typename Precision::Storage::Magnetic>(cell_to_node_average<Compute>(
        cell.template get<typename Fields::Bz>(), i, j, k));
    node.template get<typename Fields::Bnx>()(i, j, k) = bx;
    node.template get<typename Fields::Bny>()(i, j, k) = by;
    node.template get<typename Fields::Bnz>()(i, j, k) = bz;
    // A finite cell update can still overflow the ordered eight-cell sum.
    // Certify the node values before early deposition reads the next B epoch.
    if (invalid && (!::isfinite(bx) || !::isfinite(by) || !::isfinite(bz)))
      atomicExch(invalid, 1U);
  });
}

} // namespace kernels

// Stream: executor, after the cell ghosts are complete on it. Writes the
// interior nodes and sets *invalid, when given, for a non-finite value. Both
// owners outlive the stream's use.
template <class Precision, class CellOwner, class NodeOwner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_magnetic_cell_to_node(
    const CellOwner& cell, NodeOwner& node, unsigned* invalid,
    lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename CellOwner::space_type::backend_type> &&
                std::is_same_v<Backend, typename NodeOwner::space_type::backend_type>);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  data::GridShape shape;
  const auto checked = try_prepare_cell_to_node(cell, node, shape);
  if (checked != lpfx::ErrorCode::success) return checked;
  return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernels::magnetic_cell_to_node_kernel<
      Precision, decltype(cell.view()), decltype(node.view())>>(
      executor, shape.count, cell.view(), node.view(), shape, invalid);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_CELL_TO_NODE_DEVICE_HPP
