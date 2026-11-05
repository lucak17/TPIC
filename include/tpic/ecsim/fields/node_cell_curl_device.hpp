#ifndef TPIC_ECSIM_FIELDS_NODE_CELL_CURL_DEVICE_HPP
#define TPIC_ECSIM_FIELDS_NODE_CELL_CURL_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/ecsim/fields/node_cell_curl.hpp>
#include <tpic/ecsim/fields/stencil_region.hpp>

#include <cmath>
#include <type_traits>

namespace tpic::ecsim {

namespace kernels {

// Grid-stride over the samples of Region with KernelLaunch<> blocks. No
// validation flag.
template <StencilRegion Region, class Compute, class SourceView, class TargetView>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void node_cell_curl_kernel(
    SourceView source, TargetView target, NodeCellSpacing<Compute> spacing,
    StencilRegionGeometry geometry) {
  lpfx::device::for_each_grid_index(stencil_region_count<Region>(geometry), [&](const unsigned index) {
    int i, j, k;
    stencil_region_coordinates<Region>(index, geometry, i, j, k);
    const auto value = node_cell_curl(source, i, j, k, spacing);
    const auto x = target.template component<0>();
    const auto y = target.template component<1>();
    const auto z = target.template component<2>();
    x(i, j, k) = static_cast<typename decltype(x)::value_type>(value.x);
    y(i, j, k) = static_cast<typename decltype(y)::value_type>(value.y);
    z(i, j, k) = static_cast<typename decltype(z)::value_type>(value.z);
  });
}

} // namespace kernels

// Stream: executor. Writes target interiors in Compute arithmetic whatever the
// storage (curl scratch uses SolverStorage); locations select N2C or C2N. N2C
// reads interior nodes only (boundary projection done, no ghosts) and accepts
// only full; an interior projection after it waits for it. C2N regions as in
// stencil_region.hpp. The caller refreshes target ghosts before the next C2N, so
// curl-curl needs a cell exchange between. geometry from prepare_field_operator_plans.
// Compute is the arithmetic type; the solver passes its reduction type for the residual apply.
template <class Precision, StencilRegion Region = StencilRegion::full,
          class Compute = typename Precision::Compute::Electric,
          class SourceOwner, class TargetOwner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_node_cell_curl(
    const SourceOwner& source, TargetOwner& target, const StencilRegionGeometry& geometry,
    NodeCellSpacing<Compute> spacing,
    lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename SourceOwner::space_type::backend_type> &&
                std::is_same_v<Backend, typename TargetOwner::space_type::backend_type>);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  if (!std::isfinite(spacing.inverse_x) || spacing.inverse_x <= 0 ||
      !std::isfinite(spacing.inverse_y) || spacing.inverse_y <= 0 ||
      !std::isfinite(spacing.inverse_z) || spacing.inverse_z <= 0)
    return lpfx::ErrorCode::invalid_argument;
  const auto checked = try_prepare_node_cell_curl(source, target, geometry);
  if (checked != lpfx::ErrorCode::success) return checked;
  if constexpr (std::is_same_v<field_location_t<SourceOwner>, lpfx::Node>)
    if constexpr (Region != StencilRegion::full) return lpfx::ErrorCode::invalid_argument;
  return lpfx::device::launch<lpfx::device::KernelLaunch<>,
      kernels::node_cell_curl_kernel<Region, Compute,
          decltype(source.view()), decltype(target.view())>,
      lpfx::device::EmptyWork::skip>(
      executor, stencil_region_count<Region>(geometry), source.view(), target.view(), spacing,
      geometry);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_NODE_CELL_CURL_DEVICE_HPP
