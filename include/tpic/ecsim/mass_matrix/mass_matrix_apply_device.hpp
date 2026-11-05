#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_APPLY_DEVICE_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_APPLY_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/block_threads.hpp>
#include <tpic/data/mass_matrix/compact_mass_matrix_device.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/fields/wall_faces.hpp>
#include <tpic/ecsim/fields/stencil_region.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_stencil.hpp>

#include <cmath>
#include <type_traits>

namespace tpic::ecsim {

namespace kernels {

// Grid-stride over the samples of Region with KernelLaunch<mass_matrix_block_threads>
// blocks. A non-finite stored output sets *invalid and is not written.
template <class Precision, class Compute, StencilRegion Region, class Input, class Output>
__global__ __launch_bounds__(device::mass_matrix_block_threads) void mass_matrix_apply_kernel(
    const typename Precision::Storage::Mass* mass, Input input, Output output,
    data::MassMatrixLayout layout, WallFaces dual_volume, Compute inverse_volume,
    StencilRegionGeometry geometry, unsigned* invalid) {
  using Ox = typename decltype(output.template component<0>())::value_type;
  using Oy = typename decltype(output.template component<1>())::value_type;
  using Oz = typename decltype(output.template component<2>())::value_type;
  lpfx::device::for_each_grid_index(stencil_region_count<Region>(geometry), [&](const unsigned index) {
    int i, j, k;
    stencil_region_coordinates<Region>(index, geometry, i, j, k);
    const Compute row_inverse_volume =
        inverse_volume * config::convert<Compute>(dual_volume.dual_volume_factor(i, j, k));
    Compute x, y, z;
    apply_mass_matrix_node(mass, layout, input.template component<0>(),
        input.template component<1>(), input.template component<2>(),
        i, j, k, row_inverse_volume, x, y, z);
    const Ox ox = config::convert<Ox>(x);
    const Oy oy = config::convert<Oy>(y);
    const Oz oz = config::convert<Oz>(z);
    // Narrowing keeps Inf and NaN, so the stored values certify x, y and z.
    if (!::isfinite(ox) || !::isfinite(oy) || !::isfinite(oz)) {
      atomicExch(invalid, 1U);
      return;
    }
    output.template component<0>()(i,j,k) = ox;
    output.template component<1>()(i,j,k) = oy;
    output.template component<2>()(i,j,k) = oz;
  });
}

} // namespace kernels

// output = M * input over the interior nodes of Region, in Solver arithmetic,
// with the row-volume scaling applied here once; components are x/y/z whatever
// their tags and storage. Stream: executor, with the mass halo complete (its
// addition only for strict_interior, which reads no ghost record), the
// input coherent as Region requires (stencil_region.hpp) and *invalid cleared;
// input and output never alias. The caller refreshes output ghosts and applies
// the solver closures next. nodes and dual_volume come from
// prepare_field_operator_plans; inverse_volume is the inverse cell volume.
// Compute is the arithmetic type; the solver passes its reduction type for the residual apply.
template <class Precision, StencilRegion Region = StencilRegion::full,
          class Compute = typename Precision::Compute::Electric,
          class InputOwner, class OutputOwner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_mass_matrix_apply(
    const data::CompactMassMatrix<Precision, Backend>& mass,
    const InputOwner& input, OutputOwner& output, const StencilRegionGeometry& nodes,
    const WallFaces& dual_volume, Compute inverse_volume, unsigned* invalid,
    lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename InputOwner::space_type::backend_type> &&
                std::is_same_v<Backend, typename OutputOwner::space_type::backend_type>);
  static_assert(VectorFieldAt<InputOwner, lpfx::Node> && VectorFieldAt<OutputOwner, lpfx::Node>);
  if (!executor.bound() || invalid == nullptr) return lpfx::ErrorCode::invalid_argument;
  if (!mass.initialized() || !input.initialized() || !output.initialized())
    return lpfx::ErrorCode::invalid_state;
  const auto layout = mass.layout();
  const auto& cells = dual_volume.cells;
  // Subtracting from checked nodal extents avoids overflowing untrusted cells.
  if (layout.nx - 1 != cells[0] || layout.ny - 1 != cells[1] || layout.nz - 1 != cells[2] ||
      nodes.nx != layout.nx || nodes.ny != layout.ny || nodes.nz != layout.nz ||
      !field_cells_equal(input, cells[0], cells[1], cells[2]) || !has_halo(input, 1) ||
      !field_cells_equal(output, cells[0], cells[1], cells[2]))
    return lpfx::ErrorCode::invalid_argument;
  // Reject a same-owner in-place stencil before any launch.
  if (shares_storage(input, output)) return lpfx::ErrorCode::invalid_argument;
  if (!std::isfinite(inverse_volume) || inverse_volume <= Compute(0))
    return lpfx::ErrorCode::invalid_argument;
  return lpfx::device::launch<lpfx::device::KernelLaunch<device::mass_matrix_block_threads>,
      kernels::mass_matrix_apply_kernel<
      Precision, Compute, Region, decltype(input.view()), decltype(output.view())>, lpfx::device::EmptyWork::skip>(
      executor, stencil_region_count<Region>(nodes), mass.data(), input.view(), output.view(),
      layout, dual_volume, inverse_volume, nodes, invalid);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_APPLY_DEVICE_HPP
