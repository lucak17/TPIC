#ifndef TPIC_ECSIM_DEPOSITION_MOMENT_FIELD_DEVICE_HPP
#define TPIC_ECSIM_DEPOSITION_MOMENT_FIELD_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/config/domain.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/data/fields/grid_shape.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/fields/stencil_region.hpp>
#include <tpic/ecsim/fields/wall_faces.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tpic::ecsim {

// The run-constant plan of the moment closure kernels, a pure function of the
// domain, prepared once by each owner of a moment field: the physical wall
// faces, the interior-node box whose shell the dual-volume scale visits and
// the box with one ghost layer whose shell the Neumann closure fills.
struct MomentFieldPlan {
  WallFaces faces;
  StencilRegionGeometry nodes;  // Interior nodes: cells + 1 per axis.
  StencilRegionGeometry ghosts; // Interior nodes and one ghost layer: cells + 3.
  bool any_wall = false;
};

[[nodiscard]] inline MomentFieldPlan prepare_moment_field_plan(
    const config::PreparedDomain& domain) {
  constexpr const char* failed = "TPIC moments: closure plan preparation failed";
  MomentFieldPlan plan;
  lpfx::check_error(try_prepare_wall_faces(domain, plan.faces), failed);
  const int* cells = plan.faces.cells;
  lpfx::check_error(try_make_stencil_region_geometry(
      cells[0] + 1, cells[1] + 1, cells[2] + 1, plan.nodes), failed);
  lpfx::check_error(try_make_stencil_region_geometry(
      cells[0] + 3, cells[1] + 3, cells[2] + 3, plan.ghosts), failed);
  for (int axis = 0; axis < 3; ++axis)
    plan.any_wall |= plan.faces.lower[axis] || plan.faces.upper[axis];
  return plan;
}

namespace detail {

// The moment owner matches the plan's cells and has one ghost layer.
template <class Owner>
[[nodiscard]] lpfx::ErrorCode check_moment_owner(
    const Owner& field, const MomentFieldPlan& plan) noexcept {
  static_assert(components_at<Owner, lpfx::Node>(
      std::make_index_sequence<Owner::component_count>{}));
  if (!field.initialized()) return lpfx::ErrorCode::invalid_state;
  const int* cells = plan.faces.cells;
  if (!field_cells_equal(field, cells[0], cells[1], cells[2]) || !has_halo(field, 1))
    return lpfx::ErrorCode::invalid_argument;
  return lpfx::ErrorCode::success;
}

LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool moment_neumann_coordinate(
    int coordinate, int samples, bool lower, bool upper, int& source) noexcept {
  if (coordinate < 0) {
    if (!lower) return false;
    source = 1;
  } else if (coordinate >= samples) {
    if (!upper) return false;
    source = samples - 2;
  } else {
    source = coordinate;
  }
  return true;
}

} // namespace detail

namespace kernels {

// Only boundary nodes can carry a factor other than one, so the launch covers
// the shell of the interior-node box; each node is visited once. Grid-stride
// with KernelLaunch<> blocks; no validation flag.
template <class View, std::size_t... I>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void scale_moment_dual_volume_kernel(
    View field, WallFaces faces, StencilRegionGeometry nodes) {
  lpfx::device::for_each_grid_index(nodes.shell_count, [&](const unsigned index) {
    int i, j, k;
    stencil_region_coordinates<StencilRegion::shell>(index, nodes, i, j, k);
    const unsigned factor = faces.dual_volume_factor(i, j, k);
    if (factor != 1)
      ((field.template component<I>()(i, j, k) *=
            config::convert<typename decltype(field.template component<I>())::value_type>(factor)), ...);
  });
}

// The launch covers the ghost layer only: the shell of the box with one ghost
// layer, shifted by one, never an interior node. Grid-stride with
// KernelLaunch<> blocks; no validation flag.
template <class View, std::size_t... I>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void fill_moment_neumann_ghost_kernel(
    View field, WallFaces faces, StencilRegionGeometry nodes, StencilRegionGeometry ghosts) {
  lpfx::device::for_each_grid_index(ghosts.shell_count, [&](const unsigned index) {
    int i, j, k;
    stencil_region_coordinates<StencilRegion::shell>(index, ghosts, i, j, k);
    --i;
    --j;
    --k;
    int x, y, z;
    if (detail::moment_neumann_coordinate(i, nodes.nx, faces.lower[0], faces.upper[0], x) &&
        detail::moment_neumann_coordinate(j, nodes.ny, faces.lower[1], faces.upper[1], y) &&
        detail::moment_neumann_coordinate(k, nodes.nz, faces.lower[2], faces.upper[2], z))
      ((field.template component<I>()(i, j, k) = field.template component<I>()(x, y, z)), ...);
  });
}

// Zeroes every node of the halo-inclusive box. Grid-stride over shape.count
// with KernelLaunch<> blocks; no validation flag.
template <class View, std::size_t... Component>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void clear_moment_field_kernel(
    View field, int halo_x, int halo_y, int halo_z, data::GridShape shape) {
  lpfx::device::for_each_grid_index(static_cast<unsigned>(shape.count), [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, unsigned(shape.nx), unsigned(shape.ny));
    const int i = at.i - halo_x;
    const int j = at.j - halo_y;
    const int k = at.k - halo_z;
    ((field.template component<Component>()(i, j, k) = 0), ...);
  });
}

} // namespace kernels

// Deposition divided by the full cell volume; wall nodes have control volumes
// of 1/2, 1/4 and 1/8 cell on faces, edges and corners, so they are multiplied
// by 2, 4 and 8. Field periodicity, not particle boundaries, selects the walls.
// Stream: executor, exactly once, after the shared-node addition and before the
// ghost exchange. Writes interior nodes only; plan from prepare_moment_field_plan.
template <class Owner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_scale_moment_dual_volume(
    Owner& field, const MomentFieldPlan& plan, lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename Owner::space_type::backend_type>);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  const auto checked = detail::check_moment_owner(field, plan);
  if (checked != lpfx::ErrorCode::success) return checked;
  const auto begun = lpfx::device::begin_device_submission<Backend>();
  if (begun != lpfx::ErrorCode::success) return begun;
  if (!plan.any_wall) return lpfx::ErrorCode::success;
  using View = decltype(field.view());
  constexpr auto kernel = []<std::size_t... I>(std::index_sequence<I...>) {
    return kernels::scale_moment_dual_volume_kernel<View, I...>;
  }(std::make_index_sequence<Owner::component_count>{});
  return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernel, lpfx::device::EmptyWork::launch,
      lpfx::device::LaunchPreflight::admitted>(executor, plan.nodes.shell_count, field.view(),
      plan.faces, plan.nodes);
}

// Jhat Neumann closure. Stream: executor, after the ghost exchange. Mirrors
// one layer across physical walls (source node 1 or n-2 on every out-of-range
// axis); a ghost along a communicating axis keeps its exchanged value, edges
// and corners shared with a physical face included. Charge density has no
// such closure, so the caller chooses it explicitly.
template <class Owner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_fill_moment_neumann_ghost(
    Owner& field, const MomentFieldPlan& plan, lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename Owner::space_type::backend_type>);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  const auto checked = detail::check_moment_owner(field, plan);
  if (checked != lpfx::ErrorCode::success) return checked;
  const auto begun = lpfx::device::begin_device_submission<Backend>();
  if (begun != lpfx::ErrorCode::success) return begun;
  if (!plan.any_wall) return lpfx::ErrorCode::success;
  using View = decltype(field.view());
  constexpr auto kernel = []<std::size_t... I>(std::index_sequence<I...>) {
    return kernels::fill_moment_neumann_ghost_kernel<View, I...>;
  }(std::make_index_sequence<Owner::component_count>{});
  return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernel, lpfx::device::EmptyWork::launch,
      lpfx::device::LaunchPreflight::admitted>(executor, plan.ghosts.shell_count, field.view(),
      plan.faces, plan.nodes, plan.ghosts);
}

// Zeroes every logical node and ghost of a moment field, LPFx pitches kept.
// Stream: executor, before deposition. All components of the owner share the
// nodal location and halo depth.
template <class Owner, class Backend, class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_clear_moment_field(
    Owner& field, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  static_assert(std::is_same_v<Backend, typename Owner::space_type::backend_type>);
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  if (!field.initialized()) return lpfx::ErrorCode::invalid_state;
  const auto halo = field.template component_halo<0>();
  const bool matching = [&]<std::size_t... Component>(std::index_sequence<Component...>) {
    static_assert((std::is_same_v<typename decltype(field.view().template component<Component>())::location_type,
                                 lpfx::Node> && ...));
    return ((halo == field.template component_halo<Component>()) && ...);
  }(std::make_index_sequence<Owner::component_count>{});
  if (!matching) return lpfx::ErrorCode::invalid_argument;
  data::GridShape shape;
  const auto checked = data::try_make_grid_shape(field.cells(), 1,
      {{int(halo.x_minus), int(halo.y_minus), int(halo.z_minus)},
       {int(halo.x_plus), int(halo.y_plus), int(halo.z_plus)}}, shape);
  if (checked != lpfx::ErrorCode::success) return checked;
  using View = decltype(field.view());
  constexpr auto kernel = []<std::size_t... Component>(std::index_sequence<Component...>) {
    return kernels::clear_moment_field_kernel<View, Component...>;
  }(std::make_index_sequence<Owner::component_count>{});
  return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernel>(executor, diagnostic, shape.count,
      field.view(), halo.x_minus, halo.y_minus, halo.z_minus, shape);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_DEPOSITION_MOMENT_FIELD_DEVICE_HPP
