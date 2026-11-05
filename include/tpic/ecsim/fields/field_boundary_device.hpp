#ifndef TPIC_ECSIM_FIELDS_FIELD_BOUNDARY_DEVICE_HPP
#define TPIC_ECSIM_FIELDS_FIELD_BOUNDARY_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/core/meta.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/config/physics.hpp>
#include <tpic/ecsim/fields/field_boundary.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/fields/wall_faces.hpp>

#include <cstddef>
#include <type_traits>

namespace tpic::ecsim {
namespace detail {

// The row closure without an input: constrained components are zeroed.
struct NoRowInput {};

} // namespace detail

namespace kernels {

// Ghosts of the two physical faces normal to Axis, including transverse halos.
// Grid-stride over the face samples with KernelLaunch<> blocks; no validation flag.
template <int Axis, bool Node, class View>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void wall_ghost_kernel(
    View field, FieldBoundaryPlan plan) {
  constexpr int first = Axis == 0 ? 1 : 0;
  constexpr int second = Axis == 2 ? 1 : 2;
  lpfx::device::for_each_grid_index(static_cast<unsigned>(plan.face_count[Axis]),
                              [&](const unsigned index) {
    int ghost[3]{};
    const auto face = device::decode_x_fastest(index, unsigned(plan.extent[first] + 2));
    ghost[first] = face.first - 1;
    ghost[second] = face.second - 1;
    for (int side = 0; side < 2; ++side) {
      if (!(side ? plan.faces.upper[Axis] : plan.faces.lower[Axis])) continue;
      ghost[Axis] = side ? plan.extent[Axis] : -1;
      int source[3] = {ghost[0], ghost[1], ghost[2]};
      source[Axis] = side ? plan.extent[Axis] - 1 - int(Node) : int(Node);
      const int face_index = 2 * Axis + (side == 0);
      lpfx::meta::for_each_index<View::component_count>([&](auto component) {
        constexpr std::size_t Component = decltype(component)::value;
        const auto values = field.template component<Component>();
        using Value = typename decltype(values)::value_type;
        // The multiplier is -1 or 1: selecting the constant costs no FP64
        // conversion ahead of the loads.
        const Value multiplier = plan.multiplier[Component][face_index] < 0 ? Value(-1) : Value(1);
        values(ghost[0], ghost[1], ghost[2]) = multiplier * values(source[0], source[1], source[2]);
      });
    }
  });
}

// The constrained rows of the two physical faces normal to Axis. Grid-stride
// over the face samples with KernelLaunch<> blocks; no validation flag. Writes
// are 0 or the unchanged input value, so faces that share an edge or a corner
// store equal values in any order.
template <int Axis, class Out, class In>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void constrained_rows_kernel(
    Out out, In input, FieldBoundaryPlan plan) {
  constexpr int first = Axis == 0 ? 1 : 0;
  constexpr int second = Axis == 2 ? 1 : 2;
  lpfx::device::for_each_grid_index(static_cast<unsigned>(plan.face_count[Axis]),
                              [&](const unsigned index) {
    int node[3]{};
    const auto face = device::decode_x_fastest(index, unsigned(plan.extent[first]));
    node[first] = face.first;
    node[second] = face.second;
    for (int side = 0; side < 2; ++side) {
      const unsigned constrained = plan.constrained[2 * Axis + (side == 0)];
      if (!constrained) continue;
      node[Axis] = side ? plan.extent[Axis] - 1 : 0;
      lpfx::meta::for_each_index<Out::component_count>([&](auto component) {
        constexpr std::size_t Component = decltype(component)::value;
        if (!(constrained & (1U << Component))) return;
        const auto target = out.template component<Component>();
        using Value = typename decltype(target)::value_type;
        Value value = Value(0);
        if constexpr (!std::is_same_v<In, detail::NoRowInput>)
          value = Value(input.template component<Component>()(node[0], node[1], node[2]));
        target(node[0], node[1], node[2]) = value;
      });
    }
  });
}

} // namespace kernels

// Stream: executor; no halo is required. Writes the constrained interior node
// rows selected by plan: zero when In is detail::NoRowInput (the projection of
// solver vectors, smoothing passes, the right-hand side, the field commit and
// the Gauss-law charge),
// or the input values when In is a node vector owner (Maxwell image rows; a
// preceding projection has already zeroed those rows of the input). Output may
// alias input; unconstrained components and
// all ghosts are untouched. All helpers allocate and synchronize nothing;
// owners must outlive stream work.
template <class Out, class In, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_constrained_rows(
    Out& out, const In& input, const FieldBoundaryPlan& plan,
    lpfx::device::Executor<Backend> executor) {
  constexpr bool copy = !std::is_same_v<In, detail::NoRowInput>;
  static_assert(std::is_same_v<Backend, typename Out::space_type::backend_type>);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  const auto checked = check_field_boundary_owner<lpfx::Node>(out, plan, 0);
  if (checked != lpfx::ErrorCode::success) return checked;
  if constexpr (copy) {
    static_assert(std::is_same_v<Backend, typename In::space_type::backend_type>);
    static_assert(FieldAt<In, lpfx::Node> && In::component_count == Out::component_count);
    if (!input.initialized()) return lpfx::ErrorCode::invalid_state;
    if (input.cells() != out.cells()) return lpfx::ErrorCode::invalid_argument;
  }
  auto input_view = [&] {
    if constexpr (copy) return input.view();
    else return detail::NoRowInput{};
  }();
  const auto begun = lpfx::device::begin_device_submission<Backend>();
  if (begun != lpfx::ErrorCode::success) return begun;
  // One launch per axis with constrained faces, in x, y, z order.
  return lpfx::meta::for_each_until_error<3>([&](auto axis) {
    constexpr int Axis = int(decltype(axis)::value);
    if (!(plan.constrained[2 * Axis] | plan.constrained[2 * Axis + 1]))
      return lpfx::ErrorCode::success;
    return lpfx::device::launch<lpfx::device::KernelLaunch<>,
        kernels::constrained_rows_kernel<Axis, decltype(out.view()), decltype(input_view)>,
        lpfx::device::EmptyWork::launch, lpfx::device::LaunchPreflight::admitted>(
        executor, plan.face_count[Axis], out.view(), input_view, plan);
  });
}

// Stream: executor, after the LPFx full-direction halo copy. Fills both
// physical faces of each active axis, transverse halos included (unlike the Jhat
// Neumann closure): node ghosts reflect through the boundary node, cell ghosts
// copy the adjacent interior cell (the owner's location selects which). An edge
// or corner ghost keeps the last active axis' value; interiors are untouched.
template <class Owner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_wall_ghosts(
    Owner& field, const FieldBoundaryPlans& plans, lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename Owner::space_type::backend_type>);
  using Location = field_location_t<Owner>;
  constexpr bool node = std::is_same_v<Location, lpfx::Node>;
  const auto& plan = node ? plans.node_ghosts : plans.cell_ghosts;
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  const auto checked = check_field_boundary_owner<Location>(field, plan, 1);
  if (checked != lpfx::ErrorCode::success) return checked;
  const auto begun = lpfx::device::begin_device_submission<Backend>();
  if (begun != lpfx::ErrorCode::success) return begun;
  // R1: axes in x, y, z order on one stream; later axes read corner ghosts.
  return lpfx::meta::for_each_until_error<3>([&](auto axis) {
    constexpr int Axis = int(decltype(axis)::value);
    if (!plan.faces.lower[Axis] && !plan.faces.upper[Axis]) return lpfx::ErrorCode::success;
    return lpfx::device::launch<lpfx::device::KernelLaunch<>,
        kernels::wall_ghost_kernel<Axis, node, decltype(field.view())>,
        lpfx::device::EmptyWork::launch, lpfx::device::LaunchPreflight::admitted>(
        executor, plan.face_count[Axis], field.view(), plan);
  });
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_FIELD_BOUNDARY_DEVICE_HPP
