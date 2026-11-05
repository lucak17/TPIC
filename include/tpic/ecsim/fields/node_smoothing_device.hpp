#ifndef TPIC_ECSIM_FIELDS_NODE_SMOOTHING_DEVICE_HPP
#define TPIC_ECSIM_FIELDS_NODE_SMOOTHING_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/core/meta.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/ecsim/fields/cell_to_node.hpp>
#include <tpic/ecsim/fields/node_smoothing.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace tpic::ecsim {
namespace kernels {

// Grid-stride over the interior nodes with KernelLaunch<> blocks; *invalid,
// when given, flags a non-finite stored result.
template <config::EcsimSmoothingOperator Operator, class Compute,
          class SourceView, class TargetView>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void node_smoothing_kernel(
    SourceView source, TargetView target, SmoothingWeights<Operator, Compute> weights,
    data::GridShape shape, unsigned* invalid) {
  lpfx::device::for_each_grid_index(static_cast<unsigned>(shape.count), [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, shape.nx, shape.ny);
    const int i = at.i, j = at.j, k = at.k;
    lpfx::meta::for_each_index<TargetView::component_count>([&](auto component) {
      constexpr std::size_t Component = decltype(component)::value;
      const auto destination = target.template component<Component>();
      using Value = typename decltype(destination)::value_type;
      const Value value = config::convert<Value>(node_smoothing_value<Operator>(
          source.template component<Component>(), i, j, k, weights));
      destination(i, j, k) = value;
      if (invalid && !::isfinite(value)) atomicExch(invalid, 1U);
    });
  });
}

} // namespace kernels

// One smoothing pass over the interior nodes of every component, in Compute
// arithmetic (the solver's by default).
// Stream: executor, after the source nodes and their one-node halo are coherent
// on it. Writes target interiors (ghosts untouched); *invalid, when given and
// cleared by the caller, flags a non-finite result. The caller owns repeated
// passes and the exchange and closure between them. alpha is finite in [0, 1].
template <class Precision, class Compute = typename Precision::Compute::Electric,
          class SourceOwner, class TargetOwner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_node_smoothing(
    const SourceOwner& source, TargetOwner& target,
    config::EcsimSmoothingOperator operation, Compute alpha,
    unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename SourceOwner::space_type::backend_type> &&
                std::is_same_v<Backend, typename TargetOwner::space_type::backend_type>);
  static_assert(std::is_floating_point_v<Compute>);
  if (!executor.bound() || !std::isfinite(alpha) || alpha < Compute(0) || alpha > Compute(1) ||
      (operation != config::EcsimSmoothingOperator::Face7 &&
       operation != config::EcsimSmoothingOperator::Binomial27))
    return lpfx::ErrorCode::invalid_argument;
  data::GridShape shape;
  const auto checked = detail::try_prepare_node_smoothing(source, target, shape);
  if (checked != lpfx::ErrorCode::success) return checked;
  using SourceView = decltype(source.view());
  using TargetView = decltype(target.view());
  if (operation == config::EcsimSmoothingOperator::Face7)
    return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernels::node_smoothing_kernel<
        config::EcsimSmoothingOperator::Face7, Compute, SourceView, TargetView>>(
        executor, shape.count, source.view(), target.view(),
        SmoothingWeights<config::EcsimSmoothingOperator::Face7, Compute>{
            alpha, (Compute(1) - alpha) / Compute(6)},
        shape, invalid);
  return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernels::node_smoothing_kernel<
      config::EcsimSmoothingOperator::Binomial27, Compute, SourceView, TargetView>>(
      executor, shape.count, source.view(), target.view(),
      SmoothingWeights<config::EcsimSmoothingOperator::Binomial27, Compute>{}, shape,
      invalid);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_NODE_SMOOTHING_DEVICE_HPP
