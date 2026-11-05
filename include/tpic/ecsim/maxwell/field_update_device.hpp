#ifndef TPIC_ECSIM_MAXWELL_FIELD_UPDATE_DEVICE_HPP
#define TPIC_ECSIM_MAXWELL_FIELD_UPDATE_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/core/meta.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/fields/field_vector_device.hpp>
#include <tpic/ecsim/fields/node_cell_curl.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tpic::ecsim {
namespace kernels {

// Faraday update next = one * magnetic + faraday * curl(theta) over all cells:
// the node-to-cell curl of theta is evaluated in registers in Compute, then the
// two-term combination of field_vector_linear_combination_kernel is formed with
// the same expression and one narrowing to the next-field type. Grid-stride
// over all cells with KernelLaunch<> blocks; *invalid flags a non-finite value.
template <class Compute, class Next, class Magnetic, class Theta>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void faraday_update_kernel(
    Theta theta, Magnetic magnetic, Next next, NodeCellSpacing<Compute> spacing, Compute one,
    Compute faraday, data::GridShape shape, unsigned* invalid) {
  lpfx::device::for_each_grid_index(unsigned(shape.count), [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, shape.nx, shape.ny);
    const int i = at.i, j = at.j, k = at.k;
    const auto curl = node_cell_curl(theta, i, j, k, spacing);
    const Compute rotation[3] = {curl.x, curl.y, curl.z};
    lpfx::meta::for_each_index<3>([&](auto component) {
      constexpr std::size_t Component = decltype(component)::value;
      using Value = typename decltype(next.template component<Component>())::value_type;
      const Value value = config::convert<Value>(
          one * config::convert<Compute>(magnetic.template component<Component>()(i, j, k)) +
          faraday * rotation[Component]);
      next.template component<Component>()(i, j, k) = value;
      if (invalid && !::isfinite(value)) atomicExch(invalid, 1U);
    });
  });
}

} // namespace kernels

// Stream: executor, after theta's interior nodes are valid on it; magnetic is
// the old cell-centred B. Writes E from theta in Compute::Electric and
// next_cell by the Faraday update in Compute::Magnetic, flagging a non-finite
// value in *invalid; the caller exchanges the next_cell halo next. Throws on an
// invalid owner or a failed launch.
template <class Precision, class E, class B, class Theta, class Next, class Backend>
void enqueue_next_fields(E& electric, const B& magnetic, const Theta& theta, Next& next_cell,
    const NodeCellSpacing<typename Precision::Compute::Magnetic>& spacing,
    typename Precision::Compute::Electric theta_value, typename Precision::Compute::Magnetic faraday,
    unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  using Compute = typename Precision::Compute::Electric;
  using MagneticCompute = typename Precision::Compute::Magnetic;
  if (std::abs(theta_value - Compute(1)) < Compute(1e-10))
    lpfx::check_error(enqueue_field_vector_copy(electric, theta, false, invalid, executor),
                      "TPIC Maxwell: theta electric copy failed");
  else
    lpfx::check_error(enqueue_field_vector_linear_combination<Precision>(electric, invalid,
        executor, scaled(Compute(1) / theta_value, theta),
        scaled(-(Compute(1) - theta_value) / theta_value, electric)),
        "TPIC Maxwell: theta electric combination failed");
  static_assert(VectorFieldAt<Theta, lpfx::Node>);
  ecsim::check_field_vector_pair(next_cell, magnetic);
  if (!theta.initialized() || theta.cells() != next_cell.cells())
    throw std::invalid_argument("TPIC Maxwell: incompatible Faraday source");
  const auto shape = ecsim::field_vector_shape(next_cell);
  lpfx::require(executor.bound(), "TPIC Maxwell: invalid Faraday executor");
  lpfx::check_error(lpfx::device::launch<lpfx::device::KernelLaunch<>, kernels::faraday_update_kernel<
      MagneticCompute, decltype(next_cell.view()), decltype(magnetic.view()), decltype(theta.view())>>(
      executor, shape.count, theta.view(), magnetic.view(), next_cell.view(), spacing,
      MagneticCompute(1), faraday, shape, invalid), "TPIC Maxwell: Faraday update failed");
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MAXWELL_FIELD_UPDATE_DEVICE_HPP
