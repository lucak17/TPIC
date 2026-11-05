#ifndef TPIC_ECSIM_FIELDS_FIELD_VECTOR_DEVICE_HPP
#define TPIC_ECSIM_FIELDS_FIELD_VECTOR_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/core/meta.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/grid_shape.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace tpic::ecsim {

// One term of a linear combination: a coefficient times a vector field (an
// owner on the host, its view in the kernel).
template <class Compute, class Field> struct ScaledField {
  Compute coefficient;
  Field field;
};

template <class Compute, class Owner>
[[nodiscard]] ScaledField<Compute, const Owner&> scaled(Compute coefficient, const Owner& field) {
  return {coefficient, field};
}

namespace kernels {

// out = ((c0 * f0 + c1 * f1) + c2 * f2) ... per component: each load promoted
// to Compute, a left fold over the terms, one narrowing to the output type.
// Grid-stride over the interior with KernelLaunch<> blocks; *invalid, when
// given, flags a non-finite stored value.
template <class Compute, class Out, class... Fields>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void field_vector_linear_combination_kernel(
    Out out, data::GridShape shape, unsigned* invalid, ScaledField<Compute, Fields>... terms) {
  lpfx::device::for_each_grid_index(unsigned(shape.count), [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, shape.nx, shape.ny);
    const int i = at.i, j = at.j, k = at.k;
    lpfx::meta::for_each_index<3>([&](auto component) {
      constexpr std::size_t Component = decltype(component)::value;
      using Value = typename decltype(out.template component<Component>())::value_type;
      const Value value = config::convert<Value>((... + (terms.coefficient *
          Compute(terms.field.template component<Component>()(i, j, k)))));
      out.template component<Component>()(i, j, k) = value;
      if (invalid && !::isfinite(value)) atomicExch(invalid, 1U);
    });
  });
}

// Copies logical values (and halo layers when halo is nonzero), narrowing to
// the output type. Grid-stride with KernelLaunch<> blocks; *invalid, when given,
// flags a non-finite stored value.
template <class Out, class In>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void field_vector_copy_kernel(
    Out out, In in, int halo, data::GridShape shape, unsigned* invalid) {
  lpfx::device::for_each_grid_index(unsigned(shape.count), [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, shape.nx, shape.ny);
    const int i = at.i - halo, j = at.j - halo, k = at.k - halo;
    lpfx::meta::for_each_index<3>([&](auto component) {
      constexpr std::size_t Component = decltype(component)::value;
      using Value = typename decltype(out.template component<Component>())::value_type;
      const Value value = config::convert<Value>(in.template component<Component>()(i, j, k));
      out.template component<Component>()(i, j, k) = value;
      if (invalid && !::isfinite(value)) atomicExch(invalid, 1U);
    });
  });
}

} // namespace kernels

template <class Owner>
data::GridShape field_vector_shape(const Owner& owner, int halo = 0) {
  static_assert(Owner::component_count == 3);
  constexpr int node = std::is_same_v<field_location_t<Owner>, lpfx::Node>;
  if (!owner.initialized() || halo < 0 || halo > 1)
    throw std::invalid_argument("TPIC fields: invalid vector owner or halo");
  data::GridShape shape;
  if (data::try_make_grid_shape(owner.cells(), node, {{halo, halo, halo}, {halo, halo, halo}},
                                shape) != lpfx::ErrorCode::success)
    throw std::overflow_error("TPIC fields: vector extent overflow");
  return shape;
}

template <class A, class B>
void check_field_vector_pair(const A& a, const B& b, int halo = 0) {
  static_assert(VectorFieldAt<A, field_location_t<A>> && VectorFieldAt<B, field_location_t<A>>);
  if (!a.initialized() || !b.initialized() || a.cells() != b.cells())
    throw std::invalid_argument("TPIC fields: incompatible vectors");
  if (!has_halo(a, halo) || !has_halo(b, halo))
    throw std::invalid_argument("TPIC fields: vector halo is too narrow");
}

// Pointwise combinations allow output to alias any input. Only interiors are
// written; callers own ghost validity. Storage and arithmetic may differ.
// Incompatible owners are a programming error and throw; an unbound executor
// or a failed launch is returned. The terms come last so that their owners
// are deduced.
template <class Precision, class Compute = typename Precision::Compute::Electric, class Out, class Backend,
          class... Fields>
[[nodiscard]] lpfx::ErrorCode enqueue_field_vector_linear_combination(
    Out& out, unsigned* invalid, lpfx::device::Executor<Backend> executor,
    const ScaledField<Compute, const Fields&>&... terms) {
  static_assert(sizeof...(Fields) >= 2, "a linear combination has at least two terms");
  (ecsim::check_field_vector_pair(out, terms.field), ...);
  const auto shape = ecsim::field_vector_shape(out);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernels::field_vector_linear_combination_kernel<
      Compute, decltype(out.view()), decltype(terms.field.view())...>>(
      executor, shape.count, out.view(), shape, invalid,
      ScaledField<Compute, decltype(terms.field.view())>{terms.coefficient, terms.field.view()}...);
}

// Copy logical values with their pitches, including a complete one-cell halo
// when requested. No transfer/owner moves invalidate registered field bindings.
template <class Out, class In, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_field_vector_copy(Out& out, const In& in,
    bool include_halo, unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  const int halo = include_halo ? 1 : 0;
  ecsim::check_field_vector_pair(out, in, halo);
  const auto shape = ecsim::field_vector_shape(out, halo);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  return lpfx::device::launch<lpfx::device::KernelLaunch<>,
      kernels::field_vector_copy_kernel<decltype(out.view()), decltype(in.view())>>(
      executor, shape.count, out.view(), in.view(), halo, shape, invalid);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_FIELD_VECTOR_DEVICE_HPP
