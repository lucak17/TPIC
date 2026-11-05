#ifndef TPIC_ECSIM_FIELDS_NODE_SMOOTHING_HPP
#define TPIC_ECSIM_FIELDS_NODE_SMOOTHING_HPP

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/ecsim/fields/cell_to_node.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>


namespace tpic::ecsim {

// Face7 is the seven-point stencil: the centre weighted by alpha and each of
// the six face neighbours by beta = (1 - alpha) / 6, formed on the host in
// Compute. Binomial27 has fixed weights.
template <config::EcsimSmoothingOperator Operator, class Compute> struct SmoothingWeights {};

template <class Compute>
struct SmoothingWeights<config::EcsimSmoothingOperator::Face7, Compute> {
  Compute alpha, beta;
};

// The addition order below (face, edge and corner sums, each left to right)
// is part of the smoothing contract; promote each load before any arithmetic.
// Binomial27 is the tensor product [1,2,1]^3 / 64.
template <config::EcsimSmoothingOperator Operator, class Compute, class Component>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Compute node_smoothing_value(
    Component source, int i, int j, int k, SmoothingWeights<Operator, Compute> weights) noexcept {
  const Compute face_sum =
      Compute(source(i - 1, j, k)) + Compute(source(i + 1, j, k)) +
      Compute(source(i, j - 1, k)) + Compute(source(i, j + 1, k)) +
      Compute(source(i, j, k - 1)) + Compute(source(i, j, k + 1));
  if constexpr (Operator == config::EcsimSmoothingOperator::Face7) {
    return weights.alpha * Compute(source(i, j, k)) + weights.beta * face_sum;
  } else {
    const Compute edge_sum =
        Compute(source(i - 1, j - 1, k)) + Compute(source(i + 1, j - 1, k)) +
        Compute(source(i - 1, j + 1, k)) + Compute(source(i + 1, j + 1, k)) +
        Compute(source(i - 1, j, k - 1)) + Compute(source(i + 1, j, k - 1)) +
        Compute(source(i - 1, j, k + 1)) + Compute(source(i + 1, j, k + 1)) +
        Compute(source(i, j - 1, k - 1)) + Compute(source(i, j + 1, k - 1)) +
        Compute(source(i, j - 1, k + 1)) + Compute(source(i, j + 1, k + 1));
    const Compute corner_sum =
        Compute(source(i - 1, j - 1, k - 1)) +
        Compute(source(i + 1, j - 1, k - 1)) +
        Compute(source(i - 1, j + 1, k - 1)) +
        Compute(source(i + 1, j + 1, k - 1)) +
        Compute(source(i - 1, j - 1, k + 1)) +
        Compute(source(i + 1, j - 1, k + 1)) +
        Compute(source(i - 1, j + 1, k + 1)) +
        Compute(source(i + 1, j + 1, k + 1));
    return (Compute(8) * Compute(source(i, j, k)) + Compute(4) * face_sum +
            Compute(2) * edge_sum + corner_sum) * (Compute(1) / Compute(64));
  }
}

namespace detail {

template <class SourceOwner, class TargetOwner>
[[nodiscard]] lpfx::ErrorCode try_prepare_node_smoothing(
    const SourceOwner& source, const TargetOwner& target,
    data::GridShape& shape) noexcept {
  static_assert(FieldAt<SourceOwner, lpfx::Node> && FieldAt<TargetOwner, lpfx::Node> &&
                SourceOwner::component_count == TargetOwner::component_count);
  shape = {};
  if (!source.initialized() || !target.initialized()) return lpfx::ErrorCode::invalid_state;
  if (source.cells() != target.cells()) return lpfx::ErrorCode::invalid_argument;
  // Component halo depths may differ: has_halo checks every component.
  if (!has_halo(source, 1)) return lpfx::ErrorCode::invalid_argument;
  // Reject in-place stencils before launching any work.
  if (shares_storage(source, target)) return lpfx::ErrorCode::invalid_argument;
  return data::try_make_grid_shape(source.cells(), 1, {}, shape);
}

} // namespace detail
} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_NODE_SMOOTHING_HPP
