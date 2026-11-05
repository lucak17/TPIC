#ifndef TPIC_DATA_OUTPUT_GAUSS_SAMPLE_HPP
#define TPIC_DATA_OUTPUT_GAUSS_SAMPLE_HPP

#include <lpfx/data/field/component.hpp>

#include <array>
#include <cstddef>

namespace tpic::data {

// The one-component fields of the Gauss diagnostic, in the reduction type: the
// nodal charge density with its smoothing partner, and per cell the sample
// stash, the first finished residual and the last one.
template <class Precision> struct GaussFieldSchema {
  using Scalar = typename Precision::Compute::Reduction;
  using Node = lpfx::field::schema<lpfx::field::component<"q", Scalar, lpfx::field::node>>;
  using Cell = lpfx::field::schema<lpfx::field::component<"s", Scalar, lpfx::field::cell>>;
};

// The summed squares of one sample in slot and column order: the residual, its
// change since the first finished sample, the residual in the cells touching a
// physical wall, and the cell charge 4*pi*rho. The field slots are reduced when
// the sample finishes, the charge slot when it begins.
struct GaussScalarIndex {
  static constexpr std::size_t residual = 0, drift = 1, wall = 2, charge = 3;
};
inline constexpr std::size_t gauss_field_scalar_count = GaussScalarIndex::wall + 1;
inline constexpr std::size_t gauss_scalar_count = GaussScalarIndex::charge + 1;

// A finished sample, begun at cycle: root mean squares in GaussScalarIndex order.
struct GaussSample {
  int cycle = 0;
  double time = 0.0;
  std::array<double, gauss_scalar_count> rms{};
};

} // namespace tpic::data

#endif // TPIC_DATA_OUTPUT_GAUSS_SAMPLE_HPP
