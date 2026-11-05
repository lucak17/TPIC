#ifndef TPIC_DATA_OUTPUT_OUTPUT_MOMENT_SCHEMA_HPP
#define TPIC_DATA_OUTPUT_OUTPUT_MOMENT_SCHEMA_HPP

#include <lpfx/data/field/component.hpp>
#include <tpic/data/output/output_moment_quantities.hpp>

namespace tpic::data {

// Output moments are independent of the solver's rotated current Jhat.
// P is the raw mass-weighted second moment, without subtracting bulk flow.
// Reduction storage keeps diagnostic accumulation in FP64 in every compute precision.
template <class Precision> struct OutputMomentSchema {
  using Scalar = typename Precision::Compute::Reduction;
  using Rho = lpfx::field::component<"rho", Scalar, lpfx::field::node>;
  using Jx = lpfx::field::component<"jx", Scalar, lpfx::field::node>;
  using Jy = lpfx::field::component<"jy", Scalar, lpfx::field::node>;
  using Jz = lpfx::field::component<"jz", Scalar, lpfx::field::node>;
  using Pxx = lpfx::field::component<"pxx", Scalar, lpfx::field::node>;
  using Pxy = lpfx::field::component<"pxy", Scalar, lpfx::field::node>;
  using Pxz = lpfx::field::component<"pxz", Scalar, lpfx::field::node>;
  using Pyy = lpfx::field::component<"pyy", Scalar, lpfx::field::node>;
  using Pyz = lpfx::field::component<"pyz", Scalar, lpfx::field::node>;
  using Pzz = lpfx::field::component<"pzz", Scalar, lpfx::field::node>;
  using type = lpfx::field::schema<Rho, Jx, Jy, Jz, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz>;
  static_assert(type::component_count == output_moment_component_count,
                "the output-moment schema holds the components of output_moment_quantities");
};

} // namespace tpic::data

#endif // TPIC_DATA_OUTPUT_OUTPUT_MOMENT_SCHEMA_HPP
