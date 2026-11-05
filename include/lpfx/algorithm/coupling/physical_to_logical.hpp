#ifndef LPFX_ALGORITHM_COUPLING_PHYSICAL_TO_LOGICAL_HPP
#define LPFX_ALGORITHM_COUPLING_PHYSICAL_TO_LOGICAL_HPP

#include <cmath>
#include <concepts>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/algorithm/field/differential/spacing.hpp>

namespace lpfx {

// Checked setup retains exactly the six values required by the hot affine
// transform. Domain bounds and halo ownership are deliberately not part of
// this object: callers prove that the resulting stencil is addressable.
template <std::floating_point Scalar> class PhysicalToLogical3D final {
public:
  using scalar_type = Scalar;

  constexpr PhysicalToLogical3D() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Scalar origin_x() const noexcept {
    return origin_x_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Scalar origin_y() const noexcept {
    return origin_y_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Scalar origin_z() const noexcept {
    return origin_z_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Scalar
  inverse_spacing_x() const noexcept {
    return inverse_spacing_x_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Scalar
  inverse_spacing_y() const noexcept {
    return inverse_spacing_y_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Scalar
  inverse_spacing_z() const noexcept {
    return inverse_spacing_z_;
  }

  template <std::floating_point Physical>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Scalar
  logical_x(const Physical physical) const noexcept {
    return (static_cast<Scalar>(physical) - origin_x_) * inverse_spacing_x_;
  }

  template <std::floating_point Physical>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Scalar
  logical_y(const Physical physical) const noexcept {
    return (static_cast<Scalar>(physical) - origin_y_) * inverse_spacing_y_;
  }

  template <std::floating_point Physical>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Scalar
  logical_z(const Physical physical) const noexcept {
    return (static_cast<Scalar>(physical) - origin_z_) * inverse_spacing_z_;
  }

private:
  constexpr PhysicalToLogical3D(const Scalar origin_x, const Scalar origin_y,
                                const Scalar origin_z,
                                const Scalar inverse_spacing_x,
                                const Scalar inverse_spacing_y,
                                const Scalar inverse_spacing_z) noexcept
      : origin_x_(origin_x), origin_y_(origin_y), origin_z_(origin_z),
        inverse_spacing_x_(inverse_spacing_x),
        inverse_spacing_y_(inverse_spacing_y),
        inverse_spacing_z_(inverse_spacing_z) {}

  Scalar origin_x_{};
  Scalar origin_y_{};
  Scalar origin_z_{};
  Scalar inverse_spacing_x_{};
  Scalar inverse_spacing_y_{};
  Scalar inverse_spacing_z_{};

  friend struct detail::PrivateConstructionAccess;
};

static_assert(sizeof(PhysicalToLogical3D<float>) == 6 * sizeof(float));
static_assert(sizeof(PhysicalToLogical3D<double>) == 6 * sizeof(double));

// This is the only fallible layer. Rejection is transactional and collapses
// every invalid numeric setup to invalid_argument.
template <std::floating_point Scalar>
[[nodiscard]] inline ErrorCode
try_make_physical_to_logical_3d(const Scalar origin_x, const Scalar origin_y,
                                const Scalar origin_z, const Scalar spacing_x,
                                const Scalar spacing_y, const Scalar spacing_z,
                                PhysicalToLogical3D<Scalar> &output) noexcept {
  Scalar inverse_spacing_x{};
  Scalar inverse_spacing_y{};
  Scalar inverse_spacing_z{};
  if (!std::isfinite(origin_x) || !std::isfinite(origin_y) ||
      !std::isfinite(origin_z) ||
      !detail::try_invert_spacing_3d(spacing_x, spacing_y, spacing_z,
                                     inverse_spacing_x, inverse_spacing_y,
                                     inverse_spacing_z)) {
    return ErrorCode::invalid_argument;
  }
  output =
      detail::PrivateConstructionAccess::create<PhysicalToLogical3D<Scalar>>(
          origin_x, origin_y, origin_z, inverse_spacing_x, inverse_spacing_y,
          inverse_spacing_z);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
