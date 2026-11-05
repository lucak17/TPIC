#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_SPACING_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_SPACING_HPP

#include <cmath>
#include <concepts>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>

namespace lpfx {

namespace detail {

// Writes the three reciprocals only when every spacing and every reciprocal
// is finite and strictly positive.
template <std::floating_point Scalar>
[[nodiscard]] inline bool
try_invert_spacing_3d(const Scalar spacing_x, const Scalar spacing_y,
                      const Scalar spacing_z, Scalar &inverse_x,
                      Scalar &inverse_y, Scalar &inverse_z) noexcept {
  if (!std::isfinite(spacing_x) || !std::isfinite(spacing_y) ||
      !std::isfinite(spacing_z) || !(spacing_x > Scalar{0}) ||
      !(spacing_y > Scalar{0}) || !(spacing_z > Scalar{0})) {
    return false;
  }
  const Scalar x = Scalar{1} / spacing_x;
  const Scalar y = Scalar{1} / spacing_y;
  const Scalar z = Scalar{1} / spacing_z;
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
      !(x > Scalar{0}) || !(y > Scalar{0}) || !(z > Scalar{0})) {
    return false;
  }
  inverse_x = x;
  inverse_y = y;
  inverse_z = z;
  return true;
}

} // namespace detail

// The three inverse spacings consumed by differential hot paths.
template <std::floating_point Scalar> class DifferentialSpacing3D final {
public:
  using scalar_type = Scalar;

  constexpr DifferentialSpacing3D() noexcept = default;

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

private:
  constexpr DifferentialSpacing3D(const Scalar inverse_spacing_x,
                                  const Scalar inverse_spacing_y,
                                  const Scalar inverse_spacing_z) noexcept
      : inverse_spacing_x_(inverse_spacing_x),
        inverse_spacing_y_(inverse_spacing_y),
        inverse_spacing_z_(inverse_spacing_z) {}

  Scalar inverse_spacing_x_{};
  Scalar inverse_spacing_y_{};
  Scalar inverse_spacing_z_{};

  friend struct detail::PrivateConstructionAccess;
};

// Transactional: spacings whose reciprocal overflows are invalid even when
// the input itself is finite and positive.
template <std::floating_point Scalar>
[[nodiscard]] inline ErrorCode try_make_differential_spacing_3d(
    const Scalar spacing_x, const Scalar spacing_y, const Scalar spacing_z,
    DifferentialSpacing3D<Scalar> &output) noexcept {
  Scalar inverse_x{};
  Scalar inverse_y{};
  Scalar inverse_z{};
  if (!detail::try_invert_spacing_3d(spacing_x, spacing_y, spacing_z,
                                     inverse_x, inverse_y, inverse_z)) {
    return ErrorCode::invalid_argument;
  }
  output = detail::PrivateConstructionAccess::create<
      DifferentialSpacing3D<Scalar>>(inverse_x, inverse_y, inverse_z);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
