#ifndef TPIC_DATA_OUTPUT_OUTPUT_MOMENT_QUANTITIES_HPP
#define TPIC_DATA_OUTPUT_OUTPUT_MOMENT_QUANTITIES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string_view>

namespace tpic::data {

// Physical output layout, independent of rotated Jhat and compact ECSIM mass.
// Pressure is the raw mass-weighted second moment; rho is scaled by 4*pi
// only when captured, after physical dual-volume correction.
struct OutputMomentQuantity {
  std::string_view name;
  std::size_t first_component;
  std::size_t components;
  double scale;

  [[nodiscard]] constexpr std::uint16_t mask() const noexcept {
    return static_cast<std::uint16_t>(((1U << components) - 1U)
                                      << first_component);
  }
};
inline constexpr auto output_moment_quantities = std::to_array<OutputMomentQuantity>({
    {"rho", 0, 1, 4 * std::numbers::pi},
    {"J", 1, 3, 1},
    {"PXX", 4, 1, 1},
    {"PXY", 5, 1, 1},
    {"PXZ", 6, 1, 1},
    {"PYY", 7, 1, 1},
    {"PYZ", 8, 1, 1},
    {"PZZ", 9, 1, 1}});
// The pressure quantities follow rho and J.
inline constexpr std::size_t first_pressure_quantity = 2;
inline constexpr std::size_t output_moment_component_count =
    output_moment_quantities.back().first_component +
    output_moment_quantities.back().components;
inline constexpr std::size_t output_pressure_first_component =
    output_moment_quantities[first_pressure_quantity].first_component;
inline constexpr std::uint16_t all_output_moments =
    (1U << output_moment_component_count) - 1U;
// Output-moment components in output_moment_quantities order.
struct OutputMomentComponent {
  static constexpr std::size_t rho = 0, jx = 1, jy = 2, jz = 3, pxx = 4, pxy = 5, pxz = 6,
      pyy = 7, pyz = 8, pzz = 9;
};

// Selection bit of each component, and the selections that read each velocity
// component.
struct OutputMomentBit {
  using C = OutputMomentComponent;
  static constexpr unsigned rho = 1U << C::rho, jx = 1U << C::jx, jy = 1U << C::jy,
      jz = 1U << C::jz, pxx = 1U << C::pxx, pxy = 1U << C::pxy, pxz = 1U << C::pxz,
      pyy = 1U << C::pyy, pyz = 1U << C::pyz, pzz = 1U << C::pzz;
  static constexpr unsigned uses_u = jx | pxx | pxy | pxz;
  static constexpr unsigned uses_v = jy | pxy | pyy | pyz;
  static constexpr unsigned uses_w = jz | pxz | pyz | pzz;
};
static_assert(OutputMomentBit::rho == output_moment_quantities[0].mask() &&
                  (OutputMomentBit::jx | OutputMomentBit::jy | OutputMomentBit::jz) ==
                      output_moment_quantities[1].mask() &&
                  OutputMomentBit::pxx == output_moment_quantities[2].mask() &&
                  OutputMomentBit::pzz == output_moment_quantities[7].mask() &&
                  OutputMomentComponent::pzz + 1 == output_moment_component_count &&
                  OutputMomentComponent::pxx == output_pressure_first_component,
              "output-moment bits follow output_moment_quantities");

} // namespace tpic::data

#endif // TPIC_DATA_OUTPUT_OUTPUT_MOMENT_QUANTITIES_HPP
