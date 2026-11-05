#ifndef TPIC_INITIALIZATION_CASES_HARRIS_SHEET_HPP
#define TPIC_INITIALIZATION_CASES_HARRIS_SHEET_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

namespace tpic::initialization {

// The magnetic seed of a sheet: a box-scale mode and a Gaussian hump of the
// out-of-plane potential. R1: the hump is stored in the units of its
// expression in harris_sheet.cpp, so the seed of GEM and DoubleGEM is the
// constants 0.4, delta and 10 of that expression.
struct HarrisSeed {
  double mode_amplitude{}, mode_kx{}, mode_ky{};  // fraction of HarrisB0x; 2*pi/Lx, 2*pi/Ly
  double amplitude_x{}, amplitude_y{};            // hump Bx and By, fractions of HarrisB0x
  double width_x{}, width_y{};                    // Gaussian widths
  double period_x{}, period_y{};                  // phase = pi*offset/period/delta
  double slope_period_x{}, slope_period_y{};      // cosine slope times width = pi/slope_period
  bool operator==(const HarrisSeed&) const = default;
};

// The z component of the curl of the single sheet with its seed, which
// currentFromAmpere turns into the drift of the sheet species.
struct HarrisCurl {
  double center_x{}, center_y{}, delta{};
  double sheet{};                       // -HarrisB0x/delta
  double mode{}, mode_kx{}, mode_ky{};  // pertGEM*HarrisB0x*(kx + (Lx/Ly)*ky)
  // The potential amplitude of the hump, its wavenumbers and 1/width^2.
  double hump{}, hump_kx{}, hump_ky{}, hump_inverse_wx2{}, hump_inverse_wy2{};

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE double at(double x, double y) const noexcept {
    const double xm = x - center_x;
    const double ym = y - center_y;
    const double ch = ::cosh(ym / delta);
    const double sx = xm * hump_inverse_wx2;
    const double sy = ym * hump_inverse_wy2;
    const double cos_x = ::cos(hump_kx * xm);
    const double sin_x = ::sin(hump_kx * xm);
    const double cos_y = ::cos(hump_ky * ym);
    const double sin_y = ::sin(hump_ky * ym);
    return sheet / (ch * ch) + mode * ::cos(mode_kx * xm) * ::cos(mode_ky * ym) -
        hump * ::exp(-xm * sx - ym * sy) *
            (cos_y * ((4.0 * sx * sx - 2.0 * hump_inverse_wx2 - hump_kx * hump_kx) * cos_x +
                      4.0 * sx * hump_kx * sin_x) +
             cos_x * ((4.0 * sy * sy - 2.0 * hump_inverse_wy2 - hump_ky * hump_ky) * cos_y +
                      4.0 * sy * hump_ky * sin_y));
  }
  bool operator==(const HarrisCurl&) const = default;
};

struct HarrisProfile {
  bool harris_density = false;
  bool is_double = false;
  double ly{}, delta{};
  // currentFromAmpere: w0*c/(rhoINIT*sum of sign(qom)*w0), zero for a species
  // without a share and without the option. The drift along z is then
  // ampere_drift*curl/cell_shape, with the shape that scales the charge, so
  // the particle carries its share of c*curl/(4*pi) at its own position.
  double ampere_drift{};
  bool varying_thermal = false;
  HarrisCurl curl;

  // Harris weights average the two y-node densities of the containing cell,
  // which cannot be replaced by evaluating the profile at the particle position.
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample& sample, const ParticleLoadPlan& plan,
                                 double x, double y, double, const int* cell) const noexcept {
    double cell_shape = 1.0;
    if (harris_density) {
      const double lower_y = plan.origin[1] + cell[1] * plan.spacing[1];
      const double nodes = harris_shape(lower_y) + harris_shape(lower_y + plan.spacing[1]);
      sample.density = sample.density * 0.5 * nodes;  // R1: (density*0.5)*nodes
      cell_shape = 0.5 * nodes;
    }
    if (is_double) {
      const double shape = -::tanh((y - 0.5 * ly) / delta);
      sample.drift[2] = sample.drift[2] * shape;
    }
    if (ampere_drift != 0.0) {
      const double drift = ampere_drift * curl.at(x, y) / cell_shape;
      if (varying_thermal) {
        // spatiallyVaryingThermal has no physical effect: the thermal speed
        // that gives the species Maxwellian its pressure rho*(drift^2 + wth^2)
        // is the deck wth to rounding. The sum rounds to an ulp of drift^2,
        // with or without a fused multiply-add: a difference within a few of
        // them carries no information, and the deck value stays.
        const double square = drift * drift;
        const double variance = (square + sample.thermal[2] * sample.thermal[2]) - square;
        if (variance > 4.0 * std::numeric_limits<double>::epsilon() * square) sample.thermal[2] = ::sqrt(variance);
      }
      sample.drift[2] = drift;
    }
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double, double, double, double&, double&,
                                          double&) const noexcept {}
  [[nodiscard]] ParticleSupport support() const noexcept { return {}; }
  bool operator==(const HarrisProfile&) const = default;

private:
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static double sech_squared(double value) noexcept {
    const double ch = ::cosh(value);
    return 1.0 / (ch * ch);
  }
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE double harris_shape(double y) const noexcept {
    if (is_double) {
      return sech_squared((y - 0.25 * ly) / delta) +
          sech_squared((y - 0.75 * ly) / delta);
    }
    return sech_squared((y - 0.5 * ly) / delta);
  }
};

// Harris sheets. GEMHarris takes its seed and its loading options from the
// deck; GEM, GEMnoPert and plasmoidChain are fixed seeds of the same sheet,
// DoubleGEM and DoubleGEMnoPert of a double sheet.
struct HarrisSheetCase {
  static constexpr std::array<std::string_view, 6> names{
      "GEMHarris", "GEM", "GEMnoPert", "DoubleGEM", "DoubleGEMnoPert", "plasmoidChain"};
  explicit HarrisSheetCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] std::string_view particle_label() const { return label; }
  using Profile = HarrisProfile;
  [[nodiscard]] Profile profile(std::size_t index) const noexcept { return species[index]; }
  bool operator==(const HarrisSheetCase&) const = default;

  bool is_double = false;
  HarrisSeed seed;
  std::string_view label;
  std::vector<Profile> species;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_HARRIS_SHEET_HPP
