#include <tpic/initialization/cases/initial_case.hpp>

#include <cmath>

namespace tpic::initialization {

using config::SimulationParameters;

namespace {

constexpr double pi = 3.1415926535897932384626433832795;

[[nodiscard]] bool has_hump(const HarrisSeed& seed) noexcept {
  return seed.amplitude_x != 0.0 || seed.amplitude_y != 0.0;
}

// The seed the GEMHarris keys describe. The hump potential is
// (pertHump*HarrisB0x/ky) * exp(-x^2/wx^2 - y^2/wy^2) * cos(kx*x) * cos(ky*y)
// with widths deltaxHump*delta and deltayHump*delta; a negative kxHump or
// kyHump selects 2*pi/Lx or pi/Ly. kxHump = 0 gives infinite periods, a zero
// phase and a zero slope.
[[nodiscard]] HarrisSeed deck_seed(const SimulationParameters& params) {
  require_finite(params.pert_gem, "pertGEM");
  require_finite(params.pert_hump, "pertHump");
  require_finite(params.kx_hump, "kxHump");
  require_finite(params.ky_hump, "kyHump");
  require_recipe(std::isfinite(params.deltax_hump) && params.deltax_hump > 0.0,
                 "GEMHarris deltaxHump must be finite and positive");
  require_recipe(std::isfinite(params.deltay_hump) && params.deltay_hump > 0.0,
                 "GEMHarris deltayHump must be finite and positive");
  HarrisSeed seed{params.pert_gem, mode_wavenumber(1, params.grid.lx), mode_wavenumber(1, params.grid.ly)};
  if (params.pert_hump == 0.0) return seed;
  const double kx = params.kx_hump < 0.0 ? two_pi / params.grid.lx : params.kx_hump;
  const double ky = params.ky_hump < 0.0 ? .5 * two_pi / params.grid.ly : params.ky_hump;
  require_recipe(ky != 0.0, "GEMHarris kyHump must be nonzero: the hump amplitude is pertHump*HarrisB0x/kyHump");
  seed.width_x = checked_product(params.deltax_hump, params.delta, "GEMHarris hump x width");
  seed.width_y = checked_product(params.deltay_hump, params.delta, "GEMHarris hump y width");
  seed.amplitude_x = checked_quotient(params.pert_hump, checked_product(ky, seed.width_y, "GEMHarris hump y slope"),
                                      "GEMHarris hump Bx amplitude");
  seed.amplitude_y = checked_quotient(params.pert_hump, checked_product(ky, seed.width_x, "GEMHarris hump x slope"),
                                      "GEMHarris hump By amplitude");
  seed.period_x = pi / (kx * params.delta);
  seed.period_y = representable(pi / (ky * params.delta), "GEMHarris hump y period");
  seed.slope_period_x = pi / (kx * seed.width_x);
  seed.slope_period_y = pi / (ky * seed.width_y);
  return seed;
}

void validate_harris_coefficients(const SimulationParameters& params, const HarrisSheetCase& sheet) {
  const HarrisSeed& seed = sheet.seed;
  double bx_bound = std::abs(params.harris_b0x);
  double by_bound = 0.0;
  double extent_x = 0.0;
  double extent_y = 0.0;
  if (has_hump(seed) || seed.mode_amplitude != 0.0) {
    const double center_fraction = sheet.is_double ? .25 : .5;
    // The two symmetric centers have identical global displacement bounds.
    extent_x = coordinate_extent(params.grid.lx, params.grid.nxc, center_fraction * params.grid.lx);
    extent_y = coordinate_extent(params.grid.ly, params.grid.nyc, center_fraction * params.grid.ly);
  }
  if (has_hump(seed)) {
    const double axes[2][4] = {{extent_x, seed.period_x, seed.width_x, seed.slope_period_x},
                               {extent_y, seed.period_y, seed.width_y, seed.slope_period_y}};
    for (const auto& [extent, period, width, slope_period] : axes) {
      // The hump evaluates pi*offset/period/delta and cos_x*cos_y*2*offset/width.
      // exp(-r*r) may safely become zero, but an infinite trig argument or
      // gradient would then produce NaN.
      representable((checked_product(.5 * two_pi, extent, "Harris phase numerator") / period) / params.delta,
                  "Harris perturbation phase bound");
      const double gradient = representable(checked_product(2.0, extent, "Harris gradient numerator") / width,
                                          "Harris perturbation gradient bound");
      representable(gradient + .5 * two_pi / slope_period, "Harris perturbation gradient sum");
    }
    // 2*|r|*exp(-r*r) < 1, and all trigonometric factors have magnitude <=1.
    // The bounds deliberately allow both humps of a double sheet to add in magnitude.
    const double sheets = sheet.is_double ? 2.0 : 1.0;
    bx_bound = representable(bx_bound + sheets * checked_product(
        checked_product(std::abs(params.harris_b0x), std::abs(seed.amplitude_x), "Harris perturbation amplitude"),
        1.0 + .5 * two_pi / seed.slope_period_y, "Harris perturbation field bound"),
        "Harris total sheet field bound");
    by_bound = representable(sheets * checked_product(
        checked_product(std::abs(params.harris_b0x), std::abs(seed.amplitude_y), "Harris perturbation amplitude"),
        1.0 + .5 * two_pi / seed.slope_period_x, "Harris perturbation field bound"),
        "Harris perturbation field bound");
  }
  if (seed.mode_amplitude != 0.0) {
    checked_product(seed.mode_kx, extent_x, "Harris mode phase bound");
    checked_product(seed.mode_ky, extent_y, "Harris mode phase bound");
    const double amplitude = checked_product(std::abs(params.harris_b0x), std::abs(seed.mode_amplitude),
                                             "Harris mode amplitude");
    bx_bound = representable(bx_bound + checked_product(amplitude, params.grid.lx / params.grid.ly,
                                                        "Harris mode field bound"),
                             "Harris total sheet field bound");
    by_bound = representable(by_bound + amplitude, "Harris mode field bound");
  }
  representable(bx_bound + std::abs(params.b0x), "Harris Bx plus background bound");
  representable(by_bound + std::abs(params.b0y), "Harris By plus background bound");
}

// R1: the operation order of every statement is part of the initial state of
// GEM and DoubleGEM, whose seed holds the constants 0.4, delta and 10.
void add_hump(const HarrisSeed& seed, double x, double y, double center_x, double center_y,
              double delta, double sheet_b0x, double sign, FieldVector& b) {
  const double xpert = x - center_x;
  const double ypert = y - center_y;
  const double envelope = std::exp(-(xpert / seed.width_x) * (xpert / seed.width_x)
                                    -(ypert / seed.width_y) * (ypert / seed.width_y));
  const double cos_x = std::cos(pi * xpert / seed.period_x / delta);
  const double sin_x = std::sin(pi * xpert / seed.period_x / delta);
  const double cos_y = std::cos(pi * ypert / seed.period_y / delta);
  const double sin_y = std::sin(pi * ypert / seed.period_y / delta);
  b[0] += sign * sheet_b0x * seed.amplitude_x * envelope *
      (-cos_x * cos_y * 2.0 * ypert / seed.width_y - cos_x * sin_y * pi / seed.slope_period_y);
  b[1] += sign * sheet_b0x * seed.amplitude_y * envelope *
      (cos_x * cos_y * 2.0 * xpert / seed.width_x + sin_x * cos_y * pi / seed.slope_period_x);
}

// currentFromAmpere of GEMHarris: the share of every species in the curl of
// the initial field, and the refusal of a drift above c. The drift divides the
// curl by the sheet shape, so it diverges where a seed outlasts the sheet.
void admit_ampere_current(const SimulationParameters& params, HarrisSheetCase& sheet) {
  double drift_sum = 0.0;
  for (const auto& loaded : params.species) {
    require_recipe(loaded.u0 == 0.0 && loaded.v0 == 0.0,
                   "GEMHarris currentFromAmpere requires u0 = v0 = 0: the curl of the initial field is along z");
    drift_sum += (loaded.q_over_m > 0.0 ? 1.0 : -1.0) * loaded.w0;
  }
  require_recipe(std::isfinite(drift_sum) && drift_sum != 0.0,
                 "GEMHarris currentFromAmpere requires a nonzero sum of sign(qom)*w0");
  const HarrisSeed& seed = sheet.seed;
  const double b0 = params.harris_b0x;
  HarrisCurl curl;
  curl.center_x = 0.5 * params.grid.lx;
  curl.center_y = 0.5 * params.grid.ly;
  curl.delta = params.delta;
  curl.sheet = checked_quotient(-b0, params.delta, "GEMHarris sheet current");
  curl.mode = checked_product(checked_product(seed.mode_amplitude, b0, "GEMHarris mode current"),
                              seed.mode_kx + (params.grid.lx / params.grid.ly) * seed.mode_ky,
                              "GEMHarris mode current");
  curl.mode_kx = seed.mode_kx;
  curl.mode_ky = seed.mode_ky;
  if (has_hump(seed)) {
    // The potential amplitude pertHump*HarrisB0x/ky and the wavenumbers, from the seed.
    curl.hump = checked_product(checked_product(b0, seed.amplitude_x, "GEMHarris hump potential"), seed.width_y,
                                "GEMHarris hump potential");
    curl.hump_kx = pi / (seed.period_x * params.delta);
    curl.hump_ky = pi / (seed.period_y * params.delta);
    curl.hump_inverse_wx2 = representable(1.0 / (seed.width_x * seed.width_x), "GEMHarris hump x width");
    curl.hump_inverse_wy2 = representable(1.0 / (seed.width_y * seed.width_y), "GEMHarris hump y width");
  }

  // |curl| over x at one y: |cos| <= 1, |r|*exp(-r*r) <= 1/sqrt(2e), and the
  // x factor of the hump is at most 2/wx^2 + kx^2 + 4*|kx|/(wx*sqrt(2e)).
  constexpr double sqrt_2e = 2.3316439815971242;
  const double x_bound = (2.0 * curl.hump_inverse_wx2 + curl.hump_kx * curl.hump_kx) +
      4.0 * std::abs(curl.hump_kx) * std::sqrt(curl.hump_inverse_wx2) / sqrt_2e;
  const auto sech_squared = [](double value) {
    const double ch = std::cosh(value);
    return 1.0 / (ch * ch);
  };
  const double dy = params.grid.ly / params.grid.nyc;
  for (std::size_t index = 0; index < sheet.species.size(); ++index) {
    const auto& loaded = params.species[index];
    auto& profile = sheet.species[index];
    if (loaded.w0 == 0.0) continue;
    profile.ampere_drift = checked_quotient(checked_product(loaded.w0, params.c, "GEMHarris Ampere drift"),
                                            checked_product(drift_sum, loaded.rho_init, "GEMHarris Ampere drift"),
                                            "GEMHarris Ampere drift coefficient");
    profile.varying_thermal = params.spatially_varying_thermal;
    profile.curl = curl;
    // Every row of particles of the species, with the cell shape of evaluate.
    for (int j = 0; j < params.grid.nyc; ++j) {
      const double cell_shape = profile.harris_density
          ? 0.5 * (sech_squared((j * dy - curl.center_y) / curl.delta) +
                   sech_squared((j * dy + dy - curl.center_y) / curl.delta))
          : 1.0;
      for (int row = 0; row < loaded.npcely; ++row) {
        const double ym = (j + (row + 0.5) / loaded.npcely) * dy - curl.center_y;
        const double sy = ym * curl.hump_inverse_wy2;
        const double cos_y = std::cos(curl.hump_ky * ym);
        const double bound = std::abs(curl.sheet) * sech_squared(ym / curl.delta) +
            std::abs(curl.mode * std::cos(curl.mode_ky * ym)) +
            std::abs(curl.hump) * std::exp(-ym * sy) *
                (std::abs(cos_y) * x_bound +
                 std::abs((4.0 * sy * sy - 2.0 * curl.hump_inverse_wy2 - curl.hump_ky * curl.hump_ky) * cos_y +
                          4.0 * sy * curl.hump_ky * std::sin(curl.hump_ky * ym)));
        require_recipe(std::abs(profile.ampere_drift) * bound / cell_shape <= params.c,
                       "GEMHarris currentFromAmpere drift exceeds c where the sheet density vanishes");
      }
    }
  }
}

} // namespace

HarrisSheetCase::HarrisSheetCase(const SimulationParameters& params)
    : is_double(params.case_name == "DoubleGEM" || params.case_name == "DoubleGEMnoPert"),
      label(params.case_name == "plasmoidChain" ? "plasmoidChain" : is_double ? "DoubleGEM" : "GEM/Harris") {
  require_finite(params.harris_b0x, "HarrisB0x");
  require_recipe(std::isfinite(params.delta) && params.delta > 0.0, "Harris delta must be finite and positive");
  if (params.case_name == "GEMHarris") {
    seed = deck_seed(params);
  } else if (params.case_name == "GEM" || params.case_name == "DoubleGEM") {
    seed = HarrisSeed{0.0, 0.0, 0.0, 0.4, 0.4, params.delta, params.delta, 10.0, 10.0, 10.0, 10.0};
  }
  validate_harris_coefficients(params, *this);
  bool electron_seen = false;
  bool ion_seen = false;
  for (const auto& loaded : params.species) {
    bool harris_density;
    if (params.case_name == "plasmoidChain") {
      bool& seen = loaded.q_over_m < 0.0 ? electron_seen : ion_seen;
      harris_density = !seen;
      seen = true;
    } else {
      harris_density = std::abs(loaded.u0) > 1.0e-10 ||
          std::abs(loaded.v0) > 1.0e-10 || std::abs(loaded.w0) > 1.0e-10;
    }
    species.push_back({harris_density, is_double, params.grid.ly, params.delta});
  }
  if (params.case_name == "GEMHarris" && params.current_from_ampere) {
    admit_ampere_current(params, *this);
    label = "GEM/Harris, Ampere current";
  }
}

FieldVector HarrisSheetCase::magnetic(const SimulationParameters& p, double x, double y, double) const {
  FieldVector b{};
  if (is_double) {
    const double bottom = (y - 0.25 * p.grid.ly) / p.delta;
    const double top = (y - 0.75 * p.grid.ly) / p.delta;
    b[0] = p.harris_b0x * (-1.0 + std::tanh(bottom) - std::tanh(top));
  } else {
    b[0] = p.harris_b0x * std::tanh((y - 0.5 * p.grid.ly) / p.delta);
  }
  if (seed.mode_amplitude != 0.0) {
    const double xm = x - 0.5 * p.grid.lx;
    const double ym = y - 0.5 * p.grid.ly;
    b[0] += -seed.mode_amplitude * p.harris_b0x * (p.grid.lx / p.grid.ly) *
        std::cos(seed.mode_kx * xm) * std::sin(seed.mode_ky * ym);
    b[1] += seed.mode_amplitude * p.harris_b0x * std::sin(seed.mode_kx * xm) * std::cos(seed.mode_ky * ym);
  }
  if (has_hump(seed)) {
    if (is_double) {
      add_hump(seed, x, y, 0.25 * p.grid.lx, 0.25 * p.grid.ly,
               p.delta, p.harris_b0x, 1.0, b);
      add_hump(seed, x, y, 0.75 * p.grid.lx, 0.75 * p.grid.ly,
               p.delta, p.harris_b0x, -1.0, b);
    } else {
      add_hump(seed, x, y, 0.5 * p.grid.lx, 0.5 * p.grid.ly,
               p.delta, p.harris_b0x, 1.0, b);
    }
  }
  return b;
}

} // namespace tpic::initialization
