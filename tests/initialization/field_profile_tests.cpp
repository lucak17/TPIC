#include <tpic/config/domain.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/initialization/cases/initial_case.hpp>
#include <tpic/initialization/fields/field_image.hpp>

#include "../support/golden_data.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace golden = tpic::test::golden;
using tpic::config::SimulationParameters;
using tpic::initialization::FieldVector;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// The analytic fields of one parameter set and the case prepared from it.
struct Profile {
  SimulationParameters parameters;
  tpic::initialization::InitialCase initial_case;

  FieldVector magnetic(double x, double y, double z) const {
    return tpic::initialization::initial_magnetic(parameters, initial_case, x, y, z);
  }
  FieldVector electric(double x, double y, double z) const {
    return tpic::initialization::initial_electric(parameters, initial_case, x, y, z);
  }
};

Profile profile_for(const SimulationParameters& parameters) {
  return {parameters, tpic::initialization::prepare_initial_case(parameters)};
}

template <class Function>
void overflow(Function function, const char* message) {
  try {
    function();
  } catch (const std::overflow_error& error) {
    require(std::string(error.what()).starts_with("TPIC field profile:"),
            "field sampling error is missing context");
    return;
  }
  throw std::runtime_error(message);
}

SimulationParameters base(std::string name) {
  SimulationParameters parameters;
  parameters.case_name = std::move(name);
  parameters.ns = 1;
  parameters.species.resize(1);
  parameters.species[0].q_over_m = -1.0;
  parameters.species[0].rho_init = 1.0;
  parameters.b0x = .17;
  parameters.b0y = -.23;
  parameters.b0z = .31;
  return parameters;
}

void coordinate_validation() {
  const auto profile = profile_for(base("Uniform"));
  for (const double bad : {std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      FieldVector point{1.0, 2.0, 3.0};
      point[axis] = bad;
      overflow([&] { (void)profile.magnetic(point[0], point[1], point[2]); },
               "magnetic sampler accepted nonfinite coordinates");
      overflow([&] { (void)profile.electric(point[0], point[1], point[2]); },
               "electric sampler accepted nonfinite coordinates");
    }
  }
  const double large = std::numeric_limits<double>::max();
  require(profile.magnetic(-large, large, large) == FieldVector{.17, -.23, .31},
          "finite coordinates were clipped or restricted to the interior domain");
}

void sampled_overflow() {
  auto parameters = base("Weibel");
  parameters.weibel_seed_mode_x = 3;
  parameters.weibel_seed_amplitude = .1;
  parameters.weibel_seed_growth_rate = .2;
  const auto wave = profile_for(parameters);
  const double large = std::numeric_limits<double>::max();
  overflow([&] { (void)wave.magnetic(large, 0.0, 0.0); },
           "magnetic sampler accepted a nonfinite sampled component");
  overflow([&] { (void)wave.electric(large, 0.0, 0.0); },
           "electric sampler accepted a nonfinite sampled component");

  parameters = base("ForceFree");
  parameters.forcefree_b0x = 1.0e308;
  parameters.b0x = 1.0e308;
  const auto sheet = profile_for(parameters);
  overflow([&] { (void)sheet.magnetic(0.0, parameters.grid.ly, 0.0); },
           "sampler accepted overflow when adding the uniform background");
}

// Golden fixtures: every case name and parameter variant on a
// deliberately nondivisible grid. TGV wavelengths are equal in x and y
// (9.5/2 = 14.25/3), so its seeded field is divergence-free.
SimulationParameters fixture_parameters(std::string_view name) {
  SimulationParameters parameters;
  parameters.case_name = name;
  parameters.dt = .07;
  parameters.c = 1.7;
  parameters.grid.lx = 9.5;
  parameters.grid.ly = 14.25;
  parameters.grid.lz = 5.5;
  parameters.grid.nxc = 17;
  parameters.grid.nyc = 13;
  parameters.grid.nzc = 9;
  parameters.b0x = .19;
  parameters.b0y = -.23;
  parameters.b0z = .37;
  parameters.delta = 1.3;
  parameters.harris_b0x = -.47;
  parameters.forcefree_b0x = .53;
  parameters.forcefree_b0z = -.61;
  parameters.weibel_seed_mode_x = 3;
  parameters.weibel_seed_amplitude = -.043;
  parameters.weibel_seed_growth_rate = .71;
  parameters.two_stream_seed_mode_x = 2;
  parameters.two_stream_seed_density_amplitude = -.13;
  parameters.two_stream_seed_phase = .29;
  parameters.whistler_mode_x = 2;
  parameters.whistler_seed_epsilon = .037;
  parameters.whistler_phase = .41;
  parameters.whistler_field_time_offset = .23;
  parameters.whistler_particle_time_offset = -.31;
  parameters.taylor_green_vortex_mode_x = 2;
  parameters.taylor_green_vortex_mode_y = 3;
  parameters.taylor_green_vortex_mode_z = 4;
  parameters.taylor_green_vortex_velocity_amplitude = -.43;
  parameters.taylor_green_vortex_magnetic_amplitude = .17;
  parameters.taylor_green_vortex_seed_motional_e = true;
  parameters.flux_rope_b0 = -.47;
  parameters.flux_rope_width = .83;
  parameters.flux_rope_center_x = 3.17;
  parameters.flux_rope_center_y = 5.91;
  parameters.flux_rope_pressure_floor = .007;
  parameters.flux_rope_electron_pressure_fraction = .31;
  // Node 8 of the 17 cells along x.
  parameters.gaussian_expansion_center_x = 8 * (9.5 / 17);
  parameters.gaussian_expansion_sigma_x = .9;
  parameters.whistler_turbulence_epsilon = .07;
  parameters.whistler_turbulence_seed = 0xfedcba9876543210ULL;
  parameters.ns = 3;
  parameters.species.resize(3);
  for (std::size_t species = 0; species < 3; ++species) {
    auto& values = parameters.species[species];
    values.q_over_m = species == 2 ? 1.0 : -64.0;
    values.rho_init = species == 2 ? 1.2 : .6;
    values.uth = .13;
    values.vth = .17;
    values.wth = .19;
    values.u0 = species == 0 ? .11 : species == 1 ? -.11 : 0.0;
    values.npcelx = 2;
    values.npcely = 3;
    values.npcelz = 4;
  }
  if (name == "Whistler") {
    parameters.b0y = 0.0;
    parameters.b0z = 0.0;
  }
  // Pair loading requires the subcell centres.
  parameters.quiet_start = name == "GaussianExpansion" || name == "WhistlerTurbulence";
  if (name == "GaussianExpansion")
    for (auto& species : parameters.species) species.u0 = 0.0;
  if (name == "WhistlerTurbulence") {
    parameters.b0x = 0.0;
    parameters.b0y = 0.0;
  }
  return parameters;
}

struct Fixture {
  std::string label;
  SimulationParameters parameters;
};

using Modify = std::function<void(SimulationParameters&)>;

void add(std::vector<Fixture>& fixtures, std::string_view name, std::string label,
         const Modify& modify = {}) {
  auto parameters = fixture_parameters(name);
  if (modify) modify(parameters);
  fixtures.push_back({std::move(label), std::move(parameters)});
}

std::vector<Fixture> fixtures() {
  using Mode = tpic::config::WhistlerSeedMode;
  std::vector<Fixture> result;
  // Every case name, in the order of the golden table.
  constexpr std::array<std::string_view, 15> names{
      "Weibel", "TwoStream", "Whistler", "TaylorGreenVortex", "ForceFree", "fluxRope",
      "plasmoidChain", "DoubleGEM", "DoubleGEMnoPert", "GEM", "GEMnoPert", "Uniform", "GEMHarris",
      "GaussianExpansion", "WhistlerTurbulence"};
  static_assert(names.size() == tpic::config::case_names.size());
  for (const auto name : names) add(result, name, std::string(name));
  add(result, "Whistler", "Whistler approximate root",
      [](auto& p) { p.whistler_use_exact_cold_root = false; });
  add(result, "Whistler", "Whistler zero eigenmode amplitude",
      [](auto& p) { p.whistler_seed_epsilon = 0.0; });
  add(result, "Whistler", "Whistler tolerated transverse background",
      [](auto& p) { p.b0y = 1.0e-11; p.b0z = -1.0e-11; });
  const auto whistler = fixture_parameters("Whistler");
  const double omega = std::get<tpic::initialization::WhistlerCase>(
      tpic::initialization::prepare_initial_case(whistler)).omega;
  add(result, "Whistler", "Whistler large cancelling phase", [=](auto& p) {
    p.whistler_field_time_offset = 1.0e20;
    p.whistler_phase = omega * p.whistler_field_time_offset;
  });
  add(result, "Whistler", "Whistler large noise phase", [](auto& p) {
    p.whistler_seed_mode = Mode::Noise;
    p.whistler_noise_mode_max = 7;
    p.whistler_noise_seed = 0xfedcba9876543210ULL;
    p.whistler_phase = 1.0e16;
  });
  for (const double epsilon : {0.0, .07}) {
    add(result, "Whistler", "Whistler magnetic-only epsilon=" + std::to_string(epsilon),
        [=](auto& p) {
      p.whistler_seed_mode = Mode::MagneticOnly;
      p.whistler_seed_epsilon = epsilon;
      p.whistler_noise_mode_max = -1;
      p.whistler_field_time_offset = std::numeric_limits<double>::quiet_NaN();
      p.whistler_particle_time_offset = std::numeric_limits<double>::infinity();
      p.ns = 1;
      p.species.resize(1);
    });
  }
  for (const std::uint64_t seed : {0ULL, 12345ULL, 0xfedcba9876543210ULL}) {
    for (const int limit : {1, 7, 0, -1}) {
      add(result, "Whistler",
          "Whistler noise seed=" + std::to_string(seed) + " limit=" + std::to_string(limit),
          [=](auto& p) {
        p.whistler_seed_mode = Mode::Noise;
        p.whistler_noise_seed = seed;
        p.whistler_noise_mode_max = std::max(0, limit);
        p.whistler_mode_x = limit == -1 ? 0 : 3;
        p.whistler_field_time_offset = std::numeric_limits<double>::quiet_NaN();
        p.whistler_particle_time_offset = std::numeric_limits<double>::infinity();
        p.ns = 1;
        p.species.resize(1);
      });
    }
  }
  add(result, "Whistler", "Whistler zero noise amplitude", [](auto& p) {
    p.whistler_seed_mode = Mode::Noise;
    p.whistler_seed_epsilon = 0.0;
    p.whistler_mode_x = 0;
  });
  for (const double growth : {0.0, -.31})
    add(result, "Weibel", "Weibel growth=" + std::to_string(growth),
        [=](auto& p) { p.weibel_seed_growth_rate = growth; });
  add(result, "Weibel", "Weibel no seed",
      [](auto& p) { p.weibel_seed_amplitude = 0.0; p.weibel_seed_mode_x = 0; });
  add(result, "TwoStream", "TwoStream no seed", [](auto& p) {
    p.two_stream_seed_density_amplitude = 0.0;
    p.two_stream_seed_mode_x = 0;
  });
  add(result, "TwoStream", "TwoStream reordered species",
      [](auto& p) { std::swap(p.species[0], p.species[2]); });
  add(result, "TaylorGreenVortex", "TGV no motional E",
      [](auto& p) { p.taylor_green_vortex_seed_motional_e = false; });
  add(result, "TaylorGreenVortex", "TGV flow only",
      [](auto& p) { p.taylor_green_vortex_magnetic_amplitude = 0.0; });
  add(result, "TaylorGreenVortex", "TGV magnetic only",
      [](auto& p) { p.taylor_green_vortex_velocity_amplitude = 0.0; });
  add(result, "TaylorGreenVortex", "TGV no seed", [](auto& p) {
    p.taylor_green_vortex_velocity_amplitude = 0.0;
    p.taylor_green_vortex_magnetic_amplitude = 0.0;
    p.taylor_green_vortex_mode_x = p.taylor_green_vortex_mode_y = p.taylor_green_vortex_mode_z = 0;
  });
  add(result, "GEM", "Harris no sheet field", [](auto& p) { p.harris_b0x = 0.0; });
  add(result, "DoubleGEM", "Double Harris positive amplitude", [](auto& p) { p.harris_b0x = .71; });
  add(result, "ForceFree", "ForceFree no sheet field",
      [](auto& p) { p.forcefree_b0x = p.forcefree_b0z = 0.0; });
  add(result, "fluxRope", "Flux rope zero amplitude", [](auto& p) { p.flux_rope_b0 = 0.0; });
  add(result, "Uniform", "Uniform smaller transverse axes",
      [](auto& p) { p.grid.nyc = 5; p.grid.nzc = 5; });
  add(result, "GEMHarris", "GEMHarris mode only", [](auto& p) { p.pert_gem = .09; });
  add(result, "GEMHarris", "GEMHarris hump only",
      [](auto& p) { p.pert_hump = .11; p.deltax_hump = 2.5; p.deltay_hump = 1.5; });
  add(result, "GEMHarris", "GEMHarris hump kx zero",
      [](auto& p) { p.pert_hump = .11; p.kx_hump = 0.0; p.ky_hump = .9; });
  // The keys that describe the X-point seed of GEM.
  add(result, "GEMHarris", "GEMHarris GEM keys", [](auto& p) {
    constexpr double pi = 3.14159265358979323846;
    p.pert_hump = .04 * pi;
    p.deltax_hump = p.deltay_hump = 1.0;
    p.kx_hump = p.ky_hump = pi / (10.0 * p.delta);
  });
  return result;
}

// Exact perturbation and rope centres and nearby points, which need not
// coincide with a node on these grids.
std::array<FieldVector, 6> sample_points(const SimulationParameters& p) {
  return {{{0.0, 0.0, 0.0},
           {p.grid.lx * .5, p.grid.ly * .5, p.grid.lz * .5},
           {p.grid.lx * .25 + .37, p.grid.ly * .25 - .21, .17},
           {p.grid.lx * .75 - .31, p.grid.ly * .75 + .19, .71},
           {p.flux_rope_center_x, p.flux_rope_center_y, .23},
           {p.grid.lx * .5 + .13, p.grid.ly * .5 - .29, -.11}}};
}

std::string sample_line(const std::string& label, std::size_t point, char field,
                        const FieldVector& value) {
  return label + '\t' + std::to_string(point) + '\t' + field + '\t' + golden::hex(value[0]) +
         '\t' + golden::hex(value[1]) + '\t' + golden::hex(value[2]);
}

golden::Lines golden_samples(const std::vector<Fixture>& cases) {
  golden::Lines lines;
  for (const auto& fixture : cases) {
    const auto profile = profile_for(fixture.parameters);
    const auto points = sample_points(fixture.parameters);
    for (std::size_t index = 0; index < points.size(); ++index) {
      const auto& x = points[index];
      lines.push_back(sample_line(fixture.label, index, 'B', profile.magnetic(x[0], x[1], x[2])));
      lines.push_back(sample_line(fixture.label, index, 'E', profile.electric(x[0], x[1], x[2])));
    }
  }
  return lines;
}

// Golden values compare to roundoff of the precomputed coefficients.
bool sample_equal(const std::string& actual, const std::string& expected) {
  const auto a = golden::split(actual, '\t');
  const auto b = golden::split(expected, '\t');
  if (a.size() != 6 || b.size() != 6 || a[0] != b[0] || a[1] != b[1] || a[2] != b[2])
    return false;
  for (std::size_t column = 3; column < 6; ++column) {
    const double x = std::strtod(a[column].c_str(), nullptr);
    const double y = std::strtod(b[column].c_str(), nullptr);
    if (!std::isfinite(x) || !std::isfinite(y) ||
        std::abs(x - y) > 3.0e-14 * std::max({1.0, std::abs(x), std::abs(y)}))
      return false;
  }
  return true;
}

const Fixture& find(const std::vector<Fixture>& cases, std::string_view label) {
  for (const auto& fixture : cases)
    if (fixture.label == label) return fixture;
  throw std::runtime_error("missing field fixture " + std::string(label));
}

double close_tolerance(const FieldVector& a, const FieldVector& b, double scale) {
  double difference = 0.0, magnitude = 1.0;
  for (std::size_t c = 0; c < 3; ++c) {
    difference = std::max(difference, std::abs(a[c] - b[c]));
    magnitude = std::max({magnitude, std::abs(a[c]), std::abs(b[c])});
  }
  return difference / (scale * magnitude);
}

// div B by fourth-order central differences of the analytic profile.
void divergence_free(const std::vector<Fixture>& cases) {
  for (const auto* label : {"GEM", "DoubleGEM", "ForceFree", "fluxRope", "TaylorGreenVortex",
                            "GEMHarris mode only", "GEMHarris hump only", "GEMHarris hump kx zero"}) {
    const auto& fixture = find(cases, label);
    const auto profile = profile_for(fixture.parameters);
    constexpr double h = 1.0e-3;
    for (const auto& x : sample_points(fixture.parameters)) {
      double divergence = 0.0;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto at = [&](double offset) {
          auto y = x;
          y[axis] += offset;
          return profile.magnetic(y[0], y[1], y[2])[axis];
        };
        divergence += (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h);
      }
      require(std::abs(divergence) < 1.0e-9,
              std::string(label) + ": magnetic field is not divergence-free");
    }
  }
}

// The X-point seed of GEM and DoubleGEM, written with its constants: the
// statements and their operation order are part of the initial state.
void add_xpoint(const SimulationParameters& p, double x, double y, double center_x, double center_y,
                double sign, FieldVector& b) {
  constexpr double pi = 3.1415926535897932384626433832795;
  const double xpert = x - center_x;
  const double ypert = y - center_y;
  const double envelope = std::exp(-(xpert / p.delta) * (xpert / p.delta)
                                    -(ypert / p.delta) * (ypert / p.delta));
  const double cos_x = std::cos(pi * xpert / 10.0 / p.delta);
  const double sin_x = std::sin(pi * xpert / 10.0 / p.delta);
  const double cos_y = std::cos(pi * ypert / 10.0 / p.delta);
  const double sin_y = std::sin(pi * ypert / 10.0 / p.delta);
  const double coefficient = sign * p.harris_b0x * 0.4;
  b[0] += coefficient * envelope *
      (-cos_x * cos_y * 2.0 * ypert / p.delta - cos_x * sin_y * pi / 10.0);
  b[1] += coefficient * envelope *
      (cos_x * cos_y * 2.0 * xpert / p.delta + sin_x * cos_y * pi / 10.0);
}

// The Harris names with a fixed seed against that expression, the unseeded
// GEMHarris against GEMnoPert, and the GEMHarris keys of the X-point seed.
void harris_seeds(const std::vector<Fixture>& cases) {
  for (const auto* label : {"GEM", "Harris no sheet field", "DoubleGEM", "Double Harris positive amplitude"}) {
    const auto& p = find(cases, label).parameters;
    const auto profile = profile_for(p);
    for (const auto& x : sample_points(p)) {
      FieldVector expected{};
      if (p.case_name == "DoubleGEM") {
        expected[0] = p.harris_b0x * (-1.0 + std::tanh((x[1] - 0.25 * p.grid.ly) / p.delta) -
                                      std::tanh((x[1] - 0.75 * p.grid.ly) / p.delta));
        add_xpoint(p, x[0], x[1], 0.25 * p.grid.lx, 0.25 * p.grid.ly, 1.0, expected);
        add_xpoint(p, x[0], x[1], 0.75 * p.grid.lx, 0.75 * p.grid.ly, -1.0, expected);
      } else {
        expected[0] = p.harris_b0x * std::tanh((x[1] - 0.5 * p.grid.ly) / p.delta);
        add_xpoint(p, x[0], x[1], 0.5 * p.grid.lx, 0.5 * p.grid.ly, 1.0, expected);
      }
      expected[0] += p.b0x;
      expected[1] += p.b0y;
      expected[2] += p.b0z;
      const auto actual = profile.magnetic(x[0], x[1], x[2]);
#ifdef FP_FAST_FMA
      // A compiler that fuses multiply-add may contract the two forms differently.
      require(close_tolerance(actual, expected, 3.0e-14) <= 1.0,
              std::string(label) + ": seeded Harris field differs from the X-point expression");
#else
      require(actual == expected,
              std::string(label) + ": seeded Harris field is not the X-point expression bit for bit");
#endif
    }
  }
  const auto& unseeded = find(cases, "GEMHarris").parameters;
  const auto& keyed = find(cases, "GEMHarris GEM keys").parameters;
  const auto plain = profile_for(find(cases, "GEMnoPert").parameters);
  const auto seeded = profile_for(find(cases, "GEM").parameters);
  for (const auto& x : sample_points(unseeded)) {
    require(profile_for(unseeded).magnetic(x[0], x[1], x[2]) == plain.magnetic(x[0], x[1], x[2]),
            "GEMHarris without a seed is not the unseeded sheet");
    require(close_tolerance(profile_for(keyed).magnetic(x[0], x[1], x[2]), seeded.magnetic(x[0], x[1], x[2]),
                            1.0e-13) <= 1.0,
            "GEMHarris with the X-point keys differs from GEM");
  }
}

// Seeded modes repeat over one domain length along each seeded axis.
void periodic_modes(const std::vector<Fixture>& cases) {
  for (const auto* label : {"Weibel", "TwoStream", "TaylorGreenVortex", "Whistler",
                            "Whistler noise seed=12345 limit=7"}) {
    const auto& fixture = find(cases, label);
    const auto profile = profile_for(fixture.parameters);
    const auto& grid = fixture.parameters.grid;
    const std::array<double, 3> lengths{grid.lx, grid.ly, grid.lz};
    for (const auto& x : sample_points(fixture.parameters)) {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        auto y = x;
        y[axis] += lengths[axis];
        require(close_tolerance(profile.magnetic(x[0], x[1], x[2]),
                                profile.magnetic(y[0], y[1], y[2]), 1.0e-11) <= 1.0 &&
                close_tolerance(profile.electric(x[0], x[1], x[2]),
                                profile.electric(y[0], y[1], y[2]), 1.0e-11) <= 1.0,
                std::string(label) + ": seeded mode is not periodic");
      }
    }
  }
}

// The exact root satisfies the cold-plasma whistler dispersion relation
// c^2 k^2 / w^2 = 1 - wpe^2 / (w (w - Wce)) - wpi^2 / (w (w + Wci)) in units
// with wpi = 1, wpe^2 = mu, Wce = mu B0 and Wci = B0 (mu: mass ratio).
void whistler_dispersion(const std::vector<Fixture>& cases) {
  const auto& parameters = find(cases, "Whistler").parameters;
  const auto initial_case = tpic::initialization::prepare_initial_case(parameters);
  require(std::holds_alternative<tpic::initialization::WhistlerCase>(initial_case),
          "Whistler deck did not select the Whistler case");
  const auto& w = std::get<tpic::initialization::WhistlerCase>(initial_case);
  double electron = 0.0, ion = 0.0;
  for (const auto& species : parameters.species) {
    if (species.q_over_m < 0.0 && electron == 0.0) electron = -species.q_over_m;
    if (species.q_over_m > 0.0 && ion == 0.0) ion = species.q_over_m;
  }
  const double b0 = parameters.b0x, mu = electron / ion, omega = w.omega;
  const double lhs = parameters.c * parameters.c * w.wavenumber * w.wavenumber / (omega * omega);
  const double rhs = 1.0 - mu / (omega * (omega - mu * b0)) - 1.0 / (omega * (omega + b0));
  require(omega > b0 && omega < mu * b0, "Whistler root lies outside the whistler band");
  require(std::abs(lhs - rhs) <= 1.0e-9 * std::abs(lhs), "Whistler dispersion residual is not zero");
}

// Zero seed amplitudes leave exactly the uniform background and no E.
void zero_amplitudes(const std::vector<Fixture>& cases) {
  for (const auto* label : {"Weibel no seed", "TwoStream no seed", "TGV no seed",
                            "Harris no sheet field", "ForceFree no sheet field",
                            "Flux rope zero amplitude", "Whistler zero eigenmode amplitude",
                            "Whistler zero noise amplitude",
                            "Whistler magnetic-only epsilon=0.000000"}) {
    const auto& p = find(cases, label).parameters;
    const auto profile = profile_for(p);
    for (const auto& x : sample_points(p))
      require(profile.magnetic(x[0], x[1], x[2]) == FieldVector{p.b0x, p.b0y, p.b0z} &&
              profile.electric(x[0], x[1], x[2]) == FieldVector{},
              std::string(label) + ": zero amplitude changed the background");
  }
}

// Host field images of 1 and 12 logical ranks: every rank samples the profile
// at nodes (E) and cell centres (B), halos included, and its points are those
// of the single-rank image at the same global index.
void rank_split_images(const std::vector<Fixture>& cases) {
  using Fields = tpic::data::FieldSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
  for (const auto& fixture : cases) {
    const auto& parameters = fixture.parameters;
    const auto profile = profile_for(parameters);
    const auto single = tpic::config::prepare_domain(parameters.grid, parameters.topology, 0, 1);
    for (const auto dimensions : {std::array<int, 3>{1, 1, 1}, std::array<int, 3>{3, 2, 2}}) {
      auto topology = parameters.topology;
      topology.xlen = dimensions[0];
      topology.ylen = dimensions[1];
      topology.zlen = dimensions[2];
      const int size = dimensions[0] * dimensions[1] * dimensions[2];
      for (int rank = 0; rank < size; ++rank) {
        const auto domain = tpic::config::prepare_domain(parameters.grid, topology, rank, size);
        const auto image = tpic::initialization::make_host_field_image<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(
            parameters, profile.initial_case, domain);
        const auto ex = image.electric.get<Fields::Ex>();
        const auto ey = image.electric.get<Fields::Ey>();
        const auto ez = image.electric.get<Fields::Ez>();
        const auto bx = image.magnetic_cell.get<Fields::Bx>();
        const auto by = image.magnetic_cell.get<Fields::By>();
        const auto bz = image.magnetic_cell.get<Fields::Bz>();
        const std::string context = fixture.label + ", rank " + std::to_string(rank) + "/" +
                                    std::to_string(size);
        for (const bool nodal : {false, true}) {
          const double shift = nodal ? 0.0 : 0.5;
          const int extra = nodal ? 1 : 0;
          for (int k = -1; k <= domain.local_cells[2] + extra; ++k)
            for (int j = -1; j <= domain.local_cells[1] + extra; ++j)
              for (int i = -1; i <= domain.local_cells[0] + extra; ++i) {
                const std::array<int, 3> index{i, j, k};
                FieldVector x{};
                for (std::size_t axis = 0; axis < 3; ++axis) {
                  x[axis] = domain.origin[axis] + (index[axis] + shift) * domain.spacing[axis];
                  const double global = single.origin[axis] +
                      (domain.global_offset[axis] + index[axis] + shift) * single.spacing[axis];
                  require(std::abs(x[axis] - global) <= 1.0e-12 * std::max(1.0, single.lengths[axis]),
                          context + ": rank sample point differs from the single-rank point");
                }
                if (nodal)
                  require(FieldVector{ex(i, j, k), ey(i, j, k), ez(i, j, k)} ==
                              profile.electric(x[0], x[1], x[2]),
                          context + ": node E differs from the profile");
                else
                  require(FieldVector{bx(i, j, k), by(i, j, k), bz(i, j, k)} ==
                              profile.magnetic(x[0], x[1], x[2]),
                          context + ": cell B differs from the profile");
              }
        }
      }
    }
  }
}

void whistler_rejections() {
  using Mode = tpic::config::WhistlerSeedMode;
  const std::pair<const char*, Modify> cases[] = {
      {"zero amplitude still validates the parallel mode",
       [](auto& p) { p.whistler_seed_epsilon = 0.0; p.whistler_mode_y = 1; }},
      {"zero amplitude still solves the cold root",
       [](auto& p) { p.whistler_seed_epsilon = 0.0; p.grid.lx = 1000.0; p.whistler_mode_x = 1; }},
      {"invalid magnetic-only mode",
       [](auto& p) { p.whistler_seed_mode = Mode::MagneticOnly; p.whistler_mode_x = 0; }},
      {"invalid noise limit",
       [](auto& p) { p.whistler_seed_mode = Mode::Noise; p.whistler_noise_mode_max = -1; }}};
  for (const auto& [label, modify] : cases) {
    auto parameters = fixture_parameters("Whistler");
    modify(parameters);
    bool rejected = false;
    try {
      (void)profile_for(parameters);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, std::string("Whistler profile accepted: ") + label);
  }
}

// A case name outside the input grammar has no case.
void unknown_cases() {
  for (const auto name : {"", "gem", "HarrisSheet", "FluxRope"}) {
    bool rejected = false;
    try {
      (void)profile_for(fixture_parameters(name));
    } catch (const std::invalid_argument& error) {
      rejected = std::string(error.what()).find("unknown Case") != std::string::npos;
    }
    require(rejected, "unknown case name accepted: [" + std::string(name) + "]");
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = golden::parse_options(argc, argv);
    coordinate_validation();
    sampled_overflow();
    const auto cases = fixtures();
    divergence_free(cases);
    harris_seeds(cases);
    periodic_modes(cases);
    whistler_dispersion(cases);
    zero_amplitudes(cases);
    rank_split_images(cases);
    whistler_rejections();
    unknown_cases();
    golden::check(options, "field_profiles.txt", golden_samples(cases), sample_equal);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
