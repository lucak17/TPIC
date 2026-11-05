#include <tpic/initialization/cases/initial_case.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace {

using tpic::config::SimulationParameters;
using tpic::config::WhistlerSeedMode;
using tpic::initialization::prepare_initial_case;
using namespace tpic::initialization;

template <class Case> Case prepared(const SimulationParameters& params) {
  return std::get<Case>(prepare_initial_case(params));
}

std::string_view label(const SimulationParameters& params) {
  return std::visit([](const auto& selected) { return selected.particle_label(); },
                    prepare_initial_case(params));
}

// Other names and near misses have no case.
constexpr std::string_view unknown_names[] = {
    "", "uniform", "unknown_case", "unknown case", "Dipole", "gem", "GEMNOPERT", "GEMDoubleHarris",
    "testGEM3D", "HarrisSheet", "DoubleHarris", "doubleGEM", "forcefree", "weibel", "whistler",
    "twostream", "taylorgreenvortex", "PlasmoidChain", "plasmoid_chain", "PLASMOIDCHAIN",
    "plasmoidchain", "FluxRope", "fluxrope", "flux_rope", "FluxRope3D", "fluxRope3D", "FluxRopeKink",
    "fluxRopeKink", "FLUXROPE", " GEM", "GEM ", "\"GEM\"", "GEM#comment", "GEMextra",
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, const char* message, double tolerance = 2.0e-14) {
  require(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)), message);
}

template<class Function>
void rejects(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::logic_error& error) {
    require(std::string(error.what()).starts_with("TPIC recipe:"), "missing recipe error context");
    return;
  } catch (const std::overflow_error& error) {
    require(std::string(error.what()).starts_with("TPIC recipe:"), "missing recipe overflow context");
    return;
  }
  throw std::runtime_error(message);
}

SimulationParameters base(std::string name = "Uniform", int ns = 2) {
  SimulationParameters params;
  params.case_name = std::move(name);
  params.ns = ns;
  params.species.resize(static_cast<std::size_t>(ns));
  for (int index = 0; index < ns; ++index) {
    auto& species = params.species[static_cast<std::size_t>(index)];
    species.q_over_m = index % 2 == 0 ? -100.0 : 1.0;
    species.rho_init = 1.0;
    species.uth = species.vth = species.wth = 0.1;
    species.npcelx = species.npcely = species.npcelz = 1;
  }
  return params;
}

void uniform_and_sheets() {
  for (int ns = 1; ns <= 10; ++ns) {
    require(std::holds_alternative<UniformCase>(prepare_initial_case(base("Uniform", ns))) &&
                label(base("Uniform", ns)) == "uniform",
            "Uniform is not the uniform case");
    for (const auto name : unknown_names)
      rejects([&] { (void)prepare_initial_case(base(std::string(name), ns)); },
              "unknown case name accepted");
  }
  for (const std::string_view name : {"GEM", "GEMnoPert", "DoubleGEM", "DoubleGEMnoPert"}) {
    auto params = base(std::string(name));
    params.species[0].w0 = 1.0e-9;
    params.species[1].u0 = 1.0e-10;
    const auto sheet = prepared<HarrisSheetCase>(params);
    require(sheet.species.size() == 2 && sheet.species[0].harris_density &&
                !sheet.species[1].harris_density,
            "Harris drift threshold or species selection changed");
    // The seeded names hold the X-point constants 0.4, delta and 10.
    require(sheet.is_double == name.starts_with("Double") &&
                sheet.seed == (name.ends_with("noPert") ? HarrisSeed{}
                    : HarrisSeed{0.0, 0.0, 0.0, 0.4, 0.4, 0.5, 0.5, 10.0, 10.0, 10.0, 10.0}) &&
                sheet.species[0].is_double == sheet.is_double &&
                sheet.particle_label() == (sheet.is_double ? "DoubleGEM" : "GEM/Harris"),
            "Harris variant of a deck name changed");
    params.delta = 0.0;
    rejects([&] { (void)prepare_initial_case(params); }, "zero Harris delta accepted");
  }
  for (const char* name : {"plasmoidChain"}) {
    auto params = base(name, 4);
    params.species[2].u0 = 1.0;
    const auto sheet = prepared<HarrisSheetCase>(params);
    require(sheet.species[0].harris_density && sheet.species[1].harris_density &&
            !sheet.species[2].harris_density && !sheet.species[3].harris_density,
            "plasmoid sheet selection must use first species of each charge sign");
    require(!sheet.is_double && sheet.seed == HarrisSeed{} && sheet.particle_label() == "plasmoidChain",
            "plasmoidChain is not the single unperturbed sheet");
  }
  auto params = base("ForceFree");
  static_assert(std::same_as<ForceFreeCase::Profile, UniformProfile>);
  require(label(params) == "uniform", "ForceFree particle loading must remain uniform");
  params.delta = -1.0;
  rejects([&] { (void)prepare_initial_case(params); }, "negative ForceFree delta accepted");
  params = base();
  params.delta = -1.0; // Inactive case parameter remains inert.
  (void)prepare_initial_case(params);
}

void gem_harris() {
  constexpr double pi = 3.14159265358979323846;
  // Lx = Ly = 10 with 64 cells, delta = 0.5; electrons first, then ions.
  auto params = base("GEMHarris");
  params.harris_b0x = 0.02;
  params.species[0].w0 = 0.006;
  params.species[1].w0 = -0.03;
  auto sheet = prepared<HarrisSheetCase>(params);
  require(sheet.seed.mode_amplitude == 0.0 && sheet.seed.amplitude_x == 0.0 && sheet.seed.amplitude_y == 0.0 &&
              !sheet.is_double && sheet.particle_label() == "GEM/Harris" &&
              sheet.species[0].harris_density && sheet.species[0].ampere_drift == 0.0,
          "GEMHarris without keys is not the unseeded sheet");

  // The keys that give the seed of GEM, and the unseeded keys of every other name.
  params.pert_hump = 0.04 * pi;
  params.deltax_hump = params.deltay_hump = 1.0;
  params.kx_hump = params.ky_hump = pi / (10.0 * params.delta);
  sheet = prepared<HarrisSheetCase>(params);
  close(sheet.seed.amplitude_x, 0.4, "GEMHarris hump Bx amplitude");
  close(sheet.seed.amplitude_y, 0.4, "GEMHarris hump By amplitude");
  close(sheet.seed.period_x, 10.0, "GEMHarris hump x period");
  close(sheet.seed.slope_period_y, 10.0, "GEMHarris hump y slope period");
  require(sheet.seed.width_x == params.delta && sheet.seed.width_y == params.delta, "GEMHarris hump widths");
  auto preset = params;
  preset.case_name = "GEM";
  preset.current_from_ampere = preset.spatially_varying_thermal = true;
  preset.pert_gem = 1.0;
  require(prepare_initial_case(preset) == prepare_initial_case([&] {
            auto plain = base("GEM"); plain.harris_b0x = 0.02;
            plain.species[0].w0 = 0.006; plain.species[1].w0 = -0.03; return plain; }()),
          "GEM read a GEMHarris key");

  auto bad = params;
  bad.deltax_hump = 0.0;
  rejects([&] { (void)prepare_initial_case(bad); }, "zero deltaxHump accepted");
  bad = params; bad.deltay_hump = -1.0;
  rejects([&] { (void)prepare_initial_case(bad); }, "negative deltayHump accepted");
  bad = params; bad.ky_hump = 0.0;
  rejects([&] { (void)prepare_initial_case(bad); }, "kyHump = 0 accepted with a hump");
  bad = params; bad.pert_gem = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { (void)prepare_initial_case(bad); }, "nonfinite pertGEM accepted");
  auto flat = params;
  flat.kx_hump = 0.0;
  const auto flat_field = prepared<HarrisSheetCase>(flat).magnetic(flat, 3.0, 4.5, 0.0);
  require(std::isfinite(flat_field[0]) && std::isfinite(flat_field[1]), "kxHump = 0 gave a nonfinite field");

  // currentFromAmpere: the shares add up to the whole current, the profile
  // drift is the analytic curl over the cell shape, and unsafe drifts are refused.
  params.current_from_ampere = true;
  sheet = prepared<HarrisSheetCase>(params);
  const auto electrons = sheet.profile(0), ions = sheet.profile(1);
  require(sheet.particle_label() == "GEM/Harris, Ampere current", "GEMHarris Ampere label");
  close(-electrons.ampere_drift + ions.ampere_drift, params.c, "Ampere shares do not add up to c");
  const double dy = params.grid.ly / params.grid.nyc;
  const auto field = [&](double x, double y) { return sheet.magnetic(params, x, y, 0.0); };
  const auto numerical_curl = [&](double x, double y) {
    constexpr double h = 1.0e-3;
    const auto d = [&](int component, double ox, double oy) {
      return (-field(x + 2 * ox, y + 2 * oy)[component] + 8 * field(x + ox, y + oy)[component] -
              8 * field(x - ox, y - oy)[component] + field(x - 2 * ox, y - 2 * oy)[component]) / (12 * h);
    };
    return d(1, h, 0.0) - d(0, 0.0, h);
  };
  for (const auto& [x, y] : {std::pair{5.0, 5.0}, std::pair{5.3, 4.8}, std::pair{4.1, 5.6}, std::pair{6.0, 3.9}}) {
    close(ions.curl.at(x, y), numerical_curl(x, y), "analytic curl of the seeded sheet", 1.0e-7);
    ParticleLoadPlan plan{};
    plan.density = 1.0 / (4.0 * pi);
    plan.drift[2] = params.species[1].w0;
    plan.thermal[0] = plan.thermal[1] = plan.thermal[2] = 0.1;
    plan.spacing[0] = plan.spacing[1] = plan.spacing[2] = dy;
    const int cell[3] = {static_cast<int>(x / dy), static_cast<int>(y / dy), 0};
    const auto sech2 = [&](double at) {
      const double ch = std::cosh((at - 5.0) / params.delta);
      return 1.0 / (ch * ch);
    };
    const double cell_shape = 0.5 * (sech2(cell[1] * dy) + sech2(cell[1] * dy + dy));
    ParticleProfileSample sample{plan.density, 1.0, {0.0, 0.0, plan.drift[2]}, {0.1, 0.1, 0.1}};
    ions.evaluate(sample, plan, x, y, 0.0, cell);
    close(sample.density, plan.density * cell_shape, "Harris cell density");
    close(sample.drift[2], ions.ampere_drift * ions.curl.at(x, y) / cell_shape, "Ampere drift");
    require(sample.drift[0] == 0.0 && sample.drift[1] == 0.0 && sample.thermal[2] == 0.1, "Ampere in-plane drift");
  }
  params.spatially_varying_thermal = true;
  {
    const auto thermal = prepared<HarrisSheetCase>(params).profile(1);
    ParticleLoadPlan plan{};
    plan.spacing[1] = dy;
    const int cell[3] = {32, 32, 0};
    ParticleProfileSample sample{1.0, 1.0, {}, {0.1, 0.1, 0.1}};
    thermal.evaluate(sample, plan, 5.0, 5.0, 0.0, cell);
    close(sample.thermal[2], 0.1, "thermal speed of the drifting Maxwellian", 1.0e-12);
    require(sample.thermal[0] == 0.1 && sample.thermal[1] == 0.1, "in-plane thermal speed changed");
    ParticleProfileSample cold{1.0, 1.0, {}, {0.1, 0.1, 1.0e-30}};
    thermal.evaluate(cold, plan, 5.0, 5.0, 0.0, cell);
    require(cold.thermal[2] == 1.0e-30, "thermal fallback lost the deck value");
  }
  bad = params; bad.species[0].w0 = bad.species[1].w0 = 0.0;
  rejects([&] { (void)prepare_initial_case(bad); }, "Ampere current without a carrier accepted");
  bad = params; bad.species[1].u0 = 0.01;
  rejects([&] { (void)prepare_initial_case(bad); }, "Ampere current with an in-plane drift accepted");
  bad = params; bad.pert_gem = 0.01;
  rejects([&] { (void)prepare_initial_case(bad); }, "Ampere drift of the box-scale mode accepted");
  bad = params; bad.deltax_hump = 8.0; bad.deltay_hump = 4.0;
  rejects([&] { (void)prepare_initial_case(bad); }, "Ampere drift of a hump wider than the sheet accepted");
}

void weibel() {
  auto params = base("Weibel");
  params.weibel_seed_mode_x = 2;
  params.weibel_seed_amplitude = .01;
  params.weibel_seed_growth_rate = .4;
  const auto seeded = prepared<WeibelCase>(params);
  close(seeded.electric_amplitude * seeded.wavenumber, -.004,
        "Weibel growing-mode E amplitude changed");
  require(label(params) == "uniform", "Weibel particle loading must remain uniform");
  params.weibel_seed_mode_x = 0;
  rejects([&] { (void)prepare_initial_case(params); }, "nonzero Weibel seed accepted at zero mode");
  params.weibel_seed_amplitude = 0.0;
  (void)prepare_initial_case(params);
  params.weibel_seed_mode_x = -1;
  rejects([&] { (void)prepare_initial_case(params); }, "negative unseeded Weibel mode accepted");
}

void whistler() {
  auto params = base("Whistler");
  params.b0x = .1;
  params.whistler_mode_x = 4;
  params.whistler_seed_epsilon = .02;
  params.whistler_phase = .3;
  params.whistler_field_time_offset = .2;
  params.whistler_particle_time_offset = -.1;
  auto wave = prepared<WhistlerCase>(params);
  // Stored value: the exact cold-plasma dispersion root, computed once in host
  // FP64 (GCC) for Lx=10, mode=4, B0x=.1, c=1, first qom=(-100,+1). The
  // dispersion identity below checks the same root independently.
  close(wave.omega, .63473582771122661, "Whistler exact root differs from stored value");
  const double lhs = std::pow(params.c * wave.wavenumber / wave.omega, 2);
  const double rhs = 1.0 - 100.0 / (wave.omega * (wave.omega - 10.0))
                        - 1.0 / (wave.omega * (wave.omega + .1));
  close(lhs, rhs, "Whistler root fails cold-plasma dispersion", 1.0e-12);
  require(wave.species[0].electron_drift && !wave.species[1].electron_drift &&
              wave.particle_label() == "WhistlerUniform",
          "Whistler drift must select electrons only");
  params.whistler_use_exact_cold_root = false;
  close(prepared<WhistlerCase>(params).omega, .59412640890111801,
        "Whistler approximate frequency differs from stored value");

  params.whistler_use_exact_cold_root = true;
  params.whistler_mode_x = 1;
  params.whistler_seed_epsilon = 0.0;
  rejects([&] { (void)prepare_initial_case(params); }, "zero amplitude bypassed Whistler root failure");
  params.whistler_mode_x = 4;
  params.species[0].rho_init = 0.0;
  rejects([&] { (void)prepare_initial_case(params); }, "zero seed bypassed electron density validation");

  params = base("Whistler", 1);
  params.b0x = .1;
  params.whistler_mode_x = 4;
  params.whistler_seed_mode = WhistlerSeedMode::MagneticOnly;
  params.whistler_noise_mode_max = -1; // Ignored outside noise mode.
  params.whistler_field_time_offset = std::numeric_limits<double>::quiet_NaN();
  params.whistler_particle_time_offset = std::numeric_limits<double>::infinity();
  wave = prepared<WhistlerCase>(params);
  require(wave.omega == 0.0 && wave.electric_amplitude == 0.0 && wave.noise_modes.empty(),
          "magnetic-only seed must ignore cold root, missing ions, noise count and time offsets");
  params.whistler_mode_x = 0;
  rejects([&] { (void)prepare_initial_case(params); }, "magnetic-only zero mode accepted");

  params.whistler_seed_mode = WhistlerSeedMode::Noise;
  params.whistler_seed_epsilon = .02;
  params.whistler_phase = .3;
  params.whistler_noise_mode_max = 3;
  wave = prepared<WhistlerCase>(params);
  require(wave.noise_modes.size() == 3 && wave.omega == 0.0, "noise mode table or root semantics changed");
  // Stored samples of the magnetic noise field, computed once in host FP64 with
  // seed 12345 and WhistlerNoiseModeMax=3; they pin both hashed components and
  // all mode coefficients.
  constexpr std::array<std::array<double, 3>, 3> samples{{
      {0.0, .0018215841648399466, -.00099302444555924045},
      {.37, .0024493156637993719, .000053356345825403967},
      {2.0, -.00080740241329060262, -.00084065853878766952}}};
  for (const auto& sample : samples) {
    double by = 0.0, bz = 0.0;
    for (const auto& mode : wave.noise_modes) {
      by += wave.noise_amplitude * std::cos((mode.wavenumber * sample[0] + params.whistler_phase) + mode.phase_y);
      bz += wave.noise_amplitude * std::cos((mode.wavenumber * sample[0] + params.whistler_phase) + mode.phase_z);
    }
    close(by, sample[1], "Whistler y noise differs from stored value", 2.0e-17);
    close(bz, sample[2], "Whistler z noise differs from stored value", 2.0e-17);
  }
  params.whistler_noise_mode_max = 0;
  params.whistler_mode_x = -7;
  require(prepared<WhistlerCase>(params).noise_mode_max == 16, "noise default ignores negative ModeX");
  params.whistler_mode_x = 9;
  require(prepared<WhistlerCase>(params).noise_mode_max == 9, "noise fallback to ModeX changed");
  params.whistler_noise_mode_max = -1;
  rejects([&] { (void)prepare_initial_case(params); }, "negative noise maximum accepted");
  params.whistler_noise_mode_max = std::numeric_limits<int>::max();
  rejects([&] { (void)prepare_initial_case(params); }, "unbounded active noise allocation accepted");
  params.whistler_seed_epsilon = 0.0;
  require(prepared<WhistlerCase>(params).noise_modes.empty(), "zero noise seed allocated unused table");
  params.b0x = 0.0;
  rejects([&] { (void)prepare_initial_case(params); }, "zero-amplitude noise bypassed B0x constraint");
}

void two_stream() {
  auto params = base("TwoStream", 3);
  params.species[2].q_over_m = 1.0;
  params.species[1].q_over_m = -100.0;
  params.species[0].u0 = .2;
  params.species[1].u0 = -.2;
  params.species[2].rho_init = 2.0;
  params.two_stream_seed_mode_x = 3;
  params.two_stream_seed_density_amplitude = .1;
  const auto beams = prepared<TwoStreamCase>(params);
  require(beams.species[1].density_seed && !beams.species[2].density_seed && beams.particle_label() == "TwoStream",
          "TwoStream seeded species changed");
  close(beams.electric_amplitude * beams.wavenumber, -.2,
        "TwoStream E/density Gauss relation changed");
  const auto valid = params;
  params.two_stream_seed_mode_x = std::numeric_limits<int>::max();
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing TwoStream Nyquist comparison accepted");
  params = valid; params.species[2].npcelz = 2;
  rejects([&] { (void)prepare_initial_case(params); }, "unequal TwoStream PPC accepted");
  params = valid; params.species[1].u0 = -.3;
  rejects([&] { (void)prepare_initial_case(params); }, "asymmetric beams accepted");
  params = valid; params.species[2].rho_init = 3.0;
  rejects([&] { (void)prepare_initial_case(params); }, "nonneutral TwoStream densities accepted");
  params = valid; params.topology.periodic_x_particles = false;
  rejects([&] { (void)prepare_initial_case(params); }, "nonperiodic TwoStream particles accepted");
}

void tgv_and_flux_rope() {
  auto params = base("TaylorGreenVortex", 4);
  params.taylor_green_vortex_magnetic_amplitude = .1;
  const auto vortex = prepared<TaylorGreenVortexCase>(params);
  close(vortex.species[0].current_split, -100.0 / 101.0, "TGV electron current split changed");
  close(vortex.species[1].current_split, 1.0 / 101.0, "TGV ion current split changed");
  require(vortex.species[0].current_split == vortex.species[2].current_split &&
              vortex.particle_label() == "TaylorGreenVortex",
          "TGV repeats the current split for each same-sign species");
  params.grid.ly = 20.0;
  rejects([&] { (void)prepare_initial_case(params); }, "TGV unequal wavenumbers accepted");
  params.taylor_green_vortex_mode_y = 2;
  (void)prepare_initial_case(params);
  params.taylor_green_vortex_mode_z = 0;
  rejects([&] { (void)prepare_initial_case(params); }, "active TGV zero z mode accepted");
  params = base("TaylorGreenVortex", 1);
  (void)prepare_initial_case(params);
  params.taylor_green_vortex_magnetic_amplitude = .1;
  rejects([&] { (void)prepare_initial_case(params); }, "TGV magnetic seed without ions accepted");
  params.taylor_green_vortex_magnetic_amplitude = 0.0;
  params.species[0].rho_init = 0.0;
  rejects([&] { (void)prepare_initial_case(params); }, "inactive TGV zero density accepted");

  params = base("fluxRope", 4);
  params.species[2].rho_init = 3.0;
  const auto rope = prepared<FluxRopeCase>(params);
  close(rope.species[0].species_group_density, 4.0, "flux rope electron density sum changed");
  close(rope.species[1].species_group_density, 2.0, "flux rope ion density sum changed");
  close(rope.species[1].current_carrier_density, 4.0, "flux rope current carrier density changed");
  require(rope.species[0].electron_species && !rope.species[1].electron_species &&
              rope.particle_label() == "fluxRope",
          "flux rope species sign or label changed");
  params.flux_rope_electron_pressure_fraction = 1.1;
  rejects([&] { (void)prepare_initial_case(params); }, "invalid flux rope pressure partition accepted");
  params = base("fluxRope", 1);
  rejects([&] { (void)prepare_initial_case(params); }, "flux rope without positive ion density accepted");
}

void finite_and_shape_checks() {
  auto params = base("Weibel"); params.weibel_seed_amplitude = std::numeric_limits<double>::quiet_NaN();
  rejects([&] { (void)prepare_initial_case(params); }, "NaN active recipe parameter accepted");
  params = base("Whistler"); params.b0x = 1.0e308;
  params.whistler_seed_mode = WhistlerSeedMode::MagneticOnly;
  params.whistler_mode_x = 1; params.whistler_seed_epsilon = 10.0;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing derived seed accepted");
}

void analytical_coefficient_safety() {
  // These are finite input scalars that produce infinities/NaNs in the
  // analytic formulas before any particle random sampling is involved.
  auto params = base("TaylorGreenVortex");
  params.grid.lx = params.grid.ly = params.grid.lz = 1.0;
  params.taylor_green_vortex_magnetic_amplitude = 1.0e308;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing TGV B*k coefficients accepted");
  params = base("TaylorGreenVortex");
  params.taylor_green_vortex_magnetic_amplitude = 1.0e100;
  params.species[0].rho_init = 1.0e-210;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing TGV current/density drift accepted");
  params = base("TaylorGreenVortex");
  params.taylor_green_vortex_magnetic_amplitude = .1;
  params.species[0].q_over_m = -1.0e100;
  params.species[1].rho_init = 1.0e-310;
  rejects([&] { (void)prepare_initial_case(params); }, "TGV separate reciprocal density overflow accepted");
  params = base("TaylorGreenVortex");
  params.taylor_green_vortex_velocity_amplitude = 1.0e200;
  params.taylor_green_vortex_seed_motional_e = true;
  params.b0z = 1.0e200;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing TGV motional E accepted");
  params.b0z = 0.0;
  params.taylor_green_vortex_magnetic_amplitude = 1.0e200;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing TGV motional products accepted despite analytic cancellation");
  params = base("TaylorGreenVortex");
  params.taylor_green_vortex_magnetic_amplitude = 1.0e-320;
  params.grid.lx = params.grid.ly = params.grid.lz = 1.0e10;
  rejects([&] { (void)prepare_initial_case(params); }, "underflowing nonzero TGV current coefficient accepted");

  params = base("Whistler");
  params.b0x = .1;
  params.whistler_mode_x = 4;
  params.species[0].rho_init = 1.0e-320;
  rejects([&] { (void)prepare_initial_case(params); }, "zero Whistler seed accepted 0*infinite inverse density");

  params = base("fluxRope");
  for (double width : {1.0e-200, 1.0e-100, 1.0e100}) {
    params.flux_rope_width = width;
    rejects([&] { (void)prepare_initial_case(params); }, "unrepresentable flux rope width/denominator squared accepted");
  }
  params.flux_rope_width = 1.0e-70;
  (void)prepare_initial_case(params); // A narrow but representable denominator is allowed.
  params = base("fluxRope"); params.flux_rope_b0 = 1.0e200;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing flux rope B0 squared accepted");
  params.flux_rope_b0 = 1.0e-200;
  rejects([&] { (void)prepare_initial_case(params); }, "underflowing flux rope B0 squared accepted");
  params = base("fluxRope"); params.flux_rope_center_x = 1.0e200;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing flux rope displacement squared accepted");
  params = base("fluxRope"); params.flux_rope_pressure_floor = 1.0e308;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing flux rope thermal variance accepted");
  params = base("fluxRope"); params.species[0].rho_init = 1.0e-320;
  params.flux_rope_electron_pressure_fraction = 0.0;
  rejects([&] { (void)prepare_initial_case(params); }, "overflowing cold flux rope electron current drift accepted");
  params = base("fluxRope"); params.flux_rope_b0 = params.flux_rope_pressure_floor = 0.0;
  (void)prepare_initial_case(params); // Explicitly zero pressure/current remain valid.

  params = base("GEM"); params.delta = 1.0e-309;
  rejects([&] { (void)prepare_initial_case(params); }, "nonfinite Harris perturbation phase accepted");
  params.case_name = "GEMnoPert";
  (void)prepare_initial_case(params); // tanh(infinity)/sech(infinity) have finite limits.
}

} // namespace

void gaussian_expansion() {
  // Node 32 of the default 10^3 box of 64 cells.
  auto params = base("GaussianExpansion");
  params.quiet_start = true;
  params.gaussian_expansion_center_x = params.gaussian_expansion_center_y = 5;
  params.gaussian_expansion_sigma_x = 0.5;
  params.gaussian_expansion_sigma_y = 0.25;
  params.gaussian_expansion_cutoff = 4.0;
  const auto expansion = prepared<GaussianExpansionCase>(params);
  const auto profile = expansion.profile(1);
  require(expansion.profile(0) == profile && label(params) == "GaussianExpansion" &&
              expansion.electric(params, 5, 5, 5) == FieldVector{} &&
              expansion.magnetic(params, 5, 5, 5) == FieldVector{},
          "GaussianExpansion loads one shape without a field");
  require(profile.region.center[0] == 5 && profile.region.center[1] == 5 && profile.region.center[2] == 0 &&
              profile.region.radius[0] == 2 && profile.region.radius[1] == 1 && profile.region.radius[2] == 0 &&
              profile.inverse_two_variance[0] == 2 && profile.inverse_two_variance[1] == 8 &&
              profile.inverse_two_variance[2] == 0 && profile.support() == profile.region,
          "GaussianExpansion centre, cutoff and widths");
  ParticleProfileSample sample{3.0};
  profile.evaluate(sample, ParticleLoadPlan{}, 5.5, 5.25, 7.0, nullptr);
  close(sample.density, 3.0 * std::exp(-1.0), "GaussianExpansion Gaussian weight");
  params.gaussian_expansion_cutoff = 0.0;
  require(!prepared<GaussianExpansionCase>(params).profile(0).support().bounded(),
          "a zero cutoff loads every cell");

  const auto rejected = [&](const char* message, auto&& modify) {
    auto changed = params;
    modify(changed);
    rejects([&] { (void)prepare_initial_case(changed); }, message);
  };
  rejected("GaussianExpansion without a width", [](auto& p) { p.gaussian_expansion_sigma_x = p.gaussian_expansion_sigma_y = 0; });
  rejected("negative GaussianExpansion width", [](auto& p) { p.gaussian_expansion_sigma_z = -1; });
  rejected("underflowing GaussianExpansion width", [](auto& p) { p.gaussian_expansion_sigma_x = 1e200; });
  rejected("negative GaussianExpansion cutoff", [](auto& p) { p.gaussian_expansion_cutoff = -1; });
  rejected("nonfinite GaussianExpansion cutoff",
           [](auto& p) { p.gaussian_expansion_cutoff = std::numeric_limits<double>::infinity(); });
  rejected("GaussianExpansion centre off a node", [](auto& p) { p.gaussian_expansion_center_y = 5.01; });
  rejected("GaussianExpansion centre outside the box", [](auto& p) { p.gaussian_expansion_center_x = -0.15625; });
  rejected("GaussianExpansion species of unequal npcel", [](auto& p) { p.species[1].npcely = 2; });
  rejected("GaussianExpansion pairs at random positions", [](auto& p) { p.quiet_start = false; });
  rejected("drifting GaussianExpansion species", [](auto& p) { p.species[0].w0 = 0.01; });
  rejected("charged GaussianExpansion plasma", [](auto& p) { p.species[1].rho_init = 0.5; });
  rejected("GaussianExpansion species without density", [](auto& p) { p.species[0].rho_init = p.species[1].rho_init = 0; });
  // A uniform axis takes any centre.
  params.gaussian_expansion_center_z = 5.01;
  (void)prepared<GaussianExpansionCase>(params);
}

// The electron units of docs/user/whistler_turbulence.md: c = omega_pe = d_e = 1,
// B0z = |Omega_e|/omega_pe = 1/sqrt(5), m_p/m_e = 1836. The expected fields,
// flow and energies are an independent NumPy evaluation of the seed formulas
// of its Section 3 with the same phases.
void whistler_turbulence() {
  auto params = base("WhistlerTurbulence");
  params.quiet_start = true;
  params.species[0].q_over_m = -1.0;
  params.species[1].q_over_m = 1.0 / 1836.0;
  params.b0z = 1.0 / std::sqrt(5.0);
  params.grid.lx = params.grid.ly = params.grid.lz = 51.2;
  params.whistler_turbulence_epsilon = 0.1;
  const auto turbulence = prepared<WhistlerTurbulenceCase>(params);
  std::array<int, 343> seen{};
  int visits = 0;
  WhistlerTurbulenceModes::for_each_index([&](int mode, int nx, int ny, int nz) {
    require(mode == visits++ && nz != 0, "WhistlerTurbulence mode order or a mode perpendicular to B0");
    ++seen[static_cast<std::size_t>((nx + 3) + 7 * (ny + 3) + 49 * (nz + 3))];
  });
  require(visits == WhistlerTurbulenceModes::count && std::ranges::max(seen) == 1 &&
              std::ranges::equal(seen, std::views::reverse(seen)),
          "WhistlerTurbulence modes must be distinct and closed under k -> -k");

  const auto electrons = turbulence.profile(0);
  require(electrons.modes == turbulence.modes && electrons.flow_scale != 0.0 &&
              turbulence.profile(1).flow_scale == 0.0,
          "WhistlerTurbulence flow belongs to the electrons");
  const auto b = turbulence.magnetic(params, 1.3, 2.9, 4.1);
  const auto e = turbulence.electric(params, 1.3, 2.9, 4.1);
  double u = 0.0, v = 0.0, w = 0.0;
  electrons.add_velocity_seed(1.3, 2.9, 4.1, u, v, w);
  const std::array<double, 9> actual{b[0], b[1], b[2], e[0], e[1], e[2], u, v, w};
  constexpr std::array<double, 9> expected{
      0.10527339026896348, -0.02621311715757811, 0.057152509439836664,
      0.0042348672448334155, 0.017965610630977576, 0.009786525387281766,
      0.07137542510983366, -0.00498033991303902, -0.02467769790166761};
  for (std::size_t i = 0; i < actual.size(); ++i)
    close(actual[i], expected[i], "WhistlerTurbulence field or electron flow", 1.0e-12);
  // On 8^3 points the modes stay orthogonal: the energy densities in units of
  // B0^2/(8*pi) are epsilon, <|E|^2>/B0^2 and rhoINIT*<u^2>/(|qom|*B0^2).
  double magnetic = 0.0, electric = 0.0, flow = 0.0;
  for (int point = 0; point < 512; ++point) {
    const double x = point % 8 * 6.4, y = point / 8 % 8 * 6.4, z = point / 64 * 6.4;
    const auto bp = turbulence.magnetic(params, x, y, z);
    const auto ep = turbulence.electric(params, x, y, z);
    double up = 0.0, vp = 0.0, wp = 0.0;
    electrons.add_velocity_seed(x, y, z, up, vp, wp);
    magnetic += bp[0] * bp[0] + bp[1] * bp[1] + bp[2] * bp[2];
    electric += ep[0] * ep[0] + ep[1] * ep[1] + ep[2] * ep[2];
    flow += up * up + vp * vp + wp * wp;
  }
  const double energy = 512.0 * params.b0z * params.b0z;
  close(magnetic / energy, 0.1, "WhistlerTurbulence magnetic energy", 1.0e-13);
  close(electric / energy, 9.587756134410467e-4, "WhistlerTurbulence electric energy", 1.0e-13);
  close(flow / energy, 0.01678508046297854, "WhistlerTurbulence electron flow energy", 1.0e-13);

  // Electrons of one qom share the flow; a zero seed has no field and no flow.
  auto split = params;
  split.ns = 3;
  split.species.push_back(split.species[0]);
  split.species[0].rho_init = split.species[2].rho_init = 0.5;
  const auto shared = prepared<WhistlerTurbulenceCase>(split);
  require(shared.profile(0).flow_scale == electrons.flow_scale &&
              shared.profile(2).flow_scale == electrons.flow_scale,
          "WhistlerTurbulence electron species carry one flow");
  auto quiet = params;
  quiet.whistler_turbulence_epsilon = 0.0;
  const auto control = prepared<WhistlerTurbulenceCase>(quiet);
  require(control.magnetic(quiet, 1.3, 2.9, 4.1) == FieldVector{} &&
              control.electric(quiet, 1.3, 2.9, 4.1) == FieldVector{} && control.profile(0).flow_scale == 0.0,
          "zero WhistlerTurbulence seed");

  const auto rejected = [&](const char* message, auto&& modify) {
    auto changed = params;
    modify(changed);
    rejects([&] { (void)prepare_initial_case(changed); }, message);
  };
  rejected("negative WhistlerTurbulenceEpsilon", [](auto& p) { p.whistler_turbulence_epsilon = -0.1; });
  rejected("nonfinite WhistlerTurbulenceEpsilon",
           [](auto& p) { p.whistler_turbulence_epsilon = std::numeric_limits<double>::quiet_NaN(); });
  rejected("WhistlerTurbulence B0 off z", [](auto& p) { p.b0x = 0.01; });
  rejected("WhistlerTurbulence without B0", [](auto& p) { p.b0z = 0.0; });
  rejected("WhistlerTurbulence particle wall", [](auto& p) { p.topology.periodic_y_particles = false; });
  rejected("WhistlerTurbulence species of unequal npcel", [](auto& p) { p.species[1].npcelz = 2; });
  rejected("charged WhistlerTurbulence plasma", [](auto& p) { p.species[1].rho_init = 0.5; });
  rejected("WhistlerTurbulence without electron density",
           [](auto& p) { p.species[0].rho_init = p.species[1].rho_init = 0.0; });
  rejected("WhistlerTurbulence electrons of two qom", [&](auto& p) {
    p = split;
    p.species[2].q_over_m = -2.0;
  });
}

// The rows of a support against every subcell centre: on each rank of a 3 x 2
// decomposition, and summed to the whole domain's points.
void support_rows() {
  tpic::config::GridConfig grid{.lx = 9.5, .ly = 14.25, .lz = 5.5, .nxc = 17, .nyc = 13, .nzc = 9};
  tpic::config::TopologyConfig topology;
  topology.xlen = 3;
  topology.ylen = 2;
  const int ppc[3]{2, 3, 1};
  const std::array<ParticleSupport, 4> supports{{
      {{4.75, 7.125, 2.75}, {3.0, 3.0, 2.5}},     // an ellipsoid inside the box
      {{0.0, 7.125, 0.0}, {4.0, 2.0, 0.0}},       // a cylinder along z, cut by the x wall
      {{9.5, 14.25, 5.5}, {1.0, 1.0, 1.0}},       // a corner: one rank holds it
      {{4.75, 7.125, 2.75}, {0.0, 0.0, 0.0}}}};   // unbounded: every row whole
  for (const auto& support : supports) {
    std::int64_t global = 0, summed = 0;
    for (int rank = 0; rank < 6; ++rank) {
      const auto domain = tpic::config::prepare_domain(grid, topology, rank, 6);
      if (rank == 0)
        visit_support_rows(support, domain, ppc, true, [&](auto, std::int64_t points) { global += points; });
      const std::array<int, 3> extent{domain.local_cells[0] * ppc[0], domain.local_cells[1] * ppc[1],
                                      domain.local_cells[2] * ppc[2]};
      // Every local subcell, inside the support by the predicate of the rows.
      std::vector<std::int64_t> expected(static_cast<std::size_t>(extent[1] * extent[2]));
      for (int row = 0; row < extent[1] * extent[2]; ++row)
        for (int x = 0; x < extent[0]; ++x) {
          const std::array<int, 3> local{x, row % extent[1], row / extent[1]};
          double sum = 0.0;
          for (std::size_t a = 0; a < 3; ++a) {
            if (!(support.radius[a] > 0.0)) continue;
            const double d = ((static_cast<double>(domain.global_offset[a] * ppc[a] + local[a]) + 0.5) *
                                  (domain.spacing[a] / ppc[a]) - support.center[a]) / support.radius[a];
            sum += d * d;
          }
          expected[static_cast<std::size_t>(row)] += sum <= 1.0;
        }
      std::int64_t last = -1;
      visit_support_rows(support, domain, ppc, false, [&](std::int64_t first, std::int64_t points) {
        const auto row = first / extent[0];
        auto& left = expected[static_cast<std::size_t>(row)];
        require(row > last && points > 0 && first % extent[0] + points <= extent[0] && left == points,
                "support row differs from its subcells");
        left = 0;
        last = row;
        summed += points;
      });
      for (const auto left : expected) require(left == 0, "support rows missed a subcell");
    }
    require(summed == global, "local support points do not add up to the whole domain's");
  }
}

int main() {
  try {
    uniform_and_sheets();
    gem_harris();
    weibel();
    whistler();
    two_stream();
    tgv_and_flux_rope();
    gaussian_expansion();
    whistler_turbulence();
    support_rows();
    finite_and_shape_checks();
    analytical_coefficient_safety();
    std::cout << "TPIC case admission, deck names, labels and Whistler stored coefficients passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
