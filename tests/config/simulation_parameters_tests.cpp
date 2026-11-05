#include <tpic/config/simulation_parameters.hpp>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using tpic::config::EcsimSmoothingMode;
using tpic::config::EcsimSmoothingOperator;
using tpic::config::SimulationParameters;
using tpic::config::WhistlerSeedMode;

// These remain host doubles even when a device schema selects float storage.
static_assert(std::is_same_v<decltype(SimulationParameters{}.dt), double>);
static_assert(std::is_same_v<decltype(tpic::config::SpeciesConfig{}.uth), double>);
static_assert(std::is_same_v<decltype(tpic::config::SpeciesConfig{}.rho_init), double>);
static_assert(std::is_same_v<decltype(tpic::config::SpeciesConfig{}.q_over_m), double>);

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <class Function>
void expect_error(Function&& function, std::string_view fragment) {
  try {
    std::forward<Function>(function)();
  } catch (const std::exception& error) {
    require(std::string_view(error.what()).find(fragment) != std::string_view::npos,
            "Error did not contain expected text: " + std::string(fragment) +
                "; got: " + error.what());
    return;
  }
  throw std::runtime_error("Expected an error containing: " + std::string(fragment));
}

class InputFixture {
public:
  InputFixture() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int suffix = 0; suffix < 32; ++suffix) {
      directory_ = std::filesystem::temp_directory_path() /
                   ("tpic_simulation_parameters_" + std::to_string(stamp) + "_" + std::to_string(suffix));
      if (std::filesystem::create_directory(directory_)) {
        return;
      }
    }
    throw std::runtime_error("Unable to create an input-test directory");
  }

  ~InputFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  InputFixture(const InputFixture&) = delete;
  InputFixture& operator=(const InputFixture&) = delete;

  SimulationParameters read(std::string_view text) const {
    const auto path = directory_ / "inputfile";
    {
      std::ofstream output(path);
      output << text;
      output.close();
      require(static_cast<bool>(output), "Unable to write test input");
    }
    return tpic::config::load_simulation_parameters(path.string());
  }

  SimulationParameters with(std::string_view overrides) const {
    return read(base() + std::string(overrides));
  }

  static std::string base(int species = 1) {
    std::string text = "dt = 0.125\nncycles = 12\nCase = Uniform\nns = " + std::to_string(species) + "\n";
    for (const auto key : {"rhoINIT", "qom", "npcelx", "npcely", "npcelz",
                           "uth", "vth", "wth", "u0", "v0", "w0"}) {
      text += std::string(key) + " =";
      for (int i = 0; i < species; ++i) {
        text += " " + std::to_string(i + 1);
      }
      text += '\n';
    }
    return text;
  }

private:
  std::filesystem::path directory_;
};

void defaults(const InputFixture& input) {
  const auto p = input.with("");
  require(p.dt == 0.125 && p.ncycles == 12 && p.th == 1.0, "Timing defaults");
  require(p.smooth == 1.0 && p.smooth_niter == 6 &&
              p.ecsim_smoothing_mode == EcsimSmoothingMode::None &&
              p.ecsim_smoothing_operator == EcsimSmoothingOperator::Face7, "Smoothing defaults");
  require(p.c == 1 && p.b0x == 0 && p.b0y == 0 && p.b0z == 0 && p.delta == 0.5 &&
              p.harris_b0x == 0 && p.forcefree_b0x == 0 && p.forcefree_b0z == 0,
          "Background field defaults");
  require(p.pert_gem == 0 && p.pert_hump == 0 && p.deltax_hump == 8 && p.deltay_hump == 4 &&
              p.kx_hump == -1 && p.ky_hump == -1 && !p.current_from_ampere && !p.spatially_varying_thermal,
          "Harris seed defaults");
  require(!p.quiet_start, "particles start at random positions by default");
  require(p.grid.lx == 10 && p.grid.ly == 10 && p.grid.lz == 10 &&
              p.grid.nxc == 64 && p.grid.nyc == 64 && p.grid.nzc == 64, "Grid defaults");
  require(p.topology.xlen == 1 && p.topology.ylen == 1 && p.topology.zlen == 1 &&
              p.topology.periodic_x && p.topology.periodic_y && p.topology.periodic_z &&
              p.topology.periodic_x_particles && p.topology.periodic_y_particles &&
              p.topology.periodic_z_particles, "Topology defaults");
  require(p.flux_rope_b0 == 0.1 && p.flux_rope_width == 1 && p.flux_rope_center_x == 5 &&
              p.flux_rope_center_y == 5 && p.flux_rope_pressure_floor == 9e-4 &&
              p.flux_rope_electron_pressure_fraction == 0.1, "Flux-rope defaults");
  require(p.gaussian_expansion_center_x == 5 && p.gaussian_expansion_center_y == 5 &&
              p.gaussian_expansion_center_z == 5 && p.gaussian_expansion_sigma_x == 0 &&
              p.gaussian_expansion_sigma_y == 0 && p.gaussian_expansion_sigma_z == 0 &&
              p.gaussian_expansion_cutoff == 0, "Gaussian-expansion defaults");
  require(p.whistler_turbulence_epsilon == 0 && p.whistler_turbulence_seed == 1, "Whistler-turbulence defaults");
  require(p.whistler_seed_mode == WhistlerSeedMode::Eigenmode &&
              p.whistler_use_exact_cold_root && p.whistler_noise_seed == 12345 &&
              p.whistler_noise_mode_max == 0 && p.whistler_field_time_offset == 0 &&
              p.whistler_particle_time_offset == 0, "Whistler defaults");
  require(p.taylor_green_vortex_mode_x == 1 && p.taylor_green_vortex_mode_y == 1 &&
              p.taylor_green_vortex_mode_z == 1 &&
              p.taylor_green_vortex_velocity_amplitude == 0 &&
              p.taylor_green_vortex_magnetic_amplitude == 0 &&
              !p.taylor_green_vortex_seed_motional_e, "TGV defaults");
  require(p.ns == 1 && p.ns_test_particles == 0 && p.species.size() == 1 &&
              p.test_species.empty() && p.gmres_tol == 1e-3, "Species and solver defaults");
  require(p.save_dir == "data" && p.restart_dir == "data" && p.case_name == "Uniform" &&
              p.write_method.empty() && p.field_write_method.empty() &&
              p.particles_write_method.empty() && p.restart_write_method.empty() && p.sim_name.empty(),
          "Bookkeeping defaults");
  require(p.output.field_cycle == 100 && p.output.particles_cycle == 0 &&
              p.output.test_particles_cycle == 0 && p.output.restart_cycle == 0 && p.output.call_finalize &&
              p.output.diagnostics_cycle == 100 && p.output.gauss_cycle == 0 && p.output.field_tag.empty() &&
              p.output.particles_tag.empty() && p.output.moments_tag.empty(), "Output defaults");
  require(p.bc.em_x_right == 0 && p.bc.em_x_left == 0 && p.bc.em_y_right == 0 &&
              p.bc.em_y_left == 0 && p.bc.em_z_right == 0 && p.bc.em_z_left == 0 &&
              p.bc.particle_x_right == 1 && p.bc.particle_x_left == 1 &&
              p.bc.particle_y_right == 1 && p.bc.particle_y_left == 1 &&
              p.bc.particle_z_right == 1 && p.bc.particle_z_left == 1, "Boundary defaults");
}

void lexical_grammar(const InputFixture& input) {
  const auto p = input.with(
      "\n # comment\n dt \t= 2.5e-1 # last duplicate wins\n"
      "SimulationName = \t a = b; c \t\nSaveDirName = \"a b\"\n"
      "RestartDirName = \"before#after\"\n");
  require(p.dt == 0.25, "Comments, blank lines and the last duplicate");
  require(p.sim_name == "a = b; c", "Split first equals and preserve semicolon");
  require(p.save_dir == "\"a b\"" && p.restart_dir == "\"before", "Raw strings and first hash");
  require(input.with("SaveDirName =\nRestartDirName =\n").save_dir.empty(),
          "An explicit empty output directory remains empty at parse time");
  // Every rejection names the deck line; the fixture's base deck has 15 lines.
  expect_error([&] { input.with("dt = broken\n"); }, "line 16: dt = 'broken' is not a number");
  expect_error([&] { input.with("dt = 2.5e-1 seconds\n"); }, "line 16: dt = '2.5e-1 seconds' is not a number");
  expect_error([&] { input.with("ncycles = 17.5\n"); }, "line 16: ncycles = '17.5' is not an integer");
  expect_error([&] { input.with("ns = 1trailing\n"); }, "line 16: ns = '1trailing' is not an integer");
  expect_error([&] { input.with("ncycles =\n"); }, "line 16: ncycles = '' is not an integer");
  expect_error([&] { input.with("not a key value line\n"); }, "line 16: expected 'key = value'");
  expect_error([&] { input.with(" = value\n"); }, "line 16: expected 'key = value'");
  require(input.read(InputFixture::base(1)).unknown_keys.empty() &&
              input.with("DT = 1\n").unknown_keys == std::vector<std::string>{"DT (line 16)"} &&
              input.with("\n\nUnknown = 1\nAlsoUnknown = 2\n").unknown_keys ==
                  std::vector<std::string>{"Unknown (line 18)", "AlsoUnknown (line 19)"},
          "Unknown keys are ignored and listed in deck order");
  expect_error([&] { input.read("DT = 1\n"); }, "missing required key dt");
  expect_error([&] { input.read("dt = 1\n"); }, "missing required key ncycles");
  expect_error([&] { input.read("dt = 1\nncycles = 1\n"); }, "missing required key ns");
}

// Keys TPIC does not read: inert keys, alias keys and rhoINJECT. A deck may
// carry them; each is ignored and listed.
void removed_keys(const InputFixture& input) {
  for (const auto key : {
           "NonlinearECSIMNewtonMaxIter", "NonlinearECSIMGMRESRestart", "NonlinearECSIMGMRESMaxIter",
           "NonlinearECSIMParticleMaxIter", "NonlinearECSIMMaxPathSegmentsPerParticle",
           "NonlinearECSIMScalarReductionSlots", "NonlinearECSIMNewtonTol", "NonlinearECSIMGMRESTol",
           "NonlinearECSIMFDGMRESTol", "NonlinearECSIMParticleTol", "NonlinearECSIMJVPEpsilon",
           "NonlinearECSIMLineSearch", "NonlinearECSIMFrozenPathPreconditioner",
           "bcPHIfaceXright", "bcPHIfaceXleft", "bcPHIfaceYright", "bcPHIfaceYleft",
           "bcPHIfaceZright", "bcPHIfaceZleft", "PoissonCorrection", "PoissonCorrectionCycle",
           "CGtol", "NiterMover", "PushWithBatTime", "PushWithEatTime", "ImplSusceptTime",
           "ImplSusceptMode", "x_center", "y_center", "z_center", "L_square", "NpMaxNpRatio",
           "Vinj", "rhoINJECT", "pitch_angle", "energy", "ParaviewScriptPath",
           "HarrisB0", "TaylorGreenVortexAmplitude", "TestPartOutputCycle"}) {
    require(input.with(std::string(key) + " = 1\n").unknown_keys ==
                std::vector<std::string>{std::string(key) + " (line 16)"},
            "A key TPIC does not read is ignored and listed");
  }
}

void cases(const InputFixture& input) {
  for (const auto name : tpic::config::case_names)
    require(input.with("Case = " + std::string(name) + "\n").case_name == name, "Canonical case name");
  expect_error([&] {
    input.read("dt = 1\nncycles = 1\nns = 1\nrhoINIT = 1\nqom = -1\nnpcelx = 1\nnpcely = 1\n"
               "npcelz = 1\nuth = 0\nvth = 0\nwth = 0\nu0 = 0\nv0 = 0\nw0 = 0\n");
  }, "missing required key Case");
  for (const auto name : {"", "gem", "GEMDoubleHarris", "HarrisSheet", "DoubleHarris", "testGEM3D",
                          "FluxRope", "fluxrope", "flux_rope", "FluxRope3D", "fluxRope3D",
                          "FluxRopeKink", "fluxRopeKink", "PlasmoidChain", "plasmoid_chain",
                          "MysteryCase"}) {
    expect_error([&] { input.with("Case = " + std::string(name) + "\n"); },
                 "line 16: Case = '" + std::string(name) + "' is unknown (expected Uniform, GEM,");
  }
}

void output_methods(const InputFixture& input) {
  auto p = input.with("WriteMethod=hdf5\nFieldWriteMethod=vtk\nParticlesWriteMethod=adios2\n");
  require(p.write_method == "hdf5" && p.field_write_method == "vtk" &&
          p.particles_write_method == "adios2", "Independent output methods are parsed");
  p = input.with("WriteMethod=hdf5\nFieldWriteMethod=\nParticlesWriteMethod=default\n");
  require(p.write_method == "hdf5" && p.field_write_method.empty() &&
          p.particles_write_method == "default", "Empty output methods retain fallback intent");
  // Unknown and differently cased engine names are rejected at their deck line.
  for (const auto key : {"WriteMethod", "FieldWriteMethod", "ParticlesWriteMethod"})
    for (const auto value : {"unknown", "VTK", "HDF5", "Default", "bp4"})
      expect_error([&] { input.with(std::string(key) + " = " + value + "\n"); },
                   "TPIC input: line 16: " + std::string(key) + " = '" + value +
                       "' is unknown (expected default, pvtk, vtk, adios2, bp, bp5, hdf5 or h5)");
}

void booleans(const InputFixture& input) {
  const auto p = input.with("PERIODICX = false\nPERIODICX_P = true\nPERIODICY = false\nPERIODICZ = false\n");
  require(!p.topology.periodic_x && p.topology.periodic_x_particles &&
              !p.topology.periodic_y_particles && !p.topology.periodic_z_particles,
          "Independent particle periodicity and axis fallbacks");
  require(!input.with("WhistlerUseExactColdRoot = false\n").whistler_use_exact_cold_root &&
              input.with("TaylorGreenVortexSeedMotionalE = true\n").taylor_green_vortex_seed_motional_e,
          "A boolean is true or false");
  const auto harris = input.with("currentFromAmpere = 1\nspatiallyVaryingThermal = true\n");
  require(harris.current_from_ampere && harris.spatially_varying_thermal, "Harris loading options");
  require(input.with("QuietStart = true\n").quiet_start && input.with("QuietStart = 1\n").quiet_start &&
              !input.with("QuietStart = 0\n").quiet_start, "QuietStart is a boolean");
  const auto digits = input.with("PERIODICX = 0\nPERIODICY = 1\nPERIODICZ = 0\nPERIODICZ_P = 1\n");
  require(!digits.topology.periodic_x && digits.topology.periodic_y && !digits.topology.periodic_z &&
              !digits.topology.periodic_x_particles && digits.topology.periodic_z_particles,
          "A boolean is 0 or 1");
  for (const auto value : {"", "2", "01", "F", "no", "NONE", "TRUE", "False", "true trailing", "1 trailing"}) {
    expect_error([&] { input.with("PERIODICX = " + std::string(value) + "\n"); },
                 "line 16: PERIODICX = '" + std::string(value) + "' is not a boolean (0, 1, true or false)");
  }
}

void dynamic_defaults(const InputFixture& input) {
  const auto p = input.with("TaylorGreenVortexMode = 3\nTaylorGreenVortexModeY = 7\n"
                            "FieldOutputCycle = 13\nLx = 18\nLy = 26\n");
  require(p.taylor_green_vortex_mode_x == 3 && p.taylor_green_vortex_mode_y == 7 &&
              p.taylor_green_vortex_mode_z == 3, "Common TGV mode with per-axis override");
  require(p.output.diagnostics_cycle == 13 && p.flux_rope_center_x == 9 &&
              p.flux_rope_center_y == 13 && p.gaussian_expansion_center_x == 9 &&
              p.gaussian_expansion_center_y == 13, "Defaults derived from resolved values");
  const auto overrides = input.with("HarrisB0x = 2\nTaylorGreenVortexMagneticAmplitude = 4\n"
                                    "TestParticlesOutputCycle = 6\nFieldOutputCycle = 7\n"
                                    "DiagnosticsOutputCycle = 8\nDiagnosticEnergyCycle = 5\nDiagnosticGaussCycle = 4\n"
                                    "pertGEM = .5\npertHump = .25\n"
                                    "deltaxHump = 2\ndeltayHump = 3\nkxHump = 0\nkyHump = 1.5\n");
  require(overrides.pert_gem == .5 && overrides.pert_hump == .25 && overrides.deltax_hump == 2 &&
              overrides.deltay_hump == 3 && overrides.kx_hump == 0 && overrides.ky_hump == 1.5,
          "Harris seed keys");
  require(overrides.harris_b0x == 2 && overrides.taylor_green_vortex_magnetic_amplitude == 4 &&
              overrides.output.test_particles_cycle == 6 && overrides.output.diagnostics_cycle == 8 &&
              overrides.output.diagnostic_energy_cycle == 5 && overrides.output.gauss_cycle == 4,
          "Explicit values take precedence over derived defaults");
  expect_error([&] { input.with("TaylorGreenVortexMode = bad\nTaylorGreenVortexModeX = 1\n"); },
               "TaylorGreenVortexMode = 'bad' is not an integer");
}

void enum_grammar(const InputFixture& input) {
  require(input.with("ECSimSmoothing = none\n").ecsim_smoothing_mode == EcsimSmoothingMode::None &&
              input.with("ECSimSmoothing = energy_conserving\n").ecsim_smoothing_mode ==
                  EcsimSmoothingMode::EnergyConserving, "Smoothing modes");
  require(input.with("ECSimSmoothingOperator = face7\n").ecsim_smoothing_operator ==
                  EcsimSmoothingOperator::Face7 &&
              input.with("ECSimSmoothingOperator = binomial27\n").ecsim_smoothing_operator ==
                  EcsimSmoothingOperator::Binomial27, "Smoothing operators");
  require(input.with("WhistlerSeedMode = eigenmode\n").whistler_seed_mode == WhistlerSeedMode::Eigenmode &&
              input.with("WhistlerSeedMode = magnetic_only\n").whistler_seed_mode ==
                  WhistlerSeedMode::MagneticOnly &&
              input.with("WhistlerSeedMode = noise\n").whistler_seed_mode == WhistlerSeedMode::Noise,
          "Whistler seed modes");
  for (const auto token : {"OFF", "False", "0", "EnergyConserving", "energy-conserving", "NONE"})
    expect_error([&] { input.with("ECSimSmoothing = " + std::string(token) + "\n"); },
                 "line 16: ECSimSmoothing = '" + std::string(token) +
                     "' is unknown (expected none or energy_conserving)");
  for (const auto token : {"SEVEN_POINT", "7point", "face6", "27point", "27_point", "directional",
                           "Directional27", "Face7"})
    expect_error([&] { input.with("ECSimSmoothingOperator = " + std::string(token) + "\n"); },
                 "line 16: ECSimSmoothingOperator = '" + std::string(token) +
                     "' is unknown (expected face7 or binomial27)");
  for (const auto token : {"EIGEN_MODE", "eigen-mode", "magnetic-only", "magnetic", "B_ONLY",
                           "b-only", "NOISE", "magnetic_noise", "magnetic-noise"})
    expect_error([&] { input.with("WhistlerSeedMode = " + std::string(token) + "\n"); },
                 "line 16: WhistlerSeedMode = '" + std::string(token) +
                     "' is unknown (expected eigenmode, magnetic_only or noise)");
}

void lists_and_species(const InputFixture& input) {
  const auto p = input.read(InputFixture::base(2) +
      "rhoINIT = 0.1 0.2 0.3\nqom = -25 1 99\n"
      "npcelx = 2 3\nnpcely = 4 5\nnpcelz = 6 7\n"
      "uth = 0.6 0.7\nvth = 0.8 0.9\nwth = 1.1 1.2\n"
      "u0 = 1.3 1.4\nv0 = 1.5 1.6\nw0 = 1.7 1.8\nCenterInitialVelocity = true\n");
  require(p.species.size() == 2 && p.species[0].q_over_m == -25 && p.species[1].q_over_m == 1,
          "Species indexing and ignored extra entries");
  const auto& s = p.species[1];
  require(s.rho_init == 0.2 && s.npcelx == 3 && s.npcely == 5 &&
              s.npcelz == 7 && s.uth == 0.7 && s.vth == 0.9 && s.wth == 1.2 &&
              s.u0 == 1.4 && s.v0 == 1.6 && s.w0 == 1.8, "Full species assignment");
  require(p.species[0].center_initial_velocity && !s.center_initial_velocity,
          "Boolean centering and absent-tail fallback");
  require(!input.with("CenterInitialVelocity =\n").species[0].center_initial_velocity,
          "Empty optional list");
  for (const auto key : {"rhoINIT", "qom", "npcelx", "npcely", "npcelz",
                         "uth", "vth", "wth", "u0", "v0", "w0"}) {
    expect_error([&] { input.with(std::string(key) + " =\n"); },
                 "line 16: " + std::string(key) + " lists 0 values, expected at least 1");
    expect_error([&] { input.read(InputFixture::base(2) + key + " = 1\n"); },
                 std::string(key) + " lists 1 values, expected at least 2");
    expect_error([&] { input.read(InputFixture::base(2) + key + " = 1 nope 2\n"); },
                 "'nope' is not a");
  }
  expect_error([&] { input.with("qom = [-1]\n"); }, "'[-1]' is not a number");
  expect_error([&] { input.with("qom = -2suffix\n"); }, "'-2suffix' is not a number");
  expect_error([&] { input.with("uth = 3,\n"); }, "'3,' is not a number");
  require(input.with("CenterInitialVelocity = 1\n").species[0].center_initial_velocity &&
              !input.with("CenterInitialVelocity = 0\n").species[0].center_initial_velocity,
          "A list boolean is 0 or 1");
  expect_error([&] { input.with("CenterInitialVelocity = -3\n"); },
               "'-3' is not a boolean (0, 1, true or false)");
}

void test_species(const InputFixture& input) {
  const std::string extras =
      "nsTestPart = 2\nqom = -1 2 -3\nnpcelx = 1 2 3\nnpcely = 4 5 6\nnpcelz = 7 8 9\n";
  const auto p = input.with(extras);
  require(p.species.size() == 1 && p.test_species.size() == 2 &&
              p.test_species[0].q_over_m == 2 && p.test_species[1].q_over_m == -3 &&
              p.test_species[1].npcelx == 3 && p.test_species[1].npcely == 6 &&
              p.test_species[1].npcelz == 9, "Separate test-species block uses offset");
  const auto& s = p.test_species[0];
  require(s.rho_init == 0 && s.uth == 0 && s.vth == 0 && s.wth == 0 &&
              s.u0 == 0 && s.v0 == 0 && s.w0 == 0 && !s.center_initial_velocity,
          "Optional test-species tails default to zero");
  const auto tails = input.with(extras + "rhoINIT = 1 2 3\nuth = 4 5 6\n"
                                "CenterInitialVelocity = false true false\n");
  require(tails.test_species[0].rho_init == 2 && tails.test_species[1].uth == 6 &&
              tails.test_species[0].center_initial_velocity, "Present test-species tails are used");
  for (const auto key : {"qom", "npcelx", "npcely", "npcelz"}) {
    expect_error([&] { input.with(extras + key + " = 1 2\n"); },
                 std::string(key) + " lists 2 values, expected at least 3");
  }
}

void species_safety(const InputFixture& input) {
  for (int count = tpic::config::min_real_species; count <= tpic::config::max_real_species; ++count) {
    const auto p = input.read(InputFixture::base(count));
    require(p.ns == count && p.species.size() == static_cast<std::size_t>(count) &&
                p.species.back().q_over_m == count, "All 1-10 real-species counts are parsed");
  }
  for (const auto value : {"0", "-1", "-2147483648", "11", "2147483647"}) {
    expect_error([&] { input.with("ns = " + std::string(value) + "\n"); },
                 "supported real species range: 1-10");
  }
  for (const auto value : {"broken", "", "2147483648", "-2147483649"}) {
    expect_error([&] { input.with("ns = " + std::string(value) + "\n"); }, "is not an integer");
  }
  for (const auto value : {"-1", "-2147483648"}) {
    expect_error([&] { input.with("nsTestPart = " + std::string(value) + "\n"); },
                 "nsTestPart must be nonnegative");
  }
  expect_error([&] { input.with("nsTestPart = broken\n"); }, "is not an integer");
  expect_error([&] { input.with("nsTestPart = 2147483648\n"); }, "is not an integer");
  const auto int_max = std::to_string(std::numeric_limits<int>::max());
  expect_error([&] { input.with("nsTestPart = " + int_max + "\n"); },
               "ns + nsTestPart exceeds the supported integer range");
  // A representable but huge total must fail the required-list check before a
  // vector allocation. This probes both sides of the sum-overflow boundary.
  expect_error([&] {
    input.with("nsTestPart = " + std::to_string(std::numeric_limits<int>::max() - 1) + "\n");
  }, "qom lists 1 values, expected at least " + int_max);
}

} // namespace

int main() {
  try {
    const InputFixture input;
    defaults(input);
    lexical_grammar(input);
    removed_keys(input);
    cases(input);
    output_methods(input);
    booleans(input);
    dynamic_defaults(input);
    enum_grammar(input);
    lists_and_species(input);
    test_species(input);
    species_safety(input);
    std::cout << "Simulation parameters: grammar, rejections, defaults, species and safety checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Simulation parameters test failed: " << error.what() << '\n';
    return 1;
  }
}
