#ifndef TPIC_CONFIG_SIMULATION_PARAMETERS_HPP
#define TPIC_CONFIG_SIMULATION_PARAMETERS_HPP

// Runtime simulation parameters and the input-file parser that fills them.

#include <tpic/config/limits.hpp>

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace tpic::config {

// Resolved host values deliberately use double independently of device storage
// precision. Casting to an admitted precision profile belongs to kernel setup.

enum class EcsimSmoothingMode { None = 0, EnergyConserving };

enum class EcsimSmoothingOperator { Face7 = 0, Binomial27 };

enum class WhistlerSeedMode { Eigenmode = 0, MagneticOnly, Noise };

struct SpeciesConfig {
  double rho_init = 0.0;
  double uth = 0.0;
  double vth = 0.0;
  double wth = 0.0;
  double u0 = 0.0;
  double v0 = 0.0;
  double w0 = 0.0;
  double q_over_m = 0.0;
  int npcelx = 0;
  int npcely = 0;
  int npcelz = 0;
  bool center_initial_velocity = false;
};

struct BoundaryConditions {
  int em_x_right = 0;
  int em_x_left = 0;
  int em_y_right = 0;
  int em_y_left = 0;
  int em_z_right = 0;
  int em_z_left = 0;

  int particle_x_right = 1;
  int particle_x_left = 1;
  int particle_y_right = 1;
  int particle_y_left = 1;
  int particle_z_right = 1;
  int particle_z_left = 1;
};

struct TopologyConfig {
  int xlen = 1;
  int ylen = 1;
  int zlen = 1;
  bool periodic_x = true;
  bool periodic_y = true;
  bool periodic_z = true;
  bool periodic_x_particles = true;
  bool periodic_y_particles = true;
  bool periodic_z_particles = true;
};

struct GridConfig {
  double lx = 10.0;
  double ly = 10.0;
  double lz = 10.0;
  int nxc = 64;
  int nyc = 64;
  int nzc = 64;
};

struct OutputConfig {
  int field_cycle = 100;
  int particles_cycle = 0;
  int test_particles_cycle = 0;
  // Checkpoint cadence (RestartOutputCycle); CallFinalize adds the final state.
  int restart_cycle = 0;
  bool call_finalize = true;
  int diagnostics_cycle = 100;
  // Console energy cadence (DiagnosticEnergyCycle); the final state always reports.
  int diagnostic_energy_cycle = 1;
  // Gauss-law residual cadence (DiagnosticGaussCycle); nonpositive disables.
  int gauss_cycle = 0;
  std::string field_tag;
  std::string particles_tag;
  std::string moments_tag;
};

struct SimulationParameters {
  // Timing
  double dt = 0.0;
  int ncycles = 0;
  double th = 1.0;

  // Smoothing
  double smooth = 1.0;
  int smooth_niter = 6;
  EcsimSmoothingMode ecsim_smoothing_mode = EcsimSmoothingMode::None;
  EcsimSmoothingOperator ecsim_smoothing_operator = EcsimSmoothingOperator::Face7;

  // Physical constants / fields
  double c = 1.0;
  double b0x = 0.0, b0y = 0.0, b0z = 0.0;
  double delta = 0.5;
  double harris_b0x = 0.0;
  // The seed and the loading options of the GEMHarris case.
  double pert_gem = 0.0;
  double pert_hump = 0.0;
  double deltax_hump = 8.0;
  double deltay_hump = 4.0;
  double kx_hump = -1.0;
  double ky_hump = -1.0;
  bool current_from_ampere = false;
  // No physical effect: it returns the deck wth to rounding (initialization::HarrisProfile).
  bool spatially_varying_thermal = false;
  double forcefree_b0x = 0.0;
  double forcefree_b0z = 0.0;
  int weibel_seed_mode_x = 0;
  double weibel_seed_amplitude = 0.0;
  double weibel_seed_growth_rate = 0.0;
  int two_stream_seed_mode_x = 0;
  double two_stream_seed_density_amplitude = 0.0;
  double two_stream_seed_phase = 0.0;
  int whistler_mode_x = 0;
  int whistler_mode_y = 0;
  int whistler_mode_z = 0;
  double whistler_seed_epsilon = 0.0;
  double whistler_phase = 0.0;
  bool whistler_use_exact_cold_root = true;
  WhistlerSeedMode whistler_seed_mode = WhistlerSeedMode::Eigenmode;
  std::uint64_t whistler_noise_seed = 12345ULL;
  int whistler_noise_mode_max = 0;
  double whistler_field_time_offset = 0.0;
  double whistler_particle_time_offset = 0.0;
  int taylor_green_vortex_mode_x = 1;
  int taylor_green_vortex_mode_y = 1;
  int taylor_green_vortex_mode_z = 1;
  double taylor_green_vortex_velocity_amplitude = 0.0;
  double taylor_green_vortex_magnetic_amplitude = 0.0;
  bool taylor_green_vortex_seed_motional_e = false;
  double flux_rope_b0 = 0.1;
  double flux_rope_width = 1.0;
  double flux_rope_center_x = 0.0;
  double flux_rope_center_y = 0.0;
  double flux_rope_pressure_floor = 9.0e-4;
  double flux_rope_electron_pressure_fraction = 0.1;
  // GaussianExpansion: the centre (the box centre when omitted), the rms width
  // per axis (0: uniform) and the loading cutoff in widths (0: every cell).
  double gaussian_expansion_center_x = 0.0;
  double gaussian_expansion_center_y = 0.0;
  double gaussian_expansion_center_z = 0.0;
  double gaussian_expansion_sigma_x = 0.0;
  double gaussian_expansion_sigma_y = 0.0;
  double gaussian_expansion_sigma_z = 0.0;
  double gaussian_expansion_cutoff = 0.0;
  // WhistlerTurbulence: <|dB|^2>/B0^2 of the seed and the seed of its phases.
  double whistler_turbulence_epsilon = 0.0;
  std::uint64_t whistler_turbulence_seed = 1ULL;

  // Domain geometry
  GridConfig grid;
  TopologyConfig topology;

  // Species
  int ns = 0;
  int ns_test_particles = 0;
  std::vector<SpeciesConfig> species;
  std::vector<SpeciesConfig> test_species;

  // Field solver
  double gmres_tol = 1e-3;

  // I/O and bookkeeping
  std::string save_dir = "data";
  std::string restart_dir = "data";
  // One of case_names; the input parser requires the key Case.
  std::string case_name = "Uniform";
  // QuietStart: every particle of every case at the centre of its subcell, the
  // same points for every species; otherwise a random point in each subcell.
  bool quiet_start = false;
  // WriteMethod is the fallback of the field and particle methods; a nonempty
  // channel method overrides it. RestartWriteMethod has no fallback.
  std::string write_method;
  std::string field_write_method;
  std::string particles_write_method;
  std::string restart_write_method;
  std::string sim_name;
  OutputConfig output;

  BoundaryConditions bc;

  // Deck keys the parser does not read, in deck order with their lines.
  std::vector<std::string> unknown_keys;
};

// The initial-state cases an input deck names with the key Case; each is
// implemented by one case of initialization::InitialCase.
inline constexpr auto case_names = std::to_array<std::string_view>({
    "Uniform", "GEM", "GEMnoPert", "GEMHarris", "DoubleGEM", "DoubleGEMnoPert", "plasmoidChain",
    "ForceFree", "Weibel", "Whistler", "TaylorGreenVortex", "TwoStream", "fluxRope", "GaussianExpansion",
    "WhistlerTurbulence"});

// Input grammar: one 'key = value' per line; '#' starts a comment; the last
// assignment of a key wins. Every value must be read in full (a boolean is 0,
// 1, true or false, a list holds whitespace-separated values) and Case must
// name one of case_names. A violation throws std::invalid_argument naming the
// deck line and, for a value, its key. An unknown key is ignored and listed in unknown_keys.
// Resolve the defaults (including values derived from other input keys) and
// check the species counts before allocation. Recipe, geometry, precision and
// unsupported-feature checks belong to run admission.
SimulationParameters load_simulation_parameters(std::string_view path);
// Parse a captured input snapshot, e.g. after MPI startup has agreed its bytes.
SimulationParameters parse_simulation_parameters(std::istream& input);

} // namespace tpic::config

#endif // TPIC_CONFIG_SIMULATION_PARAMETERS_HPP
