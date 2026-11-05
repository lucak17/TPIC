// Golden resolved parameters: every SimulationParameters member but
// unknown_keys after parsing each deck in input/ and performance/scaling/ and each
// synthetic deck of tests/data/synthetic_decks.txt (grammar, canonical values, rejections,
// output channels, 1-10 species). The table is tests/data/golden/resolved_parameters.txt.
#include <tpic/config/simulation_parameters.hpp>

#include "../support/golden_data.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace golden = tpic::test::golden;
using tpic::config::SimulationParameters;
using tpic::config::SpeciesConfig;

template <class T> std::string text(const T& value) {
  if constexpr (std::is_same_v<T, bool>) return value ? "1" : "0";
  else if constexpr (std::is_enum_v<T>) return std::to_string(static_cast<long long>(value));
  else if constexpr (std::is_same_v<T, double>) return golden::hex(value);
  else if constexpr (std::is_same_v<T, std::string>) return golden::quoted(value);
  else return std::to_string(value);
}

class Dump {
public:
  explicit Dump(golden::Lines& lines) : lines_(lines) {}
  template <class T> void operator()(const std::string& label, const T& value) {
    lines_.push_back(label + ' ' + text(value));
  }

private:
  golden::Lines& lines_;
};

void dump_species(Dump& put, const std::string& list, const std::vector<SpeciesConfig>& species) {
  put(list + ".size", species.size());
  for (std::size_t index = 0; index < species.size(); ++index) {
    const auto& s = species[index];
    const auto prefix = list + '[' + std::to_string(index) + "].";
    put(prefix + "rho_init", s.rho_init);
    put(prefix + "uth", s.uth);
    put(prefix + "vth", s.vth);
    put(prefix + "wth", s.wth);
    put(prefix + "u0", s.u0);
    put(prefix + "v0", s.v0);
    put(prefix + "w0", s.w0);
    put(prefix + "q_over_m", s.q_over_m);
    put(prefix + "npcelx", s.npcelx);
    put(prefix + "npcely", s.npcely);
    put(prefix + "npcelz", s.npcelz);
    put(prefix + "center_initial_velocity", s.center_initial_velocity);
  }
}

void dump(const SimulationParameters& p, golden::Lines& lines) {
  Dump put(lines);
  put("dt", p.dt);
  put("ncycles", p.ncycles);
  put("th", p.th);
  put("smooth", p.smooth);
  put("smooth_niter", p.smooth_niter);
  put("ecsim_smoothing_mode", p.ecsim_smoothing_mode);
  put("ecsim_smoothing_operator", p.ecsim_smoothing_operator);
  put("c", p.c);
  put("b0x", p.b0x);
  put("b0y", p.b0y);
  put("b0z", p.b0z);
  put("delta", p.delta);
  put("harris_b0x", p.harris_b0x);
  put("pert_gem", p.pert_gem);
  put("pert_hump", p.pert_hump);
  put("deltax_hump", p.deltax_hump);
  put("deltay_hump", p.deltay_hump);
  put("kx_hump", p.kx_hump);
  put("ky_hump", p.ky_hump);
  put("current_from_ampere", p.current_from_ampere);
  put("spatially_varying_thermal", p.spatially_varying_thermal);
  put("forcefree_b0x", p.forcefree_b0x);
  put("forcefree_b0z", p.forcefree_b0z);
  put("weibel_seed_mode_x", p.weibel_seed_mode_x);
  put("weibel_seed_amplitude", p.weibel_seed_amplitude);
  put("weibel_seed_growth_rate", p.weibel_seed_growth_rate);
  put("two_stream_seed_mode_x", p.two_stream_seed_mode_x);
  put("two_stream_seed_density_amplitude", p.two_stream_seed_density_amplitude);
  put("two_stream_seed_phase", p.two_stream_seed_phase);
  put("whistler_mode_x", p.whistler_mode_x);
  put("whistler_mode_y", p.whistler_mode_y);
  put("whistler_mode_z", p.whistler_mode_z);
  put("whistler_seed_epsilon", p.whistler_seed_epsilon);
  put("whistler_phase", p.whistler_phase);
  put("whistler_use_exact_cold_root", p.whistler_use_exact_cold_root);
  put("whistler_seed_mode", p.whistler_seed_mode);
  put("whistler_noise_seed", p.whistler_noise_seed);
  put("whistler_noise_mode_max", p.whistler_noise_mode_max);
  put("whistler_field_time_offset", p.whistler_field_time_offset);
  put("whistler_particle_time_offset", p.whistler_particle_time_offset);
  put("taylor_green_vortex_mode_x", p.taylor_green_vortex_mode_x);
  put("taylor_green_vortex_mode_y", p.taylor_green_vortex_mode_y);
  put("taylor_green_vortex_mode_z", p.taylor_green_vortex_mode_z);
  put("taylor_green_vortex_velocity_amplitude", p.taylor_green_vortex_velocity_amplitude);
  put("taylor_green_vortex_magnetic_amplitude", p.taylor_green_vortex_magnetic_amplitude);
  put("taylor_green_vortex_seed_motional_e", p.taylor_green_vortex_seed_motional_e);
  put("flux_rope_b0", p.flux_rope_b0);
  put("flux_rope_width", p.flux_rope_width);
  put("flux_rope_center_x", p.flux_rope_center_x);
  put("flux_rope_center_y", p.flux_rope_center_y);
  put("flux_rope_pressure_floor", p.flux_rope_pressure_floor);
  put("flux_rope_electron_pressure_fraction", p.flux_rope_electron_pressure_fraction);
  put("gaussian_expansion_center_x", p.gaussian_expansion_center_x);
  put("gaussian_expansion_center_y", p.gaussian_expansion_center_y);
  put("gaussian_expansion_center_z", p.gaussian_expansion_center_z);
  put("gaussian_expansion_sigma_x", p.gaussian_expansion_sigma_x);
  put("gaussian_expansion_sigma_y", p.gaussian_expansion_sigma_y);
  put("gaussian_expansion_sigma_z", p.gaussian_expansion_sigma_z);
  put("gaussian_expansion_cutoff", p.gaussian_expansion_cutoff);
  put("whistler_turbulence_epsilon", p.whistler_turbulence_epsilon);
  put("whistler_turbulence_seed", p.whistler_turbulence_seed);
  put("grid.lx", p.grid.lx);
  put("grid.ly", p.grid.ly);
  put("grid.lz", p.grid.lz);
  put("grid.nxc", p.grid.nxc);
  put("grid.nyc", p.grid.nyc);
  put("grid.nzc", p.grid.nzc);
  put("topology.xlen", p.topology.xlen);
  put("topology.ylen", p.topology.ylen);
  put("topology.zlen", p.topology.zlen);
  put("topology.periodic_x", p.topology.periodic_x);
  put("topology.periodic_y", p.topology.periodic_y);
  put("topology.periodic_z", p.topology.periodic_z);
  put("topology.periodic_x_particles", p.topology.periodic_x_particles);
  put("topology.periodic_y_particles", p.topology.periodic_y_particles);
  put("topology.periodic_z_particles", p.topology.periodic_z_particles);
  put("ns", p.ns);
  put("ns_test_particles", p.ns_test_particles);
  dump_species(put, "species", p.species);
  dump_species(put, "test_species", p.test_species);
  put("gmres_tol", p.gmres_tol);
  put("save_dir", p.save_dir);
  put("restart_dir", p.restart_dir);
  put("case_name", p.case_name);
  put("quiet_start", p.quiet_start);
  put("write_method", p.write_method);
  put("field_write_method", p.field_write_method);
  put("particles_write_method", p.particles_write_method);
  put("restart_write_method", p.restart_write_method);
  put("sim_name", p.sim_name);
  put("output.field_cycle", p.output.field_cycle);
  put("output.particles_cycle", p.output.particles_cycle);
  put("output.test_particles_cycle", p.output.test_particles_cycle);
  put("output.restart_cycle", p.output.restart_cycle);
  put("output.call_finalize", p.output.call_finalize);
  put("output.diagnostics_cycle", p.output.diagnostics_cycle);
  put("output.diagnostic_energy_cycle", p.output.diagnostic_energy_cycle);
  put("output.gauss_cycle", p.output.gauss_cycle);
  put("output.field_tag", p.output.field_tag);
  put("output.particles_tag", p.output.particles_tag);
  put("output.moments_tag", p.output.moments_tag);
  put("bc.em_x_right", p.bc.em_x_right);
  put("bc.em_x_left", p.bc.em_x_left);
  put("bc.em_y_right", p.bc.em_y_right);
  put("bc.em_y_left", p.bc.em_y_left);
  put("bc.em_z_right", p.bc.em_z_right);
  put("bc.em_z_left", p.bc.em_z_left);
  put("bc.particle_x_right", p.bc.particle_x_right);
  put("bc.particle_x_left", p.bc.particle_x_left);
  put("bc.particle_y_right", p.bc.particle_y_right);
  put("bc.particle_y_left", p.bc.particle_y_left);
  put("bc.particle_z_right", p.bc.particle_z_right);
  put("bc.particle_z_left", p.bc.particle_z_left);
}

std::optional<SimulationParameters> parse(std::istream& input) {
  try {
    return tpic::config::parse_simulation_parameters(input);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

void add_deck(const std::string& id, const std::optional<SimulationParameters>& parameters,
              golden::Lines& lines) {
  lines.push_back('[' + id + (parameters ? "] accepted" : "] rejected"));
  if (parameters) dump(*parameters, lines);
}

std::vector<std::pair<std::string, std::string>> synthetic_decks(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("missing " + path.string());
  std::vector<std::pair<std::string, std::string>> decks;
  for (std::string line; std::getline(input, line);) {
    if (line.starts_with("### deck ")) decks.emplace_back(line.substr(9), std::string{});
    else if (!decks.empty()) decks.back().second += line + '\n';
  }
  if (decks.empty()) throw std::runtime_error("no synthetic decks in " + path.string());
  return decks;
}

// Hand-derived: a known channel method is kept as written, trimmed (last
// assignment wins), and an empty one stays empty.
struct OutputChannels {
  std::string_view deck, field, particles;
};
constexpr std::array<OutputChannels, 5> output_channels{{
    {"output_channels", "vtk", "adios2"},
    {"output_fallback", "", "default"},
    {"output_fields_only", "adios2", ""},
    {"output_particles_only", "", "hdf5"},
    {"output_grammar", "hdf5", "h5"}}};

void check_synthetic(const std::string& name, const std::optional<SimulationParameters>& parameters) {
  if (name.starts_with("bad_") == parameters.has_value())
    throw std::runtime_error("synthetic deck " + name + (parameters ? " was accepted" : " was rejected"));
  for (const auto& expected : output_channels)
    if (name == expected.deck && (parameters->field_write_method != expected.field ||
                                  parameters->particles_write_method != expected.particles))
      throw std::runtime_error("synthetic deck " + name + " resolved unexpected output channels");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = golden::parse_options(argc, argv);
    golden::Lines lines;
    for (const auto& deck : golden::deck_files(options.source_root)) {
      std::ifstream input(options.source_root / deck);
      if (!input) throw std::runtime_error("cannot open " + deck.generic_string());
      add_deck(deck.generic_string(), parse(input), lines);
    }
    for (const auto& [name, text] : synthetic_decks(options.source_root / "tests/data/synthetic_decks.txt")) {
      std::istringstream input(text);
      const auto parameters = parse(input);
      check_synthetic(name, parameters);
      add_deck("synthetic/" + name, parameters, lines);
    }
    golden::check(options, "resolved_parameters.txt", lines);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
