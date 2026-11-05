// Parses key = value input files into SimulationParameters.
#include <tpic/config/simulation_parameters.hpp>

#include <tpic/config/detail/errors.hpp>
#include <tpic/config/output.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tpic::config {

namespace {

std::string trimmed(std::string_view value) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  const auto first = std::find_if(value.begin(), value.end(), not_space);
  const auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
  return first < last ? std::string(first, last) : std::string();
}

constexpr detail::Errors errors{"input"};

template <class Value> constexpr std::string_view value_kind() {
  if constexpr (std::is_same_v<Value, bool>) return "a boolean (0, 1, true or false)";
  else if constexpr (std::is_floating_point_v<Value>) return "a number";
  else return "an integer";
}

// One value, read in full: no trailing text after a number, and a boolean is
// spelled 0, 1, true or false.
template <class Value> bool parse_token(std::string_view text, Value& out) {
  if constexpr (std::is_same_v<Value, bool>) {
    out = text == "1" || text == "true";
    return out || text == "0" || text == "false";
  } else {
    std::istringstream stream{std::string(text)};
    stream >> out;
    return !stream.fail() && (stream >> std::ws).eof();
  }
}

// The deck as a map from key to its last assignment. Every read marks its key,
// so that the keys nobody read are the unknown ones.
class InputDeck {
public:
  explicit InputDeck(std::istream& in) {
    std::string line;
    for (int number = 1; std::getline(in, line); ++number) {
      const auto content = trimmed(std::string_view(line).substr(0, line.find('#')));
      if (content.empty()) continue;
      const auto delimiter = content.find('=');
      auto key = trimmed(std::string_view(content).substr(0, delimiter));
      if (delimiter == std::string::npos || key.empty())
        errors.invalid("line " + std::to_string(number) + ": expected 'key = value'");
      entries_[std::move(key)] = {trimmed(std::string_view(content).substr(delimiter + 1)), number};
    }
  }

  template <class Value> void read_into(const std::string& key, Value& target) {
    if (const auto* entry = find(key)) target = parse<Value>(key, *entry);
  }

  template <class Value> Value read_optional(const std::string& key, Value fallback) {
    read_into(key, fallback);
    return fallback;
  }

  template <class Value> Value read_required(const std::string& key) {
    const auto* entry = find(key);
    if (entry == nullptr) errors.invalid("missing required key " + key);
    return parse<Value>(key, *entry);
  }

  // A whitespace-separated list of at least required values; extra values are
  // ignored. An absent key is an empty list, allowed only when nothing is required.
  template <class Value>
  std::vector<Value> read_list(const std::string& key, std::size_t required) {
    const auto* entry = find(key);
    if (entry == nullptr) {
      if (required != 0) errors.invalid("missing required key " + key);
      return {};
    }
    std::vector<Value> values;
    std::istringstream stream(entry->value);
    for (std::string token; stream >> token;) {
      Value value{};
      if (!parse_token(token, value))
        errors.invalid(at(*entry) + key + " = '" + entry->value + "': '" + token + "' is not " +
               std::string(value_kind<Value>()));
      values.push_back(value);
    }
    if (values.size() < required)
      errors.invalid(at(*entry) + key + " lists " + std::to_string(values.size()) +
             " values, expected at least " + std::to_string(required));
    return values;
  }

  // A value from a closed set of spellings.
  template <class Enum, std::size_t N>
  void read_into(const std::string& key, Enum& target,
                 const std::array<std::pair<std::string_view, Enum>, N>& choices) {
    const auto* entry = find(key);
    if (entry == nullptr) return;
    for (const auto& [name, value] : choices)
      if (entry->value == name) {
        target = value;
        return;
      }
    reject_choice(key, *entry, choices);
  }

  // An output-method value: empty (the fallback) or an engine name of
  // parse_output_engine.
  void read_output_method(const std::string& key, std::string& target) {
    const auto* entry = find(key);
    if (entry == nullptr) return;
    if (!entry->value.empty())
      (void)parse_output_engine(key, entry->value, OutputEngineKind::Default, errors.text(at(*entry)));
    target = entry->value;
  }

  void require_case_name(const std::string& key, const std::string& name) const {
    if (std::find(case_names.begin(), case_names.end(), name) == case_names.end()) {
      std::array<std::pair<std::string_view, int>, case_names.size()> choices{};
      for (std::size_t index = 0; index < case_names.size(); ++index) choices[index] = {case_names[index], 0};
      reject_choice(key, entries_.at(key), choices);
    }
  }

  // The keys no read asked for, in deck order, each with its line; they are
  // reported and ignored.
  std::vector<std::string> unknown_keys() const {
    std::vector<std::pair<int, std::string>> unread;
    for (const auto& [key, entry] : entries_)
      if (!entry.read) unread.emplace_back(entry.line, key);
    std::sort(unread.begin(), unread.end());
    std::vector<std::string> keys;
    for (const auto& [line, key] : unread) keys.push_back(key + " (line " + std::to_string(line) + ")");
    return keys;
  }

private:
  struct Entry {
    std::string value;
    int line = 0;
    bool read = false;
  };

  const Entry* find(const std::string& key) {
    const auto it = entries_.find(key);
    if (it == entries_.end()) return nullptr;
    it->second.read = true;
    return &it->second;
  }

  static std::string at(const Entry& entry) { return "line " + std::to_string(entry.line) + ": "; }

  template <class Value> static Value parse(const std::string& key, const Entry& entry) {
    if constexpr (std::is_same_v<Value, std::string>) {
      return entry.value;
    } else {
      Value value{};
      if (!parse_token(entry.value, value))
        errors.invalid(at(entry) + key + " = '" + entry.value + "' is not " + std::string(value_kind<Value>()));
      return value;
    }
  }

  template <class Choices>
  [[noreturn]] static void reject_choice(const std::string& key, const Entry& entry, const Choices& choices) {
    std::string expected;
    for (std::size_t index = 0; index < choices.size(); ++index)
      expected += (index == 0 ? "" : index + 1 == choices.size() ? " or " : ", ") +
                  std::string(choices[index].first);
    errors.invalid(at(entry) + key + " = '" + entry.value + "' is unknown (expected " + expected + ")");
  }

  std::unordered_map<std::string, Entry> entries_;
};

constexpr std::array<std::pair<std::string_view, EcsimSmoothingMode>, 2> smoothing_modes{{
    {"none", EcsimSmoothingMode::None}, {"energy_conserving", EcsimSmoothingMode::EnergyConserving}}};
constexpr std::array<std::pair<std::string_view, EcsimSmoothingOperator>, 2> smoothing_operators{{
    {"face7", EcsimSmoothingOperator::Face7}, {"binomial27", EcsimSmoothingOperator::Binomial27}}};
constexpr std::array<std::pair<std::string_view, WhistlerSeedMode>, 3> whistler_seed_modes{{
    {"eigenmode", WhistlerSeedMode::Eigenmode}, {"magnetic_only", WhistlerSeedMode::MagneticOnly},
    {"noise", WhistlerSeedMode::Noise}}};

} // namespace

SimulationParameters load_simulation_parameters(std::string_view path) {
  std::ifstream input{std::string(path)};
  if (!input) errors.invalid("unable to open input file " + std::string(path));
  return parse_simulation_parameters(input);
}

SimulationParameters parse_simulation_parameters(std::istream& input) {
  InputDeck cfg(input);
  SimulationParameters params;
  // Ordinary defaults belong to the records; derived defaults stay explicit.

  params.dt = cfg.read_required<double>("dt");
  params.ncycles = cfg.read_required<int>("ncycles");
  cfg.read_into("th", params.th);

  cfg.read_into("Smooth", params.smooth);
  cfg.read_into("SmoothNiter", params.smooth_niter);
  cfg.read_into("ECSimSmoothing", params.ecsim_smoothing_mode, smoothing_modes);
  cfg.read_into("ECSimSmoothingOperator", params.ecsim_smoothing_operator, smoothing_operators);

  cfg.read_into("SaveDirName", params.save_dir);
  cfg.read_into("RestartDirName", params.restart_dir);
  params.ns = cfg.read_required<int>("ns");
  cfg.read_into("nsTestPart", params.ns_test_particles);
  // Validate signed counts before addition, unsigned conversion or allocation.
  // The real-species range is TPIC's support contract.
  require_species_count(params.ns);
  if (params.ns_test_particles < 0) errors.invalid("nsTestPart must be nonnegative");
  if (params.ns_test_particles > std::numeric_limits<int>::max() - params.ns)
    errors.invalid("ns + nsTestPart exceeds the supported integer range");
  const auto real_species = static_cast<std::size_t>(params.ns);
  const auto test_species = static_cast<std::size_t>(params.ns_test_particles);
  const auto total_species = real_species + test_species;

  cfg.read_into("B0x", params.b0x);
  cfg.read_into("B0y", params.b0y);
  cfg.read_into("B0z", params.b0z);
  cfg.read_into("delta", params.delta);
  cfg.read_into("HarrisB0x", params.harris_b0x);
  cfg.read_into("pertGEM", params.pert_gem);
  cfg.read_into("pertHump", params.pert_hump);
  cfg.read_into("deltaxHump", params.deltax_hump);
  cfg.read_into("deltayHump", params.deltay_hump);
  cfg.read_into("kxHump", params.kx_hump);
  cfg.read_into("kyHump", params.ky_hump);
  cfg.read_into("currentFromAmpere", params.current_from_ampere);
  cfg.read_into("spatiallyVaryingThermal", params.spatially_varying_thermal);
  cfg.read_into("ForceFreeB0x", params.forcefree_b0x);
  cfg.read_into("ForceFreeB0z", params.forcefree_b0z);
  cfg.read_into("FluxRopeB0", params.flux_rope_b0);
  cfg.read_into("FluxRopeWidth", params.flux_rope_width);
  cfg.read_into("FluxRopePressureFloor", params.flux_rope_pressure_floor);
  cfg.read_into("FluxRopeElectronPressureFraction", params.flux_rope_electron_pressure_fraction);
  cfg.read_into("WeibelSeedModeX", params.weibel_seed_mode_x);
  cfg.read_into("WeibelSeedAmplitude", params.weibel_seed_amplitude);
  cfg.read_into("WeibelSeedGrowthRate", params.weibel_seed_growth_rate);
  cfg.read_into("TwoStreamSeedModeX", params.two_stream_seed_mode_x);
  cfg.read_into("TwoStreamSeedDensityAmplitude", params.two_stream_seed_density_amplitude);
  cfg.read_into("TwoStreamSeedPhase", params.two_stream_seed_phase);
  cfg.read_into("WhistlerModeX", params.whistler_mode_x);
  cfg.read_into("WhistlerModeY", params.whistler_mode_y);
  cfg.read_into("WhistlerModeZ", params.whistler_mode_z);
  cfg.read_into("WhistlerSeedEpsilon", params.whistler_seed_epsilon);
  cfg.read_into("WhistlerPhase", params.whistler_phase);
  cfg.read_into("WhistlerUseExactColdRoot", params.whistler_use_exact_cold_root);
  cfg.read_into("WhistlerSeedMode", params.whistler_seed_mode, whistler_seed_modes);
  cfg.read_into("WhistlerNoiseSeed", params.whistler_noise_seed);
  cfg.read_into("WhistlerNoiseModeMax", params.whistler_noise_mode_max);
  cfg.read_into("WhistlerFieldTimeOffset", params.whistler_field_time_offset);
  cfg.read_into("WhistlerParticleTimeOffset", params.whistler_particle_time_offset);
  // TaylorGreenVortexMode is the default of the three per-axis modes.
  const int tgv_mode = cfg.read_optional<int>("TaylorGreenVortexMode", 1);
  params.taylor_green_vortex_mode_x = cfg.read_optional<int>("TaylorGreenVortexModeX", tgv_mode);
  params.taylor_green_vortex_mode_y = cfg.read_optional<int>("TaylorGreenVortexModeY", tgv_mode);
  params.taylor_green_vortex_mode_z = cfg.read_optional<int>("TaylorGreenVortexModeZ", tgv_mode);
  cfg.read_into("TaylorGreenVortexVelocityAmplitude", params.taylor_green_vortex_velocity_amplitude);
  cfg.read_into("TaylorGreenVortexMagneticAmplitude", params.taylor_green_vortex_magnetic_amplitude);
  cfg.read_into("TaylorGreenVortexSeedMotionalE", params.taylor_green_vortex_seed_motional_e);
  cfg.read_into("GaussianExpansionSigmaX", params.gaussian_expansion_sigma_x);
  cfg.read_into("GaussianExpansionSigmaY", params.gaussian_expansion_sigma_y);
  cfg.read_into("GaussianExpansionSigmaZ", params.gaussian_expansion_sigma_z);
  cfg.read_into("GaussianExpansionCutoff", params.gaussian_expansion_cutoff);
  cfg.read_into("WhistlerTurbulenceEpsilon", params.whistler_turbulence_epsilon);
  cfg.read_into("WhistlerTurbulenceSeed", params.whistler_turbulence_seed);

  cfg.read_into("c", params.c);

  cfg.read_into("Lx", params.grid.lx);
  cfg.read_into("Ly", params.grid.ly);
  cfg.read_into("Lz", params.grid.lz);
  cfg.read_into("nxc", params.grid.nxc);
  cfg.read_into("nyc", params.grid.nyc);
  cfg.read_into("nzc", params.grid.nzc);
  params.flux_rope_center_x =
      cfg.read_optional<double>("FluxRopeCenterX", 0.5 * params.grid.lx);
  params.flux_rope_center_y =
      cfg.read_optional<double>("FluxRopeCenterY", 0.5 * params.grid.ly);
  params.gaussian_expansion_center_x =
      cfg.read_optional<double>("GaussianExpansionCenterX", 0.5 * params.grid.lx);
  params.gaussian_expansion_center_y =
      cfg.read_optional<double>("GaussianExpansionCenterY", 0.5 * params.grid.ly);
  params.gaussian_expansion_center_z =
      cfg.read_optional<double>("GaussianExpansionCenterZ", 0.5 * params.grid.lz);

  cfg.read_into("XLEN", params.topology.xlen);
  cfg.read_into("YLEN", params.topology.ylen);
  cfg.read_into("ZLEN", params.topology.zlen);
  cfg.read_into("PERIODICX", params.topology.periodic_x);
  cfg.read_into("PERIODICY", params.topology.periodic_y);
  cfg.read_into("PERIODICZ", params.topology.periodic_z);
  params.topology.periodic_x_particles = cfg.read_optional<bool>("PERIODICX_P", params.topology.periodic_x);
  params.topology.periodic_y_particles = cfg.read_optional<bool>("PERIODICY_P", params.topology.periodic_y);
  params.topology.periodic_z_particles = cfg.read_optional<bool>("PERIODICZ_P", params.topology.periodic_z);

  params.case_name = cfg.read_required<std::string>("Case");
  cfg.require_case_name("Case", params.case_name);
  cfg.read_into("QuietStart", params.quiet_start);
  cfg.read_output_method("WriteMethod", params.write_method);
  cfg.read_output_method("FieldWriteMethod", params.field_write_method);
  cfg.read_output_method("ParticlesWriteMethod", params.particles_write_method);
  cfg.read_output_method("RestartWriteMethod", params.restart_write_method);
  cfg.read_into("SimulationName", params.sim_name);

  cfg.read_into("GMREStol", params.gmres_tol);

  cfg.read_into("FieldOutputCycle", params.output.field_cycle);
  cfg.read_into("ParticlesOutputCycle", params.output.particles_cycle);
  cfg.read_into("TestParticlesOutputCycle", params.output.test_particles_cycle);
  cfg.read_into("RestartOutputCycle", params.output.restart_cycle);
  cfg.read_into("CallFinalize", params.output.call_finalize);
  params.output.diagnostics_cycle = cfg.read_optional<int>("DiagnosticsOutputCycle", params.output.field_cycle);
  cfg.read_into("DiagnosticEnergyCycle", params.output.diagnostic_energy_cycle);
  cfg.read_into("DiagnosticGaussCycle", params.output.gauss_cycle);
  cfg.read_into("FieldOutputTag", params.output.field_tag);
  cfg.read_into("ParticlesOutputTag", params.output.particles_tag);
  cfg.read_into("MomentsOutputTag", params.output.moments_tag);

  cfg.read_into("bcEMfaceXright", params.bc.em_x_right);
  cfg.read_into("bcEMfaceXleft", params.bc.em_x_left);
  cfg.read_into("bcEMfaceYright", params.bc.em_y_right);
  cfg.read_into("bcEMfaceYleft", params.bc.em_y_left);
  cfg.read_into("bcEMfaceZright", params.bc.em_z_right);
  cfg.read_into("bcEMfaceZleft", params.bc.em_z_left);

  cfg.read_into("bcPfaceXright", params.bc.particle_x_right);
  cfg.read_into("bcPfaceXleft", params.bc.particle_x_left);
  cfg.read_into("bcPfaceYright", params.bc.particle_y_right);
  cfg.read_into("bcPfaceYleft", params.bc.particle_y_left);
  cfg.read_into("bcPfaceZright", params.bc.particle_z_right);
  cfg.read_into("bcPfaceZleft", params.bc.particle_z_left);

  // Every list is read and checked before the species vectors are sized, so an
  // oversized species count fails its list check instead of an allocation.
  const auto rho_init_values = cfg.read_list<double>("rhoINIT", real_species);
  const auto q_over_m_values = cfg.read_list<double>("qom", total_species);
  const auto npcelx_values = cfg.read_list<int>("npcelx", total_species);
  const auto npcely_values = cfg.read_list<int>("npcely", total_species);
  const auto npcelz_values = cfg.read_list<int>("npcelz", total_species);
  const auto uth_values = cfg.read_list<double>("uth", real_species);
  const auto vth_values = cfg.read_list<double>("vth", real_species);
  const auto wth_values = cfg.read_list<double>("wth", real_species);
  const auto u0_values = cfg.read_list<double>("u0", real_species);
  const auto v0_values = cfg.read_list<double>("v0", real_species);
  const auto w0_values = cfg.read_list<double>("w0", real_species);
  const auto center_initial_velocity_values = cfg.read_list<bool>("CenterInitialVelocity", 0);
  params.species.resize(real_species);
  params.test_species.resize(test_species);
  // Real species precede test species in every input list. Missing optional
  // entries retain SpeciesConfig defaults; extra entries are ignored.
  const auto assign = [&](auto member, const auto& values) {
    const auto count = std::min(values.size(), total_species);
    for (std::size_t index = 0; index < count; ++index) {
      auto& species = index < real_species ? params.species[index]
                                           : params.test_species[index - real_species];
      species.*member = values[index];
    }
  };
  assign(&SpeciesConfig::rho_init, rho_init_values);
  assign(&SpeciesConfig::q_over_m, q_over_m_values);
  assign(&SpeciesConfig::npcelx, npcelx_values);
  assign(&SpeciesConfig::npcely, npcely_values);
  assign(&SpeciesConfig::npcelz, npcelz_values);
  assign(&SpeciesConfig::uth, uth_values);
  assign(&SpeciesConfig::vth, vth_values);
  assign(&SpeciesConfig::wth, wth_values);
  assign(&SpeciesConfig::u0, u0_values);
  assign(&SpeciesConfig::v0, v0_values);
  assign(&SpeciesConfig::w0, w0_values);
  assign(&SpeciesConfig::center_initial_velocity, center_initial_velocity_values);

  params.unknown_keys = cfg.unknown_keys();
  return params;
}

} // namespace tpic::config
