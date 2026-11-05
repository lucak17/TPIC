#include <tpic/config/output.hpp>

#include <tpic/config/detail/errors.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace tpic::config {
namespace {

constexpr detail::Errors errors{"output"};

struct EngineSpelling {
  std::string_view name;
  OutputEngineKind engine;
};
// Every engine spelling besides "default", which names the caller's default.
constexpr std::array<EngineSpelling, 7> engine_spellings{{
    {"pvtk", OutputEngineKind::Vtk}, {"vtk", OutputEngineKind::Vtk},
    {"adios2", OutputEngineKind::Adios2}, {"bp", OutputEngineKind::Adios2},
    {"bp5", OutputEngineKind::Adios2}, {"hdf5", OutputEngineKind::Hdf5},
    {"h5", OutputEngineKind::Hdf5}}};

bool is_digit(char value) noexcept { return value >= '0' && value <= '9'; }

bool any(const std::vector<bool> &mask) noexcept {
  return std::find(mask.begin(), mask.end(), true) != mask.end();
}

[[noreturn]] void reject_token(std::string_view key, std::string_view token,
                               std::string_view expected) {
  errors.invalid(std::string(key) + " token '" + std::string(token) + "' is unknown (expected " +
                 std::string(expected) + ")");
}

// Calls visit with each '+'-separated token of tag; an empty tag has no token
// and an empty token is rejected.
template <class Visit>
void for_each_token(std::string_view key, std::string_view tag, Visit &&visit) {
  if (tag.empty()) return;
  for (std::size_t begin = 0;;) {
    const auto end = tag.find('+', begin);
    const auto token = tag.substr(begin, end == std::string_view::npos ? end : end - begin);
    if (token.empty())
      errors.invalid(std::string(key) + " = '" + std::string(tag) + "' has an empty token");
    visit(token);
    if (end == std::string_view::npos) return;
    begin = end + 1;
  }
}

// The value of a species index; beyond the signed-int range is an overflow.
std::size_t species_index(std::string_view name, std::string_view digits) {
  int index = 0;
  for (const char character : digits) {
    const int digit = character - '0';
    if (index > (std::numeric_limits<int>::max() - digit) / 10)
      errors.overflow(std::string(name) + " species index exceeds signed-int range");
    index = index * 10 + digit;
  }
  return static_cast<std::size_t>(index);
}

std::string moment_token_forms() {
  std::string names;
  for (std::size_t q = 0; q < output_quantity_count; ++q)
    names += (q == 0 ? "" : q + 1 == output_quantity_count ? " or " : ", ") +
             std::string(data::output_moment_quantities[q].name);
  return "<quantity>, <quantity>_tot or <quantity><species index>, quantity " + names;
}

} // namespace

OutputEngineKind parse_output_engine(std::string_view key, std::string_view value,
                                     OutputEngineKind default_engine,
                                     std::string_view context) {
  if (value == "default") return default_engine;
  for (const auto &spelling : engine_spellings)
    if (value == spelling.name) return spelling.engine;
  std::string expected = "default";
  for (std::size_t index = 0; index < engine_spellings.size(); ++index)
    expected += (index + 1 == engine_spellings.size() ? " or " : ", ") +
                std::string(engine_spellings[index].name);
  throw std::invalid_argument(std::string(context) + std::string(key) + " = '" +
                              std::string(value) + "' is unknown (expected " + expected + ")");
}

std::string_view output_engine_name(OutputEngineKind engine) noexcept {
  switch (engine) {
  case OutputEngineKind::Default: return "default";
  case OutputEngineKind::Vtk: return "pvtk";
  case OutputEngineKind::Adios2: return "adios2";
  case OutputEngineKind::Hdf5: return "hdf5";
  }
  return "unknown";
}

bool output_engine_available(OutputEngineKind engine) noexcept {
  switch (engine) {
  case OutputEngineKind::Default:
  case OutputEngineKind::Vtk: return true;
  case OutputEngineKind::Adios2:
#if TPIC_HAS_ADIOS2
    return true;
#else
    return false;
#endif
  case OutputEngineKind::Hdf5:
#if TPIC_HAS_HDF5
    return true;
#else
    return false;
#endif
  }
  return false;
}

bool PreparedOutput::pressure_active() const noexcept {
  for (std::size_t q = data::first_pressure_quantity; q < output_quantity_count; ++q)
    if (total_selected[q] || any(species_selected[q])) return true;
  return false;
}

bool PreparedOutput::fields_enabled() const noexcept {
  if (field_cycle <= 0) return false;
  if (output_b || output_e || output_gauss) return true;
  for (std::size_t q = 0; q < output_quantity_count; ++q)
    if (total_selected[q] || any(species_selected[q])) return true;
  return false;
}

bool PreparedOutput::particles_enabled() const noexcept {
  return particles_cycle > 0;
}

bool PreparedOutput::checkpoints_enabled() const noexcept {
  return checkpoint_cycle > 0 && checkpoint_engine != OutputEngineKind::Default;
}

bool PreparedOutput::should_write_fields(int cycle) const noexcept {
  return field_cycle > 0 && cycle % field_cycle == 0;
}

bool PreparedOutput::should_write_diagnostics(int cycle) const noexcept {
  return diagnostics_cycle > 0 && cycle % diagnostics_cycle == 0;
}

bool PreparedOutput::should_report_energy(int cycle) const noexcept {
  return diagnostic_energy_cycle > 0 && cycle % diagnostic_energy_cycle == 0;
}

bool PreparedOutput::should_sample_gauss(int cycle, int final) const noexcept {
  return (gauss_cycle > 0 && cycle % gauss_cycle == 0) ||
         (output_gauss && (cycle + 1 == final || should_write_fields(cycle + 1)));
}

bool PreparedOutput::should_write_particles(int cycle) const noexcept {
  return particles_cycle > 0 && cycle % particles_cycle == 0;
}

bool PreparedOutput::should_write_checkpoint(int cycle) const noexcept {
  return checkpoint_cycle > 0 && cycle % checkpoint_cycle == 0;
}

PreparedOutput prepare_output(const SimulationParameters &parameters) {
  PreparedOutput prepared;
  const auto &output = parameters.output;
  const auto context = errors.text("");
  const auto common_engine = parameters.write_method.empty() ? OutputEngineKind::Vtk :
      parse_output_engine("WriteMethod", parameters.write_method, OutputEngineKind::Vtk, context);
  prepared.field_engine = parameters.field_write_method.empty() ? common_engine :
      parse_output_engine("FieldWriteMethod", parameters.field_write_method,
                          OutputEngineKind::Vtk, context);
  prepared.particle_engine = parameters.particles_write_method.empty() ? common_engine :
      parse_output_engine("ParticlesWriteMethod", parameters.particles_write_method,
                          OutputEngineKind::Vtk, context);
  prepared.save_dir = parameters.save_dir.empty() ? "data" : parameters.save_dir;
  prepared.field_cycle = output.field_cycle;
  prepared.diagnostics_cycle = output.diagnostics_cycle;
  prepared.particles_cycle = output.particles_cycle;
  prepared.diagnostic_energy_cycle = output.diagnostic_energy_cycle;
  prepared.gauss_cycle = output.gauss_cycle;
  const auto built_checkpoint_engine = output_engine_available(OutputEngineKind::Adios2) ?
      OutputEngineKind::Adios2 : output_engine_available(OutputEngineKind::Hdf5) ?
      OutputEngineKind::Hdf5 : OutputEngineKind::Default;
  prepared.checkpoint_engine = parse_output_engine("RestartWriteMethod",
      parameters.restart_write_method.empty() ? "default" : parameters.restart_write_method,
      built_checkpoint_engine, context);
  prepared.checkpoint_dir = parameters.restart_dir.empty() ? "data" : parameters.restart_dir;
  prepared.checkpoint_cycle = output.restart_cycle;
  prepared.checkpoint_final = output.call_finalize;

  const auto count = static_cast<std::size_t>(parameters.ns);
  const bool uses_default_tags = output.field_tag.empty() && output.moments_tag.empty();
  prepared.output_b = prepared.output_e = uses_default_tags;
  for_each_token("FieldOutputTag", output.field_tag, [&](std::string_view token) {
    if (token == "B") prepared.output_b = true;
    else if (token == "E") prepared.output_e = true;
    else if (token == "Gauss") prepared.output_gauss = prepared.field_cycle > 0;
    else reject_token("FieldOutputTag", token, "B, E or Gauss");
  });
  // Default tags select every species' and the total rho and J.
  for (std::size_t q = 0; q < output_quantity_count; ++q) {
    const bool by_default = uses_default_tags && q <= output_current_quantity;
    prepared.species_selected[q].assign(count, by_default);
    prepared.total_selected[q] = by_default;
  }
  std::array<bool, output_quantity_count> bare{}, indexed{};
  const auto &quantities = data::output_moment_quantities;
  for_each_token("MomentsOutputTag", output.moments_tag, [&](std::string_view token) {
    const auto quantity = std::find_if(quantities.begin(), quantities.end(),
        [&](const auto &candidate) { return token.starts_with(candidate.name); });
    if (quantity == quantities.end())
      reject_token("MomentsOutputTag", token, moment_token_forms());
    const auto q = static_cast<std::size_t>(quantity - quantities.begin());
    const auto suffix = token.substr(quantity->name.size());
    if (suffix.empty()) {
      bare[q] = true;
    } else if (suffix == "_tot") {
      prepared.total_selected[q] = true;
    } else if (std::all_of(suffix.begin(), suffix.end(), is_digit)) {
      const auto index = species_index(quantity->name, suffix);
      if (index < count) {
        prepared.species_selected[q][index] = true;
        indexed[q] = true;
      }
    } else {
      reject_token("MomentsOutputTag", token, moment_token_forms());
    }
  });
  // A bare name selects every species unless an in-range index of it is named.
  for (std::size_t q = 0; q < output_quantity_count; ++q)
    if (bare[q] && !indexed[q])
      std::fill(prepared.species_selected[q].begin(), prepared.species_selected[q].end(), true);
  return prepared;
}

} // namespace tpic::config
