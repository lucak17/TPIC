#include <tpic/config/output.hpp>

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tpic::config::OutputEngineKind;
using tpic::config::PreparedOutput;
using tpic::config::SimulationParameters;
using tpic::config::prepare_output;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function>
void rejects(Function &&function, const char *message) {
  try {
    function();
  } catch (const Exception &error) {
    require(std::string(error.what()).starts_with("TPIC output:"),
            "missing output error context");
    return;
  }
  throw std::runtime_error(message);
}

SimulationParameters parameters() {
  SimulationParameters result;
  result.ns = 3;
  return result;
}

void defaults_and_disabled_fields() {
  auto input = parameters();
  auto output = prepare_output(input);
  const std::vector<bool> all(3, true), none(3, false);
  require(output.output_b && output.output_e &&
          output.total_selected[tpic::config::output_rho_quantity] && output.total_selected[tpic::config::output_current_quantity] &&
          output.species_selected[tpic::config::output_rho_quantity] == all && output.species_selected[tpic::config::output_current_quantity] == all,
          "empty tags lost default quantities");
  require(!output.pressure_active() && output.fields_enabled(),
          "default pressure or field activity is wrong");
  require(!output.particles_enabled(), "particle output was enabled by default");
  require(output.save_dir == "data", "wrong default paths");

  input.output.moments_tag = "rho1";
  output = prepare_output(input);
  require(!output.output_b && !output.output_e &&
          !output.total_selected[tpic::config::output_rho_quantity] && !output.total_selected[tpic::config::output_current_quantity] &&
          output.species_selected[tpic::config::output_rho_quantity] == std::vector<bool>{false, true, false} &&
          output.species_selected[tpic::config::output_current_quantity] == none, "nonempty moments tag enabled default fields");

  input.output.moments_tag.clear();
  input.output.field_tag = "B";
  output = prepare_output(input);
  require(output.output_b && !output.output_e && output.species_selected[tpic::config::output_rho_quantity] == none &&
          output.species_selected[tpic::config::output_current_quantity] == none, "field-only tag enabled default moments");

  input.output.field_tag.clear();
  input.output.moments_tag = "rho5";
  output = prepare_output(input);
  require(!output.output_b && !output.output_e && !output.total_selected[tpic::config::output_rho_quantity] &&
          !output.total_selected[tpic::config::output_current_quantity] && output.species_selected[tpic::config::output_rho_quantity] == none && output.species_selected[tpic::config::output_current_quantity] == none &&
          !output.pressure_active() && !output.fields_enabled(),
          "an out-of-range species index requested a field writer");
  require(output.should_write_fields(0), "cadence was conflated with quantity selection");
  input.output.moments_tag.clear();
  input.output.field_cycle = 0;
  output = prepare_output(input);
  require(output.output_b && output.output_e && !output.fields_enabled(),
          "disabled cadence discarded the resolved selection");
}

void indexed_and_bare_precedence() {
  struct Fixture {
    const char *tag;
    std::vector<bool> species;
    bool total;
  };
  const Fixture fixtures[]{
      {"TOKEN", {true, true, true}, false},
      {"TOKEN+TOKEN0", {true, false, false}, false},
      {"TOKEN0+TOKEN", {true, false, false}, false},
      {"TOKEN+TOKEN99", {true, true, true}, false},
      {"TOKEN99", {false, false, false}, false},
      {"TOKEN_tot", {false, false, false}, true},
      {"TOKEN_tot+TOKEN+TOKEN2", {false, false, true}, true},
      {"TOKEN1+TOKEN2+TOKEN1+TOKEN99", {false, true, true}, false},
      {"TOKEN0000000000000000000000000000000000000000002", {false, false, true}, false},
      {"TOKEN2147483647", {false, false, false}, false},
  };
  const std::vector<std::string> tokens{"rho", "J", "PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ"};
  for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
    for (const auto &fixture : fixtures) {
      std::string tag(fixture.tag);
      std::size_t position = 0;
      while ((position = tag.find("TOKEN", position)) != std::string::npos) {
        tag.replace(position, 5, tokens[token_index]);
        position += tokens[token_index].size();
      }
      auto input = parameters();
      input.output.field_tag = "E";
      input.output.moments_tag = tag;
      const auto output = prepare_output(input);
      const auto &species = output.species_selected[token_index];
      const bool total = output.total_selected[token_index];
      if (species != fixture.species || total != fixture.total)
        throw std::runtime_error("selection mismatch for " + tag);
    }
  }
  auto input = parameters();
  input.output.field_tag = "E+B";
  input.output.moments_tag = "PXY1+PZZ_tot";
  const auto output = prepare_output(input);
  require(output.output_b && output.output_e && output.pressure_active(),
          "whole-token field matching or pressure selection lost");
  require(output.species_selected[tpic::config::output_rho_quantity] == std::vector<bool>(3, false) &&
          output.species_selected[tpic::config::output_current_quantity] == std::vector<bool>(3, false) &&
          output.species_selected[tpic::data::first_pressure_quantity + 0] == std::vector<bool>(3, false) &&
          output.species_selected[tpic::data::first_pressure_quantity + 1] == std::vector<bool>{false, true, false} &&
          output.total_selected[tpic::data::first_pressure_quantity + 5], "whole-token moment selection lost");
}

// Spellings that the whole-token grammar rejects, each with the token the
// rejection names.
void anchored_tag_rejections() {
  struct Rejection {
    const char *field, *moments, *token;
  };
  const Rejection rejections[]{
      {"none", "", "FieldOutputTag token 'none'"},
      {"surroundBtextE", "", "FieldOutputTag token 'surroundBtextE'"},
      {"BystanderE", "", "FieldOutputTag token 'BystanderE'"},
      {"EB", "", "FieldOutputTag token 'EB'"},
      {"B E", "", "FieldOutputTag token 'B E'"},
      {"b", "", "FieldOutputTag token 'b'"},
      {"B+", "", "FieldOutputTag = 'B+' has an empty token"},
      {"", "+rho", "MomentsOutputTag = '+rho' has an empty token"},
      {"", "rho++J", "MomentsOutputTag = 'rho++J' has an empty token"},
      {"", "rhoJ", "MomentsOutputTag token 'rhoJ'"},
      {"", "rho1rho", "MomentsOutputTag token 'rho1rho'"},
      {"", "rho5rho", "MomentsOutputTag token 'rho5rho'"},
      {"", "rho-1", "MomentsOutputTag token 'rho-1'"},
      {"", "prerho2suffix", "MomentsOutputTag token 'prerho2suffix'"},
      {"", "xrho01suffixrho_totally", "MomentsOutputTag token 'xrho01suffixrho_totally'"},
      {"", "rhorho2", "MomentsOutputTag token 'rhorho2'"},
      {"", "J_totJ2", "MomentsOutputTag token 'J_totJ2'"},
      {"", "PXX_totally", "MomentsOutputTag token 'PXX_totally'"},
      {"", "RHO", "MomentsOutputTag token 'RHO'"},
      {"", "j", "MomentsOutputTag token 'j'"},
      {"", "pxx", "MomentsOutputTag token 'pxx'"},
      {"", "rho_tot J_tot", "MomentsOutputTag token 'rho_tot J_tot'"},
      {"", "something", "MomentsOutputTag token 'something'"},
  };
  for (const auto &rejection : rejections) {
    auto input = parameters();
    input.output.field_tag = rejection.field;
    input.output.moments_tag = rejection.moments;
    try {
      (void)prepare_output(input);
    } catch (const std::invalid_argument &error) {
      const std::string message = error.what();
      require(message.starts_with("TPIC output: ") &&
                  message.find(rejection.token) != std::string::npos,
              "tag rejection lost the key or token");
      continue;
    }
    throw std::runtime_error(std::string("tag accepted: FieldOutputTag=") + rejection.field +
                             ", MomentsOutputTag=" + rejection.moments);
  }
}

void engine_aliases_and_paths() {
  auto input = parameters();
  const auto has_engines = [&](OutputEngineKind field, OutputEngineKind particles) {
    const auto output = prepare_output(input);
    return output.field_engine == field && output.particle_engine == particles;
  };
  for (const char *name : {"pvtk", "vtk"}) {
    input.write_method = name;
    require(has_engines(OutputEngineKind::Vtk, OutputEngineKind::Vtk), "VTK alias failed");
  }
  for (const char *name : {"adios2", "bp", "bp5"}) {
    input.write_method = name;
    require(has_engines(OutputEngineKind::Adios2, OutputEngineKind::Adios2), "ADIOS alias failed");
  }
  for (const char *name : {"hdf5", "h5"}) {
    input.write_method = name;
    require(has_engines(OutputEngineKind::Hdf5, OutputEngineKind::Hdf5), "HDF5 alias failed");
  }
  // An empty WriteMethod selects VTK; unknown and differently cased values are
  // rejected in invalid_channel_engines.
  input.write_method = "";
  require(has_engines(OutputEngineKind::Vtk, OutputEngineKind::Vtk),
          "an empty WriteMethod did not select VTK");
  input.write_method = "hdf5";
  input.field_write_method = "vtk";
  require(has_engines(OutputEngineKind::Vtk, OutputEngineKind::Hdf5),
          "field override changed the particle fallback");
  input.particles_write_method = "bp5";
  require(has_engines(OutputEngineKind::Vtk, OutputEngineKind::Adios2),
          "independent engine selection failed");
  input.field_write_method.clear();
  require(has_engines(OutputEngineKind::Hdf5, OutputEngineKind::Adios2),
          "empty field override did not inherit WriteMethod");
  input.field_write_method = "default";
  input.particles_write_method = "default";
  require(has_engines(OutputEngineKind::Vtk, OutputEngineKind::Vtk),
          "explicit default must select VTK instead of inheriting WriteMethod");
  input.field_write_method.clear();
  input.particles_write_method.clear();
  require(tpic::config::output_engine_name(OutputEngineKind::Default) == "default" &&
          tpic::config::output_engine_name(OutputEngineKind::Vtk) == "pvtk" &&
          tpic::config::output_engine_name(OutputEngineKind::Adios2) == "adios2" &&
          tpic::config::output_engine_name(OutputEngineKind::Hdf5) == "hdf5",
          "wrong engine reporting names");
  input.save_dir.clear();
  auto output = prepare_output(input);
  require(output.save_dir == "data", "empty path not resolved");
  // Resolution accepts these strings without attempting directory operations.
  input.save_dir = "/dev/null/tpic-output-must-not-be-created";
  output = prepare_output(input);
  require(output.save_dir == input.save_dir, "nonempty path modified");
}

void invalid_channel_engines() {
  auto input = parameters();
  input.output.field_cycle = input.output.particles_cycle = 0;
  for (const auto channel : {"WriteMethod", "FieldWriteMethod", "ParticlesWriteMethod"}) {
    for (const auto invalid : {"unknown", "VTK", "PVTK", "ADIOS2", "BP", "BP5", "hdf", " vtk"}) {
      auto& method = std::string(channel) == "WriteMethod" ? input.write_method
          : std::string(channel) == "FieldWriteMethod" ? input.field_write_method
          : input.particles_write_method;
      method = invalid;
      try {
        (void)prepare_output(input);
      } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        require(message.starts_with("TPIC output: " + std::string(channel) + " = '" + invalid +
                                    "' is unknown (expected default, pvtk, vtk, adios2, bp, bp5, hdf5 or h5)"),
                "invalid channel backend lost the input key or value");
        method.clear();
        continue;
      }
      throw std::runtime_error("unknown channel engine accepted even with disabled cadence");
    }
  }
}

void cadence_and_unsupported_requests() {
  auto input = parameters();
  input.output.field_cycle = 4;
  input.output.diagnostics_cycle = 3;
  input.output.particles_cycle = 5;
  input.output.gauss_cycle = 2;
  auto output = prepare_output(input);
  require(output.should_sample_gauss(0, 100) && !output.should_sample_gauss(3, 100) &&
              output.should_sample_gauss(4, 100) && !output.should_sample_gauss(99, 100),
          "Gauss cadence mismatch");
  require(output.should_write_fields(0) && !output.should_write_fields(5) &&
          !output.should_write_fields(6) && output.should_write_fields(8),
          "modulo field cadence mismatch");
  require(!output.should_write_diagnostics(5) && output.should_write_diagnostics(6),
          "diagnostics inherited field first-cycle behavior");
  require(output.particles_enabled() && output.should_write_particles(0) &&
          output.should_write_particles(5) && !output.should_write_particles(4),
          "particle cadence is not independent of fields/diagnostics");

  for (int interval : {0, -1, std::numeric_limits<int>::min()}) {
    input.output.field_cycle = input.output.diagnostics_cycle = input.output.particles_cycle =
        input.output.gauss_cycle = interval;
    output = prepare_output(input);
    for (int cycle : {std::numeric_limits<int>::min(), 0, 1, std::numeric_limits<int>::max()})
      require(!output.should_write_fields(cycle) && !output.should_sample_gauss(cycle, 0) &&
              !output.should_write_diagnostics(cycle) && !output.should_write_particles(cycle) &&
              !output.particles_enabled(), "nonpositive cadence was enabled");
  }
  input.output.field_cycle = input.output.diagnostics_cycle = std::numeric_limits<int>::max();
  output = prepare_output(input);
  require(output.should_write_fields(std::numeric_limits<int>::max()) &&
          output.should_write_diagnostics(std::numeric_limits<int>::max()) &&
          !output.should_write_fields(std::numeric_limits<int>::min()),
          "extreme cycle values broke guarded modulo");

  input.output.particles_cycle = 0;
  input.output.particles_tag = "position";
  input.output.test_particles_cycle = input.output.restart_cycle = 3;
  output = prepare_output(input);
  require(!output.particles_enabled() && !output.should_write_particles(0),
          "the particle tag or another cadence enabled particle output");
  require(output.should_write_checkpoint(0) && output.should_write_checkpoint(6) &&
              !output.should_write_checkpoint(4), "checkpoint cadence mismatch");
  input.output.restart_cycle = -1;
  require(!prepare_output(input).should_write_checkpoint(0), "nonpositive checkpoint cadence was enabled");

  // FieldOutputTag Gauss begins a sample one cycle before every field snapshot,
  // the final state's included, and needs field output.
  input.output.gauss_cycle = 0;
  input.output.field_cycle = 4;
  input.output.field_tag = "Gauss";
  output = prepare_output(input);
  require(output.output_gauss && !output.output_e && output.fields_enabled() &&
              output.should_sample_gauss(3, 100) && !output.should_sample_gauss(4, 100) &&
              output.should_sample_gauss(99, 100) && !output.should_sample_gauss(98, 100),
          "Gauss field samples do not precede the field snapshots");
  input.output.field_cycle = 0;
  require(!prepare_output(input).output_gauss, "Gauss output without field output");
}

void invalid_counts_and_index_overflow() {
  auto input = parameters();
  input.ns = 3;
  for (const char *token : {"rho", "J", "PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ"}) {
    for (const char *index : {"2147483648", "999999999999999999999999999999999999999"}) {
      // Overflow remains an input error even if this quantity is disabled or
      // another valid indexed match already selected a species.
      input.output.field_cycle = 0;
      input.output.moments_tag = std::string(token) + "0+" + token + index;
      rejects<std::overflow_error>([&] { (void)prepare_output(input); },
                                   "oversized output species index accepted");
    }
  }
  input.ns = 10;
  input.output.moments_tag = "rho9+J+PXX9";
  const auto output = prepare_output(input);
  require(output.species_selected[tpic::config::output_rho_quantity].size() == 10 && output.species_selected[tpic::config::output_rho_quantity][9] &&
          output.species_selected[tpic::config::output_current_quantity] == std::vector<bool>(10, true) && output.species_selected[tpic::data::first_pressure_quantity + 0][9],
          "maximum supported species count failed");
}

} // namespace

int main() {
  try {
    defaults_and_disabled_fields();
    indexed_and_bare_precedence();
    anchored_tag_rejections();
    engine_aliases_and_paths();
    invalid_channel_engines();
    cadence_and_unsupported_requests();
    invalid_counts_and_index_overflow();
    std::cout << "TPIC output checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
