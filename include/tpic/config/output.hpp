#ifndef TPIC_CONFIG_OUTPUT_HPP
#define TPIC_CONFIG_OUTPUT_HPP

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <tpic/config/simulation_parameters.hpp>
#include <tpic/data/output/output_moment_quantities.hpp>

namespace tpic::config {

// TPIC keeps VTK as the default independently of optional backend libraries.
// Preparing a request does not construct a writer or perform filesystem work.
enum class OutputEngineKind { Default, Vtk, Adios2, Hdf5 };

[[nodiscard]] std::string_view output_engine_name(OutputEngineKind engine) noexcept;
[[nodiscard]] bool output_engine_available(OutputEngineKind engine) noexcept;

// The engine that the value of an output-method key (WriteMethod,
// FieldWriteMethod, ParticlesWriteMethod or RestartWriteMethod) names:
// default, pvtk, vtk, adios2, bp, bp5, hdf5 or h5, case-sensitive; "default" names default_engine. An
// unknown value throws std::invalid_argument
// "<context><key> = '<value>' is unknown (expected ...)", where context is the
// caller's error prefix (prepare_output: "TPIC output: "; the input parser adds
// the deck line).
[[nodiscard]] OutputEngineKind parse_output_engine(std::string_view key, std::string_view value,
                                                   OutputEngineKind default_engine,
                                                   std::string_view context);

// Moment selections are indexed like data::output_moment_quantities.
inline constexpr std::size_t output_quantity_count = data::output_moment_quantities.size();
inline constexpr std::size_t output_rho_quantity = 0;
inline constexpr std::size_t output_current_quantity = 1;
static_assert(data::output_moment_quantities[output_rho_quantity].name == "rho" &&
                  data::output_moment_quantities[output_current_quantity].name == "J",
              "output quantity indices follow output_moment_quantities");

struct PreparedOutput {
  bool operator==(const PreparedOutput&) const = default;

  OutputEngineKind field_engine = OutputEngineKind::Vtk;
  OutputEngineKind particle_engine = OutputEngineKind::Vtk;
  std::string save_dir;
  // --overwrite-output or --restart; replaces only files emitted by this run.
  bool overwrite_output = false;
  bool output_b = false;
  bool output_e = false;
  // FieldOutputTag Gauss with field output on: the Gauss-law residual of the
  // sample that finishes at each field snapshot.
  bool output_gauss = false;
  // Per quantity: the species whose array is written, one entry per species,
  // and whether the total array is written.
  std::array<std::vector<bool>, output_quantity_count> species_selected;
  std::array<bool, output_quantity_count> total_selected{};

  int field_cycle = 0;
  int diagnostics_cycle = 0;
  int particles_cycle = 0;
  int diagnostic_energy_cycle = 1;
  int gauss_cycle = 0;

  // Checkpoints: RestartOutputCycle, CallFinalize, RestartDirName and the
  // engine of RestartWriteMethod; "default" resolves to ADIOS2 when it is built,
  // else HDF5, else Default (no checkpoint backend).
  OutputEngineKind checkpoint_engine = OutputEngineKind::Default;
  std::string checkpoint_dir;
  int checkpoint_cycle = 0;
  bool checkpoint_final = false;

  [[nodiscard]] bool pressure_active() const noexcept;
  [[nodiscard]] bool fields_enabled() const noexcept;
  [[nodiscard]] bool particles_enabled() const noexcept;
  // A positive cadence and a checkpoint backend.
  [[nodiscard]] bool checkpoints_enabled() const noexcept;

  // Cadence predicates. Nonpositive intervals disable output.
  [[nodiscard]] bool should_write_fields(int cycle) const noexcept;
  [[nodiscard]] bool should_write_diagnostics(int cycle) const noexcept;
  [[nodiscard]] bool should_report_energy(int cycle) const noexcept;
  // A Gauss-law sample begins at cycle and finishes at cycle + 1: at the
  // DiagnosticGaussCycle cadence and, for FieldOutputTag Gauss, one cycle before
  // each field snapshot (final the run's last cycle, whose state is always written).
  [[nodiscard]] bool should_sample_gauss(int cycle, int final) const noexcept;
  [[nodiscard]] bool should_write_particles(int cycle) const noexcept;
  [[nodiscard]] bool should_write_checkpoint(int cycle) const noexcept;
};

// Pure host resolution of output tags, engines and defaults, with no filesystem,
// MPI, device, or writer operations. parameters.ns is the admitted species count
// (config::require_species_count).
// Tags are '+'-separated tokens, each matched whole and case-sensitively.
// FieldOutputTag tokens are B, E and Gauss. A MomentsOutputTag token is a quantity
// name of data::output_moment_quantities alone (every species, unless an
// in-range indexed token of the same name is present), followed by _tot (the
// total) or followed by a decimal species index (that species if below ns,
// otherwise nothing). Any other token, including an empty one, throws
// invalid_argument; an index outside the signed-int range throws overflow_error.
// Empty tags select E, B and the total and per-species rho and J.
// Engines: see parse_output_engine. An empty WriteMethod selects the default
// engine; nonempty channel methods override WriteMethod, and the explicit
// channel value "default" selects VTK.
[[nodiscard]] PreparedOutput prepare_output(const SimulationParameters &parameters);

} // namespace tpic::config

#endif // TPIC_CONFIG_OUTPUT_HPP
