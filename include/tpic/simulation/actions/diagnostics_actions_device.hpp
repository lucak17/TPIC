#ifndef TPIC_SIMULATION_ACTIONS_DIAGNOSTICS_ACTIONS_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_DIAGNOSTICS_ACTIONS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/profiling.hpp>

#include <lpfx/distributed/transport/mpi/setup.hpp>

#include <tpic/config/precision.hpp>
#include <tpic/diagnostics/gauss_residual_device.hpp>
#include <tpic/diagnostics/scalar_diagnostics_device.hpp>
#include <tpic/io/diagnostics_writer.hpp>

#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/simulation/execution/active_backend_device.hpp>

#include <optional>
#include <string>

namespace tpic::simulation {

// The scalar diagnostics and the Gauss-law residual of the committed state,
// and their files.
template <class Precision, class SpeciesPack, class Ports> class DiagnosticsActions {
  using Diagnostics = diagnostics::ScalarDiagnostics<Precision, Backend>;
  using Gauss = diagnostics::GaussResidual<Precision, Backend>;

public:
  DiagnosticsActions(const simulation::PreparedInput &input,
                   const lpfx::distributed::CommunicationSetup &setup, RunMode mode,
                   const lpfx::device::DeviceProperties &properties,
                   data::DeviceFieldImage<Precision, Backend> &fields, SpeciesPack &species, Ports &ports,
                   Executor diagnostics_executor, Executor helper_executor)
      : input_(input), setup_(setup), fields_(fields), species_(species), ports_(ports),
        advances_(advances(mode)), diagnostics_(input.domain, input.particles.species.size(), advances_,
                                                properties, diagnostics_executor, helper_executor),
        gauss_(input.domain, input.physics, input.parameters,
               advances_ && (input.output.gauss_cycle > 0 || input.output.output_gauss), properties,
               diagnostics_executor) {}
  Diagnostics &workspace() noexcept { return diagnostics_; }
  Gauss &gauss() noexcept { return gauss_; }
  // The stream ScalarDiagnostics chooses for species s.
  [[nodiscard]] Executor executor(unsigned s) const { return diagnostics_.species_executor(s); }
  void field_energy() {
    LPFX_PROFILE_SCOPE("TPIC::FieldDiagnostics");
    diagnostics_.enqueue_field_energy(fields_.electric, fields_.magnetic_cell);
  }
  void species_scalars(unsigned s) {
    LPFX_PROFILE_SCOPE("TPIC::ParticleDiagnostics");
    diagnostics_.enqueue_species_scalars(s, species_[s].particles, input_.parameters.species[s].q_over_m);
  }
  // host_blocking_finish() issues a native collective.
  [[nodiscard]] data::ScalarDiagnosticSample host_blocking_collect(int cycle, bool final_state) {
    LPFX_PROFILE_SCOPE("TPIC::DiagnosticsCollect");
    return diagnostics_.host_blocking_finish(ports_.template get<"diagnostics">(), cycle,
                               double(cycle) * input_.parameters.dt, final_state);
  }

  void read_gauss_state(bool begin, bool finish, unsigned *invalid) {
    LPFX_PROFILE_SCOPE("TPIC::GaussDiagnostics");
    gauss_.enqueue_field(fields_.electric, begin, finish);
    if (!begin) return;
    for (std::size_t s = 0; s < input_.particles.species.size(); ++s)
      gauss_.enqueue_deposit(species_[s].particles, input_.parameters.species[s].q_over_m, invalid);
  }
  void complete_gauss_charge() {
    LPFX_PROFILE_SCOPE("TPIC::GaussCharge");
    gauss_.complete_charge(ports_.template get<"gauss.charge">(), ports_.template get<"gauss.charge.ghosts">(),
                           ports_.template get<"gauss.smooth">());
  }
  // The finished sample; rank 0 writes its gauss.csv row.
  [[nodiscard]] data::GaussSample host_blocking_collect_gauss(int cycle) {
    LPFX_PROFILE_SCOPE("TPIC::GaussCollect");
    const auto sample = gauss_.host_blocking_finish(ports_.template get<"gauss">(), cycle);
    if (setup_.rank() == 0) gauss_writer_.write(sample);
    return sample;
  }

  // A restart continues the files of the run it restarts.
  void open_scalars(const std::optional<data::PreparedRestart> &restart) {
    if (!advances_ || setup_.rank() != 0) return;
    const auto metadata = config::precision_metadata(input_.precision);
    const auto cycle = restart ? std::optional<int>(restart->checkpoint.header.cycle) : std::nullopt;
    if (input_.output.diagnostics_cycle > 0)
      diagnostics_writer_.open(input_.output, input_.particles.species.size(), metadata, cycle);
    if (gauss_.enabled()) gauss_writer_.open(input_.output, metadata, cycle);
  }
  [[nodiscard]] bool scalars_due(const data::ScalarDiagnosticSample &sample) const {
    return sample.final_state ? input_.output.diagnostics_cycle > 0
                              : input_.output.should_write_diagnostics(sample.cycle);
  }
  void write_scalars(const data::ScalarDiagnosticSample &sample, double initial_energy) {
    LPFX_PROFILE_SCOPE("TPIC::OutputDiagnostics");
    diagnostics_writer_.write(sample, initial_energy);
  }
  // The path of gauss.csv on rank 0 when the residual is on, else empty.
  std::string close_gauss() {
    const auto path = gauss_writer_.path();
    gauss_writer_.close();
    return path;
  }
  std::string close_scalars() {
    if (setup_.rank() != 0 || input_.output.diagnostics_cycle <= 0)
      return {};
    const auto path = diagnostics_writer_.path();
    diagnostics_writer_.close();
    return path;
  }

private:
  const simulation::PreparedInput &input_;
  const lpfx::distributed::CommunicationSetup &setup_;
  data::DeviceFieldImage<Precision, Backend> &fields_;
  SpeciesPack &species_;
  Ports &ports_;
  const bool advances_;
  Diagnostics diagnostics_;
  Gauss gauss_;
  io::DiagnosticsWriter diagnostics_writer_;
  io::GaussWriter gauss_writer_;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_DIAGNOSTICS_ACTIONS_DEVICE_HPP
