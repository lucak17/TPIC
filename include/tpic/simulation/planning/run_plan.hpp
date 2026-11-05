#ifndef TPIC_SIMULATION_PLANNING_RUN_PLAN_HPP
#define TPIC_SIMULATION_PLANNING_RUN_PLAN_HPP

#include <tpic/config/domain.hpp>
#include <tpic/config/numerics_options.hpp>
#include <tpic/config/output.hpp>
#include <tpic/config/particle_plan.hpp>
#include <tpic/config/physics.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/output/checkpoint.hpp>
#include <tpic/ecsim/maxwell/maxwell.hpp>
#include <tpic/ecsim/maxwell/maxwell_apply_policy.hpp>
#include <tpic/initialization/cases/initial_case.hpp>
#include <tpic/initialization/prepared_initialization.hpp>

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace tpic::simulation {

struct InputOptions {
  int rank = 0;
  // Zero derives the size from XLEN*YLEN*ZLEN for checking without MPI.
  // Startup must pass the actual communicator size for agreement checking.
  int communicator_size = 0;
  config::ParticleOrder order = config::default_particle_order;
  bool overwrite_output = false;
  // Continue from the newest complete checkpoint of RestartDirName.
  bool restart = false;
  config::MassDepositionKernel mass_matrix = config::MassDepositionKernel::scalar;
  config::PrecisionSelection precision;
  config::MassApplyMode mass_apply = config::MassApplyMode::full;
};

enum class RunMode { initialize_only, advance };

[[nodiscard]] constexpr bool advances(RunMode mode) noexcept {
  return mode == RunMode::advance;
}

// Immutable after startup. Physical diagnostics/output cadence is configured
// separately; synchronous deposition deliberately changes debugging overlap.
struct RuntimeDebugOptions {
  bool profile_stages{};
  bool synchronous_deposition{};

  [[nodiscard]] static RuntimeDebugOptions from_environment() noexcept {
    const auto enabled = [](const char* name) {
      const char* value = std::getenv(name);
      return value != nullptr && value[0] == '1' && value[1] == '\0';
    };
    return {enabled("TPIC_PROFILE_STAGES"), enabled("TPIC_DIAGNOSTIC_SYNC_DEPOSITION")};
  }
};

struct PreparedInput {
  config::SimulationParameters parameters;
  int communicator_size = 0;
  config::PreparedDomain domain;
  config::PreparedPhysics physics;
  initialization::InitialCase initial_case;
  config::PreparedOutput output;
  config::ParticlePlan particles;
  config::MassDepositionKernel mass_matrix = config::MassDepositionKernel::scalar;
  config::PrecisionSelection precision;
  config::MassApplyMode mass_apply = config::MassApplyMode::full;
  // Resolved once from mass_apply and smoothing; printed and dispatched as stored.
  ecsim::MaxwellApplySchedule maxwell_schedule = ecsim::MaxwellApplySchedule::full;
  // Warnings are reported and the run proceeds; any unsupported request rejects it.
  std::vector<std::string> warnings;
  std::vector<std::string> unsupported_requests;

  operator ecsim::MaxwellParameters() const noexcept {
    return {domain, physics, parameters.c, parameters.dt, parameters.th,
            parameters.smooth_niter, parameters.smooth, parameters.ecsim_smoothing_operator};
  }

  // No unsupported request. RunPlan::prepare rejects the input otherwise and
  // then rejects field boundaries without a device implementation and, for
  // an advancing run, unavailable output backends.
  [[nodiscard]] bool host_admitted() const noexcept { return unsupported_requests.empty(); }
};

// Snapshot streams, capture and the writers exist only when the run advances
// and a field or particle channel or checkpoints are enabled.
[[nodiscard]] inline bool snapshots_enabled(const PreparedInput& input, RunMode mode) noexcept {
  return advances(mode) && (input.output.fields_enabled() || input.output.particles_enabled() ||
                            input.output.checkpoints_enabled());
}

[[nodiscard]] PreparedInput prepare_input(config::SimulationParameters parameters,
                                         InputOptions options = {});

// One message per enabled output channel whose engine this build lacks:
// --run refuses the first, --check-input reports them as unsupported.
[[nodiscard]] std::vector<std::string> unavailable_output_backends(const PreparedInput& input);

// The checkpoint header of a run: everything but the cycle and the energy baseline.
[[nodiscard]] data::CheckpointHeader prepare_checkpoint_header(const PreparedInput& input);

// The particle-memory lines of --check-input for the selected precision profile:
// the logical bytes of the initial particle owners (alignment gaps excluded,
// one widest-attribute auxiliary included), of the sort workspace (permutation,
// bin counts/offsets and status; zero when unsorted) and of the classification
// slots with LPFx's registered counters (zero when bypassed), and the logical
// bytes of one particle record. Scan scratch, events, growth peaks, migration,
// fields and output are excluded. Byte arithmetic that is not representable
// throws overflow_error.
struct ParticleByteReport {
  std::size_t device_bytes_lower_bound{};
  std::size_t particle_record_bytes{};
};
[[nodiscard]] ParticleByteReport report_particle_bytes(const PreparedInput& input);

// One immutable owner of all host preparation. prepare is the only factory: it
// derives every record once from the parameters and options, rejects
// unsupported requests, field boundaries without a device implementation
// and, for an advancing run, unavailable output backends.
// Everything that takes a RunPlan trusts these checks. A restart from
// checkpoint also reads which pieces hold this rank's state.
class RunPlan {
  RunPlan(PreparedInput input, std::vector<initialization::ParticleLoadPlan> loads,
          std::optional<data::PreparedRestart> restart, RunMode mode, RuntimeDebugOptions debug);

public:
  static RunPlan prepare(config::SimulationParameters parameters, InputOptions options,
                         RunMode mode, RuntimeDebugOptions debug = {},
                         std::optional<data::CheckpointLocation> checkpoint = {});

  // The cycle of the first state: the checkpoint's, else 0.
  [[nodiscard]] int first_cycle() const noexcept { return restart ? restart->checkpoint.header.cycle : 0; }

  const PreparedInput input;
  // The initial particle load of each species, prepared once with the input.
  const std::vector<initialization::ParticleLoadPlan> particle_loads;
  const std::optional<data::PreparedRestart> restart;
  const RunMode mode;
  const RuntimeDebugOptions debug;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_PLANNING_RUN_PLAN_HPP
