#ifndef TPIC_SIMULATION_DRIVER_SIMULATION_DEVICE_HPP
#define TPIC_SIMULATION_DRIVER_SIMULATION_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/profiling.hpp>

#include <lpfx/distributed/runtime/host_control.hpp>
#include <lpfx/distributed/runtime/error_reporting.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>

#include <tpic/data/fields/field_storage.hpp>
#include <tpic/data/output/output_snapshot.hpp>
#include <tpic/device/device_validation.hpp>
#include <tpic/initialization/fields/field_image.hpp>
#include <tpic/io/async_output_writer.hpp>
#include <tpic/io/checkpoint.hpp>

#include <tpic/simulation/actions/deposition_actions_device.hpp>
#include <tpic/simulation/actions/diagnostics_actions_device.hpp>
#include <tpic/simulation/actions/field_advance_actions_device.hpp>
#include <tpic/simulation/actions/field_solve_actions_device.hpp>
#include <tpic/simulation/actions/local_species_actions_device.hpp>
#include <tpic/simulation/actions/mover_actions_device.hpp>
#include <tpic/simulation/actions/output_actions_device.hpp>
#include <tpic/simulation/driver/pipeline.hpp>
#include <tpic/simulation/driver/run_distributed.hpp>
#include <tpic/simulation/execution/active_backend_device.hpp>
#include <tpic/simulation/execution/execution_streams.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/simulation/reporting/startup_report.hpp>
#include <tpic/simulation/transport/transport_registry_device.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <span>
#include <type_traits>

namespace tpic::simulation {

// The physics of one run of one precision profile, particle order and mass
// deposition kernel: the device owners, the actions and the distributed
// primitives. It decides nothing about order; Pipeline does. The species pack
// and the communication are compiled for the capacity Ns; the run uses the
// species_count species of its input.
template <class Precision, bool Partial, config::MassDepositionKernel Kernel> struct Simulation {
  using Backend = simulation::Backend;
  using Validation = device::DeviceValidation<Backend>;
  using Writer = io::AsyncOutputWriter;
  using Capture = data::OwnedOutputSnapshot;
  static constexpr std::size_t Ns = config::max_real_species;
  static constexpr bool partial = Partial;
  using Transport = TransportRegistry<Precision, Ns, Partial>;
  using SpeciesPack = typename Transport::SpeciesPack;
  using Solved = ecsim::GmresResult<typename Precision::Compute::SolverReduction>;
  using SolverField = ecsim::SolverNodeOwner<Precision, Backend>;
  using Residents = lpfx::ResidentParticleRead<typename data::ParticleSchema<Precision>::type, Backend>;
  using Fields = data::FieldSchema<Precision>;

  Simulation(const RunPlan& run_plan, const lpfx::distributed::CommunicationSetup& run_setup,
             const StreamExecutors<Backend, Ns>& executors)
      : plan(run_plan), input(plan.input), setup(run_setup), domain(setup.domain()),
        properties(setup.device_properties()), rank(setup.rank()), ranks(setup.size()),
        species_count(unsigned(input.particles.species.size())),
        diagnose(plan.debug.synchronous_deposition), x(executors),
        geometry(communication::prepare_particle_migration_geometry(input.domain, input.physics)),
        mass(input.domain),
        centering(static_cast<unsigned>(input.particles.max_species_local_count()), properties, x.control),
        field_solve(input, properties, unsigned(ranks), fields, current, mass, transport, x.solver,
                    x.solver_mass, x.solver_interior),
        diagnostics(input, setup, plan.mode, properties, fields, species, transport, x.diagnostics,
                    x.diagnostics_helper),
        output(input, plan.mode, fields, magnetic_node, diagnostics.gauss().residual, species, transport,
               x.output_capture,
               x.output_transfer),
        transport(input.domain, mass), mover(input, species, transport, geometry, magnetic_node, species_count),
        local(plan, properties, domain, species, centering, transport, geometry) {
    // Owners reach final addresses before registration. Staging dies before bootstrap.
    {
      auto host = initialization::make_host_field_image<Precision>(
          input.parameters, input.initial_case, input.domain);
      // A restart overwrites the interiors with the checkpoint's E and B; the
      // E ghosts are read by nothing (no E halo exchange precedes the first
      // solve), the B ghosts are refreshed at startup.
      if (plan.restart) {
        const auto at = [](auto view) {
          return io::FieldDestination{view.core_data(), std::size_t(view.stride_y()), std::size_t(view.stride_z())};
        };
        io::read_restart_fields(*plan.restart, input.domain,
            {at(host.electric.template get<typename Fields::Ex>()),
             at(host.electric.template get<typename Fields::Ey>()),
             at(host.electric.template get<typename Fields::Ez>()),
             at(host.magnetic_cell.template get<typename Fields::Bx>()),
             at(host.magnetic_cell.template get<typename Fields::By>()),
             at(host.magnetic_cell.template get<typename Fields::Bz>())});
      }
      // The braced list uploads E before B.
      fields = {data::upload_field(host.electric, x.control),
                data::upload_field(host.magnetic_cell, x.control)};
    }
    // Allocate and publish the zero nodal-B image before the one complete LPFx
    // registration. Physical nodal ghosts stay zero at startup; cell ghosts at
    // missing neighbours retain the already uploaded analytic samples.
    magnetic_node = data::make_zero_field<typename Fields::BNode>(input.domain, x.control);
    current = data::make_zero_field<typename Fields::Current>(input.domain, x.control);
    for (std::size_t s = 0; s != Ns; ++s) {
      if (s < species_count)
        species[s].initialize(plan, s, properties, x.control, x.sort[s]);
      else
        species[s].initialize_unused();
    }
  }

  // Registration, then the two actions that resolve registered operations.
  void prepare() {
    LPFX_PROFILE_SCOPE("TPIC::Prepare");
    transport.prepare(setup, {
        .input = input, .fields = fields, .magnetic_node = magnetic_node,
        .current = current, .species = species, .centering = centering,
        .maxwell = field_solve.maxwell(), .gmres = field_solve.gmres(),
        .diagnostics = diagnostics.workspace(), .gauss = diagnostics.gauss(), .output_moments = output.moments(),
        .geometry = geometry, .executors = x});
    field_advance = std::make_unique<FieldAdvanceActions<Precision, Transport>>(
        fields, magnetic_node, field_solve.maxwell(), transport, x.control, x.field_update);
    deposition = std::make_unique<DepositionActions<Precision, Partial, Kernel, SpeciesPack, Transport>>(
        input, species, magnetic_node, field_solve.maxwell().next_node, current, mass, transport, x.current);
  }

  void host_blocking_barrier() {
    lpfx::check_mpi_error(lpfx::distributed::host_barrier(domain), "TPIC simulation: rank agreement failed");
  }
  // CPU timestamps stay on LPFx's explicit host path for every profile.
  [[nodiscard]] std::array<double, 2> host_blocking_max_over_ranks(const std::array<double, 2> local_seconds) {
    auto seconds = local_seconds;
    if (ranks > 1)
      lpfx::check_mpi_error(
          lpfx::distributed::host_reduce<double, lpfx::distributed::reduction_operation::maximum>(
              domain, std::span<const double>{local_seconds}, std::span<double>{seconds}, 0),
          "TPIC simulation: loop timing maximum-rank reduction failed");
    return seconds;
  }
  void host_blocking_report_memory() { report_initialized_memory(domain, properties.device_ordinal); }
  // The failure exit of a run. Once registration may be live, a local exception
  // must not unwind through the communication's destructor or strand another
  // rank in a collective: the main and every worker failure path end here.
  [[noreturn]] void abort(const char* message) noexcept {
    print_error(message, rank);
    lpfx::MpiCallLease lease;
    static_cast<void>(lpfx::try_acquire_shared_mpi_call(domain, lease));
    MPI_Abort(domain.bootstrap_communicator(), 1);
    std::abort();
  }

  const RunPlan& plan;
  const simulation::PreparedInput& input;
  const lpfx::distributed::CommunicationSetup& setup;
  const lpfx::MpiCallDomain& domain;
  const lpfx::device::DeviceProperties& properties;
  const int rank, ranks;
  const unsigned species_count;
  const bool diagnose;
  const StreamExecutors<Backend, Ns>& x;
  communication::ParticleMigrationGeometry geometry;
  data::DeviceFieldImage<Precision, Backend> fields;
  data::MagneticNodeOwner<Precision, Backend> magnetic_node;
  data::CurrentOwner<Precision, Backend> current;
  data::CompactMassMatrix<Precision, Backend> mass;
  SpeciesPack species;
  initialization::VelocityCentering<Precision, Backend> centering;
  FieldSolveActions<Precision, Transport> field_solve;
  DiagnosticsActions<Precision, SpeciesPack, Transport> diagnostics;
  OutputActions<Precision, SpeciesPack, Transport> output;
  // Borrowers follow owners. Pipeline::close retires distributed and GPU work.
  Transport transport;
  MoverActions<Precision, Partial, SpeciesPack, Transport> mover;
  LocalSpeciesActions<Precision, Partial, SpeciesPack, Transport> local;
  std::unique_ptr<FieldAdvanceActions<Precision, Transport>> field_advance;
  std::unique_ptr<DepositionActions<Precision, Partial, Kernel, SpeciesPack, Transport>> deposition;
};

// Runs the Simulation instantiation that the plan selects for one precision
// pair: particle order and mass deposition kernel. Explicitly instantiated
// once per compiled pair by src/simulation/simulation_dispatch.cpp. The caller
// has agreed on the
// input and selected the device; the setup and executor outlive the call.
// Only an advancing run writes diagnostics and snapshots.
template <class Precision>
[[nodiscard]] RunResult run_simulation(const RunPlan& plan,
    const lpfx::distributed::CommunicationSetup& setup, lpfx::device::ActiveExecutor executor) {
  // RunPlan::prepare rejects matrix-core with unsorted particles and run_distributed
  // agrees on device support before this dispatch.
  LPFX_PROFILE_SCOPE("TPIC::Simulation");
  const auto& input = plan.input;
  // Locals are destroyed in reverse: the runtime (writer and workers), the
  // physics, then the stream table.
  const auto run = [&]<bool Partial, config::MassDepositionKernel Kernel>() {
    ExecutionStreams<Backend> streams(setup.device_properties().device_ordinal);
    const StreamExecutors<Backend, config::max_real_species> x(
        streams, streams.admit(executor, "control"), snapshots_enabled(input, plan.mode),
        input.particles.order != config::ParticleOrder::unsorted, input.particles.species.size());
    Simulation<Precision, Partial, Kernel> simulation(plan, setup, x);
    return Pipeline<Simulation<Precision, Partial, Kernel>>(simulation, streams, x).execute();
  };
  using enum config::MassDepositionKernel;
  const bool partial = input.particles.order == config::ParticleOrder::partial;
  // The matrix kernels exist for double mass compute only; RunPlan::prepare
  // rejects the selection for every other pair.
  if constexpr (std::is_same_v<typename Precision::Compute::Mass, double>) {
    if (input.mass_matrix == matrix_core)
      return partial ? run.template operator()<true, matrix_core>() : run.template operator()<false, matrix_core>();
  } else {
    lpfx::require(input.mass_matrix != matrix_core, "TPIC simulation: matrix-core needs double mass compute");
  }
  return partial ? run.template operator()<true, scalar>() : run.template operator()<false, scalar>();
}

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_DRIVER_SIMULATION_DEVICE_HPP
