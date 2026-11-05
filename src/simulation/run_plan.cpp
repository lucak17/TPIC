#include <tpic/simulation/planning/run_plan.hpp>

#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/io/checkpoint.hpp>

#include <lpfx/data/particle/device/classification.hpp>
#include <lpfx/data/particle/layout.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace tpic::simulation {
namespace {

constexpr const char *byte_overflow = "TPIC particles: particle workspace byte count exceeds size_t";

std::size_t multiply_bytes(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::overflow_error(byte_overflow);
  return left * right;
}

std::size_t add_bytes(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right)
    throw std::overflow_error(byte_overflow);
  return left + right;
}

template <class Precision>
std::size_t species_device_bytes(const config::SpeciesParticlePlan &species, int bins,
                                 config::ParticleOrder order) {
  using Schema = typename data::ParticleSchema<Precision>::type;
  lpfx::PreparedParticleStorage<Schema> storage;
  if (lpfx::try_prepare_particle_layout(species.initial_capacity, storage) !=
      lpfx::ErrorCode::success)
    throw std::overflow_error("TPIC particles: particle layout byte count is not representable");
  // Host/device logical attribute and widest-attribute bytes agree. Deliberately
  // discard host alignment gaps: they are not a native device allocation plan.
  std::size_t sort_bytes = 0;
  if (order != config::ParticleOrder::unsorted) {
    const auto permutation = multiply_bytes(species.initial_capacity, sizeof(std::uint32_t));
    const auto bin_bytes = multiply_bytes(static_cast<std::size_t>(bins), 2 * sizeof(std::uint32_t));
    sort_bytes = add_bytes(add_bytes(permutation, bin_bytes), 2 * sizeof(std::uint32_t));
  }
  const std::size_t classification_bytes = species.classification_capacity == 0 ? 0 : add_bytes(
      multiply_bytes(species.classification_capacity, sizeof(std::uint64_t)),
      sizeof(lpfx::DeviceParticleRegisteredClassificationData));
  return add_bytes(add_bytes(storage.logical_bytes(), sort_bytes), classification_bytes);
}

} // namespace

ParticleByteReport report_particle_bytes(const PreparedInput &input) {
  return config::dispatch_precision(input.precision, [&]<class Precision> {
    using Particles = data::ParticleSchema<Precision>;
    ParticleByteReport report;
    report.particle_record_bytes = Particles::type::record_width;
    for (const auto &species : input.particles.species)
      report.device_bytes_lower_bound = add_bytes(report.device_bytes_lower_bound,
          species_device_bytes<Precision>(species, input.domain.particle_cell_count_with_halo,
                                          input.particles.order));
    return report;
  });
}

PreparedInput prepare_input(config::SimulationParameters parameters, InputOptions options) {
  // Every later preparation step relies on this count.
  config::require_species_count(parameters.ns, parameters.species.size());
  (void)config::precision_name(options.precision); // rejects an invalid selection
  PreparedInput result;
  result.precision = options.precision;
  result.mass_apply = options.mass_apply;
  result.parameters = std::move(parameters);
  const auto &p = result.parameters;
  result.physics = config::prepare_physics(p);
  result.maxwell_schedule = ecsim::resolve_maxwell_apply(result.mass_apply, result.physics.smoothing_enabled);
  result.communicator_size = options.communicator_size == 0 ?
      config::configured_process_count(p.topology) : options.communicator_size;
  result.domain = config::prepare_domain(p.grid, p.topology, options.rank, result.communicator_size);
  // Count admission precedes the recipe tables and every device allocation.
  result.particles = config::prepare_particle_plan(p, result.domain, options.order);
  result.output = config::prepare_output(p);
  // A restart replaces the outputs of the run it continues.
  result.output.overwrite_output = options.overwrite_output || options.restart;
  result.initial_case = initialization::prepare_initial_case(p);
  result.particles = initialization::prepare_bounded_particle_plan(result.particles, result.initial_case, p,
                                                                   result.domain);
  // Only a bounded support can hold no subcell centre.
  for (std::size_t s = 0; s < result.particles.species.size(); ++s)
    if (result.particles.species[s].global_count == 0)
      result.warnings.push_back("Species " + std::to_string(s) + " loads no particles: no subcell centre lies "
                                "within GaussianExpansionCutoff widths of the centre");
  result.unsupported_requests = result.physics.unsupported_requests;
  result.mass_matrix = options.mass_matrix;
  if (!config::precision_pair_compiled(options.precision))
    result.unsupported_requests.emplace_back("Precision pair " + config::precision_name(options.precision) +
                                             " is not compiled into this build "
                                             "(TPIC_BUILD_ALL_PRECISION_PAIRS, TPIC_PRECISION_PAIRS)");
  if (options.mass_matrix != config::MassDepositionKernel::scalar &&
      options.mass_matrix != config::MassDepositionKernel::matrix_core)
    throw std::invalid_argument("TPIC mass matrix: invalid kernel selection");
  if (options.mass_matrix != config::MassDepositionKernel::scalar) {
#if !TPIC_CONFIG_MATRIX_MASS
    result.unsupported_requests.emplace_back(
        "Matrix-core mass deposition requires a TPIC CUDA or HIP device build");
#endif
    if (options.order == config::ParticleOrder::unsorted)
      result.unsupported_requests.emplace_back(
          "Matrix-core mass deposition requires --order full or partial");
    if (!config::dispatch_precision(options.precision, [&]<class P> {
          return std::is_same_v<typename P::Compute::Mass, double>;
        }))
      result.unsupported_requests.emplace_back(
          "Matrix-core mass deposition requires double mass compute precision");
  }
  if (p.output.test_particles_cycle > 0)
    result.unsupported_requests.emplace_back(
        "TestParticlesOutputCycle: test-particle output is not implemented");
  if (!p.unknown_keys.empty()) {
    std::string ignored = "Unknown input keys ignored:";
    for (const auto &key : p.unknown_keys) ignored += (&key == &p.unknown_keys.front() ? " " : ", ") + key;
    result.warnings.push_back(std::move(ignored));
  }
  if (p.output.restart_cycle > 0 && result.output.checkpoint_engine == config::OutputEngineKind::Vtk)
    result.unsupported_requests.emplace_back("RestartWriteMethod: checkpoints are written with adios2 or hdf5");
  if (p.output.restart_cycle > 0 && result.output.checkpoint_engine == config::OutputEngineKind::Default)
    result.warnings.emplace_back("RestartOutputCycle: this build has no checkpoint backend (TPIC_ENABLE_ADIOS2 "
                                 "or TPIC_ENABLE_HDF5); no checkpoints will be written");

  int electrons = 0, ions = 0;
  double electron_density = 0, ion_density = 0;
  bool one_qom_per_sign = true;
  const auto qom = initialization::first_charge_magnitudes(p);
  for (const auto &species : p.species) {
    electrons += species.q_over_m < 0;
    ions += species.q_over_m > 0;
    (species.q_over_m < 0 ? electron_density : ion_density) += species.rho_init;
    one_qom_per_sign = one_qom_per_sign && std::abs(species.q_over_m) == qom[species.q_over_m > 0];
  }
  const auto* whistler = std::get_if<initialization::WhistlerCase>(&result.initial_case);
  const auto off_unit = [](double value) { return std::abs(value - 1) > 1e-12; };
  if (whistler && whistler->mode == config::WhistlerSeedMode::Eigenmode && whistler->delta_b != 0 &&
      (!one_qom_per_sign || off_unit(qom[1]) || off_unit(electron_density) || off_unit(ion_density)))
    result.warnings.emplace_back("Whistler: the eigenmode frequency assumes one qom per sign, ion qom = 1 and "
                                 "rhoINIT summing to 1 over the electrons and over the ions; this seed is not an "
                                 "exact eigenmode");
  const bool repeated_whistler_current = whistler && whistler->current_amplitude != 0 && electrons > 1;
  const bool repeated_tgv_current =
      std::holds_alternative<initialization::TaylorGreenVortexCase>(result.initial_case) &&
      p.taylor_green_vortex_magnetic_amplitude != 0 && (electrons > 1 || ions > 1);
  if (repeated_whistler_current || repeated_tgv_current)
    result.unsupported_requests.emplace_back(
        "Multiple same-sign current carriers would each carry the full wave current; this loading needs a physics decision before admission");
  return result;
}

data::CheckpointHeader prepare_checkpoint_header(const PreparedInput &input) {
  data::CheckpointHeader header;
  const auto &process = input.domain.field_neighborhood.topology().input().process_shape();
  header.species = static_cast<std::int32_t>(input.particles.species.size());
  header.processes = {static_cast<std::int32_t>(process.x()), static_cast<std::int32_t>(process.y()),
                      static_cast<std::int32_t>(process.z())};
  header.global_cells = input.domain.global_cells;
  header.lengths = input.domain.lengths;
  header.dt = input.parameters.dt;
  header.c = input.parameters.c;
  for (std::size_t face = 0; face < input.physics.field_boundary.size(); ++face) {
    header.faces[face] = static_cast<std::int32_t>(input.physics.field_boundary[face]);
    header.faces[6 + face] = static_cast<std::int32_t>(input.physics.particle_boundary[face]);
  }
  header.storage_bytes = config::dispatch_precision(input.precision, [&]<class Precision> {
    using Storage = typename Precision::Storage;
    return std::array<std::int32_t, 5>{sizeof(typename Storage::Position), sizeof(typename Storage::Velocity),
                                       sizeof(typename Storage::Weight), sizeof(typename Storage::Electric),
                                       sizeof(typename Storage::Magnetic)};
  });
  for (std::size_t s = 0; s < input.particles.species.size(); ++s) {
    header.q_over_m[s] = input.parameters.species[s].q_over_m;
    header.global_count[s] = input.particles.species[s].global_count;
  }
  return header;
}

namespace {

// A restart continues the state of its checkpoint: the mesh, the boundaries,
// the time step, the light speed, the species and the stored types are its
// own. The process grid, the compute precision, the numerics and the outputs
// may change.
void admit_restart(PreparedInput &input, const data::CheckpointHeader &checkpoint, RunMode mode) {
  const auto run = prepare_checkpoint_header(input);
  const auto reject = [&](bool differs, const std::string &what) {
    if (differs) input.unsupported_requests.push_back("Restart: " + what + " differs from the checkpoint's");
  };
  reject(run.global_cells != checkpoint.global_cells, "the mesh (nxc, nyc, nzc)");
  reject(run.lengths != checkpoint.lengths, "the domain (Lx, Ly, Lz)");
  reject(run.faces != checkpoint.faces, "a field or particle boundary");
  reject(run.dt != checkpoint.dt, "dt");
  reject(run.c != checkpoint.c, "c");
  reject(run.species != checkpoint.species || run.q_over_m != checkpoint.q_over_m, "the species (ns, qom)");
  reject(run.storage_bytes != checkpoint.storage_bytes, "the storage precision of a particle or field value");
  if (advances(mode) && input.parameters.ncycles <= checkpoint.cycle)
    input.unsupported_requests.push_back("Restart: ncycles = " + std::to_string(input.parameters.ncycles) +
                                         " is not after the checkpoint cycle " + std::to_string(checkpoint.cycle));
}

// The output backends of an advancing run.
} // namespace

std::vector<std::string> unavailable_output_backends(const PreparedInput &input) {
  std::vector<std::string> missing;
  const auto require_backend = [&](bool enabled, config::OutputEngineKind engine, const char* channel) {
    if (enabled && !config::output_engine_available(engine))
      missing.push_back(
          std::string(channel) + " output backend '" + std::string(config::output_engine_name(engine)) +
          "' is unavailable; enable " +
          (engine == config::OutputEngineKind::Adios2 ? "TPIC_ENABLE_ADIOS2" :
           engine == config::OutputEngineKind::Hdf5 ? "TPIC_ENABLE_HDF5" :
           "a supported backend") + " at build time");
  };
  require_backend(input.output.fields_enabled(), input.output.field_engine, "field");
  require_backend(input.output.particles_enabled(), input.output.particle_engine, "particle");
  require_backend(input.output.checkpoints_enabled(), input.output.checkpoint_engine, "checkpoint");
  return missing;
}

RunPlan RunPlan::prepare(config::SimulationParameters parameters, InputOptions options,
                         RunMode mode, RuntimeDebugOptions debug,
                         std::optional<data::CheckpointLocation> checkpoint) {
  auto input = prepare_input(std::move(parameters), options);
  if (checkpoint) admit_restart(input, checkpoint->header, mode);
  if (!input.host_admitted()) {
    std::string message = "TPIC run plan: unsupported requests";
    for (const auto& request : input.unsupported_requests) message += "\n" + request;
    throw std::invalid_argument(message);
  }
  // GPU startup builds the wall plans of every face.
  for (auto face : input.physics.field_boundary) {
    const auto& rule = config::field_boundary_rule(face);
    if (!rule.run_rejection.empty()) throw std::invalid_argument(std::string(rule.run_rejection));
  }
  if (advances(mode))
    if (const auto missing = unavailable_output_backends(input); !missing.empty())
      throw std::invalid_argument("TPIC " + missing.front());
  std::optional<data::PreparedRestart> restart;
  if (checkpoint) {
    restart = io::prepare_restart(std::move(*checkpoint), input.domain);
    const auto species = input.particles.species.size();
    const auto& header = restart->checkpoint.header;
    input.particles = config::prepare_particle_plan(input.parameters, input.domain, options.order,
                                                    std::span(restart->local_count).first(species),
                                                    std::span(header.global_count).first(species));
  }
  auto loads = initialization::prepare_particle_loads(input.parameters, input.domain, input.particles);
  // Restored velocities are never centred again.
  if (restart)
    for (auto& load : loads) load.center_velocity = false;
  return RunPlan(std::move(input), std::move(loads), std::move(restart), mode, debug);
}

RunPlan::RunPlan(PreparedInput prepared, std::vector<initialization::ParticleLoadPlan> loads,
                 std::optional<data::PreparedRestart> restored, RunMode mode, RuntimeDebugOptions debug)
    : input(std::move(prepared)), particle_loads(std::move(loads)), restart(std::move(restored)),
      mode(mode), debug(debug) {}

} // namespace tpic::simulation
