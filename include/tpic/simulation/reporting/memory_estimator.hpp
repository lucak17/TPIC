#ifndef TPIC_SIMULATION_REPORTING_MEMORY_ESTIMATOR_HPP
#define TPIC_SIMULATION_REPORTING_MEMORY_ESTIMATOR_HPP

#include <lpfx/core/error_reporting.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/api/field/field.hpp>
#include <lpfx/algorithm/reduce/device/workspace.hpp>
#include <lpfx/algorithm/particle/device/sort.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/data/particle/device/classification.hpp>
#include <tpic/communication/mass_halo/mass_halo_plan.hpp>
#include <tpic/communication/particles/particle_wire.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/config/output_selection.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/output/gauss_sample.hpp>
#include <tpic/data/output/output_moment_schema.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_stencil.hpp>
#include <tpic/ecsim/gmres/gmres.hpp>
#include <tpic/ecsim/maxwell/maxwell.hpp>
#include <tpic/initialization/particles/particle_load_plan.hpp>
#include <tpic/data/output/scalar_sample.hpp>
#include <tpic/data/output/output_snapshot.hpp>
#include <tpic/io/checkpoint.hpp>
#include <tpic/data/mass_matrix/mass_matrix_layout.hpp>
#include <lpfx/execution/host/channel.hpp>
#include <lpfx/execution/host/fence.hpp>
#include <lpfx/execution/host/worker.hpp>
#include <tpic/simulation/execution/deposition_ledger.hpp>
#include <tpic/simulation/execution/execution_streams.hpp>
#include <tpic/simulation/execution/execution_layout.hpp>
#include <tpic/simulation/timing/stage_timing.hpp>
#include <tpic/simulation/reporting/startup_report.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tpic::simulation {

// Sizing primitives of the estimate; the estimator tests check them directly.

// A size whose sum and product throw instead of wrapping.
class CheckedSize {
public:
  constexpr CheckedSize() noexcept = default;
  constexpr explicit CheckedSize(std::size_t value) noexcept : value_(value) {}
  [[nodiscard]] constexpr std::size_t value() const noexcept { return value_; }

  [[nodiscard]] friend constexpr CheckedSize operator+(CheckedSize a, CheckedSize b) {
    if (b.value_ > std::numeric_limits<std::size_t>::max() - a.value_)
      throw std::overflow_error("TPIC memory estimate: byte sum overflow");
    return CheckedSize(a.value_ + b.value_);
  }
  [[nodiscard]] friend constexpr CheckedSize operator*(CheckedSize a, CheckedSize b) {
    if (a.value_ != 0 && b.value_ > std::numeric_limits<std::size_t>::max() / a.value_)
      throw std::overflow_error("TPIC memory estimate: byte product overflow");
    return CheckedSize(a.value_ * b.value_);
  }
  constexpr CheckedSize& operator+=(CheckedSize other) { return *this = *this + other; }

private:
  std::size_t value_{};
};

template <class Schema, class Backend>
[[nodiscard]] std::size_t field_storage_bytes(const config::PreparedDomain& domain) {
  typename lpfx::field::device<Schema, Backend>::storage_plan_type plan;
  lpfx::HaloDepth3<std::uint32_t> halo;
  lpfx::check_error<std::invalid_argument>(
      lpfx::try_make_uniform_halo_depth3(config::PreparedDomain::field_halo, halo),
      "TPIC memory estimate: LPFx storage plan rejected");
  const auto halos = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return lpfx::make_field_halo_pack<lpfx::LogicalIndexPolicy32,
        typename Schema::template component<I>...>(
            (static_cast<void>(I), halo)...);
  }(std::make_index_sequence<Schema::component_count>{});
  lpfx::check_error<std::invalid_argument>(
      lpfx::try_prepare_field_storage(domain.field_neighborhood.local_partition().local_cells(),
                                      halos, plan),
      "TPIC memory estimate: LPFx storage plan rejected");
  return plan.total_bytes();
}

template <class Precision, class Backend>
[[nodiscard]] std::size_t particle_storage_bytes(std::uint32_t capacity) {
  lpfx::PreparedDeviceParticleStorage<typename data::ParticleSchema<Precision>::type,
                                      Backend> plan;
  lpfx::check_error<std::invalid_argument>(
      lpfx::try_prepare_device_particle_storage(capacity, plan),
      "TPIC memory estimate: LPFx storage plan rejected");
  return plan.total_bytes();
}

// Arrival chunks hold decoded attributes only, in LPFx's padded SoA layout.
// They have no widest-attribute sorting auxiliary and are not a second owner.
template <class Precision, class Backend>
[[nodiscard]] std::size_t arrival_chunk_bytes(std::uint32_t capacity) {
  lpfx::PreparedParticleLayout<typename data::ParticleSchema<Precision>::type,
                               lpfx::DeviceSpace<Backend>> plan;
  lpfx::check_error<std::invalid_argument>(lpfx::try_prepare_particle_layout(capacity, plan),
                                           "TPIC memory estimate: LPFx storage plan rejected");
  return plan.total_bytes();
}

[[nodiscard]] inline std::size_t classification_bytes(std::size_t capacity) {
  return (CheckedSize{capacity} * CheckedSize{sizeof(std::uint64_t)} +
          CheckedSize{sizeof(lpfx::DeviceParticleRegisteredClassificationData)}).value();
}

template <class Precision>
[[nodiscard]] std::size_t reduction_storage_bytes(std::size_t capacity,
    const lpfx::device::DeviceProperties& properties) {
  lpfx::PreparedDeviceReduction<typename Precision::Compute::Reduction> plan;
  lpfx::check_error<std::invalid_argument>(
      lpfx::try_prepare_device_reduction(capacity, properties, plan),
      "TPIC memory estimate: LPFx storage plan rejected");
  return plan.total_bytes();
}

[[nodiscard]] inline std::size_t sort_workspace_bytes(std::uint32_t capacity, int bins,
    const lpfx::device::DeviceProperties& properties) {
  lpfx::PreparedDeviceParticleSortWorkspace<> plan;
  lpfx::check_error<std::invalid_argument>(
      lpfx::try_prepare_device_particle_sort_workspace(capacity, bins, properties, plan),
      "TPIC memory estimate: LPFx storage plan rejected");
  return plan.total_bytes();
}

// A one-sample face/edge/corner pack has this many directed samples. Counting
// all 26 directions also covers periodic self routes and bounds physical cases.
[[nodiscard]] inline CheckedSize boundary_samples(const config::PreparedDomain& domain,
                                                  bool nodes) {
  CheckedSize interior{1}, expanded{1};
  for (const int cells : domain.local_cells) {
    const auto extent = CheckedSize{static_cast<std::size_t>(cells)} + CheckedSize{nodes ? 1U : 0U};
    interior = interior * extent;
    expanded = expanded * (extent + CheckedSize{2});
  }
  return CheckedSize{expanded.value() - interior.value()};
}

namespace detail {

// Cold full/partial/unsorted estimate for the selected precision profile,
// called before constructing the selected pack. Each persistent field owner is
// planned with its storage role; the solver work fields and the reductions
// are FP64 in every supported profile. --initialize also constructs the
// solver owners and is estimated with them; it has no snapshot or scalar
// staging. A closed one-rank population needs no hypothetical growth
// replacement. With remote ranks, retain the incoming-particle and transient-reallocation reserve.
// Transport is an allowance, independent of LPFx private allocation layouts.
// The estimate is informational: it never rejects a run.
template <class Precision, class Backend>
[[nodiscard]] MemoryEstimate estimate_precision_memory(
    const RunPlan& plan, const lpfx::device::DeviceProperties& properties) {
  using Size = CheckedSize;
  using Fields = data::FieldSchema<Precision>;
  using SolverFields = ecsim::SolverFieldSchema<Precision>;
  using OutputFields = typename data::OutputMomentSchema<Precision>::type;
  using GaussFields = data::GaussFieldSchema<Precision>;
  // Solver types the solver work fields, Krylov the GMRES basis, SolverReduction
  // the GMRES dense state and, when Krylov is narrower, its residual vectors.
  using Solver = ecsim::SolverStorage<Precision>;
  using Krylov = typename Precision::Compute::Electric;
  using SolverReduction = typename Precision::Compute::SolverReduction;
  using Reduction = typename Precision::Compute::Reduction;
  using Magnetic = typename Precision::Storage::Magnetic;
  using Current = typename Precision::Storage::Current;
  using Timing = StageTiming<Backend>;
  // Allowances for native data that TPIC does not size: LPFx peer recipes and
  // registration metadata per peer, and device descriptors with their 16-byte
  // image alignment (< 48 bytes) per field binding or species migration plan.
  constexpr std::size_t peer_metadata_allowance = 128;
  constexpr std::size_t descriptor_allowance = 64 * 1024;
  constexpr std::size_t neighbor_directions = 26;
  // Payload images per bound component: outgoing, incoming, snapshot and
  // alignment.
  constexpr std::size_t payload_images = 4;
  // Velocity centering owns its reductions with their device results and status.
  constexpr std::size_t centering_reductions = initialization::velocity_centering_reductions;
  // Wire slots per species: max(outgoing, transit) and incoming.
  constexpr std::size_t wire_slots = 2;

  const auto& input = plan.input;
  const bool profiling = plan.debug.profile_stages;
  const bool output_active = advances(plan.mode);
  MemoryEstimate result;
  // The supported particle boundaries wrap or reflect existing records; they
  // never create particles. With a one-rank particle topology (field
  // periodicity may differ) no remote owner can deliver arrivals, so
  // owner/slot/sort capacities cannot grow.
  const bool population_fixed = input.particles.fixed_local_population;
  const auto ns = input.particles.species.size();
  const bool partial = input.particles.order == config::ParticleOrder::partial;
  const bool sorted = input.particles.order != config::ParticleOrder::unsorted;
  const bool field_output = output_active && input.output.fields_enabled();
  const bool particle_output = output_active && input.output.particles_enabled();
  const bool checkpoint_output = output_active && input.output.checkpoints_enabled();
  // Particle output and checkpoints share one particle capture.
  const bool particle_capture = particle_output || checkpoint_output;
  const bool snapshot_output = field_output || particle_capture;
  // The output and the checkpoint writer threads.
  result.worker_count = ExecutionLayout::workers(ns) + (field_output || particle_output) + checkpoint_output;
  // Explicit fixed roles plus per-species local/sort and optional capture/transfer.
  result.stream_count = ExecutionLayout::streams(ns, sorted, snapshot_output);
  result.validation_count = ExecutionLayout::validations(ns);
  // Move/owner per species, with resident J+M and reader closure
  // only in partial mode, plus shared field/moment publication events. LPFx's own
  // events and validation readiness events remain native runtime overhead.
  result.dependency_event_count =
      ExecutionLayout::dependency_events(ns, partial, field_output, particle_capture);
  // The pipeline holds every species slot of the compiled capacity; only ns of
  // the local-species workers start a thread.
  constexpr std::size_t capacity = config::max_real_species;
  result.runtime_host_bytes = (Size{sizeof(ExecutionStreams<Backend>)} +
      Size{partial ? sizeof(DepositionLedger<capacity, true>) : sizeof(DepositionLedger<capacity, false>)} +
      Size{sizeof(lpfx::HostChannel<DepositionInput, 2 * capacity>)} +
      Size{capacity * sizeof(lpfx::HostChannel<MomentQuantity, 2>)} +
      Size{ExecutionLayout::host_fences(capacity) * sizeof(lpfx::HostFence)} +
      Size{ExecutionLayout::workers(capacity)} * Size{sizeof(lpfx::HostWorker<Epoch>)}).value();
  const Size timers{ExecutionLayout::stage_timers(ns)};
  result.timing_host_bytes = (timers * Size{sizeof(Timing)}).value();
  if (profiling)
    result.timing_event_count = (Size{2} * Size{timing_event_pairs(ns)} * timers).value();

  const Size electric{field_storage_bytes<typename Fields::E, Backend>(input.domain)};
  const Size magnetic_node{field_storage_bytes<typename Fields::BNode, Backend>(input.domain)};
  const Size magnetic_cell{field_storage_bytes<typename Fields::BCell, Backend>(input.domain)};
  const Size current{field_storage_bytes<typename Fields::Current, Backend>(input.domain)};
  const Size solver_node{field_storage_bytes<typename SolverFields::Node, Backend>(input.domain)};
  const Size solver_cell{field_storage_bytes<typename SolverFields::Cell, Backend>(input.domain)};
  // Startup E, B-node, J and B-cell, the Maxwell workspace, then the two
  // registered Gauss node fields, present even with the diagnostic off. Species
  // density is deposited on demand into the output moment owner, so the
  // stepping path needs no separate rho owners.
  const Size gauss_node{field_storage_bytes<typename GaussFields::Node, Backend>(input.domain)};
  result.field_device_bytes = (electric + magnetic_node + current + magnetic_cell +
      Size{8} * solver_node + solver_cell + magnetic_cell + magnetic_node + Size{2} * gauss_node).value();
  // One registered output owner is stable even with output disabled. Packing
  // and snapshot staging are allocated only for active field snapshots.
  const Size output_moments{field_storage_bytes<OutputFields, Backend>(input.domain)};
  // Field upload stages the largest owner on the host and retires the staging.
  const Size field_upload_host = Size{4} * Size{std::max({electric.value(),
      magnetic_node.value(), magnetic_cell.value(), current.value(), solver_node.value(),
      solver_cell.value(), output_moments.value()})};
  Size output_device = output_moments, output_staging, output_initial_pinned;
  Size output_pinned, output_owned;
  if (field_output) {
    const Size scalar = Size{static_cast<std::size_t>(input.domain.interior_node_count)} *
                        Size{sizeof(double)};
    const Size cell_scalar = Size{static_cast<std::size_t>(input.domain.interior_cell_count)} *
                             Size{sizeof(double)};
    // Selected immutable device/pinned slabs plus only the temporary width
    // required by total contributions without a selected species array.
    // The CPU writer owns one selected-array image, returned for reuse each cycle.
    const config::PreparedOutputSelection selection(input.output);
    Size snapshot;
    for (const auto& array : selection.arrays())
      snapshot += Size{static_cast<std::size_t>(array.components)} * (array.cell ? cell_scalar : scalar);
    output_staging = snapshot + Size{data::field_snapshot_scratch_components<Backend>(
                                         selection.scratch_component_count())} * scalar;
    output_device += output_staging;
    output_initial_pinned = snapshot;
    output_pinned = output_initial_pinned;
    output_owned = output_initial_pinned;
  }
  if (checkpoint_output) {
    // The checkpoint's E at the local nodes and B at the local cells, and the
    // writer's prefix_start of each species, (nx + 1) ny nz row starts.
    std::size_t nodes = 1, cells = 1, index = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const auto extent = static_cast<std::size_t>(input.domain.local_cells[axis]);
      nodes *= extent + 1;
      cells *= extent;
      index *= axis == 0 ? extent + 1 : extent;
    }
    output_owned += Size{3} * (Size{nodes} * Size{sizeof(typename Precision::Storage::Electric)} +
                               Size{cells} * Size{sizeof(Magnetic)}) +
                    Size{ns} * Size{index} * Size{sizeof(std::uint32_t)};
  }

  const auto mass = data::prepare_mass_matrix_layout(input.domain);
  result.mass_device_bytes =
      (Size{mass.storage_size()} * Size{sizeof(typename Precision::Storage::Mass)}).value();
  const auto nodal_boundary = boundary_samples(input.domain, true);
  const auto cell_boundary = boundary_samples(input.domain, false);
  using MassHaloPlan = communication::MassHaloPlan<Precision>;
  // Payload images and uploaded numerical records match the prepared plan
  // exactly; the per-peer allowance covers the rest of each phase.
  Size transport = Size{MassHaloPlan::prepare(input.domain).device_bytes()} +
      Size{2 * MassHaloPlan::template phase_type<true>::message_plan_type::maximum_peer_count *
           peer_metadata_allowance};
  // Halo payload bytes per boundary sample of every bound field: B-node, J
  // (add and copy), the output moments, the four bound solver vectors
  // (theta, input, smooth_a, smooth_b) and the Gauss charge (add and copy) and
  // its smoothing partner; B-cell, curl_cell and next B-cell.
  // rhs/image/mass_image/curl_node, next B-node and the Gauss cells have no
  // bound channel.
  constexpr std::size_t node_payload_width = 3 * sizeof(Magnetic) + 6 * sizeof(Current) +
      OutputFields::component_count * sizeof(Reduction) + 12 * sizeof(Solver) + 3 * sizeof(Reduction);
  constexpr std::size_t cell_payload_width =
      3 * sizeof(Magnetic) + 3 * sizeof(Solver) + 3 * sizeof(Magnetic);
  constexpr std::size_t field_bindings = 14;
  transport += (Size{node_payload_width} * nodal_boundary + Size{cell_payload_width} * cell_boundary) *
                   Size{payload_images} +
               Size{field_bindings} * Size{descriptor_allowance};

  const auto maximum_population = static_cast<std::size_t>(input.particles.max_species_local_count());
  using Wire = typename communication::ParticleMigrationWire<Precision>::Contract;
  // Each wire slot is widened by the aligned-packet slack when non-empty, as
  // LPFx prepares migration storage.
  constexpr auto wire_slot_slack = wire_slots * static_cast<std::size_t>(Wire::wire_extent_slack);
  Size particles, sort, growth, migration_reserved;
  // Every species plan carries the reciprocal per-peer and aggregate bound.
  for (const auto& species : input.particles.species) {
    const auto seed_records = static_cast<std::size_t>(species.initial_migration_capacity);
    std::uint32_t slot_capacity{};
    if (!input.particles.bypass_local_migration)
      lpfx::check_error<std::invalid_argument>(
          lpfx::try_prepare_particle_capacity(species.initial_capacity, slot_capacity),
          "TPIC memory estimate: LPFx storage plan rejected");
    const auto slots = input.particles.bypass_local_migration ? 0 : classification_bytes(slot_capacity);
    particles += Size{particle_storage_bytes<Precision, Backend>(species.initial_capacity)} + Size{slots};
    // One pinned image of the stored records per species, created lazily for
    // the active particles without owner headroom and lent to the writers.
    if (particle_capture)
      output_pinned += Size{static_cast<std::size_t>(species.local_count)} *
                       Size{data::ParticleSchema<Precision>::type::record_width};
    // Both wire images, compaction summaries and migration admission metadata.
    // LPFx pads each non-empty wire slot by the contract's aligned-packet
    // slack (zero for packed contracts); slot alignment is in the allowance.
    transport += Size{wire_slots} * Size{seed_records} * Size{Wire::record_wire_width} +
                 Size{seed_records != 0 ? wire_slot_slack : 0} + Size{descriptor_allowance};
    if (input.communicator_size > 1 && !input.particles.bypass_local_migration) {
      // Bounded reserve_particles provisions a decoded chunk up to the wire
      // seed (clamped to the local final bound), and exact incoming slots at
      // the seed. The owner-sized logical final bound remains unchanged;
      // observed arrivals can grow the retained storage.
      const auto arrival_records = std::min(species.initial_migration_capacity,
                                            species.initial_capacity);
      const auto reserved = Size{arrival_chunk_bytes<Precision, Backend>(arrival_records)} +
                            Size{classification_bytes(seed_records)};
      migration_reserved += reserved;
      transport += reserved;
      // Hole/filler lists allocate lazily from observed removals, so they are
      // not startup storage. The initial unordered compaction summary is only
      // 16 bytes, included in the metadata allowance above.
    }
    std::size_t growth_capacity{};
    if (!population_fixed) {
      // Budget old owners plus one replacement at twice initial capacity (or a
      // seed-sized arrival on an empty rank). Sum every species' simultaneously
      // live replacements: main/receive classification, decoded final chunks,
      // preserved wire images, compaction lists and optional sorter workspace.
      const auto growth_count = std::max(
          (Size{2} * Size{species.initial_capacity}).value(), seed_records);
      lpfx::check_error<std::invalid_argument>(
          lpfx::try_prepare_particle_capacity(growth_count, growth_capacity),
          "TPIC memory estimate: LPFx storage plan rejected");
      if (growth_capacity > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("TPIC memory estimate: particle growth capacity overflow");
      std::size_t growth_slots{};
      lpfx::check_error<std::invalid_argument>(
          lpfx::try_prepare_particle_capacity(growth_capacity, growth_slots),
          "TPIC memory estimate: LPFx storage plan rejected");
      if (growth_slots > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("TPIC memory estimate: classification growth capacity overflow");
      const auto growth_aux =
          Size{growth_capacity} * Size{wire_slots * Wire::record_wire_width + 2 * sizeof(std::uint32_t)} +
          Size{growth_capacity != 0 ? wire_slot_slack : 0} +
          Size{2} * Size{classification_bytes(growth_slots)} +
          Size{arrival_chunk_bytes<Precision, Backend>(static_cast<std::uint32_t>(growth_capacity))};
      growth += Size{particle_storage_bytes<Precision, Backend>(
                    static_cast<std::uint32_t>(growth_capacity))} + growth_aux;
    }
    if (sorted) {
      const int bins = input.domain.particle_cell_count_with_halo;
      sort += Size{sort_workspace_bytes(species.initial_capacity, bins, properties)};
      if (!population_fixed)
        growth += Size{sort_workspace_bytes(static_cast<std::uint32_t>(growth_capacity), bins,
                                            properties)};
      // Rotating sort reuses the particle slab's widest-attribute auxiliary,
      // already counted by particle_storage_bytes. Only bins/scan/permutation are extra.
    }
  }

  const Size centering = Size{centering_reductions} *
      Size{reduction_storage_bytes<Precision>(maximum_population, properties)} +
      Size{centering_reductions * sizeof(Reduction) + sizeof(unsigned)};
  const auto restart = static_cast<std::size_t>(input.physics.gmres_restart);
  const auto unknowns = Size{static_cast<std::size_t>(mass.interior_count)} *
                        Size{SolverFields::Node::component_count};
  const Size batch{ecsim::gmres_batch_capacity(restart)};
  constexpr bool refined = !std::is_same_v<Krylov, SolverReduction>;
  const Size residual_vectors{refined ? ecsim::gmres_residual_vector_count : 0};
  const Size solver = Size{ecsim::gmres_vector_count(restart, refined)} * unknowns * Size{sizeof(Krylov)} +
      residual_vectors * unknowns * Size{sizeof(SolverReduction)} +
      (Size{ecsim::gmres_state_count(restart)} + batch) * Size{sizeof(SolverReduction)} +
      batch * Size{ecsim::gmres_reduction_blocks(unsigned(unknowns.value()), properties)} *
          Size{sizeof(SolverReduction)};
  Size diagnostics, diagnostics_pinned;
  if (output_active) {
    // Fixed transformed contributions remain bounded after particle growth.
    // Every run reports its final energy, independently of both cadences.
    const Size scalars{data::scalar_count(ns)};
    diagnostics = (scalars + Size{2 * data::scalar_scratch_values}) *
                  Size{sizeof(Reduction)};
    diagnostics_pinned = scalars * Size{sizeof(Reduction)};
    // The Gauss values, their pinned image and the block partials of each, and
    // the stash, first and residual cells.
    if (input.output.gauss_cycle > 0 || input.output.output_gauss) {
      constexpr std::size_t gauss = data::gauss_scalar_count;
      diagnostics += Size{gauss * (1 + data::contribution_count) * sizeof(Reduction)} +
          Size{3} * Size{field_storage_bytes<typename GaussFields::Cell, Backend>(input.domain)};
      diagnostics_pinned += Size{gauss * sizeof(Reduction)};
    }
  }
  // One flag per ExecutionLayout validation slot, including in --initialize's pipeline.
  const Size validation = Size{result.validation_count} * Size{sizeof(unsigned)};
  // GMRES keeps its pinned controls even for initialize-only runs. Each
  // loading/centering status is a cold four-byte allocation, retired at setup.
  const Size persistent_pinned = validation + diagnostics_pinned +
      Size{ecsim::gmres_control_count * sizeof(SolverReduction)} + output_initial_pinned;
  // Conservative cold peak: field staging, one aligned mass plan, and the
  // largest single LPFx field descriptor upload (bounded above in transport).
  // Actual descriptor bytes include allocator padding and are budget-admitted
  // internally by LPFx; they are never retained through the timestep loop.
  // A restart reads its particles through one staging block, each attribute 64-byte aligned.
  const Size restart_staging = plan.restart ? Size{io::restart_staging_records} *
      Size{data::ParticleSchema<Precision>::type::record_width} +
      Size{data::ParticleSchema<Precision>::type::attribute_count * 64} : Size{};
  result.startup_pinned_host_estimate_bytes = (persistent_pinned + Size{std::max({
      field_upload_host.value(), restart_staging.value(), neighbor_directions * peer_metadata_allowance,
      descriptor_allowance, sizeof(unsigned)})}).value();
  const Size persistent = particles + Size{result.field_device_bytes} +
      Size{result.mass_device_bytes} + solver + centering + sort + diagnostics + output_device +
      validation;

  result.particle_device_bytes = particles.value();
  result.solver_device_bytes = solver.value();
  result.centering_device_bytes = centering.value();
  result.sort_device_bytes = sort.value();
  result.diagnostics_device_bytes = diagnostics.value();
  result.diagnostics_pinned_host_bytes = diagnostics_pinned.value();
  result.persistent_pinned_host_bytes = persistent_pinned.value();
  result.output_initial_pinned_host_bytes = output_initial_pinned.value();
  result.output_pinned_host_bytes = output_pinned.value();
  result.output_owned_host_bytes = output_owned.value();
  result.validation_device_bytes = validation.value();
  result.validation_host_bytes = validation.value();
  result.output_device_bytes = output_device.value();
  result.output_staging_device_bytes = output_staging.value();
  result.persistent_device_bytes = persistent.value();
  result.transport_device_bytes = transport.value();
  result.growth_reserve_bytes = growth.value();
  result.migration_reserved_device_bytes = migration_reserved.value();
  result.estimated_device_bytes =
      (persistent + transport + Size{result.runtime_reserve_bytes}).value();
  return result;
}

} // namespace detail

template <class Backend>
[[nodiscard]] MemoryEstimate estimate_memory(const RunPlan& plan,
                                             const lpfx::device::DeviceProperties& properties) {
  return config::dispatch_precision(plan.input.precision, [&]<class Precision> {
    return detail::estimate_precision_memory<Precision, Backend>(plan, properties);
  });
}

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_REPORTING_MEMORY_ESTIMATOR_HPP
