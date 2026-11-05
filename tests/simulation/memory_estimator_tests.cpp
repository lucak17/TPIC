#include <tpic/simulation/planning/run_plan.hpp>
#include <device/mock_backend.hpp>
#include <tpic/simulation/reporting/memory_estimator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

using Backend = lpfx::device::MockBackend;
using Mock = lpfx::device::BackendTraits<Backend>;
using Order = tpic::config::ParticleOrder;
using tpic::simulation::RunMode;
using tpic::simulation::RunPlan;
// CUDA-compatible warp width and launch limits, without a device query.
constexpr lpfx::device::DeviceProperties properties{0, 32, 1024, 2147483647, 40};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

tpic::config::SimulationParameters parameters(int ranks = 1, unsigned periodic_axes = 7) {
  tpic::config::SimulationParameters p;
  p.case_name = "Uniform";
  p.dt = .1;
  p.ncycles = 1;
  p.grid.nxc = 4 * ranks;
  p.grid.nyc = 3;
  p.grid.nzc = 2;
  p.topology.xlen = ranks;
  // Keep the field topology periodic to exercise distinct particle topology.
  p.topology.periodic_x_particles = periodic_axes & 1;
  p.topology.periodic_y_particles = periodic_axes & 2;
  p.topology.periodic_z_particles = periodic_axes & 4;
  p.output.field_cycle = 0;
  p.output.restart_cycle = 0;
  p.ns = 2;
  p.species.resize(2);
  for (int index = 0; index < p.ns; ++index) {
    auto& s = p.species[index];
    s.q_over_m = index == 0 ? -25 : 1;
    s.rho_init = 1;
    s.npcelx = s.npcely = s.npcelz = 4;
  }
  return p;
}

// Every estimate is of a prepared run plan, as at startup.
template <class Precision>
RunPlan prepared(tpic::config::SimulationParameters p, tpic::simulation::InputOptions options = {},
                 RunMode mode = RunMode::advance,
                 tpic::simulation::RuntimeDebugOptions debug = {}) {
  options.precision = {Precision::Storage::id, Precision::Compute::id};
  return RunPlan::prepare(std::move(p), options, mode, debug);
}

auto estimate(const RunPlan& plan) {
  return tpic::simulation::estimate_memory<Backend>(plan, properties);
}

// Field output with the given field and moment tags (empty tags select the
// default E, B, totals and species density and current).
tpic::config::SimulationParameters field_output(tpic::config::SimulationParameters p, int cycle,
                                                std::string field_tag, std::string moments_tag) {
  p.output.field_cycle = cycle;
  p.output.field_tag = std::move(field_tag);
  p.output.moments_tag = std::move(moments_tag);
  return p;
}

std::size_t output_host_bytes(const tpic::simulation::MemoryEstimate& memory) {
  return memory.output_pinned_host_bytes + memory.output_owned_host_bytes;
}

void accounting(const tpic::simulation::MemoryEstimate& memory) {
  const auto owners = memory.particle_device_bytes + memory.field_device_bytes +
      memory.mass_device_bytes + memory.solver_device_bytes + memory.centering_device_bytes +
      memory.sort_device_bytes + memory.diagnostics_device_bytes + memory.output_device_bytes +
      memory.validation_device_bytes;
  require(memory.persistent_device_bytes == owners, "persistent owner accounting differs");
  require(memory.estimated_device_bytes == owners + memory.transport_device_bytes +
      memory.runtime_reserve_bytes,
      "startup estimate includes hypothetical growth or omits initial storage");
}

template <class Precision>
void boundary_records_remain_local(const tpic::simulation::PreparedInput& input) {
  const auto geometry = tpic::communication::prepare_particle_migration_geometry(
      input.domain, input.physics);
  require(tpic::communication::is_local_particle_geometry(geometry),
      "closed one-rank geometry did not qualify for local motion");
  for (int axis = 0; axis < 3; ++axis) {
    for (bool upper : {false, true}) {
      tpic::communication::ParticleMigrationPoint<Precision> point;
      for (int d = 0; d < 3; ++d) {
        point.cell[d] = d == axis ? (upper ? input.domain.local_cells[d] : -1) : 0;
        point.fraction[d] = .25;
        point.velocity[d] = .125;
      }
      const auto flag = tpic::communication::classify_particle_point(
          geometry, point);
      require(flag == 0, "closed one-rank boundary record is not resident");
      for (int d = 0; d < 3; ++d)
        require(point.cell[d] >= 0 && point.cell[d] < input.domain.local_cells[d],
                "closed boundary did not retain a canonical local record");
    }
    for (bool upper : {false, true}) {
      tpic::communication::ParticleMigrationPoint<Precision> point;
      for (int d = 0; d < 3; ++d) {
        point.cell[d] = d == axis
            ? (upper ? 2 * input.domain.local_cells[d]
                     : -2 * input.domain.local_cells[d] - 1)
            : 0;
        point.fraction[d] = .25;
        point.velocity[d] = .125;
      }
      const auto flag = tpic::communication::classify_particle_point(
          geometry, point);
      require(flag == 0, "closed one-rank long flight is not resident");
      for (int d = 0; d < 3; ++d)
        require(point.cell[d] >= 0 && point.cell[d] < input.domain.local_cells[d],
                "closed long flight did not retain a canonical local record");
    }
  }
}

template <class Precision>
void topology_and_orders() {
  Mock::reset();
  for (const auto order : {Order::full, Order::partial, Order::unsorted}) {
    for (unsigned periodic_axes = 0; periodic_axes < 8; ++periodic_axes) {
      const auto local = prepared<Precision>(parameters(1, periodic_axes), {.order = order});
      boundary_records_remain_local<Precision>(local.input);
      const auto one = estimate(local);
      accounting(one);
      require(one.growth_reserve_bytes == 0, "closed one-rank population reserves impossible growth");
      require((one.sort_device_bytes == 0) == (order == Order::unsorted),
              "initial sort workspace was removed with growth reserve");
      // Equal local populations have different capacity policies: local
      // owners are exact, and full/unsorted local startup needs no slots.
      const auto remote = prepared<Precision>(parameters(2, periodic_axes), {.order = order});
      require(!tpic::communication::is_local_particle_geometry(
                  tpic::communication::prepare_particle_migration_geometry(
                      remote.input.domain, remote.input.physics)),
              "remote owner qualified for local-only motion");
      const auto two = estimate(remote);
      accounting(two);
      require(two.particle_device_bytes > one.particle_device_bytes &&
                  two.persistent_device_bytes - two.particle_device_bytes - two.sort_device_bytes ==
                  one.persistent_device_bytes - one.particle_device_bytes - one.sort_device_bytes,
              "local capacity savings changed unrelated persistent storage");
      std::size_t local_owners = 0;
      for (const auto& species : local.input.particles.species)
        local_owners += tpic::simulation::particle_storage_bytes<Precision, Backend>(
            species.initial_capacity);
      require((one.particle_device_bytes == local_owners) == (order != Order::partial),
              "local startup classification accounting disagrees with ordering");
      require(two.growth_reserve_bytes > two.particle_device_bytes + two.sort_device_bytes,
              "remote migration lost its replacement/auxiliary reserve");
      using Wire = typename tpic::communication::ParticleMigrationWire<Precision>::Contract;
      // Fields stay periodic independently of the varied particle boundaries.
      // Equal local geometry keeps outgoing/self snapshots and direction
      // records equal, so only the compact remote receive image is added.
      const auto mass_plan =
          tpic::communication::MassHaloPlan<Precision>::prepare(remote.input.domain);
      const auto mass_receive = mass_plan.incoming_bytes();
      std::size_t particle_wire = 0, reserved_arrivals = 0;
      for (const auto& species : remote.input.particles.species) {
        require(species.initial_migration_capacity > lpfx::particle_capacity_multiple,
                "remote fixture seed does not exceed one capacity quantum");
        require(species.initial_migration_capacity ==
                    tpic::config::initial_particle_migration_capacity(species.initial_capacity),
                "uniform remote seed was not derived from its owner capacity");
        require(species.initial_migration_capacity < species.initial_capacity,
                "remote fixture did not distinguish bounded arrival storage from the full owner");
        // Both non-empty wire slots carry the contract's aligned-packet
        // slack (zero for the packed layout).
        particle_wire += 2 * static_cast<std::size_t>(species.initial_migration_capacity) *
                             Wire::record_wire_width +
                         2 * static_cast<std::size_t>(Wire::wire_extent_slack);
        reserved_arrivals += tpic::simulation::arrival_chunk_bytes<Precision, Backend>(
            species.initial_migration_capacity) +
            tpic::simulation::classification_bytes(species.initial_migration_capacity);
      }
      require(two.migration_reserved_device_bytes == reserved_arrivals &&
                  one.migration_reserved_device_bytes == 0,
              "cold collective reservation omitted arrivals or invented one-rank arrivals");
      require(two.transport_device_bytes - one.transport_device_bytes ==
                  mass_receive + particle_wire + reserved_arrivals,
              "initial remote transport lost seeded storage or counts lazy compaction lists");
    }
  }
  require(Mock::device_allocation_attempts == 0 && Mock::pinned_allocations == 0 &&
              Mock::event_creations == 0 && Mock::copy_calls == 0,
          "cold memory estimate performed backend work");
}

template <class Precision>
void compact_mass_storage_accounting() {
  auto physical_parameters = parameters();
  physical_parameters.topology.periodic_x = false;
  physical_parameters.topology.periodic_y = false;
  physical_parameters.topology.periodic_z = false;
  const auto physical = prepared<Precision>(std::move(physical_parameters));
  const auto periodic = prepared<Precision>(parameters());
  using MassPlan = tpic::communication::MassHaloPlan<Precision>;
  const auto physical_plan = MassPlan::prepare(physical.input.domain);
  const auto periodic_plan = MassPlan::prepare(periodic.input.domain);
  require(physical_plan.device_bytes() == 0 &&
              periodic_plan.template phase<true>().directions().size() == 26,
          "mass accounting fixture has the wrong field topology");
  const auto physical_memory = estimate(physical);
  const auto periodic_memory = estimate(periodic);
  require(periodic_memory.transport_device_bytes >=
                  physical_memory.transport_device_bytes &&
              periodic_memory.transport_device_bytes -
                      physical_memory.transport_device_bytes ==
                  periodic_plan.device_bytes(),
          "mass transport accounting does not match compact plan storage");
}

template <class Precision>
void nonuniform_arrival_reservation() {
  auto p = parameters(2);
  p.grid.nxc = p.grid.nyc = p.grid.nzc = 5;
  p.topology.xlen = p.topology.ylen = p.topology.zlen = 2;
  std::uint32_t common_seed{};
  std::size_t largest_owner{}, smallest_owner = std::numeric_limits<std::size_t>::max();
  for (int rank = 0; rank < 8; ++rank) {
    const auto plan = prepared<Precision>(p, {.rank = rank});
    const auto memory = estimate(plan);
    std::size_t expected{};
    for (const auto& species : plan.input.particles.species) {
      if (common_seed == 0) common_seed = species.initial_migration_capacity;
      require(species.initial_migration_capacity == common_seed,
              "nonuniform ranks lost the reciprocal wire seed");
      const auto arrivals = std::min(species.initial_migration_capacity,
                                     species.initial_capacity);
      expected += tpic::simulation::arrival_chunk_bytes<Precision, Backend>(arrivals) +
          tpic::simulation::classification_bytes(species.initial_migration_capacity);
      largest_owner = std::max(largest_owner, static_cast<std::size_t>(species.initial_capacity));
      smallest_owner = std::min(smallest_owner, static_cast<std::size_t>(species.initial_capacity));
    }
    require(memory.migration_reserved_device_bytes == expected,
            "nonuniform arrival reservation exceeded the seed or local final bound");
  }
  require(largest_owner > smallest_owner,
          "nonuniform fixture did not distinguish local owner capacities");
}

template <class Precision>
void concurrent_storage() {
  // E, B, both totals, every pressure total and every species array.
  const std::string all_moments =
      "rho_tot+J_tot+rho+J+PXX_tot+PXY_tot+PXZ_tot+PYY_tot+PYZ_tot+PZZ_tot+PXX+PXY+PXZ+PYY+PYZ+PZZ";
  const auto plan = prepared<Precision>(field_output(parameters(), 1, "B+E", all_moments));
  const auto& input = plan.input;
  const auto memory = estimate(plan);
  accounting(memory);
  // Selected pinned and owned images: E/B + totals (16), species (10 each).
  const auto scalar = input.domain.interior_node_count * sizeof(double);
  require(output_host_bytes(memory) == 2 * (16 + 10 * input.particles.species.size()) * scalar,
          "owned output snapshot omitted or counted twice");
  require(memory.validation_device_bytes == (6 + input.particles.species.size()) * sizeof(unsigned) &&
              memory.validation_host_bytes == memory.validation_device_bytes,
          "independent validation slots missing from the estimate");
  const auto selected = 16 + 10 * input.particles.species.size();
  const auto scalars = tpic::data::field_scalar_count +
      tpic::data::species_scalar_count * input.particles.species.size();
  require(memory.diagnostics_device_bytes == (scalars + 2 * tpic::data::species_scalar_count *
      tpic::data::contribution_count) * sizeof(typename Precision::Compute::Reduction),
      "parallel diagnostics scratch omitted from the estimate");
  // FieldOutputTag Gauss: one cell array staged, the diagnostic's values and cells.
  const auto gauss = estimate(prepared<Precision>(field_output(parameters(), 1, "B+E+Gauss", all_moments)));
  using Reduction = typename Precision::Compute::Reduction;
  require(gauss.diagnostics_device_bytes - memory.diagnostics_device_bytes ==
                  tpic::data::gauss_scalar_count * (1 + tpic::data::contribution_count) * sizeof(Reduction) +
                      3 * tpic::simulation::field_storage_bytes<
                              typename tpic::data::GaussFieldSchema<Precision>::Cell, Backend>(input.domain) &&
              gauss.diagnostics_pinned_host_bytes - memory.diagnostics_pinned_host_bytes ==
                  tpic::data::gauss_scalar_count * sizeof(Reduction) &&
              gauss.field_device_bytes == memory.field_device_bytes &&
              gauss.output_device_bytes - memory.output_device_bytes ==
                  input.domain.interior_cell_count * sizeof(double),
          "Gauss values, cells or staged array missing from the estimate");
  const auto disabled = estimate(
      prepared<Precision>(field_output(parameters(), 0, "B+E", all_moments)));
  require(memory.output_device_bytes - disabled.output_device_bytes == selected * scalar,
          "fully selected species arrays still reserve unused packing scratch");
  require(output_host_bytes(disabled) == 0,
          "disabled field output reserves a host snapshot");
  const auto density = estimate(
      prepared<Precision>(field_output(parameters(), 1, "", "rho_tot")));
  // Totals without a selected species array need scratch only on the two-launch path.
  const auto scratch = [](std::size_t components) {
    return tpic::data::field_snapshot_scratch_components<Backend>(components);
  };
  require(density.output_device_bytes - disabled.output_device_bytes == (1 + scratch(1)) * scalar &&
              output_host_bytes(density) == 2 * scalar,
          "density-only totals do not use one scalar image and one scalar temporary");
  const auto current = estimate(
      prepared<Precision>(field_output(parameters(), 1, "", "rho_tot+J_tot")));
  require(current.output_device_bytes - disabled.output_device_bytes == (4 + scratch(3)) * scalar &&
              output_host_bytes(current) == 8 * scalar,
          "combined scalar/vector totals do not share one vector temporary");
}

template <class Precision>
void particle_snapshot_storage() {
  auto p = parameters();
  const auto disabled = estimate(prepared<Precision>(p));
  p.output.particles_cycle = 2;
  const auto plan = prepared<Precision>(p);
  const auto particles = estimate(plan);
  accounting(particles);
  std::size_t bytes{};
  for (const auto& species : plan.input.particles.species)
    bytes += species.local_count * tpic::data::ParticleSchema<Precision>::type::record_width;
  require(particles.output_device_bytes == disabled.output_device_bytes &&
              particles.output_staging_device_bytes == disabled.output_staging_device_bytes &&
              particles.output_pinned_host_bytes == bytes &&
              particles.output_owned_host_bytes == 0 && output_host_bytes(particles) == bytes,
          "particle images are not one pinned copy of the stored records or count owner headroom");
  require(particles.output_initial_pinned_host_bytes == 0 &&
              particles.persistent_pinned_host_bytes == disabled.persistent_pinned_host_bytes &&
              particles.startup_pinned_host_estimate_bytes == disabled.startup_pinned_host_estimate_bytes,
          "lazy particle host images were charged to startup pinned storage");
  require(particles.worker_count == disabled.worker_count + 1 &&
              particles.stream_count == disabled.stream_count + 2 &&
              particles.dependency_event_count == disabled.dependency_event_count + 2,
          "particle-only capture omitted its transfer streams or its readiness/transfer events");
  const auto initial = estimate(prepared<Precision>(p, {}, RunMode::initialize_only));
  require(initial.output_pinned_host_bytes == 0 && output_host_bytes(initial) == 0,
          "initialize-only run allocates particle snapshots");
  p.output.field_cycle = 1;
  const auto combined_plan = prepared<Precision>(p);
  const auto combined = estimate(combined_plan);
  require(combined.worker_count == particles.worker_count &&
              combined.stream_count == particles.stream_count &&
              combined.dependency_event_count == particles.dependency_event_count + 2,
          "combined snapshots duplicated transfer streams or lost independent event pairs");
  // The default tags: E, B, rho_tot, J_tot and rho, J per species.
  require(combined.output_pinned_host_bytes ==
              bytes + (10 + 4 * combined_plan.input.particles.species.size()) *
                          combined_plan.input.domain.interior_node_count * sizeof(double),
          "combined snapshot memory omitted a channel");
}

template <class Precision>
void bounded_particle_snapshot_storage() {
  // A population of several write chunks with migration headroom in owners:
  // the capture adds no device storage, and its host image holds the active
  // records only.
  auto p = parameters(2);
  p.grid.nxc = 64;
  p.grid.nyc = 48;
  p.grid.nzc = 32;
  const auto disabled = estimate(prepared<Precision>(p, {.order = Order::partial}));
  p.output.particles_cycle = 2;
  const auto plan = prepared<Precision>(p, {.order = Order::partial});
  const auto enabled = estimate(plan);
  accounting(enabled);
  std::size_t host_bytes{};
  for (const auto& species : plan.input.particles.species) {
    require(species.initial_capacity > species.local_count,
            "particle capture fixture lacks owner headroom");
    host_bytes += species.local_count * tpic::data::ParticleSchema<Precision>::type::record_width;
  }
  require(enabled.estimated_device_bytes == disabled.estimated_device_bytes,
          "particle capture added device storage");
  require(enabled.output_pinned_host_bytes == host_bytes && enabled.output_owned_host_bytes == 0 &&
              enabled.persistent_pinned_host_bytes == disabled.persistent_pinned_host_bytes,
          "particle host storage uses capacity instead of the lazy active image");
  require(enabled.growth_reserve_bytes == disabled.growth_reserve_bytes,
          "particle capture grows in the hypothetical migration scenario");
}

template <class Precision>
void initialize_only_storage() {
  auto p = parameters(2);
  p.output.field_cycle = 1;
  p.output.particles_cycle = 1;
  const auto run = estimate(prepared<Precision>(p, {.order = Order::partial}, RunMode::advance,
                                                {.profile_stages = true}));
  const auto initialize = estimate(prepared<Precision>(
      p, {.order = Order::partial}, RunMode::initialize_only, {.profile_stages = true}));
  accounting(initialize);
  require(initialize.field_device_bytes == run.field_device_bytes &&
              initialize.solver_device_bytes == run.solver_device_bytes &&
              initialize.particle_device_bytes == run.particle_device_bytes &&
              initialize.centering_device_bytes == run.centering_device_bytes &&
              initialize.validation_device_bytes == run.validation_device_bytes,
          "initialize-only estimate dropped structural owners or validation");
  require(initialize.output_staging_device_bytes == 0 &&
              output_host_bytes(initialize) == 0 && initialize.output_pinned_host_bytes == 0 &&
              initialize.output_device_bytes == run.output_device_bytes - run.output_staging_device_bytes,
          "initialize-only estimate allocates disabled capture or drops the moment owner");
  require(initialize.diagnostics_device_bytes == 0 && initialize.diagnostics_pinned_host_bytes == 0,
          "initialize-only estimate allocates disabled scalar diagnostics");
  require(initialize.worker_count + 1 == run.worker_count &&
              initialize.stream_count + 2 == run.stream_count &&
              initialize.dependency_event_count + 4 == run.dependency_event_count,
          "initialize-only estimate counts writer or capture resources");
  require(initialize.timing_host_bytes == run.timing_host_bytes &&
              initialize.timing_event_count == run.timing_event_count,
          "dedicated output writer was assigned nonexistent StageTiming storage");
}

template <class Precision>
void pipeline_resources() {
  using Timing = tpic::simulation::StageTiming<Backend>;
  for (const auto order : {Order::full, Order::partial, Order::unsorted}) {
    std::size_t previous_host = 0;
    for (const unsigned ns : {1U, 2U, 10U}) {
      auto p = parameters();
      p.ns = ns;
      p.species.resize(ns, p.species.front());
      const auto plan = prepared<Precision>(p, {.order = order});
      const auto disabled = estimate(plan);
      const auto profiled = estimate(
          prepared<Precision>(p, {.order = order}, RunMode::advance, {.profile_stages = true}));
      accounting(disabled);
      require(disabled.diagnostics_pinned_host_bytes == (2 + 5 * ns) * sizeof(double) &&
                  disabled.persistent_pinned_host_bytes == disabled.validation_host_bytes +
                      disabled.diagnostics_pinned_host_bytes + 2 * sizeof(double) &&
                  disabled.startup_pinned_host_estimate_bytes > disabled.persistent_pinned_host_bytes,
              "pinned readback or cold upload staging missing from report");
      require(disabled.worker_count == ns + 6 &&
                  disabled.stream_count == 11 + ns * (order == Order::unsorted ? 1 : 2),
              "pipeline reported obsolete shared streams or omitted a worker role");
      require(disabled.validation_count == ns + 6 &&
                  disabled.dependency_event_count == (order == Order::partial ? 5 : 2) * ns + 10,
              "pipeline validation or reader-generation events were omitted");
      require(disabled.timing_event_count == 0 &&
                  profiled.timing_event_count == 2 * (15 + 11 * ns) * (ns + 7),
              "optional phase-aware timing event count disagrees with prepared role timers");
      require(disabled.timing_host_bytes == (ns + 7) * sizeof(Timing) &&
                  profiled.timing_host_bytes == disabled.timing_host_bytes &&
                  disabled.runtime_host_bytes > 0 &&
                  (previous_host == 0 || disabled.runtime_host_bytes == previous_host),
              "worker control/timing host storage missing, attributed to native allocation or sized by ns");
      require(profiled.estimated_device_bytes == disabled.estimated_device_bytes &&
                  profiled.runtime_host_bytes == disabled.runtime_host_bytes,
              "native event handles were converted into a falsely exact GPU-byte count");
      previous_host = disabled.runtime_host_bytes;
      p.output.field_cycle = 1;
      const auto output_plan = prepared<Precision>(p, {.order = order});
      const auto output = estimate(output_plan);
      const auto initialization = estimate(prepared<Precision>(p, {.order = order},
                                                               RunMode::initialize_only));
      require(output.dependency_event_count == disabled.dependency_event_count + 2 &&
                  output.stream_count == disabled.stream_count + 2 &&
                  initialization.stream_count == disabled.stream_count,
              "optional transfer stream count ignored actual snapshot activation");
    }
  }
}

template <class Precision>
void simultaneous_growth_and_two_field_epochs() {
  using tpic::simulation::CheckedSize;
  using tpic::simulation::classification_bytes;
  using tpic::simulation::field_storage_bytes;
  using tpic::simulation::particle_storage_bytes;
  using Fields = tpic::data::FieldSchema<Precision>;
  using Solver = tpic::ecsim::SolverFieldSchema<Precision>;
  for (const auto order : {Order::full, Order::partial, Order::unsorted}) {
    std::size_t one_growth = 0, one_fields = 0, one_reserved = 0;
    for (const unsigned ns : {1U, 2U, 10U}) {
      auto p = parameters(2);
      p.ns = ns;
      p.species.resize(ns, p.species.front());
      const auto plan = prepared<Precision>(p, {.order = order});
      const auto& input = plan.input;
      const auto memory = estimate(plan);
      accounting(memory);
      if (ns == 1) {
        one_growth = memory.growth_reserve_bytes;
        one_fields = memory.field_device_bytes;
        one_reserved = memory.migration_reserved_device_bytes;
      }
      require(memory.growth_reserve_bytes == ns * one_growth &&
                  memory.migration_reserved_device_bytes == ns * one_reserved,
              "simultaneous species replacements were reduced to a serial high-water maximum");
      require(memory.field_device_bytes == one_fields,
              "species concurrency allocated an extra full field or magnetic epoch");
      const auto& domain = input.domain;
      const auto expected_fields = field_storage_bytes<typename Fields::E, Backend>(domain) +
          field_storage_bytes<typename Fields::Current, Backend>(domain) +
          2 * field_storage_bytes<typename Fields::BCell, Backend>(domain) +
          2 * field_storage_bytes<typename Fields::BNode, Backend>(domain) +
          8 * field_storage_bytes<typename Solver::Node, Backend>(domain) +
          field_storage_bytes<typename Solver::Cell, Backend>(domain) +
          2 * field_storage_bytes<typename tpic::data::GaussFieldSchema<Precision>::Node, Backend>(domain);
      require(memory.field_device_bytes == expected_fields, "field plan does not retain exactly two B epochs");
      std::size_t actual_particles = 0;
      for (const auto& species : input.particles.species) {
        std::uint32_t slot_capacity{};
        lpfx::check_error<std::invalid_argument>(
            lpfx::try_prepare_particle_capacity(species.initial_capacity, slot_capacity),
            "TPIC memory estimate: LPFx storage plan rejected");
        require(slot_capacity >= species.classification_capacity,
                "main-capacity reserve unexpectedly shrank the classification high-water mark");
        actual_particles += particle_storage_bytes<Precision, Backend>(species.initial_capacity) + classification_bytes(slot_capacity);
      }
      require(memory.particle_device_bytes == actual_particles,
              "reserved main classification used initial population instead of main capacity");
    }
  }
  const auto maximum = std::numeric_limits<std::size_t>::max();
  for (bool product : {false, true}) {
    bool overflow = false;
    try {
      static_cast<void>(product ? CheckedSize{maximum} * CheckedSize{2}
                                : CheckedSize{maximum} + CheckedSize{1});
    }
    catch (const std::overflow_error&) { overflow = true; }
    require(overflow, "pipeline memory arithmetic silently wrapped");
  }
  require((CheckedSize{maximum} + CheckedSize{0}).value() == maximum &&
              (CheckedSize{maximum} * CheckedSize{0}).value() == 0,
          "checked arithmetic rejected a representable boundary");
}

// Optional real-input estimate: the mock's 256-byte alignment matches native
// CUDA layout plans, and the planner allocates no particle-sized host image.
template <class Precision>
void report_input(const char* path) {
  const auto plan = prepared<Precision>(tpic::config::load_simulation_parameters(path),
                                        {.order = Order::partial}, RunMode::initialize_only);
  const auto& input = plan.input;
  const auto memory = estimate(plan);
  accounting(memory);
  constexpr double gib = double(std::size_t{1} << 30);
  std::cout << path << " (" << tpic::config::precision_name(input.precision) << "): persistent="
            << memory.persistent_device_bytes / gib << " GiB; transport="
            << memory.transport_device_bytes / gib << " GiB; growth="
            << memory.growth_reserve_bytes / gib << " GiB hypothetical extra; initial estimate="
            << memory.estimated_device_bytes / gib << " GiB; output staging="
            << memory.output_staging_device_bytes / gib << " GiB; initial pinned="
            << memory.persistent_pinned_host_bytes / gib << " GiB; snapshot host="
            << output_host_bytes(memory) / gib << " GiB\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    // The matched pairs: every build compiles them, so the run plan admits them.
    for (unsigned level = 0; level < tpic::config::precision_level_count; ++level) {
      const auto profile = tpic::config::matched_precision(static_cast<tpic::config::ComputePrecision>(level));
      tpic::config::dispatch_precision(profile, [&]<class Precision> {
        topology_and_orders<Precision>();
        compact_mass_storage_accounting<Precision>();
        nonuniform_arrival_reservation<Precision>();
        concurrent_storage<Precision>();
        particle_snapshot_storage<Precision>();
        bounded_particle_snapshot_storage<Precision>();
        initialize_only_storage<Precision>();
        pipeline_resources<Precision>();
        simultaneous_growth_and_two_field_epochs<Precision>();
        if (argc == 2) report_input<Precision>(argv[1]);
      });
    }
    require(Mock::device_allocation_attempts == 0 && Mock::pinned_allocations == 0 &&
                Mock::event_creations == 0 && Mock::copy_calls == 0,
            "pipeline resource accounting performed native allocation or submission");
    std::cout << "TPIC topology-aware memory estimate checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
