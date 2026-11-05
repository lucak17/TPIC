#include <tpic/config/domain.hpp>
#include <tpic/config/particle_plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <lpfx/data/particle/capacity.hpp>

namespace {

using tpic::config::SimulationParameters;
using tpic::config::prepare_particle_plan;
using tpic::config::prepare_domain;
using tpic::config::ParticleOrder;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function>
void rejects(Function &&function, const char *message) {
  try {
    function();
  } catch (const Exception &error) {
    require(std::string(error.what()).starts_with("TPIC particles:"),
            "missing particle preparation error context");
    return;
  }
  throw std::runtime_error(message);
}

SimulationParameters parameters(int species_count = 1) {
  SimulationParameters parameters;
  parameters.ns = species_count;
  parameters.species.resize(species_count);
  parameters.grid.nxc = parameters.grid.nyc = parameters.grid.nzc = 1;
  for (auto &species : parameters.species)
    species.npcelx = species.npcely = species.npcelz = 1;
  return parameters;
}

void migration_capacity_policy() {
  using tpic::config::initial_particle_migration_capacity;
  constexpr auto quantum = lpfx::particle_capacity_multiple;
  static_assert(quantum != 0);
  for (const std::uint32_t owner : std::array<std::uint32_t, 6>{
           0, 1, 9, 10, 641, static_cast<std::uint32_t>(std::numeric_limits<int>::max())}) {
    const auto seed = initial_particle_migration_capacity(owner);
    const auto minimum = owner / 10 + (owner % 10 != 0);
    require(seed >= minimum, "migration capacity is below ten percent");
    require(seed % quantum == 0,
            "migration capacity is not rounded to the LPFx quantum");
    require(seed == 0 || seed - quantum < minimum,
            "migration capacity was rounded by more than one quantum");
  }
  require(initial_particle_migration_capacity(0) == 0,
          "empty migration capacity did not remain empty");
  require(initial_particle_migration_capacity(1) == quantum,
          "small migration capacity did not round up to one quantum");
}

void nondivisible_counts_and_ids() {
  auto input = parameters(3);
  input.grid.nxc = 11; input.grid.nyc = 7; input.grid.nzc = 5;
  input.topology.xlen = 3; input.topology.ylen = input.topology.zlen = 2;
  input.species[1].npcelx = 2; input.species[1].npcely = 3;
  input.species[2].npcelx = input.species[2].npcely = input.species[2].npcelz = 3;
  const std::array<std::uint64_t, 3> ppc{1, 6, 27};
  const std::array<std::uint64_t, 3> global{385, 2310, 10395};
  std::array<std::uint32_t, 3> migration{};
  for (std::size_t species = 0; species < migration.size(); ++species) {
    std::uint32_t maximum_owner{};
    require(lpfx::try_prepare_particle_capacity(
                static_cast<std::uint32_t>(48 * ppc[species]), maximum_owner) ==
                lpfx::ErrorCode::success,
            "largest-rank owner capacity preparation failed");
    migration[species] = tpic::config::initial_particle_migration_capacity(maximum_owner);
  }
  std::array<std::uint64_t, 3> next_id{0, 385, 2695};
  std::uint64_t total_local = 0;
  for (int rank = 0; rank < 12; ++rank) {
    const auto domain = prepare_domain(input.grid, input.topology, rank, 12);
    const auto plan = prepare_particle_plan(input, domain);
    // A bounded load keeps each rank's IDs inside its dense interval.
    const std::array<std::uint64_t, 3> bounded_local{1, 2, 3}, bounded_global{12, 24, 36};
    const auto bounded = prepare_particle_plan(input, domain, tpic::config::ParticleOrder::full,
                                               bounded_local, bounded_global);
    require(plan.global_count == 13090, "wrong total global count");
    require(plan.local_count == 34 * static_cast<std::uint64_t>(domain.interior_cell_count),
            "local particle count assumed divisible partition");
    total_local += plan.local_count;
    for (std::size_t species = 0; species < 3; ++species) {
      const auto &actual = plan.species[species];
      require(actual.particles_per_cell == ppc[species] &&
              actual.global_count == global[species], "wrong species PPC/global count");
      require(actual.first_id == next_id[species], "rank ID intervals have gap or overlap");
      require(actual.past_last_id - actual.first_id == actual.local_count,
              "ID interval does not match initialized population");
      require(actual.initial_migration_capacity == migration[species],
              "reciprocal migration capacity differs by rank or largest owner");
      require(bounded.species[species].first_id == actual.first_id &&
                  bounded.species[species].past_last_id == actual.first_id + bounded_local[species],
              "bounded ID interval left the rank's dense interval");
      next_id[species] = actual.past_last_id;
    }
  }
  require(total_local == 13090 && next_id == std::array<std::uint64_t, 3>{385, 2695, 13090},
          "global population or species ID intervals mismatch");
}

// The byte report of these plans is tested in precision_tests.cpp.
void capacity_policy() {
  auto input = parameters();
  input.case_name = "TwoStream";
  input.species[0].npcelx = 3;
  // Count planning does not impose quiet-start pairing or remove zero-weight
  // records: loading uses one particle per requested subcell.
  input.species[0].rho_init = 0;
  const auto domain = prepare_domain(input.grid, input.topology, 0, 1);
  std::uint32_t expected_capacity{};
  require(lpfx::try_prepare_particle_capacity(std::uint32_t{3}, expected_capacity) ==
              lpfx::ErrorCode::success, "LPFx capacity preparation failed");
  const auto full = prepare_particle_plan(input, domain, ParticleOrder::full);
  const auto &species = full.species[0];
  require(full.local_count == 3 && full.global_count == 3,
          "odd PPC or zero charge changed record count");
  require(full.fixed_local_population && full.bypass_local_migration &&
          species.initial_capacity == 3 && species.classification_capacity == 0 &&
          species.initial_migration_capacity == 0,
          "fixed local full ordering retained spare owner/classification slots");
  const auto partial = prepare_particle_plan(input, domain, ParticleOrder::partial);
  const auto unsorted = prepare_particle_plan(input, domain, ParticleOrder::unsorted);
  require(partial.species[0].initial_migration_capacity == 0 &&
              unsorted.species[0].initial_migration_capacity == 0,
          "one-rank plan reserved remote migration records");
  require(partial.fixed_local_population && !partial.bypass_local_migration &&
          partial.species[0].initial_capacity == 3 &&
          partial.species[0].classification_capacity == expected_capacity,
          "local partial ordering lost the independently rounded LPFx slots");
  require(unsorted.bypass_local_migration && unsorted.species[0].initial_capacity == 3,
          "unsorted mode changed the fixed local owner");
}

void local_population_policy() {
  for (unsigned periodic_axes = 0; periodic_axes < 8; ++periodic_axes) {
    auto input = parameters();
    input.topology.periodic_x_particles = periodic_axes & 1;
    input.topology.periodic_y_particles = periodic_axes & 2;
    input.topology.periodic_z_particles = periodic_axes & 4;
    // Physical face codes on periodic axes are inactive.
    if (periodic_axes & 1) input.bc.particle_x_right = 4;
    if (periodic_axes & 2) input.bc.particle_y_left = 3;
    if (periodic_axes & 4) input.bc.particle_z_right = 0;
    const auto domain = prepare_domain(input.grid, input.topology, 0, 1);
    for (const auto order : {ParticleOrder::full, ParticleOrder::partial, ParticleOrder::unsorted}) {
      const auto plan = prepare_particle_plan(input, domain, order);
      require(plan.fixed_local_population && plan.species[0].initial_capacity == 1 &&
              plan.bypass_local_migration == (order != ParticleOrder::partial),
              "periodic/mirror local population did not select exact capacity");
    }
  }
  const auto fallback = [](SimulationParameters input, int ranks = 1) {
    const auto domain = prepare_domain(input.grid, input.topology, 0, ranks);
    const auto plan = prepare_particle_plan(input, domain);
    std::uint32_t rounded{};
    require(lpfx::try_prepare_particle_capacity(static_cast<std::uint32_t>(plan.local_count), rounded) ==
            lpfx::ErrorCode::success, "fallback test capacity failed");
    require(!plan.fixed_local_population && !plan.bypass_local_migration &&
            plan.species[0].initial_capacity == rounded &&
            plan.species[0].classification_capacity == rounded,
            "remote or unsupported population lost the compiled LPFx capacity policy");
  };
  for (int face = 0; face < 6; ++face)
    for (const int boundary : {0, 2, 3, 4, 99}) {
      auto input = parameters();
      input.topology.periodic_x_particles = input.topology.periodic_y_particles =
          input.topology.periodic_z_particles = false;
      int* boundaries[] = {&input.bc.particle_x_right, &input.bc.particle_x_left,
          &input.bc.particle_y_right, &input.bc.particle_y_left,
          &input.bc.particle_z_right, &input.bc.particle_z_left};
      *boundaries[face] = boundary;
      fallback(input);
    }
  auto input = parameters();
  input.ns_test_particles = 1;
  fallback(input);
  input = parameters();
  input.test_species.resize(1);
  fallback(input);
  input = parameters();
  input.grid.nxc = 4;
  input.topology.xlen = 2;
  fallback(input, 2);

  // The reported DoubleGEM case must fit by changing spare storage only; no
  // particle, attribute, ID or precision may disappear from this count plan.
  input = parameters(2);
  input.grid.nxc = 256; input.grid.nyc = 128; input.grid.nzc = 16;
  for (auto& species : input.species) {
    species.npcelx = species.npcely = 8; species.npcelz = 2;
  }
  const auto domain = prepare_domain(input.grid, input.topology, 0, 1);
  const auto plan = prepare_particle_plan(input, domain);
  require(plan.local_count == 134217728 && plan.global_count == plan.local_count,
          "DoubleGEM capacity policy changed the requested particle count");
  for (const auto& species : plan.species)
    require(species.initial_capacity == 67108864 && species.classification_capacity == 0 &&
            species.past_last_id - species.first_id == species.local_count,
            "DoubleGEM retained spare capacity or lost particle IDs");
}

void invalid_inputs() {
  const auto valid = parameters();
  const auto domain = prepare_domain(valid.grid, valid.topology, 0, 1);
  for (int value : {0, -1, std::numeric_limits<int>::min()}) {
    for (int axis = 0; axis < 3; ++axis) {
      auto input = valid;
      int *factors[] = {&input.species[0].npcelx, &input.species[0].npcely,
                       &input.species[0].npcelz};
      *factors[axis] = value;
      rejects<std::invalid_argument>([&] { (void)prepare_particle_plan(input, domain); },
                                    "nonpositive PPC factor accepted");
    }
  }
  rejects<std::invalid_argument>([&] { (void)prepare_particle_plan(valid, {}); },
                                "unprepared domain accepted");
  auto corrupt = domain;
  corrupt.particle_topology = {};
  rejects<std::invalid_argument>([&] { (void)prepare_particle_plan(valid, corrupt); },
                                "unprepared particle topology accepted for local policy");
}

void overflow_boundaries() {
  auto input = parameters();
  auto domain = prepare_domain(input.grid, input.topology, 0, 1);
  input.species[0].npcelx = input.species[0].npcely = 50000;
  rejects<std::overflow_error>([&] { (void)prepare_particle_plan(input, domain); },
                              "PPC signed multiplication overflow accepted");
  input.species[0].npcelx = input.species[0].npcely = input.species[0].npcelz =
      std::numeric_limits<int>::max();
  rejects<std::overflow_error>([&] { (void)prepare_particle_plan(input, domain); },
                              "PPC uint64 multiplication overflow accepted");
  input = parameters();
  input.grid.nxc = 8;
  input.species[0].npcelx = 500000000;
  domain = prepare_domain(input.grid, input.topology, 0, 1);
  rejects<std::overflow_error>([&] { (void)prepare_particle_plan(input, domain); },
                              "local launch count overflow accepted");

  // Find the capacity-policy boundary without hard-coding configured headroom.
  // The first rejected rounded capacity can exceed int while the count fits it.
  std::uint32_t low = 0, high = std::numeric_limits<int>::max();
  while (low < high) {
    const auto midpoint = low + (high - low) / 2 + 1;
    std::uint32_t capacity{};
    const bool admitted = lpfx::try_prepare_particle_capacity(midpoint, capacity) ==
        lpfx::ErrorCode::success && capacity <= std::numeric_limits<int>::max();
    if (admitted) low = midpoint;
    else high = midpoint - 1;
  }
  require(low > 0 && low < std::numeric_limits<int>::max(),
          "test requires a capacity policy with a finite positive signed range");
  input = parameters();
  input.ns_test_particles = 1; // Unsupported sources keep the rounded policy.
  input.species[0].npcelx = static_cast<int>(low);
  domain = prepare_domain(input.grid, input.topology, 0, 1);
  require(prepare_particle_plan(input, domain).local_count == low,
          "largest admitted rounded capacity rejected");
  input.species[0].npcelx += 1;
  rejects<std::overflow_error>([&] { (void)prepare_particle_plan(input, domain); },
                              "rounded capacity above signed int accepted");

  input.ns_test_particles = 0;
  input.species[0].npcelx = std::numeric_limits<int>::max();
  for (const auto order : {ParticleOrder::full, ParticleOrder::unsorted})
    require(prepare_particle_plan(input, domain, order).species[0].initial_capacity ==
                std::numeric_limits<int>::max(),
            "exact local capacity applied rounded spare-slot overflow limits");
  rejects<std::overflow_error>([&] {
    (void)prepare_particle_plan(input, domain, ParticleOrder::partial);
  }, "partial classification slots ignored the rounded signed limit");

  // Two billion tiny ranks still require only O(ns) planning storage. Each
  // species count fits uint64, while their combined global ID space overflows.
  input = parameters(10);
  input.grid.nxc = 100000; input.grid.nyc = 100000;
  input.topology.xlen = 50000; input.topology.ylen = 40000;
  for (auto &species : input.species) species.npcelx = 250000000;
  domain = prepare_domain(input.grid, input.topology, 0, 2000000000);
  rejects<std::overflow_error>([&] { (void)prepare_particle_plan(input, domain); },
                              "combined species ID space overflow accepted");
}

} // namespace

int main() {
  try {
    migration_capacity_policy();
    nondivisible_counts_and_ids();
    capacity_policy();
    local_population_policy();
    invalid_inputs();
    overflow_boundaries();
    std::cout << "TPIC particle planning checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
