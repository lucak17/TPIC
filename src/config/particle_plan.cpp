#include <tpic/config/particle_plan.hpp>

#include <tpic/config/boundary_conditions.hpp>
#include <tpic/config/detail/errors.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace tpic::config {
namespace {

constexpr detail::Errors errors{"particles"};

template <class Count>
Count multiply(Count left, Count right, const char *description) {
  if (right != 0 && left > std::numeric_limits<Count>::max() / right)
    errors.overflow(description);
  return left * right;
}

template <class Count>
Count add(Count left, Count right, const char *description) {
  if (left > std::numeric_limits<Count>::max() - right) errors.overflow(description);
  return left + right;
}

std::uint64_t product(const std::array<int, 3> &values,
                      const char *description) {
  std::uint64_t result = 1;
  for (const int value : values) {
    if (value <= 0) errors.invalid(description);
    result = multiply(result, static_cast<std::uint64_t>(value), description);
  }
  return result;
}

std::uint64_t maximum_rank_cells(const config::PreparedDomain &domain) {
  const auto shape = domain.field_neighborhood.topology().input().process_shape();
  const std::array<std::uint64_t, 3> global{
      static_cast<std::uint64_t>(domain.global_cells[0]),
      static_cast<std::uint64_t>(domain.global_cells[1]),
      static_cast<std::uint64_t>(domain.global_cells[2])};
  const std::array<std::uint64_t, 3> ranks{shape.x(), shape.y(), shape.z()};
  std::uint64_t result = 1;
  for (std::size_t axis = 0; axis < global.size(); ++axis) {
    const auto extent = global[axis] / ranks[axis] +
                        (global[axis] % ranks[axis] != 0);
    result = multiply(result, extent,
                      "largest-rank cell count exceeds uint64");
  }
  return result;
}

void require_signed_count(std::uint64_t count, const char *description) {
  if (count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    errors.overflow(description);
}

std::uint32_t capacity_for(std::uint64_t count) {
  if (count > std::numeric_limits<std::uint32_t>::max())
    errors.overflow("required particle count exceeds LPFx uint32 slots");
  std::uint32_t capacity{};
  if (lpfx::try_prepare_particle_capacity(static_cast<std::uint32_t>(count),
                                        capacity) != lpfx::ErrorCode::success)
    errors.overflow("required particle count exceeds LPFx rounded uint32 capacity");
  require_signed_count(capacity, "rounded particle capacity exceeds signed int");
  return capacity;
}

bool fixed_local_population(const SimulationParameters &parameters,
                            const config::PreparedDomain &domain) {
  if (domain.particle_topology.input().communicator_size() != 1 ||
      parameters.ns_test_particles != 0 || !parameters.test_species.empty())
    return false;
  const std::array<bool, 3> periodic{parameters.topology.periodic_x_particles,
      parameters.topology.periodic_y_particles, parameters.topology.periodic_z_particles};
  const std::array<int, 6> boundaries{parameters.bc.particle_x_right,
      parameters.bc.particle_x_left, parameters.bc.particle_y_right,
      parameters.bc.particle_y_left, parameters.bc.particle_z_right,
      parameters.bc.particle_z_left};
  for (std::size_t face = 0; face < boundaries.size(); ++face)
    if (!periodic[face / 2] && boundaries[face] != particle_mirror_code) return false;
  // No source can add particles on effective periodic/mirror faces. Unsupported
  // physical faces and test-particle requests keep the conservative capacity plan.
  return true;
}

// Prefix sum of cell counts over earlier x-fastest ranks without iterating
// over the communicator. Entire z slabs precede this slab, then y strips,
// then x blocks; offsets already include the nondivisible remainder cells.
std::uint64_t preceding_cells(const config::PreparedDomain &domain) {
  constexpr const char *error = "rank cell prefix exceeds uint64";
  const auto gx = static_cast<std::uint64_t>(domain.global_cells[0]);
  const auto gy = static_cast<std::uint64_t>(domain.global_cells[1]);
  const auto ly = static_cast<std::uint64_t>(domain.local_cells[1]);
  const auto lz = static_cast<std::uint64_t>(domain.local_cells[2]);
  const auto ox = static_cast<std::uint64_t>(domain.global_offset[0]);
  const auto oy = static_cast<std::uint64_t>(domain.global_offset[1]);
  const auto oz = static_cast<std::uint64_t>(domain.global_offset[2]);
  const auto slabs = multiply(multiply(gx, gy, error), oz, error);
  const auto strips = multiply(multiply(gx, oy, error), lz, error);
  const auto blocks = multiply(multiply(ox, ly, error), lz, error);
  return add(add(slabs, strips, error), blocks, error);
}

} // namespace

ParticlePlan prepare_particle_plan(const SimulationParameters &parameters,
                                   const config::PreparedDomain &domain,
                                   ParticleOrder order, std::span<const std::uint64_t> local_counts,
                                   std::span<const std::uint64_t> global_counts) {
  // The domain is trusted as prepared by prepare_domain from these parameters;
  // only an unprepared or inconsistent one is rejected.
  if (!domain.particle_topology.is_valid()) errors.invalid("domain must be prepared");
  config::require_canonical_partition(domain, "TPIC particles");
  const auto global_cells = product(domain.global_cells, "global cell count exceeds uint64");
  const auto local_cells = static_cast<std::uint64_t>(domain.interior_cell_count);
  const auto cell_prefix = preceding_cells(domain);
  const auto maximum_cells = maximum_rank_cells(domain);

  ParticlePlan plan;
  plan.order = order;
  plan.fixed_local_population = fixed_local_population(parameters, domain);
  plan.bypass_local_migration = plan.fixed_local_population && order != ParticleOrder::partial;
  plan.species.reserve(parameters.species.size());
  std::uint64_t species_ids = 0;
  for (std::size_t s = 0; s < parameters.species.size(); ++s) {
    const auto &input = parameters.species[s];
    SpeciesParticlePlan species;
    species.particles_per_cell = product({input.npcelx, input.npcely, input.npcelz},
                                         "PPC factors must be positive and product representable");
    require_signed_count(species.particles_per_cell, "PPC product exceeds signed int");
    species.local_count = local_counts.empty() ? multiply(local_cells, species.particles_per_cell,
                                                          "local particle count exceeds uint64")
                                               : local_counts[s];
    require_signed_count(species.local_count, "local particle count exceeds signed int");
    const auto subcells = multiply(global_cells, species.particles_per_cell, "global particle count exceeds uint64");
    species.global_count = global_counts.empty() ? subcells : global_counts[s];
    species.first_id = add(species_ids,
        multiply(cell_prefix, species.particles_per_cell, "particle ID prefix exceeds uint64"),
        "particle ID prefix exceeds uint64");
    species_ids = add(species_ids, subcells, "particle ID prefix exceeds uint64");
    species.past_last_id = add(species.first_id, species.local_count,
                               "particle ID range exceeds uint64");
    // A restarted rank may hold no record of a species.
    const auto owned = std::max<std::uint64_t>(species.local_count, 1);
    species.initial_capacity = plan.fixed_local_population
        ? static_cast<std::uint32_t>(species.local_count) : capacity_for(owned);
    species.classification_capacity = plan.bypass_local_migration ? 0 : capacity_for(owned);
    if (domain.particle_topology.input().communicator_size() != 1) {
      const auto mean_per_cell = species.global_count / global_cells + (species.global_count % global_cells != 0);
      const auto maximum_count = multiply(maximum_cells, mean_per_cell,
          "largest-rank particle count exceeds uint64");
      require_signed_count(maximum_count,
                           "largest-rank particle count exceeds signed int");
      species.initial_migration_capacity = initial_particle_migration_capacity(
          capacity_for(maximum_count));
    }
    plan.local_count = add(plan.local_count, species.local_count,
                           "summed local particle count exceeds uint64");
    plan.global_count = add(plan.global_count, species.global_count,
                            "summed global particle count exceeds uint64");
    plan.species.push_back(species);
  }
  return plan;
}

} // namespace tpic::config
