#ifndef TPIC_CONFIG_PARTICLE_PLAN_HPP
#define TPIC_CONFIG_PARTICLE_PLAN_HPP

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include <lpfx/data/particle/capacity.hpp>

#include <tpic/config/simulation_parameters.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/config/numerics_options.hpp>

namespace tpic::config {

// Ten percent of the largest rank's initial owner, rounded up first to a
// record and then to LPFx's configured particle-capacity quantum.
[[nodiscard]] constexpr std::uint32_t initial_particle_migration_capacity(
    std::uint32_t maximum_owner_capacity) noexcept {
  constexpr std::uint32_t initial_percent = 10;
  constexpr std::uint32_t multiple = lpfx::particle_capacity_multiple;
  static_assert(multiple != 0);
  const auto scaled = static_cast<std::uint64_t>(maximum_owner_capacity) *
                      initial_percent;
  const auto requested = scaled / 100 + (scaled % 100 != 0);
  const auto quanta = requested / multiple + (requested % multiple != 0);
  return static_cast<std::uint32_t>(quanta * multiple);
}

struct SpeciesParticlePlan {
  std::uint64_t particles_per_cell{};
  std::uint64_t local_count{};
  std::uint64_t global_count{};
  // Initial IDs are species-major, then x-fastest rank-major, then the
  // initializer's local record index; a species and a rank are spaced by
  // their subcells, so a bounded load leaves unused IDs. Ranges are half-open
  // and disjoint. This gives reproducibility for a fixed decomposition, not
  // across layouts.
  std::uint64_t first_id{};
  std::uint64_t past_last_id{};
  std::uint32_t initial_capacity{};
  std::uint32_t classification_capacity{};
  // Reciprocal per-peer and aggregate migration capacity, the same on every
  // rank: initial_particle_migration_capacity.
  std::uint32_t initial_migration_capacity{};

  bool operator==(const SpeciesParticlePlan&) const = default;
};

struct ParticlePlan {
  ParticleOrder order = ParticleOrder::full;
  std::vector<SpeciesParticlePlan> species;
  std::uint64_t local_count{};
  std::uint64_t global_count{};
  // One rank, no test-particle sources, and only effective periodic/mirror
  // particle faces: no other owner or boundary can increase this population.
  bool fixed_local_population{};
  // Full/unsorted local ownership uses a checked boundary-only startup pass
  // and local movers. Partial ordering still needs LPFx's migration receipt.
  bool bypass_local_migration{};

  // Largest initial local population of one species.
  [[nodiscard]] std::uint64_t max_species_local_count() const noexcept {
    std::uint64_t maximum = 0;
    for (const auto& planned : species) maximum = std::max(maximum, planned.local_count);
    return maximum;
  }

  bool operator==(const ParticlePlan&) const = default;
};

// Host preparation. Requires a domain prepared from these parameters; PPC
// factors must be positive. PPC, local initialized counts and rounded
// capacities must fit a signed int, besides LPFx's uint32 slot limit.
// Proven fixed local populations use exact initial owner counts. Other owners
// and all required classification slots use LPFx's compiled capacity policy;
// LPFx's later automatic growth is separate. A restart passes the populations
// of its checkpoint, local_counts[s] on this rank and global_counts[s] in all,
// and then leaves the ID ranges unused; a bounded load passes the points of
// its supports. The plan does not depend on the precision profile. Invalid
// inputs throw invalid_argument; checked count, capacity or ID-range
// arithmetic throws overflow_error.
[[nodiscard]] ParticlePlan prepare_particle_plan(
    const SimulationParameters &parameters, const config::PreparedDomain &domain,
    ParticleOrder order = ParticleOrder::full, std::span<const std::uint64_t> local_counts = {},
    std::span<const std::uint64_t> global_counts = {});

} // namespace tpic::config

#endif // TPIC_CONFIG_PARTICLE_PLAN_HPP
