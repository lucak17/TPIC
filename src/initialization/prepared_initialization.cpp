#include <tpic/initialization/prepared_initialization.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace tpic::initialization {

config::ParticlePlan prepare_bounded_particle_plan(const config::ParticlePlan& plan,
                                                   const InitialCase& initial_case,
                                                   const config::SimulationParameters& parameters,
                                                   const config::PreparedDomain& domain) {
  std::vector<std::uint64_t> local, global;
  std::visit([&](const auto& selected) {
    for (std::size_t s = 0; s < plan.species.size(); ++s) {
      const auto support = selected.profile(s).support();
      if (!support.bounded()) {
        local.push_back(plan.species[s].local_count);
        global.push_back(plan.species[s].global_count);
        continue;
      }
      const auto& species = parameters.species[s];
      const int ppc[3]{species.npcelx, species.npcely, species.npcelz};
      for (const bool whole_domain : {false, true}) {
        std::uint64_t points = 0;
        visit_support_rows(support, domain, ppc, whole_domain,
                           [&](std::int64_t, std::int64_t count) { points += count; });
        (whole_domain ? global : local).push_back(points);
      }
    }
  }, initial_case);
  return config::prepare_particle_plan(parameters, domain, plan.order, local, global);
}

std::vector<ParticleLoadPlan> prepare_particle_loads(
    const config::SimulationParameters& parameters, const config::PreparedDomain& domain,
    const config::ParticlePlan& particles) {
  constexpr double four_pi = 12.566370614359172953850573533118;
  std::vector<ParticleLoadPlan> loads(particles.species.size());
  for (std::size_t index = 0; index < loads.size(); ++index) {
    const auto& planned = particles.species[index];
    const auto& species = parameters.species[index];
    auto& result = loads[index];
    result.seed = 12345ULL + 100ULL * (static_cast<std::uint64_t>(domain.field_neighborhood.local_partition().rank()) *
        static_cast<std::uint64_t>(config::max_real_species) + index);
    // The volume is (dx*dy)*dz in double, in this order, not domain.cell_volume,
    // which domain admission forms with a wider intermediate.
    const double volume = (domain.spacing[0] * domain.spacing[1]) *
        domain.spacing[2];
    result.q_factor = (species.q_over_m > 0.0 ? 1.0 : -1.0) * volume /
        static_cast<double>(planned.particles_per_cell);
    result.density = species.rho_init / four_pi;
    if (!std::isfinite(result.q_factor) || result.q_factor == 0.0 ||
        !std::isfinite(result.density) ||
        (species.rho_init != 0.0 && result.density == 0.0) ||
        !std::isfinite(result.q_factor * result.density) ||
        (result.density != 0.0 && result.q_factor * result.density == 0.0)) {
      throw std::overflow_error("TPIC particle profile: macrocharge normalization is not representable");
    }
    result.drift[0] = species.u0;
    result.drift[1] = species.v0;
    result.drift[2] = species.w0;
    result.thermal[0] = species.uth;
    result.thermal[1] = species.vth;
    result.thermal[2] = species.wth;
    result.cells = data::prepare_particle_cell_map(domain);
    result.ppc[0] = species.npcelx;
    result.ppc[1] = species.npcely;
    result.ppc[2] = species.npcelz;
    for (int axis = 0; axis < 3; ++axis) {
      result.origin[axis] = domain.origin[axis];
      result.spacing[axis] = domain.spacing[axis];
      result.inverse_ppc[axis] = 1.0 / static_cast<double>(result.ppc[axis]);
    }
    result.particles_per_cell = static_cast<int>(planned.particles_per_cell);
    result.count = static_cast<std::uint32_t>(planned.local_count);
    result.capacity = planned.initial_capacity;
    result.first_id = planned.first_id;
    result.center_velocity = species.center_initial_velocity;
    result.quiet_start = parameters.quiet_start;
  }
  return loads;
}

static_assert(std::is_trivially_copyable_v<ParticleLoadPlan>);

} // namespace tpic::initialization
