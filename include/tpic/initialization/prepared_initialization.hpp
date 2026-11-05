#ifndef TPIC_INITIALIZATION_PREPARED_INITIALIZATION_HPP
#define TPIC_INITIALIZATION_PREPARED_INITIALIZATION_HPP

#include <tpic/config/particle_plan.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/initialization/cases/initial_case.hpp>
#include <tpic/initialization/particles/particle_load_plan.hpp>

#include <vector>

namespace tpic::initialization {

// plan planned again with the points of every bounded support, as a restart
// is with its checkpoint's. plan comes from the same parameters and domain; a
// case without a bounded support gets it back unchanged.
[[nodiscard]] config::ParticlePlan prepare_bounded_particle_plan(
    const config::ParticlePlan& plan, const InitialCase& initial_case,
    const config::SimulationParameters& parameters, const config::PreparedDomain& domain);

// The particle load plan of every species, in species order. The input comes
// from prepare_input and is trusted.
[[nodiscard]] std::vector<ParticleLoadPlan> prepare_particle_loads(
    const config::SimulationParameters& parameters, const config::PreparedDomain& domain,
    const config::ParticlePlan& particles);

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_PREPARED_INITIALIZATION_HPP
