#ifndef TPIC_SIMULATION_ACTIONS_MOVER_ACTIONS_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_MOVER_ACTIONS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>

#include <tpic/simulation/state/species_device.hpp>
#include <tpic/ecsim/particles/particle_mover_device.hpp>
#include <tpic/ecsim/maxwell/maxwell_device.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include <array>
#include <cstddef>

namespace tpic::simulation {

// Submits the motion and classification of one species.
template <class Precision, bool Partial, class SpeciesPack, class Ports>
class MoverActions {
  static constexpr std::size_t Ns = std::tuple_size_v<SpeciesPack>;
public:
  MoverActions(const simulation::PreparedInput& input, SpeciesPack& species, Ports& ports,
      const communication::ParticleMigrationGeometry& geometry,
      const data::MagneticNodeOwner<Precision, Backend>& magnetic, unsigned species_count)
      : species_(species), ports_(ports), geometry_(geometry), magnetic_node_(magnetic),
        bypass_(!Partial && input.particles.bypass_local_migration),
        sorted_(input.particles.order != config::ParticleOrder::unsorted) {
    for (std::size_t s = 0; s < species_count; ++s)
      mover_parameters_[s] = ecsim::prepare_particle_mover_parameters<Precision>(
          input.domain, geometry, input.parameters.species[s].q_over_m, input.parameters.dt,
          input.parameters.c);
  }
  // Never reallocates or waits on host completion.
  void move(unsigned s, const ecsim::SolverNodeOwner<Precision, Backend>& electric,
            unsigned* invalid, Executor executor) const {
    visit_species(species_, s, [&]<std::size_t S>(auto& species) {
      const auto* receipt = sorted_ ? &species.ordering.receipt() : nullptr;
      if (bypass_) {
        lpfx::check_error(ecsim::enqueue_local_particle_mover<Precision>(
                              receipt, magnetic_node_, electric, species.particles,
                              geometry_, mover_parameters_[S], invalid, executor),
                          "TPIC mover: local mover submission failed");
      } else {
        lpfx::check_error(ecsim::enqueue_migrating_particle_mover<Precision>(receipt,
            magnetic_node_, electric, species.particles, ports_.template get<"particles", S>(),
            geometry_, mover_parameters_[S], invalid, executor),
            "TPIC mover: mover/classification submission failed");
      }
    });
  }

private:
  SpeciesPack& species_;
  Ports& ports_;
  const communication::ParticleMigrationGeometry& geometry_;
  const data::MagneticNodeOwner<Precision, Backend>& magnetic_node_;
  std::array<ecsim::ParticleMoverParameters<Precision>, Ns> mover_parameters_{};
  const bool bypass_, sorted_;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_MOVER_ACTIONS_DEVICE_HPP
