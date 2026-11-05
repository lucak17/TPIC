#ifndef TPIC_INITIALIZATION_PARTICLES_PARTICLE_LOAD_PLAN_HPP
#define TPIC_INITIALIZATION_PARTICLES_PARTICLE_LOAD_PLAN_HPP

#include <tpic/data/particles/particle_cell.hpp>

#include <cstdint>

namespace tpic::initialization {

// Velocity centering reduces the charge sum and the three charge-weighted
// sums of the velocity without the loaded drift: one device reduction and one
// result per sum. The memory
// estimate sizes the centering storage with this count.
inline constexpr int velocity_centering_reductions = 4;

// Small kernel argument, independent of particle population size. Prepare once
// per species, before native memory reporting and communication registration.
struct ParticleLoadPlan {
  std::uint64_t seed{};
  // The species' uniform Maxwellian: the macrocharge factor, the internal
  // density rhoINIT/(4*pi), the configured drift and the thermal speeds.
  double q_factor{}, density{}, drift[3]{}, thermal[3]{};
  data::ParticleCellMap cells;
  double origin[3]{}, spacing[3]{}, inverse_ppc[3]{};
  int ppc[3]{};
  int particles_per_cell{};
  std::uint32_t count{}, capacity{};
  std::uint64_t first_id{};
  bool center_velocity{};
  // QuietStart: the host launches the subcell-centre instantiation of the load
  // kernel instead of the random one; no kernel reads it.
  bool quiet_start{};
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_PARTICLES_PARTICLE_LOAD_PLAN_HPP
