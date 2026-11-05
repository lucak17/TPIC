#ifndef TPIC_COMMUNICATION_PARTICLES_PARTICLE_MIGRATION_POLICY_HPP
#define TPIC_COMMUNICATION_PARTICLES_PARTICLE_MIGRATION_POLICY_HPP

#include <lpfx/api/distributed/particles.hpp>
#include <tpic/communication/particles/particle_wire.hpp>

#include <cstdint>

namespace tpic::communication {

// Migration routing rounds per step (CMake option TPIC_PARTICLE_MIGRATION_HOPS).
// One round covers every admitted flight: it stays inside the two-cell particle
// halo and every split axis has at least two cells per rank.
inline constexpr std::uint64_t particle_migration_hops = TPIC_CONFIG_PARTICLE_MIGRATION_HOPS;
static_assert(particle_migration_hops >= 1, "TPIC_PARTICLE_MIGRATION_HOPS must be positive");

// The wire/boundary identities and the hop bound belong to TPIC.
// Names, owners, traffic reservations and stream placement stay at the call site.
template <class Precision>
[[nodiscard]] auto particle_migration_policy(ParticleMigrationGeometry geometry) noexcept {
  namespace d = lpfx::distributed;
  using Descriptor = d::particle_migration_descriptor<
      typename ParticleMigrationWire<Precision>::Contract,
      lpfx::ParticleWireIdentity<0x5450494342430001>,
      lpfx::ParticleWireIdentity<0x545049434f520001>>;
  return d::migration_policy<Descriptor, ParticleDestinationFlagDecoder>(
      d::particle_semantics(ParticleReceivedOwner{geometry}, geometry), particle_migration_hops);
}

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_PARTICLES_PARTICLE_MIGRATION_POLICY_HPP
