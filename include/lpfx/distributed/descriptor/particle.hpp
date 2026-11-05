#ifndef LPFX_DISTRIBUTED_DESCRIPTOR_PARTICLE_HPP
#define LPFX_DISTRIBUTED_DESCRIPTOR_PARTICLE_HPP

#include <type_traits>

#include <lpfx/distributed/descriptor/family.hpp>
#include <lpfx/distributed/particle/wire/contract.hpp>

namespace lpfx::detail {

template <class T>
concept ParticleMigrationDescriptorIdentityType =
    ParticleWireIdentityType<T> && std::is_empty_v<T> &&
    std::is_trivially_copyable_v<T>;

} // namespace lpfx::detail

namespace lpfx::distributed {

// The compile-time statement of migration semantics. Channel identity is a
// plan fact; routing, peer traffic and capacity are runtime plan state.
template <
    lpfx::detail::ParticleWireContractType WireContract,
    lpfx::detail::ParticleMigrationDescriptorIdentityType BoundaryIdentity,
    lpfx::detail::ParticleMigrationDescriptorIdentityType OverlapIdentity>
struct particle_migration_descriptor final {
  static constexpr operation_family family = operation_family::particle;

  using contract_type = WireContract;
  using boundary_identity = BoundaryIdentity;
  using overlap_identity = OverlapIdentity;
};

namespace detail {

template <class T>
inline constexpr bool is_particle_migration_descriptor = false;

template <class Contract, class Boundary, class Overlap>
inline constexpr bool is_particle_migration_descriptor<
    particle_migration_descriptor<Contract, Boundary, Overlap>> = true;

template <class Descriptor>
concept ParticleMigrationDescriptor =
    is_particle_migration_descriptor<std::remove_cvref_t<Descriptor>>;

} // namespace detail
} // namespace lpfx::distributed

#endif
