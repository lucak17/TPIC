#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_PATTERN_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_PATTERN_HPP

#include <cstddef>
#include <cstdint>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/descriptor/particle.hpp>
#include <lpfx/distributed/particle/migration/capacity.hpp>
#include <lpfx/distributed/particle/migration/routing.hpp>

namespace lpfx {

// Routing and capacity of one prepared migration, with the descriptor
// identities that registration agrees on.
template <detail::ParticleWireContractType Contract, detail::CartesianRank Rank,
          detail::ParticleMigrationDescriptorIdentityType BoundaryDescriptor,
          detail::ParticleMigrationDescriptorIdentityType OverlapDescriptor>
class PreparedParticleMigrationPattern3D final {
public:
  using contract_type = Contract;
  using rank_type = Rank;
  using boundary_descriptor = BoundaryDescriptor;
  using overlap_descriptor = OverlapDescriptor;
  using routing_type =
      PreparedParticleRoutingPattern<Rank, typename Contract::hop_policy>;
  using capacity_type = PreparedParticleMigrationCapacity<Contract>;

  static constexpr std::size_t maximum_peer_count =
      routing_type::maximum_peer_count;
  static constexpr std::uint64_t boundary_descriptor_identity =
      BoundaryDescriptor::particle_wire_id;
  static constexpr std::uint64_t overlap_descriptor_identity =
      OverlapDescriptor::particle_wire_id;

  constexpr PreparedParticleMigrationPattern3D() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool valid() const noexcept {
    return routing_.valid() && capacity_.valid() &&
           routing_.peer_count() == capacity_.peer_count();
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Rank process_rank() const noexcept {
    const auto shape = routing_.process_shape();
    const auto coordinate = routing_.process_coordinate();
    return static_cast<Rank>(
        coordinate.x() +
        shape.x() *
            static_cast<Rank>(coordinate.y() + shape.y() * coordinate.z()));
  }

  // Hard precondition: peer_group < routing().peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr Rank
  peer_rank(const std::uint8_t peer_group) const noexcept {
    return routing_.peer_rank(peer_group);
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr const routing_type &
  routing() const noexcept {
    return routing_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr const capacity_type &
  capacity() const noexcept {
    return capacity_;
  }

private:
  constexpr PreparedParticleMigrationPattern3D(
      const routing_type &routing, const capacity_type &capacity) noexcept
      : routing_(routing), capacity_(capacity) {}

  routing_type routing_{};
  capacity_type capacity_{};

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

template <class T> inline constexpr bool particle_migration_pattern = false;

template <class Contract, class Rank, class Boundary, class Overlap>
inline constexpr bool particle_migration_pattern<
    PreparedParticleMigrationPattern3D<Contract, Rank, Boundary, Overlap>> =
    true;

template <class T>
concept ParticleMigrationPatternType = particle_migration_pattern<T>;

} // namespace detail

// invalid_argument for invalid or mismatched inputs or a peer equal to this
// process; overflow when a canonical rank does not fit uint32.
template <detail::ParticleWireContractType Contract, detail::CartesianRank Rank,
          detail::ParticleMigrationDescriptorIdentityType BoundaryDescriptor,
          detail::ParticleMigrationDescriptorIdentityType OverlapDescriptor>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_migration_pattern(
    const PreparedParticleRoutingPattern<Rank, typename Contract::hop_policy>
        &routing,
    const PreparedParticleMigrationCapacity<Contract> &capacity,
    PreparedParticleMigrationPattern3D<Contract, Rank, BoundaryDescriptor,
                                       OverlapDescriptor> &output) noexcept {
  using Pattern = PreparedParticleMigrationPattern3D<
      Contract, Rank, BoundaryDescriptor, OverlapDescriptor>;
  if (!routing.valid() || !capacity.valid() ||
      routing.peer_count() != capacity.peer_count()) {
    return ErrorCode::invalid_argument;
  }
  Rank process_rank{};
  Rank process_count{};
  if (!try_canonical_coordinate_to_rank(routing.process_shape(),
                                        routing.process_coordinate(),
                                        process_rank) ||
      !detail::checked_volume(routing.process_shape(), process_count)) {
    return ErrorCode::invalid_argument;
  }
  std::uint32_t canonical{};
  if (!detail::checked_integral_cast(process_rank, canonical) ||
      !detail::checked_integral_cast(process_count, canonical)) {
    return ErrorCode::overflow;
  }
  for (std::uint8_t peer = 0; peer != routing.peer_count(); ++peer) {
    if (routing.peer_rank(peer) == process_rank) {
      return ErrorCode::invalid_argument;
    }
    if (!detail::checked_integral_cast(routing.peer_rank(peer), canonical)) {
      return ErrorCode::overflow;
    }
  }
  output = detail::PrivateConstructionAccess::create<Pattern>(routing, capacity);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
