#ifndef LPFX_DISTRIBUTED_PLAN_PARTICLE_HPP
#define LPFX_DISTRIBUTED_PLAN_PARTICLE_HPP

#include <cstdint>

#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/descriptor/particle.hpp>
#include <lpfx/distributed/particle/migration/pattern.hpp>
#include <lpfx/distributed/particle/migration/registration.hpp>
#include <lpfx/distributed/registration/record.hpp>

namespace lpfx::distributed {

// Compile-time protocol identity of one particle channel. Runtime routing and
// traffic bounds are checked once by try_prepare_particle_plan.
template <std::uint64_t ChannelIdentity,
          detail::ParticleMigrationDescriptor Descriptor>
  requires(ChannelIdentity != 0)
struct ParticlePlan final {
  using descriptor_type = Descriptor;
  using contract_type = typename descriptor_type::contract_type;
  using boundary_identity = typename descriptor_type::boundary_identity;
  using overlap_identity = typename descriptor_type::overlap_identity;

  static constexpr std::uint64_t channel_identity = ChannelIdentity;
};

template <class Definition, lpfx::detail::CartesianRank Rank>
class PreparedParticlePlan final {
public:
  using definition_type = Definition;
  using descriptor_type = typename Definition::descriptor_type;
  using contract_type = typename descriptor_type::contract_type;
  using schema_type = typename contract_type::schema_type;
  using rank_type = Rank;
  using boundary_identity = typename descriptor_type::boundary_identity;
  using overlap_identity = typename descriptor_type::overlap_identity;
  using routing_type =
      PreparedParticleRoutingPattern<Rank, typename contract_type::hop_policy>;
  using capacity_type = PreparedParticleMigrationCapacity<contract_type>;
  using pattern_type =
      PreparedParticleMigrationPattern3D<contract_type, Rank, boundary_identity,
                                         overlap_identity>;
  using channel_type = ParticleMigrationRegistrationChannel3D<pattern_type>;
  using canonical_position_type =
      lpfx::detail::ParticleWireCanonicalPosition<contract_type>;

  static constexpr std::uint64_t channel_identity =
      Definition::channel_identity;

  constexpr PreparedParticlePlan() noexcept = default;

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return pattern_.valid();
  }

  [[nodiscard]] constexpr const pattern_type &pattern() const noexcept {
    return pattern_;
  }

  [[nodiscard]] constexpr const channel_type &
  registration_channel() const noexcept {
    return channel_;
  }

private:
  constexpr PreparedParticlePlan(const pattern_type &pattern,
                                 const channel_type &channel) noexcept
      : pattern_(pattern), channel_(channel) {}

  pattern_type pattern_{};
  channel_type channel_{};

  friend struct lpfx::detail::PrivateConstructionAccess;
};

template <class Definition, lpfx::detail::CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_plan(
    const Definition,
    const PreparedParticleRoutingPattern<
        Rank, typename Definition::contract_type::hop_policy> &routing,
    const PreparedParticleMigrationCapacity<typename Definition::contract_type>
        &capacity,
    PreparedParticlePlan<Definition, Rank> &output) noexcept {
  using Prepared = PreparedParticlePlan<Definition, Rank>;
  typename Prepared::pattern_type pattern{};
  LPFX_TRY(try_prepare_particle_migration_pattern(routing, capacity, pattern));
  ChannelId channel_id{};
  LPFX_TRY(try_make_channel_id(Definition::channel_identity, channel_id));
  typename Prepared::channel_type channel{};
  LPFX_TRY(try_prepare_particle_registration_channel(channel_id, pattern,
                                                     channel));
  output = lpfx::detail::PrivateConstructionAccess::create<Prepared>(pattern,
                                                                     channel);
  return ErrorCode::success;
}

} // namespace lpfx::distributed

#endif
