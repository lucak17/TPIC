#ifndef LPFX_API_DISTRIBUTED_PARTICLES_HPP
#define LPFX_API_DISTRIBUTED_PARTICLES_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

#include <lpfx/api/distributed/particle_wire.hpp>
#include <lpfx/api/distributed/requests.hpp>
#include <lpfx/distributed/binding/particle.hpp>

namespace lpfx::distributed {

template <class Descriptor, class Decoder, class Semantics> struct MigrationPolicy final {
  using descriptor_type = Descriptor;
  using flag_decoder_type = Decoder;
  Semantics semantics;
  std::uint64_t maximum_hops;
};
template <class Descriptor, class Decoder, class Semantics>
[[nodiscard]] constexpr auto migration_policy(Semantics semantics,
                                              std::uint64_t maximum_hops) noexcept {
  return MigrationPolicy<Descriptor, Decoder, Semantics>{std::move(semantics), maximum_hops};
}
template <class Backend> struct UsingFlags final {
  DeviceParticleClassificationSlots<Backend> *slots;
};
template <class Backend>
[[nodiscard]] constexpr auto using_flags(DeviceParticleClassificationSlots<Backend> &slots) noexcept {
  return UsingFlags<Backend>{&slots};
}
struct TrafficSeed final {
  std::uint64_t records;
  std::uint64_t arrival_records;
  bool reserve;
  bool bound_arrivals_by_owner = false;
};
struct ArrivalsUpToOwner final {};
inline constexpr ArrivalsUpToOwner arrivals_up_to_owner{};
[[nodiscard]] constexpr auto traffic_seed(std::uint64_t records,
    std::uint64_t arrival_records, bool reserve = true) noexcept {
  return TrafficSeed{records, arrival_records, reserve};
}
[[nodiscard]] constexpr auto traffic_seed(std::uint64_t records,
    ArrivalsUpToOwner, bool reserve = true) noexcept {
  return TrafficSeed{records, records, reserve, true};
}
template <bool Enabled> struct ResidentAccess final {};
template <bool Enabled> inline constexpr ResidentAccess<Enabled> resident_access{};

namespace detail {
template <class Key, class Owner, class Topology, class Policy, class Placement,
          class Backend, bool Resident>
struct ParticleRequest final {
  static constexpr bool collective_preparation = true;
  using key_type = Key;
  static constexpr std::uint64_t preparation_shape = 0;
  using definition_type = ParticlePlan<Key::identity, typename Policy::descriptor_type>;
  using plan_type = PreparedParticlePlan<definition_type, typename Topology::rank_type>;
  using contract_type = typename plan_type::contract_type;
  using user_semantics = decltype(std::declval<Policy>().semantics);
  using semantics_type = BoundParticleSemantics<user_semantics,
      typename Policy::flag_decoder_type, Backend, Resident>;
  using executor_type = decltype(std::declval<Placement>().executor);
  using binding_type = ParticleBinding<typename Placement::route_type, plan_type,
                                      Owner, executor_type, semantics_type>;
  Owner *owner;
  const Topology *topology;
  Policy policy;
  Placement placement;
  UsingFlags<Backend> flags;
  TrafficSeed traffic;

  template <class Setup>
  [[nodiscard]] ErrorCode prepare_plan(const Setup &setup, plan_type &plan) noexcept {
    // Agreement precedes all local validation and any topology collectives.
    LPFX_TRY(ObjectPreparation<Setup>::agree_words(setup,
        std::array<std::uint64_t, 2>{traffic.records, traffic.reserve}));
    std::uint64_t topology_hops = 0;
    static_cast<void>(try_particle_topology_hop_bound(*topology, topology_hops));
    AgreedParticleHopBound<typename plan_type::rank_type, typename contract_type::hop_policy> hops;
    LPFX_TRY(ObjectPreparation<Setup>::agree_hops(setup, *topology,
        std::min(policy.maximum_hops, topology_hops), hops));
    constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (!owner->valid() || owner->capacity() > maximum ||
        traffic.records > maximum / contract_type::record_wire_width ||
        (!traffic.bound_arrivals_by_owner && traffic.arrival_records > owner->capacity()))
      return ErrorCode::invalid_argument;
    typename plan_type::routing_type routing;
    LPFX_TRY(try_prepare_particle_routing_pattern(*topology, std::move(hops), routing));
    const auto capacity = routing.peer_count() ? traffic.records : 0;
    std::array<std::uint64_t, particle_migration_maximum_peer_count> peers{};
    peers.fill(capacity);
    const std::span<const std::uint64_t> bounds{peers.data(), routing.peer_count()};
    typename plan_type::capacity_type prepared;
    LPFX_TRY(try_prepare_particle_migration_capacity<contract_type>(bounds, bounds,
        capacity, capacity, capacity, owner->capacity(), prepared));
    return try_prepare_particle_plan(definition_type{}, routing, prepared, plan);
  }
  [[nodiscard]] binding_type binding(plan_type &plan) noexcept {
    return {plan, *owner, placement.executor,
        semantics_type{std::move(policy.semantics), *flags.slots, nullptr}};
  }
  template <class Program, class Session, class Setup>
  [[nodiscard]] ErrorCode reserve(Program &program, HostMemoryBudget &budget,
      Session &session, const Setup &setup, const plan_type &plan) noexcept {
    return traffic.reserve ? program.reserve(budget, session, setup,
        plan.pattern().capacity(), traffic.bound_arrivals_by_owner
            ? std::min<std::uint64_t>(traffic.arrival_records, owner->capacity())
            : traffic.arrival_records) : ErrorCode::success;
  }
};
} // namespace detail

template <field::fixed_string Key, std::uint64_t... Index, class Owner,
          class Topology, class Policy, class Placement, class Backend, bool Resident>
[[nodiscard]] constexpr auto migrate(Owner &owner, const Topology &topology,
    Policy policy, Placement placement, UsingFlags<Backend> flags, TrafficSeed traffic,
    ResidentAccess<Resident>) noexcept {
  return detail::ParticleRequest<detail::operation_key<Key, Index...>, Owner, Topology,
      Policy, Placement, Backend, Resident>{&owner, &topology, std::move(policy),
          std::move(placement), flags, traffic};
}
template <field::fixed_string Key, std::uint64_t... Index, class Owner,
          class Topology, class... Options>
void migrate(Owner &, const Topology &&, Options...) = delete;

} // namespace lpfx::distributed
#endif
