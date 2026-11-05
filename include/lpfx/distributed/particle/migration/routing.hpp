#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_ROUTING_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_ROUTING_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/particle/wire/contract.hpp>
#include <lpfx/distributed/plan/particle_destination.hpp>
#include <lpfx/domain/cartesian/peer_map.hpp>

namespace lpfx {

namespace detail {

template <class Policy>
concept FixedParticleHopPolicyType =
    ParticleHopPolicyType<Policy> &&
    (Policy::kind == ParticleHopPolicyKind::fixed);

template <class Policy>
concept AdaptiveParticleHopPolicyType =
    ParticleHopPolicyType<Policy> &&
    (Policy::kind == ParticleHopPolicyKind::adaptive);

template <CartesianRank Rank>
[[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
particle_topology_axis_hop_bound(const Rank extent,
                                 const bool periodic) noexcept {
  return periodic ? static_cast<std::uint64_t>(extent / Rank{2})
                  : static_cast<std::uint64_t>(extent - Rank{1});
}

template <CartesianRank Rank>
[[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
particle_topology_hop_bound(const CartesianExtent3<Rank> shape,
                            const CartesianPeriodicity3 periodicity) noexcept {
  std::uint64_t bound =
      particle_topology_axis_hop_bound(shape.x(), periodicity.x);
  const std::uint64_t y =
      particle_topology_axis_hop_bound(shape.y(), periodicity.y);
  const std::uint64_t z =
      particle_topology_axis_hop_bound(shape.z(), periodicity.z);
  if (y > bound) {
    bound = y;
  }
  if (z > bound) {
    bound = z;
  }
  return bound;
}

template <CartesianRank Rank>
[[nodiscard]] constexpr std::uint64_t particle_topology_hop_bound(
    const CanonicalCartesianTopologyInput3D<Rank> &input) noexcept {
  return particle_topology_hop_bound(input.process_shape(),
                                     input.periodicity());
}

} // namespace detail

// A rank-agreed adaptive hop bound for one topology. Routing preparation
// consumes it; moving it disarms the source.
template <detail::CartesianRank Rank,
          detail::AdaptiveParticleHopPolicyType HopPolicy>
class AgreedParticleHopBound final {
public:
  constexpr AgreedParticleHopBound() noexcept = default;

  constexpr AgreedParticleHopBound(AgreedParticleHopBound &&other) noexcept
      : topology_input_(other.topology_input_), bound_(other.bound_),
        active_(std::exchange(other.active_, false)) {}

  constexpr AgreedParticleHopBound &
  operator=(AgreedParticleHopBound &&other) noexcept {
    topology_input_ = other.topology_input_;
    bound_ = other.bound_;
    active_ = std::exchange(other.active_, false);
    return *this;
  }

  [[nodiscard]] constexpr bool active() const noexcept { return active_; }

  // Hard precondition: active().
  [[nodiscard]] constexpr std::uint64_t value() const noexcept {
    return bound_;
  }

  [[nodiscard]] constexpr bool matches(
      const CanonicalCartesianTopologyInput3D<Rank> &topology_input)
      const noexcept {
    return active_ && topology_input_ == topology_input;
  }

private:
  constexpr AgreedParticleHopBound(
      const CanonicalCartesianTopologyInput3D<Rank> topology_input,
      const std::uint64_t bound) noexcept
      : topology_input_(topology_input), bound_(bound), active_(true) {}

  CanonicalCartesianTopologyInput3D<Rank> topology_input_{};
  std::uint64_t bound_{};
  bool active_ = false;

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

template <CartesianRank Rank>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr void particle_route_axis(
    const Rank current, const Rank owner, const Rank extent,
    const bool periodic, std::int8_t &step, std::uint64_t &distance) noexcept {
  if (current == owner) {
    step = 0;
    distance = 0;
    return;
  }

  if (!periodic) {
    step = owner > current ? 1 : -1;
    distance = owner > current ? owner - current : current - owner;
    return;
  }

  const Rank plus = owner > current
                        ? static_cast<Rank>(owner - current)
                        : static_cast<Rank>(extent - (current - owner));
  const Rank minus = current > owner
                         ? static_cast<Rank>(current - owner)
                         : static_cast<Rank>(extent - (owner - current));
  // An even-extent antipodal tie takes the minus direction.
  step = plus < minus ? 1 : -1;
  distance = plus < minus ? plus : minus;
}

// First-hop direction and hop count from current to owner. Hard
// precondition: both coordinates lie inside shape.
template <CartesianRank Rank>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint64_t
particle_route(const CartesianExtent3<Rank> shape,
               const CartesianPeriodicity3 periodicity,
               const CartesianIndex3<Rank> current,
               const CartesianIndex3<Rank> owner,
               CartesianDirection3 &direction) noexcept {
  std::uint64_t x_distance, y_distance, z_distance;
  particle_route_axis(current.x(), owner.x(), shape.x(), periodicity.x,
                      direction.x, x_distance);
  particle_route_axis(current.y(), owner.y(), shape.y(), periodicity.y,
                      direction.y, y_distance);
  particle_route_axis(current.z(), owner.z(), shape.z(), periodicity.z,
                      direction.z, z_distance);
  std::uint64_t hops = x_distance;
  if (y_distance > hops)
    hops = y_distance;
  if (z_distance > hops)
    hops = z_distance;
  return hops;
}

template <CartesianRank Rank>
[[nodiscard]] constexpr bool
particle_route_axis_step_reachable(const Rank coordinate, const Rank extent,
                                   const bool periodic,
                                   const std::int8_t step) noexcept {
  if (step == 0) {
    return true;
  }
  if (!periodic) {
    return step < 0 ? coordinate != 0
                    : coordinate != static_cast<Rank>(extent - Rank{1});
  }
  if (extent <= Rank{1}) {
    return false;
  }
  // For extent two both native neighbors are the same rank and the antipodal
  // tie rule makes only the minus route reachable.
  return extent != Rank{2} || step < 0;
}

// Routing selects directions before the peer map deduplicates ranks, so this
// mask permits asymmetric periodic antipodal routes.
template <CartesianRank Rank>
[[nodiscard]] constexpr CartesianNeighborMask3D particle_reachable_direction_mask(
    const CanonicalCartesianTopologyInput3D<Rank> &input) noexcept {
  const auto shape = input.process_shape();
  const auto coordinate = input.process_coordinate();
  const auto periodicity = input.periodicity();
  CartesianNeighborMask3D selected{};
  for (std::size_t d = 0; d != canonical_cartesian_neighbor_directions.size();
       ++d) {
    const auto direction = canonical_cartesian_neighbor_directions[d];
    if (particle_route_axis_step_reachable(coordinate.x(), shape.x(),
                                           periodicity.x, direction.x) &&
        particle_route_axis_step_reachable(coordinate.y(), shape.y(),
                                           periodicity.y, direction.y) &&
        particle_route_axis_step_reachable(coordinate.z(), shape.z(),
                                           periodicity.z, direction.z))
      selected = selected | cartesian_neighbor_bit(d);
  }
  return selected;
}

} // namespace detail

template <detail::CartesianRank Rank>
[[nodiscard]] LPFX_HOST_DEVICE constexpr bool
try_particle_route_direction_and_hops(
    const CartesianExtent3<Rank> process_shape,
    const CartesianPeriodicity3 periodicity,
    const CartesianIndex3<Rank> current,
    const CartesianIndex3<Rank> final_owner, CartesianDirection3 &direction,
    std::uint64_t &record_hops) noexcept {
  if (process_shape.x() == 0 || process_shape.y() == 0 ||
      process_shape.z() == 0 || current.x() >= process_shape.x() ||
      current.y() >= process_shape.y() || current.z() >= process_shape.z() ||
      final_owner.x() >= process_shape.x() ||
      final_owner.y() >= process_shape.y() ||
      final_owner.z() >= process_shape.z()) {
    return false;
  }
  CartesianDirection3 prepared{};
  record_hops = detail::particle_route(process_shape, periodicity, current,
                                       final_owner, prepared);
  direction = prepared;
  return true;
}

template <detail::CartesianRank Rank>
[[nodiscard]] constexpr bool try_particle_topology_hop_bound(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    std::uint64_t &output) noexcept {
  if (!topology.is_valid()) {
    return false;
  }
  output = detail::particle_topology_hop_bound(topology.input());
  return true;
}

template <detail::CartesianRank Rank, detail::ParticleHopPolicyType HopPolicy>
class PreparedParticleRoutingPattern final {
public:
  static constexpr std::size_t maximum_peer_count =
      PreparedCartesianPeerMap3D<Rank>::maximum_peer_count;

  using rank_type = Rank;
  using hop_policy = HopPolicy;

  constexpr PreparedParticleRoutingPattern() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool valid() const noexcept {
    return process_shape_.x() != 0;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianExtent3<Rank>
  process_shape() const noexcept {
    return process_shape_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianIndex3<Rank>
  process_coordinate() const noexcept {
    return process_coordinate_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianPeriodicity3
  periodicity() const noexcept {
    return periodicity_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  topology_hop_bound() const noexcept {
    return valid() ? detail::particle_topology_hop_bound(process_shape_,
                                                         periodicity_)
                   : 0;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  round_count() const noexcept {
    return round_count_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  peer_count() const noexcept {
    return peers_.peer_count();
  }

  // Hard precondition: group < peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr Rank
  peer_rank(const std::uint8_t group) const noexcept {
    return peers_.peer_rank(group);
  }

  // Hard precondition: direction is a Cartesian neighbor direction.
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  peer_group(const CartesianDirection3 direction) const noexcept {
    return peers_.peer_group(cartesian_neighbor_index(direction));
  }

  // Bucket of a record whose final owner is final_owner: resident, or the
  // first-hop peer group. invalid_argument outside the process shape or for
  // an unprepared first hop; capacity_exceeded beyond round_count().
  // Hard precondition: valid().
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr ErrorCode
  try_route(const CartesianIndex3<Rank> final_owner,
            std::uint8_t &bucket) const noexcept {
    if (final_owner.x() >= process_shape_.x() ||
        final_owner.y() >= process_shape_.y() ||
        final_owner.z() >= process_shape_.z()) {
      return ErrorCode::invalid_argument;
    }
    CartesianDirection3 direction;
    const std::uint64_t hops =
        detail::particle_route(process_shape_, periodicity_,
                               process_coordinate_, final_owner, direction);
    if (hops > round_count_) {
      return ErrorCode::capacity_exceeded;
    }
    if (hops == 0) {
      bucket = particle_bucket_resident;
      return ErrorCode::success;
    }
    const std::uint8_t peer = peer_group(direction);
    if (peer >= peer_count()) {
      return ErrorCode::invalid_argument;
    }
    bucket = peer;
    return ErrorCode::success;
  }

private:
  constexpr PreparedParticleRoutingPattern(
      const CartesianExtent3<Rank> process_shape,
      const CartesianIndex3<Rank> process_coordinate,
      const CartesianPeriodicity3 periodicity, const std::uint64_t round_count,
      const PreparedCartesianPeerMap3D<Rank> peers) noexcept
      : process_shape_(process_shape), process_coordinate_(process_coordinate),
        periodicity_(periodicity), round_count_(round_count),
        peers_(peers) {}

  CartesianExtent3<Rank> process_shape_{};
  CartesianIndex3<Rank> process_coordinate_{};
  CartesianPeriodicity3 periodicity_{};
  std::uint64_t round_count_{};
  PreparedCartesianPeerMap3D<Rank> peers_{};

  friend struct detail::PrivateConstructionAccess;
};

// The bucket of a classified destination: discard, resident or the first-hop
// peer group. invalid_argument for an unset destination; otherwise as
// PreparedParticleRoutingPattern::try_route.
template <detail::CartesianRank Rank, detail::ParticleHopPolicyType HopPolicy>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr ErrorCode
route_bucket(const PreparedParticleRoutingPattern<Rank, HopPolicy> &routing,
             const distributed::ParticleDestination3D<Rank> &destination,
             std::uint8_t &bucket) noexcept {
  if (!destination.valid()) {
    return ErrorCode::invalid_argument;
  }
  if (destination.discarded()) {
    bucket = particle_bucket_discard;
    return ErrorCode::success;
  }
  return routing.try_route(destination.coordinate(), bucket);
}

namespace detail {

template <CartesianRank Rank, ParticleHopPolicyType HopPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_routing_pattern(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const std::uint64_t round_count,
    PreparedParticleRoutingPattern<Rank, HopPolicy> &output) noexcept {
  std::uint64_t topology_bound{};
  if (!try_particle_topology_hop_bound(topology, topology_bound)) {
    return ErrorCode::invalid_argument;
  }
  if (round_count > topology_bound) {
    return ErrorCode::capacity_exceeded;
  }
  const auto &input = topology.input();
  PreparedCartesianPeerMap3D<Rank> peers{};
  LPFX_TRY(try_prepare_cartesian_peer_map(
      topology, particle_reachable_direction_mask(input), peers));
  output = PrivateConstructionAccess::create<
      PreparedParticleRoutingPattern<Rank, HopPolicy>>(
      input.process_shape(), input.process_coordinate(), input.periodicity(),
      round_count, peers);
  return ErrorCode::success;
}

} // namespace detail

template <detail::CartesianRank Rank,
          detail::FixedParticleHopPolicyType HopPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_routing_pattern(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    PreparedParticleRoutingPattern<Rank, HopPolicy> &output) noexcept {
  return detail::try_prepare_particle_routing_pattern(
      topology, HopPolicy::fixed_hop_bound, output);
}

template <detail::CartesianRank Rank,
          detail::AdaptiveParticleHopPolicyType HopPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_routing_pattern(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    AgreedParticleHopBound<Rank, HopPolicy> &&agreement,
    PreparedParticleRoutingPattern<Rank, HopPolicy> &output) noexcept {
  if (!topology.is_valid() || !agreement.matches(topology.input())) {
    return ErrorCode::invalid_argument;
  }
  LPFX_TRY(detail::try_prepare_particle_routing_pattern(
      topology, agreement.value(), output));
  const AgreedParticleHopBound<Rank, HopPolicy> consumed{std::move(agreement)};
  static_cast<void>(consumed);
  return ErrorCode::success;
}

// Routing combined with the two classification callbacks of one prepared
// migration. Host loops and GPU kernels instantiate the same value type.
template <class Routing, class InitialOwner, class ReceivedOwner>
class ParticleMigrationProtocol final {
public:
  using routing_type = Routing;
  using rank_type = typename Routing::rank_type;

  constexpr ParticleMigrationProtocol(
      const Routing routing, const InitialOwner initial_owner,
      const ReceivedOwner received_owner) noexcept
      : routing_(routing), initial_owner_(initial_owner),
        received_owner_(received_owner) {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr const Routing &
  routing() const noexcept {
    return routing_;
  }

  // Bucket of source record lane. A callback may name a destination or,
  // through a ParticleClassificationAction, the bucket itself.
  template <class View, class Lane>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr ErrorCode
  classify_initial(const View source, const Lane lane,
                   std::uint8_t &bucket) const noexcept {
    if constexpr (distributed::detail::ParticleInitialActionCallback<
                      InitialOwner, View, Lane>) {
      distributed::ParticleClassificationAction action{};
      if (!initial_owner_(source, lane, action) || !action.valid()) {
        return ErrorCode::invalid_argument;
      }
      bucket = action.bucket();
      return bucket < particle_bucket_discard &&
                     bucket >= routing_.peer_count()
                 ? ErrorCode::invalid_argument
                 : ErrorCode::success;
    } else {
      distributed::ParticleDestination3D<rank_type> destination{};
      if (!initial_owner_(source, static_cast<std::uint32_t>(lane),
                          destination)) {
        return ErrorCode::invalid_argument;
      }
      return route_bucket(routing_, destination, bucket);
    }
  }

  // Bucket of a received record: resident means final owner.
  template <class Position>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr ErrorCode
  classify_received(const std::uint64_t round, const std::uint32_t lane,
                    const Position &position,
                    std::uint8_t &bucket) const noexcept {
    distributed::ParticleDestination3D<rank_type> destination{};
    if (!received_owner_(round, lane, position, destination)) {
      return ErrorCode::invalid_argument;
    }
    return route_bucket(routing_, destination, bucket);
  }

private:
  Routing routing_;
  [[no_unique_address]] InitialOwner initial_owner_;
  [[no_unique_address]] ReceivedOwner received_owner_;
};

} // namespace lpfx

#endif
