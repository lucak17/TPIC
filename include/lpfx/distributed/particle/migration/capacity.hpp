#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_CAPACITY_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_CAPACITY_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/fixed_array.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/distributed/particle/limits.hpp>
#include <lpfx/distributed/particle/wire/image.hpp>

namespace lpfx {

// Per-peer and summed particle bounds of one migration. Every bound is
// uint32-portable, so device counts and wire offsets fit uint32 and every
// bound times the record width fits uint64.
template <detail::ParticleWireContractType Contract>
class PreparedParticleMigrationCapacity final {
public:
  static constexpr std::size_t maximum_peer_count =
      particle_migration_maximum_peer_count;

  using contract_type = Contract;

  constexpr PreparedParticleMigrationCapacity() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool valid() const noexcept {
    return valid_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  peer_count() const noexcept {
    return peer_count_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  record_wire_bytes() const noexcept {
    return Contract::record_wire_width;
  }

  // Hard precondition: peer < peer_count(). These are per-peer bounds for
  // every round; the summed initial and transit bounds are independent.
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_peer_outgoing(const std::uint8_t peer) const noexcept {
    return maximum_peer_outgoing_[peer];
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_peer_incoming(const std::uint8_t peer) const noexcept {
    return maximum_peer_incoming_[peer];
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_outgoing() const noexcept {
    return maximum_outgoing_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_incoming() const noexcept {
    return maximum_incoming_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_transit() const noexcept {
    return maximum_transit_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_final() const noexcept {
    return maximum_final_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_outgoing_bytes() const noexcept {
    return maximum_outgoing_ * Contract::record_wire_width;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_incoming_bytes() const noexcept {
    return maximum_incoming_ * Contract::record_wire_width;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  maximum_transit_bytes() const noexcept {
    return maximum_transit_ * Contract::record_wire_width;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const PreparedParticleMigrationCapacity &left,
             const PreparedParticleMigrationCapacity &right) noexcept {
    if (left.valid_ != right.valid_ || left.peer_count_ != right.peer_count_ ||
        left.maximum_outgoing_ != right.maximum_outgoing_ ||
        left.maximum_incoming_ != right.maximum_incoming_ ||
        left.maximum_transit_ != right.maximum_transit_ ||
        left.maximum_final_ != right.maximum_final_) {
      return false;
    }
    for (std::uint8_t peer = 0; peer != left.peer_count_; ++peer) {
      if (left.maximum_peer_outgoing_[peer] !=
              right.maximum_peer_outgoing_[peer] ||
          left.maximum_peer_incoming_[peer] !=
              right.maximum_peer_incoming_[peer]) {
        return false;
      }
    }
    return true;
  }

private:
  constexpr PreparedParticleMigrationCapacity(
      const std::uint8_t peer_count,
      const std::array<std::uint64_t, maximum_peer_count> maximum_peer_outgoing,
      const std::array<std::uint64_t, maximum_peer_count> maximum_peer_incoming,
      const std::uint64_t maximum_outgoing,
      const std::uint64_t maximum_incoming, const std::uint64_t maximum_transit,
      const std::uint64_t maximum_final) noexcept
      : maximum_peer_outgoing_(maximum_peer_outgoing),
        maximum_peer_incoming_(maximum_peer_incoming),
        maximum_outgoing_(maximum_outgoing),
        maximum_incoming_(maximum_incoming), maximum_transit_(maximum_transit),
        maximum_final_(maximum_final), peer_count_(peer_count), valid_(true) {}

  detail::DeviceFixedArray<std::uint64_t, maximum_peer_count>
      maximum_peer_outgoing_{};
  detail::DeviceFixedArray<std::uint64_t, maximum_peer_count>
      maximum_peer_incoming_{};
  std::uint64_t maximum_outgoing_{};
  std::uint64_t maximum_incoming_{};
  std::uint64_t maximum_transit_{};
  std::uint64_t maximum_final_{};
  std::uint8_t peer_count_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

// invalid_argument for mismatched or oversized peer spans, a zero record
// width, or a peer bound above its summed bound; capacity_exceeded for a
// bound above UINT32_MAX; overflow when a summed byte bound overflows.
template <detail::ParticleWireContractType Contract>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_migration_capacity(
    const std::span<const std::uint64_t> maximum_peer_outgoing,
    const std::span<const std::uint64_t> maximum_peer_incoming,
    const std::uint64_t maximum_outgoing, const std::uint64_t maximum_incoming,
    const std::uint64_t maximum_transit, const std::uint64_t maximum_final,
    PreparedParticleMigrationCapacity<Contract> &output) noexcept {
  using Capacity = PreparedParticleMigrationCapacity<Contract>;
  if (maximum_peer_outgoing.size() != maximum_peer_incoming.size() ||
      maximum_peer_outgoing.size() > Capacity::maximum_peer_count ||
      Contract::record_wire_width == 0) {
    return ErrorCode::invalid_argument;
  }

  std::array<std::uint64_t, Capacity::maximum_peer_count> outgoing{};
  std::array<std::uint64_t, Capacity::maximum_peer_count> incoming{};
  for (std::size_t peer = 0; peer != maximum_peer_outgoing.size(); ++peer) {
    if (maximum_peer_outgoing[peer] > (maximum_outgoing > maximum_transit
                                           ? maximum_outgoing
                                           : maximum_transit) ||
        maximum_peer_incoming[peer] > maximum_incoming) {
      return ErrorCode::invalid_argument;
    }
    outgoing[peer] = maximum_peer_outgoing[peer];
    incoming[peer] = maximum_peer_incoming[peer];
  }

  // Peer bounds are below the summed bounds checked here.
  constexpr std::uint64_t portable = std::numeric_limits<std::uint32_t>::max();
  if (maximum_outgoing > portable || maximum_incoming > portable ||
      maximum_transit > portable || maximum_final > portable) {
    return ErrorCode::capacity_exceeded;
  }
  std::uint64_t bytes{};
  if (!detail::checked_multiply(maximum_outgoing, Contract::record_wire_width,
                                bytes) ||
      !detail::checked_multiply(maximum_incoming, Contract::record_wire_width,
                                bytes) ||
      !detail::checked_multiply(maximum_transit, Contract::record_wire_width,
                                bytes)) {
    return ErrorCode::overflow;
  }

  output = detail::PrivateConstructionAccess::create<Capacity>(
      static_cast<std::uint8_t>(maximum_peer_outgoing.size()), outgoing,
      incoming, maximum_outgoing, maximum_incoming, maximum_transit,
      maximum_final);
  return ErrorCode::success;
}

namespace detail {

// Elementwise maximum of two capacities over the same peers.
template <ParticleWireContractType Contract>
[[nodiscard]] constexpr ErrorCode try_merge_particle_migration_capacity(
    const PreparedParticleMigrationCapacity<Contract> &first,
    const PreparedParticleMigrationCapacity<Contract> &second,
    PreparedParticleMigrationCapacity<Contract> &output) noexcept {
  if (!first.valid() || !second.valid() ||
      first.peer_count() != second.peer_count()) {
    return ErrorCode::invalid_argument;
  }
  const auto maximum = [](const std::uint64_t a,
                          const std::uint64_t b) constexpr noexcept {
    return a > b ? a : b;
  };
  std::array<std::uint64_t, particle_migration_maximum_peer_count> outgoing{},
      incoming{};
  const std::uint8_t peers = first.peer_count();
  for (std::uint8_t peer = 0; peer != peers; ++peer) {
    outgoing[peer] = maximum(first.maximum_peer_outgoing(peer),
                             second.maximum_peer_outgoing(peer));
    incoming[peer] = maximum(first.maximum_peer_incoming(peer),
                             second.maximum_peer_incoming(peer));
  }
  return try_prepare_particle_migration_capacity<Contract>(
      {outgoing.data(), peers}, {incoming.data(), peers},
      maximum(first.maximum_outgoing(), second.maximum_outgoing()),
      maximum(first.maximum_incoming(), second.maximum_incoming()),
      maximum(first.maximum_transit(), second.maximum_transit()),
      maximum(first.maximum_final(), second.maximum_final()), output);
}

// The one publisher of admitted prefixes.
struct ParticlePeerPrefixAdmission final {
  template <bool EnforceCapacity, ParticleWireContractType Contract,
            ParticlePeerPrefixKind Kind>
  [[nodiscard]] LPFX_ALWAYS_INLINE static constexpr ErrorCode
  try_admit(const PreparedParticleMigrationCapacity<Contract> &capacity,
            const std::span<const std::uint64_t> counts,
            const std::uint64_t summed_limit,
            PreparedParticlePeerPrefixes<Contract, Kind> &output) noexcept {
    if (!capacity.valid() || counts.size() != capacity.peer_count()) {
      return ErrorCode::invalid_argument;
    }

    std::array<std::uint64_t, particle_migration_maximum_peer_count + 1>
        particle_offsets{};
    std::uint64_t total = 0;
    for (std::size_t peer = 0; peer != counts.size(); ++peer) {
      if constexpr (EnforceCapacity) {
        const std::uint64_t peer_limit =
            Kind == ParticlePeerPrefixKind::incoming
                ? capacity.maximum_peer_incoming(
                      static_cast<std::uint8_t>(peer))
                : capacity.maximum_peer_outgoing(
                      static_cast<std::uint8_t>(peer));
        if (counts[peer] > peer_limit) {
          return ErrorCode::capacity_exceeded;
        }
      }
      if (!checked_add(total, counts[peer], total)) {
        return ErrorCode::overflow;
      }
      // A summed limit is uint32-portable, so the byte extent fits uint64.
      if (total > summed_limit) {
        return ErrorCode::capacity_exceeded;
      }
      particle_offsets[peer + 1] = total;
    }
    if constexpr (!EnforceCapacity) {
      std::uint64_t bytes{};
      if (!checked_multiply(total, Contract::record_wire_width, bytes))
        return ErrorCode::overflow;
    }

    // Publish in place: an intermediate prepared value made compilers keep a
    // second offset array on the hot path.
    for (std::size_t peer = 0; peer != particle_offsets.size(); ++peer) {
      output.particle_offsets_[peer] = particle_offsets[peer];
    }
    output.peer_count_ = capacity.peer_count();
    return ErrorCode::success;
  }
};

} // namespace detail

// Count-only initial classification precedes allocation admission; the
// driver admits the wire storage before packing uses these prefixes.
template <detail::ParticleWireContractType Contract>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ErrorCode
try_prepare_initial_particle_outgoing_counts(
    const PreparedParticleMigrationCapacity<Contract> &capacity,
    const std::span<const std::uint64_t> outgoing_counts,
    PreparedParticleOutgoingPrefixes<Contract> &output) noexcept {
  return detail::ParticlePeerPrefixAdmission::try_admit<false>(
      capacity, outgoing_counts, UINT32_MAX, output);
}

template <detail::ParticleWireContractType Contract>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ErrorCode
try_admit_initial_particle_outgoing(
    const PreparedParticleMigrationCapacity<Contract> &capacity,
    const std::span<const std::uint64_t> outgoing_counts,
    PreparedParticleOutgoingPrefixes<Contract> &output) noexcept {
  return detail::ParticlePeerPrefixAdmission::try_admit<true>(
      capacity, outgoing_counts, capacity.maximum_outgoing(), output);
}

template <detail::ParticleWireContractType Contract>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ErrorCode
try_admit_transit_particle_outgoing(
    const PreparedParticleMigrationCapacity<Contract> &capacity,
    const std::span<const std::uint64_t> outgoing_counts,
    const std::uint64_t expected_transit_count,
    PreparedParticleOutgoingPrefixes<Contract> &output) noexcept {
  PreparedParticleOutgoingPrefixes<Contract> prepared{};
  LPFX_TRY(detail::ParticlePeerPrefixAdmission::try_admit<true>(
      capacity, outgoing_counts, capacity.maximum_transit(), prepared));
  if (prepared.total_particles() != expected_transit_count) {
    return ErrorCode::invalid_argument;
  }
  output = prepared;
  return ErrorCode::success;
}

template <detail::ParticleWireContractType Contract>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ErrorCode
try_admit_particle_incoming(
    const PreparedParticleMigrationCapacity<Contract> &capacity,
    const std::span<const std::uint64_t> incoming_counts,
    PreparedParticleIncomingPrefixes<Contract> &output) noexcept {
  return detail::ParticlePeerPrefixAdmission::try_admit<true>(
      capacity, incoming_counts, capacity.maximum_incoming(), output);
}

namespace detail {

// Largest wire record count addressable as size_t bytes, clipped to the
// portable uint32 bound. Zero for a zero record width.
[[nodiscard]] constexpr std::uint64_t
particle_migration_wire_particle_maximum(
    const std::uint64_t record_wire_width) noexcept {
  if (record_wire_width == 0)
    return 0;
  constexpr auto portable_maximum = std::uint64_t{UINT32_MAX};
  constexpr auto addressable_bytes =
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  const auto byte_maximum = addressable_bytes / record_wire_width;
  return byte_maximum < portable_maximum ? byte_maximum : portable_maximum;
}

// The particle-owner capacity policy applied to one migration bound: a bound
// that fits is kept, a genuine increase gets the configured headroom and
// rounding, clipped at the inclusive native limit (which need not be a
// capacity multiple).
[[nodiscard]] constexpr ErrorCode try_grow_particle_migration_bound(
    const std::uint64_t current, const std::uint64_t required,
    const std::uint64_t maximum, std::uint64_t &output) noexcept {
  if (current > maximum || required > maximum)
    return ErrorCode::capacity_exceeded;
  if (required <= current) {
    output = current;
    return ErrorCode::success;
  }
  std::uint64_t prepared = 0;
  LPFX_TRY(lpfx::try_prepare_particle_capacity(required, prepared));
  output = prepared < maximum ? prepared : maximum;
  return ErrorCode::success;
}

// Final and owner bounds of one round from its summed peer counts. With a
// deferred owner (device) only wire and arrival capacity is admitted: the
// cumulative final bound saturates at UINT32_MAX and owner stays 0. The
// summed-count portability check is compiled out when the per-peer maximum
// already bounds it.
template <bool DeferredOwner,
          std::uint64_t MaximumPeerParticles = std::uint64_t{UINT32_MAX}>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ErrorCode
try_particle_round_bounds(const std::uint64_t outgoing,
                          const std::uint64_t incoming,
                          const std::uint64_t residents,
                          const std::uint64_t prior_finals,
                          std::uint64_t &finals, std::uint64_t &owner) noexcept {
  if constexpr (DeferredOwner) {
    if (outgoing > UINT32_MAX || incoming > UINT32_MAX ||
        prior_finals > UINT32_MAX)
      return ErrorCode::capacity_exceeded;
    finals = incoming > UINT32_MAX - prior_finals ? UINT32_MAX
                                                  : prior_finals + incoming;
    owner = 0;
  } else {
    constexpr bool totals_fit =
        MaximumPeerParticles <=
        UINT32_MAX / particle_migration_maximum_peer_count;
    if (!checked_add(prior_finals, incoming, finals) ||
        !checked_add(residents, finals, owner))
      return ErrorCode::overflow;
    if ((!totals_fit && (outgoing > UINT32_MAX || incoming > UINT32_MAX)) ||
        owner > UINT32_MAX)
      return ErrorCode::capacity_exceeded;
  }
  return ErrorCode::success;
}

// A received record can be final, discarded or routed to any next peer, so
// each round admits the full incoming batch for every disposition before a
// payload or decode is submitted. These are storage bounds, not predictions;
// the summed transit allocation remains one batch. Exact classified finals
// and the final owner population are checked in their own phases.
template <bool DeferredOwner = false, ParticleWireContractType Contract>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_round_capacity(
    const PreparedParticleMigrationCapacity<Contract> &current,
    const std::span<const std::uint64_t> outgoing_counts,
    const std::span<const std::uint64_t> incoming_counts,
    const std::uint64_t residents, const std::uint64_t prior_finals,
    const bool transit_possible,
    PreparedParticleMigrationCapacity<Contract> &output,
    std::uint64_t &required_owner, bool &changed,
    const std::uint64_t maximum_peer_particles = UINT32_MAX,
    const bool initial_round = true) noexcept {
  if (!current.valid() || outgoing_counts.size() != current.peer_count() ||
      incoming_counts.size() != current.peer_count())
    return ErrorCode::invalid_argument;
  std::uint64_t outgoing_total = 0, incoming_total = 0;
  for (std::size_t peer = 0; peer != incoming_counts.size(); ++peer) {
    if (outgoing_counts[peer] > maximum_peer_particles ||
        incoming_counts[peer] > maximum_peer_particles)
      return ErrorCode::capacity_exceeded;
    if (!checked_add(outgoing_total, outgoing_counts[peer], outgoing_total) ||
        !checked_add(incoming_total, incoming_counts[peer], incoming_total))
      return ErrorCode::overflow;
  }
  std::uint64_t final_bound = 0, owner_bound = 0;
  LPFX_TRY(try_particle_round_bounds<DeferredOwner>(
      outgoing_total, incoming_total, residents, prior_finals, final_bound,
      owner_bound));

  const auto maximum = [](const std::uint64_t a,
                          const std::uint64_t b) constexpr noexcept {
    return a > b ? a : b;
  };
  constexpr auto portable_maximum = std::uint64_t{UINT32_MAX};
  constexpr auto wire_maximum =
      particle_migration_wire_particle_maximum(Contract::record_wire_width);
  const auto peer_maximum = maximum_peer_particles < wire_maximum
                                ? maximum_peer_particles
                                : wire_maximum;
  const auto transit = transit_possible ? incoming_total : UINT64_C(0);
  const auto peer_transit = transit < peer_maximum ? transit : peer_maximum;
  std::array<std::uint64_t, particle_migration_maximum_peer_count>
      outgoing{}, incoming{};
  std::uint64_t initial_peer_growth = 0, transit_peer_growth = 0,
                largest_peer_incoming = 0;
  for (std::uint8_t peer = 0; peer != current.peer_count(); ++peer) {
    LPFX_TRY(try_grow_particle_migration_bound(
        current.maximum_peer_outgoing(peer),
        maximum(outgoing_counts[peer], peer_transit), peer_maximum,
        outgoing[peer]));
    LPFX_TRY(try_grow_particle_migration_bound(
        current.maximum_peer_incoming(peer), incoming_counts[peer],
        peer_maximum, incoming[peer]));
    if (outgoing[peer] > current.maximum_peer_outgoing(peer)) {
      if (initial_round && outgoing_counts[peer] >= peer_transit)
        initial_peer_growth = maximum(initial_peer_growth, outgoing[peer]);
      if (!initial_round ||
          (transit_possible && peer_transit >= outgoing_counts[peer]))
        transit_peer_growth = maximum(transit_peer_growth, outgoing[peer]);
    }
    largest_peer_incoming = maximum(largest_peer_incoming, incoming[peer]);
  }
  std::uint64_t maximum_outgoing = 0, maximum_incoming = 0,
                maximum_transit = 0, maximum_final = 0;
  LPFX_TRY(try_grow_particle_migration_bound(
      current.maximum_outgoing(), initial_round ? outgoing_total : UINT64_C(0),
      wire_maximum, maximum_outgoing));
  LPFX_TRY(try_grow_particle_migration_bound(
      current.maximum_incoming(), incoming_total, wire_maximum,
      maximum_incoming));
  LPFX_TRY(try_grow_particle_migration_bound(
      current.maximum_transit(),
      maximum(transit, initial_round ? UINT64_C(0) : outgoing_total),
      wire_maximum, maximum_transit));
  LPFX_TRY(try_grow_particle_migration_bound(
      current.maximum_final(), final_bound, portable_maximum, maximum_final));

  // Retained aggregate bounds can sit between a peer's old and newly padded
  // capacities. Restore the peer <= aggregate invariant without applying
  // headroom a second time.
  maximum_incoming = maximum(maximum_incoming, largest_peer_incoming);
  maximum_outgoing = maximum(maximum_outgoing, initial_peer_growth);
  maximum_transit = maximum(maximum_transit, transit_peer_growth);
  PreparedParticleMigrationCapacity<Contract> candidate{};
  LPFX_TRY(try_prepare_particle_migration_capacity<Contract>(
      {outgoing.data(), current.peer_count()},
      {incoming.data(), current.peer_count()}, maximum_outgoing,
      maximum_incoming, maximum_transit, maximum_final, candidate));
  changed = !(candidate == current);
  output = candidate;
  required_owner = owner_bound;
  return ErrorCode::success;
}

// Admits the classified split of an admitted incoming batch and returns the
// cumulative final prefix. Hard preconditions: capacity.valid() and
// admitted_incoming <= capacity.maximum_incoming().
template <ParticleWireContractType Contract>
[[nodiscard]] LPFX_HOST_DEVICE constexpr ErrorCode
try_admit_particle_received_split(
    const PreparedParticleMigrationCapacity<Contract> &capacity,
    const std::uint64_t admitted_incoming, const std::uint64_t final_count,
    const std::uint64_t transit_count, const std::uint64_t discarded_count,
    const std::uint64_t prior_final_prefix, const std::uint64_t resident_count,
    const std::uint64_t owner_capacity, std::uint64_t &final_prefix) noexcept {
  assert(capacity.valid());
  assert(admitted_incoming <= capacity.maximum_incoming());

  // Explicit unsigned bounds keep this callable from device code.
  if (final_count > UINT64_MAX - transit_count ||
      final_count + transit_count > UINT64_MAX - discarded_count) {
    return ErrorCode::overflow;
  }
  if (final_count + transit_count + discarded_count != admitted_incoming) {
    return ErrorCode::invalid_argument;
  }
  if (prior_final_prefix > UINT64_MAX - final_count ||
      resident_count > UINT64_MAX - (prior_final_prefix + final_count) ||
      transit_count > UINT64_MAX / capacity.record_wire_bytes()) {
    return ErrorCode::overflow;
  }
  const std::uint64_t prefix = prior_final_prefix + final_count;
  if (transit_count > capacity.maximum_transit() ||
      prefix > capacity.maximum_final() ||
      resident_count + prefix > owner_capacity ||
      transit_count * capacity.record_wire_bytes() >
          capacity.maximum_transit_bytes()) {
    return ErrorCode::capacity_exceeded;
  }
  final_prefix = prefix;
  return ErrorCode::success;
}

} // namespace detail

} // namespace lpfx

#endif
