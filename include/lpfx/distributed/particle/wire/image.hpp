#ifndef LPFX_DISTRIBUTED_PARTICLE_WIRE_IMAGE_HPP
#define LPFX_DISTRIBUTED_PARTICLE_WIRE_IMAGE_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/fixed_array.hpp>
#include <lpfx/distributed/particle/limits.hpp>
#include <lpfx/distributed/particle/wire/contract.hpp>

namespace lpfx {

namespace detail {

// Storage order of the segments of one packet. Packed: entry order. Aligned:
// descending natural width alignment, ties in entry order, so every width
// stored before a segment is a multiple of its alignment.
template <ParticleWireContractType Contract, std::size_t Other,
          std::size_t Segment>
inline constexpr bool particle_wire_segment_precedes = [] {
  if constexpr (Contract::wire_layout == ParticleWireLayout::aligned) {
    constexpr std::uint64_t other =
        particle_wire_natural_alignment(Contract::wire_widths[Other]);
    constexpr std::uint64_t segment =
        particle_wire_natural_alignment(Contract::wire_widths[Segment]);
    return other > segment || (other == segment && Other < Segment);
  } else {
    return Other < Segment;
  }
}();

template <ParticleWireContractType Contract, std::size_t Segment>
  requires(Segment < Contract::wire_entry_count)
inline constexpr std::uint64_t particle_wire_segment_prefix_width =
    []<std::size_t... Other>(std::index_sequence<Other...>) {
      return (std::uint64_t{0} + ... +
              (particle_wire_segment_precedes<Contract, Other, Segment>
                   ? std::uint64_t{Contract::wire_widths[Other]}
                   : std::uint64_t{0}));
    }(std::make_index_sequence<Contract::wire_entry_count>{});

// Alignment of every lane of Segment in an image whose base is aligned to
// Contract::record_wire_alignment (wire storage slots are 64-byte aligned):
// packets start at multiples of it, the segment count times its prefix width
// into the packet, and its lanes are a width apart. The packed layout
// promises none.
template <ParticleWireContractType Contract, std::size_t Segment>
  requires(Segment < Contract::wire_entry_count)
inline constexpr std::size_t particle_wire_lane_alignment =
    Contract::wire_layout == ParticleWireLayout::aligned
        ? static_cast<std::size_t>(
              particle_wire_natural_alignment(Contract::wire_widths[Segment]))
        : std::size_t{1};

// The packets of an image are placed in peer order: each nonempty one starts
// at the end of the preceding nonempty one, rounded up to the packet
// alignment. Packets so keep their record bytes, never share a byte, and only
// the local image holds the gaps. A start cannot be derived from the peer's
// record offset alone: rounding that offset moves a packet without moving its
// successor, which would overlap them.
//
// Returns the gap bytes before the packet of peer, so the packet starts at
// particle_offsets[peer] * record_wire_width plus this padding. An empty peer
// reports where its packet would start. Hard precondition: peer does not
// exceed the peer count of the peer-major particle_offsets.
template <ParticleWireContractType Contract, class Offsets>
[[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
particle_wire_packet_padding(const Offsets &particle_offsets,
                             const std::uint8_t peer) noexcept {
  if constexpr (Contract::wire_packet_gap == 0) {
    return 0;
  } else {
    constexpr std::uint64_t mask = Contract::record_wire_alignment - 1;
    std::uint64_t padding = 0;
    for (std::uint8_t packet = 0; packet <= peer; ++packet) {
      if (packet != peer &&
          particle_offsets[packet + 1] == particle_offsets[packet]) {
        continue;
      }
      // The preceding packets end at their record bytes plus their gaps.
      const std::uint64_t end =
          static_cast<std::uint64_t>(particle_offsets[packet]) *
              Contract::record_wire_width +
          padding;
      padding += ((end + mask) & ~mask) - end;
    }
    return padding;
  }
}

// Largest extent of an image of particles records: the gaps between packets
// add at most the contract's slack. The checked form fails on overflow.
template <ParticleWireContractType Contract>
[[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
particle_wire_extent_bytes(const std::uint64_t particles) noexcept {
  if constexpr (Contract::wire_extent_slack == 0) {
    return particles * Contract::record_wire_width;
  } else {
    return particles == 0 ? 0
                          : particles * Contract::record_wire_width +
                                Contract::wire_extent_slack;
  }
}

template <ParticleWireContractType Contract, class Size>
[[nodiscard]] constexpr bool
try_particle_wire_extent_bytes(const Size particles, Size &bytes) noexcept {
  if (!checked_multiply(particles,
                        static_cast<Size>(Contract::record_wire_width),
                        bytes)) {
    return false;
  }
  if constexpr (Contract::wire_extent_slack != 0) {
    return particles == 0 ||
           checked_add(bytes, static_cast<Size>(Contract::wire_extent_slack),
                       bytes);
  } else {
    return true;
  }
}

template <class T>
concept ParticleWireImageCount =
    std::unsigned_integral<T> && (!std::same_as<T, bool>);

struct ParticlePeerPrefixAdmission;

} // namespace detail

enum class ParticlePeerPrefixKind : std::uint8_t {
  outgoing = 0,
  incoming = 1,
};

// Admitted peer-major particle offsets of one direction; the terminal offset
// at peer_count() is the total. Host admission and every device kernel
// argument derive from this one uint64 prefix. Only capacity admission
// (migration/capacity.hpp) publishes a nonempty value.
template <detail::ParticleWireContractType Contract,
          ParticlePeerPrefixKind Kind>
class PreparedParticlePeerPrefixes final {
public:
  static constexpr std::size_t maximum_peer_count =
      particle_migration_maximum_peer_count;

  static constexpr ParticlePeerPrefixKind kind = Kind;
  using contract_type = Contract;

  constexpr PreparedParticlePeerPrefixes() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  peer_count() const noexcept {
    return peer_count_;
  }

  // Hard precondition: peer <= peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  particle_offset(const std::uint8_t peer) const noexcept {
    return particle_offsets_[peer];
  }

  // Hard precondition: peer <= peer_count(). A peer packet spans
  // particle_count(peer) * Contract::record_wire_width bytes from here; the
  // next nonempty packet starts at or after its end. Linear in peer when the
  // contract has packet gaps.
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  byte_offset(const std::uint8_t peer) const noexcept {
    return particle_offsets_[peer] * Contract::record_wire_width +
           detail::particle_wire_packet_padding<Contract>(particle_offsets_,
                                                          peer);
  }

  // Hard precondition: peer < peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  particle_count(const std::uint8_t peer) const noexcept {
    return particle_offsets_[peer + 1] - particle_offsets_[peer];
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  total_particles() const noexcept {
    return particle_offsets_[peer_count_];
  }

  // The image extent, which covers every packet (and in the aligned layout
  // up to wire_extent_slack bytes past the last one).
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  total_bytes() const noexcept {
    return detail::particle_wire_extent_bytes<Contract>(total_particles());
  }

private:
  detail::DeviceFixedArray<std::uint64_t, maximum_peer_count + 1>
      particle_offsets_{};
  std::uint8_t peer_count_{};

  friend struct detail::ParticlePeerPrefixAdmission;
};

template <detail::ParticleWireContractType Contract>
using PreparedParticleOutgoingPrefixes =
    PreparedParticlePeerPrefixes<Contract, ParticlePeerPrefixKind::outgoing>;
template <detail::ParticleWireContractType Contract>
using PreparedParticleIncomingPrefixes =
    PreparedParticlePeerPrefixes<Contract, ParticlePeerPrefixKind::incoming>;

// One peer's contiguous packet stores each communicated component as an
// array. GPU lanes therefore access adjacent component values while MPI sends
// the complete packet in one call. A packet spans count * record_wire_width
// bytes and starts at a multiple of record_wire_alignment.
template <detail::ParticleWireContractType Contract, class Byte,
          detail::ParticleWireImageCount Count>
  requires(std::same_as<std::remove_const_t<Byte>, std::byte>)
class BasicParticleWireImageView final {
public:
  using contract_type = Contract;
  using pointer = Byte *;
  using count_type = Count;

  LPFX_HOST_DEVICE constexpr BasicParticleWireImageView() noexcept {}

  LPFX_HOST_DEVICE constexpr BasicParticleWireImageView(
      pointer const base, const Count particle_count) noexcept
      : base_(base), particle_count_(particle_count) {}

  template <class OtherByte>
    requires std::is_const_v<Byte> && (!std::is_const_v<OtherByte>)
  LPFX_HOST_DEVICE constexpr BasicParticleWireImageView(
      const BasicParticleWireImageView<Contract, OtherByte, Count>
          other) noexcept
      : base_(other.base()), particle_count_(other.particle_count()) {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr pointer base() const noexcept {
    return base_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Count
  particle_count() const noexcept {
    return particle_count_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool empty() const noexcept {
    return particle_count_ == 0;
  }

  // Hard precondition: particle < particle_count().
  template <std::size_t Segment>
    requires(Segment < Contract::wire_entry_count)
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr pointer
  lane(const Count particle) const noexcept {
    constexpr std::size_t width =
        detail::particle_wire_entry_width<Contract, Segment>;
    constexpr std::uint64_t prefix_width =
        detail::particle_wire_segment_prefix_width<Contract, Segment>;
    static_assert(
        prefix_width % detail::particle_wire_lane_alignment<Contract, Segment> ==
            0 &&
        Contract::record_wire_alignment %
                detail::particle_wire_lane_alignment<Contract, Segment> ==
            0);
    return base_ +
           static_cast<std::size_t>(particle_count_) *
               static_cast<std::size_t>(prefix_width) +
           static_cast<std::size_t>(particle) * width;
  }

private:
  pointer base_ = nullptr;
  Count particle_count_{};
};

template <detail::ParticleWireContractType Contract>
using DeviceParticleWireImageView =
    BasicParticleWireImageView<Contract, std::byte, std::uint32_t>;

template <detail::ParticleWireContractType Contract>
using ConstDeviceParticleWireImageView =
    BasicParticleWireImageView<Contract, const std::byte, std::uint32_t>;

template <detail::ParticleWireContractType Contract>
using HostParticleWireImageView =
    BasicParticleWireImageView<Contract, std::byte, std::size_t>;

template <detail::ParticleWireContractType Contract>
using ConstHostParticleWireImageView =
    BasicParticleWireImageView<Contract, const std::byte, std::size_t>;

// A peer-major sequence of contiguous SoA packets addressed by admitted peer
// prefixes. The offsets and the gap bytes before each packet are copied into
// the view, so it is a complete kernel argument and locates a packet without
// revisiting the preceding ones. Resolve a global record index once and reuse
// its packet view for all component loads or stores; known destination peers
// need no lookup.
template <detail::ParticleWireContractType Contract, class Byte,
          detail::ParticleWireImageCount Count>
  requires(std::same_as<std::remove_const_t<Byte>, std::byte>)
class BasicParticlePeerWireImageView final {
public:
  using contract_type = Contract;
  using pointer = Byte *;
  using count_type = Count;
  using image_type = BasicParticleWireImageView<Contract, Byte, Count>;
  // The accumulated gaps of an image fit one octet per peer.
  // cmp_less_equal: nvcc warns #186-D on `0 <= max` when the slack is 0.
  static_assert(std::cmp_less_equal(Contract::wire_extent_slack,
                                    std::numeric_limits<std::uint8_t>::max()));

  struct RecordLocation final {
    image_type image;
    Count lane;
  };

  LPFX_HOST_DEVICE constexpr BasicParticlePeerWireImageView() noexcept {}

  // Admitted capacities are uint32-portable, so every offset fits Count.
  template <ParticlePeerPrefixKind Kind>
  LPFX_HOST_DEVICE constexpr BasicParticlePeerWireImageView(
      pointer const base,
      const PreparedParticlePeerPrefixes<Contract, Kind> &prefixes) noexcept
      : base_(base), peer_count_(prefixes.peer_count()) {
    for (std::uint8_t peer = 0; peer <= peer_count_; ++peer) {
      offsets_[peer] = static_cast<Count>(prefixes.particle_offset(peer));
    }
    if constexpr (Contract::wire_packet_gap != 0) {
      for (std::uint8_t peer = 0; peer != peer_count_; ++peer) {
        paddings_[peer] = static_cast<std::uint8_t>(
            detail::particle_wire_packet_padding<Contract>(offsets_, peer));
      }
    }
  }

  template <class OtherByte>
    requires std::is_const_v<Byte> && (!std::is_const_v<OtherByte>)
  LPFX_HOST_DEVICE constexpr BasicParticlePeerWireImageView(
      const BasicParticlePeerWireImageView<Contract, OtherByte, Count> &other)
      noexcept
      : base_(other.base()), peer_count_(other.peer_count()) {
    for (std::uint8_t peer = 0; peer <= peer_count_; ++peer) {
      offsets_[peer] = other.particle_offset(peer);
    }
    for (std::uint8_t peer = 0; peer != peer_count_; ++peer) {
      paddings_[peer] = other.packet_padding(peer);
    }
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr pointer base() const noexcept {
    return base_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  peer_count() const noexcept {
    return peer_count_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr Count
  particle_count() const noexcept {
    return offsets_[peer_count_];
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool empty() const noexcept {
    return particle_count() == 0;
  }
  // Hard precondition: peer <= peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr Count
  particle_offset(const std::uint8_t peer) const noexcept {
    return offsets_[peer];
  }

  // Gap bytes of the image before the packet of peer; 0 without packet gaps.
  // Hard precondition: peer < peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  packet_padding(const std::uint8_t peer) const noexcept {
    return paddings_[peer];
  }

  // Hard precondition: peer < peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr image_type
  packet(const std::uint8_t peer) const noexcept {
    const Count count = offsets_[peer + 1] - offsets_[peer];
    std::size_t offset = static_cast<std::size_t>(offsets_[peer]) *
                         static_cast<std::size_t>(Contract::record_wire_width);
    if constexpr (Contract::wire_packet_gap != 0) {
      offset += paddings_[peer];
    }
    return {count == 0 ? nullptr : base_ + offset, count};
  }

  // Hard precondition: particle < particle_count(). Upper-bound lookup skips
  // empty peer intervals, including repeated leading/intermediate prefixes.
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr RecordLocation
  resolve(const Count particle) const noexcept {
    std::uint8_t first = 0;
    std::uint8_t last = peer_count_;
    while (last - first > 1) {
      const auto middle = static_cast<std::uint8_t>(first + (last - first) / 2);
      if (particle < offsets_[middle]) {
        last = middle;
      } else {
        first = middle;
      }
    }
    return {packet(first), static_cast<Count>(particle - offsets_[first])};
  }

private:
  pointer base_ = nullptr;
  detail::DeviceFixedArray<Count, particle_migration_maximum_peer_count + 1>
      offsets_{};
  detail::DeviceFixedArray<std::uint8_t, particle_migration_maximum_peer_count>
      paddings_{};
  std::uint8_t peer_count_{};
};

template <detail::ParticleWireContractType Contract>
using DeviceParticlePeerWireImageView =
    BasicParticlePeerWireImageView<Contract, std::byte, std::uint32_t>;
template <detail::ParticleWireContractType Contract>
using ConstDeviceParticlePeerWireImageView =
    BasicParticlePeerWireImageView<Contract, const std::byte, std::uint32_t>;
template <detail::ParticleWireContractType Contract>
using HostParticlePeerWireImageView =
    BasicParticlePeerWireImageView<Contract, std::byte, std::size_t>;
template <detail::ParticleWireContractType Contract>
using ConstHostParticlePeerWireImageView =
    BasicParticlePeerWireImageView<Contract, const std::byte, std::size_t>;

} // namespace lpfx

#endif
