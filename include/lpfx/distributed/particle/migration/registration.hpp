#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_REGISTRATION_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_REGISTRATION_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <lpfx/distributed/common/canonical_wire.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/distributed/particle/migration/pattern.hpp>
#include <lpfx/distributed/registration/record.hpp>

namespace lpfx {

struct ParticleMigrationInvariantDescriptorV1Layout final {
  inline static constexpr std::array<std::byte, 8> magic{
      std::byte{'L'}, std::byte{'P'}, std::byte{'F'}, std::byte{'x'},
      std::byte{'P'}, std::byte{'M'}, std::byte{'I'}, std::byte{'1'}};

  static constexpr std::uint16_t version = 1;
  static constexpr std::uint16_t header_extent = 106;
};

struct ParticleMigrationDirectedDescriptorV1Layout final {
  inline static constexpr std::array<std::byte, 8> magic{
      std::byte{'L'}, std::byte{'P'}, std::byte{'F'}, std::byte{'x'},
      std::byte{'P'}, std::byte{'M'}, std::byte{'D'}, std::byte{'1'}};

  static constexpr std::uint16_t version = 1;
  static constexpr std::uint16_t extent = 108;
};

namespace detail {

template <ParticleMigrationPatternType Pattern>
inline constexpr std::size_t particle_migration_invariant_descriptor_extent =
    ParticleMigrationInvariantDescriptorV1Layout::header_extent +
    particle_wire_descriptor_extent<typename Pattern::contract_type>;

template <class Layout, std::size_t Extent>
constexpr void write_particle_migration_descriptor_prefix(
    std::array<std::byte, Extent> &bytes, const std::uint16_t header_extent,
    const std::uint32_t descriptor_extent) noexcept {
  static_assert(Extent >= 16);
  write_canonical_wire_prefix(bytes, Layout::magic);
  const std::array<std::uint16_t, 2> header{Layout::version, header_extent};
  integer_codec<std::endian::big>::write_sequence(bytes.data() + 8, header);
  integer_codec<std::endian::big>::write(bytes.data() + 12, descriptor_extent);
}

template <ParticleMigrationPatternType Pattern>
[[nodiscard]] constexpr auto
make_particle_migration_invariant_descriptor(const Pattern &pattern) noexcept {
  using Contract = typename Pattern::contract_type;
  using Layout = ParticleMigrationInvariantDescriptorV1Layout;
  constexpr std::size_t extent =
      particle_migration_invariant_descriptor_extent<Pattern>;
  static_assert(extent <= std::numeric_limits<std::uint32_t>::max());

  std::array<std::byte, extent> bytes{};
  write_particle_migration_descriptor_prefix<Layout>(
      bytes, Layout::header_extent, static_cast<std::uint32_t>(extent));
  constexpr std::array<std::uint32_t, 2> schema{
      static_cast<std::uint32_t>(particle_wire_descriptor_extent<Contract>),
      static_cast<std::uint32_t>(sizeof(std::uint32_t))};
  integer_codec<std::endian::big>::write_sequence(bytes.data() + 16, schema);

  const auto &routing = pattern.routing();
  const auto shape = routing.process_shape();
  const std::array<std::uint64_t, 10> descriptor_words{
      static_cast<std::uint64_t>(shape.x()),
      static_cast<std::uint64_t>(shape.y()),
      static_cast<std::uint64_t>(shape.z()),
      routing.topology_hop_bound(),
      routing.round_count(),
      Contract::record_wire_width,
      Pattern::boundary_descriptor_identity,
      Pattern::overlap_descriptor_identity,
      Contract::hop_policy::particle_wire_id,
      Contract::hop_policy::consensus_identity};
  integer_codec<std::endian::big>::write_sequence(bytes.data() + 24,
                                                  descriptor_words);

  const auto periodicity = routing.periodicity();
  bytes[104] = std::byte{
      static_cast<unsigned char>((periodicity.x ? UINT8_C(1) : UINT8_C(0)) |
                                 (periodicity.y ? UINT8_C(2) : UINT8_C(0)) |
                                 (periodicity.z ? UINT8_C(4) : UINT8_C(0)))};
  bytes[105] =
      std::byte{static_cast<unsigned char>(Contract::hop_policy::kind)};

  constexpr auto wire = make_canonical_particle_wire_descriptor<Contract>();
  for (std::size_t index = 0; index != wire.size(); ++index) {
    bytes[Layout::header_extent + index] = std::byte{wire[index]};
  }
  return bytes;
}

// One reciprocal contract descriptor for a direct (one-hop) neighbor.
template <CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode
try_encode_particle_migration_directed_descriptor(
    const CartesianExtent3<Rank> shape, const CartesianPeriodicity3 periodicity,
    const CartesianIndex3<Rank> source_coordinate,
    const CartesianIndex3<Rank> destination_coordinate, const Rank source_rank,
    const Rank destination_rank, const std::uint64_t maximum_particles,
    const std::uint64_t record_wire_bytes, const std::uint64_t round_count,
    std::array<std::byte, ParticleMigrationDirectedDescriptorV1Layout::extent>
        &bytes) noexcept {
  std::uint32_t canonical_source_rank{};
  std::uint32_t canonical_destination_rank{};
  std::uint64_t maximum_bytes{};
  CartesianDirection3 direction{};
  std::uint64_t hop_count{};
  if (!checked_integral_cast(source_rank, canonical_source_rank) ||
      !checked_integral_cast(destination_rank, canonical_destination_rank) ||
      !checked_multiply(maximum_particles, record_wire_bytes, maximum_bytes)) {
    return ErrorCode::overflow;
  }
  if (!try_particle_route_direction_and_hops(
          shape, periodicity, source_coordinate, destination_coordinate,
          direction, hop_count) ||
      hop_count != 1) {
    return ErrorCode::invalid_argument;
  }

  using Layout = ParticleMigrationDirectedDescriptorV1Layout;
  bytes = {};
  write_particle_migration_descriptor_prefix<Layout>(
      bytes, Layout::extent, static_cast<std::uint32_t>(Layout::extent));
  const std::array<std::uint32_t, 2> ranks{canonical_source_rank,
                                           canonical_destination_rank};
  integer_codec<std::endian::big>::write_sequence(bytes.data() + 16, ranks);
  const std::array<std::uint64_t, 6> coordinates{
      static_cast<std::uint64_t>(source_coordinate.x()),
      static_cast<std::uint64_t>(source_coordinate.y()),
      static_cast<std::uint64_t>(source_coordinate.z()),
      static_cast<std::uint64_t>(destination_coordinate.x()),
      static_cast<std::uint64_t>(destination_coordinate.y()),
      static_cast<std::uint64_t>(destination_coordinate.z())};
  integer_codec<std::endian::big>::write_sequence(bytes.data() + 24,
                                                  coordinates);
  integer_codec<std::endian::big>::write(
      bytes.data() + 72, UINT32_C(1) << static_cast<std::uint32_t>(
                             cartesian_neighbor_index(direction)));
  const std::array<std::uint64_t, 4> capacities{
      maximum_particles, maximum_bytes, record_wire_bytes, round_count};
  integer_codec<std::endian::big>::write_sequence(bytes.data() + 76,
                                                  capacities);
  return ErrorCode::success;
}

} // namespace detail

template <detail::ParticleMigrationPatternType Pattern>
using ParticleMigrationRegistrationChannel3D = RegistrationChannel<
    detail::particle_migration_invariant_descriptor_extent<Pattern>,
    Pattern::maximum_peer_count,
    ParticleMigrationDirectedDescriptorV1Layout::extent>;

// One contract per peer group; there is one migration message per unique
// peer and protocol stage, so both message groups are 0.
template <detail::ParticleMigrationPatternType Pattern>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_registration_channel(
    const ChannelId channel_id, const Pattern &pattern,
    ParticleMigrationRegistrationChannel3D<Pattern> &output) noexcept {
  using Contract = typename Pattern::contract_type;
  if (!channel_id.valid() || !pattern.valid()) {
    return ErrorCode::invalid_argument;
  }
  ParticleMigrationRegistrationChannel3D<Pattern> prepared{};
  prepared.channel_id = channel_id;
  const auto invariant =
      detail::make_particle_migration_invariant_descriptor(pattern);
  if (!prepared.invariant_descriptor.try_assign(invariant)) {
    return ErrorCode::capacity_exceeded;
  }

  const auto &routing = pattern.routing();
  const auto shape = routing.process_shape();
  const auto local = routing.process_coordinate();
  for (std::uint8_t peer_group = 0; peer_group != routing.peer_count();
       ++peer_group) {
    const auto peer_rank = pattern.peer_rank(peer_group);
    CartesianIndex3<typename Pattern::rank_type> peer{};
    if (!try_canonical_rank_to_coordinate(shape, peer_rank, peer)) {
      return ErrorCode::invalid_argument;
    }
    std::uint32_t canonical_peer{};
    if (!detail::checked_integral_cast(peer_rank, canonical_peer)) {
      return ErrorCode::overflow;
    }
    std::array<std::byte, ParticleMigrationDirectedDescriptorV1Layout::extent>
        outbound{}, inbound{};
    LPFX_TRY(detail::try_encode_particle_migration_directed_descriptor(
        shape, routing.periodicity(), local, peer, pattern.process_rank(),
        peer_rank, pattern.capacity().maximum_peer_outgoing(peer_group),
        Contract::record_wire_width, routing.round_count(), outbound));
    LPFX_TRY(detail::try_encode_particle_migration_directed_descriptor(
        shape, routing.periodicity(), peer, local, peer_rank,
        pattern.process_rank(),
        pattern.capacity().maximum_peer_incoming(peer_group),
        Contract::record_wire_width, routing.round_count(), inbound));
    LPFX_TRY(prepared.try_append_contract(canonical_peer, 0, 0, outbound,
                                          inbound));
  }
  output = prepared;
  return ErrorCode::success;
}

} // namespace lpfx

#endif
