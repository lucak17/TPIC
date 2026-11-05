#ifndef LPFX_DISTRIBUTED_PARTICLE_WIRE_CONTRACT_HPP
#define LPFX_DISTRIBUTED_PARTICLE_WIRE_CONTRACT_HPP

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/data/particle/schema.hpp>
#include <lpfx/distributed/common/canonical_wire.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/distributed/particle/limits.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

enum class ParticleWireEndianness : std::uint8_t {
  little = 1,
  big = 2,
};

enum class ParticleWireTreatment : std::uint8_t {
  communicate = 1,
  rebuild_locally = 2,
};

enum class ParticleEndpointMaskEncoding : std::uint8_t {
  invariant_zero_omitted = 1,
  unsigned_octet = 2,
};

enum class ParticleHopPolicyKind : std::uint8_t {
  fixed = 1,
  adaptive = 2,
};

// Byte layout of the particle wire images of a contract; part of its canonical
// descriptor, so every rank of a migration must use the same layout. Records
// are the summed entry widths and peer messages the same bytes either way.
enum class ParticleWireLayout : std::uint8_t {
  // Segments in entry order, packets back to back. Host codecs copy bytes
  // whatever their alignment, so host routes use this.
  packed = 1,
  // Segments by descending natural alignment and packets starting at a
  // multiple of the largest one in the local image, so device codecs issue
  // one typed access per lane. Unsent gaps of at most 7 bytes separate the
  // packets of an image: each starts at the rounded-up end of the preceding
  // one (image.hpp).
  aligned = 2,
};

template <detail::checked_integer GlobalIndex, std::floating_point Fraction>
struct ParticleCanonicalPosition3D final {
  GlobalIndex global_cell_x{};
  GlobalIndex global_cell_y{};
  GlobalIndex global_cell_z{};
  Fraction fraction_x{};
  Fraction fraction_y{};
  Fraction fraction_z{};
  std::uint8_t positive_endpoint_mask{};
};

template <std::uint64_t Id>
  requires(Id != 0)
struct ParticleWireIdentity final {
  static constexpr std::uint64_t particle_wire_id = Id;
};

template <LifetimeStartEligible T, std::uint64_t Id>
  requires(Id != 0) && (sizeof(T) <= std::numeric_limits<std::uint32_t>::max())
struct ParticleWireScalar final {
  using value_type = T;

  static constexpr std::uint64_t particle_wire_id = Id;
  static constexpr std::uint32_t width = static_cast<std::uint32_t>(sizeof(T));
};

namespace detail {

inline constexpr std::uint64_t particle_wire_not_applicable_id =
    UINT64_C(0x4c5046784e2f4102);
inline constexpr std::uint64_t particle_wire_not_applicable_bound =
    std::numeric_limits<std::uint64_t>::max();

template <class T>
concept ParticleWireIdentityType = requires {
  typename std::integral_constant<std::uint64_t, static_cast<std::uint64_t>(
                                                     T::particle_wire_id)>;
  requires static_cast<std::uint64_t>(T::particle_wire_id) != 0;
};

template <class T> inline constexpr bool particle_wire_scalar = false;

template <class T, std::uint64_t Id>
inline constexpr bool particle_wire_scalar<ParticleWireScalar<T, Id>> = true;

template <class T>
concept ParticleWireScalarType = particle_wire_scalar<T>;

template <class T> struct IsRebuildLocally : std::false_type {};

template <class RebuilderTag>
struct IsRebuildLocally<RebuildLocally<RebuilderTag>> : std::true_type {
  using rebuilder_type = RebuilderTag;
};

} // namespace detail

template <detail::ParticleAttribute Attribute, std::uint64_t TagIdentity,
          std::uint64_t StorageScalarIdentity,
          detail::ParticleWireScalarType WireScalar,
          detail::ParticleWireIdentityType Codec>
  requires(TagIdentity != 0 && StorageScalarIdentity != 0) &&
          std::same_as<typename Attribute::wire_policy, Communicate> &&
          (sizeof(typename Attribute::value_type) <=
           std::numeric_limits<std::uint32_t>::max())
struct CommunicatedParticleWireAttribute final {
  using attribute_type = Attribute;
  using tag_type = typename Attribute::name_type;
  using storage_type = typename Attribute::value_type;
  using wire_scalar = WireScalar;
  using codec_type = Codec;

  static constexpr ParticleWireTreatment treatment =
      ParticleWireTreatment::communicate;
  static constexpr std::uint64_t tag_identity = TagIdentity;
  static constexpr std::uint64_t storage_scalar_identity =
      StorageScalarIdentity;
  static constexpr std::uint64_t wire_scalar_identity =
      WireScalar::particle_wire_id;
  static constexpr std::uint64_t codec_identity = Codec::particle_wire_id;
  static constexpr std::uint64_t rebuilder_identity =
      detail::particle_wire_not_applicable_id;
  static constexpr std::uint32_t wire_width = WireScalar::width;
  static constexpr std::uint32_t storage_width =
      static_cast<std::uint32_t>(sizeof(storage_type));
};

template <detail::ParticleAttribute Attribute, std::uint64_t TagIdentity,
          std::uint64_t StorageScalarIdentity>
  requires(TagIdentity != 0 && StorageScalarIdentity != 0) &&
          detail::IsRebuildLocally<typename Attribute::wire_policy>::value &&
          detail::ParticleWireIdentityType<typename detail::IsRebuildLocally<
              typename Attribute::wire_policy>::rebuilder_type> &&
          (sizeof(typename Attribute::value_type) <=
           std::numeric_limits<std::uint32_t>::max())
struct RebuiltParticleWireAttribute final {
  using attribute_type = Attribute;
  using tag_type = typename Attribute::name_type;
  using storage_type = typename Attribute::value_type;
  using rebuilder_type = typename detail::IsRebuildLocally<
      typename Attribute::wire_policy>::rebuilder_type;

  static constexpr ParticleWireTreatment treatment =
      ParticleWireTreatment::rebuild_locally;
  static constexpr std::uint64_t tag_identity = TagIdentity;
  static constexpr std::uint64_t storage_scalar_identity =
      StorageScalarIdentity;
  static constexpr std::uint64_t wire_scalar_identity =
      detail::particle_wire_not_applicable_id;
  static constexpr std::uint64_t codec_identity =
      detail::particle_wire_not_applicable_id;
  static constexpr std::uint64_t rebuilder_identity =
      rebuilder_type::particle_wire_id;
  static constexpr std::uint32_t wire_width = 0;
  static constexpr std::uint32_t storage_width =
      static_cast<std::uint32_t>(sizeof(storage_type));
};

namespace detail {

template <class T> inline constexpr bool particle_wire_attribute = false;

template <class A, std::uint64_t Tag, std::uint64_t Storage, class Wire,
          class Codec>
inline constexpr bool particle_wire_attribute<
    CommunicatedParticleWireAttribute<A, Tag, Storage, Wire, Codec>> = true;

template <class A, std::uint64_t Tag, std::uint64_t Storage>
inline constexpr bool
    particle_wire_attribute<RebuiltParticleWireAttribute<A, Tag, Storage>> =
        true;

template <class T>
concept ParticleWireAttributeContractType = particle_wire_attribute<T>;

template <class...> inline constexpr bool unique_particle_wire_tag_ids = true;

template <class First, class... Rest>
inline constexpr bool unique_particle_wire_tag_ids<First, Rest...> =
    ((First::tag_identity != Rest::tag_identity) && ...) &&
    unique_particle_wire_tag_ids<Rest...>;

template <class Schema, class... Entries, std::size_t... Indices>
[[nodiscard]] consteval bool
particle_wire_entries_match_schema(std::index_sequence<Indices...>) noexcept {
  return (std::same_as<typename Entries::attribute_type,
                       typename Schema::template attribute<Indices>> &&
          ...);
}

template <class Entry>
inline constexpr bool particle_wire_communicated =
    Entry::treatment == ParticleWireTreatment::communicate;

template <class... Entries>
inline constexpr std::size_t communicated_particle_attribute_count =
    (std::size_t{0} + ... +
     (particle_wire_communicated<Entries> ? std::size_t{1} : std::size_t{0}));

} // namespace detail

template <detail::ParticleWireScalarType DomainIndexWireScalar,
          detail::ParticleWireScalarType FractionWireScalar,
          detail::ParticleWireIdentityType PositionCodec,
          detail::ParticleWireIdentityType EndpointCodec,
          ParticleEndpointMaskEncoding EndpointMaskEncoding>
  requires std::integral<typename DomainIndexWireScalar::value_type> &&
           (!std::same_as<typename DomainIndexWireScalar::value_type, bool>) &&
           (EndpointMaskEncoding ==
                ParticleEndpointMaskEncoding::invariant_zero_omitted ||
            EndpointMaskEncoding ==
                ParticleEndpointMaskEncoding::unsigned_octet)
struct CanonicalParticleRouting final {
  using domain_index_wire_scalar = DomainIndexWireScalar;
  using fraction_wire_scalar = FractionWireScalar;
  using position_codec_type = PositionCodec;
  using endpoint_codec_type = EndpointCodec;

  static constexpr ParticleEndpointMaskEncoding endpoint_mask_encoding =
      EndpointMaskEncoding;
  static constexpr std::uint32_t endpoint_mask_width =
      EndpointMaskEncoding == ParticleEndpointMaskEncoding::unsigned_octet ? 1U
                                                                           : 0U;
  static constexpr std::size_t wire_entry_count =
      EndpointMaskEncoding == ParticleEndpointMaskEncoding::unsigned_octet
          ? std::size_t{7}
          : std::size_t{6};
};

namespace detail {

template <class T> inline constexpr bool canonical_particle_routing = false;

template <class Index, class Fraction, class Position, class Endpoint,
          ParticleEndpointMaskEncoding Encoding>
inline constexpr bool canonical_particle_routing<
    CanonicalParticleRouting<Index, Fraction, Position, Endpoint, Encoding>> =
    true;

template <class T>
concept CanonicalParticleRoutingType = canonical_particle_routing<T>;

} // namespace detail

template <detail::CanonicalParticleRoutingType Routing>
[[nodiscard]] LPFX_HOST_DEVICE constexpr bool
valid_particle_positive_endpoint_mask(const std::uint8_t mask) noexcept {
  if constexpr (Routing::endpoint_mask_encoding ==
                ParticleEndpointMaskEncoding::unsigned_octet) {
    return (mask & UINT8_C(0xf8)) == 0;
  } else {
    return mask == 0;
  }
}

template <std::uint64_t PolicyIdentity, std::uint64_t FixedHopBound>
  requires(PolicyIdentity != 0) &&
          (FixedHopBound != detail::particle_wire_not_applicable_bound)
struct FixedParticleHopPolicy final {
  static constexpr ParticleHopPolicyKind kind = ParticleHopPolicyKind::fixed;
  static constexpr std::uint64_t particle_wire_id = PolicyIdentity;
  static constexpr std::uint64_t consensus_identity =
      detail::particle_wire_not_applicable_id;
  static constexpr std::uint64_t fixed_hop_bound = FixedHopBound;
};

template <std::uint64_t PolicyIdentity,
          detail::ParticleWireIdentityType Consensus>
  requires(PolicyIdentity != 0)
struct AdaptiveParticleHopPolicy final {
  using consensus_type = Consensus;

  static constexpr ParticleHopPolicyKind kind = ParticleHopPolicyKind::adaptive;
  static constexpr std::uint64_t particle_wire_id = PolicyIdentity;
  static constexpr std::uint64_t consensus_identity =
      Consensus::particle_wire_id;
  static constexpr std::uint64_t fixed_hop_bound =
      detail::particle_wire_not_applicable_bound;
};

namespace detail {

template <class T> inline constexpr bool particle_hop_policy = false;

template <std::uint64_t Identity, std::uint64_t Bound>
inline constexpr bool
    particle_hop_policy<FixedParticleHopPolicy<Identity, Bound>> = true;

template <std::uint64_t Identity, class Consensus>
inline constexpr bool
    particle_hop_policy<AdaptiveParticleHopPolicy<Identity, Consensus>> = true;

template <class T>
concept ParticleHopPolicyType = particle_hop_policy<T>;

template <class Routing, class... Entries>
inline constexpr std::size_t particle_wire_entry_count =
    communicated_particle_attribute_count<Entries...> +
    Routing::wire_entry_count;

// Entry order: communicated attributes in schema order, then 3 domain indices,
// 3 fractions and the optional endpoint mask. The aligned layout stores the
// segments in another order (image.hpp).
template <class Routing, class... Entries>
[[nodiscard]] consteval auto make_particle_wire_widths() noexcept {
  std::array<std::uint32_t, particle_wire_entry_count<Routing, Entries...>>
      widths{};
  std::size_t ordinal = 0;
  ((particle_wire_communicated<Entries>
        ? static_cast<void>(widths[ordinal++] = Entries::wire_width)
        : static_cast<void>(0)),
   ...);
  for (std::size_t axis = 0; axis != 3; ++axis) {
    widths[ordinal++] = Routing::domain_index_wire_scalar::width;
  }
  for (std::size_t axis = 0; axis != 3; ++axis) {
    widths[ordinal++] = Routing::fraction_wire_scalar::width;
  }
  if constexpr (Routing::endpoint_mask_encoding ==
                ParticleEndpointMaskEncoding::unsigned_octet) {
    widths[ordinal] = 1;
  }
  return widths;
}

template <std::size_t Extent>
[[nodiscard]] consteval std::uint64_t particle_wire_width_sum(
    const std::array<std::uint32_t, Extent> &widths) noexcept {
  std::uint64_t total = 0;
  for (const std::uint32_t width : widths) {
    total += width;
  }
  return total;
}

// Widest typed access a device codec issues for one aligned wire lane.
inline constexpr std::uint64_t particle_wire_maximum_alignment = 8;

// Largest power of two dividing value, capped at
// particle_wire_maximum_alignment (which zero also yields).
[[nodiscard]] constexpr std::uint64_t
particle_wire_natural_alignment(const std::uint64_t value) noexcept {
  const std::uint64_t lowest = value & (~value + 1U);
  return lowest == 0 || lowest > particle_wire_maximum_alignment
             ? particle_wire_maximum_alignment
             : lowest;
}

template <ParticleWireLayout Layout, std::size_t Extent>
[[nodiscard]] consteval std::uint64_t particle_wire_record_alignment(
    const std::array<std::uint32_t, Extent> &widths) noexcept {
  std::uint64_t alignment = 1;
  if constexpr (Layout == ParticleWireLayout::aligned) {
    for (const std::uint32_t width : widths) {
      const std::uint64_t natural = particle_wire_natural_alignment(width);
      alignment = natural > alignment ? natural : alignment;
    }
  }
  return alignment;
}

// Wire segment of each attribute; UINT32_MAX for rebuilt attributes.
template <class... Entries>
[[nodiscard]] consteval auto make_particle_attribute_segments() noexcept {
  std::array<std::uint32_t, sizeof...(Entries)> segments{};
  std::uint32_t ordinal = 0;
  std::size_t attribute = 0;
  ((segments[attribute++] = particle_wire_communicated<Entries>
                                ? ordinal++
                                : std::numeric_limits<std::uint32_t>::max()),
   ...);
  return segments;
}

} // namespace detail

template <ParticleWireLayout Layout, detail::ParticleSchemaType Schema,
          ParticleWireEndianness PayloadEndianness,
          detail::ParticleHopPolicyType HopPolicy,
          detail::CanonicalParticleRoutingType Routing,
          detail::ParticleWireAttributeContractType... Attributes>
  requires(Layout == ParticleWireLayout::packed ||
           Layout == ParticleWireLayout::aligned) &&
          (sizeof...(Attributes) == Schema::attribute_count) &&
          (sizeof...(Attributes) <= std::numeric_limits<std::uint16_t>::max()) &&
          (PayloadEndianness == ParticleWireEndianness::little ||
           PayloadEndianness == ParticleWireEndianness::big) &&
          (detail::particle_wire_entry_count<Routing, Attributes...> <=
           std::numeric_limits<std::uint16_t>::max()) &&
          detail::unique_particle_wire_tag_ids<Attributes...> &&
          (detail::particle_wire_entries_match_schema<Schema, Attributes...>(
              std::make_index_sequence<sizeof...(Attributes)>{}))
struct BasicParticleWireContract final {
  using schema_type = Schema;
  using hop_policy = HopPolicy;
  using routing_type = Routing;
  using attribute_tuple = std::tuple<Attributes...>;

  static constexpr ParticleWireLayout wire_layout = Layout;
  static constexpr ParticleWireEndianness payload_endianness =
      PayloadEndianness;
  static constexpr std::size_t attribute_count = sizeof...(Attributes);
  static constexpr std::size_t communicated_attribute_count =
      detail::communicated_particle_attribute_count<Attributes...>;
  static constexpr std::size_t wire_entry_count =
      detail::particle_wire_entry_count<Routing, Attributes...>;

  template <std::size_t Index>
  using attribute_type = std::tuple_element_t<Index, attribute_tuple>;

  inline static constexpr std::array<std::uint32_t, wire_entry_count>
      wire_widths = detail::make_particle_wire_widths<Routing, Attributes...>();
  inline static constexpr std::array<std::uint32_t, attribute_count>
      attribute_segments = detail::make_particle_attribute_segments<Attributes...>();
  static constexpr std::uint64_t record_wire_width =
      detail::particle_wire_width_sum(wire_widths);
  // Alignment of image bases and packet starts (1 when packed).
  static constexpr std::uint64_t record_wire_alignment =
      detail::particle_wire_record_alignment<Layout>(wire_widths);
  // Largest unsent gap before a packet: each one starts at the end of the
  // preceding packet rounded up to that alignment (image.hpp), and every
  // packet end is a multiple of the divisor the record width shares with it.
  // 0 when the width is a multiple of the alignment, as always when packed:
  // packets are then back to back.
  static constexpr std::uint64_t wire_packet_gap =
      record_wire_alignment - std::gcd(record_wire_width, record_wire_alignment);
  // An image of n > 0 records spans at most n * record_wire_width +
  // wire_extent_slack bytes: one gap before every packet but the first.
  static constexpr std::uint64_t wire_extent_slack =
      (particle_migration_maximum_peer_count - 1) * wire_packet_gap;
};

// The packed contract keeps the byte layout of wire descriptor version 2.
template <class Schema, ParticleWireEndianness PayloadEndianness,
          class HopPolicy, class Routing, class... Attributes>
using ParticleWireContract =
    BasicParticleWireContract<ParticleWireLayout::packed, Schema,
                              PayloadEndianness, HopPolicy, Routing,
                              Attributes...>;

// For contracts whose images device codecs produce and consume.
template <class Schema, ParticleWireEndianness PayloadEndianness,
          class HopPolicy, class Routing, class... Attributes>
using AlignedParticleWireContract =
    BasicParticleWireContract<ParticleWireLayout::aligned, Schema,
                              PayloadEndianness, HopPolicy, Routing,
                              Attributes...>;

namespace detail {

template <class T> inline constexpr bool particle_wire_contract = false;

template <ParticleWireLayout Layout, class Schema,
          ParticleWireEndianness Endianness, class HopPolicy, class Routing,
          class... Attributes>
inline constexpr bool particle_wire_contract<BasicParticleWireContract<
    Layout, Schema, Endianness, HopPolicy, Routing, Attributes...>> = true;

template <class T>
concept ParticleWireContractType = particle_wire_contract<T>;

// Hard precondition: attribute Attribute is communicated.
template <ParticleWireContractType Contract, std::size_t Attribute>
inline constexpr std::size_t particle_attribute_segment_ordinal =
    Contract::attribute_segments[Attribute];

template <ParticleWireContractType Contract, std::size_t Entry>
  requires(Entry < Contract::wire_entry_count)
inline constexpr std::size_t particle_wire_entry_width =
    Contract::wire_widths[Entry];

template <ParticleWireContractType Contract>
using ParticleWireCanonicalPosition = ParticleCanonicalPosition3D<
    typename Contract::routing_type::domain_index_wire_scalar::value_type,
    typename Contract::routing_type::fraction_wire_scalar::value_type>;

} // namespace detail

// Version 2 identifies peer-major SoA packets and one payload message per peer
// in the packed layout.
struct ParticleWireDescriptorV2Layout final {
  inline static constexpr std::array<std::uint8_t, 8> magic{
      0x4c, 0x50, 0x46, 0x78, 0x50, 0x57, 0x43, 0x00};

  static constexpr std::uint16_t version = 2;
  static constexpr std::uint16_t header_extent = 103;
  static constexpr std::uint16_t attribute_extent = 57;
};

// Version 3 is version 2 in the aligned layout, so packed and aligned ranks
// fail registration against each other.
struct ParticleWireDescriptorV3Layout final {
  inline static constexpr std::array<std::uint8_t, 8> magic =
      ParticleWireDescriptorV2Layout::magic;

  static constexpr std::uint16_t version = 3;
  static constexpr std::uint16_t header_extent =
      ParticleWireDescriptorV2Layout::header_extent;
  static constexpr std::uint16_t attribute_extent =
      ParticleWireDescriptorV2Layout::attribute_extent;
};

namespace detail {

template <ParticleWireContractType Contract>
using ParticleWireDescriptorLayout =
    std::conditional_t<Contract::wire_layout == ParticleWireLayout::aligned,
                       ParticleWireDescriptorV3Layout,
                       ParticleWireDescriptorV2Layout>;

template <ParticleWireContractType Contract>
inline constexpr std::size_t particle_wire_descriptor_extent =
    canonical_wire_descriptor_extent(
        ParticleWireDescriptorLayout<Contract>::header_extent,
        ParticleWireDescriptorLayout<Contract>::attribute_extent,
        static_cast<std::uint16_t>(Contract::attribute_count));

template <ParticleWireContractType Contract, std::size_t Index,
          std::size_t Extent>
constexpr void encode_particle_wire_attribute(
    std::array<std::uint8_t, Extent> &bytes) noexcept {
  using Entry = typename Contract::template attribute_type<Index>;
  constexpr std::size_t base =
      ParticleWireDescriptorLayout<Contract>::header_extent +
      Index * ParticleWireDescriptorLayout<Contract>::attribute_extent;
  constexpr std::array<std::uint64_t, 5> identities{
      Entry::tag_identity, Entry::storage_scalar_identity,
      Entry::wire_scalar_identity, Entry::codec_identity,
      Entry::rebuilder_identity};
  constexpr std::array<std::uint32_t, 4> ordinals{
      Entry::wire_width, Entry::storage_width,
      static_cast<std::uint32_t>(Index), Contract::attribute_segments[Index]};
  integer_codec<std::endian::big>::write_sequence(bytes.data() + base,
                                                  identities);
  integer_codec<std::endian::big>::write_sequence(bytes.data() + base + 40,
                                                  ordinals);
  bytes[base + 56] = static_cast<std::uint8_t>(Entry::treatment);
}

} // namespace detail

template <detail::ParticleWireContractType Contract>
using CanonicalParticleWireDescriptor =
    std::array<std::uint8_t, detail::particle_wire_descriptor_extent<Contract>>;

template <detail::ParticleWireContractType Contract>
[[nodiscard]] constexpr CanonicalParticleWireDescriptor<Contract>
make_canonical_particle_wire_descriptor() noexcept {
  static_assert(detail::particle_wire_descriptor_extent<Contract> <=
                std::numeric_limits<std::uint32_t>::max());

  CanonicalParticleWireDescriptor<Contract> bytes{};
  using Layout = detail::ParticleWireDescriptorLayout<Contract>;
  detail::write_canonical_descriptor_prefix<Layout>(bytes,
                                                    Layout::attribute_extent);
  constexpr std::array<std::uint16_t, 3> descriptor_counts{
      static_cast<std::uint16_t>(Contract::attribute_count),
      static_cast<std::uint16_t>(Contract::wire_entry_count),
      static_cast<std::uint16_t>(Contract::communicated_attribute_count)};
  detail::integer_codec<std::endian::big>::write_sequence(bytes.data() + 14,
                                                          descriptor_counts);
  detail::integer_codec<std::endian::big>::write(
      bytes.data() + 20,
      static_cast<std::uint32_t>(
          detail::particle_wire_descriptor_extent<Contract>));
  bytes[24] = static_cast<std::uint8_t>(Contract::payload_endianness);
  bytes[25] =
      static_cast<std::uint8_t>(Contract::routing_type::endpoint_mask_encoding);
  bytes[26] = static_cast<std::uint8_t>(Contract::hop_policy::kind);
  constexpr std::array<std::uint32_t, 3> routing_widths{
      Contract::routing_type::domain_index_wire_scalar::width,
      Contract::routing_type::fraction_wire_scalar::width,
      Contract::routing_type::endpoint_mask_width};
  constexpr std::array<std::uint64_t, 8> descriptor_words{
      Contract::record_wire_width,
      Contract::routing_type::domain_index_wire_scalar::particle_wire_id,
      Contract::routing_type::fraction_wire_scalar::particle_wire_id,
      Contract::routing_type::position_codec_type::particle_wire_id,
      Contract::routing_type::endpoint_codec_type::particle_wire_id,
      Contract::hop_policy::particle_wire_id,
      Contract::hop_policy::consensus_identity,
      Contract::hop_policy::fixed_hop_bound};
  detail::integer_codec<std::endian::big>::write_sequence(bytes.data() + 27,
                                                          routing_widths);
  detail::integer_codec<std::endian::big>::write_sequence(bytes.data() + 39,
                                                          descriptor_words);
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (detail::encode_particle_wire_attribute<Contract, Index>(bytes), ...);
  }(std::make_index_sequence<Contract::attribute_count>{});
  return bytes;
}

} // namespace lpfx

#endif
