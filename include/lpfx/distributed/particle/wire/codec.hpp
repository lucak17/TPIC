#ifndef LPFX_DISTRIBUTED_PARTICLE_WIRE_CODEC_HPP
#define LPFX_DISTRIBUTED_PARTICLE_WIRE_CODEC_HPP

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/memory_intrinsics.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/data/particle/host/view.hpp>
#include <lpfx/distributed/particle/wire/image.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

namespace detail {

template <ParticleWireEndianness Endianness, class T>
inline constexpr bool particle_wire_native_order =
    sizeof(T) == 1 ||
    (Endianness == ParticleWireEndianness::little &&
     std::endian::native == std::endian::little) ||
    (Endianness == ParticleWireEndianness::big &&
     std::endian::native == std::endian::big);

// Wire lanes are dense byte storage aligned to Alignment only (1 in the packed
// layout, image.hpp), not necessarily to T. A constant-size representation
// copy lets host and GPU compilers select a naturally wide transaction; see
// copy_aligned_representation. The copy assumes what both addresses share.
template <std::size_t Alignment, class T>
inline constexpr std::size_t particle_wire_scalar_alignment =
    Alignment < alignof(T) ? Alignment : alignof(T);

template <ParticleWireEndianness Endianness, std::size_t Alignment = 1,
          LifetimeStartEligible T>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void
store_particle_wire_scalar(std::byte *const destination,
                           const T value) noexcept {
  if constexpr (particle_wire_native_order<Endianness, T>) {
    copy_aligned_representation<sizeof(T),
                                particle_wire_scalar_alignment<Alignment, T>>(
        destination, reinterpret_cast<const std::byte *>(&value));
  } else {
    const auto *const representation =
        reinterpret_cast<const unsigned char *>(&value);
    for (std::size_t byte = 0; byte != sizeof(T); ++byte) {
      destination[byte] =
          static_cast<std::byte>(representation[sizeof(T) - byte - 1]);
    }
  }
}

template <ParticleWireEndianness Endianness, LifetimeStartEligible T,
          std::size_t Alignment = 1>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE T
load_particle_wire_scalar(const std::byte *const source) noexcept {
  T value{};
  if constexpr (particle_wire_native_order<Endianness, T>) {
    copy_aligned_representation<sizeof(T),
                                particle_wire_scalar_alignment<Alignment, T>>(
        reinterpret_cast<std::byte *>(&value), source);
  } else {
    auto *const representation = reinterpret_cast<unsigned char *>(&value);
    for (std::size_t byte = 0; byte != sizeof(T); ++byte) {
      representation[byte] =
          static_cast<unsigned char>(source[sizeof(T) - byte - 1]);
    }
  }
  return value;
}

template <class Codec, class Context, class Wire, class Storage>
concept ParticleWireAttributeCodec =
    requires(const Context &context, const Storage &value, const Wire wire,
             Wire &encoded, Storage &decoded) {
      {
        Codec::try_encode(context, value, encoded)
      } noexcept -> std::same_as<bool>;
      {
        Codec::try_decode(context, wire, decoded)
      } noexcept -> std::same_as<bool>;
    };

template <class Rebuilder, class Context, class Position, class Storage>
concept ParticleWireAttributeRebuilder =
    requires(const Context &context, const Position &position,
             Storage &value) {
      {
        Rebuilder::try_rebuild(context, position, value)
      } noexcept -> std::same_as<bool>;
    };

template <class Contract, class Context, std::size_t Attribute>
inline constexpr bool particle_wire_attribute_codec = [] {
  using Entry = typename Contract::template attribute_type<Attribute>;
  using Storage = typename Entry::storage_type;
  if constexpr (particle_wire_communicated<Entry>) {
    return ParticleWireAttributeCodec<typename Entry::codec_type, Context,
                                      typename Entry::wire_scalar::value_type,
                                      Storage>;
  } else {
    return ParticleWireAttributeRebuilder<
        typename Entry::rebuilder_type, Context,
        ParticleWireCanonicalPosition<Contract>, Storage>;
  }
}();

template <class Contract, class Context, std::size_t... Attributes>
[[nodiscard]] consteval bool
particle_wire_codec_context(std::index_sequence<Attributes...>) noexcept {
  return (particle_wire_attribute_codec<Contract, Context, Attributes> && ...);
}

// Every attribute hook of Contract is a noexcept bool function of Context:
// Codec::try_encode(context, value, wire&) and try_decode(context, wire,
// value&) for communicated attributes, Rebuilder::try_rebuild(context,
// position, value&) for rebuilt ones.
template <class Context, class Contract>
concept ParticleWireCodecContext =
    ParticleWireContractType<Contract> &&
    particle_wire_codec_context<Contract, Context>(
        std::make_index_sequence<Contract::attribute_count>{});

// PositionCodec::try_encode(context, source, index, position&) is a noexcept
// bool function.
template <class Contract, class Context, class Source, class Index>
concept ParticleWirePositionEncoder =
    ParticleWireContractType<Contract> &&
    requires(const Context &context, const Source &source, const Index index,
             ParticleWireCanonicalPosition<Contract> &position) {
      {
        Contract::routing_type::position_codec_type::try_encode(
            context, source, index, position)
      } noexcept -> std::same_as<bool>;
    };

struct StoreParticleWireComponent final {
  template <std::size_t Segment, ParticleWireContractType Contract, class Byte,
            ParticleWireImageCount Count, LifetimeStartEligible Scalar>
    requires(!std::is_const_v<Byte>)
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static void
  apply(const BasicParticleWireImageView<Contract, Byte, Count> image,
        const Count lane, const Scalar value) noexcept {
    static_assert(particle_wire_entry_width<Contract, Segment> ==
                  sizeof(Scalar));
    store_particle_wire_scalar<
        Contract::payload_endianness,
        particle_wire_lane_alignment<Contract, Segment>>(
        image.template lane<Segment>(lane), value);
  }
};

struct LoadParticleWireComponent final {
  template <std::size_t Segment, ParticleWireContractType Contract, class Byte,
            ParticleWireImageCount Count, LifetimeStartEligible Scalar>
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static void
  apply(const BasicParticleWireImageView<Contract, Byte, Count> image,
        const Count lane, Scalar &value) noexcept {
    static_assert(particle_wire_entry_width<Contract, Segment> ==
                  sizeof(Scalar));
    value = load_particle_wire_scalar<
        Contract::payload_endianness, Scalar,
        particle_wire_lane_alignment<Contract, Segment>>(
        image.template lane<Segment>(lane));
  }
};

// The routing entries follow the communicated attributes in entry order
// (image.hpp orders their storage). Operation is a stateless load/store policy
// shared by host and device.
template <class Operation, ParticleWireContractType Contract, class Byte,
          ParticleWireImageCount Count>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void apply_particle_canonical_position(
    const BasicParticleWireImageView<Contract, Byte, Count> image,
    const Count lane,
    ParticleWireCanonicalPosition<Contract> &position) noexcept {
  constexpr std::size_t base = Contract::communicated_attribute_count;
  Operation::template apply<base>(image, lane, position.global_cell_x);
  Operation::template apply<base + 1>(image, lane, position.global_cell_y);
  Operation::template apply<base + 2>(image, lane, position.global_cell_z);
  Operation::template apply<base + 3>(image, lane, position.fraction_x);
  Operation::template apply<base + 4>(image, lane, position.fraction_y);
  Operation::template apply<base + 5>(image, lane, position.fraction_z);
  if constexpr (Contract::routing_type::endpoint_mask_encoding ==
                ParticleEndpointMaskEncoding::unsigned_octet) {
    Operation::template apply<base + 6>(image, lane,
                                        position.positive_endpoint_mask);
  }
}

template <ParticleWireContractType Contract, class Byte,
          ParticleWireImageCount Count>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool
load_particle_canonical_position(
    const BasicParticleWireImageView<Contract, Byte, Count> image,
    const std::type_identity_t<Count> lane,
    ParticleWireCanonicalPosition<Contract> &position) noexcept {
  apply_particle_canonical_position<LoadParticleWireComponent>(image, lane,
                                                               position);
  return valid_particle_positive_endpoint_mask<typename Contract::routing_type>(
      position.positive_endpoint_mask);
}

// Encodes the storage value of communicated attribute Attribute.
template <std::size_t Attribute, ParticleWireContractType Contract, class Byte,
          ParticleWireImageCount Count, class Value, class Context>
  requires(!std::is_const_v<Byte>)
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool
encode_particle_attribute(
    const Value &value, const Context &context,
    const BasicParticleWireImageView<Contract, Byte, Count> image,
    const std::type_identity_t<Count> lane) noexcept {
  using Entry = typename Contract::template attribute_type<Attribute>;
  typename Entry::wire_scalar::value_type encoded{};
  if (!Entry::codec_type::try_encode(context, value, encoded)) {
    return false;
  }
  StoreParticleWireComponent::template apply<
      particle_attribute_segment_ordinal<Contract, Attribute>>(image, lane,
                                                               encoded);
  return true;
}

template <ParticleWireContractType Contract, class Source, class Index,
          class Context, class Byte, ParticleWireImageCount Count>
  requires(!std::is_const_v<Byte>) &&
          ParticleWirePositionEncoder<Contract, Context, Source, Index>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool encode_particle_position(
    const Source &source, const Index index, const Context &context,
    const BasicParticleWireImageView<Contract, Byte, Count> image,
    const std::type_identity_t<Count> lane) noexcept {
  using Routing = typename Contract::routing_type;
  ParticleWireCanonicalPosition<Contract> position{};
  if (!Routing::position_codec_type::try_encode(context, source, index,
                                                position) ||
      !valid_particle_positive_endpoint_mask<Routing>(
          position.positive_endpoint_mask)) {
    return false;
  }
  apply_particle_canonical_position<StoreParticleWireComponent>(image, lane,
                                                                position);
  return true;
}

// Decodes (communicated) or rebuilds attribute Attribute of one record.
template <std::size_t Attribute, ParticleWireContractType Contract, class Byte,
          ParticleWireImageCount Count, class Context>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool decode_particle_attribute(
    const BasicParticleWireImageView<Contract, Byte, Count> image,
    const std::type_identity_t<Count> lane,
    const ParticleWireCanonicalPosition<Contract> &position,
    const Context &context,
    typename Contract::template attribute_type<Attribute>::storage_type
        &value) noexcept {
  using Entry = typename Contract::template attribute_type<Attribute>;
  if constexpr (particle_wire_communicated<Entry>) {
    typename Entry::wire_scalar::value_type wire{};
    LoadParticleWireComponent::template apply<
        particle_attribute_segment_ordinal<Contract, Attribute>>(image, lane,
                                                                 wire);
    return Entry::codec_type::try_decode(context, wire, value);
  } else {
    return Entry::rebuilder_type::try_rebuild(context, position, value);
  }
}

template <ParticleWireContractType Contract, class Source, class Index,
          class Context, class Byte, ParticleWireImageCount Count>
struct EncodeParticleAttribute final {
  const Source &source;
  Index index;
  const Context &context;
  BasicParticleWireImageView<Contract, Byte, Count> image;
  Count lane;

  template <std::size_t Attribute>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE ErrorCode
  operator()(meta::index_constant<Attribute>) const noexcept {
    if constexpr (particle_wire_communicated<
                      typename Contract::template attribute_type<Attribute>>) {
      if (!encode_particle_attribute<Attribute, Contract>(
              source.template get<Attribute>()[index], context, image, lane)) {
        return ErrorCode::invalid_argument;
      }
    }
    return ErrorCode::success;
  }
};

template <ParticleWireContractType Contract, class Byte,
          ParticleWireImageCount Count, class Context, class Sink,
          class SinkIndex>
struct DecodeParticleAttribute final {
  BasicParticleWireImageView<Contract, Byte, Count> image;
  Count lane;
  const ParticleWireCanonicalPosition<Contract> &position;
  const Context &context;
  const Sink &sink;
  SinkIndex sink_index;

  template <std::size_t Attribute>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE ErrorCode
  operator()(meta::index_constant<Attribute>) const noexcept {
    typename Contract::template attribute_type<Attribute>::storage_type
        value{};
    if (!decode_particle_attribute<Attribute, Contract>(image, lane, position,
                                                        context, value)) {
      return ErrorCode::invalid_argument;
    }
    sink.template store<Attribute>(sink_index, value);
    return ErrorCode::success;
  }
};

// Encodes source record index (attributes in schema order, then position).
template <ParticleWireContractType Contract, class Source, class Index,
          class Context, class Byte, ParticleWireImageCount Count>
  requires(!std::is_const_v<Byte>) &&
          ParticleWireCodecContext<Context, Contract> &&
          ParticleWirePositionEncoder<Contract, Context, Source, Index>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool encode_particle_record(
    const Source &source, const Index index, const Context &context,
    const BasicParticleWireImageView<Contract, Byte, Count> image,
    const std::type_identity_t<Count> lane) noexcept {
  return for_each_attribute<typename Contract::schema_type>(
             EncodeParticleAttribute<Contract, Source, Index, Context, Byte,
                                     Count>{source, index, context, image,
                                            lane}) == ErrorCode::success &&
         encode_particle_position<Contract>(source, index, context, image,
                                            lane);
}

// Decodes wire record lane and hands every attribute value, in schema order,
// to sink.template store<Attribute>(sink_index, value).
template <ParticleWireContractType Contract, class Byte,
          ParticleWireImageCount Count, class Context, class Sink,
          class SinkIndex>
  requires ParticleWireCodecContext<Context, Contract>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool decode_particle_record(
    const BasicParticleWireImageView<Contract, Byte, Count> image,
    const std::type_identity_t<Count> lane, const Context &context,
    const Sink &sink, const SinkIndex sink_index) noexcept {
  ParticleWireCanonicalPosition<Contract> position{};
  return load_particle_canonical_position(image, lane, position) &&
         for_each_attribute<typename Contract::schema_type>(
             DecodeParticleAttribute<Contract, Byte, Count, Context, Sink,
                                     SinkIndex>{image, lane, position, context,
                                                sink, sink_index}) ==
             ErrorCode::success;
}

template <ParticleWireContractType Contract, class SourceByte,
          class DestinationByte, class Count, std::size_t... Entry>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void copy_particle_wire_entries(
    const BasicParticleWireImageView<Contract, SourceByte, Count> source,
    const Count source_lane,
    const BasicParticleWireImageView<Contract, DestinationByte, Count>
        destination,
    const Count destination_lane, std::index_sequence<Entry...>) noexcept {
  (copy_aligned_representation<particle_wire_entry_width<Contract, Entry>,
                               particle_wire_lane_alignment<Contract, Entry>>(
       destination.template lane<Entry>(destination_lane),
       source.template lane<Entry>(source_lane)),
   ...);
}

template <ParticleWireContractType Contract, class SourceByte,
          class DestinationByte, ParticleWireImageCount Count>
  requires(!std::is_const_v<DestinationByte>)
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void copy_particle_wire_record(
    const BasicParticleWireImageView<Contract, SourceByte, Count> source,
    const std::type_identity_t<Count> source_lane,
    const BasicParticleWireImageView<Contract, DestinationByte, Count>
        destination,
    const std::type_identity_t<Count> destination_lane) noexcept {
  copy_particle_wire_entries(
      source, source_lane, destination, destination_lane,
      std::make_index_sequence<Contract::wire_entry_count>{});
}

// Attribute-major host pack of one peer image. The view, map and count are
// held by value so the byte stores into the image cannot force reloads of the
// loop bounds, and the named always-inline operator keeps every epoch driver
// from outlining the per-element loop.
template <ParticleWireContractType Contract, bool Const,
          ParticleMapIndex MapIndex, class CodecContext>
struct PackHostParticleWireAttribute final {
  BasicParticleView<typename Contract::schema_type, HostSpace, Const> source;
  const MapIndex *map;
  std::size_t count;
  HostParticleWireImageView<Contract> image;
  const CodecContext &context;

  template <std::size_t Attribute>
  [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
  operator()(meta::index_constant<Attribute>) const noexcept {
    if constexpr (particle_wire_communicated<
                      typename Contract::template attribute_type<Attribute>>) {
      const auto values = source.template get<Attribute>();
      for (std::size_t lane = 0; lane != count; ++lane) {
        if (!encode_particle_attribute<Attribute, Contract>(
                values[static_cast<std::size_t>(map[lane])], context, image,
                lane)) {
          return ErrorCode::invalid_argument;
        }
      }
    }
    return ErrorCode::success;
  }
};

// Kernel functors keep ordinary members: NVCC 12.6 can fail internally when
// non-empty kernel-argument members use [[no_unique_address]].
template <ParticleWireContractType Contract, class SourceAccessor,
          class CodecContext>
  requires std::is_trivially_copyable_v<SourceAccessor> &&
           std::is_trivially_copyable_v<CodecContext> &&
           ParticleWireCodecContext<CodecContext, Contract>
class DeviceInitialParticleWireWriter final {
public:
  constexpr DeviceInitialParticleWireWriter(
      const SourceAccessor source, const CodecContext context,
      const DeviceParticlePeerWireImageView<Contract> destination) noexcept
      : source_(source), context_(context), destination_(destination) {}

  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE bool
  write_peer(const std::uint32_t source, const std::uint8_t peer,
             const std::uint32_t lane) const noexcept {
    return encode_particle_record<Contract>(source_, source, context_,
                                            destination_.packet(peer), lane);
  }

private:
  SourceAccessor source_{};
  CodecContext context_{};
  DeviceParticlePeerWireImageView<Contract> destination_{};
};

template <ParticleWireContractType Contract>
class DeviceTransitParticleWireWriter final {
public:
  LPFX_HOST_DEVICE constexpr DeviceTransitParticleWireWriter(
      const ConstDeviceParticlePeerWireImageView<Contract> source,
      const DeviceParticlePeerWireImageView<Contract> destination) noexcept
      : source_(source), destination_(destination) {}

  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE bool
  write_peer(const std::uint32_t source, const std::uint8_t peer,
             const std::uint32_t lane) const noexcept {
    const auto record = source_.resolve(source);
    copy_particle_wire_record(record.image, record.lane,
                              destination_.packet(peer), lane);
    return true;
  }

private:
  ConstDeviceParticlePeerWireImageView<Contract> source_{};
  DeviceParticlePeerWireImageView<Contract> destination_{};
};

template <ParticleWireContractType Contract, class SourceAccessor,
          class CodecContext>
[[nodiscard]] constexpr auto make_device_initial_particle_wire_writer(
    const SourceAccessor source, const CodecContext context,
    const DeviceParticlePeerWireImageView<Contract> destination) noexcept {
  return DeviceInitialParticleWireWriter<Contract, SourceAccessor,
                                         CodecContext>{source, context,
                                                       destination};
}

} // namespace detail

// Packs source records source_for_wire[lane] into image lanes. The loop is
// attribute-major so each host pass streams one attribute array.
template <detail::ParticleWireContractType Contract, bool Const,
          detail::ParticleMapIndex MapIndex, class CodecContext>
  requires detail::ParticleWireCodecContext<CodecContext, Contract>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode pack_host_particle_wire_image(
    const BasicParticleView<typename Contract::schema_type, HostSpace, Const>
        source,
    const BufferView<MapIndex, HostSpace> source_for_wire,
    const HostParticleWireImageView<Contract> image,
    const CodecContext &context) noexcept {
  if (source_for_wire.size() != image.particle_count()) {
    return ErrorCode::invalid_argument;
  }
  if (image.empty()) {
    return ErrorCode::success;
  }
  LPFX_TRY(for_each_attribute<typename Contract::schema_type>(
      detail::PackHostParticleWireAttribute<Contract, Const, MapIndex,
                                            CodecContext>{
          source, source_for_wire.data(), source_for_wire.size(), image,
          context}));
  for (std::size_t lane = 0; lane != source_for_wire.size(); ++lane) {
    if (!detail::encode_particle_position<Contract>(
            source, static_cast<std::size_t>(source_for_wire[lane]), context,
            image, lane)) {
      return ErrorCode::invalid_argument;
    }
  }
  return ErrorCode::success;
}

} // namespace lpfx

#endif
