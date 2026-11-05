#ifndef LPFX_DISTRIBUTED_FIELD_REGISTRATION_HPP
#define LPFX_DISTRIBUTED_FIELD_REGISTRATION_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/common/canonical_wire.hpp>
#include <lpfx/distributed/descriptor/field.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/distributed/field/pattern.hpp>
#include <lpfx/distributed/registration/record.hpp>

namespace lpfx::detail {

// Directed descriptor of one direction and flow: a 32-byte header, then one
// record per piece. Affine records hold the two boxes; fused records also
// hold the component, role, directions, samples, message offset and bytes.
template <bool Fused> struct FieldDirectedDescriptorLayout final {
  static constexpr std::array<std::byte, 8> magic{
      std::byte{'L'},
      std::byte{'P'},
      std::byte{'F'},
      std::byte{'x'},
      std::byte{'F'},
      Fused ? std::byte{'S'} : std::byte{'C'},
      Fused ? std::byte{'C'} : std::byte{'D'},
      Fused ? std::byte{'P'} : std::byte{0}};
  static constexpr std::uint16_t version = 2;
  static constexpr std::uint16_t header_extent = 32;
  static constexpr std::uint16_t component_extent = Fused ? 136 : 96;

  [[nodiscard]] static constexpr std::size_t
  descriptor_extent(const std::size_t record_count) noexcept {
    return header_extent + record_count * component_extent;
  }
};

template <class Spec> struct FieldRegistrationTraits final {
  using layout_type = FieldDirectedDescriptorLayout<Spec::fused>;
  static constexpr std::size_t maximum_record_count =
      Spec::fused ? fused_field_maximum_pieces_per_component *
                        Spec::component_count
                  : Spec::component_count;
  static constexpr std::size_t contract_capacity =
      cartesian_neighbor_mask_count(Spec::transport_mask);
  static constexpr std::size_t pool_capacity =
      2 * (contract_capacity * layout_type::header_extent +
           Spec::maximum_active_piece_count * layout_type::component_extent);
  using channel_type = RegistrationChannel<
      FieldOperationDescriptorLayout<Spec>::extent, contract_capacity,
      layout_type::descriptor_extent(maximum_record_count), pool_capacity>;
};

template <class Coordinate>
constexpr void encode_field_registration_box(const CartesianBox3<Coordinate> box,
                                             std::byte *const bytes) noexcept {
  using Codec = integer_codec<std::endian::big>;
  const auto lower = box.lower();
  const auto upper = box.upper();
  const std::array<Coordinate, 6> values{lower.x(), lower.y(), lower.z(),
                                         upper.x(), upper.y(), upper.z()};
  for (std::size_t index = 0; index != values.size(); ++index) {
    Codec::write(bytes + index * sizeof(std::uint64_t),
                 static_cast<std::uint64_t>(
                     static_cast<std::int64_t>(values[index])));
  }
}

template <FieldFlow Flow, class Pattern>
constexpr void
encode_field_direction(const typename Pattern::direction_type &direction,
                       const std::span<std::byte> bytes) noexcept {
  using Spec = typename Pattern::spec_type;
  using Layout = FieldDirectedDescriptorLayout<Spec::fused>;
  using Codec = integer_codec<std::endian::big>;
  const auto pieces = direction.template pieces<Flow>();
  static_cast<void>(
      write_canonical_descriptor_prefix<Layout>(bytes, Layout::component_extent));
  Codec::write(bytes.data() + 16, static_cast<std::uint32_t>(bytes.size()));
  Codec::write(bytes.data() + 20, static_cast<std::uint16_t>(pieces.size()));
  Codec::write(bytes.data() + 24, static_cast<std::uint64_t>(
                                      direction.template message_bytes<Flow>()));
  std::size_t offset = 0;
  for (std::size_t record = 0; record != pieces.size(); ++record) {
    const auto &piece = pieces[record];
    std::byte *const base = bytes.data() + Layout::descriptor_extent(record);
    const std::size_t piece_bytes = Pattern::piece_bytes(piece);
    if constexpr (Spec::fused) {
      Codec::write(base, piece.component);
      base[2] = static_cast<std::byte>(piece.role);
      const std::array<std::int8_t, 6> directions{
          piece.publication.x, piece.publication.y, piece.publication.z,
          piece.shared.x,      piece.shared.y,      piece.shared.z};
      for (std::size_t axis = 0; axis != directions.size(); ++axis) {
        base[3 + axis] = static_cast<std::byte>(
            static_cast<std::uint8_t>(directions[axis]));
      }
      Codec::write(base + 16, static_cast<std::uint64_t>(piece.samples));
      Codec::write(base + 24, static_cast<std::uint64_t>(offset));
      Codec::write(base + 32, static_cast<std::uint64_t>(piece_bytes));
      encode_field_registration_box(piece.source, base + 40);
      encode_field_registration_box(piece.destination, base + 88);
    } else {
      encode_field_registration_box(piece.source, base);
      encode_field_registration_box(piece.destination, base + 48);
    }
    offset += piece_bytes;
  }
}

// Invariant descriptor and one reciprocal contract per peer direction. The
// caller sets the channel id.
template <class Pattern, class Channel>
[[nodiscard]] constexpr ErrorCode
try_prepare_field_registration_channel(const Pattern &pattern,
                                       Channel &channel) noexcept {
  using Layout = FieldDirectedDescriptorLayout<Pattern::spec_type::fused>;
  constexpr auto invariant =
      make_canonical_field_operation_descriptor<typename Pattern::spec_type>();
  if (!pattern.is_valid() || !channel.invariant_descriptor.try_assign(invariant)) {
    return ErrorCode::invalid_argument;
  }
  for (std::size_t index = 0; index != Pattern::direction_count; ++index) {
    const auto direction = pattern.direction(index);
    if (!direction.has_peer()) {
      continue;
    }
    std::uint32_t peer{};
    if (!checked_integral_cast(direction.peer_rank(), peer)) {
      return ErrorCode::invalid_argument;
    }
    LPFX_TRY(channel.try_append_contract(
        peer, direction.local_receive_message_group(),
        direction.remote_receive_message_group(),
        Layout::descriptor_extent(
            direction.template pieces<FieldFlow::outbound>().size()),
        Layout::descriptor_extent(
            direction.template pieces<FieldFlow::inbound>().size()),
        [&](const std::span<std::byte> bytes) constexpr noexcept {
          encode_field_direction<FieldFlow::outbound, Pattern>(direction, bytes);
          return ErrorCode::success;
        },
        [&](const std::span<std::byte> bytes) constexpr noexcept {
          encode_field_direction<FieldFlow::inbound, Pattern>(direction, bytes);
          return ErrorCode::success;
        }));
  }
  return ErrorCode::success;
}

} // namespace lpfx::detail

#endif
