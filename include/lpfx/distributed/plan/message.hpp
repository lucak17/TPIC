#ifndef LPFX_DISTRIBUTED_PLAN_MESSAGE_HPP
#define LPFX_DISTRIBUTED_PLAN_MESSAGE_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/common/canonical_wire.hpp>
#include <lpfx/distributed/descriptor/message.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/distributed/detail/static_vector.hpp>
#include <lpfx/distributed/registration/record.hpp>
#include <lpfx/domain/cartesian/index_policy.hpp>

namespace lpfx::distributed {

struct ByteRange final {
  std::size_t offset_bytes = 0;
  std::size_t size_bytes = 0;

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size_bytes == 0;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const ByteRange &, const ByteRange &) noexcept = default;
};

template <lpfx::detail::CartesianRank Rank>
struct FixedMessageFragment final {
  Rank peer{};
  std::uint32_t local_receive_group = 0;
  std::uint32_t remote_receive_group = 0;
  ByteRange outgoing{};
  ByteRange incoming{};

  [[nodiscard]] friend constexpr bool operator==(
      const FixedMessageFragment &,
      const FixedMessageFragment &) noexcept = default;
};

template <lpfx::detail::CartesianRank Rank>
struct PreparedPeerMessage final {
  Rank peer{};
  ByteRange outgoing{};
  ByteRange incoming{};

  [[nodiscard]] friend constexpr bool operator==(
      const PreparedPeerMessage &,
      const PreparedPeerMessage &) noexcept = default;
};

template <std::uint64_t ChannelIdentity,
          detail::FixedByteMessagesDescriptor Descriptor>
  requires(ChannelIdentity != 0)
struct MessagePlan final {
  using descriptor_type = Descriptor;

  static constexpr std::uint64_t channel_identity = ChannelIdentity;
};

template <class Definition, class Rank> class PreparedMessagePlan;

namespace detail {

struct FixedMessageInvariantDescriptorLayout final {
  inline static constexpr std::array<std::byte, 8> magic{
      std::byte{'L'}, std::byte{'P'}, std::byte{'F'}, std::byte{'x'},
      std::byte{'M'}, std::byte{'S'}, std::byte{'G'}, std::byte{'P'}};
  static constexpr std::uint16_t version = 1;
  static constexpr std::uint16_t header_extent =
      fixed_message_invariant_descriptor_extent;
  static constexpr std::uint16_t component_extent = 0;
};

struct FixedMessageDirectedDescriptorLayout final {
  inline static constexpr std::array<std::byte, 8> magic{
      std::byte{'L'}, std::byte{'P'}, std::byte{'F'}, std::byte{'x'},
      std::byte{'M'}, std::byte{'S'}, std::byte{'G'}, std::byte{'F'}};
  static constexpr std::uint16_t version = 1;
  static constexpr std::uint16_t header_extent =
      fixed_message_directed_header_extent;
  static constexpr std::uint16_t component_extent =
      fixed_message_fragment_record_extent;
  static constexpr std::size_t descriptor_extent_offset = 16;
  static constexpr std::size_t fragment_count_offset = 20;
  static constexpr std::size_t total_message_bytes_offset = 24;
};

struct FixedMessageHalfProjection final {
  std::uint32_t group = 0;
  ByteRange range{};
};

// Sorts the first count values and rejects equal neighbours.
template <class T, std::size_t Capacity, class Less, class Equal>
[[nodiscard]] constexpr bool sort_unique(std::array<T, Capacity> &values,
                                         const std::size_t count,
                                         const Less less,
                                         const Equal equal) noexcept {
  // Cold and bounded by the descriptor capacity. An insertion sort keeps GCC's
  // array-bounds analysis away from std::sort's 16-element unrolled paths on
  // small fixed arrays; equal neighbours are rejected, so order is canonical.
  for (std::size_t index = 1; index < count && index < Capacity; ++index) {
    const T value = values[index];
    std::size_t position = index;
    while (position != 0 && less(value, values[position - 1])) {
      values[position] = values[position - 1];
      --position;
    }
    values[position] = value;
  }
  for (std::size_t index = 1; index < count && index < Capacity; ++index) {
    if (equal(values[index - 1], values[index])) {
      return false;
    }
  }
  return true;
}

inline constexpr auto fixed_message_group_less =
    [](const FixedMessageHalfProjection left,
       const FixedMessageHalfProjection right) noexcept {
      return left.group < right.group;
    };
inline constexpr auto fixed_message_group_equal =
    [](const FixedMessageHalfProjection left,
       const FixedMessageHalfProjection right) noexcept {
      return left.group == right.group;
    };

template <FixedByteMessagesDescriptor Descriptor, std::size_t Capacity>
[[nodiscard]] constexpr ErrorCode try_encode_fixed_message_invariant(
    CanonicalRegistrationBytes<Capacity> &output) noexcept {
  using Layout = FixedMessageInvariantDescriptorLayout;
  if constexpr (Capacity < Layout::header_extent) {
    return ErrorCode::capacity_exceeded;
  } else {
    std::array<std::byte, Layout::header_extent> bytes{};
    lpfx::detail::write_canonical_descriptor_prefix<Layout>(bytes, 0);
    using Codec = lpfx::detail::integer_codec<std::endian::big>;
    Codec::write(bytes.data() + 16,
                 static_cast<std::uint32_t>(Layout::header_extent));
    Codec::write(bytes.data() + 20,
                 static_cast<std::uint16_t>(operation_family::message));
    Codec::write(bytes.data() + 24, Descriptor::protocol_identity);
    return output.try_assign(bytes) ? ErrorCode::success
                                    : ErrorCode::capacity_exceeded;
  }
}

template <std::size_t ProjectionCapacity>
[[nodiscard]] constexpr ErrorCode try_encode_fixed_message_direction(
    const std::array<FixedMessageHalfProjection, ProjectionCapacity>
        &projections,
    const std::size_t projection_count, const std::size_t message_bytes,
    const std::span<std::byte> bytes) noexcept {
  using Layout = FixedMessageDirectedDescriptorLayout;
  if (projection_count == 0) {
    return message_bytes == 0 ? ErrorCode::success
                              : ErrorCode::invalid_argument;
  }
  std::size_t record_bytes{};
  std::size_t descriptor_extent{};
  if (!lpfx::detail::checked_multiply(
          projection_count, fixed_message_fragment_record_extent,
          record_bytes) ||
      !lpfx::detail::checked_add(fixed_message_directed_header_extent,
                                 record_bytes, descriptor_extent)) {
    return ErrorCode::overflow;
  }
  if (projection_count > ProjectionCapacity ||
      descriptor_extent != bytes.size()) {
    return ErrorCode::capacity_exceeded;
  }
  std::uint16_t canonical_count{};
  std::uint32_t canonical_extent{};
  std::uint64_t canonical_total{};
  if (!lpfx::detail::checked_integral_cast(projection_count,
                                           canonical_count) ||
      !lpfx::detail::checked_integral_cast(descriptor_extent,
                                           canonical_extent) ||
      !lpfx::detail::checked_integral_cast(message_bytes, canonical_total)) {
    return ErrorCode::overflow;
  }

  if (!lpfx::detail::write_canonical_descriptor_prefix<Layout>(
          bytes, Layout::component_extent)) {
    return ErrorCode::capacity_exceeded;
  }
  using Codec = lpfx::detail::integer_codec<std::endian::big>;
  Codec::write(bytes.data() + Layout::descriptor_extent_offset,
               canonical_extent);
  Codec::write(bytes.data() + Layout::fragment_count_offset, canonical_count);
  Codec::write(bytes.data() + Layout::total_message_bytes_offset,
               canonical_total);

  std::uint64_t covered = 0;
  for (std::size_t index = 0; index != projection_count; ++index) {
    std::uint64_t fragment_bytes{};
    if (!lpfx::detail::checked_integral_cast(
            projections[index].range.size_bytes, fragment_bytes) ||
        !lpfx::detail::checked_add(covered, fragment_bytes, covered)) {
      return ErrorCode::overflow;
    }
    if (fragment_bytes == 0) {
      return ErrorCode::invalid_argument;
    }
    const std::size_t base =
        Layout::header_extent + index * Layout::component_extent;
    Codec::write(bytes.data() + base, projections[index].group);
    Codec::write(bytes.data() + base + 8, fragment_bytes);
  }
  if (covered != canonical_total) {
    return ErrorCode::invalid_argument;
  }
  return ErrorCode::success;
}

[[nodiscard]] constexpr ErrorCode validate_fixed_message_range(
    const ByteRange range, const std::size_t image_bytes) noexcept {
  if (range.empty()) {
    return ErrorCode::success;
  }
  std::size_t end{};
  if (!lpfx::detail::checked_add(range.offset_bytes, range.size_bytes, end)) {
    return ErrorCode::overflow;
  }
  return end <= image_bytes ? ErrorCode::success
                            : ErrorCode::invalid_argument;
}

template <std::size_t Capacity>
[[nodiscard]] constexpr ErrorCode coalesce_fixed_message_half(
    std::array<FixedMessageHalfProjection, Capacity> &projections,
    const std::size_t projection_count, ByteRange &output) noexcept {
  output = {};
  if (projection_count > Capacity) {
    return ErrorCode::capacity_exceeded;
  }
  if (projection_count == 0) {
    return ErrorCode::success;
  }
  if (!sort_unique(projections, projection_count, fixed_message_group_less,
                   fixed_message_group_equal)) {
    return ErrorCode::invalid_argument;
  }
  const auto first = projections[0].range;
  std::size_t end{};
  if (!lpfx::detail::checked_add(first.offset_bytes, first.size_bytes, end)) {
    return ErrorCode::overflow;
  }
  for (std::size_t index = 1;
       index < projection_count && index < Capacity; ++index) {
    const auto range = projections[index].range;
    if (range.offset_bytes != end ||
        !lpfx::detail::checked_add(range.offset_bytes, range.size_bytes,
                                   end)) {
      return ErrorCode::invalid_argument;
    }
  }
  output = {first.offset_bytes, end - first.offset_bytes};
  return ErrorCode::success;
}

} // namespace detail

template <std::uint64_t ChannelIdentity,
          detail::FixedByteMessagesDescriptor Descriptor,
          lpfx::detail::CartesianRank Rank>
class PreparedMessagePlan<MessagePlan<ChannelIdentity, Descriptor>, Rank>
    final {
public:
  using definition_type = MessagePlan<ChannelIdentity, Descriptor>;
  using descriptor_type = Descriptor;
  using rank_type = Rank;
  using fragment_type = FixedMessageFragment<Rank>;
  using peer_message_type = PreparedPeerMessage<Rank>;
  using channel_type = RegistrationChannel<
      detail::fixed_message_invariant_descriptor_extent,
      Descriptor::maximum_peer_count,
      detail::fixed_message_reciprocal_descriptor_capacity<Descriptor>,
      2 * (detail::fixed_message_directed_header_extent *
               Descriptor::maximum_peer_count +
           detail::fixed_message_fragment_record_extent *
               Descriptor::maximum_fragment_count)>;
  using peer_messages_type =
      lpfx::detail::static_vector<peer_message_type,
                                  Descriptor::maximum_peer_count>;

  static constexpr std::uint64_t channel_identity = ChannelIdentity;
  static constexpr std::size_t maximum_fragment_count =
      Descriptor::maximum_fragment_count;
  static constexpr std::size_t maximum_peer_count =
      Descriptor::maximum_peer_count;

  constexpr PreparedMessagePlan() noexcept = default;

  // Only try_prepare_message_plan produces a nonzero rank count.
  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return rank_count_ != 0;
  }

  [[nodiscard]] constexpr Rank rank() const noexcept { return rank_; }
  [[nodiscard]] constexpr Rank rank_count() const noexcept {
    return rank_count_;
  }
  [[nodiscard]] constexpr std::size_t outgoing_bytes() const noexcept {
    return outgoing_bytes_;
  }
  [[nodiscard]] constexpr std::size_t incoming_bytes() const noexcept {
    return incoming_bytes_;
  }
  [[nodiscard]] constexpr std::span<const peer_message_type>
  peer_messages() const noexcept {
    return peer_messages_.view();
  }
  [[nodiscard]] constexpr const channel_type &
  registration_channel() const noexcept {
    return channel_;
  }

private:
  constexpr PreparedMessagePlan(const peer_messages_type &peer_messages,
                                const channel_type &channel, const Rank rank,
                                const Rank rank_count,
                                const std::size_t outgoing_bytes,
                                const std::size_t incoming_bytes) noexcept
      : peer_messages_(peer_messages), channel_(channel), rank_(rank),
        rank_count_(rank_count), outgoing_bytes_(outgoing_bytes),
        incoming_bytes_(incoming_bytes) {}

  peer_messages_type peer_messages_{};
  channel_type channel_{};
  Rank rank_{};
  Rank rank_count_{};
  std::size_t outgoing_bytes_ = 0;
  std::size_t incoming_bytes_ = 0;

  friend struct lpfx::detail::PrivateConstructionAccess;
};

template <std::uint64_t ChannelIdentity,
          detail::FixedByteMessagesDescriptor Descriptor,
          lpfx::detail::CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode try_prepare_message_plan(
    const MessagePlan<ChannelIdentity, Descriptor>, const Rank rank,
    const Rank rank_count, const std::size_t outgoing_bytes,
    const std::size_t incoming_bytes,
    const std::span<const FixedMessageFragment<Rank>> fragments,
    PreparedMessagePlan<MessagePlan<ChannelIdentity, Descriptor>, Rank>
        &output) noexcept {
  using Prepared =
      PreparedMessagePlan<MessagePlan<ChannelIdentity, Descriptor>, Rank>;
  using Fragment = FixedMessageFragment<Rank>;
  constexpr std::size_t Capacity = Descriptor::maximum_fragment_count;

  if (rank_count == 0 || rank >= rank_count) {
    return ErrorCode::invalid_argument;
  }
  // Every peer is below rank_count, so this also bounds the native peers.
  std::uint32_t canonical_rank_count{};
  if (!lpfx::detail::checked_integral_cast(rank_count, canonical_rank_count)) {
    return ErrorCode::overflow;
  }
  if (fragments.size() > Capacity) {
    return ErrorCode::capacity_exceeded;
  }
  if (!fragments.empty() && fragments.data() == nullptr) {
    return ErrorCode::invalid_argument;
  }

  // Canonical fragments are preparation scratch. Only peer envelopes and
  // registration bytes survive preparation.
  std::array<Fragment, Capacity> canonical{};
  std::array<detail::FixedMessageHalfProjection, Capacity> projections{};
  std::size_t incoming_count = 0;
  for (std::size_t index = 0; index != fragments.size(); ++index) {
    Fragment &fragment = canonical[index];
    fragment = fragments[index];
    if (fragment.peer >= rank_count || fragment.peer == rank) {
      return ErrorCode::invalid_argument;
    }
    if (fragment.outgoing.empty()) {
      fragment.outgoing = {};
    }
    if (fragment.incoming.empty()) {
      fragment.incoming = {};
    }
    if (fragment.outgoing.empty() && fragment.incoming.empty()) {
      return ErrorCode::invalid_argument;
    }
    LPFX_TRY(detail::validate_fixed_message_range(fragment.outgoing,
                                                  outgoing_bytes));
    LPFX_TRY(detail::validate_fixed_message_range(fragment.incoming,
                                                  incoming_bytes));
    if (!fragment.incoming.empty()) {
      projections[incoming_count++] = {0, fragment.incoming};
    }
  }

  // Incoming ranges are unique and pairwise disjoint.
  if (!detail::sort_unique(
          projections, incoming_count,
          [](const detail::FixedMessageHalfProjection left,
             const detail::FixedMessageHalfProjection right) noexcept {
            return std::tie(left.range.offset_bytes, left.range.size_bytes) <
                   std::tie(right.range.offset_bytes, right.range.size_bytes);
          },
          [](const detail::FixedMessageHalfProjection left,
             const detail::FixedMessageHalfProjection right) noexcept {
            return left.range == right.range;
          })) {
    return ErrorCode::invalid_argument;
  }
  // incoming_count <= Capacity; the second bound only informs the optimizer.
  for (std::size_t index = 1; index < incoming_count && index < Capacity;
       ++index) {
    std::size_t previous_end{};
    if (!lpfx::detail::checked_add(projections[index - 1].range.offset_bytes,
                                   projections[index - 1].range.size_bytes,
                                   previous_end)) {
      return ErrorCode::overflow;
    }
    if (previous_end > projections[index].range.offset_bytes) {
      return ErrorCode::invalid_argument;
    }
  }

  const auto fragment_key = [](const Fragment &fragment) noexcept {
    return std::tuple{fragment.peer,
                      fragment.local_receive_group,
                      fragment.remote_receive_group,
                      fragment.outgoing.offset_bytes,
                      fragment.outgoing.size_bytes,
                      fragment.incoming.offset_bytes,
                      fragment.incoming.size_bytes};
  };
  if (!detail::sort_unique(
          canonical, fragments.size(),
          [&](const Fragment &left, const Fragment &right) noexcept {
            return fragment_key(left) < fragment_key(right);
          },
          [](const Fragment &left, const Fragment &right) noexcept {
            return left == right;
          })) {
    return ErrorCode::invalid_argument;
  }

  typename Prepared::channel_type channel{};
  LPFX_TRY(try_make_channel_id(ChannelIdentity, channel.channel_id));
  LPFX_TRY(detail::try_encode_fixed_message_invariant<Descriptor>(
      channel.invariant_descriptor));

  typename Prepared::peer_messages_type peer_messages{};
  std::size_t first = 0;
  while (first != fragments.size()) {
    const Rank peer = canonical[first].peer;
    std::size_t last = first + 1;
    while (last != fragments.size() && canonical[last].peer == peer) {
      ++last;
    }
    typename Prepared::peer_message_type peer_message{};
    peer_message.peer = peer;

    const auto prepare_half = [&]<bool Outgoing>(
                                  ByteRange &peer_range,
                                  const std::span<std::byte>
                                      descriptor) constexpr noexcept {
      std::size_t count = 0;
      for (std::size_t index = first; index != last; ++index) {
        const Fragment &fragment = canonical[index];
        const ByteRange range = Outgoing ? fragment.outgoing : fragment.incoming;
        if (!range.empty()) {
          projections[count++] = {Outgoing ? fragment.remote_receive_group
                                           : fragment.local_receive_group,
                                  range};
        }
      }
      LPFX_TRY(detail::coalesce_fixed_message_half(projections, count,
                                                   peer_range));
      return detail::try_encode_fixed_message_direction(
          projections, count, peer_range.size_bytes, descriptor);
    };

    std::size_t outgoing_count = 0;
    std::size_t incoming_fragments = 0;
    for (std::size_t index = first; index != last; ++index) {
      outgoing_count += !canonical[index].outgoing.empty();
      incoming_fragments += !canonical[index].incoming.empty();
    }
    const auto descriptor_extent =
        [](const std::size_t count) constexpr noexcept {
          return count == 0
                     ? std::size_t{0}
                     : detail::fixed_message_directed_header_extent +
                           count * detail::fixed_message_fragment_record_extent;
        };
    LPFX_TRY(channel.try_append_contract(
        static_cast<std::uint32_t>(peer), 0, 0,
        descriptor_extent(outgoing_count),
        descriptor_extent(incoming_fragments),
        [&](const std::span<std::byte> bytes) constexpr noexcept {
          return prepare_half.template operator()<true>(peer_message.outgoing,
                                                        bytes);
        },
        [&](const std::span<std::byte> bytes) constexpr noexcept {
          return prepare_half.template operator()<false>(peer_message.incoming,
                                                         bytes);
        }));
    if (!peer_messages.try_append(peer_message)) {
      return ErrorCode::capacity_exceeded;
    }
    first = last;
  }

  output = lpfx::detail::PrivateConstructionAccess::create<Prepared>(
      peer_messages, channel, rank, rank_count, outgoing_bytes,
      incoming_bytes);
  return ErrorCode::success;
}

} // namespace lpfx::distributed

#endif
