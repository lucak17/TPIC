#ifndef LPFX_DISTRIBUTED_DESCRIPTOR_MESSAGE_HPP
#define LPFX_DISTRIBUTED_DESCRIPTOR_MESSAGE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

#include <lpfx/distributed/descriptor/family.hpp>

namespace lpfx::distributed {

// A fixed-byte message descriptor identifies the application wire protocol
// and bounds rank-local fragments. Peers, image offsets and active counts are
// runtime plan facts and deliberately do not become template arguments.
template <std::uint64_t ProtocolIdentity, std::size_t MaximumFragments,
          std::size_t MaximumPeers = MaximumFragments>
  requires(ProtocolIdentity != 0) && (MaximumFragments != 0) &&
          (MaximumPeers != 0) && (MaximumPeers <= MaximumFragments) &&
          (MaximumFragments <=
           static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()))
struct FixedByteMessages final {
  static constexpr operation_family family = operation_family::message;
  static constexpr std::uint64_t protocol_identity = ProtocolIdentity;
  static constexpr std::size_t maximum_fragment_count = MaximumFragments;
  static constexpr std::size_t maximum_peer_count = MaximumPeers;
};

namespace detail {

template <class T> inline constexpr bool is_fixed_byte_messages = false;
template <std::uint64_t ProtocolIdentity, std::size_t MaximumFragments,
          std::size_t MaximumPeers>
inline constexpr bool is_fixed_byte_messages<
    FixedByteMessages<ProtocolIdentity, MaximumFragments, MaximumPeers>> =
    true;

template <class T>
concept FixedByteMessagesDescriptor = is_fixed_byte_messages<T>;

inline constexpr std::size_t fixed_message_invariant_descriptor_extent = 32;
inline constexpr std::size_t fixed_message_directed_header_extent = 32;
inline constexpr std::size_t fixed_message_fragment_record_extent = 16;

template <FixedByteMessagesDescriptor Descriptor>
inline constexpr std::size_t fixed_message_reciprocal_descriptor_capacity =
    fixed_message_directed_header_extent +
    fixed_message_fragment_record_extent *
        Descriptor::maximum_fragment_count;

} // namespace detail
} // namespace lpfx::distributed

#endif
