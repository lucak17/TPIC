#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_PAYLOAD_SLICES_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_PAYLOAD_SLICES_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/particle/wire/image.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

enum class CommunicationFlow : std::uint8_t { send, receive };

template <CommunicationFlow Flow>
using ParticlePayloadByte =
    std::conditional_t<Flow == CommunicationFlow::send, const std::byte,
                       std::byte>;

template <CommunicationFlow Flow>
inline constexpr ParticlePeerPrefixKind particle_payload_prefix_kind =
    Flow == CommunicationFlow::send ? ParticlePeerPrefixKind::outgoing
                                    : ParticlePeerPrefixKind::incoming;

template <CommunicationFlow Flow> struct ParticlePayloadSlice final {
  static constexpr CommunicationFlow flow = Flow;

  ParticlePayloadByte<Flow> *data = nullptr;
  std::size_t bytes = 0;
  std::uint8_t peer = 0;
};

namespace detail {

template <class Sink, class Slice>
concept ParticlePayloadSliceSink = requires(Sink &sink, const Slice slice) {
  { sink(slice) } noexcept -> std::same_as<ErrorCode>;
};

} // namespace detail

// The one packet traversal: one slice per nonempty peer packet of payload,
// whose bytes are laid out by prefixes. A packet is never split to fit a
// native count; capacity_exceeded when one exceeds maximum_native_count or
// the prefixes exceed payload.
template <CommunicationFlow Flow, detail::ParticleWireContractType Contract,
          class Space, class Sink>
  requires detail::ParticlePayloadSliceSink<Sink, ParticlePayloadSlice<Flow>>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode for_each_particle_payload_slice(
    const PreparedParticlePeerPrefixes<
        Contract, particle_payload_prefix_kind<Flow>> &prefixes,
    const BufferView<ParticlePayloadByte<Flow>, Space> payload,
    const std::uint64_t maximum_native_count, Sink &&sink) noexcept {
  if (maximum_native_count == 0) {
    return ErrorCode::invalid_argument;
  }
  if constexpr (Contract::wire_packet_gap != 0) {
    // Aligned packets: the gap before a packet start is not sent.
    for (std::uint8_t peer = 0; peer != prefixes.peer_count(); ++peer) {
      const std::uint64_t count = prefixes.particle_count(peer);
      if (count == 0) {
        continue;
      }
      std::size_t offset = 0;
      std::size_t size = 0;
      std::size_t end = 0;
      if (!detail::checked_integral_cast(prefixes.byte_offset(peer), offset) ||
          !detail::checked_integral_cast(count * Contract::record_wire_width,
                                         size) ||
          !detail::checked_add(offset, size, end) || end > payload.size() ||
          size > maximum_native_count) {
        return ErrorCode::capacity_exceeded;
      }
      LPFX_TRY(sink(ParticlePayloadSlice<Flow>{payload.data() + offset, size,
                                               peer}));
    }
    return ErrorCode::success;
  }
  std::size_t offset = 0;
  for (std::uint8_t peer = 0; peer != prefixes.peer_count(); ++peer) {
    // Admitted prefixes start at zero and increase monotonically, so the
    // preceding end is this packet's start.
    std::size_t end = 0;
    if (!detail::checked_integral_cast(prefixes.byte_offset(peer + 1), end) ||
        end > payload.size()) {
      return ErrorCode::capacity_exceeded;
    }
    const std::size_t size = end - offset;
    if (size > maximum_native_count) {
      return ErrorCode::capacity_exceeded;
    }
    if (size == 0) {
      continue;
    }
    LPFX_TRY(sink(ParticlePayloadSlice<Flow>{payload.data() + offset, size,
                                             peer}));
    offset = end;
  }
  return ErrorCode::success;
}

} // namespace lpfx

#endif
