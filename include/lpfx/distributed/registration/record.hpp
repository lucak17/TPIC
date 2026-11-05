#ifndef LPFX_DISTRIBUTED_REGISTRATION_RECORD_HPP
#define LPFX_DISTRIBUTED_REGISTRATION_RECORD_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/detail/static_vector.hpp>

namespace lpfx {

// A nonzero channel identity. Registration checks it once against the plan's
// compile-time channel_identity.
class ChannelId final {
public:
  constexpr ChannelId() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return value_ != 0; }
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }

  [[nodiscard]] friend constexpr bool
  operator==(const ChannelId &, const ChannelId &) noexcept = default;

private:
  explicit constexpr ChannelId(const std::uint64_t value) noexcept
      : value_(value) {}

  std::uint64_t value_ = 0;

  friend struct detail::PrivateConstructionAccess;
};

template <detail::checked_integer Value>
[[nodiscard]] constexpr ErrorCode try_make_channel_id(const Value value,
                                                      ChannelId &output) noexcept {
  std::uint64_t prepared{};
  if (!detail::checked_integral_cast(value, prepared)) {
    return ErrorCode::overflow;
  }
  if (prepared == 0) {
    return ErrorCode::invalid_argument;
  }
  output = detail::PrivateConstructionAccess::create<ChannelId>(prepared);
  return ErrorCode::success;
}

template <std::size_t Capacity>
using CanonicalRegistrationBytes = detail::static_vector<std::byte, Capacity>;

namespace distributed::detail {

// One directed reciprocal contract of a channel, borrowed from its owner.
struct RegistrationContractView final {
  std::uint32_t peer_rank = 0;
  std::uint32_t local_receive_message_group_id = 0;
  std::uint32_t remote_receive_message_group_id = 0;
  std::span<const std::byte> outbound_descriptor{};
  std::span<const std::byte> inbound_descriptor{};

  [[nodiscard]] constexpr auto key() const noexcept {
    return std::tuple{peer_rank, local_receive_message_group_id,
                      remote_receive_message_group_id};
  }
};

} // namespace distributed::detail

// A plan's registration record: its invariant descriptor and its directed
// contracts, kept in canonical (peer, local group, remote group) order with
// unique keys. Descriptor bytes live in one pool addressed by offsets, so the
// value stays relocatable.
template <std::size_t InvariantBytes, std::size_t MaxContracts,
          std::size_t MaxDescriptorBytes,
          std::size_t PoolBytes = 2 * MaxContracts * MaxDescriptorBytes>
  requires(PoolBytes <= std::numeric_limits<std::uint32_t>::max())
class RegistrationChannel final {
public:
  struct Slice final {
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
  };
  struct Contract final {
    std::uint32_t peer = 0;
    std::uint32_t local_group = 0;
    std::uint32_t remote_group = 0;
    Slice outbound{};
    Slice inbound{};

    [[nodiscard]] constexpr auto key() const noexcept {
      return std::tuple{peer, local_group, remote_group};
    }
  };

  static constexpr std::size_t invariant_descriptor_capacity = InvariantBytes;
  static constexpr std::size_t directed_contract_capacity = MaxContracts;
  static constexpr std::size_t reciprocal_descriptor_capacity =
      MaxDescriptorBytes;
  static constexpr std::size_t descriptor_pool_capacity = PoolBytes;

  ChannelId channel_id{};
  CanonicalRegistrationBytes<InvariantBytes> invariant_descriptor{};

  [[nodiscard]] constexpr std::size_t contract_count() const noexcept {
    return contracts_.size();
  }
  [[nodiscard]] constexpr std::span<const Contract>
  contracts() const noexcept {
    return contracts_.view();
  }
  // The used prefix of the descriptor pool that contract slices address.
  [[nodiscard]] constexpr std::span<const std::byte>
  descriptor_pool() const noexcept {
    return std::span<const std::byte>{bytes_}.first(used_bytes_);
  }
  [[nodiscard]] constexpr distributed::detail::RegistrationContractView
  contract(const std::size_t index) const noexcept {
    const Contract &entry = contracts_.view()[index];
    return {entry.peer, entry.local_group, entry.remote_group,
            resolve(entry.outbound), resolve(entry.inbound)};
  }

  // Writers fill exactly the requested, zeroed extents. A duplicate key is
  // invalid_argument; exhausted capacity is capacity_exceeded.
  template <class OutboundWriter, class InboundWriter>
  [[nodiscard]] constexpr ErrorCode try_append_contract(
      const std::uint32_t peer, const std::uint32_t local_group,
      const std::uint32_t remote_group, const std::size_t outbound_size,
      const std::size_t inbound_size, OutboundWriter &&write_outbound,
      InboundWriter &&write_inbound) noexcept {
    if (contract_count() == MaxContracts ||
        outbound_size > MaxDescriptorBytes ||
        inbound_size > MaxDescriptorBytes ||
        outbound_size > PoolBytes - used_bytes_ ||
        inbound_size > PoolBytes - used_bytes_ - outbound_size) {
      return ErrorCode::capacity_exceeded;
    }
    const Contract candidate{peer, local_group, remote_group,
                             slice(used_bytes_, outbound_size),
                             slice(used_bytes_ + outbound_size, inbound_size)};
    std::size_t position = contracts_.size();
    while (position != 0 &&
           candidate.key() < contracts_.view()[position - 1].key()) {
      --position;
    }
    if (position != 0 &&
        candidate.key() == contracts_.view()[position - 1].key()) {
      return ErrorCode::invalid_argument;
    }
    const auto outbound = std::span{bytes_}.subspan(used_bytes_, outbound_size);
    const auto inbound =
        std::span{bytes_}.subspan(used_bytes_ + outbound_size, inbound_size);
    // Failed writers may dirty the unused suffix; re-zero before every attempt.
    for (auto &byte : outbound) {
      byte = std::byte{};
    }
    for (auto &byte : inbound) {
      byte = std::byte{};
    }
    LPFX_TRY(write_outbound(outbound));
    LPFX_TRY(write_inbound(inbound));
    if (!contracts_.try_insert(position, candidate)) {
      return ErrorCode::capacity_exceeded;
    }
    used_bytes_ += static_cast<std::uint32_t>(outbound_size + inbound_size);
    return ErrorCode::success;
  }

  [[nodiscard]] constexpr ErrorCode
  try_append_contract(const std::uint32_t peer, const std::uint32_t local_group,
                      const std::uint32_t remote_group,
                      const std::span<const std::byte> outbound,
                      const std::span<const std::byte> inbound) noexcept {
    const auto copy = [](const std::span<const std::byte> source) {
      return [source](const std::span<std::byte> destination) constexpr noexcept {
        for (std::size_t index = 0; index != source.size(); ++index) {
          destination[index] = source[index];
        }
        return ErrorCode::success;
      };
    };
    return try_append_contract(peer, local_group, remote_group,
                               outbound.size(), inbound.size(), copy(outbound),
                               copy(inbound));
  }

private:
  [[nodiscard]] static constexpr Slice slice(const std::size_t offset,
                                             const std::size_t size) noexcept {
    return size == 0 ? Slice{}
                     : Slice{static_cast<std::uint32_t>(offset),
                             static_cast<std::uint32_t>(size)};
  }
  [[nodiscard]] constexpr std::span<const std::byte>
  resolve(const Slice slice) const noexcept {
    return std::span<const std::byte>{bytes_}.subspan(slice.offset, slice.size);
  }

  detail::static_vector<Contract, MaxContracts> contracts_{};
  std::array<std::byte, PoolBytes> bytes_{};
  std::uint32_t used_bytes_{};
};

} // namespace lpfx

#endif
