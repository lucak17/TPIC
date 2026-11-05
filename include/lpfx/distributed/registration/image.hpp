#ifndef LPFX_DISTRIBUTED_REGISTRATION_IMAGE_HPP
#define LPFX_DISTRIBUTED_REGISTRATION_IMAGE_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <tuple>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/distributed/registration/record.hpp>

// One rank's registration image, little endian:
//   header   magic[8] | image bytes u64 | channel count u32 | contract count u32
//   channel  id u64 | invariant offset u32, size u32 | first contract u32,
//            contract count u32
//   contract peer u32 | local group u32 | remote group u32 |
//            outbound offset u32, size u32 | inbound offset u32, size u32
//   payload  descriptor bytes addressed by the offsets above
// Channels appear in canonical channel-identity order and each channel's
// contracts in canonical key order.
namespace lpfx::distributed::detail {

using RegistrationIntegerCodec =
    lpfx::detail::integer_codec<std::endian::little>;

struct RegistrationImageLayout final {
  static constexpr std::array<std::byte, 8> magic{
      std::byte{'L'}, std::byte{'P'}, std::byte{'F'}, std::byte{'x'},
      std::byte{'R'}, std::byte{'M'}, std::byte{'0'}, std::byte{'3'}};
  static constexpr std::size_t header_bytes = 24;
  static constexpr std::size_t channel_bytes = 24;
  static constexpr std::size_t contract_bytes = 28;

  std::uint32_t channel_count = 0;
  std::uint32_t contract_count = 0;
  std::size_t contracts_offset = 0;
  std::size_t payload_offset = 0;
  std::size_t image_bytes = 0;

  [[nodiscard]] constexpr std::size_t payload_bytes() const noexcept {
    return image_bytes - payload_offset;
  }
};

[[nodiscard]] constexpr bool try_make_registration_image_layout(
    const std::size_t channels, const std::size_t contracts,
    const std::size_t payload_bytes, RegistrationImageLayout &output) noexcept {
  using namespace lpfx::detail;
  RegistrationImageLayout candidate{};
  std::size_t bytes{};
  if (payload_bytes > std::numeric_limits<std::uint32_t>::max() ||
      !checked_integral_cast(channels, candidate.channel_count) ||
      !checked_integral_cast(contracts, candidate.contract_count) ||
      !checked_multiply(channels, RegistrationImageLayout::channel_bytes,
                        bytes) ||
      !checked_add(RegistrationImageLayout::header_bytes, bytes,
                   candidate.contracts_offset) ||
      !checked_multiply(contracts, RegistrationImageLayout::contract_bytes,
                        bytes) ||
      !checked_add(candidate.contracts_offset, bytes,
                   candidate.payload_offset) ||
      !checked_add(candidate.payload_offset, payload_bytes,
                   candidate.image_bytes)) {
    return false;
  }
  output = candidate;
  return true;
}

// Plans are in canonical channel order and borrowed until encoding ends.
template <class... Plans>
[[nodiscard]] constexpr ErrorCode
try_layout_registration_image(const std::tuple<const Plans *...> &plans,
                              RegistrationImageLayout &output) noexcept {
  std::size_t contracts = 0;
  std::size_t payload = 0;
  const ErrorCode error = std::apply(
      [&](const auto *...plan) noexcept {
        ErrorCode result = ErrorCode::success;
        const auto measure = [&]<class Plan>(const Plan &source) noexcept {
          const auto &channel = source.registration_channel();
          if (!source.is_valid() ||
              channel.channel_id.value() != Plan::channel_identity ||
              channel.invariant_descriptor.empty()) {
            return ErrorCode::invalid_argument;
          }
          contracts += channel.contract_count();
          return lpfx::detail::checked_add(
                     payload,
                     channel.invariant_descriptor.size() +
                         channel.descriptor_pool().size(),
                     payload)
                     ? ErrorCode::success
                     : ErrorCode::overflow;
        };
        static_cast<void>(
            (((result = measure(*plan)) == ErrorCode::success) && ...));
        return result;
      },
      plans);
  if (error != ErrorCode::success) {
    return error;
  }
  return try_make_registration_image_layout(sizeof...(Plans), contracts,
                                            payload, output)
             ? ErrorCode::success
             : ErrorCode::overflow;
}

// Writes the image measured by try_layout_registration_image into output.
template <class... Plans>
[[nodiscard]] ErrorCode
encode_registration_image(const std::tuple<const Plans *...> &plans,
                          const RegistrationImageLayout &layout,
                          const std::span<std::byte> output) noexcept {
  using Codec = RegistrationIntegerCodec;
  if (output.size() != layout.image_bytes || output.data() == nullptr) {
    return ErrorCode::invalid_argument;
  }
  std::byte *const image = output.data();
  std::memcpy(image, RegistrationImageLayout::magic.data(),
              RegistrationImageLayout::magic.size());
  Codec::write(image + 8, std::uint64_t{layout.image_bytes});
  Codec::write(image + 16, layout.channel_count);
  Codec::write(image + 20, layout.contract_count);

  std::byte *channel_row = image + RegistrationImageLayout::header_bytes;
  std::byte *contract_row = image + layout.contracts_offset;
  std::byte *const payload = image + layout.payload_offset;
  std::uint32_t contract_index = 0;
  std::uint32_t cursor = 0;
  const auto append = [&](const std::span<const std::byte> bytes) noexcept {
    if (!bytes.empty()) {
      std::memcpy(payload + cursor, bytes.data(), bytes.size());
    }
    cursor += static_cast<std::uint32_t>(bytes.size());
  };
  std::apply(
      [&](const auto *...plan) noexcept {
        const auto encode = [&](const auto &channel) noexcept {
          const auto invariant = channel.invariant_descriptor.view();
          Codec::write(channel_row, channel.channel_id.value());
          Codec::write(channel_row + 8, cursor);
          Codec::write(channel_row + 12,
                       static_cast<std::uint32_t>(invariant.size()));
          Codec::write(channel_row + 16, contract_index);
          Codec::write(channel_row + 20,
                       static_cast<std::uint32_t>(channel.contract_count()));
          channel_row += RegistrationImageLayout::channel_bytes;
          append(invariant);
          const std::uint32_t pool = cursor;
          append(channel.descriptor_pool());
          const auto slice = [pool](const auto value) noexcept {
            return value.size == 0 ? std::uint32_t{0} : pool + value.offset;
          };
          for (const auto &contract : channel.contracts()) {
            Codec::write(contract_row, contract.peer);
            Codec::write(contract_row + 4, contract.local_group);
            Codec::write(contract_row + 8, contract.remote_group);
            Codec::write(contract_row + 12, slice(contract.outbound));
            Codec::write(contract_row + 16, contract.outbound.size);
            Codec::write(contract_row + 20, slice(contract.inbound));
            Codec::write(contract_row + 24, contract.inbound.size);
            contract_row += RegistrationImageLayout::contract_bytes;
            ++contract_index;
          }
        };
        (encode(plan->registration_channel()), ...);
      },
      plans);
  return ErrorCode::success;
}

class RegistrationContractRange final {
public:
  constexpr RegistrationContractRange() noexcept = default;
  constexpr RegistrationContractRange(
      const std::span<const std::byte> rows,
      const std::span<const std::byte> payload) noexcept
      : rows_(rows), payload_(payload) {}

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return rows_.size() / RegistrationImageLayout::contract_bytes;
  }

  // Requires index < size() on a parsed image.
  [[nodiscard]] RegistrationContractView
  operator[](const std::size_t index) const noexcept {
    using Codec = RegistrationIntegerCodec;
    const std::byte *const row =
        rows_.data() + index * RegistrationImageLayout::contract_bytes;
    const auto bytes = [&](const std::size_t at) noexcept {
      return payload_.subspan(Codec::read<std::uint32_t>(row + at),
                              Codec::read<std::uint32_t>(row + at + 4));
    };
    return {Codec::read<std::uint32_t>(row), Codec::read<std::uint32_t>(row + 4),
            Codec::read<std::uint32_t>(row + 8), bytes(12), bytes(20)};
  }

  [[nodiscard]] std::optional<RegistrationContractView>
  find(const std::uint32_t peer, const std::uint32_t local,
       const std::uint32_t remote) const noexcept {
    const auto key = std::tuple{peer, local, remote};
    std::size_t first = 0;
    std::size_t last = size();
    while (first != last) {
      const std::size_t middle = first + (last - first) / 2;
      if ((*this)[middle].key() < key) {
        first = middle + 1;
      } else {
        last = middle;
      }
    }
    if (first == size() || (*this)[first].key() != key) {
      return std::nullopt;
    }
    return (*this)[first];
  }

private:
  std::span<const std::byte> rows_{};
  std::span<const std::byte> payload_{};
};

struct RegistrationChannelView final {
  std::uint64_t channel_id = 0;
  std::span<const std::byte> invariant{};
  RegistrationContractRange contracts{};
};

class RegistrationImageView final {
public:
  constexpr RegistrationImageView() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return !bytes_.empty();
  }
  [[nodiscard]] constexpr std::uint32_t channel_count() const noexcept {
    return layout_.channel_count;
  }

  // Requires ordinal < channel_count() on a parsed image.
  [[nodiscard]] RegistrationChannelView
  channel(const std::size_t ordinal) const noexcept {
    using Codec = RegistrationIntegerCodec;
    const std::byte *const row = bytes_.data() +
                                 RegistrationImageLayout::header_bytes +
                                 ordinal * RegistrationImageLayout::channel_bytes;
    const auto payload = bytes_.subspan(layout_.payload_offset);
    return {Codec::read<std::uint64_t>(row),
            payload.subspan(Codec::read<std::uint32_t>(row + 8),
                            Codec::read<std::uint32_t>(row + 12)),
            RegistrationContractRange{
                bytes_.subspan(layout_.contracts_offset +
                                   std::size_t{Codec::read<std::uint32_t>(
                                       row + 16)} *
                                       RegistrationImageLayout::contract_bytes,
                               std::size_t{Codec::read<std::uint32_t>(
                                   row + 20)} *
                                   RegistrationImageLayout::contract_bytes),
                payload}};
  }

  // Checks the framing only; try_parse additionally checks every row.
  [[nodiscard]] static bool open(const std::span<const std::byte> bytes,
                                 RegistrationImageView &output) noexcept {
    using Codec = RegistrationIntegerCodec;
    RegistrationImageLayout layout{};
    if (bytes.size() < RegistrationImageLayout::header_bytes ||
        bytes.data() == nullptr ||
        std::memcmp(bytes.data(), RegistrationImageLayout::magic.data(),
                    RegistrationImageLayout::magic.size()) != 0 ||
        Codec::read<std::uint64_t>(bytes.data() + 8) != bytes.size()) {
      return false;
    }
    const auto channels = Codec::read<std::uint32_t>(bytes.data() + 16);
    const auto contracts = Codec::read<std::uint32_t>(bytes.data() + 20);
    // The payload extent is whatever follows the rows.
    if (!try_make_registration_image_layout(channels, contracts, 0, layout) ||
        layout.payload_offset > bytes.size() || channels == 0 ||
        !try_make_registration_image_layout(
            channels, contracts, bytes.size() - layout.payload_offset,
            layout)) {
      return false;
    }
    output.bytes_ = bytes;
    output.layout_ = layout;
    return true;
  }

  [[nodiscard]] static ErrorCode
  try_parse(const std::span<const std::byte> bytes,
            RegistrationImageView &output) noexcept {
    using Codec = RegistrationIntegerCodec;
    RegistrationImageView candidate{};
    if (!open(bytes, candidate)) {
      return ErrorCode::invalid_argument;
    }
    const auto &layout = candidate.layout_;
    const std::size_t payload = layout.payload_bytes();
    const auto within = [payload](const std::byte *const at) noexcept {
      const std::size_t offset = Codec::read<std::uint32_t>(at);
      const std::size_t size = Codec::read<std::uint32_t>(at + 4);
      return offset <= payload && size <= payload - offset;
    };
    std::uint64_t previous_channel = 0;
    std::size_t contract_cursor = 0;
    for (std::size_t ordinal = 0; ordinal != layout.channel_count; ++ordinal) {
      const std::byte *const row = bytes.data() +
                                   RegistrationImageLayout::header_bytes +
                                   ordinal * RegistrationImageLayout::channel_bytes;
      const auto id = Codec::read<std::uint64_t>(row);
      const std::size_t count = Codec::read<std::uint32_t>(row + 20);
      if (id <= previous_channel || !within(row + 8) ||
          Codec::read<std::uint32_t>(row + 12) == 0 ||
          Codec::read<std::uint32_t>(row + 16) != contract_cursor ||
          count > layout.contract_count - contract_cursor) {
        return ErrorCode::invalid_argument;
      }
      previous_channel = id;
      std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> previous_key{};
      for (std::size_t index = 0; index != count; ++index) {
        const std::byte *const contract =
            bytes.data() + layout.contracts_offset +
            (contract_cursor + index) * RegistrationImageLayout::contract_bytes;
        const auto key = std::tuple{Codec::read<std::uint32_t>(contract),
                                    Codec::read<std::uint32_t>(contract + 4),
                                    Codec::read<std::uint32_t>(contract + 8)};
        if ((index != 0 && key <= previous_key) || !within(contract + 12) ||
            !within(contract + 20)) {
          return ErrorCode::invalid_argument;
        }
        previous_key = key;
      }
      contract_cursor += count;
    }
    if (contract_cursor != layout.contract_count) {
      return ErrorCode::invalid_argument;
    }
    output = candidate;
    return ErrorCode::success;
  }

private:
  std::span<const std::byte> bytes_{};
  RegistrationImageLayout layout_{};
};

// Borrowed rank images: offsets[r]..offsets[r + 1] is rank r's image.
class GatheredRegistrationView final {
public:
  constexpr GatheredRegistrationView(
      const std::span<const std::byte> bytes,
      const std::span<const std::uint64_t> offsets) noexcept
      : bytes_(bytes), offsets_(offsets) {}

  [[nodiscard]] constexpr std::size_t rank_count() const noexcept {
    return offsets_.empty() ? 0 : offsets_.size() - 1;
  }
  [[nodiscard]] std::span<const std::byte>
  image_bytes(const std::size_t rank) const noexcept {
    if (offsets_[rank] > offsets_[rank + 1] ||
        offsets_[rank + 1] > bytes_.size()) {
      return {};
    }
    return bytes_.subspan(
        static_cast<std::size_t>(offsets_[rank]),
        static_cast<std::size_t>(offsets_[rank + 1] - offsets_[rank]));
  }
  // Requires every image to have passed try_parse.
  [[nodiscard]] RegistrationImageView
  rank(const std::size_t rank) const noexcept {
    RegistrationImageView output{};
    static_cast<void>(RegistrationImageView::open(image_bytes(rank), output));
    return output;
  }
  [[nodiscard]] constexpr std::span<const std::byte> bytes() const noexcept {
    return bytes_;
  }
  [[nodiscard]] constexpr std::span<const std::uint64_t>
  offsets() const noexcept {
    return offsets_;
  }

private:
  std::span<const std::byte> bytes_{};
  std::span<const std::uint64_t> offsets_{};
};

// All ranks register the same channels with equal invariants, and every
// directed contract has its reciprocal with matching descriptors.
[[nodiscard]] inline ErrorCode
validate_registration_images(const GatheredRegistrationView gathered) noexcept {
  const std::size_t ranks = gathered.rank_count();
  if (ranks == 0 || ranks > std::numeric_limits<std::uint32_t>::max() ||
      gathered.offsets().front() != 0 ||
      gathered.offsets().back() != gathered.bytes().size()) {
    return ErrorCode::invalid_argument;
  }
  // Parse every image before following any cross-rank reference.
  for (std::size_t rank = 0; rank != ranks; ++rank) {
    RegistrationImageView parsed{};
    if (RegistrationImageView::try_parse(gathered.image_bytes(rank), parsed) !=
        ErrorCode::success) {
      return ErrorCode::invalid_argument;
    }
  }
  const auto same = [](const std::span<const std::byte> left,
                       const std::span<const std::byte> right) noexcept {
    return std::ranges::equal(left, right);
  };
  const RegistrationImageView reference = gathered.rank(0);
  for (std::size_t rank = 1; rank != ranks; ++rank) {
    const RegistrationImageView image = gathered.rank(rank);
    if (image.channel_count() != reference.channel_count()) {
      return ErrorCode::invalid_argument;
    }
    for (std::size_t ordinal = 0; ordinal != image.channel_count(); ++ordinal) {
      const RegistrationChannelView channel = image.channel(ordinal);
      const RegistrationChannelView expected = reference.channel(ordinal);
      if (channel.channel_id != expected.channel_id ||
          !same(channel.invariant, expected.invariant)) {
        return ErrorCode::invalid_argument;
      }
    }
  }
  for (std::size_t rank = 0; rank != ranks; ++rank) {
    const RegistrationImageView image = gathered.rank(rank);
    for (std::size_t ordinal = 0; ordinal != image.channel_count(); ++ordinal) {
      const RegistrationChannelView channel = image.channel(ordinal);
      for (std::size_t index = 0; index != channel.contracts.size(); ++index) {
        const RegistrationContractView contract = channel.contracts[index];
        if (contract.peer_rank >= ranks) {
          return ErrorCode::invalid_argument;
        }
        const auto reciprocal =
            gathered.rank(contract.peer_rank)
                .channel(ordinal)
                .contracts.find(static_cast<std::uint32_t>(rank),
                                contract.remote_receive_message_group_id,
                                contract.local_receive_message_group_id);
        // Every directed edge needs a reciprocal; its bytes are compared once,
        // from the canonically smaller side.
        if (!reciprocal) {
          return ErrorCode::invalid_argument;
        }
        if (rank > contract.peer_rank ||
            (rank == contract.peer_rank &&
             contract.local_receive_message_group_id >
                 contract.remote_receive_message_group_id)) {
          continue;
        }
        const bool self_reciprocal =
            rank == contract.peer_rank &&
            contract.local_receive_message_group_id ==
                contract.remote_receive_message_group_id;
        if (!same(contract.outbound_descriptor,
                  reciprocal->inbound_descriptor) ||
            (!self_reciprocal && !same(contract.inbound_descriptor,
                                       reciprocal->outbound_descriptor))) {
          return ErrorCode::invalid_argument;
        }
      }
    }
  }
  return ErrorCode::success;
}

} // namespace lpfx::distributed::detail

#endif
