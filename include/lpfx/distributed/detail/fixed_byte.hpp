#ifndef LPFX_DISTRIBUTED_DETAIL_FIXED_BYTE_HPP
#define LPFX_DISTRIBUTED_DETAIL_FIXED_BYTE_HPP

#include <cstddef>
#include <cstdint>
#include <span>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx::distributed::detail {

// A borrowed byte range; Byte is std::byte or const std::byte.
template <class Byte> struct ByteRegion final {
  Byte *data = nullptr;
  std::size_t size_bytes = 0;

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size_bytes == 0;
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return size_bytes == 0 || data != nullptr;
  }
};

// A route identifies the peer's receive-message group. It is a cold native
// matching key, not a runtime semantic instruction.
struct FixedByteSendMessage final {
  const std::byte *data = nullptr;
  std::size_t size_bytes = 0;
  std::uint32_t peer = 0;
  std::uint32_t route = 0;
};

struct FixedByteReceiveMessage final {
  std::byte *data = nullptr;
  std::size_t size_bytes = 0;
  std::uint32_t peer = 0;
  std::uint32_t route = 0;
};

// Cold projection over unpublished descriptors. The family proves that all
// nonempty records address one image and that receive regions are disjoint.
// Only the send/receive pointer qualification specializes this helper.
template <class Message>
[[nodiscard]] constexpr ErrorCode try_coalesce_fixed_byte_messages(
    const std::span<Message> messages, std::size_t &count) noexcept {
  const auto less = [](const auto &left, const auto &right) constexpr noexcept {
    return left.peer < right.peer ||
           (left.peer == right.peer && left.data < right.data);
  };
  for (std::size_t i = 1; i < messages.size(); ++i) {
    const auto value = messages[i];
    std::size_t j = i;
    while (j != 0 && less(value, messages[j - 1])) {
      messages[j] = messages[j - 1];
      --j;
    }
    messages[j] = value;
  }
  std::size_t output = 0;
  for (const auto next : messages) {
    if (output != 0 && messages[output - 1].peer == next.peer) {
      auto &current = messages[output - 1];
      if (current.data + current.size_bytes != next.data)
        return ErrorCode::invalid_argument;
      if (!lpfx::detail::checked_add(current.size_bytes, next.size_bytes,
                                    current.size_bytes))
        return ErrorCode::overflow;
    } else {
      messages[output] = next;
      messages[output++].route = 0;
    }
  }
  count = output;
  return ErrorCode::success;
}

// Ephemeral preparation input. The family has validated image membership,
// disjoint native write storage, peer reciprocity, and fixed message extents.
// Leaves seal native recipes and discard these spans before preparation ends.
struct FixedByteExchange final {
  std::span<const FixedByteSendMessage> sends{};
  std::span<const FixedByteReceiveMessage> receives{};
  ByteRegion<const std::byte> outgoing{};
  ByteRegion<std::byte> incoming{};
  std::uint32_t rank = 0;
  std::uint32_t rank_count = 0;

  [[nodiscard]] constexpr bool has_native_work() const noexcept {
    return !sends.empty() || !receives.empty();
  }
};

} // namespace lpfx::distributed::detail

#endif
