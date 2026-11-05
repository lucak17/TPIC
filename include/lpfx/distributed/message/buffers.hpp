#ifndef LPFX_DISTRIBUTED_MESSAGE_BUFFERS_HPP
#define LPFX_DISTRIBUTED_MESSAGE_BUFFERS_HPP

#include <cstddef>

namespace lpfx::distributed {

// Two stable aggregate byte images. Space is compile-time residency state;
// the raw bases remain representable even when invalid so cold binding
// admission can diagnose a null nonempty image.
template <class Space> class FixedMessageBuffers final {
public:
  using space_type = Space;

  constexpr FixedMessageBuffers() noexcept = default;
  constexpr FixedMessageBuffers(const std::byte *const outgoing,
                                const std::size_t outgoing_bytes,
                                std::byte *const incoming,
                                const std::size_t incoming_bytes) noexcept
      : outgoing_(outgoing), incoming_(incoming),
        outgoing_bytes_(outgoing_bytes), incoming_bytes_(incoming_bytes) {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    return (outgoing_bytes_ == 0 || outgoing_ != nullptr) &&
           (incoming_bytes_ == 0 || incoming_ != nullptr);
  }

  [[nodiscard]] constexpr const std::byte *outgoing_data() const noexcept {
    return outgoing_;
  }
  [[nodiscard]] constexpr std::size_t outgoing_bytes() const noexcept {
    return outgoing_bytes_;
  }
  [[nodiscard]] constexpr std::byte *incoming_data() const noexcept {
    return incoming_;
  }
  [[nodiscard]] constexpr std::size_t incoming_bytes() const noexcept {
    return incoming_bytes_;
  }

private:
  const std::byte *outgoing_ = nullptr;
  std::byte *incoming_ = nullptr;
  std::size_t outgoing_bytes_ = 0;
  std::size_t incoming_bytes_ = 0;
};

} // namespace lpfx::distributed

#endif
