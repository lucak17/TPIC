#ifndef LPFX_BACKEND_PROPERTIES_HPP
#define LPFX_BACKEND_PROPERTIES_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace lpfx::device {

// A stable physical-device identifier suitable for cold rank-to-device
// agreement. Native backends lower their runtime UUID into this fixed value;
// process-local device ordinals deliberately remain separate.
struct DeviceIdentity final {
  static constexpr std::size_t byte_count = 16;

  std::array<std::uint8_t, byte_count> bytes{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    for (const std::uint8_t byte : bytes) {
      if (byte != 0) {
        return true;
      }
    }
    return false;
  }

  friend constexpr bool operator==(DeviceIdentity,
                                   DeviceIdentity) noexcept = default;
};

static_assert(sizeof(DeviceIdentity) == DeviceIdentity::byte_count);

// Runtime launch facts are queried once while preparing work and then passed to
// launch preparation.  Keeping only current consumers here avoids a cached copy
// of a vendor property structure and keeps backend headers out of common code.
struct DeviceProperties final {
  std::int32_t device_ordinal{};
  std::uint32_t subgroup_width{};
  std::uint32_t maximum_threads_per_block{};
  std::uint32_t maximum_grid_size_x{};
  std::uint32_t multiprocessor_count{};
};

static_assert(sizeof(DeviceProperties) == 5 * sizeof(std::uint32_t));

// A cold reporting snapshot, separate from the compact launch properties.
// Free memory is shared with other processes and is not an allocation promise.
struct DeviceMemoryInfo final {
  std::size_t free_bytes{};
  std::size_t total_bytes{};
};

} // namespace lpfx::device

#endif
