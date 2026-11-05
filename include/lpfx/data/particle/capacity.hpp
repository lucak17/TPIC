#ifndef LPFX_DATA_PARTICLE_CAPACITY_HPP
#define LPFX_DATA_PARTICLE_CAPACITY_HPP

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>

#include <lpfx/config.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx {

enum class ParticleResizeMode : std::uint8_t { preserve, discard };

inline constexpr std::uint32_t particle_capacity_headroom_percent =
    LPFX_CONFIG_PARTICLE_CAPACITY_HEADROOM_PERCENT;
inline constexpr std::uint32_t particle_capacity_multiple =
    LPFX_CONFIG_PARTICLE_CAPACITY_MULTIPLE;

namespace detail {

template <std::uint32_t Percent, std::unsigned_integral Count>
[[nodiscard]] constexpr Count
particle_capacity_with_headroom(const Count required) noexcept {
  if constexpr (Percent == 0) {
    return required;
  } else {
    constexpr auto divisor = std::gcd(Percent, std::uint32_t{100});
    constexpr auto numerator = Percent / divisor;
    constexpr auto denominator = 100U / divisor;
    constexpr auto limit = std::numeric_limits<Count>::max();
    const auto available = static_cast<Count>(limit - required);
    const auto whole = required / denominator;
    if (whole > available / numerator)
      return limit;
    const auto extra_whole = whole * numerator;
    // The remainder is below 100 and Percent fits uint32_t. Widen only
    // this product; the full required * Percent product could overflow.
    const auto extra_remainder =
        (static_cast<std::uint64_t>(required % denominator) * numerator +
         denominator - 1) /
        denominator;
    if (extra_remainder > available - extra_whole)
      return limit;
    return static_cast<Count>(required + extra_whole + extra_remainder);
  }
}

template <std::uint32_t Percent, std::uint32_t Multiple,
          std::unsigned_integral Count>
[[nodiscard]] constexpr ErrorCode
try_prepare_particle_capacity(const Count required, Count &output) noexcept {
  static_assert(std::has_single_bit(Multiple));
  constexpr auto limit = std::numeric_limits<Count>::max();
  if constexpr (Multiple > limit) {
    if (required != 0)
      return ErrorCode::capacity_exceeded;
    output = 0;
  } else {
    constexpr auto mask = static_cast<Count>(Multiple - 1);
    constexpr auto aligned_limit = static_cast<Count>(limit & ~mask);
    if (required > aligned_limit)
      return ErrorCode::capacity_exceeded;
    const auto padded = particle_capacity_with_headroom<Percent>(required);
    const auto bounded = padded > aligned_limit ? aligned_limit : padded;
    // Clipping optional spare slots ensures bounded + mask cannot overflow.
    // Keep the mask in Count so uint64_t capacities retain their upper bits.
    output = static_cast<Count>((bounded + mask) & ~mask);
  }
  return ErrorCode::success;
}

} // namespace detail

// Prepare a minimum particle count with configured headroom and count rounding.
// Only optional spare slots may be clipped at the largest representable
// multiple. An unrepresentable requirement fails without changing output.
// Prepared storage layouts remain exact; apply this before preparing a layout.
template <std::unsigned_integral Count>
[[nodiscard]] constexpr ErrorCode
try_prepare_particle_capacity(const Count required, Count &output) noexcept {
  return detail::try_prepare_particle_capacity<
      particle_capacity_headroom_percent, particle_capacity_multiple>(required,
                                                                      output);
}

namespace detail {

// Apply the same configured reserve as initial creation, based on the new
// requirement. Existing allocations that fit are reused without more padding.
[[nodiscard]] constexpr ErrorCode
try_grow_particle_capacity(const std::uint32_t current,
                           const std::uint64_t required,
                           std::uint32_t &output) noexcept {
  constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
  if (required > limit)
    return ErrorCode::capacity_exceeded;
  if (required <= current) {
    output = current;
    return ErrorCode::success;
  }
  return lpfx::try_prepare_particle_capacity(
      static_cast<std::uint32_t>(required), output);
}

} // namespace detail

} // namespace lpfx

#endif
