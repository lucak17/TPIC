#ifndef LPFX_DISTRIBUTED_DETAIL_INTEGER_CODEC_HPP
#define LPFX_DISTRIBUTED_DETAIL_INTEGER_CODEC_HPP

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <lpfx/core/attributes.hpp>

namespace lpfx::detail {

template <class Byte>
concept integer_codec_byte = std::same_as<std::remove_cv_t<Byte>, std::byte> ||
                             std::same_as<std::remove_cv_t<Byte>, std::uint8_t>;

template <class Integer>
concept integer_codec_integer =
    std::unsigned_integral<std::remove_cv_t<Integer>> &&
    !std::same_as<std::remove_cv_t<Integer>, bool>;

static_assert(std::numeric_limits<unsigned char>::digits == 8);

template <integer_codec_byte Byte>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr std::uint8_t
integer_codec_octet(const Byte value) noexcept {
  if constexpr (std::same_as<std::remove_cv_t<Byte>, std::byte>) {
    return std::to_integer<std::uint8_t>(value);
  } else {
    return value;
  }
}

template <std::endian Endian>
  requires(Endian == std::endian::big || Endian == std::endian::little)
struct integer_codec final {
  template <integer_codec_integer Integer, integer_codec_byte Byte>
  LPFX_ALWAYS_INLINE static constexpr void write(Byte *const destination,
                                                 const Integer value) noexcept {
    for (std::size_t octet = 0; octet != sizeof(Integer); ++octet) {
      const std::size_t position =
          Endian == std::endian::little ? octet : sizeof(Integer) - 1U - octet;
      destination[octet] = static_cast<Byte>(static_cast<std::uint8_t>(
          value >> static_cast<unsigned int>(position * 8U)));
    }
  }

  template <integer_codec_integer Integer, integer_codec_byte Byte>
  [[nodiscard]] LPFX_ALWAYS_INLINE static constexpr Integer
  read(const Byte *const source) noexcept {
    Integer value = 0;
    for (std::size_t octet = 0; octet != sizeof(Integer); ++octet) {
      if constexpr (Endian == std::endian::little) {
        value = static_cast<Integer>(
            value | static_cast<Integer>(integer_codec_octet(source[octet]))
                        << static_cast<unsigned int>(octet * 8U));
      } else {
        value = static_cast<Integer>(value << 8U |
                                     integer_codec_octet(source[octet]));
      }
    }
    return value;
  }

  template <integer_codec_integer Integer, integer_codec_byte Byte,
            std::size_t Count>
  LPFX_ALWAYS_INLINE static constexpr void
  write_sequence(Byte *const destination,
                 const std::array<Integer, Count> &values) noexcept {
    for (std::size_t index = 0; index != Count; ++index) {
      write(destination + index * sizeof(Integer), values[index]);
    }
  }
};

// FNV-1a over the little-endian octets of each word; never zero.
template <std::size_t Count>
[[nodiscard]] constexpr std::uint64_t
hash_words(const std::array<std::uint64_t, Count> &words) noexcept {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (std::uint64_t word : words) {
    for (unsigned octet = 0; octet != 8; ++octet) {
      hash = (hash ^ (word & 255U)) * UINT64_C(1099511628211);
      word >>= 8U;
    }
  }
  return hash == 0 ? 1 : hash;
}

} // namespace lpfx::detail

#endif
