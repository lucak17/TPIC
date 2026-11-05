#ifndef LPFX_DISTRIBUTED_COMMON_CANONICAL_WIRE_HPP
#define LPFX_DISTRIBUTED_COMMON_CANONICAL_WIRE_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include <lpfx/distributed/detail/integer_codec.hpp>

namespace lpfx::detail {

[[nodiscard]] constexpr std::size_t
canonical_wire_descriptor_extent(const std::uint16_t header_extent,
                                 const std::uint16_t entry_extent,
                                 const std::uint16_t entry_count) noexcept {
  return static_cast<std::size_t>(header_extent) +
         entry_count * static_cast<std::size_t>(entry_extent);
}

template <integer_codec_byte DestinationByte, std::size_t DestinationExtent,
          integer_codec_byte SourceByte, std::size_t SourceExtent>
constexpr void write_canonical_wire_prefix(
    std::array<DestinationByte, DestinationExtent> &destination,
    const std::array<SourceByte, SourceExtent> &source) noexcept {
  static_assert(SourceExtent <= DestinationExtent);
  for (std::size_t index = 0; index != SourceExtent; ++index) {
    destination[index] =
        static_cast<DestinationByte>(integer_codec_octet(source[index]));
  }
}

template <integer_codec_byte ActualByte, integer_codec_byte ExpectedByte,
          std::size_t Extent>
[[nodiscard]] constexpr bool canonical_wire_prefix_matches(
    const ActualByte *const actual,
    const std::array<ExpectedByte, Extent> &expected) noexcept {
  for (std::size_t index = 0; index != Extent; ++index) {
    if (integer_codec_octet(actual[index]) !=
        integer_codec_octet(expected[index])) {
      return false;
    }
  }
  return true;
}

template <class Layout, integer_codec_byte Byte>
[[nodiscard]] constexpr bool
write_canonical_descriptor_prefix(const std::span<Byte> bytes,
                                  const std::uint16_t entry_extent) noexcept {
  static_assert(Layout::magic.size() == 8);
  if (bytes.size() < 14)
    return false;
  for (std::size_t i = 0; i != Layout::magic.size(); ++i) {
    bytes[i] = static_cast<Byte>(integer_codec_octet(Layout::magic[i]));
  }
  using Codec = integer_codec<std::endian::big>;
  Codec::write(bytes.data() + 8, static_cast<std::uint16_t>(Layout::version));
  Codec::write(bytes.data() + 10,
               static_cast<std::uint16_t>(Layout::header_extent));
  Codec::write(bytes.data() + 12, entry_extent);
  return true;
}

template <class Layout, integer_codec_byte Byte, std::size_t Extent>
constexpr void
write_canonical_descriptor_prefix(std::array<Byte, Extent> &bytes,
                                  const std::uint16_t entry_extent) noexcept {
  static_assert(Extent >= 14);
  static_cast<void>(write_canonical_descriptor_prefix<Layout>(
      std::span<Byte>{bytes}, entry_extent));
}

template <class Layout, integer_codec_byte Byte>
[[nodiscard]] constexpr bool
canonical_descriptor_header_matches(const Byte *const bytes,
                                    const std::size_t extent) noexcept {
  if constexpr (Layout::header_extent < 14) {
    return false;
  } else {
    using Codec = integer_codec<std::endian::big>;
    return extent >= Layout::header_extent &&
           canonical_wire_prefix_matches(bytes, Layout::magic) &&
           Codec::template read<std::uint16_t>(bytes + 8) == Layout::version &&
           Codec::template read<std::uint16_t>(bytes + 10) ==
               Layout::header_extent &&
           Codec::template read<std::uint16_t>(bytes + 12) ==
               Layout::component_extent;
  }
}

} // namespace lpfx::detail

#endif
