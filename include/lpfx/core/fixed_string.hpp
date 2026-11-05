#ifndef LPFX_CORE_FIXED_STRING_HPP
#define LPFX_CORE_FIXED_STRING_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace lpfx::field {

// Structural compile-time string used as a stable component or operation name.
// The terminal null is retained in the object representation but excluded from
// size() and view(). Embedded nulls are rejected by named facade types.
template <std::size_t Extent> struct fixed_string final {
  static_assert(Extent > 0);

  char characters[Extent]{};

  constexpr fixed_string(const char (&text)[Extent]) noexcept {
    for (std::size_t index = 0; index != Extent; ++index) {
      characters[index] = text[index];
    }
  }

  [[nodiscard]] constexpr const char *data() const noexcept {
    return characters;
  }

  [[nodiscard]] static constexpr std::size_t size() noexcept {
    return Extent - 1;
  }

  [[nodiscard]] static constexpr bool empty() noexcept { return size() == 0; }

  [[nodiscard]] constexpr char operator[](const std::size_t index) const noexcept {
    return characters[index];
  }

  [[nodiscard]] constexpr std::string_view view() const noexcept {
    return {characters, size()};
  }

  template <std::size_t OtherExtent>
  [[nodiscard]] friend constexpr bool
  operator==(const fixed_string &left,
             const fixed_string<OtherExtent> &right) noexcept {
    if constexpr (Extent != OtherExtent) {
      return false;
    } else {
      for (std::size_t index = 0; index != Extent; ++index) {
        if (left.characters[index] != right.characters[index]) {
          return false;
        }
      }
      return true;
    }
  }
};

template <std::size_t Extent>
fixed_string(const char (&)[Extent]) -> fixed_string<Extent>;

namespace detail {

// Normalize the public string spelling to a scalar-packed type before it
// reaches kernel signatures. CUDA 12.6 host-stub generation cannot spell a
// structural NTTP containing a built-in array and emits <templ-param-object>.
// Evaluated scalar template arguments are portable and carry no runtime state.
inline constexpr std::size_t maximum_name_size = 64;

template <std::size_t Extent>
[[nodiscard]] consteval std::uint64_t
pack_name_chunk(const fixed_string<Extent> &name,
                const std::size_t chunk) noexcept {
  std::uint64_t packed = 0;
  const std::size_t first = chunk * sizeof(packed);
  for (std::size_t byte = 0;
       byte != sizeof(packed) && first + byte < name.size(); ++byte) {
    packed |= static_cast<std::uint64_t>(
                  static_cast<unsigned char>(name[first + byte]))
              << (byte * 8);
  }
  return packed;
}

template <std::size_t Size, std::uint64_t... Chunks>
  requires(Size <= maximum_name_size) &&
          (sizeof...(Chunks) * sizeof(std::uint64_t) == maximum_name_size)
struct NameTag final {
  inline static constexpr auto name = []() consteval {
    constexpr std::uint64_t chunks[]{Chunks...};
    char characters[Size + 1]{};
    for (std::size_t index = 0; index != Size; ++index) {
      characters[index] = static_cast<char>(
          (chunks[index / sizeof(std::uint64_t)] >>
           ((index % sizeof(std::uint64_t)) * 8)) &
          UINT64_C(0xff));
    }
    return fixed_string{characters};
  }();
};

template <fixed_string Name>
using name_tag_t = NameTag<
    Name.size(), pack_name_chunk(Name, 0), pack_name_chunk(Name, 1),
    pack_name_chunk(Name, 2), pack_name_chunk(Name, 3),
    pack_name_chunk(Name, 4), pack_name_chunk(Name, 5),
    pack_name_chunk(Name, 6), pack_name_chunk(Name, 7)>;

template <fixed_string Name>
[[nodiscard]] consteval bool valid_name() noexcept {
  if (Name.empty() || Name.size() > maximum_name_size ||
      Name[Name.size()] != '\0') {
    return false;
  }
  for (std::size_t index = 0; index != Name.size(); ++index) {
    if (Name[index] == '\0') {
      return false;
    }
  }
  return true;
}

} // namespace detail

} // namespace lpfx::field

#endif
