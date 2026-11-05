#ifndef LPFX_ALGORITHM_PARTICLE_KEY_HPP
#define LPFX_ALGORITHM_PARTICLE_KEY_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/particle/schema.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

namespace detail {

template <class T>
concept ParticleSortUnsignedIndex =
    checked_integer<T> && std::is_unsigned_v<T> && LifetimeStartEligible<T>;

} // namespace detail

template <detail::ParticleSortUnsignedIndex Count,
          detail::ParticleSortUnsignedIndex Offset,
          detail::ParticleSortUnsignedIndex Cursor,
          detail::ParticleSortUnsignedIndex MapIndex>
struct BasicParticleSortIndexPolicy final {
  using count_type = Count;
  using offset_type = Offset;
  using cursor_type = Cursor;
  using map_index_type = MapIndex;
};

// Names the integer particle attribute that holds each record's cell key.
template <class Name> struct ParticleAttributeKey final {
  using name_type = Name;
};

namespace detail {

template <class Policy>
concept ParticleSortIndexPolicy = requires {
  requires ParticleSortUnsignedIndex<typename Policy::count_type>;
  requires ParticleSortUnsignedIndex<typename Policy::offset_type>;
  requires ParticleSortUnsignedIndex<typename Policy::cursor_type>;
  requires ParticleSortUnsignedIndex<typename Policy::map_index_type>;
};

template <class T> inline constexpr bool particle_attribute_key = false;

template <class Name>
inline constexpr bool particle_attribute_key<ParticleAttributeKey<Name>> =
    true;

template <class T>
concept ParticleKeyAccessor = particle_attribute_key<T>;

// A class trait, not a direct alias: nvcc fails explicit-argument
// substitution of signatures that spell the key type through a dependent
// variable template while Schema is still being deduced.
template <class Schema, class KeyAccessor> struct ParticleKeyTraits final {
  static constexpr std::size_t index =
      Schema::template index_of<typename KeyAccessor::name_type>;
  using type = typename Schema::template value_type<index>;
};

template <class Schema, class KeyAccessor>
inline constexpr std::size_t particle_key_index =
    ParticleKeyTraits<Schema, KeyAccessor>::index;

template <class Schema, class KeyAccessor>
using particle_key_t = typename ParticleKeyTraits<Schema, KeyAccessor>::type;

template <class Schema, class KeyAccessor>
concept ParticleDirectCellKeyFor =
    ParticleSchemaType<Schema> && ParticleKeyAccessor<KeyAccessor> &&
    Schema::template contains<typename KeyAccessor::name_type> &&
    checked_integer<particle_key_t<Schema, KeyAccessor>>;

// The direct-cell key range contract of the device histogram (try_bin below
// accepts the same set): key is a valid bin when it is nonnegative and below
// bin_count, without a representability assumption on the key type.
template <checked_integer Key, std::unsigned_integral Bin>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool
try_direct_cell_bin_index(const Key key, const Bin bin_count,
                          Bin &output) noexcept {
  if constexpr (std::is_signed_v<Key>) {
    if (key < Key{0}) {
      return false;
    }
  }
  using UnsignedKey = std::make_unsigned_t<Key>;
  const UnsignedKey candidate = static_cast<UnsignedKey>(key);
  if constexpr (std::numeric_limits<UnsignedKey>::digits >
                std::numeric_limits<Bin>::digits) {
    if (candidate > static_cast<UnsignedKey>(std::numeric_limits<Bin>::max())) {
      return false;
    }
  }
  const Bin bin = static_cast<Bin>(candidate);
  if (bin >= bin_count) {
    return false;
  }
  output = bin;
  return true;
}

} // namespace detail

template <detail::checked_integer Key> class DirectCellBinLayout final {
public:
  using key_type = Key;

  constexpr DirectCellBinLayout() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return bin_count_ != 0;
  }
  [[nodiscard]] constexpr std::size_t bin_count() const noexcept {
    return bin_count_;
  }

  // Same accept set as detail::try_direct_cell_bin_index with one unsigned
  // compare: a negative key's unsigned image is at least max(Key) + 1, so it
  // fails the same bound as every key >= bin_count. The bound only depends on
  // bin_count_, so a loop over a local layout hoists it.
  [[nodiscard]] LPFX_ALWAYS_INLINE constexpr bool
  try_bin(const Key key, std::size_t &output) const noexcept {
    using UnsignedKey = std::make_unsigned_t<Key>;
    constexpr UnsignedKey key_max =
        static_cast<UnsignedKey>(std::numeric_limits<Key>::max());
    const UnsignedKey candidate = static_cast<UnsignedKey>(key);
    if constexpr (std::numeric_limits<UnsignedKey>::digits >
                  std::numeric_limits<std::size_t>::digits) {
      if (candidate > key_max ||
          candidate >= static_cast<UnsignedKey>(bin_count_)) {
        return false;
      }
    } else if constexpr (key_max >= std::numeric_limits<std::size_t>::max()) {
      if (static_cast<std::size_t>(candidate) >= bin_count_) {
        return false;
      }
    } else {
      constexpr std::size_t key_limit = static_cast<std::size_t>(key_max) + 1;
      const std::size_t limit = bin_count_ < key_limit ? bin_count_ : key_limit;
      if (static_cast<std::size_t>(candidate) >= limit) {
        return false;
      }
    }
    output = static_cast<std::size_t>(candidate);
    return true;
  }

  // Hard precondition: try_bin(key, ...) succeeded for this unchanged layout.
  [[nodiscard]] constexpr std::size_t
  unchecked_bin(const Key key) const noexcept {
    return static_cast<std::size_t>(key);
  }

private:
  constexpr explicit DirectCellBinLayout(const std::size_t bin_count) noexcept
      : bin_count_(bin_count) {}

  std::size_t bin_count_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::checked_integer Key, detail::checked_integer BinCount>
[[nodiscard]] constexpr bool
try_make_direct_cell_bin_layout(const BinCount requested_bin_count,
                                DirectCellBinLayout<Key> &output) noexcept {
  std::size_t bin_count{};
  if (!detail::checked_integral_cast(requested_bin_count, bin_count) ||
      bin_count == 0) {
    return false;
  }
  output = detail::PrivateConstructionAccess::create<DirectCellBinLayout<Key>>(
      bin_count);
  return true;
}

} // namespace lpfx

#endif
