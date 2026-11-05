#ifndef LPFX_DATA_PARTICLE_SCHEMA_HPP
#define LPFX_DATA_PARTICLE_SCHEMA_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

struct Communicate final {};

template <class RebuilderTag> struct RebuildLocally final {
  using rebuilder_tag = RebuilderTag;
};

// Particle attributes share one allocation alignment so that every SoA block
// remains interchangeable with the reorder carousel.  The default preserves
// the allocator's natural alignment; this policy lets performance-sensitive
// schemas request a stronger compile-time alignment guarantee.
struct DefaultAttributeAlignment final {
  static constexpr std::size_t alignment_bytes = 0;
};

template <std::size_t Bytes>
  requires(std::has_single_bit(Bytes))
struct AttributeAlignment final {
  static constexpr std::size_t alignment_bytes = Bytes;
};

namespace detail {

template <class T> inline constexpr bool particle_wire_policy = false;
template <> inline constexpr bool particle_wire_policy<Communicate> = true;
template <class RebuilderTag>
inline constexpr bool particle_wire_policy<RebuildLocally<RebuilderTag>> =
    true;

template <class T>
concept ParticleWirePolicy = particle_wire_policy<T>;

template <class T> inline constexpr bool particle_alignment_policy = false;
template <>
inline constexpr bool particle_alignment_policy<DefaultAttributeAlignment> =
    true;
template <std::size_t Bytes>
inline constexpr bool particle_alignment_policy<AttributeAlignment<Bytes>> =
    true;

template <class T>
concept ParticleAlignmentPolicy = particle_alignment_policy<T>;

} // namespace detail

template <class Name, LifetimeStartEligible T,
          detail::ParticleWirePolicy WirePolicy = Communicate,
          detail::ParticleAlignmentPolicy AlignmentPolicy =
              DefaultAttributeAlignment>
struct Attribute final {
  using name_type = Name;
  using value_type = T;
  using wire_policy = WirePolicy;
  using alignment_policy = AlignmentPolicy;
};

namespace detail {

template <class T> inline constexpr bool particle_attribute = false;

template <class Name, class T, class WirePolicy, class AlignmentPolicy>
inline constexpr bool
    particle_attribute<Attribute<Name, T, WirePolicy, AlignmentPolicy>> =
        AlignmentPolicy::alignment_bytes == 0 ||
        AlignmentPolicy::alignment_bytes % alignof(T) == 0;

template <class T>
concept ParticleAttribute = particle_attribute<T>;

// A name is first matched against attribute names, then against complete
// attribute descriptors.
template <class Name, class... Attributes>
[[nodiscard]] consteval std::size_t particle_attribute_index() noexcept {
  if constexpr (meta::contains<Name, typename Attributes::name_type...>) {
    return meta::index_of<Name, typename Attributes::name_type...>;
  } else {
    return meta::index_of<Name, Attributes...>;
  }
}

// Attribute indices by descending element width, stable for equal widths.
template <std::size_t Count>
[[nodiscard]] consteval std::array<std::size_t, Count>
particle_reorder_sequence(const std::array<std::size_t, Count> &widths,
                          const bool reverse) noexcept {
  std::array<std::size_t, Count> order{};
  for (std::size_t index = 0; index != Count; ++index) {
    order[index] = index;
  }
  for (std::size_t index = 1; index != Count; ++index) {
    const std::size_t selected = order[index];
    std::size_t position = index;
    while (position != 0 && widths[order[position - 1]] < widths[selected]) {
      order[position] = order[position - 1];
      --position;
    }
    order[position] = selected;
  }
  if (reverse) {
    for (std::size_t index = 0; index != Count / 2; ++index) {
      std::swap(order[index], order[Count - index - 1]);
    }
  }
  return order;
}

} // namespace detail

template <detail::ParticleAttribute... Attributes>
  requires(sizeof...(Attributes) > 0) &&
          meta::all_distinct<typename Attributes::name_type...>
struct ParticleSchema final {
  static constexpr std::size_t attribute_count = sizeof...(Attributes);
  using attributes_type = std::tuple<Attributes...>;
  using values_type = std::tuple<typename Attributes::value_type...>;

  template <std::size_t I>
  using attribute = std::tuple_element_t<I, attributes_type>;
  template <std::size_t I>
  using value_type = typename attribute<I>::value_type;

  // Name is an attribute name or a complete attribute descriptor.
  template <class Name>
  static constexpr bool contains =
      meta::contains<Name, typename Attributes::name_type..., Attributes...>;

  template <class Name>
    requires contains<Name>
  static constexpr std::size_t index_of =
      detail::particle_attribute_index<Name, Attributes...>();

  static constexpr std::array<std::size_t, attribute_count> widths{
      sizeof(typename Attributes::value_type)...};
  static constexpr std::size_t widest_width =
      std::max({sizeof(typename Attributes::value_type)...});
  static constexpr std::size_t record_width =
      (sizeof(typename Attributes::value_type) + ...);
  // Largest element or requested alignment of any attribute.
  static constexpr std::size_t alignment =
      std::max({std::max(alignof(typename Attributes::value_type),
                         Attributes::alignment_policy::alignment_bytes)...});

  // Reorder sweeps alternate between these orders; see ParticleSlotCarousel.
  static constexpr std::array<std::size_t, attribute_count> forward_order =
      detail::particle_reorder_sequence(widths, false);
  static constexpr std::array<std::size_t, attribute_count> reverse_order =
      detail::particle_reorder_sequence(widths, true);
};

namespace detail {

template <class T> inline constexpr bool particle_schema = false;

template <class... Attributes>
inline constexpr bool particle_schema<ParticleSchema<Attributes...>> = true;

template <class T>
concept ParticleSchemaType = particle_schema<T>;

} // namespace detail

// Calls function(meta::index_constant<I>{}) for every attribute in index
// order. When function returns ErrorCode, stops at and returns the first
// failure.
LPFX_EXEC_CHECK_DISABLE
template <detail::ParticleSchemaType Schema, class Function>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr auto
for_each_attribute(Function &&function) {
  if constexpr (std::same_as<decltype(function(meta::index_constant<0>{})),
                             ErrorCode>) {
    return meta::for_each_until_error<Schema::attribute_count>(function);
  } else {
    meta::for_each_index<Schema::attribute_count>(function);
  }
}

} // namespace lpfx

#endif
