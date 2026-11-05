#ifndef LPFX_DATA_FIELD_COMPONENT_HPP
#define LPFX_DATA_FIELD_COMPONENT_HPP

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include <lpfx/core/fixed_string.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/memory/host/allocator.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

enum class FieldPitchKind : std::uint8_t {
  packed = 0,
  aligned = 1,
  core_aligned = 2
};

// Row pitch policy. aligned rounds every row to Bytes; core_aligned also
// prefixes the block so logical x == 0 of every row is Bytes-aligned.
template <FieldPitchKind Kind, std::size_t Bytes = 0>
  requires((Kind == FieldPitchKind::packed) == (Bytes == 0) &&
           (Bytes == 0 || std::has_single_bit(Bytes)))
struct Pitch final {
  static constexpr FieldPitchKind kind = Kind;
  static constexpr std::size_t alignment_bytes = Bytes;
};

using PackedPitch = Pitch<FieldPitchKind::packed>;

template <std::size_t Bytes>
using AlignedPitch = Pitch<FieldPitchKind::aligned, Bytes>;

template <std::size_t Bytes>
using CoreAlignedPitch = Pitch<FieldPitchKind::core_aligned, Bytes>;

// Start alignment of a component block; 0 selects the space's data alignment
// (at least alignof(T)).
template <std::size_t Bytes = 0>
  requires(Bytes == 0 || std::has_single_bit(Bytes))
struct ComponentAlignment final {
  template <class Space, class T>
  static constexpr std::size_t alignment =
      Bytes != 0 ? Bytes
                 : std::max(AllocatorTraits<Space>::minimum_data_alignment,
                            alignof(T));
};

using DefaultComponentAlignment = ComponentAlignment<>;

namespace detail {

template <class T> inline constexpr bool is_pitch_policy = false;

template <FieldPitchKind Kind, std::size_t Bytes>
inline constexpr bool is_pitch_policy<Pitch<Kind, Bytes>> = true;

template <class T>
concept FieldPitchPolicy = is_pitch_policy<T>;

template <class T> inline constexpr bool is_alignment_policy = false;

template <std::size_t Bytes>
inline constexpr bool is_alignment_policy<ComponentAlignment<Bytes>> = true;

template <class T>
concept FieldAlignmentPolicy = is_alignment_policy<T>;

} // namespace detail

template <
    class Tag, LifetimeStartEligible T, detail::CartesianLocation Location,
    detail::FieldPitchPolicy PitchPolicy = PackedPitch,
    detail::FieldAlignmentPolicy AlignmentPolicy = DefaultComponentAlignment>
struct FieldComponent final {
  using tag_type = Tag;
  using value_type = T;
  using location_type = Location;
  using pitch_policy = PitchPolicy;
  using alignment_policy = AlignmentPolicy;
};

namespace detail {

template <class T> inline constexpr bool is_field_component = false;

template <class Tag, LifetimeStartEligible T, CartesianLocation Location,
          FieldPitchPolicy PitchPolicy, FieldAlignmentPolicy AlignmentPolicy>
inline constexpr bool is_field_component<
    FieldComponent<Tag, T, Location, PitchPolicy, AlignmentPolicy>> = true;

template <class T>
concept FieldComponentType = is_field_component<T>;

template <class Space, FieldComponentType Component>
inline constexpr std::size_t component_alignment =
    Component::alignment_policy::template alignment<
        Space, typename Component::value_type>;

template <class Space, FieldComponentType Component>
inline constexpr std::size_t component_start_alignment =
    Component::pitch_policy::kind == FieldPitchKind::packed
        ? component_alignment<Space, Component>
        : Component::pitch_policy::alignment_bytes;

template <class Space, class Component>
concept FieldComponentForSpace =
    FieldComponentType<Component> &&
    array_range_start_supported<typename Component::value_type, Space> &&
    component_alignment<Space, Component> >=
        DefaultComponentAlignment::alignment<Space,
                                             typename Component::value_type> &&
    (Component::pitch_policy::kind == FieldPitchKind::packed ||
     (Component::pitch_policy::alignment_bytes >=
          component_alignment<Space, Component> &&
      Component::pitch_policy::alignment_bytes %
              sizeof(typename Component::value_type) ==
          0));

// A non-empty component pack with distinct tags.
template <class... Components>
concept FieldTags = sizeof...(Components) > 0 &&
                    (FieldComponentType<Components> && ...) &&
                    meta::all_distinct<typename Components::tag_type...>;

template <class Space, class... Components>
concept FieldPack = FieldTags<Components...> &&
                    (FieldComponentForSpace<Space, Components> && ...);

// Components are addressed by their own type or by their tag.
template <class Key, class... Components>
concept FieldPackKey = meta::contains<Key, Components...> ||
                       meta::contains<Key, typename Components::tag_type...>;

template <class Key, class... Components>
[[nodiscard]] consteval std::size_t find_field_pack_index() noexcept {
  if constexpr (meta::contains<Key, Components...>) {
    return meta::index_of<Key, Components...>;
  } else {
    return meta::index_of<Key, typename Components::tag_type...>;
  }
}

template <class Key, class... Components>
  requires FieldPackKey<Key, Components...>
inline constexpr std::size_t field_pack_index =
    find_field_pack_index<Key, Components...>();

} // namespace detail

namespace field {

using cell = lpfx::Cell;
using node = lpfx::Node;
using face_x = lpfx::FaceX;
using face_y = lpfx::FaceY;
using face_z = lpfx::FaceZ;
using edge_x = lpfx::EdgeX;
using edge_y = lpfx::EdgeY;
using edge_z = lpfx::EdgeZ;

// The name is normalized to a scalar-packed tag so CUDA host stubs can spell
// the component type.
template <fixed_string Name, LifetimeStartEligible T,
          lpfx::detail::CartesianLocation Location = cell,
          lpfx::detail::FieldPitchPolicy PitchPolicy = PackedPitch,
          lpfx::detail::FieldAlignmentPolicy AlignmentPolicy =
              DefaultComponentAlignment>
  requires(detail::valid_name<Name>())
using component = FieldComponent<detail::name_tag_t<Name>, T, Location,
                                 PitchPolicy, AlignmentPolicy>;

namespace detail {

template <class T>
concept Component = lpfx::detail::FieldComponentType<std::remove_cv_t<T>>;

} // namespace detail

template <detail::Component... Components>
  requires lpfx::detail::FieldTags<Components...>
struct schema final {
  using component_tuple = std::tuple<Components...>;
  using component_list = meta::type_list<Components...>;
  static constexpr std::size_t component_count = sizeof...(Components);

  template <std::size_t Index>
  using component = std::tuple_element_t<Index, component_tuple>;

  template <class Component>
  static constexpr bool contains = meta::contains<Component, Components...>;

  template <class Component>
    requires contains<Component>
  static constexpr std::size_t index = meta::index_of<Component, Components...>;
};

namespace detail {

template <class T> inline constexpr bool is_schema = false;

template <Component... Components>
inline constexpr bool is_schema<schema<Components...>> = true;

template <class T>
concept Schema = is_schema<std::remove_cv_t<T>>;

} // namespace detail

} // namespace field

} // namespace lpfx

#endif
