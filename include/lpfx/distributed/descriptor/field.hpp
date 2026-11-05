#ifndef LPFX_DISTRIBUTED_DESCRIPTOR_FIELD_HPP
#define LPFX_DISTRIBUTED_DESCRIPTOR_FIELD_HPP

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include <lpfx/core/meta.hpp>
#include <lpfx/data/field/component.hpp>
#include <lpfx/distributed/common/canonical_wire.hpp>
#include <lpfx/distributed/common/directions.hpp>
#include <lpfx/distributed/descriptor/family.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/domain/cartesian/direction_mask.hpp>

namespace lpfx {

enum class FieldFlow : std::uint8_t { outbound = 0, inbound = 1 };

// Semantic role of one piece of the fused field operation. Every execution
// target interprets the same roles in the same order.
enum class FusedFieldPieceRole : std::uint8_t {
  shared_add = 0,
  primary_copy = 1,
  supplemental_add = 2,
};

// One component entry in a field operation descriptor. The communicated
// scalar identity is explicit because native field payloads are not a
// self-describing serialization.
template <detail::FieldComponentType Component, std::uint64_t TagIdentity,
          std::uint64_t StorageScalarIdentity>
  requires(TagIdentity != 0 && StorageScalarIdentity != 0) &&
          (sizeof(typename Component::value_type) <=
           std::numeric_limits<std::uint32_t>::max())
struct FieldCopyComponent final {
  using component_type = Component;
  using tag_type = typename Component::tag_type;
  using storage_type = typename Component::value_type;
  using location_type = typename Component::location_type;

  static constexpr std::uint64_t tag_identity = TagIdentity;
  static constexpr std::uint64_t storage_scalar_identity =
      StorageScalarIdentity;
  static constexpr std::uint32_t width =
      static_cast<std::uint32_t>(sizeof(storage_type));
};

namespace detail {

template <class T> inline constexpr bool is_field_copy_component = false;
template <FieldComponentType Component, std::uint64_t TagIdentity,
          std::uint64_t StorageScalarIdentity>
inline constexpr bool is_field_copy_component<
    FieldCopyComponent<Component, TagIdentity, StorageScalarIdentity>> = true;

template <class T>
concept FieldCopyComponentType = is_field_copy_component<T>;

template <std::uint64_t... Identities>
inline constexpr bool distinct_field_identities = [] {
  constexpr std::array<std::uint64_t, sizeof...(Identities)> values{
      Identities...};
  for (std::size_t i = 0; i != values.size(); ++i) {
    for (std::size_t j = i + 1; j != values.size(); ++j) {
      if (values[i] == values[j]) {
        return false;
      }
    }
  }
  return true;
}();

inline constexpr std::size_t fused_field_maximum_pieces_per_component = 8;
inline constexpr std::size_t fused_field_maximum_component_count =
    std::numeric_limits<std::uint16_t>::max() /
    fused_field_maximum_pieces_per_component;

// A shared sample moves only along the node axes of its location.
template <CartesianLocation Location>
[[nodiscard]] constexpr bool
field_shared_direction_is_active(const CartesianDirection3 direction) noexcept {
  return (direction.x == 0 || Location::x_is_node) &&
         (direction.y == 0 || Location::y_is_node) &&
         (direction.z == 0 || Location::z_is_node);
}

// A supplemental route: a publication direction plus a tangential shared
// direction whose sum is a Cartesian neighbor direction.
struct FusedFieldRoute final {
  std::uint8_t publication;
  std::uint8_t shared;
  std::uint8_t route;
};

template <class Visit>
constexpr void for_each_fused_field_composition(Visit &&visit) noexcept {
  for (std::size_t p = 0; p != 26; ++p) {
    for (std::size_t s = 0; s != 26; ++s) {
      const CartesianDirection3 publication =
          canonical_cartesian_neighbor_directions[p];
      const CartesianDirection3 shared =
          canonical_cartesian_neighbor_directions[s];
      const CartesianDirection3 route{
          static_cast<std::int8_t>(publication.x + shared.x),
          static_cast<std::int8_t>(publication.y + shared.y),
          static_cast<std::int8_t>(publication.z + shared.z)};
      if ((publication.x == 0 || shared.x == 0) &&
          (publication.y == 0 || shared.y == 0) &&
          (publication.z == 0 || shared.z == 0) &&
          is_cartesian_neighbor_direction(route)) {
        visit(p, s, cartesian_neighbor_index(route));
      }
    }
  }
}

// Every supplemental route in canonical (publication, shared) order.
inline constexpr auto fused_field_routes = [] {
  constexpr std::size_t count = [] {
    std::size_t n = 0;
    for_each_fused_field_composition(
        [&](std::size_t, std::size_t, std::size_t) constexpr noexcept { ++n; });
    return n;
  }();
  std::array<FusedFieldRoute, count> routes{};
  std::size_t n = 0;
  for_each_fused_field_composition(
      [&](const std::size_t p, const std::size_t s,
          const std::size_t r) constexpr noexcept {
        routes[n++] = {static_cast<std::uint8_t>(p),
                       static_cast<std::uint8_t>(s),
                       static_cast<std::uint8_t>(r)};
      });
  return routes;
}();

// Inbound pieces one fused component may receive from each direction.
template <CartesianNeighborMask3D PublicationMask, CartesianLocation Location>
inline constexpr auto fused_field_location_piece_capacities = [] {
  std::array<std::size_t, 26> capacities{};
  for (std::size_t d = 0; d != 26; ++d) {
    capacities[d] = cartesian_neighbor_mask_contains(PublicationMask, d) +
                    field_shared_direction_is_active<Location>(
                        canonical_cartesian_neighbor_directions[d]);
  }
  for (const FusedFieldRoute route : fused_field_routes) {
    if (cartesian_neighbor_mask_contains(PublicationMask, route.publication) &&
        field_shared_direction_is_active<Location>(
            canonical_cartesian_neighbor_directions[route.shared])) {
      ++capacities[route.route];
    }
  }
  return capacities;
}();

} // namespace detail
} // namespace lpfx

namespace lpfx::distributed {

enum class field_operation_kind : std::uint8_t {
  copy_to_ghost = 1,
  ghost_to_owner_add = 2,
  shared_sample_add = 3,
  shared_sample_add_copy_to_ghost = 4,
};

// The sole semantic field-operation mapping. Directions is the selected
// operation mask for affine operations and the publication mask for the fused
// operation; every derived mask and wire fact is read from this empty type.
template <field_operation_kind Kind, CartesianNeighborMask3D Directions,
          lpfx::detail::FieldCopyComponentType... WireEntries>
  requires(Kind >= field_operation_kind::copy_to_ghost &&
           Kind <= field_operation_kind::shared_sample_add_copy_to_ghost) &&
          (sizeof...(WireEntries) > 0) &&
          (sizeof...(WireEntries) <= std::numeric_limits<std::uint16_t>::max()) &&
          (is_valid_cartesian_neighbor_mask(Directions)) &&
          meta::all_distinct<typename WireEntries::tag_type...> &&
          lpfx::detail::distinct_field_identities<WireEntries::tag_identity...> &&
          (Kind == field_operation_kind::copy_to_ghost ||
           (std::floating_point<typename WireEntries::storage_type> && ...)) &&
          (Kind != field_operation_kind::shared_sample_add_copy_to_ghost ||
           sizeof...(WireEntries) <=
               lpfx::detail::fused_field_maximum_component_count)
struct FieldOperationSpec final {
  static constexpr operation_family family = operation_family::field;
  static constexpr field_operation_kind kind = Kind;
  static constexpr bool additive = Kind != field_operation_kind::copy_to_ghost;
  static constexpr bool fused =
      Kind == field_operation_kind::shared_sample_add_copy_to_ghost;

  using component_tuple = std::tuple<WireEntries...>;
  using field_components = std::tuple<typename WireEntries::component_type...>;
  static constexpr std::size_t component_count = sizeof...(WireEntries);

  template <std::size_t Index>
  using component_type = std::tuple_element_t<Index, component_tuple>;

  static constexpr std::array<std::uint32_t, component_count> component_widths{
      WireEntries::width...};

  static constexpr CartesianNeighborMask3D publication_mask = Directions;

  // Inbound pieces each canonical direction may carry.
  static constexpr std::array<std::size_t, 26> inbound_capacities = [] {
    std::array<std::size_t, 26> capacities{};
    for (std::size_t d = 0; d != 26; ++d) {
      if constexpr (fused) {
        capacities[d] = (lpfx::detail::fused_field_location_piece_capacities<
                             Directions, typename WireEntries::location_type>[d] +
                         ...);
      } else {
        capacities[d] = cartesian_neighbor_mask_contains(Directions, d)
                            ? component_count
                            : 0;
      }
    }
    return capacities;
  }();

  static constexpr CartesianNeighborMask3D transport_mask = [] {
    CartesianNeighborMask3D mask{};
    for (std::size_t d = 0; d != 26; ++d) {
      if (inbound_capacities[d] != 0) {
        mask.bits |= cartesian_neighbor_bit(d).bits;
      }
    }
    return mask;
  }();

  static constexpr std::size_t maximum_active_piece_count = [] {
    std::size_t count = 0;
    for (const std::size_t capacity : inbound_capacities) {
      count += capacity;
    }
    return count;
  }();

  static constexpr std::array<std::uint8_t, 8> descriptor_magic = [] {
    if constexpr (Kind == field_operation_kind::copy_to_ghost) {
      return std::array<std::uint8_t, 8>{0x4c, 0x50, 0x46, 0x78,
                                         0x46, 0x43, 0x47, 0x00};
    } else if constexpr (Kind == field_operation_kind::ghost_to_owner_add) {
      return std::array<std::uint8_t, 8>{0x4c, 0x50, 0x46, 0x78,
                                         0x47, 0x41, 0x44, 0x44};
    } else if constexpr (Kind == field_operation_kind::shared_sample_add) {
      return std::array<std::uint8_t, 8>{0x4c, 0x50, 0x46, 0x78,
                                         0x53, 0x41, 0x44, 0x44};
    } else {
      return std::array<std::uint8_t, 8>{0x4c, 0x50, 0x46, 0x78,
                                         0x53, 0x41, 0x43, 0x47};
    }
  }();
  static constexpr std::uint16_t descriptor_version = fused ? 2 : 3;
  static constexpr auto descriptor_masks = [] {
    if constexpr (fused) {
      return std::array<CartesianNeighborMask3D, 2>{publication_mask,
                                                    transport_mask};
    } else {
      return std::array<CartesianNeighborMask3D, 1>{transport_mask};
    }
  }();
};

namespace detail {

template <class T> inline constexpr bool is_field_operation_spec = false;
template <field_operation_kind Kind, CartesianNeighborMask3D Directions,
          class... WireEntries>
inline constexpr bool
    is_field_operation_spec<FieldOperationSpec<Kind, Directions, WireEntries...>> =
        true;

template <class Spec>
concept FieldOperationSpecType =
    is_field_operation_spec<std::remove_cvref_t<Spec>>;

} // namespace detail

template <lpfx::detail::FieldCopyComponentType... WireEntries>
using CopyToGhost = FieldOperationSpec<field_operation_kind::copy_to_ghost,
                                       all_directions_3d, WireEntries...>;

} // namespace lpfx::distributed

namespace lpfx {

template <distributed::detail::FieldOperationSpecType Spec>
struct FieldOperationDescriptorLayout final {
  static constexpr std::array<std::uint8_t, 8> magic = Spec::descriptor_magic;
  static constexpr std::uint16_t version = Spec::descriptor_version;
  static constexpr std::uint16_t header_extent =
      static_cast<std::uint16_t>(21U + 4U * Spec::descriptor_masks.size());
  static constexpr std::uint16_t component_extent = 21;
  static constexpr std::size_t extent =
      header_extent + Spec::component_count * std::size_t{component_extent};
};

// The rank-global invariant descriptor of one field operation.
template <distributed::detail::FieldOperationSpecType Spec>
[[nodiscard]] constexpr auto make_canonical_field_operation_descriptor() noexcept {
  using Layout = FieldOperationDescriptorLayout<Spec>;
  using Codec = detail::integer_codec<std::endian::big>;
  static_assert(std::endian::native == std::endian::little ||
                    std::endian::native == std::endian::big,
                "LPFx field communication requires a little- or big-endian host");
  std::array<std::byte, Layout::extent> bytes{};
  detail::write_canonical_descriptor_prefix<Layout>(bytes,
                                                    Layout::component_extent);
  Codec::write(bytes.data() + 14,
               static_cast<std::uint16_t>(Spec::component_count));
  Codec::write(bytes.data() + 16, static_cast<std::uint32_t>(Layout::extent));
  bytes[20] = static_cast<std::byte>(
      std::endian::native == std::endian::little ? 1U : 2U);
  for (std::size_t mask = 0; mask != Spec::descriptor_masks.size(); ++mask) {
    Codec::write(bytes.data() + 21 + 4 * mask, Spec::descriptor_masks[mask].bits);
  }
  meta::for_each_index<Spec::component_count>([&](const auto index) constexpr noexcept {
    constexpr std::size_t I = decltype(index)::value;
    using Entry = typename Spec::template component_type<I>;
    using Location = typename Entry::location_type;
    std::byte *const base =
        bytes.data() + Layout::header_extent + I * Layout::component_extent;
    Codec::write(base, Entry::tag_identity);
    Codec::write(base + 8, Entry::storage_scalar_identity);
    Codec::write(base + 16, Entry::width);
    base[20] = static_cast<std::byte>((Location::x_is_node ? 1U : 0U) |
                                      (Location::y_is_node ? 2U : 0U) |
                                      (Location::z_is_node ? 4U : 0U));
  });
  return bytes;
}

} // namespace lpfx

#endif
