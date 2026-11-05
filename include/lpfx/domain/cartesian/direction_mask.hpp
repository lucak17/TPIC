#ifndef LPFX_DOMAIN_CARTESIAN_DIRECTION_MASK_HPP
#define LPFX_DOMAIN_CARTESIAN_DIRECTION_MASK_HPP

#include <bit>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/attributes.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>

namespace lpfx {

// A structural 26-bit mask in canonical Cartesian-neighbor order.  The public
// data member makes the type eligible as a C++20 non-type template parameter;
// communication contracts separately require a valid, non-empty,
// opposite-closed value.
struct CartesianNeighborMask3D final {
  std::uint32_t bits{};

  [[nodiscard]] friend constexpr bool
  operator==(const CartesianNeighborMask3D &,
             const CartesianNeighborMask3D &) noexcept = default;
};

[[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianNeighborMask3D
operator|(const CartesianNeighborMask3D left,
          const CartesianNeighborMask3D right) noexcept {
  return CartesianNeighborMask3D{left.bits | right.bits};
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianNeighborMask3D
operator&(const CartesianNeighborMask3D left,
          const CartesianNeighborMask3D right) noexcept {
  return CartesianNeighborMask3D{left.bits & right.bits};
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianNeighborMask3D
cartesian_neighbor_bit(const std::size_t canonical_index) noexcept {
  return canonical_index < std::size_t{26}
             ? CartesianNeighborMask3D{std::uint32_t{1}
                                       << static_cast<unsigned>(
                                              canonical_index)}
             : CartesianNeighborMask3D{};
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianNeighborMask3D
cartesian_neighbor_bit(const CartesianDirection3 direction) noexcept {
  return is_cartesian_neighbor_direction(direction)
             ? cartesian_neighbor_bit(cartesian_neighbor_index(direction))
             : CartesianNeighborMask3D{};
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr bool
cartesian_neighbor_mask_contains(const CartesianNeighborMask3D mask,
                                 const std::size_t canonical_index) noexcept {
  return (mask.bits & cartesian_neighbor_bit(canonical_index).bits) != 0;
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr bool
cartesian_neighbor_mask_contains(const CartesianNeighborMask3D mask,
                                 const CartesianDirection3 direction) noexcept {
  return (mask.bits & cartesian_neighbor_bit(direction).bits) != 0;
}

[[nodiscard]] constexpr std::size_t
cartesian_neighbor_mask_count(const CartesianNeighborMask3D mask) noexcept {
  return static_cast<std::size_t>(std::popcount(mask.bits));
}

namespace detail {

inline constexpr std::uint32_t cartesian_neighbor_valid_bits =
    (std::uint32_t{1} << 26U) - std::uint32_t{1};

[[nodiscard]] consteval CartesianNeighborMask3D
make_cartesian_neighbor_class_mask(const std::uint8_t nonzero_axis_count) {
  CartesianNeighborMask3D output{};
  for (std::size_t index = 0;
       index != canonical_cartesian_neighbor_directions.size(); ++index) {
    const CartesianDirection3 direction =
        canonical_cartesian_neighbor_directions[index];
    const std::uint8_t count = static_cast<std::uint8_t>(
        (direction.x != 0 ? 1U : 0U) + (direction.y != 0 ? 1U : 0U) +
        (direction.z != 0 ? 1U : 0U));
    if (count == nonzero_axis_count) {
      output.bits |= cartesian_neighbor_bit(index).bits;
    }
  }
  return output;
}

} // namespace detail

inline constexpr CartesianNeighborMask3D cartesian_no_neighbor_mask{};
inline constexpr CartesianNeighborMask3D cartesian_face_neighbor_mask =
    detail::make_cartesian_neighbor_class_mask(1);
inline constexpr CartesianNeighborMask3D cartesian_edge_neighbor_mask =
    detail::make_cartesian_neighbor_class_mask(2);
inline constexpr CartesianNeighborMask3D cartesian_corner_neighbor_mask =
    detail::make_cartesian_neighbor_class_mask(3);
inline constexpr CartesianNeighborMask3D cartesian_all_neighbor_mask{
    detail::cartesian_neighbor_valid_bits};

[[nodiscard]] constexpr bool is_opposite_closed_cartesian_neighbor_mask(
    const CartesianNeighborMask3D mask) noexcept {
  if ((mask.bits & ~detail::cartesian_neighbor_valid_bits) != 0) {
    return false;
  }
  for (std::size_t index = 0;
       index != canonical_cartesian_neighbor_directions.size(); ++index) {
    if (!cartesian_neighbor_mask_contains(mask, index)) {
      continue;
    }
    if (!cartesian_neighbor_mask_contains(
            mask, opposite_cartesian_neighbor_index(index))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool
is_valid_cartesian_neighbor_mask(const CartesianNeighborMask3D mask) noexcept {
  return mask.bits != 0 && is_opposite_closed_cartesian_neighbor_mask(mask);
}

static_assert(cartesian_neighbor_mask_count(cartesian_face_neighbor_mask) == 6);
static_assert(cartesian_neighbor_mask_count(cartesian_edge_neighbor_mask) ==
              12);
static_assert(cartesian_neighbor_mask_count(cartesian_corner_neighbor_mask) ==
              8);
static_assert((cartesian_face_neighbor_mask | cartesian_edge_neighbor_mask |
               cartesian_corner_neighbor_mask) == cartesian_all_neighbor_mask);
static_assert(is_valid_cartesian_neighbor_mask(cartesian_all_neighbor_mask));

} // namespace lpfx

#endif
