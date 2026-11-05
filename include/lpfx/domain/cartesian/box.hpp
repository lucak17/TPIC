#ifndef LPFX_DOMAIN_CARTESIAN_BOX_HPP
#define LPFX_DOMAIN_CARTESIAN_BOX_HPP

#include <limits>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>

namespace lpfx::detail {

// CartesianBox3 guarantees ordered endpoints. Converting both endpoints to
// the corresponding unsigned type before subtraction also covers boxes that
// cross zero without overflowing the signed coordinate representation.
template <CartesianUnsignedWord Count, CartesianCoordinate Coordinate>
[[nodiscard]] constexpr ErrorCode
try_cartesian_box_extents(const CartesianBox3<Coordinate> box,
                          CartesianExtent3<Count> &output) noexcept {
  const auto axis_extent = [](const Coordinate lower, const Coordinate upper,
                              Count &extent) constexpr noexcept {
    using UnsignedCoordinate = std::make_unsigned_t<Coordinate>;
    return checked_integral_cast(static_cast<UnsignedCoordinate>(upper) -
                                     static_cast<UnsignedCoordinate>(lower),
                                 extent);
  };

  const auto lower = box.lower();
  const auto upper = box.upper();
  Count x{};
  Count y{};
  Count z{};
  if (!axis_extent(lower.x(), upper.x(), x) ||
      !axis_extent(lower.y(), upper.y(), y) ||
      !axis_extent(lower.z(), upper.z(), z)) {
    return ErrorCode::overflow;
  }
  return try_make_cartesian_extent3(x, y, z, output);
}

template <CartesianUnsignedWord Count>
[[nodiscard]] constexpr ErrorCode
try_cartesian_extent_count(const Count x, const Count y, const Count z,
                           Count &output) noexcept {
  Count xy{};
  return checked_multiply(x, y, xy) && checked_multiply(xy, z, output)
             ? ErrorCode::success
             : ErrorCode::overflow;
}

template <CartesianCoordinate Coordinate, CartesianUnsignedWord Count>
struct CartesianBoxPairAxis final {
  Coordinate first_lower{};
  Coordinate first_upper{};
  Coordinate second_lower{};
  Coordinate second_upper{};
  Count count{};
};

template <CartesianCoordinate Coordinate, CartesianUnsignedWord Count>
[[nodiscard]] constexpr ErrorCode
try_make_cartesian_box_pair(const CartesianBoxPairAxis<Coordinate, Count> x,
                            const CartesianBoxPairAxis<Coordinate, Count> y,
                            const CartesianBoxPairAxis<Coordinate, Count> z,
                            CartesianBox3<Coordinate> &first,
                            CartesianBox3<Coordinate> &second,
                            Count &count) noexcept {
  Count prepared_count{};
  LPFX_TRY(try_cartesian_extent_count(x.count, y.count, z.count,
                                      prepared_count));
  CartesianBox3<Coordinate> prepared_first{};
  LPFX_TRY(try_make_cartesian_box3(x.first_lower, y.first_lower, z.first_lower,
                                   x.first_upper, y.first_upper, z.first_upper,
                                   prepared_first));
  CartesianBox3<Coordinate> prepared_second{};
  LPFX_TRY(try_make_cartesian_box3(
      x.second_lower, y.second_lower, z.second_lower, x.second_upper,
      y.second_upper, z.second_upper, prepared_second));

  first = prepared_first;
  second = prepared_second;
  count = prepared_count;
  return ErrorCode::success;
}

template <CartesianUnsignedWord Count, CartesianCoordinate Coordinate>
[[nodiscard]] constexpr ErrorCode
try_cartesian_box_shape(const CartesianBox3<Coordinate> box,
                        CartesianExtent3<Count> &extents,
                        Count &count) noexcept {
  CartesianExtent3<Count> prepared_extents{};
  LPFX_TRY(try_cartesian_box_extents(box, prepared_extents));
  Count prepared_count{};
  if (!checked_volume(prepared_extents, prepared_count)) {
    return ErrorCode::overflow;
  }
  extents = prepared_extents;
  count = prepared_count;
  return ErrorCode::success;
}

template <CartesianUnsignedWord Count, CartesianCoordinate Coordinate>
[[nodiscard]] constexpr ErrorCode
validate_cartesian_box_count(const CartesianBox3<Coordinate> box,
                             const Count expected) noexcept {
  CartesianExtent3<Count> extents{};
  Count actual{};
  LPFX_TRY(try_cartesian_box_shape(box, extents, actual));
  return actual == expected ? ErrorCode::success : ErrorCode::invalid_argument;
}

template <CartesianCoordinate Coordinate>
[[nodiscard]] constexpr bool
cartesian_box_contains(const CartesianBox3<Coordinate> outer,
                       const CartesianBox3<Coordinate> inner) noexcept {
  const auto outer_lower = outer.lower();
  const auto outer_upper = outer.upper();
  const auto inner_lower = inner.lower();
  const auto inner_upper = inner.upper();
  return inner_lower.x() >= outer_lower.x() &&
         inner_upper.x() <= outer_upper.x() &&
         inner_lower.y() >= outer_lower.y() &&
         inner_upper.y() <= outer_upper.y() &&
         inner_lower.z() >= outer_lower.z() &&
         inner_upper.z() <= outer_upper.z();
}

// Half-open boxes overlap only when every axis has a non-empty intersection.
// This definition makes face/edge/corner contact and every empty box disjoint.
template <CartesianCoordinate Coordinate>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr bool
cartesian_boxes_overlap(const CartesianBox3<Coordinate> left,
                        const CartesianBox3<Coordinate> right) noexcept {
  const auto overlap = [](const Coordinate left_lower,
                          const Coordinate left_upper,
                          const Coordinate right_lower,
                          const Coordinate right_upper) constexpr noexcept {
    // All operands are pure. Eager Boolean combination avoids cloning a
    // short-circuit branch ladder at each checked axis.
    return (left_lower < left_upper) & (right_lower < right_upper) &
           (left_lower < right_upper) & (right_lower < left_upper);
  };
  const auto left_lower = left.lower();
  const auto left_upper = left.upper();
  const auto right_lower = right.lower();
  const auto right_upper = right.upper();
  return overlap(left_lower.x(), left_upper.x(), right_lower.x(),
                 right_upper.x()) &
         overlap(left_lower.y(), left_upper.y(), right_lower.y(),
                 right_upper.y()) &
         overlap(left_lower.z(), left_upper.z(), right_lower.z(),
                 right_upper.z());
}

template <CartesianCoordinate Coordinate>
[[nodiscard]] constexpr bool cartesian_boxes_have_equal_extents(
    const CartesianBox3<Coordinate> left,
    const CartesianBox3<Coordinate> right) noexcept {
  using Extent = std::make_unsigned_t<Coordinate>;
  CartesianExtent3<Extent> left_extents{};
  CartesianExtent3<Extent> right_extents{};
  return try_cartesian_box_extents(left, left_extents) == ErrorCode::success &&
         try_cartesian_box_extents(right, right_extents) ==
             ErrorCode::success &&
         left_extents == right_extents;
}

// An empty intersection succeeds with an empty box and a zero count.
template <CartesianUnsignedWord Count, CartesianCoordinate Coordinate>
[[nodiscard]] constexpr ErrorCode try_intersect_cartesian_boxes(
    const CartesianBox3<Coordinate> left, const CartesianBox3<Coordinate> right,
    CartesianBox3<Coordinate> &output, Count &count) noexcept {
  const auto intersect =
      [](const Coordinate left_lower, const Coordinate left_upper,
         const Coordinate right_lower, const Coordinate right_upper,
         Coordinate &lower, Coordinate &upper) constexpr noexcept {
        lower = left_lower < right_lower ? right_lower : left_lower;
        upper = left_upper < right_upper ? left_upper : right_upper;
        return lower < upper;
      };
  const auto left_lower = left.lower();
  const auto left_upper = left.upper();
  const auto right_lower = right.lower();
  const auto right_upper = right.upper();
  Coordinate lower_x{};
  Coordinate lower_y{};
  Coordinate lower_z{};
  Coordinate upper_x{};
  Coordinate upper_y{};
  Coordinate upper_z{};
  if (!intersect(left_lower.x(), left_upper.x(), right_lower.x(),
                 right_upper.x(), lower_x, upper_x) ||
      !intersect(left_lower.y(), left_upper.y(), right_lower.y(),
                 right_upper.y(), lower_y, upper_y) ||
      !intersect(left_lower.z(), left_upper.z(), right_lower.z(),
                 right_upper.z(), lower_z, upper_z)) {
    output = CartesianBox3<Coordinate>{};
    count = Count{0};
    return ErrorCode::success;
  }
  CartesianBox3<Coordinate> intersection{};
  LPFX_TRY(try_make_cartesian_box3(lower_x, lower_y, lower_z, upper_x, upper_y,
                                   upper_z, intersection));
  Count intersection_count{};
  CartesianExtent3<Count> intersection_extents{};
  LPFX_TRY(try_cartesian_box_shape(intersection, intersection_extents,
                                   intersection_count));
  output = intersection;
  count = intersection_count;
  return ErrorCode::success;
}

// Moves box by target_origin - source_origin; overflow when an endpoint
// leaves the Coordinate range.
template <CartesianCoordinate Coordinate>
[[nodiscard]] constexpr ErrorCode
try_translate_cartesian_box(const CartesianBox3<Coordinate> box,
                            const CartesianIndex3<Coordinate> source_origin,
                            const CartesianIndex3<Coordinate> target_origin,
                            CartesianBox3<Coordinate> &output) noexcept {
  const auto translate = [](const Coordinate value, const Coordinate source,
                            const Coordinate target,
                            Coordinate &translated) constexpr noexcept {
    using Unsigned = std::make_unsigned_t<Coordinate>;
    constexpr Unsigned sign_bit =
        Unsigned{1} << (std::numeric_limits<Unsigned>::digits - 1);
    const auto rank = [](const Coordinate coordinate) constexpr noexcept {
      return static_cast<Unsigned>(coordinate) ^ sign_bit;
    };

    const Unsigned value_rank = rank(value);
    const Unsigned source_rank = rank(source);
    const Unsigned target_rank = rank(target);
    Unsigned translated_rank{};
    if (value_rank >= source_rank) {
      if (!checked_add(target_rank, value_rank - source_rank,
                       translated_rank)) {
        return false;
      }
    } else if (!checked_subtract(target_rank, source_rank - value_rank,
                                 translated_rank)) {
      return false;
    }

    if (translated_rank >= sign_bit) {
      translated = static_cast<Coordinate>(translated_rank - sign_bit);
    } else {
      translated =
          static_cast<Coordinate>(std::numeric_limits<Coordinate>::min() +
                                  static_cast<Coordinate>(translated_rank));
    }
    return true;
  };

  const auto lower = box.lower();
  const auto upper = box.upper();
  Coordinate lower_x{};
  Coordinate lower_y{};
  Coordinate lower_z{};
  Coordinate upper_x{};
  Coordinate upper_y{};
  Coordinate upper_z{};
  if (!translate(lower.x(), source_origin.x(), target_origin.x(), lower_x) ||
      !translate(lower.y(), source_origin.y(), target_origin.y(), lower_y) ||
      !translate(lower.z(), source_origin.z(), target_origin.z(), lower_z) ||
      !translate(upper.x(), source_origin.x(), target_origin.x(), upper_x) ||
      !translate(upper.y(), source_origin.y(), target_origin.y(), upper_y) ||
      !translate(upper.z(), source_origin.z(), target_origin.z(), upper_z)) {
    return ErrorCode::overflow;
  }
  return try_make_cartesian_box3(lower_x, lower_y, lower_z, upper_x, upper_y,
                                 upper_z, output);
}

} // namespace lpfx::detail

#endif
