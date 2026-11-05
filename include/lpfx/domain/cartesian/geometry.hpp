#ifndef LPFX_DOMAIN_CARTESIAN_GEOMETRY_HPP
#define LPFX_DOMAIN_CARTESIAN_GEOMETRY_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/domain/cartesian/index_policy.hpp>

namespace lpfx {

struct CartesianIndexKind final {};
struct CartesianExtentKind final {};

// Shared representation of CartesianIndex3 and CartesianExtent3; Kind keeps
// the two distinct types.
template <detail::CartesianWord T, class Kind>
  requires(!std::same_as<Kind, CartesianExtentKind> ||
           detail::CartesianUnsignedWord<T>)
class CartesianTriple3 final {
public:
  using value_type = T;

  LPFX_HOST_DEVICE constexpr CartesianTriple3() noexcept {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr T x() const noexcept { return x_; }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr T y() const noexcept { return y_; }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr T z() const noexcept { return z_; }

  template <std::size_t Axis>
    requires(Axis < 3)
  [[nodiscard]] LPFX_HOST_DEVICE constexpr T get() const noexcept {
    if constexpr (Axis == 0) {
      return x_;
    } else if constexpr (Axis == 1) {
      return y_;
    } else {
      return z_;
    }
  }

  [[nodiscard]] friend constexpr bool
  operator==(const CartesianTriple3 &,
             const CartesianTriple3 &) noexcept = default;

private:
  LPFX_HOST_DEVICE constexpr CartesianTriple3(const T x, const T y,
                                              const T z) noexcept
      : x_(x), y_(y), z_(z) {}

  T x_{};
  T y_{};
  T z_{};

  friend struct detail::PrivateConstructionAccess;
};

template <class T>
using CartesianIndex3 = CartesianTriple3<T, CartesianIndexKind>;

template <class T>
using CartesianExtent3 = CartesianTriple3<T, CartesianExtentKind>;

namespace detail {

template <class Triple, checked_integer X, checked_integer Y,
          checked_integer Z>
[[nodiscard]] constexpr ErrorCode try_make_cartesian_triple3(
    const X x, const Y y, const Z z, Triple &output) noexcept {
  using T = typename Triple::value_type;
  T prepared_x{};
  T prepared_y{};
  T prepared_z{};
  if (!checked_integral_cast(x, prepared_x) ||
      !checked_integral_cast(y, prepared_y) ||
      !checked_integral_cast(z, prepared_z)) {
    return ErrorCode::overflow;
  }
  output = PrivateConstructionAccess::create<Triple>(prepared_x, prepared_y,
                                                     prepared_z);
  return ErrorCode::success;
}

template <CartesianUnsignedWord Count>
[[nodiscard]] constexpr bool
cartesian_extent_is_nonempty(const CartesianExtent3<Count> extents) noexcept {
  return extents.x() != 0 && extents.y() != 0 && extents.z() != 0;
}

// x * y * z converted to Output; output is written only on success.
template <CartesianUnsignedWord Count, checked_integer Output>
[[nodiscard]] constexpr bool
checked_volume(const CartesianExtent3<Count> extents, Output &output) noexcept {
  Output x{};
  Output y{};
  Output z{};
  Output xy{};
  return checked_integral_cast(extents.x(), x) &&
         checked_integral_cast(extents.y(), y) &&
         checked_integral_cast(extents.z(), z) && checked_multiply(x, y, xy) &&
         checked_multiply(xy, z, output);
}

} // namespace detail

template <detail::CartesianWord T, detail::checked_integer X,
          detail::checked_integer Y, detail::checked_integer Z>
[[nodiscard]] constexpr ErrorCode
try_make_cartesian_index3(const X x, const Y y, const Z z,
                          CartesianIndex3<T> &output) noexcept {
  return detail::try_make_cartesian_triple3(x, y, z, output);
}

template <detail::CartesianUnsignedWord T, detail::checked_integer X,
          detail::checked_integer Y, detail::checked_integer Z>
[[nodiscard]] constexpr ErrorCode
try_make_cartesian_extent3(const X x, const Y y, const Z z,
                           CartesianExtent3<T> &output) noexcept {
  return detail::try_make_cartesian_triple3(x, y, z, output);
}

// Calls function(meta::index_constant<Axis>{}) for Axis = 0, 1, 2. When the
// function returns ErrorCode, stops at and returns the first failure.
LPFX_EXEC_CHECK_DISABLE
template <class Function>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr auto
for_each_axis(Function &&function) {
  if constexpr (std::same_as<
                    std::invoke_result_t<Function &, meta::index_constant<0>>,
                    ErrorCode>) {
    return meta::for_each_until_error<3>(function);
  } else {
    meta::for_each_index<3>(function);
  }
}

// Half-open [lower, upper) box with ordered endpoints.
template <detail::CartesianCoordinate Coordinate> class CartesianBox3 final {
public:
  constexpr CartesianBox3() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianIndex3<Coordinate>
  lower() const noexcept {
    return lower_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianIndex3<Coordinate>
  upper() const noexcept {
    return upper_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const CartesianBox3 &, const CartesianBox3 &) noexcept = default;

private:
  constexpr CartesianBox3(const CartesianIndex3<Coordinate> lower,
                          const CartesianIndex3<Coordinate> upper) noexcept
      : lower_(lower), upper_(upper) {}

  CartesianIndex3<Coordinate> lower_{};
  CartesianIndex3<Coordinate> upper_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianCoordinate Coordinate,
          detail::checked_integer LowerX, detail::checked_integer LowerY,
          detail::checked_integer LowerZ, detail::checked_integer UpperX,
          detail::checked_integer UpperY, detail::checked_integer UpperZ>
[[nodiscard]] constexpr ErrorCode
try_make_cartesian_box3(const LowerX lower_x, const LowerY lower_y,
                        const LowerZ lower_z, const UpperX upper_x,
                        const UpperY upper_y, const UpperZ upper_z,
                        CartesianBox3<Coordinate> &output) noexcept {
  CartesianIndex3<Coordinate> lower{};
  CartesianIndex3<Coordinate> upper{};
  LPFX_TRY(try_make_cartesian_index3(lower_x, lower_y, lower_z, lower));
  LPFX_TRY(try_make_cartesian_index3(upper_x, upper_y, upper_z, upper));
  if (lower.x() > upper.x() || lower.y() > upper.y() || lower.z() > upper.z()) {
    return ErrorCode::invalid_argument;
  }

  output = detail::PrivateConstructionAccess::create<CartesianBox3<Coordinate>>(
      lower, upper);
  return ErrorCode::success;
}

struct CartesianDirection3 final {
  std::int8_t x{};
  std::int8_t y{};
  std::int8_t z{};

  [[nodiscard]] friend constexpr bool
  operator==(const CartesianDirection3 &,
             const CartesianDirection3 &) noexcept = default;
};

[[nodiscard]] LPFX_HOST_DEVICE constexpr bool
is_cartesian_step(const std::int8_t value) noexcept {
  return value >= -1 && value <= 1;
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr bool
is_cartesian_direction(const CartesianDirection3 direction) noexcept {
  return is_cartesian_step(direction.x) && is_cartesian_step(direction.y) &&
         is_cartesian_step(direction.z);
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr bool
is_cartesian_neighbor_direction(const CartesianDirection3 direction) noexcept {
  return is_cartesian_direction(direction) &&
         (direction.x != 0 || direction.y != 0 || direction.z != 0);
}

// Hard precondition: direction is a Cartesian direction.
[[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
cartesian_direction_ordinal(const CartesianDirection3 direction) noexcept {
  return static_cast<std::uint8_t>(
      static_cast<int>(direction.x) + 1 +
      3 * (static_cast<int>(direction.y) + 1 +
           3 * (static_cast<int>(direction.z) + 1)));
}

// Hard precondition: direction is a Cartesian neighbor direction.
[[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
cartesian_neighbor_index(const CartesianDirection3 direction) noexcept {
  const std::uint8_t ordinal = cartesian_direction_ordinal(direction);
  return static_cast<std::uint8_t>(ordinal < 13 ? ordinal : ordinal - 1);
}

// Hard precondition: direction is a Cartesian direction.
[[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianDirection3
opposite_cartesian_direction(const CartesianDirection3 direction) noexcept {
  return CartesianDirection3{static_cast<std::int8_t>(-direction.x),
                             static_cast<std::int8_t>(-direction.y),
                             static_cast<std::int8_t>(-direction.z)};
}

// Canonical index of the opposite neighbor direction. Hard precondition:
// index < 26.
[[nodiscard]] LPFX_HOST_DEVICE constexpr std::size_t
opposite_cartesian_neighbor_index(const std::size_t index) noexcept {
  return std::size_t{25} - index;
}

inline constexpr std::array<CartesianDirection3, 26>
    canonical_cartesian_neighbor_directions = []() constexpr {
      std::array<CartesianDirection3, 26> directions{};
      std::uint8_t index = 0;
      for (std::int8_t z = -1; z <= 1; ++z) {
        for (std::int8_t y = -1; y <= 1; ++y) {
          for (std::int8_t x = -1; x <= 1; ++x) {
            if (x != 0 || y != 0 || z != 0) {
              directions[index] = CartesianDirection3{x, y, z};
              ++index;
            }
          }
        }
      }
      return directions;
    }();

static_assert(canonical_cartesian_neighbor_directions[0] ==
              CartesianDirection3{-1, -1, -1});
static_assert(canonical_cartesian_neighbor_directions[12] ==
              CartesianDirection3{-1, 0, 0});
static_assert(canonical_cartesian_neighbor_directions[13] ==
              CartesianDirection3{1, 0, 0});
static_assert(canonical_cartesian_neighbor_directions[25] ==
              CartesianDirection3{1, 1, 1});
static_assert([] {
  for (std::size_t index = 0;
       index != canonical_cartesian_neighbor_directions.size(); ++index) {
    if (cartesian_neighbor_index(opposite_cartesian_direction(
            canonical_cartesian_neighbor_directions[index])) !=
        opposite_cartesian_neighbor_index(index)) {
      return false;
    }
  }
  return true;
}());

} // namespace lpfx

#endif
