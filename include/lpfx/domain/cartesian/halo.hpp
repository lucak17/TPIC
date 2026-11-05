#ifndef LPFX_DOMAIN_CARTESIAN_HALO_HPP
#define LPFX_DOMAIN_CARTESIAN_HALO_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>

namespace lpfx {

// The user-declared constructors keep the type non-aggregate so checked
// factories cannot be bypassed by brace-list narrowing; later layout
// preparation revalidates the mutable payload.
template <detail::CartesianUnsignedWord Index> struct HaloDepth3 final {
  LPFX_HOST_DEVICE constexpr HaloDepth3() noexcept {}

  Index x_minus{};
  Index x_plus{};
  Index y_minus{};
  Index y_plus{};
  Index z_minus{};
  Index z_plus{};

  // Depth on the minus face of Axis when side < 0, else on the plus face.
  template <std::size_t Axis>
    requires(Axis < 3)
  [[nodiscard]] LPFX_HOST_DEVICE constexpr Index
  get(const std::int8_t side) const noexcept {
    if constexpr (Axis == 0) {
      return side < 0 ? x_minus : x_plus;
    } else if constexpr (Axis == 1) {
      return side < 0 ? y_minus : y_plus;
    } else {
      return side < 0 ? z_minus : z_plus;
    }
  }

  [[nodiscard]] friend constexpr bool
  operator==(const HaloDepth3 &, const HaloDepth3 &) noexcept = default;

private:
  constexpr HaloDepth3(const Index x_minus_value, const Index x_plus_value,
                       const Index y_minus_value, const Index y_plus_value,
                       const Index z_minus_value,
                       const Index z_plus_value) noexcept
      : x_minus(x_minus_value), x_plus(x_plus_value), y_minus(y_minus_value),
        y_plus(y_plus_value), z_minus(z_minus_value), z_plus(z_plus_value) {}

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianUnsignedWord Index, detail::checked_integer XMinus,
          detail::checked_integer XPlus, detail::checked_integer YMinus,
          detail::checked_integer YPlus, detail::checked_integer ZMinus,
          detail::checked_integer ZPlus>
[[nodiscard]] constexpr ErrorCode
try_make_halo_depth3(const XMinus x_minus, const XPlus x_plus,
                     const YMinus y_minus, const YPlus y_plus,
                     const ZMinus z_minus, const ZPlus z_plus,
                     HaloDepth3<Index> &output) noexcept {
  Index prepared_x_minus{};
  Index prepared_x_plus{};
  Index prepared_y_minus{};
  Index prepared_y_plus{};
  Index prepared_z_minus{};
  Index prepared_z_plus{};
  if (!detail::checked_integral_cast(x_minus, prepared_x_minus) ||
      !detail::checked_integral_cast(x_plus, prepared_x_plus) ||
      !detail::checked_integral_cast(y_minus, prepared_y_minus) ||
      !detail::checked_integral_cast(y_plus, prepared_y_plus) ||
      !detail::checked_integral_cast(z_minus, prepared_z_minus) ||
      !detail::checked_integral_cast(z_plus, prepared_z_plus)) {
    return ErrorCode::overflow;
  }

  output = detail::PrivateConstructionAccess::create<HaloDepth3<Index>>(
      prepared_x_minus, prepared_x_plus, prepared_y_minus, prepared_y_plus,
      prepared_z_minus, prepared_z_plus);
  return ErrorCode::success;
}

template <detail::CartesianUnsignedWord Index, detail::checked_integer Depth>
[[nodiscard]] constexpr ErrorCode
try_make_uniform_halo_depth3(const Depth depth,
                             HaloDepth3<Index> &output) noexcept {
  return try_make_halo_depth3(depth, depth, depth, depth, depth, depth, output);
}

template <detail::CartesianIndexPolicy Policy>
class AllocatedAxisInterval final {
public:
  using coordinate_type = typename Policy::coordinate_type;
  using count_type = typename Policy::count_type;

  constexpr AllocatedAxisInterval() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr coordinate_type
  lower() const noexcept {
    return lower_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr coordinate_type
  upper() const noexcept {
    return upper_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr count_type extent() const noexcept {
    return extent_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const AllocatedAxisInterval &,
             const AllocatedAxisInterval &) noexcept = default;

private:
  constexpr AllocatedAxisInterval(const coordinate_type lower,
                                  const coordinate_type upper,
                                  const count_type extent) noexcept
      : lower_(lower), upper_(upper), extent_(extent) {}

  coordinate_type lower_{};
  coordinate_type upper_{};
  count_type extent_{};

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

template <CartesianCoordinate Coordinate, CartesianUnsignedWord Count>
[[nodiscard]] constexpr bool
try_make_negative_coordinate(const Count magnitude,
                             Coordinate &output) noexcept {
  Coordinate positive{};
  if (checked_integral_cast(magnitude, positive)) {
    output = static_cast<Coordinate>(-positive);
    return true;
  }

  using UnsignedCoordinate = std::make_unsigned_t<Coordinate>;
  constexpr UnsignedCoordinate minimum_magnitude =
      static_cast<UnsignedCoordinate>(std::numeric_limits<Coordinate>::max()) +
      UnsignedCoordinate{1};
  Count converted_minimum_magnitude{};
  if (!checked_integral_cast(minimum_magnitude, converted_minimum_magnitude) ||
      magnitude != converted_minimum_magnitude) {
    return false;
  }
  output = std::numeric_limits<Coordinate>::min();
  return true;
}

} // namespace detail

// Allocated samples [-minus_depth, interior_samples + plus_depth).
template <detail::CartesianIndexPolicy Policy,
          detail::checked_integer InteriorSamples,
          detail::checked_integer MinusDepth, detail::checked_integer PlusDepth>
[[nodiscard]] constexpr ErrorCode try_make_allocated_axis_interval(
    const InteriorSamples interior_samples, const MinusDepth minus_depth,
    const PlusDepth plus_depth,
    AllocatedAxisInterval<Policy> &output) noexcept {
  using Coordinate = typename Policy::coordinate_type;
  using Count = typename Policy::count_type;

  Count prepared_interior{};
  if (!detail::checked_integral_cast(interior_samples, prepared_interior)) {
    return ErrorCode::overflow;
  }
  Count prepared_minus{};
  Count prepared_plus{};
  if (!detail::checked_integral_cast(minus_depth, prepared_minus) ||
      !detail::checked_integral_cast(plus_depth, prepared_plus)) {
    return ErrorCode::overflow;
  }
  if (prepared_interior == 0) {
    return ErrorCode::invalid_argument;
  }

  Count upper_count{};
  if (!detail::checked_add(prepared_interior, prepared_plus, upper_count)) {
    return ErrorCode::overflow;
  }
  Coordinate upper{};
  Coordinate lower{};
  if (!detail::checked_integral_cast(upper_count, upper) ||
      !detail::try_make_negative_coordinate(prepared_minus, lower)) {
    return ErrorCode::overflow;
  }

  Count extent{};
  if (!detail::checked_add(prepared_minus, upper_count, extent)) {
    return ErrorCode::overflow;
  }

  output = detail::PrivateConstructionAccess::create<
      AllocatedAxisInterval<Policy>>(lower, upper, extent);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
