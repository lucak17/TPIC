#ifndef LPFX_ALGORITHM_COUPLING_TRILINEAR_HPP
#define LPFX_ALGORITHM_COUPLING_TRILINEAR_HPP

#include <concepts>
#include <cstddef>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/domain/cartesian/location.hpp>

namespace lpfx {

namespace detail {

template <class Component, class Location, class IndexPolicy,
          class Accumulator>
concept TrilinearSampleComponent =
    requires(const Component component,
             const typename IndexPolicy::coordinate_type i,
             const typename IndexPolicy::coordinate_type j,
             const typename IndexPolicy::coordinate_type k) {
      typename Component::location_type;
      typename Component::index_policy;
      requires std::same_as<typename Component::location_type, Location>;
      requires std::same_as<typename Component::index_policy, IndexPolicy>;
      { static_cast<Accumulator>(component(i, j, k)) };
    };

template <class Coordinate, std::floating_point Weight>
struct TrilinearAxisCoordinates final {
  Coordinate base{};
  Weight fraction{};
};

// Hard precondition: shifted and its enclosing pair of integer coordinates are
// finite and representable by Coordinate. Conversion truncates toward zero;
// the single correction supplies floor semantics for negative coordinates.
template <class Coordinate, std::floating_point Weight>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr
    TrilinearAxisCoordinates<Coordinate, Weight>
    make_trilinear_axis_coordinates(const Weight logical_coordinate,
                                    const bool sample_is_node) noexcept {
  const Weight shifted =
      logical_coordinate - (sample_is_node ? Weight{0} : Weight{0.5});
  Coordinate base = static_cast<Coordinate>(shifted);
  base = static_cast<Coordinate>(
      base - static_cast<Coordinate>(static_cast<Weight>(base) > shifted));
  return {base, shifted - static_cast<Weight>(base)};
}

template <std::floating_point Accumulator>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
trilinear_lerp(const Accumulator lower, const Accumulator upper,
               const Accumulator fraction) noexcept {
  return lower + (upper - lower) * fraction;
}

template <std::size_t Corner, std::floating_point Accumulator, class Component,
          class Stencil>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
trilinear_corner(const Component &component, const Stencil &stencil) noexcept {
  return static_cast<Accumulator>(stencil.template at<Corner>(component));
}

} // namespace detail

// Logical coordinates use the cell geometry convention: nodes lie at integer
// coordinates and cell centers at half-integers. Location selects that offset
// independently on each axis. The object retains only the lower sample index
// and fractional coordinate for each axis; no corner-weight array is stored.
template <detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy,
          std::floating_point Weight>
class TrilinearStencil3D final {
public:
  using location_type = Location;
  using index_policy = IndexPolicy;
  using coordinate_type = typename IndexPolicy::coordinate_type;
  using weight_type = Weight;

  LPFX_HOST_DEVICE constexpr TrilinearStencil3D() noexcept {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr coordinate_type
  base_x() const noexcept {
    return base_x_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr coordinate_type
  base_y() const noexcept {
    return base_y_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr coordinate_type
  base_z() const noexcept {
    return base_z_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Weight fraction_x() const noexcept {
    return fraction_x_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Weight fraction_y() const noexcept {
    return fraction_y_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr Weight fraction_z() const noexcept {
    return fraction_z_;
  }

  template <std::size_t Corner>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr coordinate_type
  i() const noexcept {
    static_assert(Corner < 8);
    return static_cast<coordinate_type>(
        base_x_ + static_cast<coordinate_type>(Corner & 1U));
  }

  template <std::size_t Corner>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr coordinate_type
  j() const noexcept {
    static_assert(Corner < 8);
    return static_cast<coordinate_type>(
        base_y_ +
        static_cast<coordinate_type>((Corner >> 1U) & 1U));
  }

  template <std::size_t Corner>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr coordinate_type
  k() const noexcept {
    static_assert(Corner < 8);
    return static_cast<coordinate_type>(
        base_z_ +
        static_cast<coordinate_type>((Corner >> 2U) & 1U));
  }

  template <std::size_t Corner>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Weight
  weight() const noexcept {
    static_assert(Corner < 8);
    const Weight wx = (Corner & 1U) != 0 ? fraction_x_ : Weight{1} - fraction_x_;
    const Weight wy = ((Corner >> 1U) & 1U) != 0
                          ? fraction_y_
                          : Weight{1} - fraction_y_;
    const Weight wz = ((Corner >> 2U) & 1U) != 0
                          ? fraction_z_
                          : Weight{1} - fraction_z_;
    return wx * wy * wz;
  }

  // The sample of component at corner Corner.
  template <std::size_t Corner, class Component>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr decltype(auto)
  at(const Component &component) const noexcept {
    return component(i<Corner>(), j<Corner>(), k<Corner>());
  }

private:
  LPFX_HOST_DEVICE constexpr TrilinearStencil3D(
      const coordinate_type base_x, const coordinate_type base_y,
      const coordinate_type base_z, const Weight fraction_x,
      const Weight fraction_y, const Weight fraction_z) noexcept
      : base_x_(base_x), base_y_(base_y), base_z_(base_z),
        fraction_x_(fraction_x), fraction_y_(fraction_y),
        fraction_z_(fraction_z) {}

  coordinate_type base_x_{};
  coordinate_type base_y_{};
  coordinate_type base_z_{};
  Weight fraction_x_{};
  Weight fraction_y_{};
  Weight fraction_z_{};

  friend struct detail::PrivateConstructionAccess;
};

// Hard precondition: every finite shifted coordinate and its enclosing sample
// pair is representable by IndexPolicy::coordinate_type. This unchecked inner
// primitive performs no bounds or halo checks.
template <detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy,
          std::floating_point Weight>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr
    TrilinearStencil3D<Location, IndexPolicy, Weight>
    make_trilinear_stencil(const Weight x, const Weight y,
                           const Weight z) noexcept {
  using Coordinate = typename IndexPolicy::coordinate_type;
  const auto x_coordinates =
      detail::make_trilinear_axis_coordinates<Coordinate>(x,
                                                          Location::x_is_node);
  const auto y_coordinates =
      detail::make_trilinear_axis_coordinates<Coordinate>(y,
                                                          Location::y_is_node);
  const auto z_coordinates =
      detail::make_trilinear_axis_coordinates<Coordinate>(z,
                                                          Location::z_is_node);
  return detail::PrivateConstructionAccess::create<
      TrilinearStencil3D<Location, IndexPolicy, Weight>>(
      x_coordinates.base, y_coordinates.base, z_coordinates.base,
      x_coordinates.fraction, y_coordinates.fraction, z_coordinates.fraction);
}

// Eight direct field loads followed by seven nested linear interpolations.
// Accumulator is explicit: LPFx does not choose a global compute/result type.
// Component and stencil locations must match at compile time.
template <std::floating_point Accumulator, class Component,
          detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy,
          std::floating_point Weight>
  requires detail::TrilinearSampleComponent<Component, Location, IndexPolicy,
                                              Accumulator>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
trilinear_sample(
    const Component component,
    const TrilinearStencil3D<Location, IndexPolicy, Weight> stencil) noexcept {
  const Accumulator fx = static_cast<Accumulator>(stencil.fraction_x());
  const Accumulator fy = static_cast<Accumulator>(stencil.fraction_y());
  const Accumulator fz = static_cast<Accumulator>(stencil.fraction_z());

  using detail::trilinear_corner;
  const Accumulator x00 = detail::trilinear_lerp(
      trilinear_corner<0, Accumulator>(component, stencil),
      trilinear_corner<1, Accumulator>(component, stencil), fx);
  const Accumulator x10 = detail::trilinear_lerp(
      trilinear_corner<2, Accumulator>(component, stencil),
      trilinear_corner<3, Accumulator>(component, stencil), fx);
  const Accumulator x01 = detail::trilinear_lerp(
      trilinear_corner<4, Accumulator>(component, stencil),
      trilinear_corner<5, Accumulator>(component, stencil), fx);
  const Accumulator x11 = detail::trilinear_lerp(
      trilinear_corner<6, Accumulator>(component, stencil),
      trilinear_corner<7, Accumulator>(component, stencil), fx);
  const Accumulator y0 = detail::trilinear_lerp(x00, x10, fy);
  const Accumulator y1 = detail::trilinear_lerp(x01, x11, fy);
  return detail::trilinear_lerp(y0, y1, fz);
}

} // namespace lpfx

#endif
