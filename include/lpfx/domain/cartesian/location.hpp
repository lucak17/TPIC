#ifndef LPFX_DOMAIN_CARTESIAN_LOCATION_HPP
#define LPFX_DOMAIN_CARTESIAN_LOCATION_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/private_construction.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>

namespace lpfx {

enum class AxisLocation : std::uint8_t { Cell = 0, Node = 1 };

template <AxisLocation X, AxisLocation Y, AxisLocation Z>
struct Location3D final {
  static constexpr AxisLocation x = X;
  static constexpr AxisLocation y = Y;
  static constexpr AxisLocation z = Z;
  static constexpr bool x_is_node = X == AxisLocation::Node;
  static constexpr bool y_is_node = Y == AxisLocation::Node;
  static constexpr bool z_is_node = Z == AxisLocation::Node;

  template <std::size_t Axis>
    requires(Axis < 3)
  static constexpr bool is_node =
      Axis == 0 ? x_is_node : (Axis == 1 ? y_is_node : z_is_node);
};

using Cell =
    Location3D<AxisLocation::Cell, AxisLocation::Cell, AxisLocation::Cell>;
using Node =
    Location3D<AxisLocation::Node, AxisLocation::Node, AxisLocation::Node>;
using FaceX =
    Location3D<AxisLocation::Node, AxisLocation::Cell, AxisLocation::Cell>;
using FaceY =
    Location3D<AxisLocation::Cell, AxisLocation::Node, AxisLocation::Cell>;
using FaceZ =
    Location3D<AxisLocation::Cell, AxisLocation::Cell, AxisLocation::Node>;
using EdgeX =
    Location3D<AxisLocation::Cell, AxisLocation::Node, AxisLocation::Node>;
using EdgeY =
    Location3D<AxisLocation::Node, AxisLocation::Cell, AxisLocation::Node>;
using EdgeZ =
    Location3D<AxisLocation::Node, AxisLocation::Node, AxisLocation::Cell>;

namespace detail {

template <class T> struct IsLocation3D : std::false_type {};

template <AxisLocation X, AxisLocation Y, AxisLocation Z>
struct IsLocation3D<Location3D<X, Y, Z>> : std::true_type {};

template <class T>
concept CartesianLocation = IsLocation3D<T>::value;

} // namespace detail

// Samples per axis: cells plus one on node axes.
template <detail::CartesianLocation Location,
          detail::CartesianIndexPolicy Policy>
[[nodiscard]] constexpr ErrorCode try_make_location_sample_extents(
    const CartesianExtent3<typename Policy::count_type> cells,
    CartesianExtent3<typename Policy::count_type> &output) noexcept {
  using Count = typename Policy::count_type;
  if (!detail::cartesian_extent_is_nonempty(cells)) {
    return ErrorCode::invalid_argument;
  }

  Count samples_x{};
  Count samples_y{};
  Count samples_z{};
  if (!detail::checked_add(cells.x(), static_cast<Count>(Location::x_is_node),
                           samples_x) ||
      !detail::checked_add(cells.y(), static_cast<Count>(Location::y_is_node),
                           samples_y) ||
      !detail::checked_add(cells.z(), static_cast<Count>(Location::z_is_node),
                           samples_z)) {
    return ErrorCode::overflow;
  }
  output = detail::PrivateConstructionAccess::create<CartesianExtent3<Count>>(
      samples_x, samples_y, samples_z);
  return ErrorCode::success;
}

// Inverse of try_make_location_sample_extents; every axis must keep at least
// one cell.
template <detail::CartesianLocation Location,
          detail::CartesianIndexPolicy Policy>
[[nodiscard]] constexpr ErrorCode try_make_location_cell_extents(
    const CartesianExtent3<typename Policy::count_type> samples,
    CartesianExtent3<typename Policy::count_type> &output) noexcept {
  using Count = typename Policy::count_type;
  constexpr Count x_offset = static_cast<Count>(Location::x_is_node);
  constexpr Count y_offset = static_cast<Count>(Location::y_is_node);
  constexpr Count z_offset = static_cast<Count>(Location::z_is_node);
  if (samples.x() <= x_offset || samples.y() <= y_offset ||
      samples.z() <= z_offset) {
    return ErrorCode::invalid_argument;
  }
  output = detail::PrivateConstructionAccess::create<CartesianExtent3<Count>>(
      static_cast<Count>(samples.x() - x_offset),
      static_cast<Count>(samples.y() - y_offset),
      static_cast<Count>(samples.z() - z_offset));
  return ErrorCode::success;
}

} // namespace lpfx

#endif
