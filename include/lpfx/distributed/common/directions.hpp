#ifndef LPFX_DISTRIBUTED_COMMON_DIRECTIONS_HPP
#define LPFX_DISTRIBUTED_COMMON_DIRECTIONS_HPP

#include <concepts>
#include <type_traits>

#include <lpfx/domain/cartesian/direction_mask.hpp>

namespace lpfx::distributed {

inline constexpr CartesianNeighborMask3D all_directions_3d =
    cartesian_all_neighbor_mask;

} // namespace lpfx::distributed

namespace lpfx::detail {

template <class Selector>
concept CartesianDirectionSelector =
    std::integral<std::remove_cvref_t<Selector>> ||
    std::same_as<std::remove_cvref_t<Selector>, CartesianDirection3>;

} // namespace lpfx::detail

#endif
