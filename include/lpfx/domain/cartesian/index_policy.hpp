#ifndef LPFX_DOMAIN_CARTESIAN_INDEX_POLICY_HPP
#define LPFX_DOMAIN_CARTESIAN_INDEX_POLICY_HPP

#include <concepts>
#include <cstdint>

namespace lpfx {

namespace detail {

template <class T>
concept CartesianCoordinate =
    std::same_as<T, std::int32_t> || std::same_as<T, std::int64_t>;

template <class T>
concept CartesianUnsignedWord =
    std::same_as<T, std::uint32_t> || std::same_as<T, std::uint64_t>;

template <class T>
concept CartesianRank = CartesianUnsignedWord<T>;

template <class T>
concept CartesianWord = CartesianCoordinate<T> || CartesianUnsignedWord<T>;

template <class Policy>
concept CartesianIndexPolicy = requires {
  typename Policy::coordinate_type;
  typename Policy::count_type;
  typename Policy::index_type;
  typename Policy::offset_type;
  typename Policy::stride_type;
  requires CartesianCoordinate<typename Policy::coordinate_type>;
  requires CartesianUnsignedWord<typename Policy::count_type>;
  requires CartesianUnsignedWord<typename Policy::offset_type>;
  requires CartesianUnsignedWord<typename Policy::stride_type>;
  requires std::same_as<typename Policy::index_type,
                        typename Policy::count_type>;
};

} // namespace detail

template <detail::CartesianCoordinate Coordinate,
          detail::CartesianUnsignedWord Count,
          detail::CartesianUnsignedWord Offset,
          detail::CartesianUnsignedWord Stride>
struct BasicLogicalIndexPolicy final {
  using coordinate_type = Coordinate;
  using count_type = Count;
  using index_type = Count;
  using offset_type = Offset;
  using stride_type = Stride;
};

using LogicalIndexPolicy32 =
    BasicLogicalIndexPolicy<std::int32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t>;
using LogicalIndexPolicy64 =
    BasicLogicalIndexPolicy<std::int64_t, std::uint64_t, std::uint64_t,
                            std::uint64_t>;

} // namespace lpfx

#endif
