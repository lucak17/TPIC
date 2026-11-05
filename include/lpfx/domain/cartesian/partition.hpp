#ifndef LPFX_DOMAIN_CARTESIAN_PARTITION_HPP
#define LPFX_DOMAIN_CARTESIAN_PARTITION_HPP

#include <array>
#include <cstdint>

#include <lpfx/core/private_construction.hpp>
#include <lpfx/domain/cartesian/topology.hpp>

namespace lpfx {

template <detail::CartesianIndexPolicy IndexPolicy,
          detail::CartesianRank Rank>
class CartesianPartition3D final {
public:
  using policy_type = IndexPolicy;
  using coordinate_type = typename IndexPolicy::coordinate_type;
  using count_type = typename IndexPolicy::count_type;
  using rank_type = Rank;

  constexpr CartesianPartition3D() noexcept = default;

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return process_shape_.x() != 0;
  }

  [[nodiscard]] constexpr CartesianExtent3<Rank>
  process_shape() const noexcept {
    return process_shape_;
  }

  [[nodiscard]] constexpr CartesianIndex3<Rank>
  process_coordinate() const noexcept {
    return process_coordinate_;
  }

  [[nodiscard]] constexpr Rank rank() const noexcept { return rank_; }

  [[nodiscard]] constexpr CartesianBox3<coordinate_type>
  global_cell_box() const noexcept {
    return global_cell_box_;
  }

  // The partition factory proved that every extent fits count_type.
  [[nodiscard]] constexpr CartesianExtent3<count_type>
  local_cells() const noexcept {
    const CartesianIndex3<coordinate_type> lower = global_cell_box_.lower();
    const CartesianIndex3<coordinate_type> upper = global_cell_box_.upper();
    return detail::PrivateConstructionAccess::create<
        CartesianExtent3<count_type>>(
        static_cast<count_type>(upper.x() - lower.x()),
        static_cast<count_type>(upper.y() - lower.y()),
        static_cast<count_type>(upper.z() - lower.z()));
  }

  [[nodiscard]] constexpr CartesianIndex3<coordinate_type>
  global_offset() const noexcept {
    return global_cell_box_.lower();
  }

private:
  constexpr CartesianPartition3D(
      const CartesianExtent3<Rank> process_shape,
      const CartesianIndex3<Rank> process_coordinate, const Rank rank,
      const CartesianBox3<coordinate_type> global_cell_box) noexcept
      : process_shape_(process_shape), process_coordinate_(process_coordinate),
        rank_(rank), global_cell_box_(global_cell_box) {}

  CartesianExtent3<Rank> process_shape_{};
  CartesianIndex3<Rank> process_coordinate_{};
  Rank rank_{};
  CartesianBox3<coordinate_type> global_cell_box_{};

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

// Block partition of one axis: the first global_cells % process_extent
// processes receive one extra cell.
template <CartesianIndexPolicy IndexPolicy, CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode
try_partition_axis(const typename IndexPolicy::count_type global_cells,
                   const Rank process_extent, const Rank process_coordinate,
                   typename IndexPolicy::coordinate_type &lower,
                   typename IndexPolicy::coordinate_type &upper) noexcept {
  using Coordinate = typename IndexPolicy::coordinate_type;
  using Count = typename IndexPolicy::count_type;

  Count extent{};
  Count coordinate{};
  if (!checked_integral_cast(process_extent, extent) ||
      !checked_integral_cast(process_coordinate, coordinate)) {
    return ErrorCode::overflow;
  }
  if (extent == 0 || coordinate >= extent) {
    return ErrorCode::invalid_argument;
  }
  if (global_cells < extent) {
    return ErrorCode::invalid_argument;
  }

  const Count base = static_cast<Count>(global_cells / extent);
  const Count remainder = static_cast<Count>(global_cells % extent);
  const Count extra = coordinate < remainder ? Count{1} : Count{0};
  Count local{};
  if (!checked_add(base, extra, local)) {
    return ErrorCode::overflow;
  }

  Count regular_offset{};
  if (!checked_multiply(base, coordinate, regular_offset)) {
    return ErrorCode::overflow;
  }
  const Count remainder_offset =
      coordinate < remainder ? coordinate : remainder;
  Count offset{};
  if (!checked_add(regular_offset, remainder_offset, offset)) {
    return ErrorCode::overflow;
  }
  Count upper_count{};
  if (!checked_add(offset, local, upper_count)) {
    return ErrorCode::overflow;
  }

  Coordinate prepared_lower{};
  Coordinate prepared_upper{};
  if (!checked_integral_cast(offset, prepared_lower) ||
      !checked_integral_cast(upper_count, prepared_upper)) {
    return ErrorCode::overflow;
  }
  lower = prepared_lower;
  upper = prepared_upper;
  return ErrorCode::success;
}

// Partition of the process at process_coordinate; global_cells is non-empty.
template <CartesianIndexPolicy IndexPolicy, CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode try_prepare_cartesian_partition_at(
    const CartesianExtent3<typename IndexPolicy::count_type> global_cells,
    const CartesianExtent3<Rank> process_shape,
    const CartesianIndex3<Rank> process_coordinate, const Rank rank,
    CartesianPartition3D<IndexPolicy, Rank> &output) noexcept {
  using Coordinate = typename IndexPolicy::coordinate_type;

  std::array<Coordinate, 3> lower{};
  std::array<Coordinate, 3> upper{};
  LPFX_TRY(for_each_axis([&](auto axis) {
    constexpr std::size_t Axis = decltype(axis)::value;
    return try_partition_axis<IndexPolicy>(
        global_cells.template get<Axis>(), process_shape.template get<Axis>(),
        process_coordinate.template get<Axis>(), lower[Axis], upper[Axis]);
  }));

  CartesianBox3<Coordinate> global_cell_box{};
  LPFX_TRY(try_make_cartesian_box3(lower[0], lower[1], lower[2], upper[0],
                                   upper[1], upper[2], global_cell_box));
  output = PrivateConstructionAccess::create<
      CartesianPartition3D<IndexPolicy, Rank>>(process_shape,
                                               process_coordinate, rank,
                                               global_cell_box);
  return ErrorCode::success;
}

} // namespace detail

template <detail::CartesianIndexPolicy IndexPolicy,
          detail::CartesianRank Rank, detail::checked_integer GlobalX,
          detail::checked_integer GlobalY, detail::checked_integer GlobalZ>
[[nodiscard]] constexpr ErrorCode
try_prepare_cartesian_partition3d(
    const GlobalX global_x, const GlobalY global_y, const GlobalZ global_z,
    const CanonicalCartesianTopologyInput3D<Rank> &topology,
    CartesianPartition3D<IndexPolicy, Rank> &output) noexcept {
  if (!topology.is_valid()) {
    return ErrorCode::invalid_argument;
  }

  CartesianExtent3<typename IndexPolicy::count_type> global_cells{};
  LPFX_TRY(try_make_cartesian_extent3(global_x, global_y, global_z,
                                      global_cells));
  if (!detail::cartesian_extent_is_nonempty(global_cells)) {
    return ErrorCode::invalid_argument;
  }
  return detail::try_prepare_cartesian_partition_at<IndexPolicy>(
      global_cells, topology.process_shape(), topology.process_coordinate(),
      topology.rank(), output);
}

} // namespace lpfx

#endif
