#ifndef LPFX_DOMAIN_CARTESIAN_PARTITION_NEIGHBORHOOD_HPP
#define LPFX_DOMAIN_CARTESIAN_PARTITION_NEIGHBORHOOD_HPP

#include <array>
#include <cstddef>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/domain/cartesian/partition.hpp>

namespace lpfx {

// A transport-independent, canonical partition neighborhood. Physical
// boundaries have no partition. Periodic self neighbors resolve directly to
// local_partition(); remote slots are ordered by the canonical direction
// index owned by the topology.
template <detail::CartesianIndexPolicy IndexPolicy,
          detail::CartesianRank Rank>
class PreparedCartesianPartitionNeighborhood3D final {
public:
  using index_policy = IndexPolicy;
  using rank_type = Rank;
  using count_type = typename IndexPolicy::count_type;
  using topology_type = PreparedCanonicalCartesianTopology3D<Rank>;
  using partition_type = CartesianPartition3D<IndexPolicy, Rank>;

  static constexpr std::size_t neighbor_count = topology_type::neighbor_count;

  constexpr PreparedCartesianPartitionNeighborhood3D() noexcept = default;

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return topology_.is_valid() && local_partition_.is_valid();
  }

  [[nodiscard]] constexpr const topology_type &topology() const noexcept {
    return topology_;
  }

  [[nodiscard]] constexpr CartesianExtent3<count_type>
  global_cells() const noexcept {
    return global_cells_;
  }

  [[nodiscard]] constexpr const partition_type &
  local_partition() const noexcept {
    return local_partition_;
  }

  // Hard precondition: is_valid() and index < neighbor_count.
  [[nodiscard]] constexpr bool
  has_neighbor_partition_by_index(const std::size_t index) const noexcept {
    return topology_.neighbor_by_index(index).has_peer();
  }

  // Hard precondition: is_valid(), index < neighbor_count and
  // has_neighbor_partition_by_index(index).
  [[nodiscard]] constexpr const partition_type &
  neighbor_partition_by_index(const std::size_t index) const noexcept {
    return topology_.neighbor_by_index(index).is_self_peer()
               ? local_partition_
               : remote_partitions_[index];
  }

private:
  constexpr PreparedCartesianPartitionNeighborhood3D(
      const topology_type topology,
      const CartesianExtent3<count_type> global_cells,
      const partition_type local_partition,
      const std::array<partition_type, neighbor_count>
          remote_partitions) noexcept
      : topology_(topology), global_cells_(global_cells),
        local_partition_(local_partition),
        remote_partitions_(remote_partitions) {}

  topology_type topology_{};
  CartesianExtent3<count_type> global_cells_{};
  partition_type local_partition_{};
  std::array<partition_type, neighbor_count> remote_partitions_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianIndexPolicy IndexPolicy,
          detail::CartesianRank Rank, detail::checked_integer GlobalX,
          detail::checked_integer GlobalY, detail::checked_integer GlobalZ>
[[nodiscard]] constexpr ErrorCode
try_prepare_cartesian_partition_neighborhood3d(
    const GlobalX global_x, const GlobalY global_y, const GlobalZ global_z,
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    PreparedCartesianPartitionNeighborhood3D<IndexPolicy, Rank>
        &output) noexcept {
  using Neighborhood =
      PreparedCartesianPartitionNeighborhood3D<IndexPolicy, Rank>;
  using Coordinate = typename IndexPolicy::coordinate_type;
  using Count = typename IndexPolicy::count_type;
  using Partition = typename Neighborhood::partition_type;

  if (!topology.is_valid()) {
    return ErrorCode::invalid_argument;
  }

  CartesianExtent3<Count> global_cells{};
  LPFX_TRY(try_make_cartesian_extent3(global_x, global_y, global_z,
                                      global_cells));
  if (!detail::cartesian_extent_is_nonempty(global_cells)) {
    return ErrorCode::invalid_argument;
  }

  // Every partition endpoint is represented by Coordinate. Validate the
  // global upper endpoint once, including ranks outside the local
  // neighborhood, rather than accepting a decomposition that fails only when
  // a distant rank is queried later.
  Coordinate coordinate_bound{};
  if (!detail::checked_integral_cast(global_cells.x(), coordinate_bound) ||
      !detail::checked_integral_cast(global_cells.y(), coordinate_bound) ||
      !detail::checked_integral_cast(global_cells.z(), coordinate_bound)) {
    return ErrorCode::overflow;
  }

  const CanonicalCartesianTopologyInput3D<Rank> &input = topology.input();
  const CartesianExtent3<Rank> shape = input.process_shape();
  Partition local_partition{};
  LPFX_TRY(detail::try_prepare_cartesian_partition_at<IndexPolicy>(
      global_cells, shape, input.process_coordinate(), input.rank(),
      local_partition));

  std::array<Partition, Neighborhood::neighbor_count> remote_partitions{};
  for (std::size_t index = 0; index != Neighborhood::neighbor_count; ++index) {
    const auto &neighbor = topology.neighbor_by_index(index);
    if (!neighbor.has_peer() || neighbor.is_self_peer()) {
      continue;
    }

    bool already_prepared = false;
    for (std::size_t previous = 0; previous != index; ++previous) {
      const auto &candidate = topology.neighbor_by_index(previous);
      if (candidate.has_peer() && !candidate.is_self_peer() &&
          candidate.peer_rank() == neighbor.peer_rank()) {
        remote_partitions[index] = remote_partitions[previous];
        already_prepared = true;
        break;
      }
    }
    if (already_prepared) {
      continue;
    }

    CartesianIndex3<Rank> peer_coordinate{};
    if (!try_canonical_rank_to_coordinate(shape, neighbor.peer_rank(),
                                          peer_coordinate)) {
      return ErrorCode::invalid_argument;
    }
    LPFX_TRY(detail::try_prepare_cartesian_partition_at<IndexPolicy>(
        global_cells, shape, peer_coordinate, neighbor.peer_rank(),
        remote_partitions[index]));
  }

  output = detail::PrivateConstructionAccess::create<Neighborhood>(
      topology, global_cells, local_partition, remote_partitions);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
