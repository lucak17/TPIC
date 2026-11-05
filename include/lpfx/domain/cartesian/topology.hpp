#ifndef LPFX_DOMAIN_CARTESIAN_TOPOLOGY_HPP
#define LPFX_DOMAIN_CARTESIAN_TOPOLOGY_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/private_construction.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>

namespace lpfx {

struct CartesianPeriodicity3 final {
  bool x{};
  bool y{};
  bool z{};

  [[nodiscard]] friend constexpr bool
  operator==(const CartesianPeriodicity3 &,
             const CartesianPeriodicity3 &) noexcept = default;
};

// Canonical ranks are x-fast: rank = x + shape.x * (y + shape.y * z).
template <detail::CartesianRank Rank>
[[nodiscard]] constexpr bool
try_canonical_coordinate_to_rank(const CartesianExtent3<Rank> shape,
                                 const CartesianIndex3<Rank> coordinate,
                                 Rank &output) noexcept {
  Rank volume{};
  if (!detail::cartesian_extent_is_nonempty(shape) ||
      !detail::checked_volume(shape, volume) || coordinate.x() >= shape.x() ||
      coordinate.y() >= shape.y() || coordinate.z() >= shape.z()) {
    return false;
  }

  const Rank y_plane = static_cast<Rank>(shape.y() * coordinate.z());
  const Rank yz = static_cast<Rank>(coordinate.y() + y_plane);
  const Rank x_planes = static_cast<Rank>(shape.x() * yz);
  output = static_cast<Rank>(coordinate.x() + x_planes);
  return true;
}

template <detail::CartesianRank Rank>
[[nodiscard]] constexpr bool
try_canonical_rank_to_coordinate(const CartesianExtent3<Rank> shape,
                                 const Rank rank,
                                 CartesianIndex3<Rank> &output) noexcept {
  Rank volume{};
  if (!detail::cartesian_extent_is_nonempty(shape) ||
      !detail::checked_volume(shape, volume) || rank >= volume) {
    return false;
  }

  const Rank after_x = static_cast<Rank>(rank / shape.x());
  output = detail::PrivateConstructionAccess::create<CartesianIndex3<Rank>>(
      static_cast<Rank>(rank % shape.x()),
      static_cast<Rank>(after_x % shape.y()),
      static_cast<Rank>(after_x / shape.y()));
  return true;
}

template <detail::CartesianRank Rank>
class CanonicalCartesianTopologyInput3D final {
public:
  constexpr CanonicalCartesianTopologyInput3D() noexcept = default;

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

  [[nodiscard]] constexpr Rank communicator_size() const noexcept {
    return communicator_size_;
  }

  [[nodiscard]] constexpr CartesianPeriodicity3 periodicity() const noexcept {
    return periodicity_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const CanonicalCartesianTopologyInput3D &,
             const CanonicalCartesianTopologyInput3D &) noexcept = default;

private:
  constexpr CanonicalCartesianTopologyInput3D(
      const CartesianExtent3<Rank> process_shape,
      const CartesianIndex3<Rank> process_coordinate, const Rank rank,
      const Rank communicator_size,
      const CartesianPeriodicity3 periodicity) noexcept
      : process_shape_(process_shape), process_coordinate_(process_coordinate),
        rank_(rank), communicator_size_(communicator_size),
        periodicity_(periodicity) {}

  CartesianExtent3<Rank> process_shape_{};
  CartesianIndex3<Rank> process_coordinate_{};
  Rank rank_{};
  Rank communicator_size_{};
  CartesianPeriodicity3 periodicity_{};

  friend struct detail::PrivateConstructionAccess;
};

template <
    detail::CartesianRank Rank, detail::checked_integer ProcessX,
    detail::checked_integer ProcessY, detail::checked_integer ProcessZ,
    detail::checked_integer CoordinateX, detail::checked_integer CoordinateY,
    detail::checked_integer CoordinateZ, detail::checked_integer RankValue,
    detail::checked_integer CommunicatorSize>
[[nodiscard]] constexpr ErrorCode
try_make_canonical_cartesian_topology_input(
    const ProcessX process_x, const ProcessY process_y,
    const ProcessZ process_z, const CoordinateX coordinate_x,
    const CoordinateY coordinate_y, const CoordinateZ coordinate_z,
    const RankValue rank_value, const CommunicatorSize communicator_size_value,
    const CartesianPeriodicity3 periodicity,
    CanonicalCartesianTopologyInput3D<Rank> &output) noexcept {
  CartesianExtent3<Rank> process_shape{};
  CartesianIndex3<Rank> process_coordinate{};
  Rank prepared_rank{};
  Rank prepared_communicator_size{};
  LPFX_TRY(try_make_cartesian_extent3(process_x, process_y, process_z,
                                      process_shape));
  LPFX_TRY(try_make_cartesian_index3(coordinate_x, coordinate_y, coordinate_z,
                                     process_coordinate));
  if (!detail::checked_integral_cast(rank_value, prepared_rank) ||
      !detail::checked_integral_cast(communicator_size_value,
                                     prepared_communicator_size)) {
    return ErrorCode::overflow;
  }
  if (!detail::cartesian_extent_is_nonempty(process_shape)) {
    return ErrorCode::invalid_argument;
  }
  Rank topology_size{};
  if (!detail::checked_volume(process_shape, topology_size)) {
    return ErrorCode::overflow;
  }

  // A matching canonical rank also implies rank < communicator size.
  Rank canonical_rank{};
  if (prepared_communicator_size != topology_size ||
      !try_canonical_coordinate_to_rank(process_shape, process_coordinate,
                                        canonical_rank) ||
      canonical_rank != prepared_rank) {
    return ErrorCode::invalid_argument;
  }

  output = detail::PrivateConstructionAccess::create<
      CanonicalCartesianTopologyInput3D<Rank>>(
      process_shape, process_coordinate, prepared_rank,
      prepared_communicator_size, periodicity);
  return ErrorCode::success;
}

// Convenience for runtime MPI configuration. The process shape normally comes
// from application input, while rank and communicator size come from the live
// communicator. Canonical x-fast coordinates are derived rather than repeated
// in application setup code.
template <detail::CartesianRank Rank, detail::checked_integer ProcessX,
          detail::checked_integer ProcessY, detail::checked_integer ProcessZ,
          detail::checked_integer RankValue,
          detail::checked_integer CommunicatorSize>
[[nodiscard]] constexpr ErrorCode
try_make_canonical_cartesian_topology_input(
    const ProcessX process_x, const ProcessY process_y,
    const ProcessZ process_z, const RankValue rank_value,
    const CommunicatorSize communicator_size_value,
    const CartesianPeriodicity3 periodicity,
    CanonicalCartesianTopologyInput3D<Rank> &output) noexcept {
  CartesianExtent3<Rank> process_shape{};
  LPFX_TRY(try_make_cartesian_extent3(process_x, process_y, process_z,
                                      process_shape));
  if (!detail::cartesian_extent_is_nonempty(process_shape)) {
    return ErrorCode::invalid_argument;
  }

  Rank prepared_rank{};
  Rank prepared_communicator_size{};
  if (!detail::checked_integral_cast(rank_value, prepared_rank) ||
      !detail::checked_integral_cast(communicator_size_value,
                                     prepared_communicator_size)) {
    return ErrorCode::overflow;
  }

  Rank topology_size{};
  if (!detail::checked_volume(process_shape, topology_size)) {
    return ErrorCode::overflow;
  }
  CartesianIndex3<Rank> process_coordinate{};
  if (prepared_communicator_size != topology_size ||
      !try_canonical_rank_to_coordinate(process_shape, prepared_rank,
                                        process_coordinate)) {
    return ErrorCode::invalid_argument;
  }

  output = detail::PrivateConstructionAccess::create<
      CanonicalCartesianTopologyInput3D<Rank>>(
      process_shape, process_coordinate, prepared_rank,
      prepared_communicator_size, periodicity);
  return ErrorCode::success;
}

namespace detail {

template <CartesianRank Rank>
[[nodiscard]] constexpr bool
try_neighbor_axis_coordinate(const Rank coordinate, const Rank process_extent,
                             const std::int8_t step, const bool periodic,
                             Rank &neighbor) noexcept {
  if (process_extent == 0 || coordinate >= process_extent ||
      !is_cartesian_step(step)) {
    return false;
  }
  if (step == 0) {
    neighbor = coordinate;
    return true;
  }
  if (step < 0) {
    if (coordinate != 0) {
      neighbor = static_cast<Rank>(coordinate - 1);
      return true;
    }
    if (periodic) {
      neighbor = static_cast<Rank>(process_extent - 1);
      return true;
    }
    return false;
  }

  if (coordinate != static_cast<Rank>(process_extent - 1)) {
    neighbor = static_cast<Rank>(coordinate + 1);
    return true;
  }
  if (periodic) {
    neighbor = 0;
    return true;
  }
  return false;
}

} // namespace detail

template <detail::CartesianRank Rank> class CartesianNeighbor3 final {
public:
  constexpr CartesianNeighbor3() noexcept = default;

  [[nodiscard]] constexpr CartesianDirection3 direction() const noexcept {
    return direction_;
  }

  // Hard precondition: has_peer() is true.
  [[nodiscard]] constexpr Rank peer_rank() const noexcept { return peer_rank_; }

  [[nodiscard]] constexpr bool has_peer() const noexcept {
    return (flags_ & has_peer_flag) != 0;
  }

  [[nodiscard]] constexpr bool is_self_peer() const noexcept {
    return (flags_ & self_peer_flag) != 0;
  }

private:
  static constexpr std::uint8_t has_peer_flag = 1;
  static constexpr std::uint8_t self_peer_flag = 2;

  constexpr CartesianNeighbor3(const CartesianDirection3 direction,
                               const Rank peer_rank, const bool has_peer,
                               const bool self_peer) noexcept
      : direction_(direction),
        flags_(static_cast<std::uint8_t>((has_peer ? has_peer_flag : 0) |
                                         (self_peer ? self_peer_flag : 0))),
        peer_rank_(peer_rank) {}

  CartesianDirection3 direction_{};
  std::uint8_t flags_{};
  Rank peer_rank_{};

  friend struct detail::PrivateConstructionAccess;
};

// Canonical transport-free topology. Communication protocols layer their own
// peer metadata on this deterministic neighbor table.
template <detail::CartesianRank Rank>
class PreparedCanonicalCartesianTopology3D final {
public:
  using rank_type = Rank;
  static constexpr std::size_t neighbor_count = 26;

  constexpr PreparedCanonicalCartesianTopology3D() noexcept = default;

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return input_.is_valid();
  }

  [[nodiscard]] constexpr const CanonicalCartesianTopologyInput3D<Rank> &
  input() const noexcept {
    return input_;
  }

  // Hard precondition: is_valid() and index < neighbor_count.
  [[nodiscard]] constexpr const CartesianNeighbor3<Rank> &
  neighbor_by_index(const std::size_t index) const noexcept {
    return neighbors_[index];
  }

  // Hard precondition: is_valid() and direction is a Cartesian neighbor
  // direction.
  [[nodiscard]] constexpr const CartesianNeighbor3<Rank> &
  neighbor(const CartesianDirection3 direction) const noexcept {
    return neighbors_[cartesian_neighbor_index(direction)];
  }

private:
  constexpr PreparedCanonicalCartesianTopology3D(
      const CanonicalCartesianTopologyInput3D<Rank> input,
      const std::array<CartesianNeighbor3<Rank>, neighbor_count>
          neighbors) noexcept
      : input_(input), neighbors_(neighbors) {}

  CanonicalCartesianTopologyInput3D<Rank> input_{};
  std::array<CartesianNeighbor3<Rank>, neighbor_count> neighbors_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode
try_prepare_canonical_cartesian_topology(
    const CanonicalCartesianTopologyInput3D<Rank> &input,
    PreparedCanonicalCartesianTopology3D<Rank> &output) noexcept {
  using Topology = PreparedCanonicalCartesianTopology3D<Rank>;
  if (!input.is_valid()) {
    return ErrorCode::invalid_argument;
  }

  const CartesianExtent3<Rank> shape = input.process_shape();
  const CartesianIndex3<Rank> coordinate = input.process_coordinate();
  const CartesianPeriodicity3 periodicity = input.periodicity();
  std::array<CartesianNeighbor3<Rank>, Topology::neighbor_count> neighbors{};

  for (std::size_t index = 0; index < neighbors.size(); ++index) {
    const CartesianDirection3 direction =
        canonical_cartesian_neighbor_directions[index];
    Rank neighbor_x{};
    Rank neighbor_y{};
    Rank neighbor_z{};
    const bool has_peer =
        detail::try_neighbor_axis_coordinate(coordinate.x(), shape.x(),
                                             direction.x, periodicity.x,
                                             neighbor_x) &&
        detail::try_neighbor_axis_coordinate(coordinate.y(), shape.y(),
                                             direction.y, periodicity.y,
                                             neighbor_y) &&
        detail::try_neighbor_axis_coordinate(
            coordinate.z(), shape.z(), direction.z, periodicity.z, neighbor_z);

    Rank peer{};
    if (has_peer &&
        !try_canonical_coordinate_to_rank(
            shape,
            detail::PrivateConstructionAccess::create<CartesianIndex3<Rank>>(
                neighbor_x, neighbor_y, neighbor_z),
            peer)) {
      return ErrorCode::invalid_argument;
    }
    neighbors[index] =
        detail::PrivateConstructionAccess::create<CartesianNeighbor3<Rank>>(
            direction, peer, has_peer, has_peer && peer == input.rank());
  }

  output = detail::PrivateConstructionAccess::create<Topology>(input,
                                                               neighbors);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
