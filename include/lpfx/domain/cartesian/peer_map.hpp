#ifndef LPFX_DOMAIN_CARTESIAN_PEER_MAP_HPP
#define LPFX_DOMAIN_CARTESIAN_PEER_MAP_HPP

#include <cstddef>
#include <cstdint>

#include <lpfx/core/fixed_array.hpp>
#include <lpfx/domain/cartesian/direction_mask.hpp>
#include <lpfx/domain/cartesian/topology.hpp>

namespace lpfx {

// Local peer slots are ordered by rank, not reciprocal slot identities. The
// selection mask belongs to the caller: particle reachability may be
// asymmetric. Fields discard this value after layout; particle classification
// retains it.
template <detail::CartesianRank Rank> class PreparedCartesianPeerMap3D final {
public:
  using rank_type = Rank;
  static constexpr std::size_t maximum_peer_count = 26;
  static constexpr std::uint8_t absent_peer = UINT8_MAX;

  constexpr PreparedCartesianPeerMap3D() noexcept {
    for (std::size_t d = 0; d != maximum_peer_count; ++d)
      direction_peers_[d] = absent_peer;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  peer_count() const noexcept {
    return count_;
  }

  // Precondition: peer < peer_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr Rank
  peer_rank(const std::uint8_t peer) const noexcept {
    return ranks_[peer];
  }

  // Precondition: direction < 26. Unselected, boundary and self entries are
  // absent.
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  peer_group(const std::size_t direction) const noexcept {
    return direction_peers_[direction];
  }

private:
  detail::DeviceFixedArray<Rank, maximum_peer_count> ranks_{};
  detail::DeviceFixedArray<std::uint8_t, maximum_peer_count> direction_peers_{};
  std::uint8_t count_{};

  template <detail::CartesianRank R>
  friend constexpr ErrorCode try_prepare_cartesian_peer_map(
      const PreparedCanonicalCartesianTopology3D<R> &, CartesianNeighborMask3D,
      PreparedCartesianPeerMap3D<R> &) noexcept;
};

// One checked core per rank type, independent of schemas, channels and
// backends. Empty/asymmetric masks are valid. Failure leaves the output
// unchanged.
template <detail::CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode try_prepare_cartesian_peer_map(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const CartesianNeighborMask3D selected,
    PreparedCartesianPeerMap3D<Rank> &output) noexcept {
  if (!topology.is_valid() ||
      (selected.bits & ~cartesian_all_neighbor_mask.bits) != 0)
    return ErrorCode::invalid_argument;

  PreparedCartesianPeerMap3D<Rank> candidate{};
  for (std::size_t d = 0; d != candidate.maximum_peer_count; ++d) {
    const auto &neighbor = topology.neighbor_by_index(d);
    if (!cartesian_neighbor_mask_contains(selected, d) ||
        !neighbor.has_peer() || neighbor.is_self_peer())
      continue;
    const Rank rank = neighbor.peer_rank();
    std::uint8_t peer = 0;
    while (peer != candidate.count_ && candidate.ranks_[peer] < rank)
      ++peer;
    if (peer != candidate.count_ && candidate.ranks_[peer] == rank)
      continue;
    for (auto next = candidate.count_; next != peer; --next)
      candidate.ranks_[next] = candidate.ranks_[next - 1];
    candidate.ranks_[peer] = rank;
    ++candidate.count_;
  }
  for (std::size_t d = 0; d != candidate.maximum_peer_count; ++d) {
    const auto &neighbor = topology.neighbor_by_index(d);
    if (!cartesian_neighbor_mask_contains(selected, d) ||
        !neighbor.has_peer() || neighbor.is_self_peer())
      continue;
    std::uint8_t peer = 0;
    while (candidate.ranks_[peer] != neighbor.peer_rank())
      ++peer;
    candidate.direction_peers_[d] = peer;
  }
  output = candidate;
  return ErrorCode::success;
}

} // namespace lpfx

#endif
