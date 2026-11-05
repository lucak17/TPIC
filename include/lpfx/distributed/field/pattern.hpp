#ifndef LPFX_DISTRIBUTED_FIELD_PATTERN_HPP
#define LPFX_DISTRIBUTED_FIELD_PATTERN_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/common/directions.hpp>
#include <lpfx/distributed/descriptor/field.hpp>
#include <lpfx/distributed/field/geometry.hpp>
#include <lpfx/domain/cartesian/direction_mask.hpp>
#include <lpfx/domain/cartesian/peer_map.hpp>
#include <lpfx/domain/cartesian/topology.hpp>

namespace lpfx {

enum class FieldPeerKind : std::uint8_t {
  physical_boundary = 0,
  self_peer = 1,
  remote_peer = 2,
  masked_out = 3,
};

// Rank-local geometry a field pattern is prepared from.
template <distributed::detail::FieldOperationSpecType Spec,
          detail::CartesianIndexPolicy IndexPolicy>
struct FieldRankGeometry3D final {
  using index_policy = IndexPolicy;
  using count_type = typename IndexPolicy::count_type;

  CartesianExtent3<count_type> cell_extents{};
  std::array<HaloDepth3<count_type>, Spec::component_count> component_halos{};

  [[nodiscard]] friend constexpr bool
  operator==(const FieldRankGeometry3D &,
             const FieldRankGeometry3D &) noexcept = default;
};

// Shared-sample add reads no halo.
template <distributed::detail::FieldOperationSpecType Spec,
          detail::CartesianIndexPolicy IndexPolicy>
  requires(Spec::kind == distributed::field_operation_kind::shared_sample_add)
struct FieldRankGeometry3D<Spec, IndexPolicy> final {
  using index_policy = IndexPolicy;
  using count_type = typename IndexPolicy::count_type;

  CartesianExtent3<count_type> cell_extents{};

  [[nodiscard]] friend constexpr bool
  operator==(const FieldRankGeometry3D &,
             const FieldRankGeometry3D &) noexcept = default;
};

namespace detail {

template <FieldFlow Flow>
inline constexpr std::size_t field_flow_index = static_cast<std::size_t>(Flow);

[[nodiscard]] constexpr bool
field_peer_kind_has_peer(const FieldPeerKind kind) noexcept {
  return kind == FieldPeerKind::self_peer || kind == FieldPeerKind::remote_peer;
}

template <class Spec, CartesianRank Rank>
[[nodiscard]] constexpr FieldPeerKind
field_peer_kind(const PreparedCanonicalCartesianTopology3D<Rank> &topology,
                const std::size_t direction) noexcept {
  if (!cartesian_neighbor_mask_contains(Spec::transport_mask, direction)) {
    return FieldPeerKind::masked_out;
  }
  const auto &neighbor = topology.neighbor_by_index(direction);
  if (!neighbor.has_peer()) {
    return FieldPeerKind::physical_boundary;
  }
  return neighbor.is_self_peer() ? FieldPeerKind::self_peer
                                 : FieldPeerKind::remote_peer;
}

template <CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr FieldPiece<IndexPolicy>
reverse_field_piece(const FieldPiece<IndexPolicy> &piece) noexcept {
  FieldPiece<IndexPolicy> reversed{};
  reversed.source = piece.destination;
  reversed.destination = piece.source;
  reversed.samples = piece.samples;
  return reversed;
}

// Pieces of every canonical direction and flow in one statically bounded
// segment each, in geometry enumeration order.
template <class Spec, CartesianIndexPolicy IndexPolicy>
class FieldPieceSegments final {
public:
  using piece_type = FieldPiece<IndexPolicy>;

  // An outbound fused piece is its receiver's inbound piece.
  template <FieldFlow Flow>
  [[nodiscard]] static constexpr std::size_t
  capacity(const std::size_t direction) noexcept {
    if constexpr (Flow == FieldFlow::outbound && Spec::fused) {
      return Spec::inbound_capacities[opposite_cartesian_neighbor_index(
          direction)];
    } else {
      return Spec::inbound_capacities[direction];
    }
  }

  template <FieldFlow Flow>
  static constexpr std::array<std::size_t, 27> starts = [] {
    std::array<std::size_t, 27> result{};
    for (std::size_t d = 0; d != 26; ++d) {
      result[d + 1] = result[d] + capacity<Flow>(d);
    }
    return result;
  }();

  template <FieldFlow Flow>
  [[nodiscard]] constexpr std::span<const piece_type>
  pieces(const std::size_t direction) const noexcept {
    constexpr std::size_t flow = field_flow_index<Flow>;
    return {pieces_[flow].data() + starts<Flow>[direction],
            directions_[direction].piece_counts[flow]};
  }

  [[nodiscard]] constexpr std::size_t
  message_bytes(const std::size_t flow,
                const std::size_t direction) const noexcept {
    return directions_[direction].message_bytes[flow];
  }

  [[nodiscard]] constexpr std::size_t
  image_offset(const std::size_t flow,
               const std::size_t direction) const noexcept {
    return directions_[direction].image_offsets[flow];
  }

  [[nodiscard]] constexpr std::array<std::size_t, 2> &
  image_offsets(const std::size_t direction) noexcept {
    return directions_[direction].image_offsets;
  }

  template <FieldFlow Flow>
  [[nodiscard]] constexpr ErrorCode
  try_append(const std::size_t direction, const piece_type &piece) noexcept {
    constexpr std::size_t flow = field_flow_index<Flow>;
    Direction &entry = directions_[direction];
    if (entry.piece_counts[flow] == capacity<Flow>(direction)) {
      return ErrorCode::capacity_exceeded;
    }
    std::size_t samples{};
    std::size_t bytes{};
    std::size_t total{};
    if (!checked_integral_cast(piece.samples, samples) ||
        !checked_multiply(
            samples, std::size_t{Spec::component_widths[piece.component]},
            bytes) ||
        !checked_add(entry.message_bytes[flow], bytes, total)) {
      return ErrorCode::overflow;
    }
    pieces_[flow][starts<Flow>[direction] + entry.piece_counts[flow]] = piece;
    ++entry.piece_counts[flow];
    entry.message_bytes[flow] = total;
    return ErrorCode::success;
  }

private:
  struct Direction final {
    std::array<std::size_t, 2> image_offsets{};
    std::array<std::size_t, 2> message_bytes{};
    std::array<std::uint16_t, 2> piece_counts{};
  };

  std::array<Direction, 26> directions_{};
  std::array<std::array<piece_type, Spec::maximum_active_piece_count>, 2>
      pieces_{};
};

} // namespace detail

template <distributed::detail::FieldOperationSpecType Spec,
          detail::CartesianIndexPolicy IndexPolicy,
          detail::CartesianRank Rank = std::uint32_t>
class PreparedFieldPattern3D;

// One canonical direction of a prepared pattern. Borrows the pattern.
template <distributed::detail::FieldOperationSpecType Spec,
          detail::CartesianIndexPolicy IndexPolicy, detail::CartesianRank Rank>
class PreparedFieldDirectionView3D final {
public:
  using pattern_type = PreparedFieldPattern3D<Spec, IndexPolicy, Rank>;
  using piece_type = FieldPiece<IndexPolicy>;

  [[nodiscard]] constexpr std::size_t index() const noexcept { return index_; }

  [[nodiscard]] constexpr CartesianDirection3 direction() const noexcept {
    return canonical_cartesian_neighbor_directions[index_];
  }

  [[nodiscard]] constexpr FieldPeerKind peer_kind() const noexcept {
    return pattern_->peer_kind(index_);
  }

  [[nodiscard]] constexpr bool has_peer() const noexcept {
    return detail::field_peer_kind_has_peer(peer_kind());
  }

  [[nodiscard]] constexpr bool is_self_peer() const noexcept {
    return peer_kind() == FieldPeerKind::self_peer;
  }

  // Hard precondition: has_peer().
  [[nodiscard]] constexpr Rank peer_rank() const noexcept {
    return pattern_->topology().neighbor_by_index(index_).peer_rank();
  }

  [[nodiscard]] constexpr std::uint32_t
  local_receive_message_group() const noexcept {
    return static_cast<std::uint32_t>(index_);
  }

  [[nodiscard]] constexpr std::uint32_t
  remote_receive_message_group() const noexcept {
    return static_cast<std::uint32_t>(opposite_cartesian_neighbor_index(index_));
  }

  template <FieldFlow Flow>
  [[nodiscard]] constexpr std::span<const piece_type> pieces() const noexcept {
    return pattern_->segments_.template pieces<Flow>(index_);
  }

  template <FieldFlow Flow>
  [[nodiscard]] constexpr std::size_t message_bytes() const noexcept {
    return pattern_->segments_.message_bytes(detail::field_flow_index<Flow>,
                                             index_);
  }

  template <FieldFlow Flow>
  [[nodiscard]] constexpr std::size_t image_offset_bytes() const noexcept {
    return pattern_->segments_.image_offset(detail::field_flow_index<Flow>,
                                            index_);
  }

private:
  constexpr PreparedFieldDirectionView3D(const pattern_type &pattern,
                                         const std::size_t index) noexcept
      : pattern_(&pattern), index_(index) {}

  const pattern_type *pattern_;
  std::size_t index_;

  friend pattern_type;
};

// The oriented pieces of one field operation for every canonical direction,
// with their message and image offsets.
template <distributed::detail::FieldOperationSpecType Spec,
          detail::CartesianIndexPolicy IndexPolicy, detail::CartesianRank Rank>
class PreparedFieldPattern3D final {
public:
  using spec_type = Spec;
  using index_policy = IndexPolicy;
  using rank_type = Rank;
  using topology_type = PreparedCanonicalCartesianTopology3D<Rank>;
  using geometry_type = FieldRankGeometry3D<Spec, IndexPolicy>;
  using piece_type = FieldPiece<IndexPolicy>;
  using direction_type = PreparedFieldDirectionView3D<Spec, IndexPolicy, Rank>;

  static constexpr std::size_t direction_count = 26;
  static constexpr std::size_t component_count = Spec::component_count;
  static constexpr CartesianNeighborMask3D transport_mask = Spec::transport_mask;

  constexpr PreparedFieldPattern3D() noexcept = default;

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return topology_.is_valid();
  }

  [[nodiscard]] constexpr const topology_type &topology() const noexcept {
    return topology_;
  }

  [[nodiscard]] constexpr const geometry_type &local_geometry() const noexcept {
    return local_geometry_;
  }

  template <detail::CartesianDirectionSelector Selector>
  [[nodiscard]] constexpr direction_type
  direction(const Selector selector) const noexcept {
    return direction_type{*this, direction_index(selector)};
  }

  template <detail::CartesianDirectionSelector Selector>
  [[nodiscard]] static constexpr bool
  direction_enabled(const Selector selector) noexcept {
    return cartesian_neighbor_mask_contains(transport_mask,
                                            direction_index(selector));
  }

  [[nodiscard]] constexpr FieldPeerKind
  peer_kind(const std::size_t direction) const noexcept {
    return detail::field_peer_kind<Spec>(topology_, direction);
  }

  [[nodiscard]] static constexpr std::size_t
  piece_bytes(const piece_type &piece) noexcept {
    return static_cast<std::size_t>(piece.samples) *
           static_cast<std::size_t>(Spec::component_widths[piece.component]);
  }

  template <FieldFlow Flow>
  [[nodiscard]] constexpr std::size_t remote_image_bytes() const noexcept {
    std::size_t bytes{};
    for (std::size_t d = 0; d != direction_count; ++d) {
      if (peer_kind(d) == FieldPeerKind::remote_peer) {
        bytes += segments_.message_bytes(detail::field_flow_index<Flow>, d);
      }
    }
    return bytes;
  }

  [[nodiscard]] constexpr std::size_t self_image_bytes() const noexcept {
    std::size_t bytes{};
    for (std::size_t d = 0; d != direction_count; ++d) {
      if (peer_kind(d) == FieldPeerKind::self_peer) {
        bytes += segments_.message_bytes(
            detail::field_flow_index<FieldFlow::outbound>, d);
      }
    }
    return bytes;
  }

private:
  template <detail::CartesianDirectionSelector Selector>
  [[nodiscard]] static constexpr std::size_t
  direction_index(const Selector selector) noexcept {
    if constexpr (std::same_as<std::remove_cvref_t<Selector>,
                               CartesianDirection3>) {
      return cartesian_neighbor_index(selector);
    } else {
      return static_cast<std::size_t>(selector);
    }
  }

  constexpr PreparedFieldPattern3D(
      const topology_type &topology, const geometry_type &local_geometry,
      const detail::FieldPieceSegments<Spec, IndexPolicy> &segments) noexcept
      : topology_(topology), local_geometry_(local_geometry),
        segments_(segments) {}

  topology_type topology_{};
  geometry_type local_geometry_{};
  detail::FieldPieceSegments<Spec, IndexPolicy> segments_{};

  friend direction_type;
  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

template <FieldFlow Flow, class Segments, class Piece>
[[nodiscard]] constexpr ErrorCode
try_append_nonempty_field_piece(Segments &segments, const std::size_t direction,
                                const std::size_t component,
                                Piece piece) noexcept {
  if (piece.samples == 0) {
    return ErrorCode::success;
  }
  piece.component = static_cast<std::uint16_t>(component);
  return segments.template try_append<Flow>(direction, piece);
}

// Every component of every peer direction has one outbound and one inbound
// piece, empty ones included.
template <class Spec, CartesianIndexPolicy IndexPolicy, CartesianRank Rank,
          class Geometry>
[[nodiscard]] constexpr ErrorCode try_prepare_affine_field_pieces(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const Geometry &local, const std::array<Geometry, 26> &peers,
    FieldPieceSegments<Spec, IndexPolicy> &segments) noexcept {
  using Kind = distributed::field_operation_kind;
  using Piece = FieldPiece<IndexPolicy>;
  for (std::size_t d = 0; d != 26; ++d) {
    if (!field_peer_kind_has_peer(field_peer_kind<Spec>(topology, d))) {
      continue;
    }
    const CartesianDirection3 direction =
        canonical_cartesian_neighbor_directions[d];
    const Geometry &peer = peers[d];
    LPFX_TRY(meta::for_each_until_error<Spec::component_count>(
        [&](const auto index) constexpr noexcept {
          constexpr std::size_t C = decltype(index)::value;
          using Location =
              typename Spec::template component_type<C>::location_type;
          Piece outbound{};
          Piece inbound{};
          if constexpr (Spec::kind == Kind::shared_sample_add) {
            LPFX_TRY((try_prepare_shared_sample_add_region<Location, IndexPolicy>(
                direction, local.cell_extents, peer.cell_extents, inbound)));
            outbound = reverse_field_piece(inbound);
          } else {
            Piece receive{};
            Piece send{};
            LPFX_TRY((try_prepare_field_copy_region<Location, IndexPolicy>(
                direction, local.cell_extents, local.component_halos[C],
                peer.cell_extents, receive)));
            LPFX_TRY((try_prepare_field_copy_region<Location, IndexPolicy>(
                opposite_cartesian_direction(direction), peer.cell_extents,
                peer.component_halos[C], local.cell_extents, send)));
            if constexpr (Spec::kind == Kind::ghost_to_owner_add) {
              // Ghost-to-owner add runs each ghost copy backwards.
              outbound = reverse_field_piece(receive);
              inbound = reverse_field_piece(send);
            } else {
              outbound = send;
              inbound = receive;
            }
          }
          outbound.component = static_cast<std::uint16_t>(C);
          inbound.component = static_cast<std::uint16_t>(C);
          LPFX_TRY(segments.template try_append<FieldFlow::outbound>(d, outbound));
          return segments.template try_append<FieldFlow::inbound>(d, inbound);
        }));
  }
  return ErrorCode::success;
}

// Per component: receiver-local inbound shared, primary and supplemental
// pieces, then the same three kinds outbound, each built in the receiving
// rank's frame so its wire order equals the receiver's inbound list.
template <class Spec, CartesianIndexPolicy IndexPolicy, CartesianRank Rank,
          class Geometry>
[[nodiscard]] constexpr ErrorCode try_prepare_fused_field_pieces(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const Geometry &local, const std::array<Geometry, 26> &peers,
    FieldPieceSegments<Spec, IndexPolicy> &segments) noexcept {
  using Piece = FieldPiece<IndexPolicy>;
  constexpr CartesianNeighborMask3D publications = Spec::publication_mask;
  std::array<bool, 26> peer{};
  for (std::size_t d = 0; d != 26; ++d) {
    peer[d] = topology.neighbor_by_index(d).has_peer();
  }
  return meta::for_each_until_error<Spec::component_count>(
      [&](const auto index) constexpr noexcept {
        // Declared here, not in the enclosing function: clang rejects a
        // constant-initialized reference of an enclosing function inside an
        // instantiated generic lambda instead of capturing it.
        constexpr auto &directions = canonical_cartesian_neighbor_directions;
        constexpr std::size_t C = decltype(index)::value;
        using Location = typename Spec::template component_type<C>::location_type;
        constexpr FieldFlow in = FieldFlow::inbound;
        constexpr FieldFlow out = FieldFlow::outbound;
        for (std::size_t d = 0; d != 26; ++d) {
          if (!field_shared_direction_is_active<Location>(directions[d]) ||
              !peer[d]) {
            continue;
          }
          Piece piece{};
          LPFX_TRY((try_prepare_fused_field_shared_region<Location, IndexPolicy>(
              directions[d], local.cell_extents, peers[d].cell_extents, piece)));
          LPFX_TRY(try_append_nonempty_field_piece<in>(segments, d, C, piece));
        }
        for (std::size_t d = 0; d != 26; ++d) {
          if (!cartesian_neighbor_mask_contains(publications, d) || !peer[d]) {
            continue;
          }
          Piece piece{};
          LPFX_TRY((try_prepare_fused_field_primary_region<Location, IndexPolicy>(
              directions[d], local.cell_extents, local.component_halos[C],
              peers[d].cell_extents, piece)));
          LPFX_TRY(try_append_nonempty_field_piece<in>(segments, d, C, piece));
        }
        for (const FusedFieldRoute route : fused_field_routes) {
          if (!cartesian_neighbor_mask_contains(publications,
                                                route.publication) ||
              !peer[route.publication] ||
              !field_shared_direction_is_active<Location>(
                  directions[route.shared]) ||
              !peer[route.route]) {
            continue;
          }
          Piece piece{};
          LPFX_TRY((try_prepare_fused_field_supplemental_region<Location,
                                                                IndexPolicy>(
              directions[route.publication], directions[route.shared],
              local.cell_extents, local.component_halos[C],
              peers[route.publication].cell_extents,
              peers[route.route].cell_extents, piece)));
          LPFX_TRY(try_append_nonempty_field_piece<in>(segments, route.route, C,
                                                       piece));
        }
        for (std::size_t d = 0; d != 26; ++d) {
          const CartesianDirection3 shared =
              directions[opposite_cartesian_neighbor_index(d)];
          if (!field_shared_direction_is_active<Location>(shared) || !peer[d]) {
            continue;
          }
          Piece piece{};
          LPFX_TRY((try_prepare_fused_field_shared_region<Location, IndexPolicy>(
              shared, peers[d].cell_extents, local.cell_extents, piece)));
          LPFX_TRY(try_append_nonempty_field_piece<out>(segments, d, C, piece));
        }
        for (std::size_t d = 0; d != 26; ++d) {
          const std::size_t transport = opposite_cartesian_neighbor_index(d);
          if (!cartesian_neighbor_mask_contains(publications, d) ||
              !peer[transport]) {
            continue;
          }
          Piece piece{};
          LPFX_TRY((try_prepare_fused_field_primary_region<Location, IndexPolicy>(
              directions[d], peers[transport].cell_extents,
              peers[transport].component_halos[C], local.cell_extents, piece)));
          LPFX_TRY(try_append_nonempty_field_piece<out>(segments, transport, C,
                                                        piece));
        }
        for (const FusedFieldRoute route : fused_field_routes) {
          const std::size_t transport =
              opposite_cartesian_neighbor_index(route.route);
          const std::size_t source =
              opposite_cartesian_neighbor_index(route.shared);
          if (!cartesian_neighbor_mask_contains(publications,
                                                route.publication) ||
              !field_shared_direction_is_active<Location>(
                  directions[route.shared]) ||
              !peer[transport] || !peer[source]) {
            continue;
          }
          Piece piece{};
          LPFX_TRY((try_prepare_fused_field_supplemental_region<Location,
                                                                IndexPolicy>(
              directions[route.publication], directions[route.shared],
              peers[transport].cell_extents,
              peers[transport].component_halos[C], peers[source].cell_extents,
              local.cell_extents, piece)));
          LPFX_TRY(try_append_nonempty_field_piece<out>(segments, transport, C,
                                                        piece));
        }
        return ErrorCode::success;
      });
}

// Remote images are laid out in canonical direction order unless several
// directions share a peer; a peer envelope then orders its sends by the
// receiver's inbound direction. Ghost-to-owner add keeps the layout of the
// copy it inverts, so its outbound flow takes the inbound slot rule.
template <bool Inverse, class Spec, CartesianIndexPolicy IndexPolicy,
          CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode assign_field_image_offsets(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    FieldPieceSegments<Spec, IndexPolicy> &segments) noexcept {
  constexpr std::size_t send = Inverse ? 1 : 0;
  constexpr std::size_t receive = 1 - send;
  std::size_t remote_send{};
  std::size_t remote_receive{};
  std::size_t self_send{};
  std::size_t remote_directions{};
  for (std::size_t d = 0; d != 26; ++d) {
    const FieldPeerKind kind = field_peer_kind<Spec>(topology, d);
    if (!field_peer_kind_has_peer(kind)) {
      continue;
    }
    auto &offsets = segments.image_offsets(d);
    const std::size_t bytes = segments.message_bytes(send, d);
    if (kind == FieldPeerKind::self_peer) {
      offsets[send] = self_send;
      if (!checked_add(self_send, bytes, self_send)) {
        return ErrorCode::overflow;
      }
    } else {
      ++remote_directions;
      offsets[send] = remote_send;
      offsets[receive] = remote_receive;
      if (!checked_add(remote_send, bytes, remote_send) ||
          !checked_add(remote_receive, segments.message_bytes(receive, d),
                       remote_receive)) {
        return ErrorCode::overflow;
      }
    }
  }
  if (remote_directions < 2) {
    return ErrorCode::success;
  }
  PreparedCartesianPeerMap3D<Rank> peers{};
  LPFX_TRY(try_prepare_cartesian_peer_map(topology, Spec::transport_mask, peers));
  if (remote_directions == peers.peer_count()) {
    return ErrorCode::success;
  }
  remote_send = 0;
  remote_receive = 0;
  for (std::uint8_t peer = 0; peer != peers.peer_count(); ++peer) {
    for (std::size_t r = 0; r != 26; ++r) {
      const std::size_t s = opposite_cartesian_neighbor_index(r);
      if (peers.peer_group(s) == peer) {
        segments.image_offsets(s)[send] = remote_send;
        if (!checked_add(remote_send, segments.message_bytes(send, s),
                         remote_send)) {
          return ErrorCode::overflow;
        }
      }
      if (peers.peer_group(r) == peer) {
        segments.image_offsets(r)[receive] = remote_receive;
        if (!checked_add(remote_receive, segments.message_bytes(receive, r),
                         remote_receive)) {
          return ErrorCode::overflow;
        }
      }
    }
  }
  return ErrorCode::success;
}

// A self inbound direction consumes the image its opposite outbound direction
// packs; both must carry identical pieces.
template <bool Inverse, class Spec, CartesianIndexPolicy IndexPolicy,
          CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode bind_field_self_images(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    FieldPieceSegments<Spec, IndexPolicy> &segments) noexcept {
  constexpr std::size_t send = Inverse ? 1 : 0;
  constexpr std::size_t receive = 1 - send;
  for (std::size_t d = 0; d != 26; ++d) {
    if (field_peer_kind<Spec>(topology, d) != FieldPeerKind::self_peer) {
      continue;
    }
    const std::size_t opposite = opposite_cartesian_neighbor_index(d);
    const auto sent = segments.template pieces<FieldFlow::outbound>(d);
    const auto received =
        segments.template pieces<FieldFlow::inbound>(opposite);
    if (field_peer_kind<Spec>(topology, opposite) != FieldPeerKind::self_peer ||
        sent.size() != received.size()) {
      return ErrorCode::invalid_argument;
    }
    for (std::size_t piece = 0; piece != sent.size(); ++piece) {
      if (!(sent[piece] == received[piece])) {
        return ErrorCode::invalid_argument;
      }
    }
    segments.image_offsets(opposite)[receive] =
        segments.image_offsets(d)[send];
  }
  return ErrorCode::success;
}

// Ghost sources and owner destinations stay on their side of each interior.
template <class Spec, CartesianIndexPolicy IndexPolicy, CartesianRank Rank,
          class Geometry>
[[nodiscard]] constexpr ErrorCode validate_ghost_to_owner_add_pieces(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const Geometry &local, const std::array<Geometry, 26> &peers,
    const FieldPieceSegments<Spec, IndexPolicy> &segments) noexcept {
  return meta::for_each_until_error<Spec::component_count>(
      [&](const auto index) constexpr noexcept {
        constexpr std::size_t C = decltype(index)::value;
        using Location = typename Spec::template component_type<C>::location_type;
        CartesianBox3<typename IndexPolicy::coordinate_type> local_interior{};
        LPFX_TRY((try_make_field_interior_box<Location, IndexPolicy>(
            local.cell_extents, local_interior)));
        for (std::size_t d = 0; d != 26; ++d) {
          if (!field_peer_kind_has_peer(field_peer_kind<Spec>(topology, d))) {
            continue;
          }
          CartesianBox3<typename IndexPolicy::coordinate_type> peer_interior{};
          LPFX_TRY((try_make_field_interior_box<Location, IndexPolicy>(
              peers[d].cell_extents, peer_interior)));
          const auto &outbound =
              segments.template pieces<FieldFlow::outbound>(d)[C];
          const auto &inbound = segments.template pieces<FieldFlow::inbound>(d)[C];
          if (cartesian_boxes_overlap(outbound.source, local_interior) ||
              !cartesian_box_contains(local_interior, inbound.destination) ||
              !cartesian_box_contains(peer_interior, outbound.destination) ||
              cartesian_boxes_overlap(inbound.source, peer_interior)) {
            return ErrorCode::invalid_argument;
          }
        }
        return ErrorCode::success;
      });
}

} // namespace detail

// Classifies every canonical direction, builds its oriented pieces and lays
// out their images. Disabled and boundary directions keep no pieces.
template <class Spec, detail::CartesianIndexPolicy IndexPolicy,
          detail::CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode try_prepare_field_pattern(
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const typename PreparedFieldPattern3D<Spec, IndexPolicy, Rank>::geometry_type
        &local_geometry,
    const std::array<typename PreparedFieldPattern3D<Spec, IndexPolicy,
                                                     Rank>::geometry_type,
                     26> &peer_geometries,
    PreparedFieldPattern3D<Spec, IndexPolicy, Rank> &output) noexcept {
  using Pattern = PreparedFieldPattern3D<Spec, IndexPolicy, Rank>;
  constexpr bool inverse =
      Spec::kind == distributed::field_operation_kind::ghost_to_owner_add;
  if (!topology.is_valid() ||
      !detail::cartesian_extent_is_nonempty(local_geometry.cell_extents)) {
    return ErrorCode::invalid_argument;
  }
  for (std::size_t d = 0; d != Pattern::direction_count; ++d) {
    const auto &neighbor = topology.neighbor_by_index(d);
    if (!cartesian_neighbor_mask_contains(Spec::transport_mask, d) ||
        !neighbor.has_peer()) {
      continue;
    }
    const auto &peer = peer_geometries[d];
    if (!detail::cartesian_extent_is_nonempty(peer.cell_extents) ||
        (neighbor.is_self_peer() && peer != local_geometry)) {
      return ErrorCode::invalid_argument;
    }
  }

  detail::FieldPieceSegments<Spec, IndexPolicy> segments{};
  if constexpr (Spec::fused) {
    LPFX_TRY(detail::try_prepare_fused_field_pieces(topology, local_geometry,
                                                    peer_geometries, segments));
  } else {
    LPFX_TRY(detail::try_prepare_affine_field_pieces(topology, local_geometry,
                                                     peer_geometries, segments));
  }
  LPFX_TRY(detail::assign_field_image_offsets<inverse>(topology, segments));
  LPFX_TRY(detail::bind_field_self_images<inverse>(topology, segments));
  if constexpr (inverse) {
    LPFX_TRY(detail::validate_ghost_to_owner_add_pieces(
        topology, local_geometry, peer_geometries, segments));
  }
  output = detail::PrivateConstructionAccess::create<Pattern>(
      topology, local_geometry, segments);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
