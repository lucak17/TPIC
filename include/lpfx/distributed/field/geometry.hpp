#ifndef LPFX_DISTRIBUTED_FIELD_GEOMETRY_HPP
#define LPFX_DISTRIBUTED_FIELD_GEOMETRY_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/descriptor/field.hpp>
#include <lpfx/domain/cartesian/box.hpp>
#include <lpfx/domain/cartesian/halo.hpp>
#include <lpfx/domain/cartesian/index_policy.hpp>
#include <lpfx/domain/cartesian/location.hpp>

namespace lpfx {

// One oriented piece of a field operation: samples of the source box are
// copied or added into the destination box. Remote boxes are in the peer's
// frame. Affine pieces keep the default role and zero directions.
template <detail::CartesianIndexPolicy IndexPolicy> struct FieldPiece final {
  using coordinate_type = typename IndexPolicy::coordinate_type;
  using count_type = typename IndexPolicy::count_type;

  CartesianBox3<coordinate_type> source{};
  CartesianBox3<coordinate_type> destination{};
  count_type samples{};
  std::uint16_t component{};
  FusedFieldPieceRole role{};
  CartesianDirection3 publication{};
  CartesianDirection3 shared{};

  [[nodiscard]] friend constexpr bool operator==(const FieldPiece &,
                                                 const FieldPiece &) noexcept =
      default;
};

namespace detail {

template <std::size_t Axis>
[[nodiscard]] constexpr std::int8_t
direction_axis(const CartesianDirection3 direction) noexcept {
  if constexpr (Axis == 0) {
    return direction.x;
  } else if constexpr (Axis == 1) {
    return direction.y;
  } else {
    return direction.z;
  }
}

// Box [0, samples) of a location's interior samples.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_make_field_interior_box(
    const CartesianExtent3<typename IndexPolicy::count_type> cells,
    CartesianBox3<typename IndexPolicy::coordinate_type> &output) noexcept {
  using Coordinate = typename IndexPolicy::coordinate_type;
  CartesianExtent3<typename IndexPolicy::count_type> samples{};
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(cells,
                                                                    samples)));
  return try_make_cartesian_box3(Coordinate{0}, Coordinate{0}, Coordinate{0},
                                 samples.x(), samples.y(), samples.z(), output);
}

// The local ghost region of direction and the peer samples that fill it:
// destination is the local ghost box, source the peer box in its frame.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_field_copy_region(
    const CartesianDirection3 direction,
    const CartesianExtent3<typename IndexPolicy::count_type> local_cells,
    const HaloDepth3<typename IndexPolicy::count_type> local_halo,
    const CartesianExtent3<typename IndexPolicy::count_type> peer_cells,
    FieldPiece<IndexPolicy> &output) noexcept {
  using Count = typename IndexPolicy::count_type;
  using Coordinate = typename IndexPolicy::coordinate_type;
  using Axis = CartesianBoxPairAxis<Coordinate, Count>;
  CartesianExtent3<Count> local{};
  CartesianExtent3<Count> peer{};
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(local_cells,
                                                                    local)));
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(peer_cells,
                                                                    peer)));
  std::array<Axis, 3> axes{};
  LPFX_TRY(for_each_axis([&](const auto axis_index) constexpr noexcept {
    constexpr std::size_t A = decltype(axis_index)::value;
    const std::int8_t side = direction_axis<A>(direction);
    const Count local_samples = local.template get<A>();
    const Count peer_samples = peer.template get<A>();
    Axis &axis = axes[A];
    if (side == 0) {
      Coordinate upper{};
      if (local_samples != peer_samples) {
        return ErrorCode::invalid_argument;
      }
      if (!checked_integral_cast(local_samples, upper)) {
        return ErrorCode::overflow;
      }
      axis = {0, upper, 0, upper, local_samples};
      return ErrorCode::success;
    }
    // The peer source skips its node endpoint adjacent to the receiver. A
    // full-peer-width halo may reach the far endpoint.
    const Count width = local_halo.template get<A>(side);
    const Count inset = static_cast<Count>(Location::template is_node<A>);
    Count local_end{};
    Count peer_lower{};
    Count peer_upper{};
    bool local_ok = false;
    bool peer_ok = false;
    if (side < 0) {
      axis.first_upper = 0;
      local_ok = try_make_negative_coordinate(width, axis.first_lower);
      peer_ok = checked_subtract(peer_samples, inset, peer_upper) &&
                width <= peer_upper &&
                checked_subtract(peer_upper, width, peer_lower) &&
                checked_integral_cast(peer_lower, axis.second_lower) &&
                checked_integral_cast(peer_upper, axis.second_upper);
    } else {
      local_ok = checked_add(local_samples, width, local_end) &&
                 checked_integral_cast(local_samples, axis.first_lower) &&
                 checked_integral_cast(local_end, axis.first_upper);
      peer_ok = checked_subtract(peer_samples, inset, peer_upper) &&
                width <= peer_upper &&
                checked_integral_cast(inset, axis.second_lower) &&
                checked_add(inset, width, peer_lower) &&
                checked_integral_cast(peer_lower, axis.second_upper);
    }
    if (!peer_ok) {
      return ErrorCode::capacity_exceeded;
    }
    if (!local_ok) {
      return ErrorCode::overflow;
    }
    axis.count = width;
    return ErrorCode::success;
  }));
  return try_make_cartesian_box_pair(axes[0], axes[1], axes[2],
                                     output.destination, output.source,
                                     output.samples);
}

// Samples shared by local and peer across direction: destination is the
// local box, source the peer box in its frame. An inactive direction for
// Location is an empty region.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_shared_sample_add_region(
    const CartesianDirection3 direction,
    const CartesianExtent3<typename IndexPolicy::count_type> local_cells,
    const CartesianExtent3<typename IndexPolicy::count_type> peer_cells,
    FieldPiece<IndexPolicy> &output) noexcept {
  using Count = typename IndexPolicy::count_type;
  using Coordinate = typename IndexPolicy::coordinate_type;
  using Axis = CartesianBoxPairAxis<Coordinate, Count>;
  CartesianExtent3<Count> local{};
  CartesianExtent3<Count> peer{};
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(local_cells,
                                                                    local)));
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(peer_cells,
                                                                    peer)));
  if (!field_shared_direction_is_active<Location>(direction)) {
    output.source = {};
    output.destination = {};
    output.samples = 0;
    return ErrorCode::success;
  }
  std::array<Axis, 3> axes{};
  LPFX_TRY(for_each_axis([&](const auto axis_index) constexpr noexcept {
    constexpr std::size_t A = decltype(axis_index)::value;
    const std::int8_t side = direction_axis<A>(direction);
    const Count local_samples = local.template get<A>();
    const Count peer_samples = peer.template get<A>();
    Count first_lower{};
    Count first_upper{};
    Count second_lower{};
    Count second_upper{};
    Axis &axis = axes[A];
    if (side == 0) {
      if (local_samples != peer_samples) {
        return ErrorCode::invalid_argument;
      }
      first_upper = local_samples;
      second_upper = peer_samples;
      axis.count = local_samples;
    } else if (side < 0) {
      if (!checked_subtract(peer_samples, Count{1}, second_lower)) {
        return ErrorCode::overflow;
      }
      first_upper = Count{1};
      second_upper = peer_samples;
      axis.count = Count{1};
    } else {
      if (!checked_subtract(local_samples, Count{1}, first_lower)) {
        return ErrorCode::overflow;
      }
      first_upper = local_samples;
      second_upper = Count{1};
      axis.count = Count{1};
    }
    return checked_integral_cast(first_lower, axis.first_lower) &&
                   checked_integral_cast(first_upper, axis.first_upper) &&
                   checked_integral_cast(second_lower, axis.second_lower) &&
                   checked_integral_cast(second_upper, axis.second_upper)
               ? ErrorCode::success
               : ErrorCode::overflow;
  }));
  return try_make_cartesian_box_pair(axes[0], axes[1], axes[2],
                                     output.destination, output.source,
                                     output.samples);
}

// A fused publication must leave the node endpoint of every node axis it
// crosses outside the receiver's halo.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_validate_fused_field_publication(
    const CartesianDirection3 publication,
    const HaloDepth3<typename IndexPolicy::count_type> receiver_halo,
    const CartesianExtent3<typename IndexPolicy::count_type>
        publication_source_cells) noexcept {
  using Count = typename IndexPolicy::count_type;
  CartesianExtent3<Count> samples{};
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(
      publication_source_cells, samples)));
  bool shallow = true;
  for_each_axis([&](const auto axis_index) constexpr noexcept {
    constexpr std::size_t A = decltype(axis_index)::value;
    const std::int8_t side = direction_axis<A>(publication);
    if (side == 0 || !Location::template is_node<A>) {
      return;
    }
    const Count width = receiver_halo.template get<A>(side);
    Count available{};
    shallow = shallow &&
              checked_subtract(samples.template get<A>(), Count{1}, available) &&
              (width == 0 || width < available);
  });
  return shallow ? ErrorCode::success : ErrorCode::capacity_exceeded;
}

// Receiver-frame shared-add region of a fused operation.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_fused_field_shared_region(
    const CartesianDirection3 shared,
    const CartesianExtent3<typename IndexPolicy::count_type> receiver_cells,
    const CartesianExtent3<typename IndexPolicy::count_type> source_cells,
    FieldPiece<IndexPolicy> &output) noexcept {
  LPFX_TRY((try_prepare_shared_sample_add_region<Location, IndexPolicy>(
      shared, receiver_cells, source_cells, output)));
  output.role = FusedFieldPieceRole::shared_add;
  output.publication = {};
  output.shared = shared;
  return ErrorCode::success;
}

// Receiver-frame primary ghost copy of a fused operation.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_fused_field_primary_region(
    const CartesianDirection3 publication,
    const CartesianExtent3<typename IndexPolicy::count_type> receiver_cells,
    const HaloDepth3<typename IndexPolicy::count_type> receiver_halo,
    const CartesianExtent3<typename IndexPolicy::count_type>
        publication_source_cells,
    FieldPiece<IndexPolicy> &output) noexcept {
  LPFX_TRY((try_validate_fused_field_publication<Location, IndexPolicy>(
      publication, receiver_halo, publication_source_cells)));
  LPFX_TRY((try_prepare_field_copy_region<Location, IndexPolicy>(
      publication, receiver_cells, receiver_halo, publication_source_cells,
      output)));
  output.role = FusedFieldPieceRole::primary_copy;
  output.publication = publication;
  output.shared = {};
  return ErrorCode::success;
}

// Receiver-frame supplemental add: the contributor's shared samples that the
// publication source has not yet summed when it publishes into the
// receiver's ghost. Cold and statically specialized; kept out of line so the
// component builders are not cloned around it.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] LPFX_NOINLINE constexpr ErrorCode
try_prepare_fused_field_supplemental_region(
    const CartesianDirection3 publication, const CartesianDirection3 shared,
    const CartesianExtent3<typename IndexPolicy::count_type> receiver_cells,
    const HaloDepth3<typename IndexPolicy::count_type> receiver_halo,
    const CartesianExtent3<typename IndexPolicy::count_type>
        publication_source_cells,
    const CartesianExtent3<typename IndexPolicy::count_type>
        contributor_source_cells,
    FieldPiece<IndexPolicy> &output) noexcept {
  LPFX_TRY((try_validate_fused_field_publication<Location, IndexPolicy>(
      publication, receiver_halo, publication_source_cells)));
  FieldPiece<IndexPolicy> copy{};
  LPFX_TRY((try_prepare_field_copy_region<Location, IndexPolicy>(
      publication, receiver_cells, receiver_halo, publication_source_cells,
      copy)));
  FieldPiece<IndexPolicy> contribution{};
  LPFX_TRY((try_prepare_shared_sample_add_region<Location, IndexPolicy>(
      shared, publication_source_cells, contributor_source_cells,
      contribution)));
  output.role = FusedFieldPieceRole::supplemental_add;
  output.publication = publication;
  output.shared = shared;
  CartesianBox3<typename IndexPolicy::coordinate_type> intersection{};
  typename IndexPolicy::count_type samples{};
  LPFX_TRY(try_intersect_cartesian_boxes(copy.source, contribution.destination,
                                         intersection, samples));
  if (samples == 0) {
    output.source = {};
    output.destination = {};
    output.samples = 0;
    return ErrorCode::success;
  }
  LPFX_TRY(try_translate_cartesian_box(
      intersection, contribution.destination.lower(),
      contribution.source.lower(), output.source));
  LPFX_TRY(try_translate_cartesian_box(intersection, copy.source.lower(),
                                       copy.destination.lower(),
                                       output.destination));
  output.samples = samples;
  return ErrorCode::success;
}

} // namespace detail
} // namespace lpfx

#endif
