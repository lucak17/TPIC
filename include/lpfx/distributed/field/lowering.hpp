#ifndef LPFX_DISTRIBUTED_FIELD_LOWERING_HPP
#define LPFX_DISTRIBUTED_FIELD_LOWERING_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/distributed/detail/byte_interval.hpp>
#include <lpfx/distributed/field/owner.hpp>
#include <lpfx/distributed/field/pattern.hpp>
#include <lpfx/domain/cartesian/halo.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx {

// The pass type owns semantic meaning; a prepared work record retains only
// geometry, offsets and the topology route selected during preparation.
enum class FieldWorkAction : std::uint8_t { pack, overwrite, add };
enum class FieldSemanticPass : std::uint8_t {
  source, overwrite, add, shared_add, supplemental_add
};
enum class FieldSourceRequirement : std::uint8_t {
  field, remote_image, immutable_snapshot
};
enum class FieldRuntimeRouteClass : std::uint8_t {
  remote, self_direct, self_snapshot
};

struct PreparedFieldRoute final {
  FieldRuntimeRouteClass route{};
  std::size_t image_offset_bytes{};
};

namespace detail {

template <FieldSemanticPass... Passes>
struct FieldPassSequence {
  using pass_sequence = FieldPassSequence;
  static constexpr std::size_t pass_count = sizeof...(Passes);
  template <FieldSemanticPass Pass>
  static constexpr bool pass_is_active = ((Pass == Passes) || ...);
  template <FieldSemanticPass Pass>
  static constexpr std::size_t pass_index = [] {
    constexpr std::array passes{Passes...};
    for (std::size_t i = 0; i != passes.size(); ++i)
      if (passes[i] == Pass) return i;
    return passes.size();
  }();

  template <class Visitor>
  [[nodiscard]] static constexpr ErrorCode
  try_for_each_active_pass(Visitor &&visitor) noexcept {
    ErrorCode failure = ErrorCode::success;
    const auto visit = [&]<FieldSemanticPass Pass>() constexpr noexcept {
      if (failure == ErrorCode::success)
        failure = visitor(std::integral_constant<FieldSemanticPass, Pass>{});
    };
    (visit.template operator()<Passes>(), ...);
    return failure;
  }
};

template <distributed::field_operation_kind Kind>
struct FieldSchedulePolicy;

template <>
struct FieldSchedulePolicy<distributed::field_operation_kind::copy_to_ghost>
    : FieldPassSequence<FieldSemanticPass::source, FieldSemanticPass::overwrite> {};
template <>
struct FieldSchedulePolicy<distributed::field_operation_kind::ghost_to_owner_add>
    : FieldPassSequence<FieldSemanticPass::source, FieldSemanticPass::add> {};
template <>
struct FieldSchedulePolicy<distributed::field_operation_kind::shared_sample_add>
    : FieldPassSequence<FieldSemanticPass::source, FieldSemanticPass::add> {};
template <>
struct FieldSchedulePolicy<
    distributed::field_operation_kind::shared_sample_add_copy_to_ghost>
    : FieldPassSequence<FieldSemanticPass::source, FieldSemanticPass::overwrite,
                        FieldSemanticPass::shared_add,
                        FieldSemanticPass::supplemental_add> {};

// Both host and device lowerings instantiate this mapping directly. Route is
// runtime topology data; a lowering may dispatch once per prepared box/block
// to these typed cases. It must not invent a second action/source policy.
template <distributed::field_operation_kind Kind, FieldSemanticPass Pass>
struct FieldPassTraits final {
  static_assert(FieldSchedulePolicy<Kind>::template pass_is_active<Pass>);
  template <FieldRuntimeRouteClass Route>
  static constexpr bool route_is_supported = [] {
    if constexpr (Route == FieldRuntimeRouteClass::remote)
      return true;
    else if constexpr (Route == FieldRuntimeRouteClass::self_snapshot)
      return Kind != distributed::field_operation_kind::ghost_to_owner_add;
    else
      return (Kind == distributed::field_operation_kind::copy_to_ghost &&
              Pass == FieldSemanticPass::source) ||
             (Kind == distributed::field_operation_kind::ghost_to_owner_add &&
              Pass == FieldSemanticPass::add);
  }();
  template <FieldRuntimeRouteClass Route>
  static constexpr FieldWorkAction action = [] {
    if constexpr (Pass == FieldSemanticPass::source) {
      if constexpr (Kind == distributed::field_operation_kind::copy_to_ghost &&
                    Route == FieldRuntimeRouteClass::self_direct)
        return FieldWorkAction::overwrite;
      return FieldWorkAction::pack;
    } else if constexpr (Pass == FieldSemanticPass::overwrite) {
      return FieldWorkAction::overwrite;
    } else {
      return FieldWorkAction::add;
    }
  }();
  template <FieldRuntimeRouteClass Route>
  static constexpr FieldSourceRequirement source = [] {
    if constexpr (Pass == FieldSemanticPass::source ||
                  Route == FieldRuntimeRouteClass::self_direct)
      return FieldSourceRequirement::field;
    else if constexpr (Route == FieldRuntimeRouteClass::self_snapshot)
      return FieldSourceRequirement::immutable_snapshot;
    else
      return FieldSourceRequirement::remote_image;
  }();
};

template <FieldSemanticPass Pass>
inline constexpr FieldFlow field_pass_flow =
    Pass == FieldSemanticPass::source ? FieldFlow::outbound : FieldFlow::inbound;

// Execution images: 0 outgoing, 1 incoming, 2 self snapshot.
template <FieldSemanticPass Pass>
[[nodiscard]] constexpr std::size_t
field_image_index(const FieldRuntimeRouteClass route) noexcept {
  return route == FieldRuntimeRouteClass::self_snapshot
             ? 2
             : (Pass == FieldSemanticPass::source ? 0 : 1);
}

struct NoFieldSelfCopy final {};

// Preparation-only route facts. Copy-to-ghost self routes keep a snapshot
// offset per component and outbound direction; direct routes hold the
// sentinel, which a nonempty snapshot offset cannot equal because its checked
// end must also fit size_t.
template <class Spec> struct PreparedFieldSchedule final {
  static constexpr bool copy =
      Spec::kind == distributed::field_operation_kind::copy_to_ghost;
  static constexpr std::size_t direct = std::numeric_limits<std::size_t>::max();
  [[no_unique_address]] std::conditional_t<
      copy, std::array<std::array<std::size_t, 26>, Spec::component_count>,
      NoFieldSelfCopy> self_copy{};
  std::size_t snapshot_bytes{};
};

// A direct write may share a launch with every outbound read. It is safe only
// when it cannot change any source and cannot race another direct
// destination. All other self routes are captured into immutable storage.
// Hard precondition: outbound is a self-peer direction.
template <std::size_t C, class Pattern>
[[nodiscard]] constexpr FieldRuntimeRouteClass
classify_copy_to_ghost_self_route(const Pattern &pattern,
                                  const std::size_t outbound) noexcept {
  constexpr FieldFlow out = FieldFlow::outbound;
  constexpr FieldFlow in = FieldFlow::inbound;
  const auto &source = pattern.direction(outbound).template pieces<out>()[C];
  if (Pattern::piece_bytes(source) == 0) {
    return FieldRuntimeRouteClass::self_direct;
  }
  const auto &destination =
      pattern.direction(opposite_cartesian_neighbor_index(outbound))
          .template pieces<in>()[C];
  for (std::size_t other = 0; other != Pattern::direction_count; ++other) {
    const auto view = pattern.direction(other);
    if (!view.has_peer()) {
      continue;
    }
    const auto &other_source = view.template pieces<out>()[C];
    if (Pattern::piece_bytes(other_source) != 0 &&
        cartesian_boxes_overlap(destination.destination, other_source.source)) {
      return FieldRuntimeRouteClass::self_snapshot;
    }
    if (!view.is_self_peer() || other == outbound) {
      continue;
    }
    const auto &other_destination =
        pattern.direction(opposite_cartesian_neighbor_index(other))
            .template pieces<in>()[C];
    if (Pattern::piece_bytes(other_destination) != 0 &&
        cartesian_boxes_overlap(destination.destination,
                                other_destination.destination)) {
      return FieldRuntimeRouteClass::self_snapshot;
    }
  }
  return FieldRuntimeRouteClass::self_direct;
}

template <class Pattern>
[[nodiscard]] constexpr ErrorCode try_prepare_field_schedule(
    const Pattern &pattern,
    PreparedFieldSchedule<typename Pattern::spec_type> &output) noexcept {
  using Schedule = PreparedFieldSchedule<typename Pattern::spec_type>;
  if (!pattern.is_valid()) {
    return ErrorCode::invalid_argument;
  }
  output = {};
  if constexpr (Schedule::copy) {
    return meta::for_each_until_error<Pattern::component_count>(
        [&](const auto index) constexpr noexcept {
          constexpr std::size_t C = decltype(index)::value;
          auto &offsets = output.self_copy[C];
          offsets.fill(Schedule::direct);
          for (std::size_t d = 0; d != Pattern::direction_count; ++d) {
            const auto direction = pattern.direction(d);
            if (!direction.is_self_peer() ||
                classify_copy_to_ghost_self_route<C>(pattern, d) !=
                    FieldRuntimeRouteClass::self_snapshot) {
              continue;
            }
            offsets[d] = output.snapshot_bytes;
            if (!checked_add(output.snapshot_bytes,
                             Pattern::piece_bytes(
                                 direction.template pieces<FieldFlow::outbound>()[C]),
                             output.snapshot_bytes)) {
              return ErrorCode::overflow;
            }
          }
          return ErrorCode::success;
        });
  } else if constexpr (Pattern::spec_type::kind !=
                       distributed::field_operation_kind::ghost_to_owner_add) {
    output.snapshot_bytes = pattern.self_image_bytes();
  }
  return ErrorCode::success;
}

template <FieldSemanticPass Pass, std::size_t C, class Pattern, class Sink>
[[nodiscard]] constexpr ErrorCode try_emit_field_route(
    const Pattern &pattern,
    const PreparedFieldSchedule<typename Pattern::spec_type> &schedule,
    const std::size_t d, const typename Pattern::piece_type &piece,
    std::size_t image_offset, Sink &sink) noexcept {
  using Schedule = PreparedFieldSchedule<typename Pattern::spec_type>;
  FieldRuntimeRouteClass route = FieldRuntimeRouteClass::remote;
  if (pattern.peer_kind(d) == FieldPeerKind::self_peer) {
    if constexpr (Schedule::copy) {
      const std::size_t outbound = Pass == FieldSemanticPass::source
                                       ? d
                                       : opposite_cartesian_neighbor_index(d);
      const std::size_t offset = schedule.self_copy[C][outbound];
      route = offset == Schedule::direct ? FieldRuntimeRouteClass::self_direct
                                         : FieldRuntimeRouteClass::self_snapshot;
      if constexpr (Pass == FieldSemanticPass::overwrite) {
        if (route == FieldRuntimeRouteClass::self_direct) {
          return ErrorCode::success;
        }
      }
      if (route == FieldRuntimeRouteClass::self_snapshot) {
        image_offset = offset;
      }
    } else if constexpr (Pattern::spec_type::kind ==
                         distributed::field_operation_kind::ghost_to_owner_add) {
      // Ghost sources and owner destinations occupy disjoint regions. Each
      // inbound route contributes once after all remote sources are captured.
      if constexpr (Pass == FieldSemanticPass::source) {
        return ErrorCode::success;
      }
      route = FieldRuntimeRouteClass::self_direct;
    } else {
      route = FieldRuntimeRouteClass::self_snapshot;
    }
  }
  return sink(meta::index_constant<C>{}, d, piece,
              PreparedFieldRoute{route, image_offset});
}

// Emits the accepted pieces of one direction in stored order. A piece's image
// offset counts every preceding piece of the direction, of any component or
// role; the pattern proved these sums representable.
template <FieldSemanticPass Pass, std::size_t C, class Pattern, class Accept,
          class Sink>
[[nodiscard]] constexpr ErrorCode try_emit_field_direction(
    const Pattern &pattern,
    const PreparedFieldSchedule<typename Pattern::spec_type> &schedule,
    const std::size_t d, const Accept &accept, Sink &sink) noexcept {
  constexpr FieldFlow flow = field_pass_flow<Pass>;
  const auto direction = pattern.direction(d);
  if (!direction.has_peer()) {
    return ErrorCode::success;
  }
  std::size_t offset = direction.template image_offset_bytes<flow>();
  for (const auto &piece : direction.template pieces<flow>()) {
    const std::size_t bytes = Pattern::piece_bytes(piece);
    if (piece.component == C && bytes != 0 && accept(piece)) {
      LPFX_TRY((try_emit_field_route<Pass, C>(pattern, schedule, d, piece,
                                              offset, sink)));
    }
    offset += bytes;
  }
  return ErrorCode::success;
}

// The one work traversal: pass (caller) -> component -> canonical direction
// (supplemental pieces nested in canonical (publication, shared) order) ->
// stored piece. sink(index_constant<C>, direction, piece, route) -> ErrorCode.
template <FieldSemanticPass Pass, class Pattern, class Sink>
[[nodiscard]] constexpr ErrorCode try_for_each_field_work(
    const Pattern &pattern,
    const PreparedFieldSchedule<typename Pattern::spec_type> &schedule,
    Sink &&sink) noexcept {
  using Spec = typename Pattern::spec_type;
  using Piece = typename Pattern::piece_type;
  return meta::for_each_until_error<Spec::component_count>(
      [&](const auto index) constexpr noexcept {
        constexpr std::size_t C = decltype(index)::value;
        if constexpr (Pass == FieldSemanticPass::supplemental_add) {
          for (const FusedFieldRoute route : fused_field_routes) {
            const CartesianDirection3 publication =
                canonical_cartesian_neighbor_directions[route.publication];
            const CartesianDirection3 shared =
                canonical_cartesian_neighbor_directions[route.shared];
            const auto accept = [&](const Piece &piece) constexpr noexcept {
              return piece.role == FusedFieldPieceRole::supplemental_add &&
                     piece.publication == publication && piece.shared == shared;
            };
            LPFX_TRY((try_emit_field_direction<Pass, C>(pattern, schedule,
                                                        route.route, accept,
                                                        sink)));
          }
        } else {
          for (std::size_t d = 0; d != Pattern::direction_count; ++d) {
            const auto accept = [&](const Piece &piece) constexpr noexcept {
              if constexpr (Spec::fused && Pass == FieldSemanticPass::overwrite)
                return piece.role == FusedFieldPieceRole::primary_copy;
              else if constexpr (Pass == FieldSemanticPass::shared_add)
                return piece.role == FusedFieldPieceRole::shared_add &&
                       piece.shared ==
                           canonical_cartesian_neighbor_directions[d];
              else
                return true;
            };
            LPFX_TRY((try_emit_field_direction<Pass, C>(pattern, schedule, d,
                                                        accept, sink)));
          }
        }
        return ErrorCode::success;
      });
}

template <class Coordinate> struct FieldRouteBoxes final {
  CartesianBox3<Coordinate> source{};
  CartesianBox3<Coordinate> destination{};
};

// Field-frame boxes of one emitted route. A direct self route reads or writes
// the local box of its opposite-direction piece instead of an image.
template <FieldSemanticPass Pass, std::size_t C, class Pattern>
[[nodiscard]] constexpr auto resolve_field_route_box(
    const Pattern &pattern, const std::size_t d,
    const typename Pattern::piece_type &piece,
    const PreparedFieldRoute route) noexcept {
  constexpr auto kind = Pattern::spec_type::kind;
  FieldRouteBoxes<typename Pattern::index_policy::coordinate_type> boxes{
      piece.source, piece.destination};
  if constexpr ((kind == distributed::field_operation_kind::copy_to_ghost &&
                 Pass == FieldSemanticPass::source) ||
                (kind == distributed::field_operation_kind::ghost_to_owner_add &&
                 Pass == FieldSemanticPass::add)) {
    if (route.route == FieldRuntimeRouteClass::self_direct) {
      const auto opposite =
          pattern.direction(opposite_cartesian_neighbor_index(d));
      if constexpr (Pass == FieldSemanticPass::source) {
        boxes.destination =
            opposite.template pieces<FieldFlow::inbound>()[C].destination;
      } else {
        boxes.source = opposite.template pieces<FieldFlow::outbound>()[C].source;
      }
    }
  }
  return boxes;
}

// True when some remote direction exchanges bytes.
template <class Pattern>
[[nodiscard]] constexpr bool
field_needs_native_work(const Pattern &pattern) noexcept {
  for (std::size_t d = 0; d != Pattern::direction_count; ++d) {
    const auto direction = pattern.direction(d);
    if (direction.peer_kind() == FieldPeerKind::remote_peer &&
        (direction.template message_bytes<FieldFlow::outbound>() != 0 ||
         direction.template message_bytes<FieldFlow::inbound>() != 0)) {
      return true;
    }
  }
  return false;
}

} // namespace detail
} // namespace lpfx

namespace lpfx::distributed::detail {

struct FieldExecutionImages final {
  std::span<std::byte> outgoing;
  std::span<std::byte> incoming;
};

template <class Spec>
inline constexpr bool field_execution_has_direct_routes =
    []<FieldSemanticPass... Passes>(
        lpfx::detail::FieldPassSequence<Passes...>) constexpr noexcept {
      return (lpfx::detail::FieldPassTraits<Spec::kind, Passes>::
                  template route_is_supported<
                      FieldRuntimeRouteClass::self_direct> ||
              ...);
    }(typename lpfx::detail::FieldSchedulePolicy<Spec::kind>::pass_sequence{});

// Both concrete lowerings use the same cold storage and geometry admission.
// There is no action or source requirement in these records: those facts
// remain in FieldPassTraits and the typed pass/component array index.
template <class Spec> struct FieldExecutionLayout final {
  using schedule_type = lpfx::detail::FieldSchedulePolicy<Spec::kind>;
  static constexpr std::size_t bucket_count =
      schedule_type::pass_count * Spec::component_count;
  lpfx::detail::PreparedFieldSchedule<Spec> schedule{};
  std::array<std::size_t, bucket_count> counts{};
  std::array<std::size_t, bucket_count> offsets{};
  std::array<std::size_t, 3> image_bytes{};
  std::array<std::size_t, 3> image_offsets{};
  std::size_t work_bytes{};
  std::size_t total_bytes{};

  template <FieldSemanticPass Pass, std::size_t Component>
  static constexpr std::size_t bucket_index =
      schedule_type::template pass_index<Pass> * Spec::component_count +
      Component;
};

[[nodiscard]] constexpr bool append_field_region(const std::size_t bytes,
                                                 const std::size_t alignment,
                                                 std::size_t &cursor,
                                                 std::size_t &offset) noexcept {
  if (bytes == 0) {
    offset = cursor;
    return true;
  }
  return lpfx::detail::checked_align_up(cursor, alignment, offset) &&
         lpfx::detail::checked_add(offset, bytes, cursor);
}

// ImageAlignment aligns each image region's start within the allocation.
// Message offsets and sizes live inside an image and are independent of it.
// Host images are byte streams (1); device lowering requests typed access.
template <class Work, std::size_t ImageAlignment = 1, class Pattern>
[[nodiscard]] constexpr ErrorCode prepare_field_execution_layout(
    const Pattern &pattern,
    FieldExecutionLayout<typename Pattern::spec_type> &layout) noexcept {
  static_assert(std::has_single_bit(ImageAlignment));
  using Layout = FieldExecutionLayout<typename Pattern::spec_type>;
  LPFX_TRY(lpfx::detail::try_prepare_field_schedule(pattern, layout.schedule));
  LPFX_TRY(Layout::schedule_type::try_for_each_active_pass(
      [&]<FieldSemanticPass Pass>(
          std::integral_constant<FieldSemanticPass, Pass>) constexpr noexcept {
        return lpfx::detail::try_for_each_field_work<Pass>(
            pattern, layout.schedule,
            [&](const auto component, std::size_t, const auto &,
                PreparedFieldRoute) constexpr noexcept {
              auto &count =
                  layout.counts[Layout::template bucket_index<Pass,
                                                              component.value>];
              return lpfx::detail::checked_add(count, std::size_t{1}, count)
                         ? ErrorCode::success
                         : ErrorCode::overflow;
            });
      }));
  for (std::size_t bucket = 0; bucket != Layout::bucket_count; ++bucket) {
    std::size_t bytes{};
    if (!lpfx::detail::checked_byte_count<Work>(layout.counts[bucket], bytes) ||
        !append_field_region(bytes, alignof(Work), layout.work_bytes,
                             layout.offsets[bucket])) {
      return ErrorCode::overflow;
    }
  }
  layout.image_bytes[0] =
      pattern.template remote_image_bytes<FieldFlow::outbound>();
  layout.image_bytes[1] =
      pattern.template remote_image_bytes<FieldFlow::inbound>();
  layout.image_bytes[2] = layout.schedule.snapshot_bytes;
  layout.total_bytes = layout.work_bytes;
  for (std::size_t image = 0; image != 3; ++image) {
    if (!append_field_region(layout.image_bytes[image], ImageAlignment,
                             layout.total_bytes, layout.image_offsets[image])) {
      return ErrorCode::overflow;
    }
  }
  return ErrorCode::success;
}

template <class Work>
inline constexpr std::size_t field_host_allocation_alignment =
    alignof(Work) > DefaultAllocator<HostSpace>::minimum_data_alignment
        ? alignof(Work)
        : DefaultAllocator<HostSpace>::minimum_data_alignment;

template <class Work>
using FieldHostAllocation = lpfx::detail::RawHostAllocation<
    HostAlignedAllocator<field_host_allocation_alignment<Work>>>;

template <class Work>
[[nodiscard]] Result<FieldHostAllocation<Work>>
allocate_field_host_storage(HostMemoryBudget &budget,
                            const std::size_t bytes) noexcept {
  using Allocator = HostAlignedAllocator<field_host_allocation_alignment<Work>>;
  using Output = Result<FieldHostAllocation<Work>>;
  HostMemoryRequirement requirement{};
  if (!requirement.try_add(bytes, Allocator::minimum_data_alignment)) {
    return Output::from_error(ErrorCode::overflow);
  }
  auto admission = budget.try_admit(requirement);
  LPFX_TRY_RESULT(admission, Output);
  return lpfx::detail::allocate_host_raw<Allocator>(
      std::move(admission).value(), bytes);
}

template <class Work>
[[nodiscard]] Work *start_field_work_array(std::byte *const base,
                                           const std::size_t offset,
                                           const std::size_t count) noexcept {
  static_assert(std::is_trivially_copyable_v<Work> &&
                std::is_trivially_destructible_v<Work> &&
                std::is_aggregate_v<Work>);
  if (count == 0) {
    return nullptr;
  }
  return std::launder(static_cast<Work *>(
      std::memmove(base + offset, base + offset, count * sizeof(Work))));
}

template <class Spec>
[[nodiscard]] std::span<std::byte>
field_execution_image(std::byte *const base,
                      const FieldExecutionLayout<Spec> &layout,
                      const std::size_t image) noexcept {
  return {layout.image_bytes[image] == 0 ? nullptr
                                         : base + layout.image_offsets[image],
          layout.image_bytes[image]};
}

struct FieldExecutionBoxShape final {
  std::size_t x{}, y{}, z{};
  std::size_t stride_y_bytes{}, stride_z_bytes{};
  std::size_t bytes{};
};

// Allocation geometry of one owner component, prepared once from host
// metadata even when the component addresses device storage.
template <lpfx::detail::CartesianIndexPolicy Policy>
struct FieldComponentGeometry final {
  CartesianBox3<typename Policy::coordinate_type> allocated{};
  std::size_t stride_y_bytes{};
  std::size_t stride_z_bytes{};
  bool has_core{};
};

template <class Pattern>
using FieldComponentGeometries =
    std::array<FieldComponentGeometry<typename Pattern::index_policy>,
               Pattern::component_count>;

template <std::size_t C, class Pattern, class Owner, class View>
[[nodiscard]] constexpr ErrorCode prepare_field_component_geometry(
    const Pattern &pattern, const Owner &owner, const View component,
    FieldComponentGeometry<typename Pattern::index_policy> &output) noexcept {
  using Policy = typename Pattern::index_policy;
  using T = typename View::value_type;
  CartesianExtent3<typename Policy::count_type> interior{};
  LPFX_TRY((try_make_location_sample_extents<typename View::location_type,
                                             Policy>(
      pattern.local_geometry().cell_extents, interior)));
  const auto halo = owner.template component_halo<C>();
  AllocatedAxisInterval<Policy> x{}, y{}, z{};
  LPFX_TRY(try_make_allocated_axis_interval<Policy>(interior.x(), halo.x_minus,
                                                    halo.x_plus, x));
  LPFX_TRY(try_make_allocated_axis_interval<Policy>(interior.y(), halo.y_minus,
                                                    halo.y_plus, y));
  LPFX_TRY(try_make_allocated_axis_interval<Policy>(interior.z(), halo.z_minus,
                                                    halo.z_plus, z));
  LPFX_TRY(try_make_cartesian_box3(x.lower(), y.lower(), z.lower(), x.upper(),
                                   y.upper(), z.upper(), output.allocated));
  if constexpr (std::same_as<typename Owner::space_type, HostSpace>) {
    const auto layout = component.layout();
    const auto allocation = layout.allocation_extents();
    if (layout.interior_extents() != interior || layout.halo() != halo ||
        allocation.x() != x.extent() || allocation.y() != y.extent() ||
        allocation.z() != z.extent()) {
      return ErrorCode::invalid_argument;
    }
  }
  std::size_t stride_y{}, stride_z{}, allocation_x{}, allocation_y{},
      minimum_z{};
  if (!lpfx::detail::checked_integral_cast(component.stride_y(), stride_y) ||
      !lpfx::detail::checked_integral_cast(component.stride_z(), stride_z) ||
      !lpfx::detail::checked_integral_cast(x.extent(), allocation_x) ||
      !lpfx::detail::checked_integral_cast(y.extent(), allocation_y) ||
      !lpfx::detail::checked_multiply(stride_y, allocation_y, minimum_z) ||
      !lpfx::detail::checked_byte_count<T>(stride_y, output.stride_y_bytes) ||
      !lpfx::detail::checked_byte_count<T>(stride_z, output.stride_z_bytes)) {
    return ErrorCode::overflow;
  }
  output.has_core = component.core_data() != nullptr;
  return stride_y < allocation_x || stride_z < minimum_z
             ? ErrorCode::invalid_argument
             : ErrorCode::success;
}

template <class T, lpfx::detail::CartesianIndexPolicy Policy>
[[nodiscard]] constexpr ErrorCode
prepare_field_box_shape(const FieldComponentGeometry<Policy> &component,
                        const CartesianBox3<typename Policy::coordinate_type> box,
                        FieldExecutionBoxShape &shape) noexcept {
  using Count = typename Policy::count_type;
  if (!lpfx::detail::cartesian_box_contains(component.allocated, box)) {
    return ErrorCode::capacity_exceeded;
  }
  CartesianExtent3<Count> extents{};
  Count samples{};
  LPFX_TRY(lpfx::detail::try_cartesian_box_shape(box, extents, samples));
  std::size_t sample_count{};
  if (!lpfx::detail::checked_integral_cast(extents.x(), shape.x) ||
      !lpfx::detail::checked_integral_cast(extents.y(), shape.y) ||
      !lpfx::detail::checked_integral_cast(extents.z(), shape.z) ||
      !lpfx::detail::checked_integral_cast(samples, sample_count) ||
      !lpfx::detail::checked_byte_count<T>(sample_count, shape.bytes)) {
    return ErrorCode::overflow;
  }
  shape.stride_y_bytes = component.stride_y_bytes;
  shape.stride_z_bytes = component.stride_z_bytes;
  return shape.bytes != 0 && !component.has_core ? ErrorCode::invalid_argument
                                                 : ErrorCode::success;
}

// Component buffers must be disjoint: source safety is proved by coordinate
// boxes within each component, so an independent component cannot overwrite
// those sources through an aliased allocation. Check the conservative address
// span, including row/plane padding, once before any native initialization.
template <class Pattern, class Owner, class View>
[[nodiscard]] ErrorCode validate_field_component_storage(
    const Owner &owner, const View field,
    const FieldComponentGeometries<Pattern> &geometries) noexcept {
  constexpr std::size_t count = Pattern::component_count;
  if constexpr (count < 2) {
    return ErrorCode::success;
  } else {
    std::array<lpfx::detail::ByteInterval, count> intervals{};
    LPFX_TRY(meta::for_each_until_error<count>([&](const auto index) noexcept {
      constexpr std::size_t C = decltype(index)::value;
      const auto component = field.template component<C>();
      using T = typename std::remove_cvref_t<decltype(component)>::value_type;
      FieldExecutionBoxShape shape{};
      LPFX_TRY(prepare_field_box_shape<T>(geometries[C], geometries[C].allocated,
                                          shape));
      // Empty allocation spans neither require an address nor participate in
      // alias comparisons; guard the final-row/plane subtraction explicitly.
      if (shape.bytes == 0) {
        return ErrorCode::success;
      }
      const auto halo = owner.template component_halo<C>();
      std::size_t before_core{}, term{}, span{};
      if (!lpfx::detail::checked_byte_count<T>(
              static_cast<std::size_t>(halo.x_minus), before_core) ||
          !lpfx::detail::checked_multiply(
              static_cast<std::size_t>(halo.y_minus), shape.stride_y_bytes,
              term) ||
          !lpfx::detail::checked_add(before_core, term, before_core) ||
          !lpfx::detail::checked_multiply(
              static_cast<std::size_t>(halo.z_minus), shape.stride_z_bytes,
              term) ||
          !lpfx::detail::checked_add(before_core, term, before_core) ||
          !lpfx::detail::checked_byte_count<T>(shape.x, span) ||
          !lpfx::detail::checked_multiply(shape.y - 1, shape.stride_y_bytes,
                                          term) ||
          !lpfx::detail::checked_add(span, term, span) ||
          !lpfx::detail::checked_multiply(shape.z - 1, shape.stride_z_bytes,
                                          term) ||
          !lpfx::detail::checked_add(span, term, span)) {
        return ErrorCode::overflow;
      }
      const auto core = reinterpret_cast<std::uintptr_t>(component.core_data());
      std::uintptr_t offset{}, extent{};
      if (!lpfx::detail::checked_integral_cast(before_core, offset) ||
          !lpfx::detail::checked_integral_cast(span, extent) || core < offset ||
          core - offset > std::numeric_limits<std::uintptr_t>::max() - extent) {
        return ErrorCode::overflow;
      }
      intervals[C] = {core - offset, core - offset + extent};
      return ErrorCode::success;
    }));
    std::sort(intervals.begin(), intervals.end(),
              [](const auto left, const auto right) noexcept {
                return left.begin < right.begin;
              });
    for (std::size_t i = 1; i != intervals.size(); ++i)
      if (!lpfx::detail::disjoint(intervals[i - 1], intervals[i]))
        return ErrorCode::invalid_argument;
    return ErrorCode::success;
  }
}

// Owner checks and per-component allocation geometry from host metadata only,
// even when the field addresses denote device storage.
template <class Pattern, class Owner>
[[nodiscard]] ErrorCode
prepare_field_owner_geometry(const Pattern &pattern, Owner &owner,
                             FieldComponentGeometries<Pattern> &output) noexcept {
  if (!pattern.is_valid() || !owner.valid() || !owner.initialized() ||
      owner.cells() != pattern.local_geometry().cell_extents) {
    return ErrorCode::invalid_argument;
  }
  // A type requirement: nvcc rejects a requires-expression naming a captured
  // variable inside the generic lambda.
  constexpr bool has_halos =
      requires(const typename Pattern::geometry_type &geometry) {
        geometry.component_halos;
      };
  const auto field = owner.view();
  LPFX_TRY(meta::for_each_until_error<Pattern::component_count>(
      [&](const auto index) noexcept {
        constexpr std::size_t C = decltype(index)::value;
        if constexpr (has_halos) {
          if (owner.template component_halo<C>() !=
              pattern.local_geometry().component_halos[C]) {
            return ErrorCode::invalid_argument;
          }
        }
        return prepare_field_component_geometry<C>(
            pattern, owner, field.template component<C>(), output[C]);
      }));
  return validate_field_component_storage<Pattern>(owner, field, output);
}

// Field-frame shapes of one route: the field side of a remote or snapshot
// route, both sides of a direct route.
template <FieldSemanticPass Pass, std::size_t C, class Pattern>
[[nodiscard]] constexpr ErrorCode prepare_field_route_shapes(
    const Pattern &pattern, const FieldComponentGeometries<Pattern> &geometries,
    const std::size_t d, const typename Pattern::piece_type &piece,
    const PreparedFieldRoute route,
    lpfx::detail::FieldRouteBoxes<typename Pattern::index_policy::coordinate_type>
        &boxes,
    FieldExecutionBoxShape &source,
    FieldExecutionBoxShape &destination) noexcept {
  using T = typename Pattern::spec_type::template component_type<C>::storage_type;
  boxes = lpfx::detail::resolve_field_route_box<Pass, C>(pattern, d, piece, route);
  const std::size_t bytes = Pattern::piece_bytes(piece);
  const bool direct = route.route == FieldRuntimeRouteClass::self_direct;
  if (Pass == FieldSemanticPass::source || direct) {
    LPFX_TRY(prepare_field_box_shape<T>(geometries[C], boxes.source, source));
    if (source.bytes != bytes) {
      return ErrorCode::invalid_argument;
    }
  }
  if (Pass != FieldSemanticPass::source || direct) {
    LPFX_TRY(prepare_field_box_shape<T>(geometries[C], boxes.destination,
                                        destination));
    if (destination.bytes != bytes) {
      return ErrorCode::invalid_argument;
    }
  }
  return direct && (source.x != destination.x || source.y != destination.y ||
                    source.z != destination.z)
             ? ErrorCode::invalid_argument
             : ErrorCode::success;
}

// The complete binding check, run before any registration or native resource.
template <class Plan, class Owner>
[[nodiscard]] ErrorCode validate_field_execution_binding(const Plan &plan,
                                                         Owner &owner) noexcept {
  using Pattern = typename Plan::pattern_type;
  using Spec = typename Plan::spec_type;
  const Pattern &pattern = plan.pattern();
  FieldComponentGeometries<Pattern> geometries{};
  LPFX_TRY(prepare_field_owner_geometry(pattern, owner, geometries));
  lpfx::detail::PreparedFieldSchedule<Spec> schedule{};
  LPFX_TRY(lpfx::detail::try_prepare_field_schedule(pattern, schedule));
  return lpfx::detail::FieldSchedulePolicy<Spec::kind>::try_for_each_active_pass(
      [&]<FieldSemanticPass Pass>(
          std::integral_constant<FieldSemanticPass, Pass>) noexcept {
        return lpfx::detail::try_for_each_field_work<Pass>(
            pattern, schedule,
            [&](const auto component, const std::size_t d, const auto &piece,
                const PreparedFieldRoute route) noexcept {
              lpfx::detail::FieldRouteBoxes<
                  typename Pattern::index_policy::coordinate_type>
                  boxes{};
              FieldExecutionBoxShape source{}, destination{};
              return prepare_field_route_shapes<Pass, component.value>(
                  pattern, geometries, d, piece, route, boxes, source,
                  destination);
            });
      });
}

} // namespace lpfx::distributed::detail

#endif
