#ifndef LPFX_DISTRIBUTED_FIELD_HOST_HPP
#define LPFX_DISTRIBUTED_FIELD_HOST_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/field/lowering.hpp>

namespace lpfx::distributed::detail {

// Scalar widths with an integer carrier: their loops are written out in
// execute_host_field_box so the compiler sees plain loop nests.
template <class T>
inline constexpr bool host_field_native_lane =
    sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8;

template <std::size_t Bytes> struct HostFieldRepresentation;
template <> struct HostFieldRepresentation<1> {
  using type = std::uint8_t;
};
template <> struct HostFieldRepresentation<2> {
  using type = std::uint16_t;
};
template <> struct HostFieldRepresentation<4> {
  using type = std::uint32_t;
};
template <> struct HostFieldRepresentation<8> {
  using type = std::uint64_t;
};
template <> struct HostFieldRepresentation<16> {
  using type = std::array<std::uint64_t, 2>;
};

template <class T>
LPFX_ALWAYS_INLINE void
copy_host_field_representation(const std::byte *const source,
                               std::byte *const destination) noexcept {
  if constexpr (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                sizeof(T) == 8 || sizeof(T) == 16) {
    // Materialize integer carriers so LLVM keeps scalar-width lane copies as
    // load/store operations rather than opaque memcpy calls in strided loops.
    typename HostFieldRepresentation<sizeof(T)>::type bits;
    std::memcpy(&bits, source, sizeof(T));
    std::memcpy(destination, &bits, sizeof(T));
  } else {
    std::memcpy(destination, source, sizeof(T));
  }
}

template <bool Strided> struct HostFieldBoxWork;

// A compact image and a strided field need only one pair of field strides.
template <> struct HostFieldBoxWork<false> final {
  const std::byte *source;
  std::byte *destination;
  std::size_t field_stride_y, field_stride_z;
  std::size_t x, y, z;
};

// Direct local routes can have two independently strided operands. Cold
// binding also resolves image-backed routes to this same branch-free form.
template <> struct HostFieldBoxWork<true> final {
  const std::byte *source;
  std::byte *destination;
  std::size_t source_stride_y, source_stride_z;
  std::size_t destination_stride_y, destination_stride_z;
  std::size_t x, y, z;
};

template <class T, FieldWorkAction Action>
LPFX_ALWAYS_INLINE void execute_host_field_lane(
    const std::byte *const source, std::byte *const destination,
    const std::size_t source_stride, const std::size_t destination_stride,
    const std::size_t lane) noexcept {
  if constexpr (Action == FieldWorkAction::add) {
    T value;
    std::memcpy(&value, source + lane * source_stride, sizeof(T));
    *reinterpret_cast<T *>(destination + lane * destination_stride) += value;
  } else {
    copy_host_field_representation<T>(source + lane * source_stride,
                                      destination + lane * destination_stride);
  }
}

template <class T, FieldWorkAction Action>
LPFX_ALWAYS_INLINE void execute_host_field_line(
    const std::byte *const source, std::byte *const destination,
    const std::size_t source_stride, const std::size_t destination_stride,
    const std::size_t count) noexcept {
  if constexpr (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                sizeof(T) == 8) {
    LPFX_INDEPENDENT_LOOP
    for (std::size_t lane = 0; lane != count; ++lane) {
      execute_host_field_lane<T, Action>(source, destination, source_stride,
                                         destination_stride, lane);
    }
  } else {
    // Wider/custom representations retain their exact bytes without a SIMD
    // promise unsupported by the host compiler (including x87 long double).
    for (std::size_t lane = 0; lane != count; ++lane) {
      execute_host_field_lane<T, Action>(source, destination, source_stride,
                                         destination_stride, lane);
    }
  }
}

template <class T, FieldWorkAction Action, bool Strided>
LPFX_ALWAYS_INLINE void
execute_host_field_box(const HostFieldBoxWork<Strided> work) noexcept {
  const std::byte *const source = work.source;
  std::byte *const destination = work.destination;
  std::size_t sy{}, sz{}, dy{}, dz{};
  if constexpr (Strided) {
    sy = work.source_stride_y;
    sz = work.source_stride_z;
    dy = work.destination_stride_y;
    dz = work.destination_stride_z;
  } else if constexpr (Action == FieldWorkAction::pack) {
    sy = work.field_stride_y;
    sz = work.field_stride_z;
    dy = work.x * sizeof(T);
    dz = work.y * dy;
  } else {
    sy = work.x * sizeof(T);
    sz = work.y * sy;
    dy = work.field_stride_y;
    dz = work.field_stride_z;
  }
  const auto x_count = work.x, y_count = work.y, z_count = work.z;
  if (x_count == 1) {
    if (y_count == 1) {
      if constexpr (host_field_native_lane<T>) {
        LPFX_INDEPENDENT_LOOP
        for (std::size_t z = 0; z != z_count; ++z) {
          if constexpr (Action == FieldWorkAction::add) {
            T value;
            std::memcpy(&value, source + z * sz, sizeof(T));
            *reinterpret_cast<T *>(destination + z * dz) += value;
          } else {
            typename HostFieldRepresentation<sizeof(T)>::type bits;
            std::memcpy(&bits, source + z * sz, sizeof(T));
            std::memcpy(destination + z * dz, &bits, sizeof(T));
          }
        }
      } else {
        execute_host_field_line<T, Action>(source, destination, sz, dz,
                                           z_count);
      }
    } else {
      for (std::size_t z = 0; z != z_count; ++z) {
        if constexpr (host_field_native_lane<T>) {
          LPFX_INDEPENDENT_LOOP
          for (std::size_t y = 0; y != y_count; ++y) {
            if constexpr (Action == FieldWorkAction::add) {
              T value;
              std::memcpy(&value, source + z * sz + y * sy, sizeof(T));
              *reinterpret_cast<T *>(destination + z * dz + y * dy) += value;
            } else {
              typename HostFieldRepresentation<sizeof(T)>::type bits;
              std::memcpy(&bits, source + z * sz + y * sy, sizeof(T));
              std::memcpy(destination + z * dz + y * dy, &bits, sizeof(T));
            }
          }
        } else {
          execute_host_field_line<T, Action>(
              source + z * sz, destination + z * dz, sy, dy, y_count);
        }
      }
    }
    return;
  }
  for (std::size_t z = 0; z != z_count; ++z) {
    for (std::size_t y = 0; y != y_count; ++y) {
      const auto *source_row = source + z * sz + y * sy;
      auto *destination_row = destination + z * dz + y * dy;
      if constexpr (Action == FieldWorkAction::add) {
        if constexpr (host_field_native_lane<T>) {
          LPFX_INDEPENDENT_LOOP
          for (std::size_t x = 0; x != x_count; ++x) {
            T value;
            std::memcpy(&value, source_row + x * sizeof(T), sizeof(T));
            reinterpret_cast<T *>(destination_row)[x] += value;
          }
        } else {
          execute_host_field_line<T, Action>(source_row, destination_row,
                                             sizeof(T), sizeof(T), x_count);
        }
      } else {
        std::memcpy(destination_row, source_row, x_count * sizeof(T));
      }
    }
  }
}

template <class Pattern, class Owner>
  requires lpfx::detail::FieldOwnerMatchesOperationSpec<
               Owner, typename Pattern::spec_type, typename Pattern::index_policy> &&
           std::same_as<typename Owner::space_type, HostSpace>
class HostFieldExecution final {
public:
  using pattern_type = Pattern;
  using owner_type = Owner;
  using spec_type = typename Pattern::spec_type;
  using schedule_type = lpfx::detail::FieldSchedulePolicy<spec_type::kind>;
  using layout_type = FieldExecutionLayout<spec_type>;
  static constexpr bool strided = field_execution_has_direct_routes<spec_type>;
  using work_type = HostFieldBoxWork<strided>;
  static constexpr bool asynchronous = false;

  HostFieldExecution() noexcept = default;
  LPFX_NONMOVABLE(HostFieldExecution);

  // Validates the plan against the owner and binds every work record.
  template <class Plan, class Executor>
    requires std::same_as<typename Plan::pattern_type, pattern_type>
  [[nodiscard]] ErrorCode prepare(HostMemoryBudget &budget, const Plan &plan,
                                  Owner &owner, const Executor &,
                                  FieldExecutionImages &images) noexcept {
    if (!allocation_.empty()) {
      return ErrorCode::invalid_state;
    }
    const pattern_type &pattern = plan.pattern();
    layout_type layout{};
    LPFX_TRY(prepare_field_execution_layout<work_type>(pattern, layout));
    FieldComponentGeometries<Pattern> geometries{};
    LPFX_TRY(prepare_field_owner_geometry(pattern, owner, geometries));
    auto allocated =
        allocate_field_host_storage<work_type>(budget, layout.total_bytes);
    LPFX_TRY(allocated);
    auto allocation = std::move(allocated).value();
    std::array<std::span<work_type>, layout_type::bucket_count> work{};
    for (std::size_t i = 0; i != work.size(); ++i) {
      work[i] = {start_field_work_array<work_type>(allocation.raw_base(),
                                                   layout.offsets[i],
                                                   layout.counts[i]),
                 layout.counts[i]};
    }
    std::array<std::size_t, layout_type::bucket_count> cursors{};
    const auto field = owner.view();
    LPFX_TRY(schedule_type::try_for_each_active_pass(
        [&]<FieldSemanticPass Pass>(
            std::integral_constant<FieldSemanticPass, Pass>) noexcept {
          return lpfx::detail::try_for_each_field_work<Pass>(
              pattern, layout.schedule,
              [&](const auto component, const std::size_t d, const auto &piece,
                  const PreparedFieldRoute route) noexcept {
                constexpr auto bucket =
                    layout_type::template bucket_index<Pass, component.value>;
                return bind_box<Pass, component.value>(
                    pattern, geometries,
                    field.template component<component.value>(), d, piece,
                    route, allocation.raw_base(), layout,
                    work[bucket][cursors[bucket]++]);
              });
        }));
    images = {field_execution_image(allocation.raw_base(), layout, 0),
              field_execution_image(allocation.raw_base(), layout, 1)};
    work_ = work;
    allocation_ = std::move(allocation);
    return ErrorCode::success;
  }

  template <FieldSemanticPass Pass, std::size_t Component = 0>
  [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode execute() noexcept {
    static_assert(schedule_type::template pass_is_active<Pass>);
    if constexpr (Component != spec_type::component_count) {
      using T =
          typename spec_type::template component_type<Component>::storage_type;
      using Traits = lpfx::detail::FieldPassTraits<spec_type::kind, Pass>;
      // Source capture and a proven-safe direct overwrite are both exact
      // representation copies. Their cold pointer binding is the only change.
      constexpr auto action =
          Traits::template action<FieldRuntimeRouteClass::remote>;
      for (const auto &work :
           work_[layout_type::template bucket_index<Pass, Component>]) {
        execute_host_field_box<T, action>(work);
      }
      return execute<Pass, Component + 1>();
    }
    return ErrorCode::success;
  }

  [[nodiscard]] ErrorCode reset() noexcept {
    work_ = {};
    allocation_.reset_capacity();
    return ErrorCode::success;
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }

private:
  template <FieldSemanticPass Pass, std::size_t Component, class View,
            class Piece>
  [[nodiscard]] static ErrorCode
  bind_box(const Pattern &pattern,
           const FieldComponentGeometries<Pattern> &geometries,
           const View component, const std::size_t d, const Piece &piece,
           const PreparedFieldRoute route, std::byte *const base,
           const layout_type &layout, work_type &output) noexcept {
    using T = typename View::value_type;
    lpfx::detail::FieldRouteBoxes<typename View::coordinate_type> boxes{};
    FieldExecutionBoxShape source{}, destination{};
    LPFX_TRY((prepare_field_route_shapes<Pass, Component>(
        pattern, geometries, d, piece, route, boxes, source, destination)));
    const std::byte *source_pointer{};
    std::byte *destination_pointer{};
    const bool direct = route.route == FieldRuntimeRouteClass::self_direct;
    if (Pass == FieldSemanticPass::source || direct) {
      const auto lower = boxes.source.lower();
      source_pointer = reinterpret_cast<const std::byte *>(
          &component(lower.x(), lower.y(), lower.z()));
    }
    if (Pass != FieldSemanticPass::source || direct) {
      const auto lower = boxes.destination.lower();
      destination_pointer = reinterpret_cast<std::byte *>(
          &component(lower.x(), lower.y(), lower.z()));
    }
    if (!direct) {
      const auto region = field_execution_image(
          base, layout, lpfx::detail::field_image_index<Pass>(route.route));
      const auto &shape =
          Pass == FieldSemanticPass::source ? source : destination;
      if (route.image_offset_bytes > region.size() ||
          shape.bytes > region.size() - route.image_offset_bytes) {
        return ErrorCode::invalid_argument;
      }
      auto compact = shape;
      compact.stride_y_bytes = shape.x * sizeof(T);
      compact.stride_z_bytes = compact.stride_y_bytes * shape.y;
      if constexpr (Pass == FieldSemanticPass::source) {
        destination = compact;
        destination_pointer = region.data() + route.image_offset_bytes;
      } else {
        source = compact;
        source_pointer = region.data() + route.image_offset_bytes;
      }
    }
    if constexpr (strided) {
      output = {source_pointer,
                destination_pointer,
                source.stride_y_bytes,
                source.stride_z_bytes,
                destination.stride_y_bytes,
                destination.stride_z_bytes,
                source.x,
                source.y,
                source.z};
    } else {
      const auto &field_shape =
          Pass == FieldSemanticPass::source ? source : destination;
      output = {source_pointer,
                destination_pointer,
                field_shape.stride_y_bytes,
                field_shape.stride_z_bytes,
                source.x,
                source.y,
                source.z};
    }
    return ErrorCode::success;
  }

  FieldHostAllocation<work_type> allocation_{};
  std::array<std::span<work_type>, layout_type::bucket_count> work_{};
};

} // namespace lpfx::distributed::detail

#endif
