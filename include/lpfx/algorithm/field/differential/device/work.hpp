#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_DEVICE_WORK_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_DEVICE_WORK_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/memory/space.hpp>
#include <lpfx/algorithm/device_kernel.hpp>
#include <lpfx/algorithm/field/differential/operators.hpp>

namespace lpfx {

// Every target interior of Op, concatenated in target order. The default value
// is the canonical empty operation; nonempty values come only from
// try_prepare_device_stencil_work.
template <class Op, detail::CartesianIndexPolicy IndexPolicy>
class PreparedDeviceStencilWork final {
public:
  using operator_type = Op;
  using index_policy = IndexPolicy;
  using count_type = typename IndexPolicy::count_type;
  using coordinate_type = typename IndexPolicy::coordinate_type;
  static constexpr std::size_t target_count = Op::target_count;

  constexpr PreparedDeviceStencilWork() noexcept = default;

  template <std::size_t Target = 0>
    requires(Target < target_count)
  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianExtent3<count_type>
  target_extents() const noexcept {
    return extents_[Target];
  }

  template <std::size_t Target = 0>
    requires(Target < target_count)
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  target_sample_count() const noexcept {
    return sample_counts_[Target];
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  total_sample_count() const noexcept {
    return sample_counts_[count_slots - 1];
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool empty() const noexcept {
    return total_sample_count() == 0;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const PreparedDeviceStencilWork &,
             const PreparedDeviceStencilWork &) noexcept = default;

private:
  // A single target's count is the total; several targets store each count
  // followed by the total.
  static constexpr std::size_t count_slots =
      target_count == 1 ? 1 : target_count + 1;

  constexpr PreparedDeviceStencilWork(
      const CartesianExtent3<count_type> (&extents)[target_count],
      const std::uint64_t (&sample_counts)[target_count],
      const std::uint64_t total) noexcept {
    static_assert(detail::packed_trivially_copyable<
                  PreparedDeviceStencilWork,
                  CartesianExtent3<count_type>[target_count],
                  std::uint64_t[count_slots]>);
    for (std::size_t target = 0; target != target_count; ++target) {
      extents_[target] = extents[target];
      if constexpr (target_count != 1) {
        sample_counts_[target] = sample_counts[target];
      }
    }
    sample_counts_[count_slots - 1] = total;
  }

  CartesianExtent3<count_type> extents_[target_count]{};
  std::uint64_t sample_counts_[count_slots]{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianIndexPolicy IndexPolicy>
using PreparedDeviceCurlEdgeToFaceWork =
    PreparedDeviceStencilWork<detail::CurlEdgeToFace, IndexPolicy>;

template <detail::CartesianIndexPolicy IndexPolicy>
using PreparedDeviceCurlFaceToEdgeWork =
    PreparedDeviceStencilWork<detail::CurlFaceToEdge, IndexPolicy>;

template <detail::CartesianIndexPolicy IndexPolicy>
using PreparedDeviceDivergenceEdgeToNodeWork =
    PreparedDeviceStencilWork<detail::DivergenceEdgeToNode, IndexPolicy>;

namespace detail {

// Per-axis maximum of the target interiors of a three-target stencil. Every
// target interior starts at the origin, so this box is their union's bounding
// box; its volume fits std::uint64_t for every prepared work.
struct StencilUnionBox final {
  std::uint64_t x;
  std::uint64_t y;
  std::uint64_t z;
};

[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint64_t
stencil_axis_maximum(const std::uint64_t a, const std::uint64_t b,
                     const std::uint64_t c) noexcept {
  const std::uint64_t ab = a > b ? a : b;
  return ab > c ? ab : c;
}

template <class Count>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr StencilUnionBox
stencil_union_box(const CartesianExtent3<Count> e0,
                  const CartesianExtent3<Count> e1,
                  const CartesianExtent3<Count> e2) noexcept {
  return {stencil_axis_maximum(e0.x(), e1.x(), e2.x()),
          stencil_axis_maximum(e0.y(), e1.y(), e2.y()),
          stencil_axis_maximum(e0.z(), e1.z(), e2.z())};
}

template <class Op, CartesianIndexPolicy IndexPolicy>
  requires(Op::target_count == 3)
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr StencilUnionBox
stencil_union_box(
    const PreparedDeviceStencilWork<Op, IndexPolicy> work) noexcept {
  return stencil_union_box(work.template target_extents<0>(),
                           work.template target_extents<1>(),
                           work.template target_extents<2>());
}

// Points of the fused traversal: the union box volume.
template <class Op, CartesianIndexPolicy IndexPolicy>
  requires(Op::target_count == 3)
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint64_t
device_stencil_union_count(
    const PreparedDeviceStencilWork<Op, IndexPolicy> work) noexcept {
  const StencilUnionBox box = stencil_union_box(work);
  return box.x * box.y * box.z;
}

// Crossover of the two traversals of a three-target stencil (ncu, base
// clocks, double curls, sm_80 code on an 84-SM RTX 5080). The fused kernel
// launches a third as many threads, each evaluating every target formula at
// the point clamped into that target's interior. It loses while its grid does
// not fill the device several times over (it breaks even near 1.4e5 union
// points, about one wave) and on thin boxes, where the clamped evaluations are
// mostly redundant (an axis of 1, 2, 8 or 16 cells loses up to 45% even at
// millions of points). Every measured box with both bounds met gains at least
// 3.6%, up to 1.42x; a box below either bound keeps the concatenated kernel.
inline constexpr std::uint64_t device_stencil_union_minimum_count =
    std::uint64_t{1} << 19;
inline constexpr std::uint64_t device_stencil_union_minimum_axis = 33;

// Host-side traversal choice of a nonempty work: the fused union-box kernel
// for a three-target Op whose union box meets both crossover bounds, the
// concatenated per-target kernel otherwise.
template <class Op, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr bool device_stencil_selects_union_traversal(
    const PreparedDeviceStencilWork<Op, IndexPolicy> work) noexcept {
  if constexpr (Op::target_count == 3) {
    const StencilUnionBox box = stencil_union_box(work);
    return box.x >= device_stencil_union_minimum_axis &&
           box.y >= device_stencil_union_minimum_axis &&
           box.z >= device_stencil_union_minimum_axis &&
           box.x * box.y * box.z >= device_stencil_union_minimum_count;
  } else {
    return false;
  }
}

} // namespace detail

// Transactional. A domain with any zero cell extent is the empty operation;
// otherwise every target extent must fit the signed logical coordinates used
// by device view indexing, and the concatenated sample count and, for three
// targets, the union box volume must fit std::uint64_t (overflow).
template <class Op, detail::CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_device_stencil_work(
    const CartesianExtent3<typename IndexPolicy::count_type> cell_extents,
    PreparedDeviceStencilWork<Op, IndexPolicy> &output) noexcept {
  using Work = PreparedDeviceStencilWork<Op, IndexPolicy>;
  if (!detail::cartesian_extent_is_nonempty(cell_extents)) {
    output = Work{};
    return ErrorCode::success;
  }

  CartesianExtent3<typename IndexPolicy::count_type>
      extents[Work::target_count]{};
  std::uint64_t sample_counts[Work::target_count]{};
  std::uint64_t total_sample_count = 0;
  const ErrorCode error =
      meta::for_each_until_error<Work::target_count>([&](auto target) {
        constexpr std::size_t index = decltype(target)::value;
        LPFX_TRY((detail::try_prepare_device_location_samples<
                  typename Op::template target_location<index>, IndexPolicy>(
            cell_extents, extents[index], sample_counts[index])));
        return detail::checked_add(total_sample_count, sample_counts[index],
                                   total_sample_count)
                   ? ErrorCode::success
                   : ErrorCode::overflow;
      });
  LPFX_TRY(error);
  if constexpr (Work::target_count == 3) {
    const detail::StencilUnionBox box =
        detail::stencil_union_box(extents[0], extents[1], extents[2]);
    std::uint64_t box_xy = 0;
    std::uint64_t box_volume = 0;
    if (!detail::checked_multiply(box.x, box.y, box_xy) ||
        !detail::checked_multiply(box_xy, box.z, box_volume)) {
      return ErrorCode::overflow;
    }
  }
  output = detail::PrivateConstructionAccess::create<Work>(
      extents, sample_counts, total_sample_count);
  return ErrorCode::success;
}

namespace detail {

// Kernel argument: the source components followed by the target components.
template <class Sources, class Targets> struct DeviceStencilBinding final {
  Sources sources{};
  Targets targets{};
};

template <class Coordinate> struct StencilSampleIndex final {
  Coordinate i;
  Coordinate j;
  Coordinate k;
};

template <class Coordinate, class Count>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr StencilSampleIndex<
    Coordinate>
decompose_stencil_sample(const CartesianExtent3<Count> extents,
                         const std::uint64_t local_sample) noexcept {
  const std::uint64_t extent_x = static_cast<std::uint64_t>(extents.x());
  const std::uint64_t extent_y = static_cast<std::uint64_t>(extents.y());
  const std::uint64_t xy = extent_x * extent_y;
  const std::uint64_t z = local_sample / xy;
  const std::uint64_t xy_sample = local_sample - z * xy;
  const std::uint64_t y = xy_sample / extent_x;
  const std::uint64_t x = xy_sample - y * extent_x;
  return {static_cast<Coordinate>(x), static_cast<Coordinate>(y),
          static_cast<Coordinate>(z)};
}

template <class Op, std::floating_point Accumulator, std::size_t Target,
          class Sources, class Targets, class Coordinate,
          std::floating_point SpacingScalar>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void store_device_stencil_sample(
    const DeviceStencilBinding<Sources, Targets> &binding,
    const StencilSampleIndex<Coordinate> index,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  const auto &target = stencil_component<Target>(binding.targets);
  using Value = typename std::remove_cvref_t<decltype(target)>::value_type;
  target(index.i, index.j, index.k) =
      static_cast<Value>(Op::template apply<Accumulator, Target>(
          binding.sources.x, binding.sources.y, binding.sources.z, index.i,
          index.j, index.k, spacing));
}

// Writes the target sample at linear_sample < work.total_sample_count().
template <std::floating_point Accumulator, class Sources, class Targets,
          class Op, CartesianIndexPolicy IndexPolicy,
          std::floating_point SpacingScalar>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void apply_device_stencil_sample(
    const DeviceStencilBinding<Sources, Targets> binding,
    const PreparedDeviceStencilWork<Op, IndexPolicy> work,
    const DifferentialSpacing3D<SpacingScalar> spacing,
    const std::uint64_t linear_sample) noexcept {
  using Coordinate = typename IndexPolicy::coordinate_type;
  if constexpr (Op::target_count == 1) {
    store_device_stencil_sample<Op, Accumulator, 0>(
        binding,
        decompose_stencil_sample<Coordinate>(work.target_extents(),
                                             linear_sample),
        spacing);
  } else {
    static_assert(Op::target_count == 3);
    const std::uint64_t count_x = work.template target_sample_count<0>();
    const std::uint64_t count_y = work.template target_sample_count<1>();
    CartesianExtent3<typename IndexPolicy::count_type> extents{};
    std::uint64_t local_sample{};
    if (linear_sample < count_x) {
      extents = work.template target_extents<0>();
      local_sample = linear_sample;
    } else if (linear_sample < count_x + count_y) {
      extents = work.template target_extents<1>();
      local_sample = linear_sample - count_x;
    } else {
      extents = work.template target_extents<2>();
      local_sample = linear_sample - count_x - count_y;
    }
    const StencilSampleIndex<Coordinate> index =
        decompose_stencil_sample<Coordinate>(extents, local_sample);
    if (linear_sample < count_x) {
      store_device_stencil_sample<Op, Accumulator, 0>(binding, index,
                                                      spacing);
    } else if (linear_sample < count_x + count_y) {
      store_device_stencil_sample<Op, Accumulator, 1>(binding, index,
                                                      spacing);
    } else {
      store_device_stencil_sample<Op, Accumulator, 2>(binding, index,
                                                      spacing);
    }
  }
}

// Hard precondition: extent > 0. v clamped into [0, extent).
template <class Coordinate, class Count>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Coordinate
clamp_stencil_coordinate(const std::uint64_t v, const Count extent) noexcept {
  const std::uint64_t last = static_cast<std::uint64_t>(extent) - 1U;
  return static_cast<Coordinate>(v < last ? v : last);
}

// Target's formula at (i, j, k) clamped into its nonempty interior: every
// point of the union box reads only samples some target point reads, so the
// loads need no predicate.
template <class Op, std::floating_point Accumulator, std::size_t Target,
          class Coordinate, class Sources, class Targets, class Count,
          std::floating_point SpacingScalar>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Accumulator
clamped_stencil_value(
    const DeviceStencilBinding<Sources, Targets> &binding,
    const CartesianExtent3<Count> extents, const std::uint64_t i,
    const std::uint64_t j, const std::uint64_t k,
    const DifferentialSpacing3D<SpacingScalar> spacing) noexcept {
  return Op::template apply<Accumulator, Target>(
      binding.sources.x, binding.sources.y, binding.sources.z,
      clamp_stencil_coordinate<Coordinate>(i, extents.x()),
      clamp_stencil_coordinate<Coordinate>(j, extents.y()),
      clamp_stencil_coordinate<Coordinate>(k, extents.z()), spacing);
}

template <std::size_t Target, class Sources, class Targets, class Coordinate,
          std::floating_point Accumulator>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void
store_stencil_value(const DeviceStencilBinding<Sources, Targets> &binding,
                    const StencilSampleIndex<Coordinate> index,
                    const Accumulator value) noexcept {
  const auto &target = stencil_component<Target>(binding.targets);
  using Value = typename std::remove_cvref_t<decltype(target)>::value_type;
  target(index.i, index.j, index.k) = static_cast<Value>(value);
}

// Hard precondition: (i, j, k) lies inside stencil_union_box(work). Writes
// every target whose interior contains (i, j, k) with the same formula, cast
// and store as apply_device_stencil_sample, so a traversal of the union box
// reads each source sample once for all the targets that use it. The three
// formulas run unpredicated and before any store, so their loads are issued
// together.
template <std::floating_point Accumulator, class Sources, class Targets,
          class Op, CartesianIndexPolicy IndexPolicy,
          std::floating_point SpacingScalar>
  requires(Op::target_count == 3)
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void apply_device_stencil_union_point(
    const DeviceStencilBinding<Sources, Targets> &binding,
    const PreparedDeviceStencilWork<Op, IndexPolicy> work,
    const DifferentialSpacing3D<SpacingScalar> spacing, const std::uint64_t i,
    const std::uint64_t j, const std::uint64_t k) noexcept {
  using Coordinate = typename IndexPolicy::coordinate_type;
  const auto e0 = work.template target_extents<0>();
  const auto e1 = work.template target_extents<1>();
  const auto e2 = work.template target_extents<2>();
  const Accumulator v0 = clamped_stencil_value<Op, Accumulator, 0, Coordinate>(
      binding, e0, i, j, k, spacing);
  const Accumulator v1 = clamped_stencil_value<Op, Accumulator, 1, Coordinate>(
      binding, e1, i, j, k, spacing);
  const Accumulator v2 = clamped_stencil_value<Op, Accumulator, 2, Coordinate>(
      binding, e2, i, j, k, spacing);
  const StencilSampleIndex<Coordinate> index{static_cast<Coordinate>(i),
                                             static_cast<Coordinate>(j),
                                             static_cast<Coordinate>(k)};
  if (i < e0.x() && j < e0.y() && k < e0.z()) {
    store_stencil_value<0>(binding, index, v0);
  }
  if (i < e1.x() && j < e1.y() && k < e1.z()) {
    store_stencil_value<1>(binding, index, v1);
  }
  if (i < e2.x() && j < e2.y() && k < e2.z()) {
    store_stencil_value<2>(binding, index, v2);
  }
}

// The one launch seam of the stencil family. A specialization provides
//   template <std::floating_point Accumulator, ...>
//   static error_type enqueue(DeviceStencilBinding<Sources, Targets>,
//                             PreparedDeviceStencilWork<Op, IndexPolicy>,
//                             DifferentialSpacing3D<S>, stream_type,
//                             bool &accepted) noexcept;
// that writes every target sample of a nonempty work exactly once (through
// apply_device_stencil_sample over the concatenated order, or, when
// device_stencil_selects_union_traversal(work), through
// apply_device_stencil_union_point over stencil_union_box) and follows the
// device::checked_kernel_launch acceptance protocol.
template <class Backend> struct DeviceStencilLaunch;

} // namespace detail

} // namespace lpfx

#endif
