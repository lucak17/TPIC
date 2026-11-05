#ifndef LPFX_ALGORITHM_REDUCE_DEVICE_WORKSPACE_HPP
#define LPFX_ALGORITHM_REDUCE_DEVICE_WORKSPACE_HPP

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/fixed_array.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/domain/cartesian/halo.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/algorithm/reduce/operation.hpp>

namespace lpfx {

namespace detail {

[[nodiscard]] consteval bool
valid_device_block_threads(const std::uint32_t threads) noexcept {
  return threads >= 32 && threads <= 1024 && std::has_single_bit(threads);
}

// Shared by the scan and reduction launch policies.
template <class Policy>
concept DeviceBlockLaunchPolicy = std::is_empty_v<Policy> && requires {
  { Policy::block_threads } -> std::convertible_to<std::uint32_t>;
  requires valid_device_block_threads(Policy::block_threads);
};

template <class Policy>
concept DeviceReductionLaunchPolicyType =
    DeviceBlockLaunchPolicy<Policy> && requires {
      { Policy::items_per_thread } -> std::convertible_to<std::uint32_t>;
      { Policy::maximum_blocks } -> std::convertible_to<std::uint32_t>;
      requires Policy::items_per_thread != 0;
      requires Policy::maximum_blocks != 0;
    };

inline constexpr std::size_t maximum_device_level_count = 8;

// Element counts of a hierarchical device pass: first, ceil(first / Width),
// ... while greater than one; Terminal also keeps the final single-element
// level. Width >= 32 bounds the depth of any 32-bit first count.
template <std::uint64_t Width, bool Terminal, class Counts>
  requires(Width >= 32)
LPFX_ALWAYS_INLINE constexpr std::uint8_t
make_device_level_counts(std::uint32_t current, Counts &counts) noexcept {
  std::uint8_t level_count = 0;
  while (current > 1) {
    counts[level_count++] = current;
    current = static_cast<std::uint32_t>(
        1U + (static_cast<std::uint64_t>(current) - 1U) / Width);
  }
  if constexpr (Terminal) {
    counts[level_count++] = current;
  }
  return level_count;
}

static_assert([] {
  std::array<std::uint32_t, maximum_device_level_count> counts{};
  return make_device_level_counts<32, true>(0xffffffffU, counts);
}() == maximum_device_level_count);

// Level i holds counts[i] elements packed after level i - 1.
struct DeviceLevelLayout final {
  std::array<std::uint32_t, maximum_device_level_count> counts{};
  std::array<std::size_t, maximum_device_level_count> byte_offsets{};
  std::size_t total_bytes{};
  std::uint8_t level_count{};

  [[nodiscard]] friend constexpr bool
  operator==(const DeviceLevelLayout &,
             const DeviceLevelLayout &) noexcept = default;
};

// Returns false when the byte size overflows.
template <class T, std::uint64_t Width, bool Terminal>
[[nodiscard]] constexpr bool
try_make_device_level_layout(const std::uint32_t first,
                             DeviceLevelLayout &output) noexcept {
  DeviceLevelLayout layout{};
  layout.level_count =
      make_device_level_counts<Width, Terminal>(first, layout.counts);
  for (std::size_t level = 0; level != layout.level_count; ++level) {
    layout.byte_offsets[level] = layout.total_bytes;
    std::size_t level_bytes{};
    if (!checked_byte_count<T>(layout.counts[level], level_bytes) ||
        !checked_add(layout.total_bytes, level_bytes, layout.total_bytes)) {
      return false;
    }
  }
  output = layout;
  return true;
}

} // namespace detail

template <std::uint32_t BlockThreads, std::uint32_t ItemsPerThread,
          std::uint32_t MaximumBlocks>
  requires(detail::valid_device_block_threads(BlockThreads) &&
           ItemsPerThread != 0 && MaximumBlocks != 0)
struct DeviceReductionLaunchPolicy final {
  static constexpr std::uint32_t block_threads = BlockThreads;
  static constexpr std::uint32_t items_per_thread = ItemsPerThread;
  static constexpr std::uint32_t maximum_blocks = MaximumBlocks;
};

using DefaultDeviceReductionLaunchPolicy =
    DeviceReductionLaunchPolicy<256, 8, 4096>;

namespace detail {

template <DeviceReductionLaunchPolicyType LaunchPolicy>
inline constexpr std::uint64_t device_reduction_segment_width =
    static_cast<std::uint64_t>(LaunchPolicy::block_threads) *
    static_cast<std::uint64_t>(LaunchPolicy::items_per_thread);

} // namespace detail

// Block count of every level of one contiguous reduction epoch.
class PreparedDeviceReductionWork final {
public:
  constexpr PreparedDeviceReductionWork() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  sample_count() const noexcept {
    return sample_count_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  level_count() const noexcept {
    return level_count_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool empty() const noexcept {
    return sample_count_ == 0;
  }

  // Hard precondition: level < level_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint32_t
  level_block_count(const std::size_t level) const noexcept {
    return level_block_counts_[level];
  }

  [[nodiscard]] friend constexpr bool
  operator==(const PreparedDeviceReductionWork &left,
             const PreparedDeviceReductionWork &right) noexcept {
    if (left.sample_count_ != right.sample_count_ ||
        left.level_count_ != right.level_count_) {
      return false;
    }
    for (std::size_t level = 0; level != detail::maximum_device_level_count;
         ++level) {
      if (left.level_block_counts_[level] != right.level_block_counts_[level]) {
        return false;
      }
    }
    return true;
  }

private:
  constexpr PreparedDeviceReductionWork(
      const std::uint64_t sample_count,
      const std::array<std::uint32_t, detail::maximum_device_level_count>
          level_block_counts,
      const std::uint8_t level_count) noexcept
      : sample_count_(sample_count), level_block_counts_(level_block_counts),
        level_count_(level_count) {}

  std::uint64_t sample_count_{};
  detail::DeviceFixedArray<std::uint32_t, detail::maximum_device_level_count>
      level_block_counts_{};
  std::uint8_t level_count_{};

  friend struct detail::PrivateConstructionAccess;
};

// Row-segmented reduction of one location's sample interior. Kernels receive
// this object by value.
template <detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy =
              DefaultDeviceReductionLaunchPolicy>
class PreparedDeviceFieldReductionWork final {
public:
  using location_type = Location;
  using index_policy = IndexPolicy;
  using launch_policy = LaunchPolicy;
  using count_type = typename IndexPolicy::count_type;

  constexpr PreparedDeviceFieldReductionWork() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianExtent3<count_type>
  cell_extents() const noexcept {
    return cell_extents_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianExtent3<count_type>
  sample_extents() const noexcept {
    return sample_extents_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  sample_count() const noexcept {
    return reduction_.sample_count();
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  x_segment_count() const noexcept {
    return x_segment_count_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t
  segment_count() const noexcept {
    return segment_count_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t
  level_count() const noexcept {
    return reduction_.level_count();
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool empty() const noexcept {
    return reduction_.empty();
  }

  // Hard precondition: level < level_count().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint32_t
  level_block_count(const std::size_t level) const noexcept {
    return reduction_.level_block_count(level);
  }

  [[nodiscard]] friend constexpr bool
  operator==(const PreparedDeviceFieldReductionWork &,
             const PreparedDeviceFieldReductionWork &) noexcept = default;

private:
  constexpr PreparedDeviceFieldReductionWork(
      const CartesianExtent3<count_type> cell_extents,
      const CartesianExtent3<count_type> sample_extents,
      const std::uint64_t x_segment_count, const std::uint64_t segment_count,
      const PreparedDeviceReductionWork reduction) noexcept
      : cell_extents_(cell_extents), sample_extents_(sample_extents),
        x_segment_count_(x_segment_count), segment_count_(segment_count),
        reduction_(reduction) {}

  CartesianExtent3<count_type> cell_extents_{};
  CartesianExtent3<count_type> sample_extents_{};
  std::uint64_t x_segment_count_{};
  std::uint64_t segment_count_{};
  PreparedDeviceReductionWork reduction_{};

  friend struct detail::PrivateConstructionAccess;
};

// Capacity-bounded partial hierarchy shared by every epoch of one reduction.
template <detail::DeviceReductionAccumulator Accumulator,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy =
              DefaultDeviceReductionLaunchPolicy>
class PreparedDeviceReduction final {
public:
  using accumulator_type = Accumulator;
  using launch_policy = LaunchPolicy;

  constexpr PreparedDeviceReduction() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return valid_ && capacity_ == 0;
  }
  [[nodiscard]] constexpr std::uint64_t capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] constexpr std::uint32_t block_limit() const noexcept {
    return block_limit_;
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return levels_.total_bytes;
  }
  [[nodiscard]] constexpr std::uint8_t level_count() const noexcept {
    return levels_.level_count;
  }

  // Hard precondition: level < level_count().
  [[nodiscard]] constexpr std::uint32_t
  level_block_count(const std::size_t level) const noexcept {
    return levels_.counts[level];
  }

  // Hard precondition: level < level_count().
  [[nodiscard]] constexpr std::size_t
  level_byte_offset(const std::size_t level) const noexcept {
    return levels_.byte_offsets[level];
  }

  [[nodiscard]] friend constexpr bool
  operator==(const PreparedDeviceReduction &,
             const PreparedDeviceReduction &) noexcept = default;

private:
  constexpr PreparedDeviceReduction(
      const std::uint64_t capacity, const std::uint32_t block_limit,
      const detail::DeviceLevelLayout levels) noexcept
      : levels_(levels), capacity_(capacity), block_limit_(block_limit),
        valid_(true) {}

  detail::DeviceLevelLayout levels_{};
  std::uint64_t capacity_{};
  std::uint32_t block_limit_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

template <class Work>
inline constexpr bool is_device_field_reduction_work = false;

template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy,
          DeviceReductionLaunchPolicyType LaunchPolicy>
inline constexpr bool is_device_field_reduction_work<
    PreparedDeviceFieldReductionWork<Location, IndexPolicy, LaunchPolicy>> =
    true;

template <class Work>
concept DeviceReductionWorkType =
    std::same_as<Work, PreparedDeviceReductionWork> ||
    is_device_field_reduction_work<Work>;

// Field work carries its launch policy; contiguous work uses the reduction's.
// A variable template, not `typename Work::launch_policy` in a disjunction:
// nvcc's host output evaluates that member eagerly.
template <class Work, class LaunchPolicy>
inline constexpr bool device_reduction_work_policy_matches =
    std::same_as<Work, PreparedDeviceReductionWork>;

template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy,
          DeviceReductionLaunchPolicyType WorkPolicy, class LaunchPolicy>
inline constexpr bool device_reduction_work_policy_matches<
    PreparedDeviceFieldReductionWork<Location, IndexPolicy, WorkPolicy>,
    LaunchPolicy> = std::same_as<WorkPolicy, LaunchPolicy>;

// Hard preconditions: sample_count, first_work_count and block_limit are
// nonzero.
template <DeviceReductionLaunchPolicyType LaunchPolicy>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr PreparedDeviceReductionWork
make_device_reduction_work(const std::uint64_t sample_count,
                           const std::uint64_t first_work_count,
                           const std::uint32_t block_limit) noexcept {
  const std::uint32_t first =
      first_work_count < block_limit
          ? static_cast<std::uint32_t>(first_work_count)
          : block_limit;
  std::array<std::uint32_t, maximum_device_level_count> counts{};
  const std::uint8_t level_count =
      make_device_level_counts<device_reduction_segment_width<LaunchPolicy>,
                               true>(first, counts);
  return PrivateConstructionAccess::create<PreparedDeviceReductionWork>(
      sample_count, counts, level_count);
}

} // namespace detail

template <detail::DeviceReductionAccumulator Accumulator,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy,
          detail::checked_integer Capacity>
[[nodiscard]] constexpr ErrorCode try_prepare_device_reduction(
    const Capacity requested_capacity,
    const device::DeviceProperties &properties,
    PreparedDeviceReduction<Accumulator, LaunchPolicy> &output) noexcept {
  using Prepared = PreparedDeviceReduction<Accumulator, LaunchPolicy>;
  std::uint64_t capacity{};
  if (!detail::checked_integral_cast(requested_capacity, capacity)) {
    return ErrorCode::overflow;
  }
  if (capacity == 0) {
    output = detail::PrivateConstructionAccess::create<Prepared>(
        std::uint64_t{0}, std::uint32_t{0}, detail::DeviceLevelLayout{});
    return ErrorCode::success;
  }
  if (properties.maximum_threads_per_block < LaunchPolicy::block_threads ||
      properties.maximum_grid_size_x == 0) {
    return ErrorCode::invalid_argument;
  }
  const std::uint32_t block_limit =
      properties.maximum_grid_size_x < LaunchPolicy::maximum_blocks
          ? properties.maximum_grid_size_x
          : LaunchPolicy::maximum_blocks;

  // One first-level partial per input is the geometry-independent upper bound.
  // Contiguous and row-specialized active work normally use far fewer blocks.
  const std::uint32_t first = capacity < block_limit
                                  ? static_cast<std::uint32_t>(capacity)
                                  : block_limit;
  detail::DeviceLevelLayout levels{};
  if (!detail::try_make_device_level_layout<
          Accumulator, detail::device_reduction_segment_width<LaunchPolicy>,
          true>(first, levels)) {
    return ErrorCode::overflow;
  }
  output = detail::PrivateConstructionAccess::create<Prepared>(
      capacity, block_limit, levels);
  return ErrorCode::success;
}

template <detail::DeviceReductionAccumulator Accumulator,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy,
          detail::checked_integer Count>
[[nodiscard]] constexpr ErrorCode try_prepare_device_reduction_work(
    const PreparedDeviceReduction<Accumulator, LaunchPolicy> &prepared,
    const Count requested_count, PreparedDeviceReductionWork &output) noexcept {
  std::uint64_t count{};
  if (!prepared.valid()) {
    return ErrorCode::invalid_state;
  }
  if (!detail::checked_integral_cast(requested_count, count)) {
    return ErrorCode::overflow;
  }
  if (count > prepared.capacity()) {
    return ErrorCode::capacity_exceeded;
  }
  if (count == 0) {
    output = {};
    return ErrorCode::success;
  }
  constexpr std::uint64_t segment_width =
      detail::device_reduction_segment_width<LaunchPolicy>;
  output = detail::make_device_reduction_work<LaunchPolicy>(
      count, 1U + (count - 1U) / segment_width, prepared.block_limit());
  return ErrorCode::success;
}

// A zero cell axis is an empty domain for every staggered location and yields
// empty work.
template <detail::DeviceReductionAccumulator Accumulator,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy,
          detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_device_field_reduction_work(
    const PreparedDeviceReduction<Accumulator, LaunchPolicy> &prepared,
    const CartesianExtent3<typename IndexPolicy::count_type> cell_extents,
    PreparedDeviceFieldReductionWork<Location, IndexPolicy, LaunchPolicy>
        &output) noexcept {
  using Count = typename IndexPolicy::count_type;
  using Work =
      PreparedDeviceFieldReductionWork<Location, IndexPolicy, LaunchPolicy>;
  if (!prepared.valid()) {
    return ErrorCode::invalid_state;
  }
  if (!detail::cartesian_extent_is_nonempty(cell_extents)) {
    output = {};
    return ErrorCode::success;
  }
  CartesianExtent3<Count> samples{};
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(
      cell_extents, samples)));
  AllocatedAxisInterval<IndexPolicy> interval{};
  LPFX_TRY(for_each_axis([&](const auto axis) {
    return try_make_allocated_axis_interval<IndexPolicy>(
        samples.template get<decltype(axis)::value>(), 0U, 0U, interval);
  }));

  std::uint64_t count{};
  if (!detail::checked_volume(samples, count)) {
    return ErrorCode::overflow;
  }
  if (count > prepared.capacity()) {
    return ErrorCode::capacity_exceeded;
  }
  // count did not overflow and every axis is nonzero, so neither does rows.
  const std::uint64_t rows = static_cast<std::uint64_t>(samples.y()) *
                             static_cast<std::uint64_t>(samples.z());
  const std::uint64_t x_segment_count =
      detail::ceil_div(static_cast<std::uint64_t>(samples.x()),
                       detail::device_reduction_segment_width<LaunchPolicy>);
  std::uint64_t segment_count{};
  if (!detail::checked_multiply(x_segment_count, rows, segment_count)) {
    return ErrorCode::overflow;
  }
  output = detail::PrivateConstructionAccess::create<Work>(
      cell_extents, samples, x_segment_count, segment_count,
      detail::make_device_reduction_work<LaunchPolicy>(count, segment_count,
                                                       prepared.block_limit()));
  return ErrorCode::success;
}

namespace detail {

// One segment of segmented_reduction_kernel: elements
// [index * width, (index + 1) * width) of a row, bounded by end.
template <class Elements> struct ReductionSegment final {
  std::uint64_t index;
  std::uint64_t end;
  Elements elements;
};

// Contiguous input: the whole range is one row. Transform is false for the
// partial levels, whose elements are already accumulators.
template <bool Transform, class Count> struct LinearReductionSource final {
  using extent_type = Count;

  [[nodiscard]] static constexpr Count
  extent(const PreparedDeviceReductionWork work) noexcept {
    return work.sample_count();
  }

  template <std::uint64_t SegmentWidth>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr std::uint64_t
  segment_count(const Count count) noexcept {
    return 1U + (static_cast<std::uint64_t>(count) - 1U) / SegmentWidth;
  }

  template <class Semantics, class... Pointer>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr ReductionSegment<
      ReductionElements<Semantics, Transform, Pointer...>>
  segment(const std::uint64_t index, const Count count,
          const Pointer... pointers) noexcept {
    return {index, count, {pointers...}};
  }
};

// Field interior: x_segment_count() segments per (j, k) row.
template <class Work> struct FieldReductionSource final {
  using extent_type = Work;

  [[nodiscard]] static constexpr Work extent(const Work work) noexcept {
    return work;
  }

  template <std::uint64_t SegmentWidth>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr std::uint64_t
  segment_count(const Work work) noexcept {
    return work.segment_count();
  }

  template <class Semantics, class... Component>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr ReductionSegment<
      ReductionElements<Semantics, true, typename Component::element_type *...>>
  segment(const std::uint64_t index, const Work work,
          const Component... components) noexcept {
    using Coordinate = typename Work::index_policy::coordinate_type;
    const auto extents = work.sample_extents();
    const std::uint64_t size_y = static_cast<std::uint64_t>(extents.y());
    const std::uint64_t row = index / work.x_segment_count();
    const Coordinate j = static_cast<Coordinate>(row % size_y);
    const Coordinate k = static_cast<Coordinate>(row / size_y);
    return {index - row * work.x_segment_count(),
            static_cast<std::uint64_t>(extents.x()),
            {components.row_data(j, k)...}};
  }
};

using RangeReductionSource = LinearReductionSource<true, std::uint64_t>;
using PartialsReductionSource = LinearReductionSource<false, std::uint32_t>;

template <class Work>
using device_reduction_source_t =
    std::conditional_t<std::same_as<Work, PreparedDeviceReductionWork>,
                       RangeReductionSource, FieldReductionSource<Work>>;

// Launch seam, specialized statically by a device-language leaf:
//   enqueue_identity<Operation, Accumulator>(output, stream, accepted)
//   enqueue<Operation, Accumulator, LaunchPolicy>(work, scratch, prepared,
//       stream, accepted, final_output, operands...)
// Operands are one or two (dot) const pointers for PreparedDeviceReductionWork
// and one or two device component views for field work. A null final_output
// leaves the last level in scratch.
template <class Backend> struct DeviceReductionLaunch;

} // namespace detail

} // namespace lpfx

#endif
