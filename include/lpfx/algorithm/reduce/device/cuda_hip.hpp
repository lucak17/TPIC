#ifndef LPFX_ALGORITHM_REDUCE_DEVICE_CUDA_HIP_HPP
#define LPFX_ALGORITHM_REDUCE_DEVICE_CUDA_HIP_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/algorithm/reduce/device/workspace.hpp>
#include <lpfx/algorithm/reduce/operation.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace lpfx::detail {

template <class Semantics> struct DeviceReductionCombine final {
  using accumulator_type = typename Semantics::accumulator_type;

  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE accumulator_type
  operator()(const accumulator_type left,
             const accumulator_type right) const noexcept {
    return Semantics::combine(left, right);
  }
};

// Pointer operands become restrict-qualified kernel parameters.
template <class T> struct KernelOperand final {
  using type = T;
};

template <class T> struct KernelOperand<T *> final {
  using type = T *LPFX_RESTRICT;
};

// Combines items [0, Items) of one segment: item i is element
// base + BlockThreads * i. Bounded tests each index against the row end.
template <std::uint32_t Items, bool Bounded, std::uint64_t BlockThreads,
          class Semantics, class Row>
LPFX_DEVICE LPFX_ALWAYS_INLINE void
accumulate_segment_items(typename Semantics::accumulator_type &value,
                         const Row &row, const std::uint64_t base) noexcept {
#pragma unroll
  for (std::uint32_t item = 0; item != Items; ++item) {
    const std::uint64_t index = base + BlockThreads * item;
    if (!Bounded || index < row.end) {
      value = Semantics::combine(value, row.elements(index));
    }
  }
}

// Loads the in-range items [0, sizeof...(Item)) of a run of a partial
// segment, item i being element base + BlockThreads * i, then combines them
// in increasing order up to the block-uniform live item count of the run
// (1 <= live_items <= sizeof...(Item)), so warps issue no combine past it.
// Each item has exactly one combine site, predicated on its own range test as
// in the unsplit body: a product (dot, squared norm) then has a single use
// next to its sum, so FMA contraction is the same as in the unsplit body on
// every target. A dispatch to one body per live count gave the first product
// several uses; sm_120 codegen hoisted it into a separate mul.f64, which
// changed binary64 dot results in the last bit.
template <std::uint64_t BlockThreads, class Semantics, class Row,
          std::uint32_t... Item>
LPFX_DEVICE LPFX_ALWAYS_INLINE void accumulate_live_run(
    typename Semantics::accumulator_type &value, const Row &row,
    const std::uint64_t base, const std::uint64_t live_items,
    std::integer_sequence<std::uint32_t, Item...>) noexcept {
  using Raw = decltype(row.elements.load(std::uint64_t{}));
  Raw raw[sizeof...(Item)];
#pragma unroll
  for (std::uint32_t item = 0; item != sizeof...(Item); ++item) {
    const std::uint64_t index = base + BlockThreads * item;
    if (index < row.end) {
      raw[item] = row.elements.load(index);
    }
  }
#pragma unroll
  for (std::uint32_t item = 0; item != sizeof...(Item); ++item) {
    if (item == live_items) {
      break;
    }
    if (base + BlockThreads * item < row.end) {
      value = Semantics::combine(value, row.elements.apply(raw[item]));
    }
  }
}

// Width of the runs a partial segment is staged in. Staging all
// items_per_thread loads at once keeps them all live: 16 doubles (32
// registers) for a binary64 dot at the default policy, which raised the
// kernel to 46-62 registers. Runs of 4 keep it near the unsplit body's
// register count. With single-site combines (accumulate_live_run) runs of 4
// also measured equal or faster than one run for binary32 operands
// accumulated in binary64 (sm_120: dot 48 -> 40 registers, up to 3% faster).
inline constexpr std::uint32_t device_reduction_partial_run = 4U;

template <std::uint32_t Items, std::uint64_t BlockThreads, class Semantics,
          class Row, std::uint32_t... Run>
LPFX_DEVICE LPFX_ALWAYS_INLINE void accumulate_partial_runs(
    typename Semantics::accumulator_type &value, const Row &row,
    const std::uint64_t base, const std::uint64_t live_items,
    std::integer_sequence<std::uint32_t, Run...>) noexcept {
  constexpr std::uint64_t width = device_reduction_partial_run;
  // Runs in increasing item order; a run executes only while live items
  // remain, and combines min(live_items - first item, run width) of them.
  (void)((live_items > Run * width
              ? (accumulate_live_run<BlockThreads, Semantics>(
                     value, row, base + BlockThreads * Run * width,
                     live_items - Run * width < width
                         ? live_items - Run * width
                         : width,
                     std::make_integer_sequence<
                         std::uint32_t,
                         Items - Run * width < width ? Items - Run * width
                                                     : width>{}),
                 true)
              : false) &&
         ...);
}

// Partial segment (remaining < segment width). Each thread combines exactly
// its in-range items in increasing order, whichever staging is selected.
template <std::uint64_t BlockThreads, class Semantics, class Row,
          std::uint32_t... Item>
LPFX_DEVICE LPFX_ALWAYS_INLINE void accumulate_partial_segment(
    typename Semantics::accumulator_type &value, const Row &row,
    const std::uint64_t base, const std::uint64_t remaining,
    const std::integer_sequence<std::uint32_t, Item...> items) noexcept {
  constexpr std::uint32_t item_count = sizeof...(Item);
  constexpr std::uint32_t width = device_reduction_partial_run;
  const std::uint64_t live_items = 1U + (remaining - 1U) / BlockThreads;
  if constexpr (item_count > width) {
    accumulate_partial_runs<item_count, BlockThreads, Semantics>(
        value, row, base, live_items,
        std::make_integer_sequence<std::uint32_t,
                                   (item_count + width - 1U) / width>{});
  } else {
    accumulate_live_run<BlockThreads, Semantics>(value, row, base, live_items,
                                                 items);
  }
}

// Field rows split full segments from the one partial segment ending each
// row. Only binary64 accumulation gains from it, and only on reduced-rate
// FP64 targets (device_reduced_rate_fp64, reduce/operation.hpp):
// predicated-off FP64 combines still occupy the slow FP64 pipe there. The
// split raises static registers (sm_80: 38-40 against 26-32) and code size,
// so full-rate FP64 targets (sm_60/70/80/90/100, CDNA) compile the fully
// predicated body. For binary32 and for the linear sources (at most one
// partial segment) the extra block-uniform branch costs more than it saves
// (measured), so those keep the fully predicated body everywhere.
template <class Semantics, class LaunchPolicy, class Source>
inline constexpr bool device_reduction_splits_segments =
    device_reduced_rate_fp64 &&
    !std::same_as<Source, RangeReductionSource> &&
    !std::same_as<Source, PartialsReductionSource> &&
    std::same_as<typename Semantics::accumulator_type, double> &&
    LaunchPolicy::items_per_thread > 1U;

// Each block strides over the source's segments; every thread accumulates
// items_per_thread elements per segment, then the block publishes one partial.
template <class Semantics, class LaunchPolicy, class Source, class... Operand>
__global__ __launch_bounds__(LaunchPolicy::block_threads) void segmented_reduction_kernel(
    const typename KernelOperand<Operand>::type... operands,
    const typename Source::extent_type extent,
    typename Semantics::accumulator_type *LPFX_RESTRICT const partials) {
  using Accumulator = typename Semantics::accumulator_type;
  constexpr std::uint64_t block_threads = LaunchPolicy::block_threads;
  constexpr std::uint32_t items = LaunchPolicy::items_per_thread;
  constexpr std::uint64_t segment_width =
      device_reduction_segment_width<LaunchPolicy>;
  __shared__ Accumulator
      warp_scratch[device::subgroup::block_warp_count<LaunchPolicy::block_threads>];

  Accumulator value = Semantics::identity();
  const std::uint64_t segment_count =
      Source::template segment_count<segment_width>(extent);
  for (std::uint64_t segment = static_cast<std::uint64_t>(blockIdx.x);
       segment < segment_count;
       segment += static_cast<std::uint64_t>(gridDim.x)) {
    const auto row =
        Source::template segment<Semantics>(segment, extent, operands...);
    const std::uint64_t first = row.index * segment_width;
    const std::uint64_t base = first + static_cast<std::uint64_t>(threadIdx.x);
    if constexpr (device_reduction_splits_segments<Semantics, LaunchPolicy,
                                                   Source>) {
      // Block-uniform: a full segment needs no per-item test.
      const std::uint64_t remaining = row.end - first;
      if (remaining >= segment_width) {
        accumulate_segment_items<items, false, block_threads, Semantics>(
            value, row, base);
      } else {
        accumulate_partial_segment<block_threads, Semantics>(
            value, row, base, remaining,
            std::make_integer_sequence<std::uint32_t, items>{});
      }
    } else {
      accumulate_segment_items<items, true, block_threads, Semantics>(
          value, row, base);
    }
  }
  value = device::subgroup::block_reduce<LaunchPolicy::block_threads>(
      value, warp_scratch, Semantics::identity(),
      DeviceReductionCombine<Semantics>{});
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = value;
  }
}

template <class Semantics>
__global__ void
device_reduction_identity_kernel(typename Semantics::accumulator_type *const output) {
  *output = Semantics::identity();
}

template <device::DeviceKernelLaunchBackend Backend>
struct CudaHipDeviceReductionLaunch {
  using traits_type = device::BackendTraits<Backend>;
  using error_type = typename traits_type::error_type;
  using stream_type = typename traits_type::stream_type;

  template <ReductionOperation Operation, DeviceReductionAccumulator Accumulator>
  [[nodiscard]] static error_type
  enqueue_identity(Accumulator *const output, const stream_type stream,
                   bool &accepted) noexcept {
    return device::checked_kernel_launch<Backend>(accepted, [&] {
      device_reduction_identity_kernel<ReductionSemantics<Operation, Accumulator>>
          <<<1, 1, 0, stream>>>(output);
    });
  }

  template <ReductionOperation Operation, DeviceReductionAccumulator Accumulator,
            DeviceReductionLaunchPolicyType LaunchPolicy, class Work,
            class... Operand>
  [[nodiscard]] static error_type
  enqueue(const Work work, std::byte *const scratch,
          const PreparedDeviceReduction<Accumulator, LaunchPolicy> &prepared,
          const stream_type stream, bool &accepted,
          Accumulator *const final_output, const Operand... operands) noexcept {
    static_assert((std::is_trivially_copyable_v<Operand> && ...));
    using Semantics = ReductionSemantics<Operation, Accumulator>;
    using Source = device_reduction_source_t<Work>;
    constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
    const auto level_output = [&](const std::size_t level) noexcept {
      return final_output != nullptr && level + 1 == work.level_count()
                 ? final_output
                 : reinterpret_cast<Accumulator *>(
                       scratch + prepared.level_byte_offset(level));
    };

    error_type error = device::checked_kernel_launch<Backend>(accepted, [&] {
      segmented_reduction_kernel<Semantics, LaunchPolicy, Source, Operand...>
          <<<work.level_block_count(0), block_threads, 0, stream>>>(
              operands..., Source::extent(work), level_output(0));
    });
    if (!traits_type::succeeded(error)) {
      return error;
    }
    for (std::size_t level = 1; level != work.level_count(); ++level) {
      segmented_reduction_kernel<Semantics, LaunchPolicy,
                                 PartialsReductionSource, const Accumulator *>
          <<<work.level_block_count(level), block_threads, 0, stream>>>(
              reinterpret_cast<const Accumulator *>(
                  scratch + prepared.level_byte_offset(level - 1U)),
              work.level_block_count(level - 1U), level_output(level));
      error = traits_type::take_last_launch_error();
      if (!traits_type::succeeded(error)) {
        return error;
      }
    }
    return traits_type::success_error;
  }
};

#if defined(__CUDACC__)
template <>
struct DeviceReductionLaunch<device::CudaBackend> final
    : CudaHipDeviceReductionLaunch<device::CudaBackend> {};
#endif

#if defined(__HIPCC__)
template <>
struct DeviceReductionLaunch<device::HipBackend> final
    : CudaHipDeviceReductionLaunch<device::HipBackend> {};
#endif

} // namespace lpfx::detail

#endif

#endif
