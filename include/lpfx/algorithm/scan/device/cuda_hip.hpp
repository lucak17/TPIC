#ifndef LPFX_ALGORITHM_SCAN_DEVICE_CUDA_HIP_HPP
#define LPFX_ALGORITHM_SCAN_DEVICE_CUDA_HIP_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/algorithm/scan/device/workspace.hpp>
#include <lpfx/algorithm/scan/operation.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace lpfx::detail {

// Integral combines are exact in any grouping, so integral blocks use a
// two-level warp-shuffle scan (two block barriers). Floating-point values keep
// the shared-memory up/down-sweep tree and its exact combine order. The
// warp-count bound keeps the warp totals scannable by one warp; any other
// block size (e.g. 32 threads on wave64) keeps the tree.
template <class T, std::uint32_t BlockThreads>
inline constexpr bool device_scan_uses_warp_shuffle =
    std::integral<T> && sizeof(T) <= sizeof(unsigned long long) &&
    BlockThreads % device::subgroup::detail::native_warp_size == 0 &&
    BlockThreads / device::subgroup::detail::native_warp_size <=
        device::subgroup::detail::native_warp_size;

// Native shuffle carrier of an integral scan value: exact round trip.
template <class T>
using device_scan_shuffle_type = std::conditional_t<
    sizeof(T) <= sizeof(unsigned int),
    std::conditional_t<std::is_signed_v<T>, int, unsigned int>,
    std::conditional_t<std::is_signed_v<T>, long long, unsigned long long>>;

// Requires the complete, converged physical warp.
template <class T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
device_scan_shuffle_up(const T value, const std::uint32_t delta) noexcept {
  return static_cast<T>(device::subgroup::warp_shuffle_up(
      static_cast<device_scan_shuffle_type<T>>(value), delta));
}

template <bool Inclusive, bool ApplyInitial, bool PublishTotals,
          class Operation, class T, std::uint32_t BlockThreads>
LPFX_DEVICE LPFX_ALWAYS_INLINE void
device_scan_block_warp_shuffle(const T *const source, T *const destination,
                               const std::uint32_t count, const T initial,
                               T *const block_totals) noexcept {
  constexpr std::uint32_t warp_width =
      device::subgroup::detail::native_warp_size;
  constexpr std::uint32_t warp_count = BlockThreads / warp_width;
  __shared__ T warp_totals[warp_count];

  const std::uint32_t thread = static_cast<std::uint32_t>(threadIdx.x);
  const std::uint32_t lane = thread % warp_width;
  const std::uint32_t warp = thread / warp_width;
  const std::uint32_t block = static_cast<std::uint32_t>(blockIdx.x);
  const std::uint32_t index = block * BlockThreads + thread;
  const T input = index < count ? source[index] : Operation::identity();

  T inclusive = input;
#pragma unroll
  for (std::uint32_t delta = 1; delta < warp_width; delta <<= 1) {
    const T lower = device_scan_shuffle_up(inclusive, delta);
    if (lane >= delta) {
      inclusive = Operation::combine(lower, inclusive);
    }
  }
  if (lane == warp_width - 1) {
    warp_totals[warp] = inclusive;
  }
  __syncthreads();
  if (warp == 0) {
    T total = lane < warp_count ? warp_totals[lane] : Operation::identity();
#pragma unroll
    for (std::uint32_t delta = 1; delta < warp_count; delta <<= 1) {
      const T lower = device_scan_shuffle_up(total, delta);
      if (lane >= delta) {
        total = Operation::combine(lower, total);
      }
    }
    if (lane < warp_count) {
      warp_totals[lane] = total;
    }
  }
  __syncthreads();

  // The whole warp shuffles here, converged.
  T prefix = inclusive;
  if constexpr (!Inclusive) {
    prefix = device_scan_shuffle_up(inclusive, 1U);
    if (lane == 0) {
      prefix = Operation::identity();
    }
  }
  if constexpr (PublishTotals) {
    if (thread == 0) {
      T total = warp_totals[warp_count - 1];
      if constexpr (ApplyInitial) {
        if (block == 0) {
          total = Operation::combine(initial, total);
        }
      }
      block_totals[block] = total;
    }
  }
  if (index < count) {
    if (warp != 0) {
      prefix = Operation::combine(warp_totals[warp - 1], prefix);
    }
    if constexpr (ApplyInitial) {
      if (block == 0) {
        prefix = Operation::combine(initial, prefix);
      }
    }
    destination[index] = prefix;
  }
}

template <bool Inclusive, bool ApplyInitial, bool PublishTotals,
          class Operation, class T, std::uint32_t BlockThreads>
LPFX_DEVICE LPFX_ALWAYS_INLINE void
device_scan_block_tree(const T *const source, T *const destination,
                       const std::uint32_t count, const T initial,
                       T *const block_totals) noexcept {
  constexpr std::uint32_t block_threads = BlockThreads;
  __shared__ T shared[block_threads];

  const std::uint32_t lane = static_cast<std::uint32_t>(threadIdx.x);
  const std::uint32_t block = static_cast<std::uint32_t>(blockIdx.x);
  const std::uint32_t index = block * block_threads + lane;
  const T input = index < count ? source[index] : Operation::identity();
  shared[lane] = input;

  __syncthreads();
  for (std::uint32_t offset = 1; offset < block_threads; offset <<= 1) {
    const std::uint32_t tree_index = (lane + 1) * (offset << 1) - 1;
    if (tree_index < block_threads) {
      shared[tree_index] =
          Operation::combine(shared[tree_index - offset], shared[tree_index]);
    }
    __syncthreads();
  }
  if (lane == 0) {
    if constexpr (PublishTotals) {
      T total = shared[block_threads - 1];
      if constexpr (ApplyInitial) {
        if (block == 0) {
          total = Operation::combine(initial, total);
        }
      }
      block_totals[block] = total;
    }
    shared[block_threads - 1] = Operation::identity();
  }
  __syncthreads();
  for (std::uint32_t offset = block_threads >> 1; offset != 0; offset >>= 1) {
    const std::uint32_t tree_index = (lane + 1) * (offset << 1) - 1;
    if (tree_index < block_threads) {
      const T lower = shared[tree_index - offset];
      shared[tree_index - offset] = shared[tree_index];
      shared[tree_index] = Operation::combine(shared[tree_index], lower);
    }
    __syncthreads();
  }

  if (index < count) {
    T prefix = shared[lane];
    if constexpr (ApplyInitial) {
      if (block == 0) {
        prefix = Operation::combine(initial, prefix);
      }
    }
    if constexpr (Inclusive) {
      destination[index] = Operation::combine(prefix, input);
    } else {
      destination[index] = prefix;
    }
  }
}

template <bool Inclusive, bool ApplyInitial, bool PublishTotals,
          class Operation, ScanValue T, DeviceScanLaunchPolicyType LaunchPolicy>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void device_scan_blocks_kernel(
    const T *const source, T *const destination, const std::uint32_t count,
    const T initial, T *const block_totals) {
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  if constexpr (device_scan_uses_warp_shuffle<T, block_threads>) {
    device_scan_block_warp_shuffle<Inclusive, ApplyInitial, PublishTotals,
                                   Operation, T, block_threads>(
        source, destination, count, initial, block_totals);
  } else {
    device_scan_block_tree<Inclusive, ApplyInitial, PublishTotals, Operation, T,
                           block_threads>(source, destination, count, initial,
                                          block_totals);
  }
}

template <class Operation, ScanValue T, DeviceScanLaunchPolicyType LaunchPolicy>
__global__ __launch_bounds__(LaunchPolicy::block_threads) void device_scan_add_block_prefixes_kernel(
    T *const values, const std::uint32_t count, const T *const block_prefixes) {
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  const std::uint32_t index =
      static_cast<std::uint32_t>(blockIdx.x) * block_threads +
      static_cast<std::uint32_t>(threadIdx.x);
  if (index < count) {
    const std::uint32_t block = index / block_threads;
    values[index] = Operation::combine(block_prefixes[block], values[index]);
  }
}

template <device::DeviceKernelLaunchBackend Backend>
struct CudaHipDeviceScanLaunch {
  template <bool Inclusive, class Operation, ScanValue T,
            DeviceScanLaunchPolicyType LaunchPolicy>
    requires ScanOperationFor<Operation, T>
  [[nodiscard]] static typename device::BackendTraits<Backend>::error_type
  enqueue(const T *const source, T *const destination,
          const std::uint32_t count, const T initial, std::byte *const scratch,
          const PreparedDeviceScanWorkspace<T, LaunchPolicy> &prepared,
          const typename device::BackendTraits<Backend>::stream_type stream,
          bool &accepted) noexcept {
    using Traits = device::BackendTraits<Backend>;
    constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
    const std::uint32_t base_block_count =
        device_block_count(count, block_threads);
    // Level counts of this count; the byte offsets are the capacity's.
    std::array<std::uint32_t, maximum_device_level_count> level_counts{};
    const std::size_t level_count =
        make_device_level_counts<block_threads, false>(base_block_count,
                                                       level_counts);

    const auto level_pointer = [&](const std::size_t level) noexcept {
      return reinterpret_cast<T *>(scratch + prepared.level_byte_offset(level));
    };

    if (level_count == 0) {
      return device::checked_kernel_launch<Backend>(accepted, [&] {
        device_scan_blocks_kernel<Inclusive, true, false, Operation, T,
                                  LaunchPolicy>
            <<<base_block_count, block_threads, 0, stream>>>(
                source, destination, count, initial, nullptr);
      });
    }

    typename Traits::error_type error =
        device::checked_kernel_launch<Backend>(accepted, [&] {
          device_scan_blocks_kernel<Inclusive, true, true, Operation, T,
                                    LaunchPolicy>
              <<<base_block_count, block_threads, 0, stream>>>(
                  source, destination, count, initial, level_pointer(0));
        });
    if (!Traits::succeeded(error)) {
      return error;
    }

    for (std::size_t level = 0; level + 1 < level_count; ++level) {
      T *const values = level_pointer(level);
      const std::uint32_t value_count = level_counts[level];
      device_scan_blocks_kernel<false, false, true, Operation, T, LaunchPolicy>
          <<<device_block_count(value_count, block_threads), block_threads, 0,
             stream>>>(values, values, value_count, Operation::identity(),
                       level_pointer(level + 1));
      error = Traits::take_last_launch_error();
      if (!Traits::succeeded(error)) {
        return error;
      }
    }

    const std::size_t top_level = level_count - 1;
    T *const top_values = level_pointer(top_level);
    device_scan_blocks_kernel<false, false, false, Operation, T, LaunchPolicy>
        <<<1, block_threads, 0, stream>>>(top_values, top_values,
                                          level_counts[top_level],
                                          Operation::identity(), nullptr);
    error = Traits::take_last_launch_error();
    if (!Traits::succeeded(error)) {
      return error;
    }

    for (std::size_t higher = top_level; higher != 0; --higher) {
      const std::size_t lower = higher - 1;
      const std::uint32_t lower_count = level_counts[lower];
      device_scan_add_block_prefixes_kernel<Operation, T, LaunchPolicy>
          <<<device_block_count(lower_count, block_threads), block_threads, 0,
             stream>>>(level_pointer(lower), lower_count,
                       level_pointer(higher));
      error = Traits::take_last_launch_error();
      if (!Traits::succeeded(error)) {
        return error;
      }
    }

    device_scan_add_block_prefixes_kernel<Operation, T, LaunchPolicy>
        <<<base_block_count, block_threads, 0, stream>>>(destination, count,
                                                         level_pointer(0));
    return Traits::take_last_launch_error();
  }
};

#if defined(__CUDACC__)
template <>
struct DeviceScanLaunch<device::CudaBackend> final
    : CudaHipDeviceScanLaunch<device::CudaBackend> {};
#endif

#if defined(__HIPCC__)
template <>
struct DeviceScanLaunch<device::HipBackend> final
    : CudaHipDeviceScanLaunch<device::HipBackend> {};
#endif

} // namespace lpfx::detail

#endif

#endif
