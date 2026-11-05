#ifndef LPFX_ALGORITHM_PARTICLE_DEVICE_CUDA_HIP_SORT_HPP
#define LPFX_ALGORITHM_PARTICLE_DEVICE_CUDA_HIP_SORT_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <lpfx/algorithm/particle/device/launch.hpp>
#include <lpfx/algorithm/particle/key.hpp>
#include <lpfx/algorithm/scan/device/cuda_hip.hpp>
#include <lpfx/algorithm/scan/operation.hpp>
#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/core/memory_intrinsics.hpp>
#include <lpfx/data/particle/device/view.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace lpfx::detail {

// Sorting after a mover usually leaves many neighboring lanes in the same
// cell. Reserve their histogram/cursor range once, then assign unique lane
// ordinals within that range. Matching also handles noncontiguous equal keys
// and partial warps; no sorted-input assumption enters correctness.
template <bool NeedOrdinal>
[[nodiscard]] __device__ inline std::uint32_t reserve_device_particle_sort_bin(
    std::uint32_t *const counters, const std::uint32_t bin) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const unsigned participants = __activemask();
  const unsigned peers = __match_any_sync(participants, bin);
  const unsigned lane = static_cast<unsigned>(threadIdx.x) & 31U;
  const unsigned leader = static_cast<unsigned>(__ffs(peers) - 1);
  std::uint32_t first = 0;
  if (lane == leader)
    first = atomicAdd(counters + bin, static_cast<std::uint32_t>(__popc(peers)));
  if constexpr (NeedOrdinal) {
    first = __shfl_sync(peers, first, leader);
    return first + static_cast<std::uint32_t>(__popc(peers & ((1U << lane) - 1U)));
  } else {
    return 0;
  }
#else
  return atomicAdd(counters + bin, std::uint32_t{1});
#endif
}

__device__ inline void publish_device_particle_sort_failure(
    DeviceParticleSortSummary *const summary,
    const DeviceParticleSortFailureWord failure) noexcept {
  static_cast<void>(
      atomicCAS(&summary->failure,
                static_cast<std::uint32_t>(DeviceParticleSortFailureWord::none),
                static_cast<std::uint32_t>(failure)));
}

// A nonempty sort scans the counts into offsets[1, bin_count] and uses those
// words as the permutation cursors: bin b's cursor offsets[b + 1] starts at the
// exclusive prefix of b and ends at that of b + 1, so a successful permutation
// leaves the published image [0, prefixes..., particle_count]. Only offsets[0]
// needs clearing here. A failed sort skips the permutation, so its image is
// unspecified; consumers test the failure word before reading bins.
template <bool Empty, bool HasUpstream,
          DeviceScanLaunchPolicyType LaunchPolicy>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void clear_device_particle_sort_counts_kernel(
    const std::uint32_t bin_count, std::uint32_t *const counts,
    DeviceParticleSortSummary *const summary, std::uint32_t *const offsets,
    const std::uint32_t *const upstream_failure) {
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  const std::uint32_t bin =
      static_cast<std::uint32_t>(blockIdx.x) * block_threads +
      static_cast<std::uint32_t>(threadIdx.x);
  if (bin < bin_count) {
    counts[bin] = 0;
    if constexpr (Empty)
      offsets[bin] = 0;
  }
  if (bin == 0) {
    if constexpr (HasUpstream)
      summary->failure = upstream_failure == nullptr ? 0 : *upstream_failure;
    else
      summary->failure = 0;
    if constexpr (Empty)
      offsets[bin_count] = 0;
    else
      offsets[0] = 0;
  }
}

template <bool HasUpstream, class Key,
          DeviceScanLaunchPolicyType LaunchPolicy>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void histogram_device_particle_sort_kernel(
    const std::uint32_t particle_count, const std::uint32_t bin_count,
    const Key *const keys, std::uint32_t *const counts,
    DeviceParticleSortSummary *const summary,
    const std::uint32_t *const upstream_failure) {
  // This word is finalized by compaction and remains immutable throughout sort;
  // reading the histogram's concurrently atomically updated summary is unsafe.
  if constexpr (HasUpstream) {
    if (upstream_failure != nullptr && *upstream_failure != 0)
      return;
  }
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  const std::uint32_t particle =
      static_cast<std::uint32_t>(blockIdx.x) * block_threads +
      static_cast<std::uint32_t>(threadIdx.x);
  if (particle < particle_count) {
    const auto key = keys[particle];
    std::uint32_t bin{};
    if (!try_direct_cell_bin_index(key, bin_count, bin)) {
      publish_device_particle_sort_failure(
          summary, DeviceParticleSortFailureWord::invalid_key);
      return;
    }
    static_cast<void>(reserve_device_particle_sort_bin<false>(counts, bin));
  }
}

template <class Key, DeviceScanLaunchPolicyType LaunchPolicy>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void permutation_device_particle_sort_kernel(
    const std::uint32_t particle_count, const Key *const keys,
    std::uint32_t *const cursors, std::uint32_t *const permutation,
    const DeviceParticleSortSummary *const summary) {
  if (summary->failure != 0)
    return;
  const std::uint32_t particle =
      static_cast<std::uint32_t>(blockIdx.x) * LaunchPolicy::block_threads +
      static_cast<std::uint32_t>(threadIdx.x);
  if (particle < particle_count) {
    const auto bin = static_cast<std::uint32_t>(keys[particle]);
    permutation[particle] = reserve_device_particle_sort_bin<true>(cursors, bin);
  }
}

// Fixed-size representation copies start the destination implicit lifetime,
// including when this raw slot previously contained a different attribute type.
// The compiler resolves sizeof(T) and pointer alignment in each instantiation.
template <class T, DeviceScanLaunchPolicyType LaunchPolicy>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void scatter_device_particle_attribute_kernel(
    const std::uint32_t count, const T *const source, T *const destination,
    const std::uint32_t *const permutation,
    const DeviceParticleSortSummary *const summary) {
  if (summary->failure != 0)
    return;
  const std::uint32_t particle =
      static_cast<std::uint32_t>(blockIdx.x) * LaunchPolicy::block_threads +
      static_cast<std::uint32_t>(threadIdx.x);
  if (particle < count) {
    ::memcpy(
        assume_aligned_pointer<alignof(T)>(destination + permutation[particle]),
        assume_aligned_pointer<alignof(T)>(source + particle), sizeof(T));
  }
}

template <device::DeviceKernelLaunchBackend Backend>
struct CudaHipDeviceParticleSortLaunch {
  using Traits = device::BackendTraits<Backend>;

  template <bool Forward, class Schema, class LaunchPolicy,
            std::size_t... Stage>
  static typename Traits::error_type scatter_attributes(
      const std::uint32_t count,
      const BasicParticleView<Schema, DeviceSpace<Backend>, true> source,
      const BasicParticleView<Schema, DeviceSpace<Backend>, false> destination,
      const std::uint32_t *const permutation,
      const DeviceParticleSortSummary *const summary,
      const typename Traits::stream_type stream,
      std::index_sequence<Stage...>) noexcept {
    constexpr auto order =
        Forward ? Schema::forward_order : Schema::reverse_order;
    constexpr auto threads = LaunchPolicy::block_threads;
    const auto blocks = device_block_count(count, threads);
    auto error = Traits::success_error;
    const auto launch = [&]<std::size_t I>() noexcept {
      if (!Traits::succeeded(error))
        return;
      using T = typename Schema::template value_type<I>;
      scatter_device_particle_attribute_kernel<T, LaunchPolicy>
          <<<blocks, threads, 0, stream>>>(count, source.template get<I>(),
                                           destination.template get<I>(),
                                           permutation, summary);
      error = Traits::take_last_launch_error();
    };
    (launch.template operator()<order[Stage]>(), ...);
    return error;
  }

  template <ParticleKeyAccessor KeyAccessor, ParticleSchemaType Schema,
            DeviceScanLaunchPolicyType LaunchPolicy, bool HasUpstream>
  [[nodiscard]] static typename Traits::error_type enqueue(
      const std::uint32_t particle_count, const std::uint32_t bin_count,
      const BasicParticleView<Schema, DeviceSpace<Backend>, true> source,
      const BasicParticleView<Schema, DeviceSpace<Backend>, false> destination,
      std::uint32_t *const counts, std::uint32_t *const offsets,
      std::uint32_t *const permutation,
      DeviceParticleSortSummary *const summary,
      const std::uint32_t *const upstream_failure,
      std::byte *const scan_scratch,
      const PreparedDeviceScanWorkspace<std::uint32_t, LaunchPolicy>
          &scan_workspace,
      const bool forward, const typename Traits::stream_type stream,
      bool &accepted) noexcept {
    constexpr std::uint32_t threads = LaunchPolicy::block_threads;
    const auto particle_blocks =
        device_block_count(particle_count, threads);
    const auto bin_blocks = device_block_count(bin_count, threads);
    auto error = device::checked_kernel_launch<Backend>(accepted, [&]() noexcept {
      if (particle_count == 0) {
        clear_device_particle_sort_counts_kernel<true, HasUpstream,
                                                 LaunchPolicy>
            <<<bin_blocks, threads, 0, stream>>>(bin_count, counts, summary,
                                                 offsets, upstream_failure);
      } else {
        clear_device_particle_sort_counts_kernel<false, HasUpstream,
                                                 LaunchPolicy>
            <<<bin_blocks, threads, 0, stream>>>(bin_count, counts, summary,
                                                 offsets, upstream_failure);
      }
    });
    if (!Traits::succeeded(error) || particle_count == 0)
      return error;
    constexpr std::size_t key_index = particle_key_index<Schema, KeyAccessor>;
    using Key = particle_key_t<Schema, KeyAccessor>;
    histogram_device_particle_sort_kernel<HasUpstream, Key, LaunchPolicy>
        <<<particle_blocks, threads, 0, stream>>>(
            particle_count, bin_count, source.template get<key_index>(), counts,
            summary, upstream_failure);
    error = Traits::take_last_launch_error();
    if (!Traits::succeeded(error))
      return error;
    bool scan_accepted = false;
    error = DeviceScanLaunch<Backend>::template enqueue<
        false, ScanSum<std::uint32_t>, std::uint32_t, LaunchPolicy>(
        counts, offsets + 1, bin_count, std::uint32_t{0}, scan_scratch,
        scan_workspace, stream, scan_accepted);
    if (!Traits::succeeded(error))
      return error;
    if (!scan_accepted)
      std::abort();
    permutation_device_particle_sort_kernel<Key, LaunchPolicy>
        <<<particle_blocks, threads, 0, stream>>>(
            particle_count, source.template get<key_index>(), offsets + 1,
            permutation, summary);
    error = Traits::take_last_launch_error();
    if (!Traits::succeeded(error))
      return error;
    using Stages = std::make_index_sequence<Schema::attribute_count>;
    if (!forward)
      return scatter_attributes<false, Schema, LaunchPolicy>(
          particle_count, source, destination, permutation, summary, stream,
          Stages{});
    return scatter_attributes<true, Schema, LaunchPolicy>(
        particle_count, source, destination, permutation, summary, stream,
        Stages{});
  }
};

#if defined(__CUDACC__)
template <>
struct DeviceParticleSortLaunch<device::CudaBackend> final
    : CudaHipDeviceParticleSortLaunch<device::CudaBackend> {};
#endif
#if defined(__HIPCC__)
template <>
struct DeviceParticleSortLaunch<device::HipBackend> final
    : CudaHipDeviceParticleSortLaunch<device::HipBackend> {};
#endif

} // namespace lpfx::detail

#endif

#endif
