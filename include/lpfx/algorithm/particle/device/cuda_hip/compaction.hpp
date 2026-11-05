#ifndef LPFX_ALGORITHM_PARTICLE_DEVICE_CUDA_HIP_COMPACTION_HPP
#define LPFX_ALGORITHM_PARTICLE_DEVICE_CUDA_HIP_COMPACTION_HPP

#include <bit>
#include <cstdint>
#include <type_traits>

#include <lpfx/algorithm/particle/device/launch.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/data/particle/device/view.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace lpfx::detail {

static_assert(std::is_same_v<std::uint32_t, unsigned int>);

template <class LaunchPolicy, class RetainedClassification,
          class ClassificationReader>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void count_device_unstable_hole_fill_blocks_kernel(
    const std::uint32_t particle_count,
    const std::uint32_t expected_retained_count,
    const ClassificationReader classification,
    DeviceUnstableHoleFillBlockPrefix *const block_counts) {
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  constexpr std::uint32_t sub_bucket_count = LaunchPolicy::count_sub_buckets;
  static_assert(block_threads >= 32 && block_threads <= 1024);
  static_assert(sub_bucket_count != 0 && sub_bucket_count <= 32 &&
                (sub_bucket_count & (sub_bucket_count - 1)) == 0);
  constexpr std::uint32_t shared_count = 2 * sub_bucket_count;
  __shared__ std::uint32_t shared_counts[shared_count];

  const std::uint32_t lane = threadIdx.x;
  if constexpr (block_threads >= shared_count) {
    if (lane < shared_count) {
      shared_counts[lane] = 0;
    }
  } else {
    for (std::uint32_t index = lane; index < shared_count;
         index += block_threads) {
      shared_counts[index] = 0;
    }
  }
  __syncthreads();

  const std::uint32_t source = blockIdx.x * block_threads + lane;
  if (source < particle_count) {
    const bool retained =
        RetainedClassification::retained(classification[source]);
    std::uint32_t category = 2;
    if (source < expected_retained_count) {
      category = retained ? 2 : 0;
    } else {
      category = retained ? 1 : 2;
    }
    if (category < 2) {
      const std::uint32_t sub_bucket = lane & (sub_bucket_count - 1);
      static_cast<void>(
          atomicAdd(shared_counts + category * sub_bucket_count + sub_bucket,
                    std::uint32_t{1}));
    }
  }
  __syncthreads();

  if (lane < 2) {
    std::uint32_t count{};
#pragma unroll
    for (std::uint32_t sub_bucket = 0; sub_bucket != sub_bucket_count;
         ++sub_bucket) {
      count += shared_counts[lane * sub_bucket_count + sub_bucket];
    }
    if (lane == 0) {
      block_counts[blockIdx.x].hole = count;
    } else {
      block_counts[blockIdx.x].filler = count;
    }
  }
}

// Live lanes of one physical warp of a LaunchPolicy block. A 32-thread block
// occupies half of a HIP wave64; only its live lanes take part in cross-lane
// operations.
template <class LaunchPolicy>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint32_t
hole_fill_live_warp_lanes() noexcept {
  constexpr std::uint32_t threads = LaunchPolicy::block_threads;
  constexpr std::uint32_t warp_threads = device::subgroup::warp_size();
  return threads < warp_threads ? threads : warp_threads;
}

// Turns warp-exclusive (hole, filler) prefixes into block-exclusive prefixes
// in thread order and returns the block totals. The publishing lane of each
// warp supplies that warp's totals; one barrier orders them. Every block
// thread enters converged. The caller must pass a block barrier before the
// next call rewrites warp_holes and warp_fillers.
template <class LaunchPolicy>
LPFX_DEVICE LPFX_ALWAYS_INLINE void add_preceding_hole_fill_warps(
    std::uint32_t &hole, std::uint32_t &filler, const bool publish,
    const std::uint32_t warp_hole_total, const std::uint32_t warp_filler_total,
    std::uint32_t *const warp_holes, std::uint32_t *const warp_fillers,
    std::uint32_t &hole_total, std::uint32_t &filler_total) noexcept {
  constexpr std::uint32_t threads = LaunchPolicy::block_threads;
  static_assert(threads >= 32 && threads <= 1024);
  static_assert((threads & (threads - 1)) == 0);
  constexpr std::uint32_t warp_threads = device::subgroup::warp_size();
  constexpr std::uint32_t warp_count =
      (threads + warp_threads - 1) / warp_threads;
  const auto warp = static_cast<std::uint32_t>(threadIdx.x) / warp_threads;
  if (publish) {
    warp_holes[warp] = warp_hole_total;
    warp_fillers[warp] = warp_filler_total;
  }
  __syncthreads();
  std::uint32_t holes_all = 0;
  std::uint32_t fillers_all = 0;
#pragma unroll
  for (std::uint32_t index = 0; index != warp_count; ++index) {
    const std::uint32_t warp_hole = warp_holes[index];
    const std::uint32_t warp_filler = warp_fillers[index];
    if (index == warp) {
      hole += holes_all;
      filler += fillers_all;
    }
    holes_all += warp_hole;
    fillers_all += warp_filler;
  }
  hole_total = holes_all;
  filler_total = fillers_all;
}

// Exclusive block scan of one (hole, filler) count pair per thread in thread
// order: an in-warp shuffle scan, then the preceding warps' totals.
template <class LaunchPolicy>
LPFX_DEVICE LPFX_ALWAYS_INLINE void block_exclusive_scan_hole_fill_pair(
    std::uint32_t &hole, std::uint32_t &filler,
    std::uint32_t *const warp_holes, std::uint32_t *const warp_fillers,
    std::uint32_t &hole_total, std::uint32_t &filler_total) noexcept {
  constexpr std::uint32_t live_lanes =
      hole_fill_live_warp_lanes<LaunchPolicy>();
  constexpr auto participants =
      device::subgroup::warp_low_lane_mask(live_lanes);
  const auto warp_lane = device::subgroup::warp_lane();
  std::uint32_t inclusive_hole = hole;
  std::uint32_t inclusive_filler = filler;
#pragma unroll
  for (std::uint32_t delta = 1; delta < live_lanes; delta <<= 1) {
    const std::uint32_t lower_hole =
        device::subgroup::warp_shuffle_up(participants, inclusive_hole, delta);
    const std::uint32_t lower_filler = device::subgroup::warp_shuffle_up(
        participants, inclusive_filler, delta);
    if (warp_lane >= delta) {
      inclusive_hole += lower_hole;
      inclusive_filler += lower_filler;
    }
  }
  hole = inclusive_hole - hole;
  filler = inclusive_filler - filler;
  add_preceding_hole_fill_warps<LaunchPolicy>(
      hole, filler, warp_lane == live_lanes - 1, inclusive_hole,
      inclusive_filler, warp_holes, warp_fillers, hole_total, filler_total);
}

// Exclusive block ranks of one (hole, filler) flag pair per thread in thread
// order: warp ballots rank the flags, then the preceding warps' totals.
template <class LaunchPolicy>
LPFX_DEVICE LPFX_ALWAYS_INLINE void block_exclusive_rank_hole_fill_flags(
    const bool hole, const bool filler, std::uint32_t &hole_rank,
    std::uint32_t &filler_rank, std::uint32_t *const warp_holes,
    std::uint32_t *const warp_fillers, std::uint32_t &hole_total,
    std::uint32_t &filler_total) noexcept {
  constexpr auto participants = device::subgroup::warp_low_lane_mask(
      hole_fill_live_warp_lanes<LaunchPolicy>());
  const auto warp_lane = device::subgroup::warp_lane();
  const auto hole_mask = device::subgroup::warp_ballot(participants, hole);
  const auto filler_mask = device::subgroup::warp_ballot(participants, filler);
  const auto preceding = device::subgroup::warp_low_lane_mask(warp_lane);
  hole_rank = device::subgroup::warp_population_count(hole_mask & preceding);
  filler_rank =
      device::subgroup::warp_population_count(filler_mask & preceding);
  add_preceding_hole_fill_warps<LaunchPolicy>(
      hole_rank, filler_rank, warp_lane == 0,
      device::subgroup::warp_population_count(hole_mask),
      device::subgroup::warp_population_count(filler_mask), warp_holes,
      warp_fillers, hole_total, filler_total);
}

// Trivially default-constructible staging copy of a block pair: CUDA/HIP reject
// __shared__ arrays whose element constructor initializes members.
struct alignas(2 * sizeof(std::uint32_t)) StagedHoleFillPair final {
  std::uint32_t hole;
  std::uint32_t filler;
};
static_assert(std::is_trivially_default_constructible_v<StagedHoleFillPair>);

// Shared slot of staged block pair i: one pad pair per 16 keeps both the
// coalesced (i = r * threads + lane) and the raking (i = lane * rake + r)
// accesses free of bank conflicts for every power-of-two rake up to 16.
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint32_t
staged_hole_fill_slot(const std::uint32_t index) noexcept {
  return index + index / 16;
}

// One block scans every block's pair. Each chunk of block_threads * rake
// entries is staged in shared memory with coalesced accesses; each thread
// rakes rake contiguous entries around one block scan, carrying across chunks.
template <class LaunchPolicy>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void scan_device_unstable_hole_fill_block_prefixes_kernel(
    const std::uint32_t block_count, const std::uint32_t maximum_moves,
    DeviceUnstableHoleFillBlockPrefix *const block_prefixes,
    DeviceUnstableHoleFillSummary *const summary) {
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  static_assert(block_threads >= 32 && block_threads <= 1024);
  static_assert((block_threads & (block_threads - 1)) == 0);
  constexpr std::uint32_t warp_count =
      (block_threads + device::subgroup::warp_size() - 1) /
      device::subgroup::warp_size();
  // At most 16 KiB of staged pairs (plus padding) for every block size.
  // Rake 8 balances large counts (fewer chunks) against the partial single
  // chunk of small counts; 16 is faster above ~1M particles only.
  constexpr std::uint32_t rake =
      2048 / block_threads < 8 ? 2048 / block_threads : 8;
  constexpr std::uint32_t chunk = block_threads * rake;
  __shared__ std::uint32_t warp_holes[warp_count];
  __shared__ std::uint32_t warp_fillers[warp_count];
  __shared__ StagedHoleFillPair staged[staged_hole_fill_slot(chunk)];

  const std::uint32_t lane = threadIdx.x;
  std::uint32_t hole_count = 0;
  std::uint32_t filler_count = 0;
  for (std::uint32_t base = 0; base < block_count; base += chunk) {
#pragma unroll
    for (std::uint32_t r = 0; r != rake; ++r) {
      const std::uint32_t index = base + r * block_threads + lane;
      const DeviceUnstableHoleFillBlockPrefix counts =
          index < block_count ? block_prefixes[index]
                              : DeviceUnstableHoleFillBlockPrefix{};
      staged[staged_hole_fill_slot(r * block_threads + lane)] =
          StagedHoleFillPair{counts.hole, counts.filler};
    }
    __syncthreads();
    std::uint32_t hole = 0;
    std::uint32_t filler = 0;
#pragma unroll
    for (std::uint32_t r = 0; r != rake; ++r) {
      const StagedHoleFillPair counts =
          staged[staged_hole_fill_slot(lane * rake + r)];
      hole += counts.hole;
      filler += counts.filler;
    }
    std::uint32_t chunk_holes{};
    std::uint32_t chunk_fillers{};
    block_exclusive_scan_hole_fill_pair<LaunchPolicy>(
        hole, filler, warp_holes, warp_fillers, chunk_holes, chunk_fillers);
    hole += hole_count;
    filler += filler_count;
#pragma unroll
    for (std::uint32_t r = 0; r != rake; ++r) {
      const std::uint32_t slot = staged_hole_fill_slot(lane * rake + r);
      const StagedHoleFillPair counts = staged[slot];
      staged[slot] = StagedHoleFillPair{hole, filler};
      hole += counts.hole;
      filler += counts.filler;
    }
    __syncthreads();
#pragma unroll
    for (std::uint32_t r = 0; r != rake; ++r) {
      const std::uint32_t index = base + r * block_threads + lane;
      if (index < block_count) {
        const StagedHoleFillPair prefix =
            staged[staged_hole_fill_slot(r * block_threads + lane)];
        block_prefixes[index] =
            DeviceUnstableHoleFillBlockPrefix{prefix.hole, prefix.filler};
      }
    }
    hole_count += chunk_holes;
    filler_count += chunk_fillers;
    // Orders these staged reads and warp-total reads before the next chunk.
    __syncthreads();
  }

  if (lane == 0) {
    const bool valid =
        hole_count == filler_count && hole_count <= maximum_moves;
    summary->hole_count = hole_count;
    summary->filler_count = filler_count;
    summary->move_count = valid ? hole_count : 0;
    summary->failure = static_cast<std::uint32_t>(
        valid ? DeviceUnstableHoleFillFailureWord::none
              : DeviceUnstableHoleFillFailureWord::invariant_mismatch);
  }
}

template <class LaunchPolicy, class RetainedClassification,
          class ClassificationReader>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void scatter_device_unstable_hole_fill_lists_kernel(
    const std::uint32_t particle_count,
    const std::uint32_t expected_retained_count,
    const ClassificationReader classification,
    const DeviceUnstableHoleFillBlockPrefix *const block_prefixes,
    std::uint32_t *const holes, std::uint32_t *const fillers,
    const DeviceUnstableHoleFillSummary *const summary) {
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  static_assert(block_threads >= 32 && block_threads <= 1024);
  static_assert((block_threads & (block_threads - 1)) == 0);
  constexpr std::uint32_t warp_count =
      (block_threads + device::subgroup::warp_size() - 1) /
      device::subgroup::warp_size();
  __shared__ std::uint32_t warp_holes[warp_count];
  __shared__ std::uint32_t warp_fillers[warp_count];

  // Grid-uniform: every thread of every block returns or enters the scan.
  if (summary->failure !=
      static_cast<std::uint32_t>(DeviceUnstableHoleFillFailureWord::none)) {
    return;
  }

  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t source = blockIdx.x * block_threads + lane;
  bool hole = false;
  bool filler = false;
  if (source < particle_count) {
    const bool retained =
        RetainedClassification::retained(classification[source]);
    hole = source < expected_retained_count && !retained;
    filler = source >= expected_retained_count && retained;
  }
  std::uint32_t hole_rank{};
  std::uint32_t filler_rank{};
  std::uint32_t hole_total{};
  std::uint32_t filler_total{};
  block_exclusive_rank_hole_fill_flags<LaunchPolicy>(
      hole, filler, hole_rank, filler_rank, warp_holes, warp_fillers,
      hole_total, filler_total);

  const DeviceUnstableHoleFillBlockPrefix block_prefix =
      block_prefixes[blockIdx.x];
  if (hole) {
    holes[block_prefix.hole + hole_rank] = source;
  }
  if (filler) {
    const std::uint32_t ascending_rank =
        block_prefix.filler + filler_rank;
    fillers[summary->filler_count - 1 - ascending_rank] = source;
  }
}

template <class LaunchPolicy, class Mover>
__global__
__launch_bounds__(LaunchPolicy::block_threads) void move_device_unstable_hole_fill_records_kernel(
    const std::uint32_t *const holes, const std::uint32_t *const fillers,
    DeviceUnstableHoleFillSummary *const summary, Mover mover) {
  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  static_assert(block_threads >= 32 && block_threads <= 1024);
  const std::uint32_t move = blockIdx.x * block_threads + threadIdx.x;
  if (move < summary->move_count) {
    mover(holes[move], fillers[move]);
  }
}

// Each source flag is loaded once. Ballots give each selected lane its rank;
// only the small per-warp counts need a block prefix. Two block barriers and
// at most two global reservations replace a full shared-memory particle scan.
// Failed classification counts can exceed maximum_moves, so guard list writes
// against the maximum-move allocation before the mover audits it.
template <class LaunchPolicy, class RetainedClassification, class Reader>
__global__ __launch_bounds__(LaunchPolicy::block_threads)
void build_device_unordered_hole_fill_lists_kernel(
    const std::uint32_t particle_count, const std::uint32_t retained_count,
    const Reader classification, std::uint32_t *const holes,
    std::uint32_t *const fillers, DeviceUnstableHoleFillSummary *const summary) {
  constexpr auto threads = LaunchPolicy::block_threads;
  constexpr auto warp_threads = device::subgroup::warp_size();
  constexpr auto warp_count = (threads + warp_threads - 1) / warp_threads;
  __shared__ std::uint32_t hole_prefixes[warp_count], filler_prefixes[warp_count];
  __shared__ std::uint32_t bases[2];
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  const auto warp = lane / warp_threads;
  const auto warp_lane = device::subgroup::warp_lane();
  const auto source = static_cast<std::uint32_t>(blockIdx.x) * threads + lane;
  bool hole = false, filler = false;
  if (source < particle_count) {
    const bool keep = RetainedClassification::retained(classification[source]);
    hole = source < retained_count && !keep;
    filler = source >= retained_count && keep;
  }
  // A 32-thread launch can occupy half of a HIP wave64. Every launched lane,
  // including the last block's out-of-range sources, participates in ballots.
  constexpr auto participants = device::subgroup::warp_low_lane_mask(
      threads < warp_threads ? threads : warp_threads);
  const auto hole_mask = device::subgroup::warp_ballot(participants, hole);
  const auto filler_mask = device::subgroup::warp_ballot(participants, filler);
  const auto preceding = device::subgroup::warp_low_lane_mask(warp_lane);
  const auto hole_rank = device::subgroup::warp_population_count(hole_mask & preceding);
  const auto filler_rank = device::subgroup::warp_population_count(filler_mask & preceding);
  if (warp_lane == 0) {
    hole_prefixes[warp] = device::subgroup::warp_population_count(hole_mask);
    filler_prefixes[warp] = device::subgroup::warp_population_count(filler_mask);
  }
  __syncthreads();
  if (lane == 0) {
    std::uint32_t hole_total{}, filler_total{};
#pragma unroll
    for (std::uint32_t index = 0; index != warp_count; ++index) {
      const auto holes_in_warp = hole_prefixes[index];
      const auto fillers_in_warp = filler_prefixes[index];
      hole_prefixes[index] = hole_total;
      filler_prefixes[index] = filler_total;
      hole_total += holes_in_warp;
      filler_total += fillers_in_warp;
    }
    bases[0] = hole_total == 0 ? 0 : atomicAdd(&summary->hole_count, hole_total);
    bases[1] = filler_total == 0 ? 0 : atomicAdd(&summary->filler_count, filler_total);
  }
  __syncthreads();
  const auto removed = particle_count - retained_count;
  const auto capacity = retained_count < removed ? retained_count : removed;
  if (hole) {
    const auto destination = bases[0] + hole_prefixes[warp] + hole_rank;
    if (destination < capacity) holes[destination] = source;
  }
  if (filler) {
    const auto destination = bases[1] + filler_prefixes[warp] + filler_rank;
    if (destination < capacity) fillers[destination] = source;
  }
}

template <class LaunchPolicy, class Mover>
__global__ __launch_bounds__(LaunchPolicy::block_threads)
void move_device_unordered_hole_fill_records_kernel(
    const std::uint32_t maximum_moves, const std::uint32_t *const holes,
    const std::uint32_t *const fillers, DeviceUnstableHoleFillSummary *const summary,
    Mover mover) {
  const auto move = static_cast<std::uint32_t>(blockIdx.x) *
                        LaunchPolicy::block_threads + threadIdx.x;
  // The preceding kernel completed on this stream. Every block checks those
  // immutable counts itself; none depends on another block publishing status.
  const auto hole_count = summary->hole_count;
  const auto filler_count = summary->filler_count;
  const bool valid = hole_count == filler_count && hole_count <= maximum_moves;
  if (move == 0) {
    summary->move_count = valid ? hole_count : 0;
    summary->failure = static_cast<std::uint32_t>(
        valid ? DeviceUnstableHoleFillFailureWord::none :
                DeviceUnstableHoleFillFailureWord::invariant_mismatch);
  }
  if (valid && move < hole_count)
    mover(holes[move], fillers[move]);
}

template <device::DeviceKernelLaunchBackend Backend>
struct CudaHipUnstableParticleCompactionLaunch {
  template <class LaunchPolicy, class RetainedClassification,
            ParticleSchemaType Schema, class ClassificationReader>
  [[nodiscard]] static typename device::BackendTraits<Backend>::error_type
  enqueue(
      const std::uint32_t particle_count,
      const std::uint32_t expected_retained_count,
      const ClassificationReader classification, std::uint32_t *const holes,
      std::uint32_t *const fillers,
      DeviceUnstableHoleFillBlockPrefix *const block_prefixes,
      DeviceUnstableHoleFillSummary *const summary,
      const BasicParticleView<Schema, DeviceSpace<Backend>, false> particles,
      const typename device::BackendTraits<Backend>::stream_type stream,
      bool &accepted) noexcept {
    using Traits = device::BackendTraits<Backend>;
    constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
    const std::uint32_t block_count =
        particle_count / block_threads +
        static_cast<std::uint32_t>(particle_count % block_threads != 0);
    const std::uint32_t discarded_count =
        particle_count - expected_retained_count;
    const std::uint32_t maximum_moves =
        expected_retained_count < discarded_count ? expected_retained_count
                                                   : discarded_count;
    const std::uint32_t move_block_count =
        maximum_moves / block_threads +
        static_cast<std::uint32_t>(maximum_moves % block_threads != 0);

    // Reject and consume a stale launch error before this operation enqueues
    // work, then consume the slot after every concrete launch.
    typename Traits::error_type error = Traits::take_last_launch_error();
    if (!Traits::succeeded(error)) {
      return error;
    }

    if constexpr (LaunchPolicy::unordered_lists) {
      error = Traits::memset_device_async(summary, 0, sizeof(*summary), stream);
      if (!Traits::succeeded(error)) return error;
      accepted = true;
      // These exact counts were validated by migration's classification
      // admission. Neither case requires a list or any record movement.
      if (maximum_moves == 0) return Traits::success_error;
      build_device_unordered_hole_fill_lists_kernel<LaunchPolicy, RetainedClassification>
          <<<block_count, block_threads, 0, stream>>>(
              particle_count, expected_retained_count, classification, holes,
              fillers, summary);
      error = Traits::take_last_launch_error();
      if (!Traits::succeeded(error)) return error;
      move_device_unordered_hole_fill_records_kernel<LaunchPolicy>
          <<<move_block_count, block_threads, 0, stream>>>(
              maximum_moves, holes, fillers, summary,
              make_device_particle_record_mover(particles));
      return Traits::take_last_launch_error();
    } else {

    count_device_unstable_hole_fill_blocks_kernel<LaunchPolicy,
                                                  RetainedClassification>
        <<<block_count, block_threads, 0, stream>>>(
            particle_count, expected_retained_count, classification,
            block_prefixes);
    accepted = true;
    error = Traits::take_last_launch_error();
    if (!Traits::succeeded(error)) {
      return error;
    }
    scan_device_unstable_hole_fill_block_prefixes_kernel<LaunchPolicy>
        <<<1, block_threads, 0, stream>>>(block_count, maximum_moves,
                                          block_prefixes, summary);
    error = Traits::take_last_launch_error();
    if (!Traits::succeeded(error)) {
      return error;
    }

    // Count/scan still validates inconsistent endpoint counts. Scatter has no
    // writes on success and returns immediately on failure when no move can
    // exist, so it is omitted with its second classification read.
    if (maximum_moves == 0) return Traits::success_error;

    scatter_device_unstable_hole_fill_lists_kernel<LaunchPolicy,
                                                   RetainedClassification>
        <<<block_count, block_threads, 0, stream>>>(
            particle_count, expected_retained_count, classification,
            block_prefixes, holes, fillers, summary);
    error = Traits::take_last_launch_error();
    if (!Traits::succeeded(error)) {
      return error;
    }

    if (move_block_count != 0) {
      move_device_unstable_hole_fill_records_kernel<LaunchPolicy>
          <<<move_block_count, block_threads, 0, stream>>>(
              holes, fillers, summary,
              make_device_particle_record_mover(particles));
      return Traits::take_last_launch_error();
    }
    return Traits::success_error;
    }
  }
};

#if defined(__CUDACC__)
template <>
struct DeviceUnstableParticleCompactionLaunch<device::CudaBackend> final
    : CudaHipUnstableParticleCompactionLaunch<device::CudaBackend> {};
#endif

#if defined(__HIPCC__)
template <>
struct DeviceUnstableParticleCompactionLaunch<device::HipBackend> final
    : CudaHipUnstableParticleCompactionLaunch<device::HipBackend> {};
#endif

} // namespace lpfx::detail

#endif

#endif
