#ifndef LPFX_ALGORITHM_PARTICLE_DEVICE_LAUNCH_HPP
#define LPFX_ALGORITHM_PARTICLE_DEVICE_LAUNCH_HPP

#include <bit>
#include <cstdint>

#include <lpfx/backend/traits.hpp>
#include <lpfx/core/attributes.hpp>

namespace lpfx {

// Kernel-facing records and launch seams of the device sort and compaction
// owners, shared with their CUDA/HIP leaves.

struct DeviceParticleSortSummary final {
  std::uint32_t failure{};
};

static_assert(sizeof(DeviceParticleSortSummary) == sizeof(std::uint32_t));

namespace detail {

enum class DeviceParticleSortFailureWord : std::uint32_t {
  none = 0,
  invalid_key = 1,
};

// Native launch syntax belongs to a device-language specialization with a
// static
//   enqueue<KeyAccessor, Schema, LaunchPolicy, HasUpstream>(particle_count,
//       bin_count, source, destination, counts, offsets, permutation,
//       summary, upstream_failure, scan_scratch, scan_workspace, forward,
//       stream, bool &accepted)
// The incomplete primary prevents a host fallback, registry, callback, or
// runtime backend branch from entering particle sorting.
template <class Backend> struct DeviceParticleSortLaunch;

template <class Backend>
concept DeviceParticleSortBackend =
    device::DeviceAllocationBackend<Backend> &&
    device::PinnedHostAllocationBackend<Backend> &&
    device::DeviceEventBackend<Backend> && device::DeviceCopyBackend<Backend>;

} // namespace detail

template <std::uint32_t BlockThreads, std::uint32_t CountSubBuckets,
          bool UnorderedLists = false>
  requires(BlockThreads >= 32 && BlockThreads <= 1024 &&
           std::has_single_bit(BlockThreads) && CountSubBuckets <= 32 &&
           std::has_single_bit(CountSubBuckets))
struct DeviceUnstableHoleFillLaunchPolicy final {
  static constexpr std::uint32_t block_threads = BlockThreads;
  static constexpr std::uint32_t count_sub_buckets = CountSubBuckets;
  static constexpr bool unordered_lists = UnorderedLists;
};

using DefaultDeviceUnstableHoleFillLaunchPolicy =
    DeviceUnstableHoleFillLaunchPolicy<256, 8>;

// Migration has already admitted exact classification counts. List ordering
// may follow block arrival order; all-retained/all-removed need no flag scan.
using DeviceMigrationHoleFillLaunchPolicy =
    DeviceUnstableHoleFillLaunchPolicy<256, 8, true>;

// One byte per record; 1 keeps the record.
struct CanonicalDeviceRetainedClassification final {
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr bool
  retained(const std::uint8_t value) noexcept {
    return value == std::uint8_t{1};
  }
};

struct DeviceUnstableHoleFillBlockPrefix final {
  std::uint32_t hole{};
  std::uint32_t filler{};
};

struct DeviceUnstableHoleFillSummary final {
  std::uint32_t hole_count{};
  std::uint32_t filler_count{};
  std::uint32_t move_count{};
  std::uint32_t failure{};
};

static_assert(sizeof(DeviceUnstableHoleFillBlockPrefix) == 8);
static_assert(sizeof(DeviceUnstableHoleFillSummary) == 16);

namespace detail {

enum class DeviceUnstableHoleFillFailureWord : std::uint32_t {
  none = 0,
  invariant_mismatch = 1,
};

// Kernel syntax belongs to a device-language leaf. A backend that supports
// particle compaction specializes this trait with a static
//   enqueue<LaunchPolicy, RetainedClassification, Schema>(particle_count,
//       expected_retained_count, classification, holes, fillers,
//       block_prefixes, summary, particles, stream, bool &accepted)
// whose writes, including invalid-classification paths, fit
// min(retained, removed) entries per list. The incomplete primary prevents a
// host fallback or a runtime backend branch.
template <class Backend> struct DeviceUnstableParticleCompactionLaunch;

template <class Backend>
concept DeviceParticleCompactionBackend =
    device::DeviceAllocationBackend<Backend> &&
    device::PinnedHostAllocationBackend<Backend> &&
    device::DeviceEventBackend<Backend> && device::DeviceCopyBackend<Backend>;

} // namespace detail

} // namespace lpfx

#endif
