#ifndef LPFX_BACKEND_CUDA_HIP_SUBGROUP_HPP
#define LPFX_BACKEND_CUDA_HIP_SUBGROUP_HPP

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/core/attributes.hpp>

namespace lpfx::device::subgroup {

// Public warp vocabulary. HIP masks are always 64-bit, including on wave32
// targets. The fixed-width type prevents truncation while CUDA conversion is
// eliminated after inlining.
using warp_mask_type = std::uint64_t;

inline constexpr std::uint32_t maximum_warp_size = 64;

// Pass a mask through this tag when its value is known at device compile time.
// Reductions can then remove partial-tile bookkeeping entirely.
template <warp_mask_type Participants>
using warp_mask_constant = std::integral_constant<warp_mask_type, Participants>;

template <class T>
concept warp_shuffle_scalar =
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, long> || std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

namespace detail {

// A concrete CUDA/HIP device image has a compile-time physical warp size.
// ROCm 6.4 stopped exposing warpSize as a compile-time constant, so use AMD's
// documented target-family selector instead. HIP does not support changing the
// wavefront size through -mwavefrontsize64: gfx8/gfx9 are wave64 and newer
// targets are wave32. HIP-on-NVIDIA and native CUDA are always warp32.
#if defined(__CUDA_ARCH__) ||                                                  \
    (defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_NVIDIA__))
inline constexpr std::uint32_t native_warp_size = 32;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
#if defined(__GFX8__) || defined(__GFX9__)
inline constexpr std::uint32_t native_warp_size = 64;
#else
inline constexpr std::uint32_t native_warp_size = 32;
#endif
#else
// Non-device parsing fallback. Host code obtains the active device's value
// from DeviceProperties::subgroup_width instead of this device-only API.
inline constexpr std::uint32_t native_warp_size = 1;
#endif

template <std::uint32_t RequestedWidth>
inline constexpr std::uint32_t resolved_logical_width =
    RequestedWidth == 0 ? native_warp_size : RequestedWidth;

[[nodiscard]] LPFX_HOST_DEVICE constexpr warp_mask_type
low_lane_mask(const std::uint32_t lane_count) noexcept {
  return lane_count == 0
             ? warp_mask_type{0}
             : (lane_count >= maximum_warp_size
                    ? ~warp_mask_type{0}
                    : (warp_mask_type{1} << lane_count) - warp_mask_type{1});
}

inline constexpr warp_mask_type full_mask_value =
    low_lane_mask(native_warp_size);

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr warp_mask_type
full_mask() noexcept {
  return full_mask_value;
}

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr warp_mask_type
native_mask(const warp_mask_type mask) noexcept {
  if constexpr (native_warp_size <= 32U) {
    return static_cast<warp_mask_type>(static_cast<std::uint32_t>(mask));
  } else {
    return mask;
  }
}

template <std::uint32_t LogicalWidth>
LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr void
validate_logical_width() noexcept {
  // Not std::has_single_bit: it is a host function to nvcc.
  static_assert(LogicalWidth != 0 && (LogicalWidth & (LogicalWidth - 1U)) == 0,
                "a logical warp size must be a non-zero power of two");
  static_assert(LogicalWidth <= maximum_warp_size,
                "a logical warp cannot exceed LPFx's maximum warp size");
#if LPFX_DEVICE_PASS
  static_assert(LogicalWidth <= native_warp_size,
                "a logical warp cannot exceed the target's native warp size");
#endif
}

template <warp_mask_type Participants>
LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr void
validate_constant_participants() noexcept {
  static_assert(Participants != 0, "a warp participant mask cannot be empty");
#if LPFX_DEVICE_PASS
  static_assert((Participants & ~full_mask_value) == 0,
                "a warp participant mask names a lane outside the target's "
                "native warp");
#endif
}

template <std::uint32_t LogicalWidth, warp_mask_type Participants>
inline constexpr bool contains_only_complete_logical_warps = [] {
  constexpr warp_mask_type logical_mask = low_lane_mask(LogicalWidth);
  for (std::uint32_t base = 0; base < maximum_warp_size; base += LogicalWidth) {
    const warp_mask_type tile = (Participants >> base) & logical_mask;
    if (tile != 0 && tile != logical_mask) {
      return false;
    }
  }
  return true;
}();

template <class Mask> inline constexpr bool is_mask_constant = false;
template <warp_mask_type Participants>
inline constexpr bool is_mask_constant<warp_mask_constant<Participants>> = true;

// Exact one-dimensional, power-of-two block sizes of at most 1024 threads.
template <std::uint32_t BlockThreads>
concept static_block_threads =
    BlockThreads <= 1024 && std::has_single_bit(BlockThreads);

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE std::uint32_t
block_thread_rank() noexcept {
#if LPFX_DEVICE_PASS
  return static_cast<std::uint32_t>(threadIdx.x) +
         static_cast<std::uint32_t>(blockDim.x) *
             (static_cast<std::uint32_t>(threadIdx.y) +
              static_cast<std::uint32_t>(blockDim.y) *
                  static_cast<std::uint32_t>(threadIdx.z));
#else
  return 0;
#endif
}

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE std::uint32_t
block_thread_count() noexcept {
#if LPFX_DEVICE_PASS
  return static_cast<std::uint32_t>(blockDim.x) *
         static_cast<std::uint32_t>(blockDim.y) *
         static_cast<std::uint32_t>(blockDim.z);
#else
  return 1;
#endif
}

#if defined(__HIP_DEVICE_COMPILE__) &&                                         \
    !defined(HIP_DISABLE_WARP_SYNC_BUILTINS) &&                                \
    (defined(__HIP_PLATFORM_NVIDIA__) ||                                       \
     (defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7) ||                 \
     (defined(HIP_ENABLE_WARP_SYNC_BUILTINS) && defined(HIP_VERSION_MAJOR) &&  \
      HIP_VERSION_MAJOR == 6 && defined(HIP_VERSION_MINOR) &&                  \
      HIP_VERSION_MINOR >= 2))
#define LPFX_DETAIL_HIP_SYNC_CROSS_LANE 1
#else
#define LPFX_DETAIL_HIP_SYNC_CROSS_LANE 0
#endif

#if defined(__CUDA_ARCH__) || LPFX_DETAIL_HIP_SYNC_CROSS_LANE
#define LPFX_DETAIL_SYNC_CROSS_LANE 1
#else
#define LPFX_DETAIL_SYNC_CROSS_LANE 0
#endif

#if defined(__CUDA_ARCH__) ||                                                  \
    (defined(__HIP_DEVICE_COMPILE__) &&                                        \
     !defined(HIP_DISABLE_WARP_SYNC_BUILTINS) &&                               \
     (defined(__HIP_PLATFORM_NVIDIA__) ||                                      \
      (defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7)))
#define LPFX_DETAIL_SYNC_WARP 1
#else
#define LPFX_DETAIL_SYNC_WARP 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define LPFX_DETAIL_NATIVE_WARP_REDUCE 1
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) &&      \
    !defined(HIP_DISABLE_WARP_SYNC_BUILTINS) && defined(HIP_VERSION_MAJOR) &&  \
    HIP_VERSION_MAJOR >= 7
#define LPFX_DETAIL_NATIVE_WARP_REDUCE 1
#else
#define LPFX_DETAIL_NATIVE_WARP_REDUCE 0
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) &&        \
    defined(HIP_ENABLE_EXTRA_WARP_SYNC_TYPES)
template <class T>
concept native_reduce_scalar =
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
    std::is_same_v<T, long long> || std::is_same_v<T, unsigned long long> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;
#else
template <class T>
concept native_reduce_scalar =
    std::is_same_v<T, int> || std::is_same_v<T, unsigned int>;
#endif

// The participant-mask argument type of the native *_sync intrinsics.
#if defined(__CUDA_ARCH__)
using sync_mask_type = unsigned int;
#else
using sync_mask_type = unsigned long long;
#endif

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr sync_mask_type
sync_mask(const warp_mask_type participants) noexcept {
  return static_cast<sync_mask_type>(participants);
}

} // namespace detail

// A runtime participant mask or a warp_mask_constant.
template <class Mask>
concept warp_participants = detail::is_mask_constant<Mask> ||
                            std::convertible_to<Mask, warp_mask_type>;

// The complete physical warp; requires converged callers.
using full_warp_mask = warp_mask_constant<detail::full_mask_value>;

// --------------------------------------------------------------------------
// Warp primitives
// --------------------------------------------------------------------------

// This is a C++ constant expression in every concrete CUDA/HIP device image.
// Host dispatch code must query DeviceProperties::subgroup_width instead.
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint32_t
warp_size() noexcept {
  return detail::native_warp_size;
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr warp_mask_type
warp_low_lane_mask(const std::uint32_t lane_count) noexcept {
  return detail::low_lane_mask(lane_count);
}

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE std::uint32_t
warp_lane() noexcept {
#if defined(__CUDA_ARCH__)
  std::uint32_t result = 0;
  asm("mov.u32 %0, %%laneid;" : "=r"(result));
  return result;
#elif defined(__HIP_DEVICE_COMPILE__)
  return static_cast<std::uint32_t>(__lane_id());
#else
  return 0;
#endif
}

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE warp_mask_type
warp_lane_bit() noexcept {
  if constexpr (detail::native_warp_size <= 32U) {
    // Keep a wave32 shift scalar and zero-extend once.
    return static_cast<warp_mask_type>(1U << warp_lane());
  } else {
    return warp_mask_type{1} << warp_lane();
  }
}

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE warp_mask_type
warp_active_mask() noexcept {
#if defined(__CUDA_ARCH__)
  return static_cast<warp_mask_type>(__activemask());
#elif defined(__HIP_DEVICE_COMPILE__)
  return static_cast<warp_mask_type>(__ballot(1));
#else
  return warp_mask_type{1};
#endif
}

[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE std::uint32_t
warp_population_count(const warp_mask_type mask) noexcept {
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint32_t>(__popc(static_cast<unsigned int>(mask)));
#elif defined(__HIP_DEVICE_COMPILE__)
  if constexpr (detail::native_warp_size <= 32U) {
    return static_cast<std::uint32_t>(__popc(static_cast<unsigned int>(mask)));
  } else {
    return static_cast<std::uint32_t>(
        __popcll(static_cast<unsigned long long>(mask)));
  }
#else
  warp_mask_type remaining = mask;
  std::uint32_t count = 0;
  while (remaining != 0) {
    remaining &= remaining - warp_mask_type{1};
    ++count;
  }
  return count;
#endif
}

// Return the least-significant set lane, or -1 for an empty mask.
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE std::int32_t
warp_first_lane(const warp_mask_type mask) noexcept {
#if defined(__CUDA_ARCH__)
  return static_cast<std::int32_t>(__ffs(static_cast<unsigned int>(mask))) - 1;
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
  // HIP masks are 64-bit even on wave32. ROCm 6.4's shipped headers expose
  // this 64-bit last-bit entry point, but not the documented u32 spelling.
  return static_cast<std::int32_t>(
      __lastbit_u32_u64(static_cast<unsigned long long>(mask)));
#elif defined(__HIP_DEVICE_COMPILE__)
  return static_cast<std::int32_t>(__ffs(static_cast<unsigned int>(mask))) - 1;
#else
  if (mask == 0) {
    return -1;
  }
  std::int32_t result = 0;
  warp_mask_type remaining = mask;
  while ((remaining & warp_mask_type{1}) == 0) {
    remaining >>= 1U;
    ++result;
  }
  return result;
#endif
}

// Every lane named in participants must reach this call with the same mask.
// Calling lanes must be named in participants; lanes outside the mask do not
// call it. CUDA and ROCm 7+ use the synchronized native primitive. Legacy AMD
// HIP uses a wave barrier and therefore requires already-converged callers.
LPFX_DEVICE LPFX_ALWAYS_INLINE void
warp_synchronize(const warp_mask_type participants) noexcept {
#if LPFX_DETAIL_SYNC_WARP
  __syncwarp(detail::sync_mask(participants));
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
  static_cast<void>(participants);
  __threadfence_block();
  __builtin_amdgcn_wave_barrier();
  __threadfence_block();
#else
  static_cast<void>(participants);
#endif
}

// The no-mask overload requires a converged, complete physical warp.
LPFX_DEVICE LPFX_ALWAYS_INLINE void warp_synchronize() noexcept {
  warp_synchronize(detail::full_mask());
}

// Calling lanes must be named in participants, and every named lane must call
// with the same mask. Legacy AMD HIP forms the execution mask through ordinary
// converged/divergent control flow and intersects its native 64-bit ballot.
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE warp_mask_type
warp_ballot(const warp_mask_type participants, const bool predicate) noexcept {
#if defined(__CUDA_ARCH__)
  return static_cast<warp_mask_type>(
      __ballot_sync(detail::sync_mask(participants), static_cast<int>(predicate)));
#elif LPFX_DETAIL_HIP_SYNC_CROSS_LANE
  return static_cast<warp_mask_type>(__ballot_sync(
             detail::sync_mask(participants), static_cast<int>(predicate))) &
         participants;
#elif defined(__HIP_DEVICE_COMPILE__)
  return static_cast<warp_mask_type>(__ballot(static_cast<int>(predicate))) &
         participants;
#else
  return predicate ? participants & warp_mask_type{1} : warp_mask_type{0};
#endif
}

// The no-mask overload requires a converged, complete physical warp.
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE warp_mask_type
warp_ballot(const bool predicate) noexcept {
  return warp_ballot(detail::full_mask(), predicate);
}

// LogicalWidth is checked at device compile time and must be a power of two no
// greater than the native width. Omit it for the native size, or write (for
// example) warp_shuffle<4>(...) for a statically tiled warp. The source and
// every calling lane must be named in participants, and named lanes must use
// the same mask. Overloads without participants require a converged, complete
// physical warp; partial or divergent callers pass warp_active_mask() or an
// explicitly constructed participant mask.
template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle(const warp_mask_type participants, const T value,
             const std::uint32_t source_lane) noexcept {
  constexpr std::uint32_t width = detail::resolved_logical_width<LogicalWidth>;
  detail::validate_logical_width<width>();
#if LPFX_DETAIL_SYNC_CROSS_LANE
  return __shfl_sync(detail::sync_mask(participants), value,
                     static_cast<int>(source_lane), static_cast<int>(width));
#elif defined(__HIP_DEVICE_COMPILE__)
  static_cast<void>(participants);
  return __shfl(value, static_cast<int>(source_lane), static_cast<int>(width));
#else
  static_cast<void>(participants);
  static_cast<void>(source_lane);
  return value;
#endif
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle(const T value, const std::uint32_t source_lane) noexcept {
  return warp_shuffle<LogicalWidth>(detail::full_mask(), value, source_lane);
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle_up(const warp_mask_type participants, const T value,
                const std::uint32_t delta) noexcept {
  constexpr std::uint32_t width = detail::resolved_logical_width<LogicalWidth>;
  detail::validate_logical_width<width>();
#if LPFX_DETAIL_SYNC_CROSS_LANE
  return __shfl_up_sync(detail::sync_mask(participants), value, delta,
                        static_cast<int>(width));
#elif defined(__HIP_DEVICE_COMPILE__)
  static_cast<void>(participants);
  return __shfl_up(value, delta, static_cast<int>(width));
#else
  static_cast<void>(participants);
  static_cast<void>(delta);
  return value;
#endif
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle_up(const T value, const std::uint32_t delta) noexcept {
  return warp_shuffle_up<LogicalWidth>(detail::full_mask(), value, delta);
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle_down(const warp_mask_type participants, const T value,
                  const std::uint32_t delta) noexcept {
  constexpr std::uint32_t width = detail::resolved_logical_width<LogicalWidth>;
  detail::validate_logical_width<width>();
#if LPFX_DETAIL_SYNC_CROSS_LANE
  return __shfl_down_sync(detail::sync_mask(participants), value, delta,
                          static_cast<int>(width));
#elif defined(__HIP_DEVICE_COMPILE__)
  static_cast<void>(participants);
  return __shfl_down(value, delta, static_cast<int>(width));
#else
  static_cast<void>(participants);
  static_cast<void>(delta);
  return value;
#endif
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle_down(const T value, const std::uint32_t delta) noexcept {
  return warp_shuffle_down<LogicalWidth>(detail::full_mask(), value, delta);
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle_xor(const warp_mask_type participants, const T value,
                 const std::uint32_t lane_mask) noexcept {
  constexpr std::uint32_t width = detail::resolved_logical_width<LogicalWidth>;
  detail::validate_logical_width<width>();
#if LPFX_DETAIL_SYNC_CROSS_LANE
  return __shfl_xor_sync(detail::sync_mask(participants), value,
                         static_cast<int>(lane_mask), static_cast<int>(width));
#elif defined(__HIP_DEVICE_COMPILE__)
  static_cast<void>(participants);
  return __shfl_xor(value, static_cast<int>(lane_mask),
                    static_cast<int>(width));
#else
  static_cast<void>(participants);
  static_cast<void>(lane_mask);
  return value;
#endif
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_shuffle_xor(const T value, const std::uint32_t lane_mask) noexcept {
  return warp_shuffle_xor<LogicalWidth>(detail::full_mask(), value, lane_mask);
}

// --------------------------------------------------------------------------
// Reduction machinery
// --------------------------------------------------------------------------
namespace detail {

struct sum final {
  template <class T>
  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr T
  operator()(const T left, const T right) const noexcept {
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      // Every shuffle lane evaluates the tree, including lanes whose result is
      // not returned. Unsigned arithmetic gives signed reductions defined
      // modulo 2^N without adding instructions on CUDA or HIP.
      using unsigned_type = std::make_unsigned_t<T>;
      return static_cast<T>(static_cast<unsigned_type>(left) +
                            static_cast<unsigned_type>(right));
    } else {
      return left + right;
    }
  }
};

struct minimum final {
  template <class T>
  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr T
  operator()(const T left, const T right) const noexcept {
    return right < left ? right : left;
  }
};

struct maximum final {
  template <class T>
  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr T
  operator()(const T left, const T right) const noexcept {
    return left < right ? right : left;
  }
};

template <std::uint32_t Width, class Operation, class T>
inline constexpr bool uses_native_reduce =
    LPFX_DETAIL_NATIVE_WARP_REDUCE && Width == native_warp_size &&
    native_reduce_scalar<T> &&
    (std::is_same_v<Operation, sum> || std::is_same_v<Operation, minimum> ||
     std::is_same_v<Operation, maximum>);

// Called only when uses_native_reduce holds.
template <class Operation, class T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
native_reduce(const T value, const warp_mask_type participants) noexcept {
#if LPFX_DETAIL_NATIVE_WARP_REDUCE
  if constexpr (std::is_same_v<Operation, sum>) {
    return __reduce_add_sync(sync_mask(participants), value);
  } else if constexpr (std::is_same_v<Operation, minimum>) {
    return __reduce_min_sync(sync_mask(participants), value);
  } else {
    return __reduce_max_sync(sync_mask(participants), value);
  }
#else
  static_cast<void>(participants);
  return value;
#endif
}

enum class Exchange { down, bounded_down, butterfly };

// One stage of a Width-lane tree: combine with the lane Offset away. A bounded
// stage ignores source lanes past the tile's active run.
template <std::uint32_t Width, Exchange Kind, class Operation>
struct HalvingStep final {
  warp_mask_type participants;
  Operation operation;
  std::uint32_t logical_lane;
  std::uint32_t active_count;

  template <warp_shuffle_scalar T, std::uint32_t Offset>
  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T operator()(
      const T value, std::integral_constant<std::uint32_t, Offset>) const noexcept {
    if constexpr (Kind == Exchange::butterfly) {
      return operation(value,
                       warp_shuffle_xor<Width>(participants, value, Offset));
    } else {
      const T other = warp_shuffle_down<Width>(participants, value, Offset);
      if constexpr (Kind == Exchange::bounded_down) {
        return logical_lane + Offset < active_count ? operation(value, other)
                                                    : value;
      } else {
        return operation(value, other);
      }
    }
  }
};

template <std::uint32_t Width, warp_shuffle_scalar T, class Step,
          std::size_t... Stage>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
for_each_halving(T value, const Step &step,
                 std::index_sequence<Stage...>) noexcept {
  ((value = step(value,
                 std::integral_constant<std::uint32_t, (Width >> (Stage + 1U))>{})),
   ...);
  return value;
}

template <std::uint32_t Width>
inline constexpr std::size_t halving_stage_count =
    static_cast<std::size_t>(std::bit_width(Width)) - 1U;

// Applies step at offsets Width / 2, Width / 4, ..., 1 as a compile-time
// unrolled fold.
template <std::uint32_t Width, warp_shuffle_scalar T, class Step>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
for_each_halving(const T value, const Step &step) noexcept {
  return for_each_halving<Width>(
      value, step, std::make_index_sequence<halving_stage_count<Width>>{});
}

// Complete tiles. A shuffle-down tree is valid in each tile leader; a
// broadcast (butterfly) tree is valid in every lane.
template <std::uint32_t Width, bool Broadcast, warp_shuffle_scalar T,
          class Operation>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
reduce_tree(const T value, const warp_mask_type participants,
            const Operation operation) noexcept {
  constexpr Exchange kind = Broadcast ? Exchange::butterfly : Exchange::down;
  return for_each_halving<Width>(
      value, HalvingStep<Width, kind, Operation>{participants, operation, 0U, 0U});
}

// Partial tiles: each tile's named lanes form one contiguous run whose lowest
// lane is the leader.
template <std::uint32_t Width, bool Broadcast, warp_shuffle_scalar T,
          class Operation>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
reduce_partial(const T value, const warp_mask_type participants,
               const Operation operation) noexcept {
  const std::uint32_t physical_lane = warp_lane();
  warp_mask_type tile_mask = participants;
  std::uint32_t tile_base = 0;
  if constexpr (Width != native_warp_size) {
    tile_base = physical_lane & ~(Width - 1U);
    tile_mask &= low_lane_mask(Width) << tile_base;
  }
  const std::uint32_t leader =
      static_cast<std::uint32_t>(warp_first_lane(tile_mask));
  const T reduced = for_each_halving<Width>(
      value, HalvingStep<Width, Exchange::bounded_down, Operation>{
                 participants, operation, physical_lane - leader,
                 warp_population_count(tile_mask)});
  if constexpr (Broadcast) {
    return warp_shuffle<Width>(participants, reduced, leader - tile_base);
  } else {
    return reduced;
  }
}

template <std::uint32_t Width, bool Broadcast, warp_shuffle_scalar T,
          warp_participants Mask, class Operation>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
reduce(const T value, const Mask participants,
       const Operation operation) noexcept {
  validate_logical_width<Width>();
  if constexpr (is_mask_constant<Mask>) {
    validate_constant_participants<Mask::value>();
    constexpr warp_mask_type native = native_mask(Mask::value);
    if constexpr (uses_native_reduce<Width, Operation, T>) {
      return native_reduce<Operation>(value, native);
    } else if constexpr (contains_only_complete_logical_warps<Width, native>) {
      return reduce_tree<Width, Broadcast>(value, native, operation);
    } else {
      return reduce_partial<Width, Broadcast>(value, native, operation);
    }
  } else {
    const warp_mask_type native =
        native_mask(static_cast<warp_mask_type>(participants));
    if constexpr (uses_native_reduce<Width, Operation, T>) {
      return native_reduce<Operation>(value, native);
    } else {
      // The full-warp test stays uniform across all named lanes.
      if (native == full_mask()) {
        return reduce_tree<Width, Broadcast>(value, native, operation);
      }
      return reduce_partial<Width, Broadcast>(value, native, operation);
    }
  }
}

template <std::uint32_t Width, warp_shuffle_scalar T, class Operation>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
block_warp_reduce(const T value, const Operation operation) noexcept {
  constexpr warp_mask_type participants = low_lane_mask(Width);
  if constexpr (std::is_same_v<Operation, sum>) {
    return reduce<Width, false>(value, warp_mask_constant<participants>{},
                                operation);
  } else {
    return reduce_tree<Width, false>(value, participants, operation);
  }
}

// Reproduce the identity-padded tree without shuffling values known to be
// padding. Keeping the combines preserves signed zero and arbitrary pure
// operations; simply omitting identity lanes would change their results.
template <std::uint32_t Width, std::uint32_t ValueCount, warp_shuffle_scalar T,
          class Operation>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
block_warp_padding(T value, T padding, const Operation operation) noexcept {
#if LPFX_DEVICE_PASS
#pragma unroll
#endif
  for (std::uint32_t offset = Width / 2U; offset >= ValueCount; offset /= 2U) {
    value = operation(value, padding);
    padding = operation(padding, padding);
  }
  return value;
}

} // namespace detail

#undef LPFX_DETAIL_NATIVE_WARP_REDUCE
#undef LPFX_DETAIL_SYNC_WARP
#undef LPFX_DETAIL_SYNC_CROSS_LANE
#undef LPFX_DETAIL_HIP_SYNC_CROSS_LANE

// --------------------------------------------------------------------------
// Warp and block reductions
// --------------------------------------------------------------------------

// Each tile's named lanes must form one contiguous run. The reduced value is
// valid in that run's lowest lane (every named lane for warp_all_reduce_sum).
// A compile-time LogicalWidth removes every unused shuffle stage; omitting it
// selects the target's native size. Omitting participants is the converged
// full-warp fast path; pass an explicit mask for partial or divergent
// execution. When a fixed mask contains complete logical tiles, pass
// warp_mask_constant<mask>{}; the compiler then selects the same unchecked
// tree as hand-written shuffles without runtime bookkeeping.
template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T,
          warp_participants Mask = full_warp_mask>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_reduce_sum(const T value, const Mask participants = {}) noexcept {
  return detail::reduce<detail::resolved_logical_width<LogicalWidth>, false>(
      value, participants, detail::sum{});
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T,
          warp_participants Mask = full_warp_mask>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_reduce_min(const T value, const Mask participants = {}) noexcept {
  return detail::reduce<detail::resolved_logical_width<LogicalWidth>, false>(
      value, participants, detail::minimum{});
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T,
          warp_participants Mask = full_warp_mask>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_reduce_max(const T value, const Mask participants = {}) noexcept {
  return detail::reduce<detail::resolved_logical_width<LogicalWidth>, false>(
      value, participants, detail::maximum{});
}

template <std::uint32_t LogicalWidth = 0, warp_shuffle_scalar T,
          warp_participants Mask = full_warp_mask>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
warp_all_reduce_sum(const T value, const Mask participants = {}) noexcept {
  return detail::reduce<detail::resolved_logical_width<LogicalWidth>, true>(
      value, participants, detail::sum{});
}

// Device-image constant for sizing scratch in a statically sized block. Like
// warp_size(), this is not a host-side device-property query.
template <std::uint32_t BlockThreads>
  requires detail::static_block_threads<BlockThreads>
inline constexpr std::uint32_t block_warp_count =
    (BlockThreads + detail::native_warp_size - 1U) / detail::native_warp_size;

// Exact one-dimensional, power-of-two block size; every block thread enters
// converged. Scratch holds block_warp_count<BlockThreads> elements. The result
// is valid only in thread zero; synchronize before reusing scratch. Operation
// is a pure combine defined for all intermediate lane values, including lanes
// whose results are discarded. Identity pads the final warp reduction. A
// single-warp block stays in registers; larger blocks exchange one value per
// warp through scratch and use one block barrier.
template <std::uint32_t BlockThreads, warp_shuffle_scalar T, class Operation>
  requires detail::static_block_threads<BlockThreads>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
block_reduce(T value, T *LPFX_RESTRICT const scratch, const T identity,
             const Operation operation) noexcept {
#if LPFX_DEVICE_PASS
  constexpr std::uint32_t native_width = detail::native_warp_size;
  constexpr std::uint32_t width =
      BlockThreads < native_width ? BlockThreads : native_width;
  constexpr std::uint32_t warp_count = block_warp_count<BlockThreads>;
  const std::uint32_t thread = static_cast<std::uint32_t>(threadIdx.x);
  const std::uint32_t warp = thread / native_width;
  value = detail::block_warp_reduce<width>(value, operation);
  if constexpr (warp_count == 1U) {
    // Only thread zero is observable. Its final tree consists entirely of
    // identity padding, so neither shared memory nor another shuffle is needed.
    return detail::block_warp_padding<width, 1U>(value, identity, operation);
  }
  if (warp_lane() == 0U) {
    scratch[warp] = value;
  }
  __syncthreads();

  if (warp == 0U) {
    value = thread < warp_count ? scratch[thread] : identity;
    if constexpr (std::is_same_v<Operation, detail::sum> &&
                  (std::is_integral_v<T> ||
                   detail::uses_native_reduce<native_width, Operation, T>)) {
      // Keep hardware reduction instructions available on targets
      // that provide them; narrowing the logical width would disable them.
      value = detail::block_warp_reduce<width>(value, operation);
    } else {
      value = detail::block_warp_padding<width, warp_count>(value, identity,
                                                          operation);
      // Every lane in the first physical warp participates. Logical tiles
      // remove unnecessary stages; only the first tile's result is observable.
      value = detail::reduce_tree<warp_count, false>(
          value, detail::low_lane_mask(width), operation);
    }
  }
#else
  static_cast<void>(scratch);
  static_cast<void>(identity);
  static_cast<void>(operation);
#endif
  return value;
}

template <std::uint32_t BlockThreads, warp_shuffle_scalar T>
  requires detail::static_block_threads<BlockThreads>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
block_reduce_sum(const T value, T *LPFX_RESTRICT const scratch) noexcept {
  return block_reduce<BlockThreads>(value, scratch, T{}, detail::sum{});
}

// Sums each component of values over one block around a single barrier: the
// steps of block_reduce_sum run for every component, so each sum is bitwise
// that of block_reduce_sum on its component. The contract is
// block_reduce_sum's; the sums replace values in thread zero only.
template <std::uint32_t BlockThreads, warp_shuffle_scalar T,
          std::size_t Components>
  requires detail::static_block_threads<BlockThreads>
LPFX_DEVICE LPFX_ALWAYS_INLINE void
block_reduce_sum(T (&values)[Components],
                 T (&scratch)[Components][block_warp_count<BlockThreads>]) noexcept {
#if LPFX_DEVICE_PASS
  constexpr std::uint32_t native_width = detail::native_warp_size;
  constexpr std::uint32_t width =
      BlockThreads < native_width ? BlockThreads : native_width;
  constexpr std::uint32_t warp_count = block_warp_count<BlockThreads>;
  const std::uint32_t thread = static_cast<std::uint32_t>(threadIdx.x);
  const std::uint32_t warp = thread / native_width;
#pragma unroll
  for (std::size_t component = 0; component < Components; ++component) {
    values[component] =
        detail::block_warp_reduce<width>(values[component], detail::sum{});
    if constexpr (warp_count == 1U) {
      values[component] = detail::block_warp_padding<width, 1U>(
          values[component], T{}, detail::sum{});
    } else if (warp_lane() == 0U) {
      scratch[component][warp] = values[component];
    }
  }
  if constexpr (warp_count > 1U) {
    __syncthreads();
    if (warp != 0U) {
      return;
    }
#pragma unroll
    for (std::size_t component = 0; component < Components; ++component) {
      T value = thread < warp_count ? scratch[component][thread] : T{};
      if constexpr (std::is_integral_v<T> ||
                    detail::uses_native_reduce<native_width, detail::sum, T>) {
        value = detail::block_warp_reduce<width>(value, detail::sum{});
      } else {
        value = detail::block_warp_padding<width, warp_count>(value, T{},
                                                            detail::sum{});
        value = detail::reduce_tree<warp_count, false>(
            value, detail::low_lane_mask(width), detail::sum{});
      }
      values[component] = value;
    }
  }
#else
  static_cast<void>(values);
  static_cast<void>(scratch);
#endif
}

// scratch must provide at least ceil(block-thread-count / warp_size())
// elements; maximum_warp_size / 2 elements always suffice. Every live block
// thread must enter once in converged block control flow. The result is valid
// only in block thread zero. Callers need a block barrier before reusing
// scratch; omitting the trailing barrier keeps the terminal path at one sync.
template <warp_shuffle_scalar T>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T
block_reduce_sum(T value, T *LPFX_RESTRICT const scratch) noexcept {
#if LPFX_DEVICE_PASS
  constexpr std::uint32_t native_width = detail::native_warp_size;
  const std::uint32_t thread_rank = detail::block_thread_rank();
  const std::uint32_t thread_count = detail::block_thread_count();
  const std::uint32_t warp_index =
      native_width == 64U ? thread_rank >> 6U : thread_rank >> 5U;
  value = warp_reduce_sum<native_width>(value, warp_active_mask());
  if (warp_lane() == 0U) {
    scratch[warp_index] = value;
  }
  __syncthreads();

  if (warp_index == 0) {
    const std::uint32_t warp_count = native_width == 64U
                                         ? (thread_count + 63U) >> 6U
                                         : (thread_count + 31U) >> 5U;
    value = thread_rank < warp_count ? scratch[thread_rank] : T{};
    value = warp_reduce_sum<native_width>(value, warp_active_mask());
  }
#else
  static_cast<void>(scratch);
#endif
  return value;
}

} // namespace lpfx::device::subgroup

#endif
