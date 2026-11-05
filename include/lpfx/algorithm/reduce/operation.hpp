#ifndef LPFX_ALGORITHM_REDUCE_OPERATION_HPP
#define LPFX_ALGORITHM_REDUCE_OPERATION_HPP

#include <concepts>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>

namespace lpfx {

enum class ReductionOperation : std::uint8_t {
  sum = 1,
  squared_norm = 2,
  minimum = 3,
  maximum = 4,
  absolute_maximum = 5,
  dot = 6,
};

// An empty result means the reduced input had no element. Empty is not an
// error, and no arithmetic identity is published as a result.
template <std::floating_point Accumulator>
class [[nodiscard]] ReductionResult final {
public:
  constexpr ReductionResult() noexcept = default;

  [[nodiscard]] static constexpr ReductionResult
  from_value(const Accumulator value) noexcept {
    return ReductionResult{value};
  }

  [[nodiscard]] constexpr bool has_value() const noexcept { return has_value_; }

  [[nodiscard]] constexpr bool empty() const noexcept { return !has_value_; }

  // Hard precondition: has_value() is true.
  [[nodiscard]] constexpr Accumulator value() const noexcept { return value_; }

private:
  constexpr explicit ReductionResult(const Accumulator value) noexcept
      : value_(value), has_value_(true) {}

  Accumulator value_{};
  bool has_value_{};
};

namespace detail {

template <class T>
concept ReductionAccumulator = std::floating_point<T>;

template <class T>
concept DeviceReductionAccumulator =
    std::same_as<T, float> || std::same_as<T, double>;

template <class T>
concept ReductionInput =
    std::floating_point<T> || (std::integral<T> && !std::same_as<T, bool>);

// Hard precondition: divisor != 0.
template <std::unsigned_integral T>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr T
ceil_div(const T value, const T divisor) noexcept {
  return static_cast<T>(value / divisor +
                        static_cast<T>(value % divisor != 0));
}

template <std::floating_point T>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr T
positive_infinity() noexcept {
  if constexpr (std::same_as<T, float>) {
    return __builtin_huge_valf();
  } else if constexpr (std::same_as<T, double>) {
    return __builtin_huge_val();
  } else {
    return __builtin_huge_vall();
  }
}

// Architecture gate of the binary64 reduction tunings (bit-level selection
// below; full/partial segment split in reduce/device/cuda_hip.hpp). Both
// trade extra integer work, code size and registers for fewer FP64
// instructions, which pays off only where the FP64 pipe runs at a small
// fraction of the FP32 rate; they were measured on such a GPU (RTX 5080,
// 1/64). The value is fixed per device pass from the architecture being
// compiled, so full-rate FP64 targets compile the ungated code:
// - CUDA: compute capabilities whose binary64 add/mul/FMA throughput is at
//   most 1/16 of binary32 in NVIDIA's arithmetic-instruction throughput
//   table: 5.0/5.2/5.3, 6.1/6.2, 7.5, 8.6/8.7/8.9, 12.0/12.1. Full-rate
//   (6.0, 7.0/7.2, 8.0, 9.0, 10.0) and any architecture not listed stay
//   ungated until measured.
// - HIP on AMD: RDNA consumer targets (gfx10, gfx11, gfx12 families; binary64
//   at 1/16 to 1/32 of binary32). GCN/CDNA (gfx8, gfx9: MI50 to MI300) stay
//   ungated.
// HIP on NVIDIA follows the CUDA rule (its device pass defines
// __CUDA_ARCH__). Host passes are never gated in.
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 500 || __CUDA_ARCH__ == 520 || __CUDA_ARCH__ == 530 ||    \
    __CUDA_ARCH__ == 610 || __CUDA_ARCH__ == 620 || __CUDA_ARCH__ == 750 ||    \
    __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 ||    \
    __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
inline constexpr bool device_reduced_rate_fp64 = true;
#else
inline constexpr bool device_reduced_rate_fp64 = false;
#endif
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) &&      \
    (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
inline constexpr bool device_reduced_rate_fp64 = true;
#else
inline constexpr bool device_reduced_rate_fp64 = false;
#endif

// Reduced-rate FP64 device passes (device_reduced_rate_fp64) select binary64
// minima/maxima and absolute values with integer tests, which keep them off
// the FP64 pipe. binary32 keeps the ordered forms: the FP32 pipe gains
// nothing, and binary32 compares may flush subnormals (nvcc -ftz=true, HIP
// -fgpu-flush-denormals-to-zero), which a bit-level absolute value would not.
// No device compiler flushes binary64 subnormals. Host passes keep the
// ordered-compare forms (vectorization gates).
template <class T>
concept DeviceBitSelectDouble =
    device_reduced_rate_fp64 && std::same_as<T, double>;

// value != value as an integer test: after dropping the sign, the bits exceed
// those of infinity exactly for every NaN payload.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool
is_nan_bits(const double value) noexcept {
  return static_cast<std::uint64_t>(__builtin_bit_cast(std::uint64_t, value)
                                    << 1U) > std::uint64_t{0xffe0000000000000ULL};
}

// Once selected, a NaN remains selected; a NaN on the right is selected
// explicitly instead of being discarded by an ordered compare. The device
// form selects identically for every input: with left not NaN,
// !(right >= left) is exactly (right != right || right < left).
template <class T>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr T
nan_propagating_minimum(const T left, const T right) noexcept {
  if constexpr (DeviceBitSelectDouble<T>) {
    return is_nan_bits(left) ? left : (!(right >= left) ? right : left);
  } else if constexpr (std::floating_point<T>) {
    return left != left ? left
                        : (right != right || right < left ? right : left);
  } else {
    return right < left ? right : left;
  }
}

template <class T>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr T
nan_propagating_maximum(const T left, const T right) noexcept {
  if constexpr (DeviceBitSelectDouble<T>) {
    return is_nan_bits(left) ? left : (!(right <= left) ? right : left);
  } else if constexpr (std::floating_point<T>) {
    return left != left ? left
                        : (right != right || right > left ? right : left);
  } else {
    return right > left ? right : left;
  }
}

template <ReductionOperation Operation, ReductionAccumulator Accumulator>
struct ReductionSemantics final {
  using accumulator_type = Accumulator;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr Accumulator
  identity() noexcept {
    if constexpr (Operation == ReductionOperation::minimum) {
      return positive_infinity<Accumulator>();
    } else if constexpr (Operation == ReductionOperation::maximum) {
      return -positive_infinity<Accumulator>();
    } else {
      return Accumulator{};
    }
  }

  template <ReductionInput Value>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr Accumulator
  transform(const Value value) noexcept
    requires(Operation != ReductionOperation::dot)
  {
    const Accumulator converted = static_cast<Accumulator>(value);
    if constexpr (Operation == ReductionOperation::squared_norm) {
      return converted * converted;
    } else if constexpr (Operation == ReductionOperation::absolute_maximum) {
      // Canonicalize either signed zero to positive zero. Ordered comparisons
      // leave a NaN unchanged so the later combine propagates it.
      if constexpr (DeviceBitSelectDouble<Accumulator>) {
        // Same result as below: every non-NaN value loses its sign bit
        // (-0 -> +0, x < 0 -> -x); a NaN is returned unchanged.
        return is_nan_bits(converted)
                   ? converted
                   : __builtin_bit_cast(
                         double, static_cast<std::uint64_t>(
                                     __builtin_bit_cast(std::uint64_t,
                                                        converted) &
                                     std::uint64_t{0x7fffffffffffffffULL}));
      } else {
        if (converted == Accumulator{}) {
          return Accumulator{};
        }
        return converted < Accumulator{} ? -converted : converted;
      }
    } else {
      return converted;
    }
  }

  template <ReductionInput Left, ReductionInput Right>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr Accumulator
  transform(const Left left, const Right right) noexcept
    requires(Operation == ReductionOperation::dot)
  {
    return static_cast<Accumulator>(left) * static_cast<Accumulator>(right);
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr Accumulator
  combine(const Accumulator left, const Accumulator right) noexcept {
    if constexpr (Operation == ReductionOperation::sum ||
                  Operation == ReductionOperation::squared_norm ||
                  Operation == ReductionOperation::dot) {
      return left + right;
    } else if constexpr (Operation == ReductionOperation::minimum) {
      return nan_propagating_minimum(left, right);
    } else {
      static_assert(Operation == ReductionOperation::maximum ||
                    Operation == ReductionOperation::absolute_maximum);
      return nan_propagating_maximum(left, right);
    }
  }
};

// Element i of one reduction input: Semantics::transform of one element or of
// a dot pair, or the untransformed element of a partial-result level.
// operator()(i) is apply(load(i)); device kernels may issue the loads of a
// segment before deciding which of them to combine.
template <class Semantics, bool Transform, class... Pointer>
struct ReductionElements;

template <class Left, class Right> struct ReductionOperandPair final {
  Left left;
  Right right;
};

template <class Semantics, bool Transform, class Pointer>
struct ReductionElements<Semantics, Transform, Pointer> final {
  Pointer data;

  template <class Index>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr auto
  load(const Index index) const noexcept {
    return data[index];
  }

  template <class Raw>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr
      typename Semantics::accumulator_type
      apply(const Raw raw) noexcept {
    if constexpr (Transform) {
      return Semantics::transform(raw);
    } else {
      return raw;
    }
  }

  template <class Index>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr
      typename Semantics::accumulator_type
      operator()(const Index index) const noexcept {
    if constexpr (Transform) {
      return Semantics::transform(data[index]);
    } else {
      return data[index];
    }
  }
};

template <class Semantics, class Left, class Right>
struct ReductionElements<Semantics, true, Left, Right> final {
  Left left;
  Right right;

  template <class Index>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr auto
  load(const Index index) const noexcept {
    return ReductionOperandPair<std::remove_cvref_t<decltype(left[index])>,
                                std::remove_cvref_t<decltype(right[index])>>{
        left[index], right[index]};
  }

  template <class Raw>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr
      typename Semantics::accumulator_type
      apply(const Raw raw) noexcept {
    return Semantics::transform(raw.left, raw.right);
  }

  template <class Index>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr
      typename Semantics::accumulator_type
      operator()(const Index index) const noexcept {
    return Semantics::transform(left[index], right[index]);
  }
};

} // namespace detail

} // namespace lpfx

#endif
