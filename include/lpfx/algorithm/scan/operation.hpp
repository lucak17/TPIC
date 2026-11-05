#ifndef LPFX_ALGORITHM_SCAN_OPERATION_HPP
#define LPFX_ALGORITHM_SCAN_OPERATION_HPP

#include <concepts>
#include <limits>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/algorithm/reduce/operation.hpp>

namespace lpfx {

namespace detail {

template <class T>
concept ScanValue = (std::integral<T> || std::floating_point<T>) &&
                    (!std::same_as<std::remove_cv_t<T>, bool>) &&
                    std::same_as<T, std::remove_cv_t<T>>;

template <class Operation, class T>
concept ScanOperationFor =
    ScanValue<T> && std::is_empty_v<Operation> &&
    std::is_trivially_copyable_v<Operation> &&
    requires(const T left, const T right) {
      typename Operation::value_type;
      requires std::same_as<typename Operation::value_type, T>;
      { Operation::identity() } noexcept -> std::same_as<T>;
      { Operation::combine(left, right) } noexcept -> std::same_as<T>;
    };

// Minimum and maximum identities are namespace-scope constants evaluated on
// the host at compile time. CUDA admits constexpr scalars other than long
// double in device code, so device kernels read the same constants without
// calling a host-only numeric_limits function or needing
// --expt-relaxed-constexpr. Floating point uses the infinities (not the
// finite extremes) so that an infinite input is itself selected; every NaN is
// then selected by the first combine that sees it.
template <ScanValue T>
[[nodiscard]] consteval T scan_extreme_identity(const bool minimum) noexcept {
  if constexpr (std::floating_point<T>) {
    return minimum ? std::numeric_limits<T>::infinity()
                   : -std::numeric_limits<T>::infinity();
  } else {
    return minimum ? std::numeric_limits<T>::max()
                   : std::numeric_limits<T>::lowest();
  }
}

template <ScanValue T>
inline constexpr T scan_minimum_identity = scan_extreme_identity<T>(true);

template <ScanValue T>
inline constexpr T scan_maximum_identity = scan_extreme_identity<T>(false);

} // namespace detail

// Scan operations are stateless compile-time policies. Custom policies follow
// the same identity/combine contract and must be associative. Device scans may
// group combines differently from the serial left fold, so floating-point
// sums and products need not be bitwise identical. Signed arithmetic must not
// overflow. Minimum and maximum preserve the earliest NaN in scan order and
// otherwise the earliest of equal operands (e.g. +0 before -0); both rules are
// associative, so device minimum and maximum scans are bitwise identical to
// the serial host scan.
template <detail::ScanValue T> struct ScanSum final {
  using value_type = T;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  identity() noexcept {
    return T{};
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  combine(const T left, const T right) noexcept {
    return static_cast<T>(left + right);
  }
};

template <detail::ScanValue T> struct ScanProduct final {
  using value_type = T;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  identity() noexcept {
    return T{1};
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  combine(const T left, const T right) noexcept {
    // Unsigned operands narrower than int would multiply as signed int.
    if constexpr (std::is_unsigned_v<T>)
      return static_cast<T>(
          static_cast<std::common_type_t<T, unsigned>>(left) * right);
    else
      return static_cast<T>(left * right);
  }
};

template <detail::ScanValue T> struct ScanMinimum final {
  using value_type = T;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  identity() noexcept {
    return detail::scan_minimum_identity<T>;
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  combine(const T left, const T right) noexcept {
    return detail::nan_propagating_minimum(left, right);
  }
};

template <detail::ScanValue T> struct ScanMaximum final {
  using value_type = T;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  identity() noexcept {
    return detail::scan_maximum_identity<T>;
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr T
  combine(const T left, const T right) noexcept {
    return detail::nan_propagating_maximum(left, right);
  }
};

} // namespace lpfx

#endif
