#ifndef LPFX_ALGORITHM_SCAN_HOST_HPP
#define LPFX_ALGORITHM_SCAN_HOST_HPP

#include <cstddef>

#include <lpfx/core/attributes.hpp>
#include <lpfx/algorithm/scan/operation.hpp>

namespace lpfx {

// Direct left-to-right scans of one contiguous host range. Source and
// destination may be identical. Otherwise their active ranges must not
// overlap. For nonzero count both pointers must be non-null. These are hard
// preconditions so the hot loop performs no validation or policy dispatch.
// Empty scans accept null pointers and perform no operation call.
template <class Operation, detail::ScanValue T>
  requires detail::ScanOperationFor<Operation, T>
LPFX_ALWAYS_INLINE constexpr void
run_serial_inclusive_scan(const T *const source, T *const destination,
                          const std::size_t count) noexcept {
  T prefix = Operation::identity();
  for (std::size_t index = 0; index != count; ++index) {
    const T input = source[index];
    prefix = Operation::combine(prefix, input);
    destination[index] = prefix;
  }
}

template <class Operation, detail::ScanValue T>
  requires detail::ScanOperationFor<Operation, T>
LPFX_ALWAYS_INLINE constexpr void
run_serial_exclusive_scan(const T *const source, T *const destination,
                          const std::size_t count, const T initial) noexcept {
  T prefix = initial;
  for (std::size_t index = 0; index != count; ++index) {
    const T input = source[index];
    destination[index] = prefix;
    prefix = Operation::combine(prefix, input);
  }
}

template <class Operation, detail::ScanValue T>
  requires detail::ScanOperationFor<Operation, T>
LPFX_ALWAYS_INLINE constexpr void
run_serial_exclusive_scan(const T *const source, T *const destination,
                          const std::size_t count) noexcept {
  run_serial_exclusive_scan<Operation>(source, destination, count,
                                       Operation::identity());
}

} // namespace lpfx

#endif
