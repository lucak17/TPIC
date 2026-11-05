#ifndef LPFX_ALGORITHM_DEVICE_KERNEL_HPP
#define LPFX_ALGORITHM_DEVICE_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>
#include <lpfx/domain/cartesian/halo.hpp>
#include <lpfx/domain/cartesian/location.hpp>

namespace lpfx::detail {

inline constexpr std::uint32_t device_grid_block_threads = 256;

// A fixed cap keeps direct launches valid on every supported device while
// grid-stride traversal covers arbitrarily large prepared work.
inline constexpr std::uint32_t device_grid_maximum_blocks = 2520;

// Hard preconditions: total > 0.
[[nodiscard]] constexpr std::uint32_t
grid_stride_block_count(const std::uint64_t total) noexcept {
  constexpr std::uint64_t block_threads = device_grid_block_threads;
  std::uint64_t desired_blocks = 1U + (total - 1U) / block_threads;
  if (desired_blocks > device_grid_maximum_blocks) {
    desired_blocks = device_grid_maximum_blocks;
  }
  return static_cast<std::uint32_t>(desired_blocks);
}

// Hard preconditions: sample < total and stride > 0. The subtraction-first
// test prevents a traversal ending near UINT64_MAX from wrapping to a small
// sample index.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool
advance_grid_stride_sample(const std::uint64_t total,
                           const std::uint64_t stride,
                           std::uint64_t &sample) noexcept {
  if (total - sample <= stride) {
    return false;
  }
  sample += stride;
  return true;
}

// Kernel-argument layout guard: T is copied bytewise and holds exactly the
// listed members plus trailing alignment.
template <class T, class... Members>
inline constexpr bool packed_trivially_copyable =
    std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T> &&
    sizeof(T) == ((sizeof(Members) + ... + std::size_t{0}) + alignof(T) - 1) /
                     alignof(T) * alignof(T);

// Location samples over nonempty cells, with every axis addressable by
// IndexPolicy coordinates, and their product. All failures are overflow.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_prepare_device_location_samples(
    const CartesianExtent3<typename IndexPolicy::count_type> cells,
    CartesianExtent3<typename IndexPolicy::count_type> &samples,
    std::uint64_t &sample_count) noexcept {
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(cells,
                                                                     samples)));
  AllocatedAxisInterval<IndexPolicy> interval{};
  LPFX_TRY(for_each_axis([&](auto axis) {
    return try_make_allocated_axis_interval<IndexPolicy>(
        samples.template get<decltype(axis)::value>(), 0U, 0U, interval);
  }));
  return checked_volume(samples, sample_count) ? ErrorCode::success
                                               : ErrorCode::overflow;
}

} // namespace lpfx::detail

#endif
