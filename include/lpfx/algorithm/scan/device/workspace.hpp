#ifndef LPFX_ALGORITHM_SCAN_DEVICE_WORKSPACE_HPP
#define LPFX_ALGORITHM_SCAN_DEVICE_WORKSPACE_HPP

#include <cstddef>
#include <cstdint>
#include <utility>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/algorithm/reduce/device/workspace.hpp>
#include <lpfx/algorithm/scan/operation.hpp>

namespace lpfx {

template <std::uint32_t BlockThreads>
  requires(detail::valid_device_block_threads(BlockThreads))
struct DeviceScanLaunchPolicy final {
  static constexpr std::uint32_t block_threads = BlockThreads;
};

using DefaultDeviceScanLaunchPolicy = DeviceScanLaunchPolicy<256>;

namespace detail {

template <class Policy>
concept DeviceScanLaunchPolicyType = DeviceBlockLaunchPolicy<Policy>;

// Hard precondition: block_threads != 0.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint32_t
device_block_count(const std::uint32_t count,
                   const std::uint32_t block_threads) noexcept {
  return ceil_div(count, block_threads);
}

} // namespace detail

// Block-total levels of the hierarchical scan of up to capacity() elements.
template <detail::ScanValue T, detail::DeviceScanLaunchPolicyType LaunchPolicy =
                                   DefaultDeviceScanLaunchPolicy>
class PreparedDeviceScanWorkspace final {
public:
  using value_type = T;
  using launch_policy = LaunchPolicy;

  constexpr PreparedDeviceScanWorkspace() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::uint32_t capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] constexpr std::uint8_t level_count() const noexcept {
    return levels_.level_count;
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return levels_.total_bytes;
  }

  // Hard precondition: level < level_count().
  [[nodiscard]] constexpr std::uint32_t
  level_element_count(const std::size_t level) const noexcept {
    return levels_.counts[level];
  }

  // Hard precondition: level < level_count().
  [[nodiscard]] constexpr std::size_t
  level_byte_offset(const std::size_t level) const noexcept {
    return levels_.byte_offsets[level];
  }

  [[nodiscard]] friend constexpr bool
  operator==(const PreparedDeviceScanWorkspace &,
             const PreparedDeviceScanWorkspace &) noexcept = default;

private:
  constexpr PreparedDeviceScanWorkspace(
      const std::uint32_t capacity,
      const detail::DeviceLevelLayout levels) noexcept
      : levels_(levels), capacity_(capacity), valid_(true) {}

  detail::DeviceLevelLayout levels_{};
  std::uint32_t capacity_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ScanValue T,
          detail::DeviceScanLaunchPolicyType LaunchPolicy =
              DefaultDeviceScanLaunchPolicy,
          detail::checked_integer Capacity>
[[nodiscard]] constexpr ErrorCode try_prepare_device_scan_workspace(
    const Capacity requested_capacity,
    const device::DeviceProperties &properties,
    PreparedDeviceScanWorkspace<T, LaunchPolicy> &output) noexcept {
  std::uint32_t capacity{};
  if (!detail::checked_integral_cast(requested_capacity, capacity)) {
    return ErrorCode::overflow;
  }

  constexpr std::uint32_t block_threads = LaunchPolicy::block_threads;
  const std::uint32_t base_block_count =
      detail::device_block_count(capacity, block_threads);
  if (capacity != 0 && (properties.maximum_threads_per_block < block_threads ||
                        properties.maximum_grid_size_x < base_block_count)) {
    return ErrorCode::invalid_argument;
  }

  detail::DeviceLevelLayout levels{};
  if (!detail::try_make_device_level_layout<T, block_threads, false>(
          base_block_count, levels)) {
    return ErrorCode::overflow;
  }
  output = detail::PrivateConstructionAccess::create<
      PreparedDeviceScanWorkspace<T, LaunchPolicy>>(capacity, levels);
  return ErrorCode::success;
}

template <detail::ScanValue T, device::DeviceAllocationBackend Backend,
          detail::DeviceScanLaunchPolicyType LaunchPolicy =
              DefaultDeviceScanLaunchPolicy>
class DeviceScanWorkspace final {
public:
  using value_type = T;
  using backend_type = Backend;
  using launch_policy = LaunchPolicy;
  using prepared_type = PreparedDeviceScanWorkspace<T, LaunchPolicy>;
  using traits_type = device::BackendTraits<Backend>;

  constexpr DeviceScanWorkspace() noexcept = default;

  DeviceScanWorkspace(DeviceScanWorkspace &&other) noexcept
      : allocation_(std::move(other.allocation_)),
        prepared_(std::exchange(other.prepared_, {})) {}

  DeviceScanWorkspace &operator=(DeviceScanWorkspace &&) = delete;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return prepared_.valid();
  }
  [[nodiscard]] constexpr std::uint32_t capacity() const noexcept {
    return prepared_.capacity();
  }
  [[nodiscard]] constexpr std::size_t scratch_capacity_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }
  [[nodiscard]] constexpr const prepared_type &prepared() const noexcept {
    return prepared_;
  }
  // Level storage handed to the launch leaf.
  [[nodiscard]] constexpr std::byte *scratch() noexcept {
    return allocation_.raw_base();
  }

  // Hard precondition: all submitted scans using this workspace are complete.
  [[nodiscard]] typename traits_type::error_type reset_capacity() noexcept {
    const typename traits_type::error_type error = allocation_.reset();
    prepared_ = {};
    return error;
  }

private:
  DeviceScanWorkspace(device::DeviceAllocation<Backend> &&allocation,
                      const prepared_type prepared) noexcept
      : allocation_(std::move(allocation)), prepared_(prepared) {}

  device::DeviceAllocation<Backend> allocation_{};
  prepared_type prepared_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ScanValue T, device::DeviceAllocationBackend Backend,
          detail::DeviceScanLaunchPolicyType LaunchPolicy>
[[nodiscard]] Result<DeviceScanWorkspace<T, Backend, LaunchPolicy>>
make_device_scan_workspace(
    const PreparedDeviceScanWorkspace<T, LaunchPolicy> &prepared) noexcept {
  using Workspace = DeviceScanWorkspace<T, Backend, LaunchPolicy>;
  if (!prepared.valid()) {
    return Result<Workspace>::from_error(ErrorCode::invalid_argument);
  }
  device::DeviceAllocation<Backend> allocation{};
  if (!device::BackendTraits<Backend>::succeeded(
          device::DeviceAllocation<Backend>::create(prepared.total_bytes(),
                                                    allocation))) {
    return Result<Workspace>::from_error(ErrorCode::allocation_failed);
  }
  return Result<Workspace>::from_value(
      detail::PrivateConstructionAccess::create<Workspace>(
          std::move(allocation), prepared));
}

namespace detail {

// Launch seam, specialized statically by a device-language leaf:
//   enqueue<Inclusive, Operation, T, LaunchPolicy>(source, destination, count,
//       initial, scratch, prepared, stream, accepted)
template <class Backend> struct DeviceScanLaunch;

} // namespace detail

} // namespace lpfx

#endif
