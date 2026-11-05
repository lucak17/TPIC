#ifndef LPFX_ALGORITHM_SCAN_DEVICE_SCAN_HPP
#define LPFX_ALGORITHM_SCAN_DEVICE_SCAN_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/error.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/algorithm/scan/device/workspace.hpp>
#include <lpfx/algorithm/scan/operation.hpp>

#if LPFX_NATIVE_DEVICE_OPERATIONS
#include <lpfx/algorithm/scan/device/cuda_hip.hpp>
#endif

namespace lpfx {

namespace detail {

template <class Backend, bool Inclusive, class Operation, class T,
          class LaunchPolicy>
concept DeviceScanLaunchable =
    ScanOperationFor<Operation, T> && device::DeviceBackend<Backend> &&
    DeviceScanLaunchPolicyType<LaunchPolicy> &&
    requires(const T *const source, T *const destination,
             const std::uint32_t count, const T initial,
             std::byte *const scratch,
             const PreparedDeviceScanWorkspace<T, LaunchPolicy> &prepared,
             const typename device::BackendTraits<Backend>::stream_type stream,
             bool &accepted) {
      {
        DeviceScanLaunch<Backend>::template enqueue<Inclusive, Operation, T,
                                                    LaunchPolicy>(
            source, destination, count, initial, scratch, prepared, stream,
            accepted)
      } noexcept
          -> std::same_as<typename device::BackendTraits<Backend>::error_type>;
    };

template <bool Inclusive, class Operation, ScanValue T,
          device::DeviceAllocationBackend Backend,
          DeviceScanLaunchPolicyType LaunchPolicy>
  requires DeviceScanLaunchable<Backend, Inclusive, Operation, T, LaunchPolicy>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_scan(
    const T *const source, T *const destination, const std::uint32_t count,
    const T initial, DeviceScanWorkspace<T, Backend, LaunchPolicy> &workspace,
    const device::Executor<Backend> executor) noexcept {
  if (count == 0) {
    return ErrorCode::success;
  }
  if (!workspace.valid()) {
    return ErrorCode::invalid_state;
  }
  if (count > workspace.capacity()) {
    return ErrorCode::capacity_exceeded;
  }
  if (source == nullptr || destination == nullptr) {
    return ErrorCode::invalid_argument;
  }
  if (!executor.bound()) {
    return ErrorCode::invalid_state;
  }

  bool accepted = false;
  const auto error =
      DeviceScanLaunch<Backend>::template enqueue<Inclusive, Operation, T,
                                                  LaunchPolicy>(
          source, destination, count, initial, workspace.scratch(),
          workspace.prepared(), executor.stream(), accepted);
  return device::finish_device_enqueue<Backend>(error, accepted);
}

} // namespace detail

// Hierarchical stream-ordered scans of a contiguous device range. Source and
// destination may be identical; otherwise their active ranges must not
// overlap. Workspace use may be sequenced repeatedly on one stream, but must
// not overlap on independent streams. Storage and workspace remain live until
// later work on the same stream proves completion. Submission does not
// allocate, create, query, or synchronize.
template <class Operation, detail::ScanValue T,
          device::DeviceAllocationBackend Backend,
          detail::DeviceScanLaunchPolicyType LaunchPolicy>
  requires detail::DeviceScanLaunchable<Backend, true, Operation, T,
                                        LaunchPolicy>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_inclusive_scan(
    const T *const source, T *const destination, const std::uint32_t count,
    DeviceScanWorkspace<T, Backend, LaunchPolicy> &workspace,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_scan<true, Operation>(
      source, destination, count, Operation::identity(), workspace, executor);
}

template <class Operation, detail::ScanValue T,
          device::DeviceAllocationBackend Backend,
          detail::DeviceScanLaunchPolicyType LaunchPolicy>
  requires detail::DeviceScanLaunchable<Backend, false, Operation, T,
                                        LaunchPolicy>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_exclusive_scan(
    const T *const source, T *const destination, const std::uint32_t count,
    const T initial, DeviceScanWorkspace<T, Backend, LaunchPolicy> &workspace,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_scan<false, Operation>(
      source, destination, count, initial, workspace, executor);
}

template <class Operation, detail::ScanValue T,
          device::DeviceAllocationBackend Backend,
          detail::DeviceScanLaunchPolicyType LaunchPolicy>
  requires detail::DeviceScanLaunchable<Backend, false, Operation, T,
                                        LaunchPolicy>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_exclusive_scan(
    const T *const source, T *const destination, const std::uint32_t count,
    DeviceScanWorkspace<T, Backend, LaunchPolicy> &workspace,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_scan<false, Operation>(
      source, destination, count, Operation::identity(), workspace, executor);
}

} // namespace lpfx

#endif
