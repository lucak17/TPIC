#ifndef LPFX_BACKEND_TRAITS_HPP
#define LPFX_BACKEND_TRAITS_HPP

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include <lpfx/backend/properties.hpp>

namespace lpfx::device {

struct CudaBackend final {};
struct HipBackend final {};

// A backend exists only when its native specialization is included. Keeping
// the primary template incomplete prevents disabled backends from silently
// acquiring placeholder handles or host fallbacks.
template <class Backend> struct BackendTraits;

namespace detail {

template <class Result, class Backend>
concept native_status =
    std::same_as<Result, typename BackendTraits<Backend>::error_type>;

} // namespace detail

// Error vocabulary and the borrowed stream handle shared by every role below.
template <class Backend>
concept DeviceBackend =
    requires(const typename BackendTraits<Backend>::error_type error,
             const typename BackendTraits<Backend>::stream_type stream) {
      {
        BackendTraits<Backend>::success_error
      } -> std::convertible_to<typename BackendTraits<Backend>::error_type>;
      { BackendTraits<Backend>::succeeded(error) } noexcept
          -> std::same_as<bool>;
      { BackendTraits<Backend>::native_error_name(error) } noexcept
          -> std::same_as<const char *>;
      { BackendTraits<Backend>::native_error_string(error) } noexcept
          -> std::same_as<const char *>;
      {
        BackendTraits<Backend>::null_stream
      } -> std::convertible_to<typename BackendTraits<Backend>::stream_type>;
      { stream != BackendTraits<Backend>::null_stream } noexcept
          -> std::convertible_to<bool>;
    };

template <class Backend>
concept DeviceAllocationBackend =
    DeviceBackend<Backend> &&
    requires(std::byte *&allocation, std::byte *const owned,
             const std::size_t bytes) {
      { BackendTraits<Backend>::allocate_device(allocation, bytes) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::deallocate_device(owned) } noexcept
          -> detail::native_status<Backend>;
    } &&
    std::has_single_bit(static_cast<std::size_t>(
        BackendTraits<Backend>::minimum_device_allocation_alignment));

template <class Backend>
concept PinnedHostAllocationBackend =
    DeviceBackend<Backend> &&
    requires(std::byte *&allocation, std::byte *const owned,
             const std::size_t bytes) {
      {
        BackendTraits<Backend>::allocate_pinned_host(allocation, bytes)
      } noexcept -> detail::native_status<Backend>;
      { BackendTraits<Backend>::deallocate_pinned_host(owned) } noexcept
          -> detail::native_status<Backend>;
    };

// Setup-time stream ownership and admission. Submission only borrows a
// stream through Executor and never needs this role.
template <class Backend>
concept DeviceStreamBackend =
    DeviceBackend<Backend> &&
    requires(typename BackendTraits<Backend>::stream_type &output,
             const typename BackendTraits<Backend>::stream_type stream,
             bool &nonblocking) {
      { BackendTraits<Backend>::create_stream(output) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::destroy_stream(stream) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::synchronize_stream(stream) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::is_default_stream(stream) } noexcept
          -> std::same_as<bool>;
      {
        BackendTraits<Backend>::query_stream_nonblocking(stream, nonblocking)
      } noexcept -> detail::native_status<Backend>;
    };

// Optional stream priorities, required only by high-priority streams.
template <class Backend>
concept DeviceStreamPriorityBackend =
    DeviceStreamBackend<Backend> &&
    requires(typename BackendTraits<Backend>::stream_type &output,
             const int priority, int &least, int &greatest) {
      { BackendTraits<Backend>::create_stream(output, priority) } noexcept
          -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::query_stream_priority_range(least, greatest)
      } noexcept -> detail::native_status<Backend>;
    };

// Stream-ordered copies and fills. The pitch limit applies to the current
// device and must validate every native 2D restriction.
template <class Backend>
concept DeviceCopyBackend =
    DeviceBackend<Backend> &&
    requires(void *const destination, const void *const source,
             const std::size_t bytes, const std::size_t pitch,
             const std::size_t rows, const unsigned char value,
             std::size_t &max_pitch_bytes,
             const typename BackendTraits<Backend>::stream_type stream) {
      {
        BackendTraits<Backend>::copy_host_to_device_async(destination, source,
                                                          bytes, stream)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::copy_device_to_host_async(destination, source,
                                                          bytes, stream)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::copy_device_to_device_async(destination, source,
                                                            bytes, stream)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::copy_host_to_device_2d_async(
            destination, pitch, source, pitch, bytes, rows, stream)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::copy_device_to_host_2d_async(
            destination, pitch, source, pitch, bytes, rows, stream)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::query_host_device_max_copy_pitch(max_pitch_bytes)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::memset_device_async(destination, value, bytes,
                                                    stream)
      } noexcept -> detail::native_status<Backend>;
    };

// Dependency events are timing-disabled; profiling creates separate timing
// events. stream_wait_event only inserts a device-side dependency.
template <class Backend>
concept DeviceEventBackend =
    DeviceBackend<Backend> &&
    requires(typename BackendTraits<Backend>::event_type &output,
             const typename BackendTraits<Backend>::event_type event,
             const typename BackendTraits<Backend>::stream_type stream,
             const typename BackendTraits<Backend>::error_type error,
             float &milliseconds) {
      {
        BackendTraits<Backend>::null_event
      } -> std::convertible_to<typename BackendTraits<Backend>::event_type>;
      { event != BackendTraits<Backend>::null_event } noexcept
          -> std::convertible_to<bool>;
      { BackendTraits<Backend>::create_event(output) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::create_timing_event(output) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::destroy_event(event) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::record_event(event, stream) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::stream_wait_event(stream, event) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::query_event(event) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::synchronize_event(event) } noexcept
          -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::event_elapsed_time(milliseconds, event, event)
      } noexcept -> detail::native_status<Backend>;
      { BackendTraits<Backend>::not_ready(error) } noexcept
          -> std::same_as<bool>;
    };

// CUDA/HIP launch leaves consume the runtime's per-thread launch-error slot.
template <class Backend>
concept DeviceKernelLaunchBackend =
    DeviceBackend<Backend> && requires {
      { BackendTraits<Backend>::take_last_launch_error() } noexcept
          -> detail::native_status<Backend>;
    };

// Cold availability, selection, launch properties and physical affinity.
// Device selection is local to the calling CPU thread.
template <class Backend>
concept DeviceDiscoveryBackend =
    DeviceBackend<Backend> &&
    requires(const typename BackendTraits<Backend>::error_type error,
             const typename BackendTraits<Backend>::stream_type stream,
             const void *const pointer, const int ordinal, int &count,
             DeviceMemoryInfo &memory, DeviceProperties &properties,
             DeviceIdentity &identity, std::int32_t &device_ordinal) {
      { BackendTraits<Backend>::runtime_unavailable(error) } noexcept
          -> std::same_as<bool>;
      { BackendTraits<Backend>::query_device_count(count) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::select_device(ordinal) } noexcept
          -> detail::native_status<Backend>;
      { BackendTraits<Backend>::query_current_device_memory(memory) } noexcept
          -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::query_current_device_properties(properties)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::query_current_device_identity(identity,
                                                              device_ordinal)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::query_stream_device_ordinal(stream,
                                                            device_ordinal)
      } noexcept -> detail::native_status<Backend>;
      {
        BackendTraits<Backend>::query_device_pointer_ordinal(pointer,
                                                             device_ordinal)
      } noexcept -> detail::native_status<Backend>;
    };

// A complete native runtime, as provided by CUDA and HIP.
template <class Backend>
concept DeviceRuntimeBackend =
    DeviceAllocationBackend<Backend> && PinnedHostAllocationBackend<Backend> &&
    DeviceStreamBackend<Backend> && DeviceCopyBackend<Backend> &&
    DeviceEventBackend<Backend> && DeviceKernelLaunchBackend<Backend> &&
    DeviceDiscoveryBackend<Backend>;

} // namespace lpfx::device

#endif
