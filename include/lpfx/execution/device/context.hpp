#ifndef LPFX_EXECUTION_DEVICE_CONTEXT_HPP
#define LPFX_EXECUTION_DEVICE_CONTEXT_HPP

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx::device {

// Call once in a host worker's setup before it submits device work. Failure
// is reported to the caller's startup/error policy; no implicit device switch
// is hidden in Executor or DeviceEvent submission.
template <DeviceDiscoveryBackend Backend>
[[nodiscard]] auto select_device(const int ordinal) noexcept ->
    typename BackendTraits<Backend>::error_type {
  return BackendTraits<Backend>::select_device(ordinal);
}

// Cold admission of an application-owned stream whose creation device is
// known. Older native runtimes without exact stream-affinity queries verify
// the current creation device under that provenance contract. Special default
// handles must be rejected before making any native query.
template <class Backend>
  requires DeviceStreamBackend<Backend> && DeviceDiscoveryBackend<Backend>
[[nodiscard]] ErrorCode try_validate_explicit_executor(
    const Executor<Backend> executor,
    const std::int32_t expected_device_ordinal) noexcept {
  using Traits = BackendTraits<Backend>;
  if (expected_device_ordinal < 0 || !executor.bound() ||
      Traits::is_default_stream(executor.stream())) {
    return ErrorCode::invalid_argument;
  }
  bool nonblocking = false;
  if (!Traits::succeeded(
          Traits::query_stream_nonblocking(executor.stream(), nonblocking))) {
    return ErrorCode::backend_failure;
  }
  if (!nonblocking) {
    return ErrorCode::invalid_argument;
  }
  std::int32_t actual_ordinal = -1;
  if (!Traits::succeeded(Traits::query_stream_device_ordinal(
          executor.stream(), actual_ordinal))) {
    return ErrorCode::backend_failure;
  }
  return actual_ordinal == expected_device_ordinal ? ErrorCode::success
                                                 : ErrorCode::invalid_argument;
}

} // namespace lpfx::device

#endif
