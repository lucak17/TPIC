#ifndef LPFX_BACKEND_ERROR_REPORTING_HPP
#define LPFX_BACKEND_ERROR_REPORTING_HPP

#include <lpfx/backend/launch.hpp>
#include <lpfx/core/error_reporting.hpp>

namespace lpfx::device {

// A native CUDA/HIP error of Backend, formatted through its traits.
template <DeviceBackend Backend> struct DeviceError final {
  typename BackendTraits<Backend>::error_type native;
};

template <DeviceBackend Backend>
inline int format_error(char *buffer, std::size_t size,
                        const DeviceError<Backend> error,
                        const char *context) noexcept {
  using Traits = BackendTraits<Backend>;
  const bool failed = !Traits::succeeded(error.native);
  const char *const name =
      failed ? Traits::native_error_name(error.native) : nullptr;
  const char *const message =
      failed ? Traits::native_error_string(error.native) : nullptr;
  return std::snprintf(
      buffer, size, "%s: native_code=%d native_name=%s native_message=%s",
      lpfx::detail::error_text(context), static_cast<int>(error.native),
      lpfx::detail::error_text(name), lpfx::detail::error_text(message));
}

inline int format_error(char *buffer, std::size_t size, const ErrorCode code,
                        const DeviceSubmissionDiagnostic &diagnostic,
                        const char *context) noexcept {
  if (!diagnostic.has_native_error())
    return std::snprintf(buffer, size, "%s: lpfx=%s(%u) native_stage=none",
                         lpfx::detail::error_text(context),
                         error_code_name(code), static_cast<unsigned>(code));
  return std::snprintf(buffer, size,
                       "%s: lpfx=%s(%u) native_stage=%s native_code=%d "
                       "native_name=%s native_message=%s",
                       lpfx::detail::error_text(context), error_code_name(code),
                       static_cast<unsigned>(code),
                       device_submission_stage_name(diagnostic.stage),
                       diagnostic.native_code,
                       lpfx::detail::error_text(diagnostic.native_name),
                       lpfx::detail::error_text(diagnostic.native_message));
}

#if LPFX_HAS_EXCEPTIONS

template <DeviceBackend Backend>
inline void
check_device_error(const typename BackendTraits<Backend>::error_type error,
                   const char *context) {
  if (!BackendTraits<Backend>::succeeded(error))
    lpfx::detail::throw_error<std::runtime_error>(context,
                                                  DeviceError<Backend>{error});
}

// Explicitly consumes one last-launch error. It never synchronizes the device.
template <DeviceKernelLaunchBackend Backend>
inline void check_device_launch(const char *context) {
  check_device_error<Backend>(BackendTraits<Backend>::take_last_launch_error(),
                              context);
}

// Checked event creation, record and wait: each performs the single
// DeviceEvent call it names. The backend is deduced from the event; the
// executor must belong to it.
template <DeviceEventBackend Backend>
inline void create_event(DeviceEvent<Backend> &event, const char *context) {
  check_device_error<Backend>(DeviceEvent<Backend>::create(event), context);
}

template <DeviceEventBackend Backend>
inline void record_event(DeviceEvent<Backend> &event,
                         const Executor<Backend> executor,
                         const char *context) {
  check_device_error<Backend>(event.record(executor), context);
}

template <DeviceEventBackend Backend>
inline void wait_event(const DeviceEvent<Backend> &event,
                       const Executor<Backend> executor, const char *context) {
  check_device_error<Backend>(event.wait_on(executor), context);
}

inline void
check_device_submission(const ErrorCode code,
                        const DeviceSubmissionDiagnostic &diagnostic,
                        const char *context) {
  if (code != ErrorCode::success)
    lpfx::detail::throw_error<std::runtime_error>(context, code, diagnostic);
}

#endif

} // namespace lpfx::device

#endif // LPFX_BACKEND_ERROR_REPORTING_HPP
