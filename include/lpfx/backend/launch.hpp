#ifndef LPFX_BACKEND_LAUNCH_HPP
#define LPFX_BACKEND_LAUNCH_HPP

#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx::device {

// Grid blocks every supported device accepts in each grid dimension.
inline constexpr unsigned kernel_grid_block_limit = 65535;

// Block size of a kernel family without a measured reason for another one.
inline constexpr unsigned default_block_threads = 256;

// Launch geometry for kernels that cover their work with a grid-stride loop.
// WorkPerBlock may count threads, cells owned by warps, or tiles. Both it and
// the thread count are compile-time constants; no device query is needed. A
// kernel's __launch_bounds__ names the same Threads constant. The cap is
// applied before narrowing to unsigned.
template <unsigned Threads = default_block_threads,
          unsigned WorkPerBlock = Threads>
struct KernelLaunch final {
  static_assert(Threads > 0, "kernel launches require threads");
  static_assert(WorkPerBlock > 0, "kernel launches require work per block");

  static constexpr unsigned threads = Threads;
  static constexpr unsigned work_per_block = WorkPerBlock;

  // Empty work returns zero blocks. A runtime limit further bounds the grid of
  // a reduction whose scratch is sized per block.
  [[nodiscard]] static constexpr unsigned
  blocks(const std::size_t count,
         const unsigned limit = kernel_grid_block_limit) noexcept {
    const std::size_t wanted =
        count / work_per_block + (count % work_per_block != 0);
    const unsigned capped =
        limit < kernel_grid_block_limit ? limit : kernel_grid_block_limit;
    return wanted < capped ? static_cast<unsigned>(wanted) : capped;
  }

  // Reductions and validation kernels may need a block even for empty input.
  // As with blocks(), a zero runtime limit disables the grid entirely.
  [[nodiscard]] static constexpr unsigned
  blocks_at_least_one(const std::size_t count,
                      const unsigned limit = kernel_grid_block_limit) noexcept {
    return blocks(count == 0 ? 1 : count, limit);
  }
};

template <DeviceBackend Backend>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ErrorCode
to_error_code(const typename BackendTraits<Backend>::error_type error) noexcept {
  return BackendTraits<Backend>::succeeded(error) ? ErrorCode::success
                                                  : ErrorCode::backend_failure;
}

// For native calls whose failure leaves no recoverable state.
template <DeviceBackend Backend>
LPFX_ALWAYS_INLINE void abort_on_failure(
    const typename BackendTraits<Backend>::error_type error) noexcept {
  if (!BackendTraits<Backend>::succeeded(error)) {
    std::abort();
  }
}

// Launch-leaf protocol. The previous last-launch error is consumed first: a
// failure there belongs to an earlier submission and is returned with
// accepted == false. Otherwise launch() enqueues the work, accepted becomes
// true and the launch's own error is returned.
template <DeviceKernelLaunchBackend Backend, class Launch>
[[nodiscard]] LPFX_ALWAYS_INLINE typename BackendTraits<Backend>::error_type
checked_kernel_launch(bool &accepted, Launch &&launch) noexcept {
  using Traits = BackendTraits<Backend>;
  accepted = false;
  const typename Traits::error_type previous = Traits::take_last_launch_error();
  if (!Traits::succeeded(previous)) {
    return previous;
  }
  launch();
  accepted = true;
  return Traits::take_last_launch_error();
}

// Completes a launch leaf's result: a failure before acceptance is a
// recoverable backend_failure, while a failure after acceptance or a success
// without acceptance aborts.
template <DeviceBackend Backend>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
finish_device_enqueue(const typename BackendTraits<Backend>::error_type error,
                      const bool accepted) noexcept {
  const bool succeeded = BackendTraits<Backend>::succeeded(error);
  if (succeeded != accepted) {
    std::abort();
  }
  return succeeded ? ErrorCode::success : ErrorCode::backend_failure;
}

enum class DeviceSubmissionStage : unsigned char {
  none,
  preflight,
  submission,
  completion
};

// Failure-only detail captured before a native status becomes ErrorCode.
// The runtime owns the name/message strings; this record borrows them.
struct DeviceSubmissionDiagnostic {
  DeviceSubmissionStage stage{DeviceSubmissionStage::none};
  int native_code{};
  const char *native_name{};
  const char *native_message{};

  [[nodiscard]] bool has_native_error() const noexcept {
    return stage != DeviceSubmissionStage::none;
  }

  void reset() noexcept { *this = {}; }
};

[[nodiscard]] inline const char *
device_submission_stage_name(const DeviceSubmissionStage stage) noexcept {
  switch (stage) {
  case DeviceSubmissionStage::none: return "none";
  case DeviceSubmissionStage::preflight: return "preflight";
  case DeviceSubmissionStage::submission: return "submission";
  case DeviceSubmissionStage::completion: return "completion";
  }
  return "unknown";
}

// An omitted diagnostic is a compile-time policy. This empty tag also avoids
// consuming an argument register on SysV, unlike a runtime null pointer.
struct DisabledDeviceSubmissionDiagnostic {};

namespace detail {

template <class Diagnostic>
inline constexpr bool captures_device_submission_diagnostic =
    std::is_same_v<std::remove_cvref_t<Diagnostic>,
                   DeviceSubmissionDiagnostic *>;

template <class Diagnostic>
concept device_submission_diagnostic =
    captures_device_submission_diagnostic<Diagnostic> ||
    std::is_same_v<std::remove_cvref_t<Diagnostic>,
                   DisabledDeviceSubmissionDiagnostic> ||
    std::is_same_v<std::remove_cvref_t<Diagnostic>, std::nullptr_t>;

template <DeviceBackend Backend, device_submission_diagnostic Diagnostic>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode capture_device_submission_error(
    const typename BackendTraits<Backend>::error_type error,
    const DeviceSubmissionStage stage, const Diagnostic diagnostic) noexcept {
  using Traits = BackendTraits<Backend>;
  if (Traits::succeeded(error)) return ErrorCode::success;
  if constexpr (captures_device_submission_diagnostic<Diagnostic>) {
    if (diagnostic != nullptr) {
      diagnostic->stage = stage;
      diagnostic->native_code = static_cast<int>(error);
      diagnostic->native_name = Traits::native_error_name(error);
      diagnostic->native_message = Traits::native_error_string(error);
    }
  }
  return ErrorCode::backend_failure;
}

} // namespace detail

// Reset before semantic argument checks, so a rejected call cannot retain a
// previous native failure. Beginning a submission deliberately does not reset:
// argument admission and native preflight are separate boundaries.
template <detail::device_submission_diagnostic Diagnostic>
LPFX_ALWAYS_INLINE void
reset_device_submission_diagnostic(const Diagnostic diagnostic) noexcept {
  if constexpr (detail::captures_device_submission_diagnostic<Diagnostic>) {
    if (diagnostic != nullptr) diagnostic->reset();
  }
}

// Consume the calling host thread's last-launch slot immediately before a
// device operation. Any failure here belongs to an earlier submission.
template <DeviceKernelLaunchBackend Backend,
          detail::device_submission_diagnostic Diagnostic =
              DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
begin_device_submission(const Diagnostic diagnostic = {}) noexcept {
  return detail::capture_device_submission_error<Backend>(
      BackendTraits<Backend>::take_last_launch_error(),
      DeviceSubmissionStage::preflight, diagnostic);
}

// A nonnull record opts into producer-stream completion as well as native
// error capture. Omitted/null diagnostics keep successful calls asynchronous;
// a failed native submission never synchronizes. No application debug settings
// or failure formatting are consulted here.
template <DeviceBackend Backend,
          detail::device_submission_diagnostic Diagnostic =
              DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode finish_device_submission(
    const typename BackendTraits<Backend>::error_type submission_error,
    const Executor<Backend> executor, const Diagnostic diagnostic = {}) noexcept {
  const auto result = detail::capture_device_submission_error<Backend>(
      submission_error, DeviceSubmissionStage::submission, diagnostic);
  if constexpr (detail::captures_device_submission_diagnostic<Diagnostic>) {
    static_assert(DeviceStreamBackend<Backend>);
    if (result != ErrorCode::success || diagnostic == nullptr) return result;
    return detail::capture_device_submission_error<Backend>(
        BackendTraits<Backend>::synchronize_stream(executor.stream()),
        DeviceSubmissionStage::completion, diagnostic);
  }
  return result;
}

template <DeviceKernelLaunchBackend Backend,
          detail::device_submission_diagnostic Diagnostic =
              DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
finish_device_launch(const Executor<Backend> executor,
                     const Diagnostic diagnostic = {}) noexcept {
  return finish_device_submission<Backend>(
      BackendTraits<Backend>::take_last_launch_error(), executor, diagnostic);
}

} // namespace lpfx::device

#endif
