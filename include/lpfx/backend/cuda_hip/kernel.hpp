#ifndef LPFX_BACKEND_CUDA_HIP_KERNEL_HPP
#define LPFX_BACKEND_CUDA_HIP_KERNEL_HPP

// What an application header that defines and launches kernels needs from the
// backend: the vendor kernel runtime, the grid-stride loop and the launch
// statement. It compiles only under the CUDA or HIP compiler, so including it
// first marks a header as device-only.
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#error LPFX_KERNEL_HEADER_REQUIRES_CUDA_OR_HIP_COMPILATION
#endif

#include <cstddef>
#include <utility>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx::device {

// Calls body(index) for every index in [0, count) assigned to this thread by a
// one-dimensional grid-stride loop. Index keeps the caller's loop type; the
// body returns where a hand-written loop would continue. count plus the grid
// size must be representable in Index.
template <class Index, class Body>
LPFX_DEVICE LPFX_ALWAYS_INLINE void for_each_grid_index(const Index count,
                                                        Body &&body) {
  const Index stride = blockDim.x * gridDim.x;
  for (Index index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += stride)
    body(index);
}

// Grid and block of a launch whose geometry is not Shape::blocks(work):
// single-thread launches, two-dimensional grids, runtime-capped grids and
// grids that keep one block for empty input.
struct LaunchGrid {
  dim3 blocks;
  unsigned threads;
};

// Zero work either launches the computed grid or returns success after the
// preflight check, without a launch.
enum class EmptyWork { launch, skip };

// The launch-error slot is consumed by launch, or was consumed by the
// caller's admission through begin_device_submission.
enum class LaunchPreflight { check, admitted };

// The kernel is a non-type template parameter, so the call resolves at compile
// time on both backends; every kernel template argument is spelled, because a
// launch through a named specialization does not deduce. Each launch is always
// inlined and submits exactly what a hand-written one would: the given grid
// and block, no dynamic shared memory, the executor's stream. The bracket
// around it consumes a stale launch error and fails without launching, and
// reports the launch's own error; no synchronisation is added unless a
// diagnostic record is passed.
//
// Grid Shape::blocks(work) of Shape::threads, for a KernelLaunch Shape.
template <class Shape, auto Kernel, EmptyWork Empty = EmptyWork::launch,
          LaunchPreflight Preflight = LaunchPreflight::check,
          DeviceKernelLaunchBackend Backend,
          detail::device_submission_diagnostic Diagnostic, class... Arguments>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
launch(const Executor<Backend> executor, const Diagnostic diagnostic,
       const std::size_t work, Arguments &&...arguments) {
  if constexpr (Preflight == LaunchPreflight::check) {
    const ErrorCode begun = begin_device_submission<Backend>(diagnostic);
    if (begun != ErrorCode::success)
      return begun;
  }
  if constexpr (Empty == EmptyWork::skip)
    if (work == 0)
      return ErrorCode::success;
  Kernel<<<Shape::blocks(work), Shape::threads, 0, executor.stream()>>>(
      std::forward<Arguments>(arguments)...);
  return finish_device_launch(executor, diagnostic);
}

template <class Shape, auto Kernel, EmptyWork Empty = EmptyWork::launch,
          LaunchPreflight Preflight = LaunchPreflight::check,
          DeviceKernelLaunchBackend Backend, class... Arguments>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
launch(const Executor<Backend> executor, const std::size_t work,
       Arguments &&...arguments) {
  return launch<Shape, Kernel, Empty, Preflight>(
      executor, DisabledDeviceSubmissionDiagnostic{}, work,
      std::forward<Arguments>(arguments)...);
}

#if LPFX_HAS_EXCEPTIONS

// Host code that reports by exception: the same bracket, with the native
// error formatted into context.
template <auto Kernel, DeviceKernelLaunchBackend Backend, class... Arguments>
LPFX_ALWAYS_INLINE void launch_or_throw(const char *context,
                                        const Executor<Backend> executor,
                                        const LaunchGrid grid,
                                        Arguments &&...arguments) {
  check_device_launch<Backend>(context);
  Kernel<<<grid.blocks, grid.threads, 0, executor.stream()>>>(
      std::forward<Arguments>(arguments)...);
  check_device_launch<Backend>(context);
}

template <class Shape, auto Kernel, DeviceKernelLaunchBackend Backend,
          class... Arguments>
LPFX_ALWAYS_INLINE void launch_or_throw(const char *context,
                                        const Executor<Backend> executor,
                                        const std::size_t work,
                                        Arguments &&...arguments) {
  launch_or_throw<Kernel>(context, executor,
                          LaunchGrid{Shape::blocks(work), Shape::threads},
                          std::forward<Arguments>(arguments)...);
}

#endif

} // namespace lpfx::device

#endif // LPFX_BACKEND_CUDA_HIP_KERNEL_HPP
