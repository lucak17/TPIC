#ifndef LPFX_EXECUTION_HOST_OPENMP_HPP
#define LPFX_EXECUTION_HOST_OPENMP_HPP

#include <lpfx/config.hpp>
#include <lpfx/core/attributes.hpp>

// HIP/CUDA compiler drivers parse a translation unit once for the host and
// once per device architecture. _OPENMP is a host-pass contract; the host
// parse still rejects consumers that did not link the OpenMP component.
#if !LPFX_CONFIG_ENABLE_OPENMP
#error LPFX_EXPECTED_OPENMP_COMPONENT_NOT_CONFIGURED
#elif !defined(_OPENMP) && !LPFX_DEVICE_PASS
#error LPFX_EXPECTED_OPENMP_TARGET_REQUIRED
#endif

#include <lpfx/core/error.hpp>
#include <lpfx/execution/host/serial.hpp>

namespace lpfx {

// Every for_each_index callback submitted to this executor must have no
// loop-carried dependency and must not rely on iteration order, irrespective
// of requested worker count. The one-worker/default case retains that contract
// and therefore uses the same SIMD-capable traversal as UnsequencedExecutor.
class OpenMPExecutor final {
public:
  constexpr OpenMPExecutor() noexcept = default;

  [[nodiscard]] LPFX_ALWAYS_INLINE constexpr int
  requested_worker_capacity() const noexcept {
    return requested_worker_capacity_;
  }

private:
  constexpr explicit OpenMPExecutor(const int requested_worker_capacity) noexcept
      : requested_worker_capacity_(requested_worker_capacity) {}

  int requested_worker_capacity_ = 1;

  friend constexpr ErrorCode
  try_make_openmp_executor(int, OpenMPExecutor &) noexcept;
};

// Transactional checked construction. The default executor is the valid
// single-worker configuration, so failure leaves output usable and unchanged.
[[nodiscard]] constexpr ErrorCode
try_make_openmp_executor(const int requested_worker_capacity,
                         OpenMPExecutor &output) noexcept {
  if (requested_worker_capacity < 1) {
    return ErrorCode::invalid_argument;
  }
  output = OpenMPExecutor{requested_worker_capacity};
  return ErrorCode::success;
}

template <detail::CartesianIndexPolicy IndexPolicy, class Count, class Function>
  requires std::same_as<Count, typename IndexPolicy::count_type> &&
           detail::NothrowHostIndexFunction<Function, Count>
LPFX_ALWAYS_INLINE void for_each_index(const OpenMPExecutor &executor,
                                       const Count count,
                                       Function &&function) noexcept {
  if (count == Count{0}) {
    return;
  }

  const int requested_worker_capacity = executor.requested_worker_capacity();
  if (requested_worker_capacity == 1) {
    const UnsequencedExecutor unsequenced_executor{};
    for_each_index<IndexPolicy>(unsequenced_executor, count, function);
    return;
  }

#pragma omp parallel for simd default(none) firstprivate(count)                \
    shared(function) schedule(static) num_threads(requested_worker_capacity)
  for (Count index = Count{0}; index < count; ++index) {
    function(Count{index});
  }
}

} // namespace lpfx

#endif
