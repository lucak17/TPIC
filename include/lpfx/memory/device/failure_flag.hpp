#ifndef LPFX_MEMORY_DEVICE_FAILURE_FLAG_HPP
#define LPFX_MEMORY_DEVICE_FAILURE_FLAG_HPP

#include <cstdlib>
#include <memory>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx::device {

// One reusable unsigned producer flag and its event-bounded host capture.
// Kernels and callers define the bits and interpret the completed value; a
// successful read reports transfer completion, including for a nonzero flag.
// This owner is confined to one host worker at a time. Capture may transfer
// ownership to a checking worker; other consumers need their own dependency.
// The producer stream must outlive capture/teardown. Callers must retire later
// accesses through data() before clearing or destroying the owner.
template <DeviceAllocationBackend Backend>
  requires PinnedHostAllocationBackend<Backend> && DeviceCopyBackend<Backend> &&
           DeviceEventBackend<Backend> && DeviceStreamBackend<Backend>
class DeviceFailureFlag final {
  using Traits = BackendTraits<Backend>;
  enum class State { idle, producing, captured, checked };

public:
  DeviceFailureFlag() noexcept = default;
  LPFX_NONMOVABLE(DeviceFailureFlag);

  ~DeviceFailureFlag() noexcept {
    // A failed capture can have submitted its copy without recording the
    // event. It remains producing so teardown protects both allocations.
    if (state_ == State::captured && !Traits::succeeded(ready_.synchronize()))
      std::abort();
    if (state_ == State::producing &&
        !Traits::succeeded(Traits::synchronize_stream(producer_.stream())))
      std::abort();
  }

  // Setup only: failed creation leaves the empty output unchanged.
  [[nodiscard]] static ErrorCode create(DeviceFailureFlag &output) noexcept {
    if (!output.empty())
      return ErrorCode::invalid_state;
    DeviceAllocation<Backend> device;
    PinnedHostAllocation<Backend> host;
    DeviceEvent<Backend> ready;
    if (!Traits::succeeded(decltype(device)::create(sizeof(unsigned), device)) ||
        !Traits::succeeded(decltype(host)::create(sizeof(unsigned), host)))
      return ErrorCode::allocation_failed;
    if (!Traits::succeeded(decltype(ready)::create(ready)))
      return ErrorCode::backend_failure;
    std::construct_at(reinterpret_cast<unsigned *>(host.raw_base()), 0U);
    output.device_ = std::move(device);
    output.host_ = std::move(host);
    output.ready_ = std::move(ready);
    return ErrorCode::success;
  }

  [[nodiscard]] bool empty() const noexcept { return device_.empty(); }
  [[nodiscard]] unsigned *data() noexcept {
    return reinterpret_cast<unsigned *>(device_.raw_base());
  }

  [[nodiscard]] ErrorCode clear(Executor<Backend> executor) noexcept {
    if (empty() || (state_ != State::idle && state_ != State::checked))
      return ErrorCode::invalid_state;
    if (!executor.bound())
      return ErrorCode::invalid_argument;
    producer_ = executor;
    state_ = State::producing;
    return to_error_code<Backend>(Traits::memset_device_async(
        data(), 0, sizeof(unsigned), executor.stream()));
  }

  // Capture before queuing unrelated later work on this producer stream.
  [[nodiscard]] ErrorCode capture(Executor<Backend> executor) noexcept {
    if (state_ != State::producing)
      return ErrorCode::invalid_state;
    if (!executor.bound() || executor.stream() != producer_.stream())
      return ErrorCode::invalid_argument;
    if (!Traits::succeeded(Traits::copy_device_to_host_async(
            host_.raw_base(), data(), sizeof(unsigned), executor.stream())) ||
        !Traits::succeeded(ready_.record(executor)))
      return ErrorCode::backend_failure;
    state_ = State::captured;
    return ErrorCode::success;
  }

  // Pending and failed reads leave value unchanged and retain cleanup duty.
  [[nodiscard]] Status try_read(unsigned &value) noexcept {
    return finish_read<false>(value);
  }
  [[nodiscard]] Status read(unsigned &value) noexcept {
    return finish_read<true>(value);
  }
  [[nodiscard]] ErrorCode wait_on(Executor<Backend> executor) const noexcept {
    if (state_ != State::captured && state_ != State::checked)
      return ErrorCode::invalid_state;
    if (!executor.bound())
      return ErrorCode::invalid_argument;
    return to_error_code<Backend>(ready_.wait_on(executor));
  }

private:
  template <bool Wait> Status finish_read(unsigned &value) noexcept {
    if (state_ != State::captured)
      return Status::failure(ErrorCode::invalid_state);
    const Status observed = ready_.template advance<Wait>();
    if (!observed.is_success())
      return observed;
    value = *reinterpret_cast<const unsigned *>(host_.raw_base());
    state_ = State::checked;
    return Status::success();
  }

  DeviceAllocation<Backend> device_;
  PinnedHostAllocation<Backend> host_;
  DeviceEvent<Backend> ready_;
  Executor<Backend> producer_{Traits::null_stream};
  State state_{State::idle};
};

} // namespace lpfx::device

#endif
