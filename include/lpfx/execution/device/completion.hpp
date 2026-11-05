#ifndef LPFX_EXECUTION_DEVICE_COMPLETION_HPP
#define LPFX_EXECUTION_DEVICE_COMPLETION_HPP

#include <cstddef>
#include <cstdlib>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx::detail {

enum class DeviceCopyDirection { to_host, to_device };

// Submit an accepted transfer: native failure is fatal and a failed copy never
// records an event. The caller proves buffer lifetime and byte extents.
// Borrow the executor so its stream need not be spilled across the native copy.
template <DeviceCopyDirection Direction, class Backend>
  requires device::DeviceCopyBackend<Backend> &&
           device::DeviceEventBackend<Backend>
LPFX_ALWAYS_INLINE void
copy_and_record(void *destination, const void *source, const std::size_t bytes,
                const device::Executor<Backend> &executor,
                device::DeviceEvent<Backend> &event) noexcept {
  using Traits = device::BackendTraits<Backend>;
  const auto copied = [&] {
    if constexpr (Direction == DeviceCopyDirection::to_host)
      return Traits::copy_device_to_host_async(destination, source, bytes,
                                               executor.stream());
    else
      return Traits::copy_host_to_device_async(destination, source, bytes,
                                               executor.stream());
  }();
  if (!Traits::succeeded(copied) || !Traits::succeeded(event.record(executor)))
    std::abort();
}

// Pinned host bytes plus the event that marks their device-to-host readback.
// A zero-byte readback owns nothing. Destruction frees the pinned bytes before
// returning the budget lease and requires the readback to have quiesced.
template <class Backend>
  requires device::PinnedHostAllocationBackend<Backend> &&
           device::DeviceCopyBackend<Backend> &&
           device::DeviceEventBackend<Backend>
class PinnedReadback final {
public:
  // Admits bytes against budget when it is non-null, then allocates the pinned
  // bytes (allocation_failed) and the event (backend_failure). Returns
  // Result<PinnedReadback>, deduced because Result needs a complete type.
  [[nodiscard]] static auto
  create(HostMemoryBudget *const budget, const std::size_t bytes) noexcept {
    using Traits = device::BackendTraits<Backend>;
    PinnedReadback result;
    if (bytes == 0) {
      return Result<PinnedReadback>::from_value(std::move(result));
    }
    if (budget != nullptr) {
      HostMemoryRequirement requirement{};
      static_cast<void>(requirement.try_add(bytes, alignof(std::max_align_t)));
      auto admission = budget->try_admit(requirement);
      LPFX_TRY_RESULT(admission, Result<PinnedReadback>);
      result.lease_ = std::move(admission).value();
    }
    if (!Traits::succeeded(device::PinnedHostAllocation<Backend>::create(
            bytes, result.bytes_))) {
      return Result<PinnedReadback>::from_error(ErrorCode::allocation_failed);
    }
    if (!Traits::succeeded(device::DeviceEvent<Backend>::create(result.event_))) {
      return Result<PinnedReadback>::from_error(ErrorCode::backend_failure);
    }
    return Result<PinnedReadback>::from_value(std::move(result));
  }

  // Copies bytes (at most the created size) from device_source and records
  // the event. Native failure aborts.
  LPFX_ALWAYS_INLINE void
  enqueue(const void *const device_source, const std::size_t bytes,
          const device::Executor<Backend> &executor) noexcept {
    copy_and_record<DeviceCopyDirection::to_host>(bytes_.raw_base(),
                                                  device_source, bytes,
                                                  executor, event_);
  }

  // See DeviceEvent::advance. The bytes are readable after success.
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() const noexcept {
    return event_.template advance<Wait>();
  }

  [[nodiscard]] std::byte *data() noexcept { return bytes_.raw_base(); }
  [[nodiscard]] const std::byte *data() const noexcept {
    return bytes_.raw_base();
  }

private:
  HostMemoryLease lease_{};
  device::PinnedHostAllocation<Backend> bytes_{};
  device::DeviceEvent<Backend> event_{};
};

} // namespace lpfx::detail

#endif
