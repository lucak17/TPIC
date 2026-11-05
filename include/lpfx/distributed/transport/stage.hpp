#ifndef LPFX_DISTRIBUTED_TRANSPORT_STAGE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_STAGE_HPP

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed {

// Direct-MPI ordering policy. The execution path proves prior GPU writes to
// read and all prior accesses to write complete before invoking
// before_mpi_access; this policy supplies the remaining access/visibility
// contract and never waits for producers. The conservative receive-side
// stream synchronization remains explicit.
struct StreamSynchronizedMpiOrdering final {
  template <device::DeviceStreamBackend Backend, class ReadView,
            class WriteView>
    requires requires(const ReadView &read, const WriteView &write) {
      { read.size() } noexcept -> std::convertible_to<std::size_t>;
      { write.size() } noexcept -> std::convertible_to<std::size_t>;
    }
  [[nodiscard]] static ErrorCode before_mpi_access(
      const device::Executor<Backend> &, const ReadView &,
      const WriteView &) noexcept {
    return ErrorCode::success;
  }

  template <device::DeviceStreamBackend Backend, class WriteView>
    requires requires(const WriteView &write) {
      { write.size() } noexcept -> std::convertible_to<std::size_t>;
    }
  [[nodiscard]] static ErrorCode
  after_mpi_write(const device::Executor<Backend> executor,
                  const WriteView &write) noexcept {
    return after_mpi_write(executor, write.size() != 0);
  }

  // Prepared fixed messages already know whether native receives wrote any
  // bytes; their declared image can also contain unused receive capacity.
  template <device::DeviceStreamBackend Backend>
  [[nodiscard]] static ErrorCode
  after_mpi_write(const device::Executor<Backend> executor,
                  const bool has_write) noexcept {
    if (!has_write) {
      return ErrorCode::success;
    }
    if (!executor.bound()) {
      return ErrorCode::invalid_argument;
    }
    return device::to_error_code<Backend>(
        device::BackendTraits<Backend>::synchronize_stream(executor.stream()));
  }
};

// Direct-MPI ordering without the receive-side synchronization: an epoch
// completes with its MPI requests; work enqueued later on the binding
// executor, including a field apply, is ordered after it, and other streams
// wait on an event recorded there.
struct StreamOrderedMpiOrdering final {
  [[nodiscard]] static ErrorCode before_mpi_access(const auto &, const auto &,
                                                   const auto &) noexcept {
    return ErrorCode::success;
  }
};

namespace detail {

template <class Ordering, class Executor, class ReadView, class WriteView>
concept DirectMpiOrdering = std::is_empty_v<Ordering> &&
    requires(const Executor &executor, const ReadView &read,
             const WriteView &write) {
      { Ordering::before_mpi_access(executor, read, write) }
          noexcept -> std::same_as<ErrorCode>;
    };

// The optional post-write step of an ordering, run after native receives
// wrote bytes; without it the receive leaves no post-native work.
template <class Ordering, class Executor, class WriteView>
concept MpiPostWriteOrdering =
    requires(const Executor &executor, const WriteView &write) {
      Ordering::after_mpi_write(executor, write);
    };

template <class Backend, class Byte>
[[nodiscard]] constexpr BufferView<Byte, DeviceSpace<Backend>>
device_byte_view(Byte *const data, const std::size_t size_bytes) noexcept {
  return lpfx::detail::make_buffer_view<Byte, DeviceSpace<Backend>>(data,
                                                                    size_bytes);
}

// One native observation of the last record() of event; failure is fatal. The
// status is built from native results directly: a recoverable Status
// intermediary adds stack traffic to enclosing wait paths with GCC/NVCC.
template <bool Wait, class Backend>
  requires device::DeviceEventBackend<Backend>
[[nodiscard]] LPFX_ALWAYS_INLINE Status
advance_or_abort(const device::DeviceEvent<Backend> &event) noexcept {
  if constexpr (Wait) {
    device::abort_on_failure<Backend>(event.synchronize());
  } else {
    const auto result = event.query();
    if (device::BackendTraits<Backend>::not_ready(result))
      return Status::pending();
    device::abort_on_failure<Backend>(result);
  }
  return Status::success();
}

// Pinned host images of a staged route (outgoing/input first,
// incoming/output second) and the event that completes their copies. Both
// images are admitted against the budget; the pinned bytes are freed before
// the lease is returned.
template <class Backend>
  requires device::PinnedHostAllocationBackend<Backend> &&
           device::DeviceEventBackend<Backend>
class PinnedStagingPair final {
  using Traits = device::BackendTraits<Backend>;
  using Allocation = device::PinnedHostAllocation<Backend>;

public:
  // Failure leaves the pair empty.
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &budget,
                                      const std::size_t first_bytes,
                                      const std::size_t second_bytes,
                                      const std::size_t alignment) noexcept {
    HostMemoryRequirement requirement{};
    if (!requirement.try_add(first_bytes, alignment) ||
        !requirement.try_add(second_bytes, alignment)) {
      return ErrorCode::overflow;
    }
    auto admission = budget.try_admit(requirement);
    LPFX_TRY(admission);
    Allocation first{};
    Allocation second{};
    if (!Traits::succeeded(Allocation::create(first_bytes, first)) ||
        !Traits::succeeded(Allocation::create(second_bytes, second))) {
      return ErrorCode::allocation_failed;
    }
    device::DeviceEvent<Backend> event{};
    if (!Traits::succeeded(device::DeviceEvent<Backend>::create(event))) {
      return ErrorCode::backend_failure;
    }
    admission_ = std::move(admission).value();
    first_ = std::move(first);
    second_ = std::move(second);
    event_ = std::move(event);
    return ErrorCode::success;
  }

  [[nodiscard]] std::byte *first() const noexcept {
    return const_cast<std::byte *>(first_.raw_base());
  }
  [[nodiscard]] std::byte *second() const noexcept {
    return const_cast<std::byte *>(second_.raw_base());
  }
  [[nodiscard]] std::size_t first_bytes() const noexcept {
    return first_.capacity_bytes();
  }
  [[nodiscard]] std::size_t second_bytes() const noexcept {
    return second_.capacity_bytes();
  }
  [[nodiscard]] device::DeviceEvent<Backend> &event() noexcept {
    return event_;
  }
  [[nodiscard]] bool admitted() const noexcept { return admission_.active(); }
  [[nodiscard]] bool empty() const noexcept {
    return !admission_.active() && first_.empty() && second_.empty() &&
           event_.empty();
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return first_.capacity_bytes() + second_.capacity_bytes();
  }

  template <bool Wait> [[nodiscard]] Status advance() const noexcept {
    return advance_or_abort<Wait>(event_);
  }

  void reset() noexcept {
    event_ = device::DeviceEvent<Backend>{};
    second_ = Allocation{};
    first_ = Allocation{};
    admission_ = HostMemoryLease{};
  }

private:
  HostMemoryLease admission_{};
  Allocation first_{};
  Allocation second_{};
  device::DeviceEvent<Backend> event_{};
};

// Completion of native work that the library enqueues on the executor's
// stream (XCCL): one event recorded after the work and observed by the host.
// The fixed and collective event stages derive from it.
template <class Backend>
  requires device::DeviceEventBackend<Backend>
class EventCompletion {
public:
  static constexpr bool has_post_native = true;

  [[nodiscard]] bool empty() const noexcept { return completion_.empty(); }
  // Records the completion after the native work enqueued on executor. The
  // fixed family's hook also passes whether receives wrote bytes; the event
  // covers both cases.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_post(const device::Executor<Backend> &executor, bool = false) noexcept {
    device::abort_on_failure<Backend>(completion_.record(executor));
    return Status::pending();
  }
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance_post() const noexcept {
    return advance_or_abort<Wait>(completion_);
  }
  void reset() noexcept { completion_ = {}; }

protected:
  [[nodiscard]] ErrorCode create_completion() noexcept {
    return device::to_error_code<Backend>(
        device::DeviceEvent<Backend>::create(completion_));
  }

private:
  device::DeviceEvent<Backend> completion_{};
};

} // namespace detail

} // namespace lpfx::distributed

#endif
