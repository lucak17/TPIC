#ifndef LPFX_DISTRIBUTED_DETAIL_FIXED_EXCHANGE_PROGRAM_HPP
#define LPFX_DISTRIBUTED_DETAIL_FIXED_EXCHANGE_PROGRAM_HPP

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <utility>

#include <lpfx/backend/launch.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/distributed/detail/fixed_byte.hpp>
#include <lpfx/distributed/field/submission.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

struct PreparedOperationProgramAccess;

enum class FixedExchangePhase : std::uint8_t {
  empty,
  ready,
  pre_native,
  native,
  post_native,
  apply,
  complete
};

LPFX_ALWAYS_INLINE void
fixed_exchange_require_success(const ErrorCode error) noexcept {
  if (error != ErrorCode::success)
    std::abort();
}
LPFX_ALWAYS_INLINE void
fixed_exchange_require_status(const Status status) noexcept {
  if (status.is_failure() || status.requires_conclusion())
    std::abort();
}

// One fixed-byte exchange per epoch: the execution optionally captures its
// sources, the port exchanges the fixed images, and the execution applies the
// received images after transport success. An Execution provides
//   space_type, has_source, has_apply, asynchronous;
//   static try_validate(setup, binding) and needs_native_work(binding);
//   prepare(budget, setup, binding, executor, emit), where emit consumes the
//     FixedByteExchange of the prepared images;
//   capture() when has_source, apply() when has_apply,
//   record_completion() and advance_apply<Wait>() when asynchronous;
//   reset() -> ErrorCode and workspace_bytes().
// A Port with completes_on_stream completes the exchange with its native work;
// its asynchronous apply then completes at enqueue, ordered on the executor.
template <class Port, class Execution, class Executor>
class FixedExchangeProgram final {
public:
  using port_type = Port;
  using execution_type = Execution;
  using executor_type = Executor;
  using space_type = typename Execution::space_type;

  constexpr FixedExchangeProgram() noexcept = default;
  LPFX_NONMOVABLE(FixedExchangeProgram);

  [[nodiscard]] bool empty() const noexcept {
    return phase_ == FixedExchangePhase::empty && port_.empty();
  }

  [[nodiscard]] bool is_valid() const noexcept {
    return phase_ >= FixedExchangePhase::ready &&
           (native_work_ ? port_.is_valid() : port_.empty());
  }

  template <class Setup, class Binding>
  [[nodiscard]] static ErrorCode try_validate(const Setup &setup,
                                              const Binding &binding) noexcept {
    return Execution::try_validate(setup, binding);
  }

  template <class Binding>
  [[nodiscard]] static bool needs_native_work(const Binding &binding) noexcept {
    return Execution::needs_native_work(binding);
  }

  // Consumes a binding accepted by try_validate; neither the binding nor its
  // plan is retained.
  template <class Session, class Setup, class Binding>
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &budget,
                                      Session &session, const Setup &setup,
                                      Binding &&binding,
                                      const int tag_base) noexcept {
    if (!empty()) {
      return ErrorCode::invalid_state;
    }
    executor_.emplace(std::move(binding).take_executor());
    const ErrorCode error = execution_.prepare(
        budget, setup, binding, executor(),
        [&](const FixedByteExchange &shape) noexcept {
          native_work_ = shape.has_native_work();
          return native_work_
                     ? port_.try_prepare_resources(
                           budget, session, setup, shape, executor(),
                           std::span<const int>{&tag_base, 1})
                     : ErrorCode::success;
        });
    if (error != ErrorCode::success) {
      reset();
      return error;
    }
    phase_ = FixedExchangePhase::ready;
    return ErrorCode::success;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start() noexcept {
    assert(phase_ == FixedExchangePhase::ready);
    if constexpr (Execution::has_source) {
      fixed_exchange_require_success(execution_.capture());
    }
    if constexpr (Execution::has_apply) {
      // Select transport or semantic completion before publishing the phase.
      FixedExchangePhase next = FixedExchangePhase::ready;
      auto result = start_exchange(next);
      if (!result.is_pending())
        result = apply(next);
      phase_ = next;
      return result;
    } else {
      return start_exchange(phase_);
    }
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status release() noexcept {
    assert(phase_ == FixedExchangePhase::complete);
    if constexpr (Port::has_native_completion) {
      if (native_work_) {
        const auto result = port_.release_native();
        fixed_exchange_require_status(result);
        if (result.is_pending())
          return result;
      }
    }
    phase_ = FixedExchangePhase::ready;
    return Status::success();
  }

  // Every remote image is exchanged once per epoch, including by fused
  // operations with several dependent passes.
  [[nodiscard]] FieldSubmission first_native_submission() const noexcept
    requires(Execution::has_apply)
  {
    if (!native_work_)
      return FieldSubmission::no_native_work;
    return phase_ < FixedExchangePhase::native ? FieldSubmission::pending
                                               : FieldSubmission::submitted;
  }

  void release_resources_for_close() noexcept {
    if (phase_ != FixedExchangePhase::ready) {
      std::abort();
    }
    if constexpr (Port::completes_on_stream) {
      // Completed epochs may still have queued work that reads the images.
      using Backend = typename Port::device_backend_type;
      device::abort_on_failure<Backend>(
          device::BackendTraits<Backend>::synchronize_stream(executor().stream()));
    }
    reset();
  }

  void reset() noexcept {
    if (phase_ >= FixedExchangePhase::pre_native) {
      std::abort();
    }
    port_.reset_resources();
    fixed_exchange_require_success(execution_.reset());
    executor_.reset();
    native_work_ = false;
    phase_ = FixedExchangePhase::empty;
  }

  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return execution_.workspace_bytes() + port_.workspace_bytes();
  }

private:
  friend struct PreparedOperationProgramAccess;

  // An opted-in port enqueues its post-native work, if any, on the executor
  // that applies asynchronously; the apply event then covers both, and
  // release still follows that event.
  static constexpr bool post_on_apply_stream =
      Execution::asynchronous &&
      requires(Port &port, const Executor &executor) {
        { port.enqueue_post_on_apply_stream(executor) } noexcept
            -> std::same_as<ErrorCode>;
      };

  [[nodiscard]] LPFX_ALWAYS_INLINE const Executor &executor() const noexcept {
    return executor_.get();
  }

  template <bool Wait> [[nodiscard]] LPFX_ALWAYS_INLINE Status drive() noexcept {
    if (phase_ == FixedExchangePhase::complete)
      return Status::success();
    if constexpr (Execution::asynchronous) {
      if (phase_ == FixedExchangePhase::apply) {
        const auto result = execution_.template advance_apply<Wait>();
        fixed_exchange_require_status(result);
        if (result.is_pending())
          return result;
        phase_ = FixedExchangePhase::complete;
        return Status::success();
      }
    }
    const auto result = drive_exchange<Wait>();
    if constexpr (Execution::has_apply) {
      if (result.is_pending())
        return result;
      return apply(phase_);
    } else {
      return result;
    }
  }

  [[nodiscard]] static LPFX_ALWAYS_INLINE Status
  complete_exchange(FixedExchangePhase &phase) noexcept {
    if constexpr (!Execution::has_apply)
      phase = FixedExchangePhase::complete;
    return Status::success();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status submit() noexcept {
    const auto result = port_.submit(executor());
    fixed_exchange_require_status(result);
    return result;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_post(FixedExchangePhase &phase) noexcept {
    if constexpr (Port::has_post_native) {
      if constexpr (post_on_apply_stream) {
        fixed_exchange_require_success(
            port_.enqueue_post_on_apply_stream(executor()));
      } else {
        const auto result = port_.start_post(executor());
        fixed_exchange_require_status(result);
        if (result.is_pending()) {
          phase = FixedExchangePhase::post_native;
          return result;
        }
      }
    }
    return complete_exchange(phase);
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_exchange(FixedExchangePhase &phase) noexcept {
    if (!native_work_)
      return complete_exchange(phase);
    if constexpr (Port::has_pre_native) {
      const auto result = port_.start_pre(executor());
      fixed_exchange_require_status(result);
      if (result.is_pending()) {
        phase = FixedExchangePhase::pre_native;
        return result;
      }
    }
    const auto result = submit();
    if (result.is_pending()) {
      phase = FixedExchangePhase::native;
      return result;
    }
    return start_post(phase);
  }

  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status drive_exchange() noexcept {
    if constexpr (Port::has_post_native && !post_on_apply_stream) {
      if (phase_ == FixedExchangePhase::post_native) {
        const auto result = port_.template advance_post<Wait>();
        fixed_exchange_require_status(result);
        if (result.is_pending())
          return result;
        return complete_exchange(phase_);
      }
    }
    auto native = Status::pending();
    if constexpr (Port::has_pre_native) {
      if (phase_ == FixedExchangePhase::pre_native) {
        const auto pre = port_.template advance_pre<Wait>(executor());
        fixed_exchange_require_status(pre);
        if (pre.is_pending())
          return pre;
        native = submit();
        if (native.is_pending()) {
          phase_ = FixedExchangePhase::native;
          if constexpr (!Wait)
            return native;
        }
      } else if (phase_ != FixedExchangePhase::native) {
        std::abort();
      }
    } else if (phase_ != FixedExchangePhase::native) {
      std::abort();
    }
    // Poll returns at submission; wait may advance the new native work now.
    if constexpr (Port::has_native_completion) {
      if (native.is_pending()) {
        native = port_.template advance_native<Wait>();
        fixed_exchange_require_status(native);
        if (native.is_pending())
          return native;
      }
    }
    return start_post(phase_);
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  apply(FixedExchangePhase &phase) noexcept {
    execution_.apply();
    if constexpr (Execution::asynchronous && !Port::completes_on_stream) {
      fixed_exchange_require_success(execution_.record_completion());
      phase = FixedExchangePhase::apply;
      return Status::pending();
    } else {
      phase = FixedExchangePhase::complete;
      return Status::success();
    }
  }

  [[no_unique_address]] ExecutorSlot<space_type, Executor> executor_{};
  [[no_unique_address]] Execution execution_{};
  [[no_unique_address]] Port port_{};
  FixedExchangePhase phase_ = FixedExchangePhase::empty;
  bool native_work_ = false;
};

} // namespace lpfx::distributed::detail

#endif
