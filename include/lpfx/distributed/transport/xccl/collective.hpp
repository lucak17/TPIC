#ifndef LPFX_DISTRIBUTED_TRANSPORT_XCCL_COLLECTIVE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_XCCL_COLLECTIVE_HPP

#include <cstdint>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/collective/port.hpp>
#include <lpfx/distributed/transport/xccl/session.hpp>

namespace lpfx::distributed::detail {

// XCCL collective call policy for CollectivePort: borrows the shared session
// and retains no executor, event, semantic operation, or active state.
// Native::submit constrains the supported kind/reduction/scalar combinations.
template <XcclNativeAdapter Native, collective_kind Kind,
          reduction_operation Reduction, class T, bool ExactInPlace>
class XcclCollectiveCall final {
public:
  using value_type = T;
  using session_type = XcclSession<Native>;

  [[nodiscard]] constexpr bool empty() const noexcept {
    return session_ == nullptr;
  }

  [[nodiscard]] static constexpr bool
  valid_recipe(const CollectivePreparedShape &shape, const int rank,
               const int rank_count, const bool require_native_work) noexcept {
    return rank >= 0 && rank < rank_count &&
           shape.rank == static_cast<std::uint32_t>(rank) &&
           shape.rank_count == static_cast<std::uint32_t>(rank_count) &&
           (!require_native_work || shape.has_native_work());
  }

  [[nodiscard]] bool
  is_valid(const CollectivePreparedShape &shape) const noexcept {
    return shape.has_native_work() && session_ != nullptr &&
           session_->ready();
  }

  template <class Session, class Setup, class Executor>
    requires XcclSessionProvider<Session, Native>
  [[nodiscard]] ErrorCode try_prepare(Session &shared_session,
                                      const Setup &setup,
                                      const Executor &executor) noexcept {
    if (!setup.has_device_affinity() || !executor.bound()) {
      return ErrorCode::invalid_state;
    }
    session_type &session = shared_session.xccl();
    if (!session.ready()) {
      return ErrorCode::invalid_state;
    }
    session_ = &session;
    return ErrorCode::success;
  }

  template <std::uint32_t Root, class Operands, class Executor>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  submit(Operands &&operands, const Executor &executor) noexcept {
    const auto communicator = session_->native_handle();
    return session_->submit(executor, [&]() noexcept {
      const auto args = operands();
      return Native::template submit<Kind, Reduction, T, ExactInPlace, Root>(
          args.count, args.input, args.output, communicator, executor.stream());
    });
  }

  void reset() noexcept { session_ = nullptr; }

private:
  session_type *session_ = nullptr;
};

} // namespace lpfx::distributed::detail

#endif
