#ifndef LPFX_DISTRIBUTED_COMMON_EXECUTOR_HPP
#define LPFX_DISTRIBUTED_COMMON_EXECUTOR_HPP

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/execution/host/serial.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed::detail {

template <class Space, class Executor>
inline constexpr bool communication_executor_matches_space = false;

template <>
inline constexpr bool
    communication_executor_matches_space<HostSpace, SerialExecutor> = true;

template <device::DeviceBackend Backend>
inline constexpr bool communication_executor_matches_space<
    DeviceSpace<Backend>, device::Executor<Backend>> = true;

template <class Space, class Executor>
concept CommunicationExecutorForSpace =
    communication_executor_matches_space<Space, Executor>;

template <class Space, class Executor>
  requires CommunicationExecutorForSpace<Space, Executor>
[[nodiscard]] constexpr bool
communication_executor_valid(const Executor &executor) noexcept {
  if constexpr (std::same_as<Space, HostSpace>) {
    return true;
  } else {
    return executor.bound();
  }
}

// The executor a program takes from its binding, held at its final address.
// Host executors are stateless; device executors are not default
// constructible and occupy one optional slot.
template <class Space, class Executor>
  requires CommunicationExecutorForSpace<Space, Executor>
class ExecutorSlot final {
  static constexpr bool host = std::same_as<Space, HostSpace>;

public:
  void emplace(Executor executor) noexcept {
    if constexpr (host) {
      value_ = std::move(executor);
    } else {
      value_.emplace(std::move(executor));
    }
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE const Executor &get() const noexcept {
    if constexpr (host) {
      return value_;
    } else {
      return *value_;
    }
  }

  void reset() noexcept {
    if constexpr (!host) {
      value_.reset();
    }
  }

private:
  [[no_unique_address]] std::conditional_t<host, Executor,
                                           std::optional<Executor>> value_{};
};

} // namespace lpfx::distributed::detail

#endif
