#ifndef LPFX_DISTRIBUTED_BINDING_MESSAGE_HPP
#define LPFX_DISTRIBUTED_BINDING_MESSAGE_HPP

#include <cstdint>

#include <lpfx/distributed/binding/bound_operation.hpp>
#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/distributed/message/buffers.hpp>
#include <lpfx/distributed/plan/message.hpp>

namespace lpfx::distributed {

namespace detail {

template <std::uint64_t ChannelIdentity, class Descriptor, class Rank,
          class Space, class Executor>
struct BindingFamily<
    PreparedMessagePlan<MessagePlan<ChannelIdentity, Descriptor>, Rank>,
    FixedMessageBuffers<Space>, Executor, NoSemantics> {
  static constexpr bool compatible =
      CommunicationExecutorForSpace<Space, Executor>;

  [[nodiscard]] static constexpr bool
  valid(const FixedMessageBuffers<Space> &buffers) noexcept {
    return buffers.valid();
  }
};

} // namespace detail

template <class Transport, class Plan, class Buffers, class Executor>
using MessageBinding = BoundOperation<Transport, Plan, Buffers, Executor>;

} // namespace lpfx::distributed

#endif
