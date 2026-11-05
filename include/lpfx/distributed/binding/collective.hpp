#ifndef LPFX_DISTRIBUTED_BINDING_COLLECTIVE_HPP
#define LPFX_DISTRIBUTED_BINDING_COLLECTIVE_HPP

#include <concepts>

#include <lpfx/distributed/binding/bound_operation.hpp>
#include <lpfx/distributed/collective_operand.hpp>
#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/distributed/plan/collective.hpp>

namespace lpfx::distributed {

namespace detail {

template <class Definition, CollectiveOperand Operand, class Executor>
struct BindingFamily<PreparedCollectivePlan<Definition>, Operand, Executor,
                     NoSemantics> {
  static constexpr bool compatible =
      std::same_as<typename Definition::value_type,
                   typename Operand::value_type> &&
      CommunicationExecutorForSpace<typename Operand::space_type, Executor>;

  [[nodiscard]] static constexpr bool valid(const Operand &operand) noexcept {
    return operand.valid();
  }
};

} // namespace detail

template <class Transport, class Plan, class Operand, class Executor>
using CollectiveBinding = BoundOperation<Transport, Plan, Operand, Executor>;

} // namespace lpfx::distributed

#endif
