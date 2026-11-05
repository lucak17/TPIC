#ifndef LPFX_DISTRIBUTED_BINDING_FIELD_HPP
#define LPFX_DISTRIBUTED_BINDING_FIELD_HPP

#include <lpfx/distributed/binding/bound_operation.hpp>
#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/distributed/field/owner.hpp>
#include <lpfx/distributed/plan/field.hpp>

namespace lpfx::distributed {

namespace detail {

template <class Definition, class IndexPolicy, class Rank, class Owner,
          class Executor>
struct BindingFamily<PreparedFieldPlan<Definition, IndexPolicy, Rank>, Owner,
                     Executor, NoSemantics> {
  static constexpr bool compatible = [] {
    if constexpr (lpfx::detail::FieldOwnerMatchesOperationSpec<
                      Owner,
                      typename PreparedFieldPlan<Definition, IndexPolicy,
                                                 Rank>::spec_type,
                      IndexPolicy>) {
      return CommunicationExecutorForSpace<typename Owner::space_type,
                                           Executor>;
    } else {
      return false;
    }
  }();

  [[nodiscard]] static constexpr bool valid(const Owner &owner) noexcept {
    return owner.valid() && owner.initialized();
  }
};

} // namespace detail

template <class Transport, class Plan, class Owner, class Executor>
using FieldBinding = BoundOperation<Transport, Plan, Owner, Executor>;

} // namespace lpfx::distributed

#endif
