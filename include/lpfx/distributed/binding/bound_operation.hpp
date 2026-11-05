#ifndef LPFX_DISTRIBUTED_BINDING_BOUND_OPERATION_HPP
#define LPFX_DISTRIBUTED_BINDING_BOUND_OPERATION_HPP

#include <concepts>
#include <type_traits>
#include <utility>

#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/distributed/descriptor/family.hpp>

namespace lpfx::distributed {

namespace detail {

struct NoSemantics final {};

// Each family specializes this for its prepared plan type: compatible admits
// the resource, executor and semantics types; valid(resource) checks the
// borrowed resource at preparation. An optional bindable = false keeps a
// compatible combination away from the generic bind().
template <class Plan, class Resource, class Executor, class Semantics>
struct BindingFamily {
  static constexpr bool compatible = false;
};

template <class Plan, class Resource, class Executor, class Semantics>
concept CompatibleBinding =
    BindingFamily<Plan, Resource, Executor, Semantics>::compatible;

template <class Family>
inline constexpr bool family_bindable = [] {
  if constexpr (requires { Family::bindable; }) {
    return Family::bindable;
  } else {
    return true;
  }
}();

template <class... Semantics> struct BindingSemantics final {
  using type = NoSemantics;
};
template <class Semantics> struct BindingSemantics<Semantics> final {
  using type = Semantics;
};

} // namespace detail

// A binding is a consumed preparation value. It borrows the address-stable
// plan and resource and owns the executor and semantic state that preparation
// moves into the final program. Its transport route is type state only.
template <class Transport, class Plan, class Resource, class Executor,
          class Semantics = detail::NoSemantics>
  requires detail::CompatibleBinding<Plan, Resource, Executor, Semantics>
class BoundOperation final {
public:
  using transport_type = Transport;
  using plan_type = Plan;
  using definition_type = typename Plan::definition_type;
  using resource_type = Resource;
  using executor_type = Executor;
  using semantics_type = Semantics;
  using space_type = typename Resource::space_type;

  static constexpr operation_family family = Plan::descriptor_type::family;

  // Preparation moves both states out of a noexcept path.
  static_assert(std::is_nothrow_move_constructible_v<Executor> &&
                std::is_nothrow_destructible_v<Executor>);
  static_assert(std::is_nothrow_move_constructible_v<Semantics> &&
                std::is_nothrow_destructible_v<Semantics>);

  constexpr BoundOperation(const Plan &plan, Resource &resource,
                           Executor executor, Semantics semantics = {}) noexcept
      : plan_(&plan), resource_(&resource), executor_(std::move(executor)),
        semantics_(std::move(semantics)) {}
  BoundOperation(const Plan &&, Resource &, Executor, Semantics = {}) = delete;

  constexpr BoundOperation(BoundOperation &&) noexcept = default;
  BoundOperation(const BoundOperation &) = delete;
  BoundOperation &operator=(const BoundOperation &) = delete;
  BoundOperation &operator=(BoundOperation &&) = delete;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return plan_->is_valid() &&
           detail::BindingFamily<Plan, Resource, Executor, Semantics>::valid(
               *resource_) &&
           detail::communication_executor_valid<space_type>(executor_);
  }

  [[nodiscard]] constexpr const Plan &plan() const noexcept { return *plan_; }
  [[nodiscard]] constexpr Resource &resource() const noexcept {
    return *resource_;
  }
  [[nodiscard]] constexpr const Executor &executor() const noexcept {
    return executor_;
  }

  // Preparation consumes each state exactly once into the final program.
  [[nodiscard]] constexpr Executor take_executor() && noexcept {
    return std::move(executor_);
  }
  [[nodiscard]] constexpr Semantics &&take_semantics() && noexcept {
    return std::move(semantics_);
  }

private:
  const Plan *plan_;
  Resource *resource_;
  [[no_unique_address]] Executor executor_;
  [[no_unique_address]] Semantics semantics_;
};

// Binds a prepared plan to its resource on a route. Particle bindings pass
// their semantics; other families pass none.
template <class Transport, class Plan, class Resource, class Executor,
          class... Semantics>
  requires(sizeof...(Semantics) <= 1) &&
          detail::CompatibleBinding<
              Plan, Resource, Executor,
              typename detail::BindingSemantics<Semantics...>::type> &&
          detail::family_bindable<detail::BindingFamily<
              Plan, Resource, Executor,
              typename detail::BindingSemantics<Semantics...>::type>>
[[nodiscard]] constexpr auto bind(const Plan &plan, Resource &resource,
                                  Executor executor,
                                  Semantics... semantics) noexcept {
  return BoundOperation<Transport, Plan, Resource, Executor,
                        typename detail::BindingSemantics<Semantics...>::type>{
      plan, resource, std::move(executor), std::move(semantics)...};
}

// Bindings borrow prepared plans; a temporary plan would dangle.
template <class Transport, class Plan, class Resource, class Executor,
          class... Semantics>
  requires(sizeof...(Semantics) <= 1)
void bind(const Plan &&, Resource &, Executor, Semantics...) = delete;

namespace detail {

template <class T> inline constexpr bool is_bound_operation = false;
template <class Transport, class Plan, class Resource, class Executor,
          class Semantics>
inline constexpr bool is_bound_operation<
    BoundOperation<Transport, Plan, Resource, Executor, Semantics>> = true;

template <class T, operation_family Family>
concept FamilyBinding = is_bound_operation<std::remove_cvref_t<T>> &&
                        std::remove_cvref_t<T>::family == Family;

template <class T>
concept DistributedCollectiveBinding =
    FamilyBinding<T, operation_family::collective>;
template <class T>
concept DistributedFieldBinding = FamilyBinding<T, operation_family::field>;
template <class T>
concept DistributedMessageBinding = FamilyBinding<T, operation_family::message>;
template <class T>
concept DistributedParticleBinding =
    FamilyBinding<T, operation_family::particle>;

} // namespace detail
} // namespace lpfx::distributed

#endif
