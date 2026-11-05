#ifndef LPFX_API_DISTRIBUTED_REQUESTS_HPP
#define LPFX_API_DISTRIBUTED_REQUESTS_HPP

#include <lpfx/api/distributed/key.hpp>
#include <lpfx/api/field/field.hpp>
#include <lpfx/data/field/wire_traits.hpp>
#include <lpfx/distributed/binding/field.hpp>
#include <lpfx/distributed/binding/collective.hpp>
#include <lpfx/distributed/binding/message.hpp>
#include <lpfx/distributed/field/owner.hpp>

namespace lpfx::distributed {

template <class Route, class Executor> struct Endpoint final {
  using route_type = Route;
  Executor executor;
};
template <class Route, class Executor>
[[nodiscard]] constexpr auto on(Executor executor) noexcept {
  return Endpoint<Route, Executor>{std::move(executor)};
}

struct UniformHalos final {};
inline constexpr UniformHalos uniform_halos{};

template <class Neighborhood, class Route, class Executor>
struct FieldEndpoint final {
  using neighborhood_type = Neighborhood;
  using route_type = Route;
  const Neighborhood *neighborhood;
  Executor executor;
};
template <class Neighborhood> struct FieldScope final {
  const Neighborhood *neighborhood;
  template <class Route, class Executor>
  [[nodiscard]] constexpr auto on(Executor executor) const noexcept {
    return FieldEndpoint<Neighborhood, Route, Executor>{neighborhood,
                                                       std::move(executor)};
  }
};
template <class Neighborhood>
[[nodiscard]] constexpr auto fields(const Neighborhood &neighborhood,
                                     UniformHalos) noexcept {
  return FieldScope<Neighborhood>{&neighborhood};
}
template <class Neighborhood> void fields(const Neighborhood &&, UniformHalos) = delete;

namespace detail {

template <class Owner, class Geometry, std::size_t... I>
constexpr void inferred_field_halos(const Owner &owner, Geometry &geometry,
                                    std::index_sequence<I...>) noexcept {
  if constexpr (requires { geometry.component_halos; })
    ((geometry.component_halos[I] = owner.template component_halo<I>()), ...);
}

template <class T> inline constexpr bool borrowed_field_selection = false;
template <class Owner, class... Components>
inline constexpr bool borrowed_field_selection<
    field::basic_selection<Owner, Components...>> = true;

template <class Owner, bool Borrowed = borrowed_field_selection<Owner>>
struct RequestResource final {
  Owner *owner;
  constexpr explicit RequestResource(Owner &value) noexcept : owner(&value) {}
  [[nodiscard]] constexpr Owner &get() noexcept { return *owner; }
};
template <class Owner> struct RequestResource<Owner, true> final {
  Owner owner;
  constexpr explicit RequestResource(Owner value) noexcept : owner(value) {}
  [[nodiscard]] constexpr Owner &get() noexcept { return owner; }
};

template <field_operation_kind Kind, class Components> struct InferredFieldSpec;
template <field_operation_kind Kind, class... Components>
struct InferredFieldSpec<Kind, std::tuple<Components...>> {
  using type = FieldOperationSpec<Kind, all_directions_3d,
      FieldCopyComponent<Components,
          operation_key<Components::tag_type::name>::identity,
          field::wire_traits<typename Components::value_type>::identity>...>;
};

template <class Key, field_operation_kind Kind, class Owner, class Placement>
struct FieldRequest final {
  static constexpr bool collective_preparation = true;
  using key_type = Key;
  using neighborhood_type = typename Placement::neighborhood_type;
  using definition_type = FieldPlan<Key::identity,
      typename InferredFieldSpec<Kind,
          lpfx::detail::mutable_field_owner_components_t<Owner>>::type>;
  using plan_type = PreparedFieldPlan<definition_type, typename Owner::index_policy,
                                      typename neighborhood_type::rank_type>;
  using binding_type = decltype(bind<typename Placement::route_type>(
      std::declval<const plan_type &>(), std::declval<Owner &>(),
      std::declval<Placement>().executor));
  // Catalog agreement must cover the size of the following halo collective,
  // before ranks with different component packs enter family preparation.
  static constexpr std::uint64_t preparation_shape = definition_type::spec_type::component_count;
  RequestResource<Owner> resource;
  Placement placement;

  template <class Setup>
  [[nodiscard]] ErrorCode prepare_plan(const Setup &setup, plan_type &plan) noexcept {
    auto &owner = resource.get();
    const bool local_valid = owner.valid() && owner.initialized() &&
        placement.neighborhood->is_valid() &&
        owner.cells() == placement.neighborhood->local_partition().local_cells();
    constexpr auto count = definition_type::spec_type::component_count;
    std::array<std::uint64_t, count * 6> halos{};
    if (local_valid) [&]<std::size_t... I>(std::index_sequence<I...>) {
      const auto store = [&]<std::size_t C>() {
        const auto h = owner.template component_halo<C>();
        const std::array<std::uint64_t, 6> values{
            h.x_minus, h.x_plus, h.y_minus, h.y_plus, h.z_minus, h.z_plus};
        for (std::size_t i = 0; i != 6; ++i) halos[6 * C + i] = values[i];
      };
      (store.template operator()<I>(), ...);
    }(std::make_index_sequence<count>{});
    // All ranks take the same path, even when local storage is invalid.
    auto error = ObjectPreparation<Setup>::agree_words(setup, halos);
    if (error != ErrorCode::success) return error;
    if (!local_valid) return ErrorCode::invalid_argument;
    const auto geometry = [&](const auto &partition,
                              typename plan_type::geometry_type &output) noexcept {
      output.cell_extents = partition.local_cells();
      inferred_field_halos(owner, output, std::make_index_sequence<count>{});
      return ErrorCode::success;
    };
    return try_prepare_field_plan(definition_type{}, *placement.neighborhood, geometry, plan);
  }
  [[nodiscard]] auto binding(plan_type &plan) noexcept {
    return bind<typename Placement::route_type>(plan, resource.get(), placement.executor);
  }
};

template <class Key, class Operand, class Descriptor, class Placement>
struct CollectiveRequest final {
  static constexpr bool collective_preparation = false;
  using key_type = Key;
  static constexpr std::uint64_t preparation_shape = 0;
  using definition_type = CollectivePlan<Key::identity, typename Operand::value_type, Descriptor>;
  using plan_type = PreparedCollectivePlan<definition_type>;
  using binding_type = decltype(bind<typename Placement::route_type>(
      std::declval<const plan_type &>(), std::declval<Operand &>(),
      std::declval<Placement>().executor));
  Operand operand;
  Placement placement;
  template <class Setup>
  [[nodiscard]] ErrorCode prepare_plan(const Setup &setup, plan_type &plan) noexcept {
    if (!operand.valid()) return ErrorCode::invalid_argument;
    return try_prepare_collective_plan(definition_type{}, operand.input_count(),
        operand.output_count(), setup.size(), plan);
  }
  [[nodiscard]] auto binding(plan_type &plan) noexcept {
    return bind<typename Placement::route_type>(plan, operand, placement.executor);
  }
};

template <class Key, class Plan, class Resource, class Placement>
struct PreparedRequest final {
  static constexpr bool collective_preparation = false;
  using key_type = Key;
  static constexpr std::uint64_t preparation_shape = 0;
  using definition_type = typename Plan::definition_type;
  // The bridge borrows an already prepared plan instead of retaining a copy.
  using plan_type = const Plan *;
  using binding_type = decltype(bind<typename Placement::route_type>(
      std::declval<const Plan &>(), std::declval<Resource &>(),
      std::declval<Placement>().executor));
  const Plan *plan;
  Resource *resource;
  Placement placement;
  template <class Setup>
  [[nodiscard]] ErrorCode prepare_plan(const Setup &, plan_type &output) noexcept {
    output = plan;
    return plan->is_valid() ? ErrorCode::success : ErrorCode::invalid_argument;
  }
  [[nodiscard]] auto binding(plan_type &value) noexcept {
    return bind<typename Placement::route_type>(*value, *resource, placement.executor);
  }
};

template <class Key, field_operation_kind Kind, class Owner, class Placement>
[[nodiscard]] constexpr auto field_request(Owner &&owner, Placement placement) noexcept {
  using O = std::remove_cvref_t<Owner>;
  return FieldRequest<Key, Kind, O, Placement>{
      RequestResource<O>{std::forward<Owner>(owner)}, std::move(placement)};
}
} // namespace detail

#define LPFX_FIELD_REQUEST(Name, Kind) \
  template <field::fixed_string Key, std::uint64_t... Index, class Owner, class Placement> \
    requires lpfx::detail::MutableFieldOwner3D<std::remove_cvref_t<Owner>> && \
      (std::is_lvalue_reference_v<Owner &&> || \
       detail::borrowed_field_selection<std::remove_cvref_t<Owner>>) \
  [[nodiscard]] constexpr auto Name(Owner &&owner, Placement placement) noexcept { \
    return detail::field_request<detail::operation_key<Key, Index...>, \
        field_operation_kind::Kind>(std::forward<Owner>(owner), std::move(placement)); \
  }
LPFX_FIELD_REQUEST(copy_ghosts, copy_to_ghost)
LPFX_FIELD_REQUEST(sum_shared, shared_sample_add)
#undef LPFX_FIELD_REQUEST

template <reduction_operation Operation> struct Reduction final {};
inline constexpr Reduction<reduction_operation::sum> sum{};
inline constexpr Reduction<reduction_operation::minimum> minimum{};
inline constexpr Reduction<reduction_operation::maximum> maximum{};
inline constexpr Reduction<reduction_operation::product> product{};

template <field::fixed_string Key, std::uint64_t... Index, detail::CollectiveOperand Operand,
          reduction_operation Operation, class Placement>
[[nodiscard]] constexpr auto all_reduce(Operand operand, Reduction<Operation>,
                                        Placement placement) noexcept {
  return detail::CollectiveRequest<detail::operation_key<Key, Index...>, Operand,
      all_reduce_descriptor<Operation>, Placement>{operand, std::move(placement)};
}
template <field::fixed_string Key, std::uint64_t... Index, class Plan, class Resource,
          class Placement>
[[nodiscard]] constexpr auto prepared(const Plan &plan, Resource &resource,
                                      Placement placement) noexcept {
  return detail::PreparedRequest<detail::operation_key<Key, Index...>, Plan, Resource,
      Placement>{&plan, &resource, std::move(placement)};
}
template <field::fixed_string Key, std::uint64_t... Index, class Plan, class Resource,
          class Placement>
void prepared(const Plan &&, Resource &, Placement) = delete;

} // namespace lpfx::distributed
#endif
