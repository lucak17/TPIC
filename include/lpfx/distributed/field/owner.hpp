#ifndef LPFX_DISTRIBUTED_FIELD_OWNER_HPP
#define LPFX_DISTRIBUTED_FIELD_OWNER_HPP

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/data/field/view.hpp>
#include <lpfx/distributed/descriptor/field.hpp>

namespace lpfx::detail {

// Communication consumes a field's structural interface rather than its
// owner class, so a zero-copy selection can present exactly the components a
// contract names while the original owner keeps the allocation.
template <class View> struct MutableFieldPackViewTraits final {
  static constexpr bool recognized = false;
};

template <CartesianIndexPolicy IndexPolicy, class Space,
          FieldComponentType... Components>
struct MutableFieldPackViewTraits<
    FieldPackView3D<IndexPolicy, Space, false, Components...>>
    final {
  static constexpr bool recognized = true;
  using index_policy = IndexPolicy;
  using space_type = Space;
  using component_tuple = std::tuple<Components...>;
};

template <class Owner, class = void> struct MutableFieldOwnerTraits final {
  static constexpr bool recognized = false;
};

template <class Owner>
struct MutableFieldOwnerTraits<
    Owner, std::void_t<typename Owner::index_policy, typename Owner::space_type,
                       decltype(std::declval<Owner &>().view())>>
    final {
  using view_type =
      std::remove_cvref_t<decltype(std::declval<Owner &>().view())>;
  using view_traits = MutableFieldPackViewTraits<view_type>;
  static constexpr bool recognized = [] {
    if constexpr (view_traits::recognized) {
      return std::same_as<typename Owner::index_policy,
                          typename view_traits::index_policy> &&
             std::same_as<typename Owner::space_type,
                          typename view_traits::space_type>;
    } else {
      return false;
    }
  }();
};

template <class Owner>
using mutable_field_owner_view_t =
    typename MutableFieldOwnerTraits<Owner>::view_type;

template <class Owner>
using mutable_field_owner_components_t =
    typename MutableFieldOwnerTraits<Owner>::view_traits::component_tuple;

template <class Owner, std::size_t... I>
[[nodiscard]] consteval bool
field_owner_halos_are_readable(std::index_sequence<I...>) noexcept {
  using Halo = HaloDepth3<typename Owner::index_policy::count_type>;
  return (requires(const Owner &owner) {
    { owner.template component_halo<I>() } noexcept -> std::same_as<Halo>;
  } && ...);
}

template <class Owner>
concept MutableFieldOwner3D =
    std::same_as<Owner, std::remove_cvref_t<Owner>> &&
    MutableFieldOwnerTraits<Owner>::recognized &&
    requires(Owner &owner, const Owner &constant) {
      { constant.valid() } noexcept -> std::same_as<bool>;
      { constant.initialized() } noexcept -> std::same_as<bool>;
      { owner.view() } noexcept -> std::same_as<mutable_field_owner_view_t<Owner>>;
      {
        constant.cells()
      } noexcept
          -> std::same_as<CartesianExtent3<typename Owner::index_policy::count_type>>;
    } &&
    field_owner_halos_are_readable<Owner>(std::make_index_sequence<
        std::tuple_size_v<mutable_field_owner_components_t<Owner>>>{});

template <class Owner, class Spec, class IndexPolicy>
concept FieldOwnerMatchesOperationSpec =
    MutableFieldOwner3D<Owner> &&
    std::same_as<typename Owner::index_policy, IndexPolicy> &&
    std::same_as<mutable_field_owner_components_t<Owner>,
                 typename Spec::field_components>;

} // namespace lpfx::detail

#endif
