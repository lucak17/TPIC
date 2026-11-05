#ifndef TPIC_ECSIM_FIELDS_FIELD_CHECKS_HPP
#define TPIC_ECSIM_FIELDS_FIELD_CHECKS_HPP

#include <lpfx/domain/cartesian/location.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

// The owner checks of the field operators: compile-time shape and location,
// and the run-time predicates each enqueue function maps to its error codes.
namespace tpic::ecsim {
namespace detail {

template <class Owner, class Location, std::size_t... Component>
[[nodiscard]] constexpr bool components_at(std::index_sequence<Component...>) noexcept {
  return (std::is_same_v<Location, typename decltype(std::declval<const Owner&>().view()
      .template component<Component>())::location_type> && ...);
}

} // namespace detail

// A field owner whose components all live at Location.
template <class Owner, class Location>
concept FieldAt =
    detail::components_at<Owner, Location>(std::make_index_sequence<Owner::component_count>{});

template <class Owner, class Location>
concept VectorFieldAt = Owner::component_count == 3 && FieldAt<Owner, Location>;

// The location shared by the components of a FieldAt owner.
template <class Owner>
using field_location_t =
    typename decltype(std::declval<const Owner&>().view().template component<0>())::location_type;

// Every component has at least depth ghost layers on all six sides.
template <class Owner>
[[nodiscard]] bool has_halo(const Owner& owner, int depth) noexcept {
  return [&]<std::size_t... Component>(std::index_sequence<Component...>) {
    const auto enough = [depth](auto halo) {
      return halo.x_minus >= depth && halo.x_plus >= depth &&
             halo.y_minus >= depth && halo.y_plus >= depth &&
             halo.z_minus >= depth && halo.z_plus >= depth;
    };
    return (enough(owner.template component_halo<Component>()) && ...);
  }(std::make_index_sequence<Owner::component_count>{});
}

// Some component of a starts at the storage of some component of b. Distinct
// owning allocations cannot partially overlap, so this detects in-place use.
template <class A, class B>
[[nodiscard]] bool shares_storage(const A& a, const B& b) noexcept {
  const auto starts = []<class Owner, std::size_t... C>(const Owner& owner, std::index_sequence<C...>) {
    return std::array<const void*, sizeof...(C)>{owner.view().template component<C>().core_data()...};
  };
  for (const auto left : starts(a, std::make_index_sequence<A::component_count>{}))
    for (const auto right : starts(b, std::make_index_sequence<B::component_count>{}))
      if (left == right) return true;
  return false;
}

// The owner's logical cells are nx * ny * nz.
template <class Owner>
[[nodiscard]] bool field_cells_equal(const Owner& owner, int nx, int ny, int nz) noexcept {
  const auto cells = owner.cells();
  return std::cmp_equal(cells.x(), nx) && std::cmp_equal(cells.y(), ny) &&
         std::cmp_equal(cells.z(), nz);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_FIELD_CHECKS_HPP
