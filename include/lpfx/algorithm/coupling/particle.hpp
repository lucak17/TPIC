#ifndef LPFX_ALGORITHM_COUPLING_PARTICLE_HPP
#define LPFX_ALGORITHM_COUPLING_PARTICLE_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/algorithm/coupling/physical_to_logical.hpp>
#include <lpfx/algorithm/coupling/trilinear.hpp>
#include <lpfx/data/particle/layout.hpp>
#include <lpfx/data/particle/schema.hpp>

namespace lpfx {

namespace detail {

// A class trait, not a direct alias: nvcc fails explicit-argument
// substitution of signatures that spell the value type through a dependent
// variable template while Schema is still being deduced.
template <class Schema, class Name> struct ParticleValueTraits final {
  static constexpr std::size_t index = Schema::template index_of<Name>;
  using type = typename Schema::template value_type<index>;
};

template <class Schema, class Name>
using particle_value_t = typename ParticleValueTraits<Schema, Name>::type;

template <class Schema, class Name>
concept FloatingParticleAttribute =
    Schema::template contains<Name> &&
    std::floating_point<particle_value_t<Schema, Name>>;

// Three position attributes and one scalar attribute (gathered output or
// deposited amount), all distinct and floating-point.
template <class Schema, class PositionXTag, class PositionYTag,
          class PositionZTag, class ScalarTag>
concept TrilinearParticleScalarSchema =
    ParticleSchemaType<Schema> &&
    meta::all_distinct<PositionXTag, PositionYTag, PositionZTag, ScalarTag> &&
    FloatingParticleAttribute<Schema, PositionXTag> &&
    FloatingParticleAttribute<Schema, PositionYTag> &&
    FloatingParticleAttribute<Schema, PositionZTag> &&
    FloatingParticleAttribute<Schema, ScalarTag>;

template <class T>
concept TrilinearDepositionScalar =
    std::same_as<std::remove_cv_t<T>, float> ||
    std::same_as<std::remove_cv_t<T>, double>;

template <class Schema, class PositionXTag, class PositionYTag,
          class PositionZTag, class AmountTag>
concept TrilinearParticleDepositionSchema =
    TrilinearParticleScalarSchema<Schema, PositionXTag, PositionYTag,
                                  PositionZTag, AmountTag> &&
    TrilinearDepositionScalar<particle_value_t<Schema, PositionXTag>> &&
    TrilinearDepositionScalar<particle_value_t<Schema, PositionYTag>> &&
    TrilinearDepositionScalar<particle_value_t<Schema, PositionZTag>> &&
    TrilinearDepositionScalar<particle_value_t<Schema, AmountTag>>;

template <class Component, class Space, class Accumulator>
concept TrilinearParticleGatherComponent = requires {
  typename Component::space_type;
  typename Component::location_type;
  typename Component::index_policy;
  requires std::same_as<typename Component::space_type, Space>;
  requires CartesianLocation<typename Component::location_type>;
  requires CartesianIndexPolicy<typename Component::index_policy>;
  requires TrilinearSampleComponent<
      Component, typename Component::location_type,
      typename Component::index_policy, Accumulator>;
};

template <class Component, class Space, class Accumulator>
concept TrilinearParticleDepositionComponent =
    requires(const Component component,
             const typename Component::coordinate_type i,
             const typename Component::coordinate_type j,
             const typename Component::coordinate_type k) {
      typename Component::element_type;
      typename Component::value_type;
      typename Component::coordinate_type;
      typename Component::space_type;
      typename Component::location_type;
      typename Component::index_policy;
      requires std::same_as<typename Component::space_type, Space>;
      requires CartesianLocation<typename Component::location_type>;
      requires CartesianIndexPolicy<typename Component::index_policy>;
      requires TrilinearDepositionScalar<typename Component::value_type>;
      requires std::same_as<
          std::remove_const_t<typename Component::element_type>,
          typename Component::value_type>;
      requires(!std::is_const_v<typename Component::element_type>);
      requires TrilinearDepositionScalar<Accumulator>;
      { component(i, j, k) } -> std::same_as<typename Component::value_type &>;
    };

template <class Component, std::floating_point PhysicalX,
          std::floating_point PhysicalY, std::floating_point PhysicalZ,
          std::floating_point GeometryScalar>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr auto
make_particle_trilinear_stencil(
    const PhysicalX physical_x, const PhysicalY physical_y,
    const PhysicalZ physical_z,
    const PhysicalToLogical3D<GeometryScalar> geometry) noexcept {
  return make_trilinear_stencil<typename Component::location_type,
                                typename Component::index_policy>(
      geometry.logical_x(physical_x), geometry.logical_y(physical_y),
      geometry.logical_z(physical_z));
}

template <std::floating_point Accumulator, std::floating_point PhysicalX,
          std::floating_point PhysicalY, std::floating_point PhysicalZ,
          class Component, std::floating_point GeometryScalar>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Accumulator
trilinear_sample_physical_position(
    const PhysicalX physical_x, const PhysicalY physical_y,
    const PhysicalZ physical_z, const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry) noexcept {
  return trilinear_sample<Accumulator>(
      component, make_particle_trilinear_stencil<Component>(
                     physical_x, physical_y, physical_z, geometry));
}

template <std::floating_point FieldScalar, std::floating_point Accumulator,
          std::floating_point Amount, std::floating_point Weight>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr FieldScalar
trilinear_deposition_contribution(const Amount amount,
                                  const Weight weight) noexcept {
  return static_cast<FieldScalar>(static_cast<Accumulator>(amount) *
                                  static_cast<Accumulator>(weight));
}

// Adds one corner's share of amount to the component sample.
template <std::floating_point Accumulator, class Component, class Stencil,
          std::floating_point Amount>
struct SerialTrilinearDeposit final {
  const Component &component;
  const Stencil &stencil;
  const Amount amount;

  template <std::size_t Corner>
  LPFX_ALWAYS_INLINE constexpr void
  operator()(meta::index_constant<Corner>) const noexcept {
    stencil.template at<Corner>(component) +=
        trilinear_deposition_contribution<typename Component::value_type,
                                          Accumulator>(
            amount, stencil.template weight<Corner>());
  }
};

// Serial-only accumulation in corner order 0..7; no weight array is retained.
template <std::floating_point Accumulator, class Component,
          CartesianLocation Location, CartesianIndexPolicy IndexPolicy,
          std::floating_point Weight, std::floating_point Amount>
  requires TrilinearParticleDepositionComponent<
               Component, typename Component::space_type, Accumulator> &&
           std::same_as<typename Component::location_type, Location> &&
           std::same_as<typename Component::index_policy, IndexPolicy>
LPFX_ALWAYS_INLINE constexpr void deposit_trilinear_component_serial(
    const Component component,
    const TrilinearStencil3D<Location, IndexPolicy, Weight> stencil,
    const Amount amount) noexcept {
  meta::for_each_index<8>(
      SerialTrilinearDeposit<Accumulator, Component,
                             TrilinearStencil3D<Location, IndexPolicy, Weight>,
                             Amount>{component, stencil, amount});
}

} // namespace detail

// Hard preconditions: particle_index is published; the transformed physical
// position and its enclosing logical sample pair are representable by the
// component index policy; all eight samples lie in allocated field storage.
// The inner primitive performs no clamping, bounds check, dispatch, or
// allocation. All four SoA roles are selected and validated at compile time.
template <std::floating_point Accumulator, class PositionXTag,
          class PositionYTag, class PositionZTag, class OutputTag,
          detail::ParticleSchemaType Schema, class Space, class Component,
          std::floating_point GeometryScalar>
  requires detail::TrilinearParticleScalarSchema<
               Schema, PositionXTag, PositionYTag, PositionZTag, OutputTag> &&
           detail::TrilinearParticleGatherComponent<Component, Space,
                                                    Accumulator>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr void gather_trilinear_particle(
    const BasicParticleView<Schema, Space, false> particles,
    const typename BasicParticleView<Schema, Space, false>::count_type
        particle_index,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry) noexcept {
  using Output = detail::particle_value_t<Schema, OutputTag>;

  particles.template get<OutputTag>()[particle_index] = static_cast<Output>(
      detail::trilinear_sample_physical_position<Accumulator>(
          particles.template get<PositionXTag>()[particle_index],
          particles.template get<PositionYTag>()[particle_index],
          particles.template get<PositionZTag>()[particle_index], component,
          geometry));
}

} // namespace lpfx

#endif
