#ifndef LPFX_ALGORITHM_COUPLING_HOST_HPP
#define LPFX_ALGORITHM_COUPLING_HOST_HPP

#include <concepts>
#include <cstddef>

#include <lpfx/core/attributes.hpp>
#include <lpfx/algorithm/coupling/particle.hpp>
#include <lpfx/data/particle/host/view.hpp>

namespace lpfx {

// Stateless direct traversal of the published host view. Owners participate by
// passing owner.view(); no plan, scratch, allocation, or retained execution
// object is needed. Particle attributes and component storage must not overlap.
// Each iteration writes one distinct output lane, so the compiler may safely
// vectorize the traversal even though field samples are gathered indirectly.
template <std::floating_point Accumulator, class PositionXTag,
          class PositionYTag, class PositionZTag, class OutputTag,
          detail::ParticleSchemaType Schema, class Component,
          std::floating_point GeometryScalar>
  requires detail::TrilinearParticleScalarSchema<
               Schema, PositionXTag, PositionYTag, PositionZTag, OutputTag> &&
           detail::TrilinearParticleGatherComponent<Component, HostSpace,
                                                    Accumulator>
LPFX_ALWAYS_INLINE constexpr void run_serial_particle_trilinear_gather(
    const BasicParticleView<Schema, HostSpace, false> particles,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry) noexcept {
  using Output = detail::particle_value_t<Schema, OutputTag>;

  // Resolve typed SoA bases once: get() in the loop would put a launder into
  // every iteration and inhibit vectorization.
  const auto *const position_x = particles.template get<PositionXTag>().data();
  const auto *const position_y = particles.template get<PositionYTag>().data();
  const auto *const position_z = particles.template get<PositionZTag>().data();
  auto *const output = particles.template get<OutputTag>().data();

  LPFX_INDEPENDENT_LOOP
  for (std::size_t particle = 0; particle != particles.size(); ++particle) {
    output[particle] = static_cast<Output>(
        detail::trilinear_sample_physical_position<Accumulator>(
            position_x[particle], position_y[particle], position_z[particle],
            component, geometry));
  }
}

// Adds one published particle's scalar amount to eight field samples. The
// field is neither cleared nor replaced. All numeric and addressability
// requirements are hard preconditions proved outside this inner primitive.
template <std::floating_point Accumulator, class PositionXTag,
          class PositionYTag, class PositionZTag, class AmountTag,
          detail::ParticleSchemaType Schema, bool ParticleConst,
          class Component, std::floating_point GeometryScalar>
  requires detail::TrilinearParticleDepositionSchema<
               Schema, PositionXTag, PositionYTag, PositionZTag, AmountTag> &&
           detail::TrilinearParticleDepositionComponent<Component, HostSpace,
                                                        Accumulator> &&
           detail::TrilinearDepositionScalar<GeometryScalar>
LPFX_ALWAYS_INLINE constexpr void deposit_trilinear_particle(
    const BasicParticleView<Schema, HostSpace, ParticleConst> particles,
    const typename BasicParticleView<Schema, HostSpace,
                                     ParticleConst>::count_type particle_index,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry) noexcept {
  const auto stencil = detail::make_particle_trilinear_stencil<Component>(
      particles.template get<PositionXTag>()[particle_index],
      particles.template get<PositionYTag>()[particle_index],
      particles.template get<PositionZTag>()[particle_index], geometry);
  detail::deposit_trilinear_component_serial<Accumulator>(
      component, stencil, particles.template get<AmountTag>()[particle_index]);
}

// Stateless traversal of the published prefix. Updates are applied in
// particle order and corner order 0..7, so fixed host inputs have one explicit
// serial floating-point order. No plan, scratch, allocation, or state exists.
template <std::floating_point Accumulator, class PositionXTag,
          class PositionYTag, class PositionZTag, class AmountTag,
          detail::ParticleSchemaType Schema, bool ParticleConst,
          class Component, std::floating_point GeometryScalar>
  requires detail::TrilinearParticleDepositionSchema<
               Schema, PositionXTag, PositionYTag, PositionZTag, AmountTag> &&
           detail::TrilinearParticleDepositionComponent<Component, HostSpace,
                                                        Accumulator> &&
           detail::TrilinearDepositionScalar<GeometryScalar>
LPFX_ALWAYS_INLINE constexpr void run_serial_particle_trilinear_deposition(
    const BasicParticleView<Schema, HostSpace, ParticleConst> particles,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry) noexcept {
  const auto *const position_x = particles.template get<PositionXTag>().data();
  const auto *const position_y = particles.template get<PositionYTag>().data();
  const auto *const position_z = particles.template get<PositionZTag>().data();
  const auto *const amount = particles.template get<AmountTag>().data();

  for (std::size_t particle = 0; particle != particles.size(); ++particle) {
    detail::deposit_trilinear_component_serial<Accumulator>(
        component,
        detail::make_particle_trilinear_stencil<Component>(
            position_x[particle], position_y[particle], position_z[particle],
            geometry),
        amount[particle]);
  }
}

} // namespace lpfx

#endif
