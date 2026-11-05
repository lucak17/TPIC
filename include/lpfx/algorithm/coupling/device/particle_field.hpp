#ifndef LPFX_ALGORITHM_COUPLING_DEVICE_PARTICLE_FIELD_HPP
#define LPFX_ALGORITHM_COUPLING_DEVICE_PARTICLE_FIELD_HPP

#include <concepts>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/error.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/algorithm/coupling/device/launch.hpp>
#include <lpfx/algorithm/coupling/particle.hpp>
#include <lpfx/data/particle/device/view.hpp>

namespace lpfx::detail {

template <TrilinearCoupling Op, class Accumulator, class PositionXTag,
          class PositionYTag, class PositionZTag, class ScalarTag,
          class Schema, class Backend, bool Const, class Component,
          class GeometryScalar>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_particle_trilinear(
    const BasicParticleView<Schema, DeviceSpace<Backend>, Const> &particles,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry,
    const device::Executor<Backend> executor) noexcept {
  if (particles.size() == 0) {
    return ErrorCode::success;
  }
  if (!executor.bound()) {
    return ErrorCode::invalid_state;
  }
  bool accepted = false;
  const auto error = DeviceParticleTrilinearLaunch<Backend>::template enqueue<
      Op, Accumulator, PositionXTag, PositionYTag, PositionZTag, ScalarTag>(
      particles, component, geometry, executor.stream(), accepted);
  return device::finish_device_enqueue<Backend>(error, accepted);
}

} // namespace lpfx::detail

#if LPFX_NATIVE_DEVICE_OPERATIONS
#include <lpfx/algorithm/coupling/device/cuda_hip.hpp>
#endif

namespace lpfx {

// Enqueues one direct gather per published particle. Stream order is the sole
// completion and lifetime contract: particles, field storage, and geometry must
// remain usable until the caller has observed completion of later work on the
// same stream. No owner, event, scratch, plan, or conclusion state is retained.
template <std::floating_point Accumulator, class PositionXTag,
          class PositionYTag, class PositionZTag, class OutputTag,
          detail::ParticleSchemaType Schema, device::DeviceBackend Backend,
          class Component, std::floating_point GeometryScalar>
  requires detail::TrilinearParticleScalarSchema<
               Schema, PositionXTag, PositionYTag, PositionZTag, OutputTag> &&
           detail::TrilinearParticleGatherComponent<
               Component, DeviceSpace<Backend>, Accumulator> &&
           detail::DeviceParticleTrilinearLaunchable<
               Backend, detail::TrilinearCoupling::gather, Accumulator,
               PositionXTag, PositionYTag, PositionZTag, OutputTag, Schema,
               false, Component, GeometryScalar>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
enqueue_device_particle_trilinear_gather(
    const BasicParticleView<Schema, DeviceSpace<Backend>, false> &particles,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_particle_trilinear<
      detail::TrilinearCoupling::gather, Accumulator, PositionXTag,
      PositionYTag, PositionZTag, OutputTag>(particles, component, geometry,
                                             executor);
}

// Enqueues one additive eight-corner deposition per published particle.
// Caller stream order is the completion, visibility, exclusivity, and borrowed
// lifetime contract. The operation owns no resource or state and never clears
// or publishes either owner.
template <std::floating_point Accumulator, class PositionXTag,
          class PositionYTag, class PositionZTag, class AmountTag,
          detail::ParticleSchemaType Schema, device::DeviceBackend Backend,
          bool ParticleConst, class Component,
          std::floating_point GeometryScalar>
  requires detail::TrilinearParticleDepositionSchema<
               Schema, PositionXTag, PositionYTag, PositionZTag, AmountTag> &&
           detail::TrilinearParticleDepositionComponent<
               Component, DeviceSpace<Backend>, Accumulator> &&
           detail::TrilinearDepositionScalar<GeometryScalar> &&
           detail::DeviceParticleTrilinearLaunchable<
               Backend, detail::TrilinearCoupling::deposit, Accumulator,
               PositionXTag, PositionYTag, PositionZTag, AmountTag, Schema,
               ParticleConst, Component, GeometryScalar>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
enqueue_device_particle_trilinear_deposition(
    const BasicParticleView<Schema, DeviceSpace<Backend>, ParticleConst>
        &particles,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_particle_trilinear<
      detail::TrilinearCoupling::deposit, Accumulator, PositionXTag,
      PositionYTag, PositionZTag, AmountTag>(particles, component, geometry,
                                             executor);
}

} // namespace lpfx

#endif
