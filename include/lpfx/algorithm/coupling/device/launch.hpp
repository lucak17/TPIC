#ifndef LPFX_ALGORITHM_COUPLING_DEVICE_LAUNCH_HPP
#define LPFX_ALGORITHM_COUPLING_DEVICE_LAUNCH_HPP

#include <concepts>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/algorithm/coupling/particle.hpp>
#include <lpfx/data/particle/device/view.hpp>

namespace lpfx::detail {

enum class TrilinearCoupling : std::uint8_t {
  gather,  // scalar = trilinear sample of the component
  deposit, // component += scalar * corner weight, atomically
};

// One fixed launch shape keeps the stateless seam direct.
inline constexpr std::uint32_t device_particle_trilinear_block_threads = 256;

// Exact kernel-facing source: the four attributes read or written plus the
// published count, so unrelated attributes never enlarge the kernel ABI.
// Scalar is the mutable gather output or the const deposited amount.
template <std::floating_point PositionX, std::floating_point PositionY,
          std::floating_point PositionZ, class Scalar>
  requires std::floating_point<std::remove_const_t<Scalar>>
struct DeviceParticleTrilinearBinding final {
  const PositionX *position_x{};
  const PositionY *position_y{};
  const PositionZ *position_z{};
  Scalar *scalar{};
  std::uint32_t count{};
};
static_assert(std::is_trivially_copyable_v<
              DeviceParticleTrilinearBinding<double, double, double, double>>);

template <TrilinearCoupling Op, class Schema, class ScalarTag>
using trilinear_particle_scalar_t =
    std::conditional_t<Op == TrilinearCoupling::gather,
                       particle_value_t<Schema, ScalarTag>,
                       const particle_value_t<Schema, ScalarTag>>;

template <TrilinearCoupling Op, class PositionXTag, class PositionYTag,
          class PositionZTag, class ScalarTag, ParticleSchemaType Schema,
          class Backend, bool Const>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr DeviceParticleTrilinearBinding<
    particle_value_t<Schema, PositionXTag>,
    particle_value_t<Schema, PositionYTag>,
    particle_value_t<Schema, PositionZTag>,
    trilinear_particle_scalar_t<Op, Schema, ScalarTag>>
make_device_particle_trilinear_binding(
    const BasicParticleView<Schema, DeviceSpace<Backend>, Const>
        &particles) noexcept {
  return {particles.template get<PositionXTag>(),
          particles.template get<PositionYTag>(),
          particles.template get<PositionZTag>(),
          particles.template get<ScalarTag>(), particles.size()};
}

// Launch seam, specialized statically by a device-language leaf:
//   enqueue<Op, Accumulator, PositionXTag, PositionYTag, PositionZTag,
//           ScalarTag>(particles, component, geometry, stream, accepted)
template <class Backend> struct DeviceParticleTrilinearLaunch;

template <class Backend, TrilinearCoupling Op, class Accumulator,
          class PositionXTag, class PositionYTag, class PositionZTag,
          class ScalarTag, class Schema, bool Const, class Component,
          class GeometryScalar>
concept DeviceParticleTrilinearLaunchable =
    device::DeviceBackend<Backend> &&
    requires(
        const BasicParticleView<Schema, DeviceSpace<Backend>, Const> &particles,
        const Component component,
        const PhysicalToLogical3D<GeometryScalar> geometry,
        const typename device::BackendTraits<Backend>::stream_type stream,
        bool &accepted) {
      {
        DeviceParticleTrilinearLaunch<Backend>::template enqueue<
            Op, Accumulator, PositionXTag, PositionYTag, PositionZTag,
            ScalarTag>(particles, component, geometry, stream, accepted)
      } noexcept
          -> std::same_as<typename device::BackendTraits<Backend>::error_type>;
    };

} // namespace lpfx::detail

#endif
