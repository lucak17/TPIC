#ifndef LPFX_ALGORITHM_COUPLING_DEVICE_CUDA_HIP_HPP
#define LPFX_ALGORITHM_COUPLING_DEVICE_CUDA_HIP_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/algorithm/coupling/device/launch.hpp>
#include <lpfx/algorithm/coupling/particle.hpp>
#include <lpfx/algorithm/reduce/operation.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace lpfx::detail {

#if defined(__CUDACC__)
template <class FieldScalar, int Architecture>
inline constexpr bool cuda_native_deposition_atomic_supported =
    !std::same_as<FieldScalar, double> || Architecture >= 600;
#endif

// Atomically adds one corner's share of amount to the component sample.
template <std::floating_point Accumulator, class Component, class Stencil,
          std::floating_point Amount>
struct AtomicTrilinearDeposit final {
  const Component &component;
  const Stencil &stencil;
  const Amount amount;

  template <std::size_t Corner>
  LPFX_DEVICE LPFX_ALWAYS_INLINE void
  operator()(meta::index_constant<Corner>) const noexcept {
    using FieldScalar = typename Component::value_type;
#if defined(__CUDA_ARCH__)
    static_assert(
        cuda_native_deposition_atomic_supported<FieldScalar, __CUDA_ARCH__>,
        "LPFx double deposition requires CUDA sm_60 or newer");
#endif
    static_cast<void>(atomicAdd(
        &stencil.template at<Corner>(component),
        trilinear_deposition_contribution<FieldScalar, Accumulator>(
            amount, stencil.template weight<Corner>())));
  }
};

// Corners 0..7 in order; device-only so the device functor needs no
// host-device caller.
template <class Deposit, std::size_t... Corner>
LPFX_DEVICE LPFX_ALWAYS_INLINE void
deposit_trilinear_corners(const Deposit &deposit,
                          std::index_sequence<Corner...>) noexcept {
  (deposit(meta::index_constant<Corner>{}), ...);
}

template <TrilinearCoupling Op, std::floating_point Accumulator,
          std::floating_point PositionX, std::floating_point PositionY,
          std::floating_point PositionZ, class Scalar, class Component,
          std::floating_point GeometryScalar>
__global__ __launch_bounds__(device_particle_trilinear_block_threads) void device_particle_trilinear_kernel(
    const DeviceParticleTrilinearBinding<PositionX, PositionY, PositionZ,
                                         Scalar>
        particles,
    const Component component,
    const PhysicalToLogical3D<GeometryScalar> geometry) {
  constexpr std::uint32_t block_threads =
      device_particle_trilinear_block_threads;
  const std::uint32_t particle =
      static_cast<std::uint32_t>(blockIdx.x) * block_threads +
      static_cast<std::uint32_t>(threadIdx.x);
  if (particle < particles.count) {
    if constexpr (Op == TrilinearCoupling::gather) {
      particles.scalar[particle] =
          static_cast<Scalar>(trilinear_sample_physical_position<Accumulator>(
              particles.position_x[particle], particles.position_y[particle],
              particles.position_z[particle], component, geometry));
    } else {
      const auto stencil = make_particle_trilinear_stencil<Component>(
          particles.position_x[particle], particles.position_y[particle],
          particles.position_z[particle], geometry);
      deposit_trilinear_corners(
          AtomicTrilinearDeposit<Accumulator, Component, decltype(stencil),
                                 std::remove_const_t<Scalar>>{
              component, stencil, particles.scalar[particle]},
          std::make_index_sequence<8>{});
    }
  }
}

template <device::DeviceKernelLaunchBackend Backend>
struct CudaHipDeviceParticleTrilinearLaunch {
  template <TrilinearCoupling Op, std::floating_point Accumulator,
            class PositionXTag, class PositionYTag, class PositionZTag,
            class ScalarTag, ParticleSchemaType Schema, bool Const,
            class Component, std::floating_point GeometryScalar>
  [[nodiscard]] static typename device::BackendTraits<Backend>::error_type
  enqueue(const BasicParticleView<Schema, DeviceSpace<Backend>, Const>
              &particles,
          const Component component,
          const PhysicalToLogical3D<GeometryScalar> geometry,
          const typename device::BackendTraits<Backend>::stream_type stream,
          bool &accepted) noexcept {
    constexpr std::uint32_t block_threads =
        device_particle_trilinear_block_threads;
    const auto binding =
        make_device_particle_trilinear_binding<Op, PositionXTag, PositionYTag,
                                               PositionZTag, ScalarTag>(
            particles);
    const std::uint32_t block_count = ceil_div(binding.count, block_threads);
    return device::checked_kernel_launch<Backend>(accepted, [&] {
      device_particle_trilinear_kernel<
          Op, Accumulator, particle_value_t<Schema, PositionXTag>,
          particle_value_t<Schema, PositionYTag>,
          particle_value_t<Schema, PositionZTag>,
          trilinear_particle_scalar_t<Op, Schema, ScalarTag>, Component,
          GeometryScalar><<<block_count, block_threads, 0, stream>>>(
          binding, component, geometry);
    });
  }
};

#if defined(__CUDACC__)
template <>
struct DeviceParticleTrilinearLaunch<device::CudaBackend> final
    : CudaHipDeviceParticleTrilinearLaunch<device::CudaBackend> {};
#endif

#if defined(__HIPCC__)
template <>
struct DeviceParticleTrilinearLaunch<device::HipBackend> final
    : CudaHipDeviceParticleTrilinearLaunch<device::HipBackend> {};
#endif

} // namespace lpfx::detail

#endif

#endif
