#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_DEVICE_CUDA_HIP_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_DEVICE_CUDA_HIP_HPP

#include <concepts>
#include <cstdint>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/algorithm/device_kernel.hpp>
#include <lpfx/algorithm/field/differential/device/work.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace lpfx::detail {

template <std::floating_point Accumulator, class Sources, class Targets,
          class Op, CartesianIndexPolicy IndexPolicy,
          std::floating_point SpacingScalar>
__global__ __launch_bounds__(device_grid_block_threads) void device_stencil_kernel(
    const DeviceStencilBinding<Sources, Targets> binding,
    const PreparedDeviceStencilWork<Op, IndexPolicy> work,
    const DifferentialSpacing3D<SpacingScalar> spacing) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * device_grid_block_threads +
      static_cast<std::uint64_t>(threadIdx.x);
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * device_grid_block_threads;
  const std::uint64_t total = work.total_sample_count();
  for (std::uint64_t sample = first; sample < total;) {
    apply_device_stencil_sample<Accumulator>(binding, work, spacing, sample);
    if (!advance_grid_stride_sample(total, stride, sample)) {
      break;
    }
  }
}

// Fused traversal of a three-target stencil over stencil_union_box(work): one
// pass reads each source sample for every target that uses it, instead of one
// pass per target.
template <std::floating_point Accumulator, class Sources, class Targets,
          class Op, CartesianIndexPolicy IndexPolicy,
          std::floating_point SpacingScalar>
  requires(Op::target_count == 3)
__global__ __launch_bounds__(device_grid_block_threads) void device_stencil_union_kernel(
    const DeviceStencilBinding<Sources, Targets> binding,
    const PreparedDeviceStencilWork<Op, IndexPolicy> work,
    const DifferentialSpacing3D<SpacingScalar> spacing) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * device_grid_block_threads +
      static_cast<std::uint64_t>(threadIdx.x);
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * device_grid_block_threads;
  const StencilUnionBox box = stencil_union_box(work);
  const std::uint64_t xy = box.x * box.y;
  const std::uint64_t total = xy * box.z;
  for (std::uint64_t sample = first; sample < total;) {
    const std::uint64_t k = sample / xy;
    const std::uint64_t xy_sample = sample - k * xy;
    const std::uint64_t j = xy_sample / box.x;
    const std::uint64_t i = xy_sample - j * box.x;
    apply_device_stencil_union_point<Accumulator>(binding, work, spacing, i, j,
                                                  k);
    if (!advance_grid_stride_sample(total, stride, sample)) {
      break;
    }
  }
}

// Both traversals are compiled for a three-target Op; the host picks one per
// launch from the prepared extents (device_stencil_selects_union_traversal),
// so neither kernel carries a traversal branch.
template <device::DeviceKernelLaunchBackend Backend>
struct CudaHipDeviceStencilLaunch {
  template <std::floating_point Accumulator, class Sources, class Targets,
            class Op, CartesianIndexPolicy IndexPolicy,
            std::floating_point SpacingScalar>
  [[nodiscard]] static typename device::BackendTraits<Backend>::error_type
  enqueue(const DeviceStencilBinding<Sources, Targets> binding,
          const PreparedDeviceStencilWork<Op, IndexPolicy> work,
          const DifferentialSpacing3D<SpacingScalar> spacing,
          const typename device::BackendTraits<Backend>::stream_type stream,
          bool &accepted) noexcept {
    if constexpr (Op::target_count == 3) {
      if (device_stencil_selects_union_traversal(work)) {
        const std::uint32_t block_count =
            grid_stride_block_count(device_stencil_union_count(work));
        return device::checked_kernel_launch<Backend>(accepted, [&] {
          device_stencil_union_kernel<Accumulator>
              <<<block_count, device_grid_block_threads, 0, stream>>>(
                  binding, work, spacing);
        });
      }
    }
    const std::uint32_t block_count =
        grid_stride_block_count(work.total_sample_count());
    return device::checked_kernel_launch<Backend>(accepted, [&] {
      device_stencil_kernel<Accumulator>
          <<<block_count, device_grid_block_threads, 0, stream>>>(
              binding, work, spacing);
    });
  }
};

#if defined(__CUDACC__)
template <>
struct DeviceStencilLaunch<device::CudaBackend> final
    : CudaHipDeviceStencilLaunch<device::CudaBackend> {};
#endif

#if defined(__HIPCC__)
template <>
struct DeviceStencilLaunch<device::HipBackend> final
    : CudaHipDeviceStencilLaunch<device::HipBackend> {};
#endif

} // namespace lpfx::detail

#endif

#endif
