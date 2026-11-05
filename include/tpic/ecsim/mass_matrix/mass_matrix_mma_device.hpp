#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MMA_DEVICE_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MMA_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/config.hpp>

// Keep PTX and the CUDA-only entry point out of every non-NVIDIA translation
// unit. Scalar deposition remains available independently of this header.
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA

#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_sorted_cell_device.hpp>
#include <tpic/ecsim/particles/sorted_cell_device.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// The FP64 MMA kernel body exists only in sm_80+ device images. Host passes
// and older device images compile the flag-setting body instead.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define TPIC_MASS_MMA_DEVICE_TARGET 1
#else
#define TPIC_MASS_MMA_DEVICE_TARGET 0
#endif

namespace tpic::ecsim::detail {

// FP64 m8n8k4 MMA shape: a warp of 32 lanes, K = 4 particles per instruction
// and two output registers per lane.
inline constexpr int mass_mma_k = 4;
inline constexpr int mass_mma_outputs_per_lane = 2;

#if TPIC_MASS_MMA_DEVICE_TARGET
// FP64 m8n8k4 MMA operand map: each lane carries A[row][k]
// and B[k][column], with row=column=lane/4 and k=lane%4. The two output registers
// hold D[row][2*k] and D[row][2*k+1]. Matrices never pass through shared memory.
// The statement has no side effect beyond its outputs, so it is not volatile.
__device__ __forceinline__ void ptx_mma_m8n8k4_f64(
    double& d0, double& d1, double a0, double b0, double c0, double c1) noexcept {
  asm(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
      "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
      : "=d"(d0), "=d"(d1)
      : "d"(a0), "d"(b0), "d"(c0), "d"(c1));
}
#endif

} // namespace tpic::ecsim::detail

namespace tpic::ecsim::kernels {

// One warp per cell (SortedCellLaunch), cached B, 64-particle register chunks
// and nine FP64 accumulators per output register of a lane. Narrows only at the
// canonical-pair atomic flush. *invalid flags an invalid record or a non-finite
// entry.
template <class Precision, class ParticleView, class MagneticView>
__global__ __launch_bounds__(ecsim::sorted_cell_threads) void sorted_mass_matrix_mma_kernel(
    ParticleView particles, MagneticView magnetic, typename Precision::Storage::Mass* mass,
    data::MassMatrixLayout layout, data::ParticleCellMap cells,
    typename Precision::Compute::Mass beta_over_c,
    ecsim::ParticleBins bins, unsigned* invalid) {
  using Compute = typename Precision::Compute::Mass;
  static_assert(std::is_same_v<Compute, double>);
#if TPIC_MASS_MMA_DEVICE_TARGET
  namespace subgroup = lpfx::device::subgroup;
  constexpr unsigned width = 32;
  constexpr unsigned cells_per_block = ecsim::cells_per_block<width>;
  constexpr unsigned particles_per_lane = 2;
  constexpr unsigned chunk_size = particles_per_lane * width;
  static_assert(subgroup::warp_size() == width || subgroup::warp_size() == 1);
  __shared__ Compute cell_magnetic[cells_per_block][ecsim::cached_cell_values];
  const unsigned lane = subgroup::warp_lane();
  const unsigned warp = threadIdx.x / width;
  const int corner_a = static_cast<int>(lane / detail::mass_mma_k);
  const unsigned particle_in_group = lane % detail::mass_mma_k;

  // Never read a particle view or bin offset after producer failure.
  if (!bins.ready(invalid)) return;

  // The PTX output map covers the 8x8 spatial pairs. Exactly 36 output pairs
  // survive the final flush.
  int stencil[detail::mass_mma_outputs_per_lane];
  #pragma unroll
  for (unsigned slot = 0; slot < detail::mass_mma_outputs_per_lane; ++slot)
    stencil[slot] = ecsim::mass_lane_stencil<width>(lane, slot);

  ecsim::for_each_sorted_cell_broadcast<cells_per_block>(bins, cells, warp, invalid,
      [&](const unsigned bin, const unsigned first, const unsigned count, const data::GridIndex cell) {
    // Public LPFx component access respects each field's independent pitch.
    const auto cached_magnetic = ecsim::cache_cell_magnetic<Precision>(
        magnetic, cell, cell_magnetic[warp]);
    const ecsim::KnownBin source{static_cast<std::int32_t>(bin), cell};
    const auto node = cic_corner(cell, corner_a);
    const auto vertex = layout.vertex_index(node.i, node.j, node.k);
    std::size_t base[detail::mass_mma_outputs_per_lane];
    #pragma unroll
    for (unsigned slot = 0; slot < detail::mass_mma_outputs_per_lane; ++slot)
      base[slot] = stencil[slot] >= 0 ? layout.entry_index(vertex, stencil[slot]) : 0;

    Compute accumulated[data::mass_tensor_entries][detail::mass_mma_outputs_per_lane]{};
    for (unsigned chunk = 0; chunk < count; chunk += chunk_size) {
      Compute rx[particles_per_lane]{}, ry[particles_per_lane]{}, rz[particles_per_lane]{};
      Compute alpha_scaled[particles_per_lane][data::mass_tensor_entries]{};
      #pragma unroll
      for (unsigned reg = 0; reg < particles_per_lane; ++reg) {
        const unsigned offset = chunk + reg * width + lane;
        if (offset < count)
          ecsim::sample_mass_particle<Precision>(particles, cached_magnetic, source,
              beta_over_c, first + offset, invalid, rx[reg], ry[reg], rz[reg], alpha_scaled[reg]);
      }

      #pragma unroll
      for (unsigned reg = 0; reg < particles_per_lane; ++reg) {
        const unsigned reg_offset = chunk + reg * width;
        if (reg_offset >= count) break;
        #pragma unroll
        for (unsigned sub = 0; sub < width / detail::mass_mma_k; ++sub) {
          if (reg_offset + sub * detail::mass_mma_k >= count) break;
          const unsigned source_lane = sub * detail::mass_mma_k + particle_in_group;
          const Compute x = subgroup::warp_shuffle(rx[reg], source_lane);
          const Compute y = subgroup::warp_shuffle(ry[reg], source_lane);
          const Compute z = subgroup::warp_shuffle(rz[reg], source_lane);
          const Compute weight = cic_corner_weight(corner_a, x, y, z);
          // A[a,p] = W[a,p]*alpha[p]; B[p,b] = W[b,p]. The same
          // register weight supplies both operand maps. Zero-padded alpha
          // makes the final group safe while every lane executes each MMA.
          #pragma unroll
          for (int component = 0; component < data::mass_tensor_entries; ++component) {
            const Compute alpha = subgroup::warp_shuffle(alpha_scaled[reg][component], source_lane);
            detail::ptx_mma_m8n8k4_f64(accumulated[component][0], accumulated[component][1],
                weight * alpha, weight, accumulated[component][0], accumulated[component][1]);
          }
        }
      }
    }

    // Adjacent cells share nodes. Only the 36 canonical spatial pairs emit
    // global atomics, at most 36x9 per cell regardless of particle occupancy.
    #pragma unroll
    for (unsigned slot = 0; slot < detail::mass_mma_outputs_per_lane; ++slot) {
      if (stencil[slot] < 0) continue;
      Compute block[data::mass_tensor_entries];
      #pragma unroll
      for (int component = 0; component < data::mass_tensor_entries; ++component)
        block[component] = accumulated[component][slot];
      ecsim::add_mass_block(mass, base[slot], block, invalid);
    }
    // Complete cache readers before this warp begins the next cell.
    subgroup::warp_synchronize();
  });
#else
  // Startup rejects matrix-core below sm_80; a direct launch on such a device
  // image fails visibly instead of doing no work.
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(invalid, 1U);
#endif
}

} // namespace tpic::ecsim::kernels

namespace tpic::ecsim {

// Cold startup admission queries the actual specialization, including whether
// the runtime can resolve a compatible compiled image for this device.
template <class Precision, class ParticleView, class MagneticView>
[[nodiscard]] bool matrix_core_mass_kernel_available(int device) noexcept {
  int major = 0;
  cudaFuncAttributes attributes{};
  // An older PTX image can JIT on a newer GPU but contains the sm<80 stub.
  return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
      major >= 8 && cudaFuncGetAttributes(&attributes,
          kernels::sorted_mass_matrix_mma_kernel<Precision, ParticleView, MagneticView>) == cudaSuccess &&
      attributes.ptxVersion >= 80 && attributes.binaryVersion >= 80;
}

} // namespace tpic::ecsim

#endif // LPFX_CONFIG_DEVICE_BACKEND_CUDA

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MMA_DEVICE_HPP
