#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MFMA_DEVICE_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MFMA_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/config.hpp>

#if LPFX_CONFIG_DEVICE_BACKEND_HIP
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_mfma_layout.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_sorted_cell_device.hpp>
#include <tpic/ecsim/particles/sorted_cell_device.hpp>

#include <cstdint>
#include <type_traits>

namespace tpic::ecsim {

inline constexpr unsigned mass_mfma4_threads = 4 * MassMfma4Layout::width;
// Four wave64 cells per block.
using MassMfma4Launch = lpfx::device::KernelLaunch<mass_mfma4_threads,
    cells_per_block<MassMfma4Layout::width, mass_mfma4_threads>>;

static __device__ __constant__ MassLaneStencilMap<MassMfma4Layout> device_mass_mfma4_stencil_map =
    make_mass_lane_stencil_map<MassMfma4Layout>();

} // namespace tpic::ecsim

namespace tpic::ecsim::kernels {

// Sorted MFMA4 mass kernel (MassMfma4Launch): one wave64 owns a cell, four
// waves per block, one particle per lane in a 64-particle chunk. Nine MFMA4
// instructions accumulate each group of four particles; all sums stay FP64
// until the flush. *invalid flags an invalid record or a non-finite entry.
template <class Precision, class ParticleView, class MagneticView>
__global__ __launch_bounds__(ecsim::mass_mfma4_threads) void sorted_mass_matrix_mfma4_kernel(
    ParticleView particles, MagneticView magnetic, typename Precision::Storage::Mass* mass,
    data::MassMatrixLayout layout, data::ParticleCellMap cells,
    typename Precision::Compute::Mass beta_over_c,
    ecsim::ParticleBins bins, unsigned* invalid) {
  using Compute = typename Precision::Compute::Mass;
  static_assert(std::is_same_v<Compute, double>);
#if TPIC_MASS_MFMA4_DEVICE_TARGET
  using Map = ecsim::MassMfma4Layout;
  namespace subgroup = lpfx::device::subgroup;
  constexpr unsigned width = Map::width;
  constexpr unsigned cells_per_block = ecsim::cells_per_block<width, ecsim::mass_mfma4_threads>;
  static_assert(subgroup::warp_size() == width || subgroup::warp_size() == 1);
  __shared__ Compute cell_magnetic[cells_per_block][ecsim::cached_cell_values];
  const unsigned lane = subgroup::warp_lane();
  const unsigned warp = threadIdx.x / width;
  const int stencil = ecsim::device_mass_mfma4_stencil_map.value[lane][0];

  if (!bins.ready(invalid)) return;
  ecsim::for_each_sorted_cell_broadcast<cells_per_block>(bins, cells, warp, invalid,
      [&](const unsigned bin, const unsigned first, const unsigned count, const data::GridIndex cell) {
    const auto cached_magnetic = ecsim::cache_cell_magnetic<Precision>(
        magnetic, cell, cell_magnetic[warp]);
    const ecsim::KnownBin source{static_cast<std::int32_t>(bin), cell};
    Compute accumulated[data::mass_tensor_entries]{};
    for (unsigned chunk = 0; chunk < count; chunk += width) {
      Compute rx{}, ry{}, rz{}, alpha_scaled[data::mass_tensor_entries]{};
      if (chunk + lane < count)
        ecsim::sample_mass_particle<Precision>(particles, cached_magnetic, source,
            beta_over_c, first + chunk + lane, invalid, rx, ry, rz, alpha_scaled);
      const unsigned remaining = count - chunk;
      const unsigned populated = remaining < width ? remaining : width;
      for (unsigned sub = 0; sub < populated; sub += Map::group_particles) {
        const unsigned source_lane = sub + Map::particle(lane);
        const Compute x = subgroup::warp_shuffle(rx, source_lane);
        const Compute y = subgroup::warp_shuffle(ry, source_lane);
        const Compute z = subgroup::warp_shuffle(rz, source_lane);
        // A padded or rejected source has zero fractions and zero alpha: its
        // weight is 0 or 1, its A operand weight * alpha is +0, and every
        // product with it contributes +0, so the weight needs no zero select.
        const Compute weight = cic_corner_weight(Map::input_corner(lane), x, y, z);
        const Compute partner = subgroup::warp_shuffle(weight, Map::partner_source(lane));
        // Pad operands, never predicate MFMA: all 64 lanes participate even
        // for a partial four-particle group or a rejected particle record.
        #pragma unroll
        for (int component = 0; component < data::mass_tensor_entries; ++component) {
          const Compute alpha = subgroup::warp_shuffle(alpha_scaled[component], source_lane);
          accumulated[component] = __builtin_amdgcn_mfma_f64_4x4x4f64(
              weight * alpha, partner, accumulated[component], 0, 0, 0);
        }
      }
    }
    if (stencil >= 0) {
      const auto node = cic_corner(cell, Map::output_a(lane));
      ecsim::add_mass_block(mass,
          layout.entry_index(layout.vertex_index(node.i, node.j, node.k), stencil),
          accumulated, invalid);
    }
    subgroup::warp_synchronize();
  });
#else
  // Keep scalar builds usable on other HIP targets. Startup rejects matrix-core
  // there; direct unsupported launches fail visibly instead of doing no work.
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(invalid, 1U);
#endif
}

} // namespace tpic::ecsim::kernels

namespace tpic::ecsim {

// Cold startup admission queries the actual specialization, including whether
// the runtime can resolve a compatible compiled image for this device.
template <class Precision, class ParticleView, class MagneticView>
[[nodiscard]] bool matrix_core_mass_kernel_available(int device) noexcept {
  hipDeviceProp_t properties{};
  hipFuncAttributes attributes{};
  return hipGetDeviceProperties(&properties, device) == hipSuccess &&
      mass_mfma4_supported_architecture(properties.gcnArchName, properties.warpSize) &&
      hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(
          kernels::sorted_mass_matrix_mfma4_kernel<Precision, ParticleView, MagneticView>)) == hipSuccess;
}

} // namespace tpic::ecsim
#endif // LPFX_CONFIG_DEVICE_BACKEND_HIP

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MFMA_DEVICE_HPP
