#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_SCALAR_DEVICE_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_SCALAR_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/block_threads.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/data/mass_matrix/compact_mass_matrix_device.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_sorted_cell_device.hpp>
#include <tpic/ecsim/particles/cic.hpp>
#include <tpic/ecsim/particles/particle_response.hpp>
#include <tpic/ecsim/particles/particle_sample_device.hpp>
#include <tpic/ecsim/particles/sorted_cell_device.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tpic::ecsim {

namespace detail {

template <class Precision, class ParticleView, class MagneticView>
__device__ LPFX_ALWAYS_INLINE void deposit_mass_matrix_particle(
    ParticleView particles, MagneticView magnetic, typename Precision::Storage::Mass* mass,
    data::MassMatrixLayout layout, data::ParticleCellMap cells,
    typename Precision::Compute::Mass beta_over_c,
    std::uint32_t index, unsigned* invalid) {
  using Particles = data::ParticleSchema<Precision>;
  using Compute = typename Precision::Compute::Mass;
  using Mass = typename Precision::Storage::Mass;
  data::GridIndex cell;
  if (!DecodeKey{cells}.resolve(particles.template get<typename Particles::Cell>()[index], cell)) {
    atomicExch(invalid, 1U);
    return;
  }
  const auto fraction = data::load_fractions<Precision>(particles, index);
  const Compute rx = config::convert<Compute>(fraction.x), ry = config::convert<Compute>(fraction.y),
                rz = config::convert<Compute>(fraction.z);
  // The stored fractions are tested: a fraction below one can round to one in a
  // narrower compute type, which is still a valid weight.
  if (!data::in_unit_cell(fraction.x, fraction.y, fraction.z)) {
    atomicExch(invalid, 1U);
    return;
  }
  const auto sample = sample_particle<Precision>(magnetic, cell, rx, ry, rz, beta_over_c);
  const Compute charge = config::convert<Compute>(particles.template get<typename Particles::Charge>()[index]);
  const Compute scale = beta_over_c * charge;
  // Each alpha is checked: a non-finite B makes a diagonal response entry
  // NaN, and a non-finite scale makes alpha[0] non-finite, so these nine
  // checks cover B and the scale. The norm-square witness of the sorted path
  // would miss an alpha that overflows from |a| just above 1 times a scale
  // near the largest finite value.
  Compute alpha_scaled[data::mass_tensor_entries];
  bool finite = true;
  #pragma unroll
  for (int component = 0; component < data::mass_tensor_entries; ++component) {
    alpha_scaled[component] = sample.response.a[component] * scale;
    finite = finite && ::isfinite(alpha_scaled[component]);
  }
  if (!finite) {
    atomicExch(invalid, 1U);
    return;
  }
  // R1: each corner vertex is the cell's vertex plus the corner's strides, so
  // that only one index stays live through the 36 pair blocks instead of the
  // decoded cell (D/MP 60 -> 56 registers, 32 -> 36 resident warps).
  const auto cell_vertex = layout.vertex_index(cell.i, cell.j, cell.k);
  #pragma unroll
  for (int a = 0; a < cic_corner_count; ++a) {
    const auto corner = cic_corner({}, a);
    const auto vertex = cell_vertex + static_cast<std::size_t>(
        corner.i + corner.j * layout.stride_y + corner.k * layout.stride_z);
    #pragma unroll
    for (int b = 0; b < cic_corner_count; ++b) {
      const int stencil = mass_pair_stencil(a, b);
      if (stencil < 0) continue;
      const Compute pair_weight = sample.weights.value[a] * sample.weights.value[b];
      if (pair_weight == Compute(0)) continue;
      const auto base = layout.entry_index(vertex, stencil);
      #pragma unroll
      for (int component = 0; component < data::mass_tensor_entries; ++component) {
        // Finite alpha times a pair weight in [0, 1] is finite in Compute.
        const Mass amount = config::convert<Mass>(alpha_scaled[component] * pair_weight);
        if constexpr (narrowing_may_overflow<Mass, Compute>) {
          if (!::isfinite(amount)) {
            atomicExch(invalid, 1U);
            continue;
          }
        }
        if (amount != Mass(0)) atomicAdd(mass + base + component, amount);
      }
    }
  }
}

} // namespace detail

namespace kernels {

// Grid-stride over the records [first, first + count) with
// KernelLaunch<particle_block_threads> blocks; each contribution is narrowed to
// Mass and added atomically. *invalid flags an invalid record or a non-finite
// contribution.
template <class Precision, class ParticleView, class MagneticView>
__global__ __launch_bounds__(device::particle_block_threads) void unsorted_mass_matrix_kernel(
    ParticleView particles, MagneticView magnetic, typename Precision::Storage::Mass* mass,
    data::MassMatrixLayout layout, data::ParticleCellMap cells,
    typename Precision::Compute::Mass beta_over_c,
    std::uint32_t first, std::uint32_t count, unsigned* invalid) {
  lpfx::device::for_each_grid_index(count, [&](const unsigned offset) {
    detail::deposit_mass_matrix_particle<Precision>(particles, magnetic, mass, layout, cells, beta_over_c, first + offset, invalid);
  });
}

// One native warp per cell bin (SortedCellLaunch); lanes own corner pairs
// (MassSubgroupLayout) and accumulate per cell before one atomic flush per
// canonical pair. *invalid flags an invalid record or a non-finite entry.
template <class Precision, class ParticleView, class MagneticView>
__global__ __launch_bounds__(ecsim::sorted_cell_threads) void sorted_mass_matrix_scalar_kernel(
    ParticleView particles, MagneticView magnetic, typename Precision::Storage::Mass* mass,
    data::MassMatrixLayout layout, data::ParticleCellMap cells,
    typename Precision::Compute::Mass beta_over_c,
    ecsim::ParticleBins bins, unsigned* invalid) {
  using Compute = typename Precision::Compute::Mass;
  using Mass = typename Precision::Storage::Mass;
  namespace subgroup = lpfx::device::subgroup;
  // LPFx reports one during host-only parsing; concrete device images select
  // their native CUDA warp32 or HIP wave32/wave64 width.
  constexpr unsigned width = subgroup::warp_size() > 1 ? subgroup::warp_size() :
      ecsim::smallest_subgroup_width;
  static_assert(width == 32 || width == 64);
  constexpr unsigned cells_per_block = ecsim::cells_per_block<width>;
  using Lanes = ecsim::MassSubgroupLayout<width>;
  constexpr unsigned slots = Lanes::slots;
  constexpr unsigned lanes_per_corner = Lanes::lanes_per_corner;
  __shared__ Compute cell_magnetic[cells_per_block][ecsim::cached_cell_values];
  const unsigned lane = subgroup::warp_lane();
  const unsigned warp = threadIdx.x / width;

  // Each native warp owns a cell, including all of its particle chunks. All
  // lanes participate in shuffles: tail/invalid particle lanes supply zeros.
  // No particle views or bin offsets are read after producer failure.
  if (!bins.ready(invalid)) return;

  // Rectangular 8x8 corner pairs give each lane one (wave64) or two (warp32)
  // tensor accumulators. Only the 36 canonical pairs are flushed. Lane
  // ownership is fixed, so load its direct compile-time-generated mapping.
  const int corner_a = Lanes::output_a(lane);
  int corner_b[slots], stencil[slots];
  #pragma unroll
  for (unsigned slot = 0; slot < slots; ++slot) {
    corner_b[slot] = Lanes::output_b(lane, slot);
    stencil[slot] = ecsim::mass_lane_stencil<width>(lane, slot);
  }

  ecsim::for_each_sorted_cell_broadcast<cells_per_block>(bins, cells, warp, invalid,
      [&](const unsigned bin, const unsigned first, const unsigned count, const data::GridIndex cell) {
    // Load each cell corner once, shared by every particle in this cell.
    const auto cached_magnetic = ecsim::cache_cell_magnetic<Precision>(
        magnetic, cell, cell_magnetic[warp]);
    const ecsim::KnownBin source{static_cast<std::int32_t>(bin), cell};

    const auto node = cic_corner(cell, corner_a);
    const auto vertex = layout.vertex_index(node.i, node.j, node.k);
    std::size_t base[slots];
    Mass accumulated[slots][data::mass_tensor_entries];
    #pragma unroll
    for (unsigned slot = 0; slot < slots; ++slot) {
      base[slot] = stencil[slot] >= 0 ? layout.entry_index(vertex, stencil[slot]) : 0;
      #pragma unroll
      for (int component = 0; component < data::mass_tensor_entries; ++component)
        accumulated[slot][component] = Mass(0);
    }

    for (unsigned chunk = 0; chunk < count; chunk += width) {
      Compute rx = 0, ry = 0, rz = 0;
      Compute alpha_scaled[data::mass_tensor_entries]{};
      const unsigned offset = chunk + lane;
      if (offset < count) {
        ecsim::sample_mass_particle<Precision>(particles, cached_magnetic, source,
            beta_over_c, first + offset, invalid, rx, ry, rz, alpha_scaled);
      }

      const unsigned remaining = count - chunk;
      const unsigned populated = remaining < width ? remaining : width;
      for (unsigned source_lane = 0; source_lane < populated; ++source_lane) {
        const Compute x = subgroup::warp_shuffle(rx, source_lane);
        const Compute y = subgroup::warp_shuffle(ry, source_lane);
        const Compute z = subgroup::warp_shuffle(rz, source_lane);
        // Every row weight already exists in representative lanes, so
        // exchange those values instead of independently rebuilding each
        // column weight.
        const Compute wa = cic_corner_weight(corner_a, x, y, z);
        Compute pair_weight[slots];
        #pragma unroll
        for (unsigned slot = 0; slot < slots; ++slot) {
          const int b = corner_b[slot];
          const Compute wb = subgroup::warp_shuffle(
              wa, static_cast<unsigned>(b) * lanes_per_corner);
          pair_weight[slot] = wa * wb;
        }
        #pragma unroll
        for (int component = 0; component < data::mass_tensor_entries; ++component) {
          const Compute alpha = subgroup::warp_shuffle(alpha_scaled[component], source_lane);
          #pragma unroll
          for (unsigned slot = 0; slot < slots; ++slot) {
            if (stencil[slot] < 0) continue;
            // R1: narrow to Mass before the addition for mixed storage; when
            // Mass == Compute the compiler may contract this multiply-add.
            // Validated alpha and weights in [0,1] give a finite Compute
            // product. Narrowing/accumulation overflow leaves this cell sum
            // inf or NaN (including opposite-sign overflow cancellation),
            // so the final flush detects it without a per-product branch.
            const Mass amount = config::convert<Mass>(alpha * pair_weight[slot]);
            accumulated[slot][component] += amount;
          }
        }
      }
    }

    // Adjacent cells still share nodes. The final flush needs at most 36x9
    // global atomics per occupied cell, independent of particle occupancy.
    #pragma unroll
    for (unsigned slot = 0; slot < slots; ++slot)
      if (stencil[slot] >= 0) ecsim::add_mass_block(mass, base[slot], accumulated[slot], invalid);
    // Complete all cache readers before this warp starts its next cell.
    subgroup::warp_synchronize();
  });
}

} // namespace kernels

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_SCALAR_DEVICE_HPP
