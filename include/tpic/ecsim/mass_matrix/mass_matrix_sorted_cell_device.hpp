#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_SORTED_CELL_DEVICE_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_SORTED_CELL_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_stencil.hpp>
#include <tpic/ecsim/particles/cic.hpp>
#include <tpic/ecsim/particles/particle_response.hpp>
#include <tpic/ecsim/particles/particle_sample_device.hpp>
#include <tpic/ecsim/particles/sorted_cell_device.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tpic::ecsim {

// The native subgroup width selects one of these compile-time-generated maps.
// CUDA uses the 32x2 map for both scalar and matrix-core deposition; AMD scalar
// deposition selects the 32x2 or 64x1 map for the target's wavefront width.
static __device__ __constant__ MassLaneStencilMap<MassSubgroupLayout<32>>
    device_mass_lane_stencil_map_32 = make_mass_lane_stencil_map<MassSubgroupLayout<32>>();
static __device__ __constant__ MassLaneStencilMap<MassSubgroupLayout<64>>
    device_mass_lane_stencil_map_64 = make_mass_lane_stencil_map<MassSubgroupLayout<64>>();

template <unsigned Width>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE int mass_lane_stencil(
    unsigned lane, unsigned slot) noexcept {
  static_assert(Width == 32 || Width == 64);
  if constexpr (Width == 32)
    return device_mass_lane_stencil_map_32.value[lane][slot];
  else
    return device_mass_lane_stencil_map_64.value[lane][slot];
}

// The cell loop of the sorted scalar, mma and mfma4 kernels: each subgroup takes
// bin blockIdx.x * CellsPerBlock + warp, grid-strided, and calls
// body(bin, first, count, cell) for every occupied interior bin. Lane 0 reads
// the bin range and decodes the cell, and the subgroup receives both by
// shuffles; empty and rejected bins and their reporting are those of
// open_sorted_cell. R1: the uniform per-lane open_sorted_cell, or this open
// as a bool-returning helper, costs these kernels registers (CUDA scalar mixed
// storage 72 -> 80, 28 -> 24 resident warps; matrix-core 124 -> 128) and
// per-cell instructions.
template <unsigned CellsPerBlock, class Body>
LPFX_DEVICE LPFX_ALWAYS_INLINE void for_each_sorted_cell_broadcast(
    const ParticleBins& bins, data::ParticleCellMap cells, unsigned warp,
    unsigned* invalid, Body&& body) {
  namespace subgroup = lpfx::device::subgroup;
  const bool leader = subgroup::warp_lane() == 0;
  for (unsigned bin = blockIdx.x * CellsPerBlock + warp;
       bin < bins.view.bin_count; bin += gridDim.x * CellsPerBlock) {
    unsigned first = 0, count = 0, range_valid = 0;
    if (leader) range_valid = bins.range(bin, first, count, invalid);
    range_valid = subgroup::warp_shuffle(range_valid, 0);
    first = subgroup::warp_shuffle(first, 0);
    count = subgroup::warp_shuffle(count, 0);
    if (!range_valid || count == 0) continue;
    data::GridIndex cell{};
    if (leader) cell = cells.decode(static_cast<int>(bin));
    cell.i = subgroup::warp_shuffle(cell.i, 0);
    cell.j = subgroup::warp_shuffle(cell.j, 0);
    cell.k = subgroup::warp_shuffle(cell.k, 0);
    if (cells.interior_bin(cell.i, cell.j, cell.k) < 0) {
      if (leader) atomicExch(invalid, 1U);
      continue;
    }
    body(bin, first, count, cell);
  }
}

// Per-particle staging of the sorted scalar and matrix-core kernels: the
// fractions and the scaled response alpha = a * (beta_over_c * charge).
// No early return on purpose: every lane loads its record, and an invalid record
// supplies zero fractions and alpha so that every lane can still join the
// following shuffles and matrix instructions.
template <class Precision, class ParticleView, class MagneticView>
LPFX_DEVICE LPFX_ALWAYS_INLINE void sample_mass_particle(
    ParticleView particles, MagneticView magnetic, KnownBin source,
    typename Precision::Compute::Mass beta_over_c,
    std::uint32_t index, unsigned* invalid,
    typename Precision::Compute::Mass& rx,
    typename Precision::Compute::Mass& ry,
    typename Precision::Compute::Mass& rz,
    typename Precision::Compute::Mass (&alpha_scaled)[data::mass_tensor_entries]) noexcept {
  using Particles = data::ParticleSchema<Precision>;
  using Compute = typename Precision::Compute::Mass;
  const auto key = particles.template get<typename Particles::Cell>()[index];
  const auto fraction = data::load_fractions<Precision>(particles, index);
  rx = config::convert<Compute>(fraction.x);
  ry = config::convert<Compute>(fraction.y);
  rz = config::convert<Compute>(fraction.z);
  data::GridIndex cell;
  // R1: '&', not '&&': the key and fraction loads issue together instead of
  // the fraction loads waiting for the key comparison.
  // The stored fractions are tested: a fraction below one can round to one in a
  // narrower compute type, which is still a valid weight.
  bool valid = source.resolve(key, cell) & data::in_unit_cell(fraction.x, fraction.y, fraction.z);
  if (valid) {
    const auto sample = sample_particle<Precision>(magnetic, cell, rx, ry, rz, beta_over_c);
    const Compute scale = beta_over_c *
        config::convert<Compute>(particles.template get<typename Particles::Charge>()[index]);
    // A finite norm_square proves that the response denominator and all nine
    // bounded response entries are finite.
    valid = ::isfinite(sample.response.norm_square) && ::isfinite(scale);
    #pragma unroll
    for (int component = 0; component < data::mass_tensor_entries; ++component)
      alpha_scaled[component] = sample.response.a[component] * scale;
  }
  if (!valid) {
    atomicExch(invalid, 1U);
    rx = ry = rz = Compute(0);
    #pragma unroll
    for (int component = 0; component < data::mass_tensor_entries; ++component)
      alpha_scaled[component] = Compute(0);
  }
}

// Flush one 3x3 mass block of a canonical spatial pair: narrow each entry to
// Mass, flag a non-finite one, add a nonzero one. The caller passes the nine
// entries as a register array, whatever its accumulator layout.
template <class Mass, class Value>
LPFX_DEVICE LPFX_ALWAYS_INLINE void add_mass_block(
    Mass* mass, std::size_t base, const Value (&values)[data::mass_tensor_entries],
    unsigned* invalid) noexcept {
  #pragma unroll
  for (int component = 0; component < data::mass_tensor_entries; ++component) {
    const Mass amount = config::convert<Mass>(values[component]);
    if (!::isfinite(amount)) atomicExch(invalid, 1U);
    else if (amount != Mass(0)) atomicAdd(mass + base + component, amount);
  }
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_SORTED_CELL_DEVICE_HPP
