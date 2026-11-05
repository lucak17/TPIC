#ifndef TPIC_ECSIM_PARTICLES_PARTICLE_SAMPLE_DEVICE_HPP
#define TPIC_ECSIM_PARTICLES_PARTICLE_SAMPLE_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/attributes.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/particles/particle_cell.hpp>
#include <tpic/data/particles/particle_fractions.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/ecsim/particles/cic.hpp>
#include <tpic/ecsim/particles/particle_response.hpp>

#include <cstdint>

// The per-particle sample shared by the current, mass-matrix and mover
// kernels: where the cell comes from (a compile-time tag) and the
// gather-and-response core. The mover takes the cell sources and the
// magnetic gather but builds the response after its electric gather (R1, see
// push_and_classify_particle). Each caller keeps its own load order,
// validation and early-return or branch-free structure around these pieces.
// Every helper is force-inlined so that contraction, decided after inlining,
// is the same as in the caller's own expression.
namespace tpic::ecsim {

// Cell source of an unsorted record: range test, decode and interior test of
// its stored key, in this order.
struct DecodeKey {
  data::ParticleCellMap cells;

  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE bool resolve(
      std::int32_t key, data::GridIndex& cell) const noexcept {
    if (key < 0 || key >= cells.count) return false;
    cell = cells.decode(key);
    return cells.contains_interior(cell.i, cell.j, cell.k);
  }
};

// Cell source of a sorted record: the caller opened an interior bin and holds
// its decoded cell. The bin is below bins.view.bin_count, which
// device_wait_particle_bins or device_wait_resident_particle_bins checked
// against cells.count, so key == bin implies the range and interior tests of
// DecodeKey.
struct KnownBin {
  std::int32_t bin;
  data::GridIndex cell;

  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE bool resolve(
      std::int32_t key, data::GridIndex& resolved) const noexcept {
    resolved = cell;
    return key == bin;
  }
};

template <class Compute> struct CicVector {
  Compute x, y, z;
};

// Nodal B at the particle: Bnx, Bny, Bnz with the deposition weights.
template <class Precision, class Compute, class MagneticView>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE CicVector<Compute> gather_cic_magnetic(
    MagneticView magnetic, data::GridIndex cell, const CicWeights<Compute>& weights) noexcept {
  using Fields = data::FieldSchema<Precision>;
  return {interpolate_cic(magnetic.template get<typename Fields::Bnx>(), cell, weights),
          interpolate_cic(magnetic.template get<typename Fields::Bny>(), cell, weights),
          interpolate_cic(magnetic.template get<typename Fields::Bnz>(), cell, weights)};
}

template <class Compute> struct ParticleSample {
  CicWeights<Compute> weights;
  CicVector<Compute> magnetic;
  ParticleResponse<Compute> response;
};

// The gather-and-response core: CIC weights of the canonical fractions, the
// gathered B and the response tensor built from it. Callers validate the
// fractions first.
template <class Precision, class Compute, class MagneticView, class Fraction>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE ParticleSample<Compute> sample_particle(
    MagneticView magnetic, data::GridIndex cell, Fraction x, Fraction y, Fraction z,
    Compute beta_over_c) noexcept {
  const auto weights = make_cic_weights<Compute>(x, y, z);
  const auto b = gather_cic_magnetic<Precision>(magnetic, cell, weights);
  return {weights, b, ParticleResponse<Compute>(b.x, b.y, b.z, beta_over_c)};
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_PARTICLES_PARTICLE_SAMPLE_DEVICE_HPP
