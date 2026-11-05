#ifndef TPIC_DATA_PARTICLES_PARTICLE_FRACTIONS_HPP
#define TPIC_DATA_PARTICLES_PARTICLE_FRACTIONS_HPP

#include <lpfx/core/attributes.hpp>
#include <tpic/data/particles/particle_schema.hpp>

#include <cstdint>

// The cell-relative fractions of a stored particle record: their load and
// the canonical-range predicate. Shared by the deposition, mass-matrix and
// mover kernels, the startup ownership pass and the output-moment kernel;
// each caller keeps its own validation order and reporting form. Both
// helpers are force-inlined so that the caller's code stays its own.
namespace tpic::data {

template <class Position> struct ParticleFractions {
  Position x, y, z;
};

template <class Precision, class ParticleView>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE ParticleFractions<typename Precision::Storage::Position>
load_fractions(ParticleView particles, std::uint32_t index) noexcept {
  using Particles = ParticleSchema<Precision>;
  return {particles.template get<typename Particles::Rx>()[index],
          particles.template get<typename Particles::Ry>()[index],
          particles.template get<typename Particles::Rz>()[index]};
}

// Canonical cell-relative fractions lie in [0, 1) on every axis. NaN fails.
template <class Fraction>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool in_unit_cell(
    Fraction x, Fraction y, Fraction z) noexcept {
  return x >= Fraction(0) && x < Fraction(1) && y >= Fraction(0) && y < Fraction(1) &&
      z >= Fraction(0) && z < Fraction(1);
}

} // namespace tpic::data

#endif // TPIC_DATA_PARTICLES_PARTICLE_FRACTIONS_HPP
