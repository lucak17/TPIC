#ifndef TPIC_DATA_PARTICLES_PARTICLE_SCHEMA_HPP
#define TPIC_DATA_PARTICLES_PARTICLE_SCHEMA_HPP

#include <cstdint>
#include <lpfx/api/particle/schema.hpp>

namespace tpic::communication {

// The migration codec rebuilds the rank-local cell key from the global cell
// coordinates and the fractions from the canonical position. The policy is a
// schema contract; communication/particles/particle_wire.hpp defines the
// final-arrival rebuilders.
struct RebuildLocalCell;
template <unsigned Axis> struct RebuildParticleFraction;

} // namespace tpic::communication

namespace tpic::data {

template <class Precision> struct ParticleSchema {
  using Rx = lpfx::particle::attribute<"rx", typename Precision::Storage::Position,
      lpfx::RebuildLocally<communication::RebuildParticleFraction<0>>>;
  using Ry = lpfx::particle::attribute<"ry", typename Precision::Storage::Position,
      lpfx::RebuildLocally<communication::RebuildParticleFraction<1>>>;
  using Rz = lpfx::particle::attribute<"rz", typename Precision::Storage::Position,
      lpfx::RebuildLocally<communication::RebuildParticleFraction<2>>>;
  using U = lpfx::particle::attribute<"u", typename Precision::Storage::Velocity>;
  using V = lpfx::particle::attribute<"v", typename Precision::Storage::Velocity>;
  using W = lpfx::particle::attribute<"w", typename Precision::Storage::Velocity>;
  // Signed macrocharge; deposition and diagnostics use it with its sign.
  using Charge = lpfx::particle::attribute<"q", typename Precision::Storage::Weight>;
  using Cell = lpfx::particle::attribute<"cell", std::int32_t,
      lpfx::RebuildLocally<communication::RebuildLocalCell>>;
  using Id = lpfx::particle::attribute<"id", std::uint64_t>;
  using type = lpfx::particle::schema<Rx, Ry, Rz, U, V, W, Charge, Cell, Id>;
};

} // namespace tpic::data

#endif // TPIC_DATA_PARTICLES_PARTICLE_SCHEMA_HPP
