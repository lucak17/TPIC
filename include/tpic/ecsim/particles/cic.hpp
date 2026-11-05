#ifndef TPIC_ECSIM_PARTICLES_CIC_HPP
#define TPIC_ECSIM_PARTICLES_CIC_HPP

#include <lpfx/core/attributes.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/particles/particle_cell.hpp>

#include <type_traits>

namespace tpic::ecsim {

// A CIC stencil touches the eight corners of its cell, in cic_corner order;
// the mass matrix couples every ordered pair of them.
inline constexpr int cic_corner_count = 8;
inline constexpr int cic_corner_pair_count = cic_corner_count * cic_corner_count;

// Preserve the integer cell and stored fractions separately. Adding a large
// cell coordinate to a fraction before making an LPFx stencil loses precision.
template <class Compute> struct CicWeights {
  Compute value[cic_corner_count];
};

template <class Compute, class Position>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE CicWeights<Compute> make_cic_weights(
    Position rx, Position ry, Position rz) noexcept {
  static_assert(std::is_floating_point_v<Compute>);
  // Callers validate canonical [0,1) fractions before gathering. Retain LPFx's
  // corner order and (wx * wy) * wz grouping without decomposing the already
  // relative coordinates into integer bases and fractions a second time.
  const Compute x = config::convert<Compute>(rx), y = config::convert<Compute>(ry),
                z = config::convert<Compute>(rz);
  const Compute cx = Compute(1) - x, cy = Compute(1) - y, cz = Compute(1) - z;
  return {{cx * cy * cz, x * cy * cz, cx * y * cz, x * y * cz,
           cx * cy * z, x * cy * z, cx * y * z, x * y * z}};
}

// One CIC corner weight from the cell-relative fractions, with the same
// left-associative (wx * wy) * wz grouping and complements as make_cic_weights.
template <class Compute>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Compute cic_corner_weight(
    int corner, Compute x, Compute y, Compute z) noexcept {
  return ((corner & 1) ? x : Compute(1) - x) *
      ((corner & 2) ? y : Compute(1) - y) *
      ((corner & 4) ? z : Compute(1) - z);
}

// A finite Compute value times a CIC weight in [0, 1] is finite in Compute;
// only narrowing the product to a smaller storage type can overflow. Product
// checks exist only where this is true.
template <class Storage, class Compute>
inline constexpr bool narrowing_may_overflow = sizeof(Storage) < sizeof(Compute);

[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr data::GridIndex cic_corner(
    data::GridIndex cell, int corner) noexcept {
  return {cell.i + (corner & 1), cell.j + ((corner >> 1) & 1),
          cell.k + ((corner >> 2) & 1)};
}

template <class Compute, class Component>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Compute interpolate_cic(
    Component field, data::GridIndex cell,
    const CicWeights<Compute>& weights) noexcept {
  static_assert(std::is_same_v<typename Component::location_type, lpfx::Node>);
  Compute value = 0;
  // Sum the eight corners x-fastest with the deposition weights; LPFx's nested
  // lerp gather has different rounding and does not reuse these weights.
  for (int corner = 0; corner < cic_corner_count; ++corner) {
    const auto node = cic_corner(cell, corner);
    value += weights.value[corner] * config::convert<Compute>(field(node.i, node.j, node.k));
  }
  return value;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_PARTICLES_CIC_HPP
