#ifndef TPIC_DATA_PARTICLES_PARTICLE_CELL_HPP
#define TPIC_DATA_PARTICLES_PARTICLE_CELL_HPP

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <lpfx/core/attributes.hpp>

#include <tpic/config/domain.hpp>
#include <tpic/config/precision.hpp>

namespace tpic::data {

struct GridIndex {
  int i, j, k;
};

// Rank-local particle keys index an x-fastest grid with
// config::PreparedDomain::particle_halo halo cells per side.
// They are never LPFx field allocation indices: decode before field access.
struct ParticleCellMap {
  int nx, ny, nz;
  // Halo-inclusive extents along x and y.
  int ext_nx, ext_ny;
  int count;

  // Local cell count along axis 0 (x), 1 (y) or 2 (z).
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr int extent(int axis) const noexcept {
    return axis == 0 ? nx : (axis == 1 ? ny : nz);
  }

  // Coordinates must lie in [-halo, local_cells + halo) on each axis.
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::int32_t encode(
      int i, int j, int k) const noexcept {
    constexpr int halo = config::PreparedDomain::particle_halo;
    return (i + halo) + ext_nx * ((j + halo) + ext_ny * (k + halo));
  }

  // key must lie in [0, count).
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr GridIndex decode(
      std::int32_t key) const noexcept {
    constexpr int halo = config::PreparedDomain::particle_halo;
    const int yz = key / ext_nx;
    return {key % ext_nx - halo, yz % ext_ny - halo, yz / ext_ny - halo};
  }

  // True for an owned (interior) cell.
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool contains_interior(
      int i, int j, int k) const noexcept {
    return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
  }

  // Compact x-fastest bin for owned cells; migrate halo particles first.
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr int interior_bin(
      int i, int j, int k) const noexcept {
    if (!contains_interior(i, j, k)) return -1;
    return i + nx * (j + ny * k);
  }
};

static_assert(std::is_trivial_v<ParticleCellMap> &&
              std::is_standard_layout_v<ParticleCellMap>);

// Precondition: a domain from config::prepare_domain, whose key count is
// already checked to fit int32. A hand-built domain without counts is rejected.
[[nodiscard]] inline ParticleCellMap prepare_particle_cell_map(
    const config::PreparedDomain& domain) {
  static_assert(config::PreparedDomain::particle_halo == 2);
  constexpr int halo_width = 2 * config::PreparedDomain::particle_halo;
  if (domain.particle_cell_count_with_halo <= 0)
    throw std::invalid_argument("TPIC particle cells: prepared domain counts must be positive");
  return {domain.local_cells[0], domain.local_cells[1], domain.local_cells[2],
          domain.local_cells[0] + halo_width, domain.local_cells[1] + halo_width,
          domain.particle_cell_count_with_halo};
}

// Startup sampler contract: fraction is finite and in [0,1], and cell has room
// for one increment. Carry both an exact RNG endpoint and a storage cast that
// rounds up to one, preserving the rounded position without an inward clamp.
// A carry across the local upper boundary deliberately produces a halo cell;
// migration/boundaries must run before interior binning and deposition.
template <class Position>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Position canonicalize_particle_fraction(
    int& cell, double fraction) noexcept {
  static_assert(std::is_floating_point_v<Position>);
  const Position stored = config::convert<Position>(fraction);
  if (stored == Position(1)) {
    ++cell;
    return Position(0);
  }
  return stored;
}

} // namespace tpic::data

#endif // TPIC_DATA_PARTICLES_PARTICLE_CELL_HPP
