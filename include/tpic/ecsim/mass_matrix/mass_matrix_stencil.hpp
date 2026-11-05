#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_STENCIL_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_STENCIL_HPP

#include <tpic/config/precision.hpp>
#include <tpic/ecsim/particles/cic.hpp>
#include <tpic/data/mass_matrix/mass_matrix_layout.hpp>

#include <type_traits>

namespace tpic::ecsim {

namespace detail {
struct MassPairStencils { int value[8][8]{}; };

[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr MassPairStencils make_mass_pair_stencils() noexcept {
  MassPairStencils result;
  for (int a = 0; a < 8; ++a) {
    for (int b = 0; b < 8; ++b) {
      const auto first = cic_corner({}, a), second = cic_corner({}, b);
      int index = -1;
      for (int stencil = 0; stencil < data::mass_stencil_count; ++stencil) {
        const auto offset = data::mass_stencil_offset(stencil);
        if (offset.i == second.i - first.i &&
            offset.j == second.j - first.j &&
            offset.k == second.k - first.k) index = stencil;
      }
      result.value[a][b] = index;
    }
  }
  return result;
}
} // namespace detail

// Generated from the one canonical offset definition, shared by deposition
// and apply. Negative means that the reverse spatial pair owns the block.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr int mass_pair_stencil(int a, int b) noexcept {
  constexpr auto pairs = detail::make_mass_pair_stencils();
  return pairs.value[a][b];
}

static_assert([] {
  constexpr auto pairs = detail::make_mass_pair_stencils();
  int canonical_pairs = 0;
  for (int a = 0; a < 8; ++a) {
    for (int b = 0; b < 8; ++b) {
      canonical_pairs += pairs.value[a][b] >= 0;
      if (a == b) {
        if (pairs.value[a][b] != 0) return false;
      } else if ((pairs.value[a][b] >= 0) ==
                 (pairs.value[b][a] >= 0)) return false;
    }
  }
  return canonical_pairs == 36;
}());

// Lane ownership of the rectangular 8x8 corner-pair product in the sorted
// scalar and PTX mma kernels: lane owns first corner output_a(lane) and the
// slots consecutive second corners output_b(lane, slot). Width 32 gives two
// slots per lane, width 64 one.
template <unsigned Width> struct MassSubgroupLayout {
  static_assert(Width == 32 || Width == 64);
  static constexpr unsigned width = Width;
  static constexpr unsigned slots = cic_corner_pair_count / Width;
  static constexpr unsigned lanes_per_corner = Width / cic_corner_count;
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr int output_a(unsigned lane) {
    return static_cast<int>(lane / lanes_per_corner);
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr int output_b(unsigned lane, unsigned slot) {
    return static_cast<int>(slots * (lane % lanes_per_corner) + slot);
  }
};

// The canonical stencil of each lane's output pair, or -1 where the reverse
// pair owns the block. Layout gives width, slots, output_a and output_b.
template <class Layout> struct MassLaneStencilMap {
  int value[Layout::width][Layout::slots]{};
};

// The one lane-table generator of the sorted mass kernels.
template <class Layout>
[[nodiscard]] constexpr MassLaneStencilMap<Layout> make_mass_lane_stencil_map() noexcept {
  MassLaneStencilMap<Layout> result;
  for (unsigned lane = 0; lane < Layout::width; ++lane)
    for (unsigned slot = 0; slot < Layout::slots; ++slot)
      result.value[lane][slot] =
          mass_pair_stencil(Layout::output_a(lane), Layout::output_b(lane, slot));
  return result;
}

// Every canonical pair is owned by exactly one lane slot.
template <class Layout>
[[nodiscard]] constexpr bool valid_mass_lane_stencil_map() noexcept {
  constexpr auto map = make_mass_lane_stencil_map<Layout>();
  int canonical_pairs = 0;
  for (unsigned lane = 0; lane < Layout::width; ++lane)
    for (unsigned slot = 0; slot < Layout::slots; ++slot)
      canonical_pairs += map.value[lane][slot] >= 0;
  return canonical_pairs == 36;
}

static_assert(valid_mass_lane_stencil_map<MassSubgroupLayout<32>>());
static_assert(valid_mass_lane_stencil_map<MassSubgroupLayout<64>>());

namespace detail {
template <class Compute, class Mass>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void mass_block_product(
    const Mass* block, Compute vx, Compute vy, Compute vz,
    Compute& x, Compute& y, Compute& z) noexcept {
  x = vx * config::convert<Compute>(block[0]) + vy * config::convert<Compute>(block[1]) + vz * config::convert<Compute>(block[2]);
  y = vx * config::convert<Compute>(block[3]) + vy * config::convert<Compute>(block[4]) + vz * config::convert<Compute>(block[5]);
  z = vx * config::convert<Compute>(block[6]) + vy * config::convert<Compute>(block[7]) + vz * config::convert<Compute>(block[8]);
}
} // namespace detail

// Reconstruct center + 13 forward + 13 backward blocks. Spatial symmetry
// never transposes the generally nonsymmetric alpha tensor. Input fields need
// coherent one-node ghosts and the coefficients need additive assembly/copy;
// the caller supplies row inverse volume including physical dual-volume factors.
template <class Compute, class Mass, class Vx, class Vy, class Vz>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void apply_mass_matrix_node(
    const Mass* coefficients, data::MassMatrixLayout layout, Vx vx, Vy vy, Vz vz,
    int i, int j, int k, Compute row_inverse_volume,
    Compute& x, Compute& y, Compute& z) noexcept {
  static_assert(std::is_floating_point_v<Compute>);
  const auto vertex = layout.vertex_index(i, j, k);
  detail::mass_block_product(coefficients + layout.entry_index(vertex, 0),
      config::convert<Compute>(vx(i,j,k)), config::convert<Compute>(vy(i,j,k)),
      config::convert<Compute>(vz(i,j,k)), x, y, z);
  #pragma unroll
  for (int stencil = 1; stencil < data::mass_stencil_count; ++stencil) {
    const auto offset = data::mass_stencil_offset(stencil);
    // The block of entry_vertex's stencil entry times the input at (ni, nj, nk).
    const auto accumulate = [&](const auto entry_vertex, const int ni, const int nj, const int nk) {
      Compute bx, by, bz;
      detail::mass_block_product(coefficients + layout.entry_index(entry_vertex, stencil),
          config::convert<Compute>(vx(ni,nj,nk)), config::convert<Compute>(vy(ni,nj,nk)),
          config::convert<Compute>(vz(ni,nj,nk)), bx, by, bz);
      x += bx; y += by; z += bz;
    };
    // Forward neighbour with this vertex's entry, then backward neighbour with
    // the neighbour vertex's entry: this order is part of the apply contract.
    accumulate(vertex, i + offset.i, j + offset.j, k + offset.k);
    const int ni = i - offset.i, nj = j - offset.j, nk = k - offset.k;
    accumulate(layout.vertex_index(ni, nj, nk), ni, nj, nk);
  }
  x *= row_inverse_volume;
  y *= row_inverse_volume;
  z *= row_inverse_volume;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_STENCIL_HPP
