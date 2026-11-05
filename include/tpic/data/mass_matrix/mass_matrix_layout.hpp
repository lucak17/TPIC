#ifndef TPIC_DATA_MASS_MATRIX_MASS_MATRIX_LAYOUT_HPP
#define TPIC_DATA_MASS_MATRIX_MASS_MATRIX_LAYOUT_HPP

#include <tpic/data/particles/particle_cell.hpp>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace tpic::data {

inline constexpr int mass_stencil_count = 14;
inline constexpr int mass_tensor_entries = 9;
inline constexpr int mass_entries_per_node = mass_stencil_count * mass_tensor_entries;

// Canonical spatial-pair order of the stored mass stencil, one entry per
// symmetric node pair; this order is part of the mass layout. The (-1,1,1)
// entry is intentional; canonical does not mean lexicographically positive.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr data::GridIndex mass_stencil_offset(
    int stencil) noexcept {
  constexpr data::GridIndex offsets[mass_stencil_count] = {
      {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,0}, {1,-1,0},
      {1,0,1}, {1,0,-1}, {0,1,1}, {0,1,-1}, {1,-1,1},
      {1,1,-1}, {-1,1,1}, {1,1,1}};
  return offsets[stencil];
}

// A contiguous record of mass_entries_per_node scalars per logical node. Its one-node
// halo and x-fastest record strides are independent of LPFx field pitches.
// Construct through prepare_mass_matrix_layout before using indices.
struct MassMatrixLayout {
  int nx = 0, ny = 0, nz = 0;
  int extent_x = 0, extent_y = 0, extent_z = 0;
  int stride_y = 0, stride_z = 0;
  int interior_count = 0, vertex_count = 0;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::size_t vertex_index(
      int i, int j, int k) const noexcept {
    return static_cast<std::size_t>((i + 1) + (j + 1) * stride_y + (k + 1) * stride_z);
  }
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::size_t entry_index(
      std::size_t vertex, int stencil) const noexcept {
    return vertex * mass_entries_per_node + stencil * mass_tensor_entries;
  }
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::size_t storage_size() const noexcept {
    return static_cast<std::size_t>(vertex_count) * mass_entries_per_node;
  }
  [[nodiscard]] constexpr bool operator==(const MassMatrixLayout&) const noexcept = default;
};
// Passed by value to kernels.
static_assert(std::is_trivially_copyable_v<MassMatrixLayout> &&
              std::is_standard_layout_v<MassMatrixLayout>);

// Precondition: a domain from config::prepare_domain, whose node counts are
// already checked to fit int. A hand-built domain without counts is rejected.
// vertex_index spells the one-node field halo as a literal.
[[nodiscard]] inline MassMatrixLayout prepare_mass_matrix_layout(
    const config::PreparedDomain& domain) {
  static_assert(config::PreparedDomain::field_halo == 1);
  constexpr int nodal_halo_width = 1 + 2 * config::PreparedDomain::field_halo;
  if (domain.interior_node_count <= 0 || domain.field_node_count_with_halo <= 0)
    throw std::invalid_argument("TPIC mass matrix layout: prepared domain counts must be positive");
  const int extent_x = domain.local_cells[0] + nodal_halo_width;
  const int extent_y = domain.local_cells[1] + nodal_halo_width;
  const int extent_z = domain.local_cells[2] + nodal_halo_width;
  return {domain.local_cells[0] + 1, domain.local_cells[1] + 1, domain.local_cells[2] + 1,
          extent_x, extent_y, extent_z, extent_x, extent_x * extent_y,
          domain.interior_node_count, domain.field_node_count_with_halo};
}

} // namespace tpic::data

#endif // TPIC_DATA_MASS_MATRIX_MASS_MATRIX_LAYOUT_HPP
