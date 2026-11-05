#ifndef TPIC_ECSIM_FIELDS_WALL_FACES_HPP
#define TPIC_ECSIM_FIELDS_WALL_FACES_HPP

#include <lpfx/core/attributes.hpp>
#include <tpic/config/domain.hpp>

#include <utility>

namespace tpic::ecsim {

// Relative inverse nodal control volume. Multiply this by inverse_cell_volume
// when converting volume-free quantities, or apply it alone after deposition
// that already divided by the full cell volume. FIELD topology selects walls.
struct WallFaces {
  int cells[3]{};
  bool lower[3]{};
  bool upper[3]{};

  // Number of physical walls through node (i, j, k): at most one per axis.
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE unsigned boundary_count(int i, int j, int k) const noexcept {
    return (lower[0] && i == 0) + (upper[0] && i == cells[0]) +
           (lower[1] && j == 0) + (upper[1] && j == cells[1]) +
           (lower[2] && k == 0) + (upper[2] && k == cells[2]);
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE unsigned dual_volume_factor(int i, int j, int k) const noexcept {
    return 1U << boundary_count(i, j, k);
  }

  // 1 / dual_volume_factor(i, j, k), exact (a power of two). The division by
  // a power of two compiles to an exponent subtraction, without a branch.
  template <class Real>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE Real inverse_dual_volume_factor(
      int i, int j, int k) const noexcept {
    return Real(1) / Real(dual_volume_factor(i, j, k));
  }
};

[[nodiscard]] inline lpfx::ErrorCode try_prepare_wall_faces(
    const config::PreparedDomain& domain, WallFaces& result) noexcept {
  result = {};
  if (!domain.field_neighborhood.is_valid())
    return lpfx::ErrorCode::invalid_argument;
  const auto& neighborhood = domain.field_neighborhood;
  const auto cells = neighborhood.local_partition().local_cells();
  if (!std::cmp_equal(cells.x(), domain.local_cells[0]) ||
      !std::cmp_equal(cells.y(), domain.local_cells[1]) ||
      !std::cmp_equal(cells.z(), domain.local_cells[2]))
    return lpfx::ErrorCode::invalid_argument;
  for (int axis = 0; axis < 3; ++axis) {
    if (domain.local_cells[axis] <= 0) return lpfx::ErrorCode::invalid_argument;
    result.cells[axis] = domain.local_cells[axis];
    lpfx::CartesianDirection3 lower{}, upper{};
    if (axis == 0) { lower.x = -1; upper.x = 1; }
    if (axis == 1) { lower.y = -1; upper.y = 1; }
    if (axis == 2) { lower.z = -1; upper.z = 1; }
    result.lower[axis] = !neighborhood.topology().neighbor(lower).has_peer();
    result.upper[axis] = !neighborhood.topology().neighbor(upper).has_peer();
  }
  return lpfx::ErrorCode::success;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_WALL_FACES_HPP
