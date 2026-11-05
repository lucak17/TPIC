#ifndef TPIC_ECSIM_FIELDS_STENCIL_REGION_HPP
#define TPIC_ECSIM_FIELDS_STENCIL_REGION_HPP

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <tpic/device/thread_indexing.hpp>

#include <cstdint>
#include <limits>

namespace tpic::ecsim {

// Target sample regions of a stencil launch. full is every sample;
// strict_interior is the samples whose stencil reads no source ghost; shell is
// the rest. strict_interior may run before or during the source ghost exchange
// (which writes ghosts only) and concurrently with shell on another executor;
// full and shell need completed source ghosts. Both parts complete before any
// whole-target consumer. A one-cell local extent makes strict_interior empty.
// Geometry describes logical samples, never their ghosts or a field's pitches;
// prepare it from checked owners.
enum class StencilRegion { full, strict_interior, shell };

struct StencilRegionGeometry {
  int nx{}, ny{}, nz{};
  unsigned full_count{}, interior_count{}, shell_count{};
};

[[nodiscard]] inline lpfx::ErrorCode try_make_stencil_region_geometry(
    int nx, int ny, int nz, StencilRegionGeometry& output) noexcept {
  output = {};
  constexpr auto limit = std::uint64_t(std::numeric_limits<int>::max());
  std::uint64_t full = 1, interior = 1;
  const int extents[] = {nx, ny, nz};
  for (const int extent : extents) {
    if (extent <= 0) return lpfx::ErrorCode::invalid_argument;
    if (full > limit / std::uint64_t(extent)) return lpfx::ErrorCode::overflow;
    full *= std::uint64_t(extent);
    interior *= std::uint64_t(extent > 2 ? extent - 2 : 0);
  }
  output = {nx, ny, nz, unsigned(full), unsigned(interior), unsigned(full - interior)};
  return lpfx::ErrorCode::success;
}

template<StencilRegion Region>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr unsigned stencil_region_count(
    StencilRegionGeometry geometry) noexcept {
  static_assert(Region == StencilRegion::full || Region == StencilRegion::strict_interior ||
                Region == StencilRegion::shell);
  if constexpr (Region == StencilRegion::strict_interior) return geometry.interior_count;
  if constexpr (Region == StencilRegion::shell) return geometry.shell_count;
  return geometry.full_count;
}

// Requires a prepared geometry and index < stencil_region_count<Region>(). An
// empty interior has no coordinates: callers return before launch/division.
// Every branch is an x-fastest decode of its own box, shifted into place.
template<StencilRegion Region>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void stencil_region_coordinates(unsigned index,
    StencilRegionGeometry geometry, int& i, int& j, int& k) noexcept {
  static_assert(Region == StencilRegion::full || Region == StencilRegion::strict_interior ||
                Region == StencilRegion::shell);
  if constexpr (Region == StencilRegion::strict_interior) {
    const auto at = device::decode_x_fastest(index, unsigned(geometry.nx - 2),
                                             unsigned(geometry.ny - 2));
    i = at.i + 1;
    j = at.j + 1;
    k = at.k + 1;
    return;
  }
  if constexpr (Region == StencilRegion::shell) {
    // Also handles one-/two-sample axes without duplicating opposite faces.
    if (geometry.interior_count != 0) {
      const unsigned z_faces = 2U * unsigned(geometry.nx) * unsigned(geometry.ny);
      const unsigned y_faces = 2U * unsigned(geometry.nx) * unsigned(geometry.nz - 2);
      if (index < z_faces) {
        const auto at = device::decode_x_fastest(index, unsigned(geometry.nx),
                                                 unsigned(geometry.ny));
        i = at.i;
        j = at.j;
        k = at.k * (geometry.nz - 1);
      } else if ((index -= z_faces) < y_faces) {
        const auto at = device::decode_x_fastest(index, unsigned(geometry.nx), 2U);
        i = at.i;
        j = at.j * (geometry.ny - 1);
        k = at.k + 1;
      } else {
        index -= y_faces;
        const auto at = device::decode_x_fastest(index, 2U, unsigned(geometry.ny - 2));
        i = at.i * (geometry.nx - 1);
        j = at.j + 1;
        k = at.k + 1;
      }
      return;
    }
  }
  const auto at = device::decode_x_fastest(index, unsigned(geometry.nx), unsigned(geometry.ny));
  i = at.i;
  j = at.j;
  k = at.k;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_STENCIL_REGION_HPP
