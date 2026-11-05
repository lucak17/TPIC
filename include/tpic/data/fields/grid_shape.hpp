#ifndef TPIC_DATA_FIELDS_GRID_SHAPE_HPP
#define TPIC_DATA_FIELDS_GRID_SHAPE_HPP

#include <lpfx/core/error.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace tpic::data {

// The extents of one grid view and their product, passed by value to kernels.
struct GridShape {
  int nx = 0, ny = 0, nz = 0;
  int count = 0;
};
static_assert(std::is_standard_layout_v<GridShape> && sizeof(GridShape) == 16,
              "GridShape is a by-value kernel argument of four int");

// Ghost layers below and above the interior on each axis.
struct GridHalo {
  int lower[3]{};
  int upper[3]{};
};

// The one checked shape factory. Each extent is cells + nodal + lower + upper.
// A non-positive cell count is invalid_argument; a cell count of INT_MAX (the
// node stencil addresses n + 1 signed), an extent or the product above INT_MAX
// is overflow.
template <class Cells>
[[nodiscard]] lpfx::ErrorCode try_make_grid_shape(
    Cells cells, int nodal, const GridHalo& halo, GridShape& shape) noexcept {
  shape = {};
  constexpr auto limit = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  std::uint64_t count = 1;
  int extents[3]{};
  int axis = 0;
  for (const auto n : {cells.x(), cells.y(), cells.z()}) {
    if (n <= 0) return lpfx::ErrorCode::invalid_argument;
    if (static_cast<std::uint64_t>(n) >= limit) return lpfx::ErrorCode::overflow;
    const auto extent = static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(nodal) +
                        static_cast<std::uint64_t>(halo.lower[axis]) +
                        static_cast<std::uint64_t>(halo.upper[axis]);
    if (extent > limit || count > limit / extent) return lpfx::ErrorCode::overflow;
    count *= extent;
    extents[axis++] = static_cast<int>(extent);
  }
  shape = {extents[0], extents[1], extents[2], static_cast<int>(count)};
  return lpfx::ErrorCode::success;
}

} // namespace tpic::data

#endif // TPIC_DATA_FIELDS_GRID_SHAPE_HPP
