#include <tpic/ecsim/fields/stencil_region.hpp>

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace tpic::ecsim;

void require(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}

void partition() {
  for (int nx = 1; nx <= 9; ++nx)
    for (int ny = 1; ny <= 7; ++ny)
      for (int nz = 1; nz <= 5; ++nz) {
        StencilRegionGeometry geometry;
        require(try_make_stencil_region_geometry(nx, ny, nz, geometry) == lpfx::ErrorCode::success,
                "valid stencil region geometry rejected");
        require(geometry.full_count == unsigned(nx * ny * nz) &&
                geometry.interior_count + geometry.shell_count == geometry.full_count,
                "stencil region counts do not partition full samples");
        std::vector<unsigned> visits(geometry.full_count);
        const auto visit = [&]<StencilRegion Region>() {
          for (unsigned n = 0; n < stencil_region_count<Region>(geometry); ++n) {
            int i, j, k;
            stencil_region_coordinates<Region>(n, geometry, i, j, k);
            require(i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz,
                    "stencil region escaped logical samples");
            const bool inside = i > 0 && i < nx - 1 && j > 0 && j < ny - 1 &&
                                k > 0 && k < nz - 1;
            require(inside == (Region == StencilRegion::strict_interior),
                    "stencil region has incorrect membership");
            ++visits[unsigned(i + nx * (j + ny * k))];
          }
        };
        visit.template operator()<StencilRegion::strict_interior>();
        visit.template operator()<StencilRegion::shell>();
        for (unsigned n = 0; n < geometry.full_count; ++n) {
          require(visits[n] == 1, "stencil regions overlap or omit samples");
          int i, j, k;
          stencil_region_coordinates<StencilRegion::full>(n, geometry, i, j, k);
          require(unsigned(i + nx * (j + ny * k)) == n, "full stencil order is not x-fastest");
          if (geometry.interior_count == 0) {
            int si, sj, sk;
            stencil_region_coordinates<StencilRegion::shell>(n, geometry, si, sj, sk);
            require(i == si && j == sj && k == sk, "empty-core shell does not use full indexing");
          }
        }
      }
}

void checked_geometry() {
  constexpr int maximum = std::numeric_limits<int>::max();
  for (const auto extents : {std::array{0, 2, 3}, std::array{2, -1, 3},
                             std::array{2, 3, 0}}) {
    StencilRegionGeometry geometry{1, 2, 3, 4, 5, 6};
    require(try_make_stencil_region_geometry(extents[0], extents[1], extents[2], geometry) ==
            lpfx::ErrorCode::invalid_argument && geometry.full_count == 0 && geometry.nx == 0,
            "invalid geometry was admitted or left a published result");
  }
  for (const auto extents : {std::array{maximum, 2, 1}, std::array{1291, 1291, 1291},
                             std::array{maximum, maximum, maximum}}) {
    StencilRegionGeometry geometry;
    require(try_make_stencil_region_geometry(extents[0], extents[1], extents[2], geometry) ==
            lpfx::ErrorCode::overflow && geometry.full_count == 0,
            "overflowing region geometry accepted");
  }
  for (const auto extents : {std::array{maximum, 1, 1}, std::array{1, maximum, 1},
                             std::array{1, 1, maximum}, std::array{1290, 1290, 1290}}) {
    StencilRegionGeometry geometry;
    require(try_make_stencil_region_geometry(extents[0], extents[1], extents[2], geometry) ==
            lpfx::ErrorCode::success, "representable large geometry rejected");
    for (unsigned n : {0U, geometry.shell_count - 1}) {
      int i, j, k;
      stencil_region_coordinates<StencilRegion::shell>(n, geometry, i, j, k);
      require(i >= 0 && i < geometry.nx && j >= 0 && j < geometry.ny && k >= 0 && k < geometry.nz,
              "large shell indexing overflowed");
    }
  }
}
} // namespace

int main() {
  try {
    partition();
    checked_geometry();
    std::cout << "TPIC stencil region tests passed\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
