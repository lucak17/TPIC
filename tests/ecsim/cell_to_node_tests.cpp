#include <lpfx/api/field/field.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/ecsim/fields/cell_to_node.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Schema> auto make_field(int nx, int ny, int nz, int halo = 1) {
  auto made = lpfx::field::make<Schema>(lpfx::field::options{
      .cells = {nx, ny, nz}, .halo = halo});
  require(made.has_value(), "test field allocation failed");
  return std::move(made).value();
}

template <class Precision> void stencil_properties() {
  using F = tpic::data::FieldSchema<Precision>;
  auto cell = make_field<typename F::BCell>(7, 3, 1);
  auto node = make_field<typename F::BNode>(7, 3, 1);
  tpic::data::GridShape shape;
  require(tpic::ecsim::try_prepare_cell_to_node(cell, node, shape) == lpfx::ErrorCode::success &&
          shape.nx == 8 && shape.ny == 4 && shape.nz == 2 && shape.count == 64,
          "C2N logical node extents incorrect");
  // Averaging a trilinear polynomial at the eight cell centers must yield
  // its analytic value at the node, including the six faces and corners.
  const auto polynomial = [](double x, double y, double z) {
    return 1.0 + 2*x - 3*y + 4*z + 2*x*y - 4*y*z + 8*x*z + 16*x*y*z;
  };
  const auto visit = [&]<std::size_t C>() {
    const auto source = cell.view().template component<C>();
    for (int k = -1; k <= 1; ++k)
      for (int j = -1; j <= 3; ++j)
        for (int i = -1; i <= 7; ++i)
          source(i,j,k) = static_cast<typename Precision::Storage::Magnetic>(
              (C + 1) * polynomial(i + .5, j + .5, k + .5));
    for (int k = 0; k < shape.nz; ++k)
      for (int j = 0; j < shape.ny; ++j)
        for (int i = 0; i < shape.nx; ++i) {
          const auto average = tpic::ecsim::cell_to_node_average<double>(source, i,j,k);
          require(average == (C + 1) * polynomial(i,j,k), "C2N trilinear reproduction failed");
        }
    // Mixed storage must load into double before the sum loses a unit.
    const std::array<double, 8> cancellation{1.e8, 1., -1.e8, 0., 0., 0., 0., 0.};
    for (int c = 0; c < 8; ++c)
      source((c & 1) - 1, ((c >> 1) & 1) - 1, ((c >> 2) & 1) - 1) =
          static_cast<typename Precision::Storage::Magnetic>(cancellation[c]);
    require(tpic::ecsim::cell_to_node_average<double>(source, 0,0,0) == .125,
            "C2N loads were not promoted before addition");
  };
  visit.template operator()<0>();
  visit.template operator()<1>();
  visit.template operator()<2>();
}

void invalid_storage() {
  using F = tpic::data::FieldSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
  auto cell = make_field<typename F::BCell>(2, 3, 4);
  auto node = make_field<typename F::BNode>(2, 3, 4);
  auto other = make_field<typename F::BNode>(3, 3, 4);
  auto no_halo = make_field<typename F::BCell>(2, 3, 4, 0);
  lpfx::field::host<typename F::BCell> invalid;
  tpic::data::GridShape shape;
  require(tpic::ecsim::try_prepare_cell_to_node(invalid, node, shape) == lpfx::ErrorCode::invalid_state,
          "uninitialized cell field accepted");
  require(tpic::ecsim::try_prepare_cell_to_node(cell, other, shape) == lpfx::ErrorCode::invalid_argument,
          "mismatched field geometry accepted");
  require(tpic::ecsim::try_prepare_cell_to_node(no_halo, node, shape) == lpfx::ErrorCode::invalid_argument,
          "missing source halo accepted");
  const auto check = [&](int x, int y, int z, lpfx::ErrorCode expected) {
    lpfx::CartesianExtent3<std::uint32_t> cells;
    require(lpfx::try_make_cartesian_extent3(x,y,z,cells) == lpfx::ErrorCode::success,
            "test extent construction failed");
    require(tpic::data::try_make_grid_shape(cells, 1, {}, shape) == expected,
            "signed launch guard failed");
    if (expected != lpfx::ErrorCode::success)
      require(shape.count == 0, "failed preparation exposed launch work");
  };
  check(1,1,1, lpfx::ErrorCode::success);
  check(0,1,1, lpfx::ErrorCode::invalid_argument);
  check(2147483647,1,1, lpfx::ErrorCode::overflow);
  check(65535,65535,1, lpfx::ErrorCode::overflow);
}

} // namespace

int main() {
  try {
    stencil_properties<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>();
    stencil_properties<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>();
    stencil_properties<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>();
    invalid_storage();
    std::cout << "TPIC C2N polynomial, precision and extent checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
