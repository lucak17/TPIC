// Properties of the run-constant launch inputs prepared once per run: the
// field wall plans and their owner checks, and the smoothing sums.
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/ecsim/fields/field_boundary.hpp>
#include <tpic/ecsim/fields/node_smoothing.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tpic;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

config::PreparedDomain domain(bool periodic, int rank = 0, int ranks = 1) {
  config::GridConfig grid;
  grid.nxc = 6; grid.nyc = 5; grid.nzc = 4;
  grid.lx = 3; grid.ly = 2.5; grid.lz = 2;
  config::TopologyConfig topology;
  topology.xlen = ranks;
  topology.periodic_x = topology.periodic_y = topology.periodic_z = periodic;
  return config::prepare_domain(grid, topology, rank, ranks);
}

config::PreparedPhysics walled_physics() {
  config::PreparedPhysics physics{};
  for (int face = 0; face < 6; ++face)
    physics.field_boundary[face] = face < 2 ? config::FieldBoundary::perfect_magnetic_conductor
                                            : config::FieldBoundary::perfect_electric_conductor;
  return physics;
}

// The owner surface the wall-plan owner check reads.
struct Halo { int x_minus, x_plus, y_minus, y_plus, z_minus, z_plus; };
template <class Location> struct Component { using location_type = Location; };
template <class Location> struct View {
  template <std::size_t> Component<Location> component() const { return {}; }
};
template <class Location, class Cells> struct Owner {
  static constexpr std::size_t component_count = 3;
  Cells extent;
  int halo;
  bool initialized() const { return true; }
  Cells cells() const { return extent; }
  View<Location> view() const { return {}; }
  template <std::size_t> Halo component_halo() const { return {halo, halo, halo, halo, halo, halo}; }
};

void boundary_plans() {
  for (const bool periodic : {true, false}) {
    const auto d = domain(periodic);
    const auto physics = walled_physics();
    const auto cells = d.field_neighborhood.local_partition().local_cells();
    const auto plans = ecsim::prepare_field_boundary_plans(d, physics);
    const auto check = [&]<class Location>(const ecsim::FieldBoundaryPlan& plan, int halo) {
      const Owner<Location, decltype(cells)> owner{cells, 1};
      require(ecsim::check_field_boundary_owner<Location>(owner, plan, halo) ==
                  lpfx::ErrorCode::success, "owner of the prepared plan rejected");
      const Owner<Location, decltype(cells)> shallow{cells, halo - 1};
      if (halo > 0)
        require(ecsim::check_field_boundary_owner<Location>(shallow, plan, halo) ==
                    lpfx::ErrorCode::invalid_argument, "owner without ghosts accepted");
    };
    check.template operator()<lpfx::Node>(plans.node_ghosts, 1);
    check.template operator()<lpfx::Cell>(plans.cell_ghosts, 1);
    check.template operator()<lpfx::Node>(plans.constrained_rows, 0);
    if (periodic) continue;
    // Faces 0 and 1 are magnetic conductors (x), the others electric ones. At
    // an electric y wall, E_y and B_x/B_z are even; E_x/E_z and B_y are odd.
    // At a magnetic x wall it is the dual: E_x and B_y/B_z are odd.
    const auto& node = plans.node_ghosts.multiplier;
    const auto& cell = plans.cell_ghosts.multiplier;
    require(node[1][2] == 1 && node[0][2] == -1 && node[2][2] == -1 &&
                cell[1][2] == -1 && cell[0][2] == 1 && cell[2][2] == 1,
            "conducting component reflection changed");
    require(node[0][0] == -1 && node[1][0] == 1 && node[2][0] == 1 &&
                cell[0][0] == 1 && cell[1][0] == -1 && cell[2][0] == -1,
            "magnetic-conductor component reflection changed");
    require(plans.constrained_rows.constrained[2] == 0b101U &&
                plans.constrained_rows.constrained[0] == 0b001U,
            "zeroed wall rows of the electric or the magnetic conductor changed");
  }
}

void smoothing_sums() {
  const auto constant = [](int, int, int) { return 2.0; };
  const auto linear = [](int i, int j, int k) { return double(i + 2 * j + 3 * k); };
  using Operator = config::EcsimSmoothingOperator;
  require(ecsim::node_smoothing_value<Operator::Face7, double>(constant, 1, 1, 1, {0.5, 0.25}) ==
              0.5 * 2.0 + 0.25 * 12.0, "Face7 constant sum changed");
  require(ecsim::node_smoothing_value<Operator::Binomial27, double>(constant, 1, 1, 1, {}) == 2.0,
          "Binomial27 does not preserve a constant");
  require(ecsim::node_smoothing_value<Operator::Binomial27, double>(linear, 3, 4, 5, {}) ==
              linear(3, 4, 5), "Binomial27 does not preserve a linear field");
  require(ecsim::node_smoothing_value<Operator::Face7, double>(linear, 3, 4, 5, {0.25, 0.125}) ==
              0.25 * linear(3, 4, 5) + 0.125 * 6 * linear(3, 4, 5),
          "Face7 linear sum changed");
}
} // namespace

int main() {
  try {
    boundary_plans();
    smoothing_sums();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
