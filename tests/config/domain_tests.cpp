#include <tpic/config/domain.hpp>
#include <tpic/config/simulation_parameters.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function>
void rejects(Function &&function, const char *message) {
  try {
    function();
  } catch (const Exception &error) {
    require(std::string(error.what()).starts_with("TPIC domain:"),
            "missing domain error context");
    return;
  }
  throw std::runtime_error(message);
}

void nondivisible_partition() {
  tpic::config::GridConfig grid;
  grid.nxc = 11; grid.nyc = 7; grid.nzc = 5;
  grid.lx = 22.0; grid.ly = 21.0; grid.lz = 20.0;
  tpic::config::TopologyConfig topology;
  topology.xlen = 3; topology.ylen = 2; topology.zlen = 2;
  topology.periodic_x = false;
  topology.periodic_x_particles = true;
  topology.periodic_y_particles = false;
  std::vector<int> coverage(11 * 7 * 5);
  for (int rank = 0; rank != 12; ++rank) {
    const auto domain = tpic::config::prepare_domain(grid, topology, rank, 12);
    const std::array<int, 3> expected_coordinate{rank % 3, (rank / 3) % 2, rank / 6};
    const auto coordinate =
        domain.field_neighborhood.local_partition().process_coordinate();
    require(std::array<int, 3>{static_cast<int>(coordinate.x()),
                static_cast<int>(coordinate.y()), static_cast<int>(coordinate.z())} ==
                expected_coordinate,
            "rank is not x fastest");
    require(domain.spacing == std::array<double, 3>{2, 3, 4}, "wrong spacing");
    require(domain.cell_volume == 24.0, "wrong cell volume");
    require(domain.field_neighborhood.topology().input().periodicity() == lpfx::CartesianPeriodicity3{false, true, true},
            "wrong field periodicity");
    require(domain.particle_topology.input().periodicity() == lpfx::CartesianPeriodicity3{true, false, true},
            "particle periodicity followed field periodicity");
    const std::array<int, 3> dims{3, 2, 2};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const int base = domain.global_cells[axis] / dims[axis];
      const int remainder = domain.global_cells[axis] % dims[axis];
      const int coordinate = expected_coordinate[axis];
      require(domain.local_cells[axis] == base + (coordinate < remainder),
              "wrong remainder assignment");
      require(domain.global_offset[axis] == base * coordinate +
              (coordinate < remainder ? coordinate : remainder), "wrong global offset");
      require(domain.origin[axis] == domain.spacing[axis] * domain.global_offset[axis],
              "wrong physical origin");
    }
    for (int z = 0; z < domain.local_cells[2]; ++z)
      for (int y = 0; y < domain.local_cells[1]; ++y)
        for (int x = 0; x < domain.local_cells[0]; ++x)
          ++coverage[(z + domain.global_offset[2]) * 77 +
                     (y + domain.global_offset[1]) * 11 + x + domain.global_offset[0]];
  }
  for (int count : coverage) require(count == 1, "partition has a gap or overlap");
}

void single_cell_axes() {
  tpic::config::GridConfig grid;
  grid.nxc = grid.nyc = grid.nzc = 1;
  tpic::config::TopologyConfig topology;
  topology.periodic_x_particles = topology.periodic_y_particles =
      topology.periodic_z_particles = false;
  const auto domain = tpic::config::prepare_domain(grid, topology, 0, 1);
  require(domain.interior_cell_count == 1 && domain.interior_node_count == 8,
          "one-cell nodal extent mismatch");
  require(domain.field_node_count_with_halo == 64, "field node halo extent mismatch");
  require(domain.particle_cell_count_with_halo == 125, "particle halo extent mismatch");
  topology.periodic_x = false;
  rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                                "nonperiodic one-cell axis accepted");
  topology.periodic_x = true;
  topology.xlen = 2;
  grid.nxc = 3;
  rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 2); },
                                "multi-rank axis with one cell on a rank accepted");
}

void invalid_geometry() {
  const tpic::config::GridConfig valid_grid;
  const tpic::config::TopologyConfig valid_topology;
  for (int rank : {-1, 1})
    rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(valid_grid, valid_topology, rank, 1); },
                                  "invalid rank accepted");
  rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(valid_grid, valid_topology, 0, 0); },
                                "zero communicator size accepted");
  rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(valid_grid, valid_topology, 0, 2); },
                                "communicator mismatch accepted");
  for (double length : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
    auto grid = valid_grid;
    grid.lx = length;
    rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(grid, valid_topology, 0, 1); },
                                  "invalid physical length accepted");
  }
  auto grid = valid_grid;
  grid.nxc = 0;
  rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(grid, valid_topology, 0, 1); },
                                "zero global cells accepted");
  auto topology = valid_topology;
  topology.zlen = -1;
  rejects<std::invalid_argument>([&] { (void)tpic::config::prepare_domain(valid_grid, topology, 0, 1); },
                                "negative process extent accepted");
}

void representability() {
  tpic::config::GridConfig grid;
  tpic::config::TopologyConfig topology;
  grid.nxc = std::numeric_limits<int>::max();
  grid.nyc = grid.nzc = 1;
  rejects<std::overflow_error>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                              "nodal signed extent overflow accepted");
  grid.nxc = grid.nyc = 50000;
  rejects<std::overflow_error>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                              "flattened signed count overflow accepted");
  grid.nxc = grid.nyc = grid.nzc = 1000;
  rejects<std::overflow_error>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                              "three-component compact overflow accepted");
  grid.nxc = grid.nyc = 100000;
  topology.xlen = topology.ylen = 50000;
  rejects<std::overflow_error>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                              "process-count overflow accepted");
  grid = {};
  topology = {};
  grid.lx = std::numeric_limits<double>::denorm_min();
  rejects<std::overflow_error>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                              "spacing underflow accepted");
  grid.lx = grid.ly = grid.lz = 1e200;
  rejects<std::overflow_error>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                              "volume overflow accepted");
  grid.lx = grid.ly = grid.lz = 1e-110;
  rejects<std::overflow_error>([&] { (void)tpic::config::prepare_domain(grid, topology, 0, 1); },
                              "volume underflow accepted");
  grid.lx = 1e300; grid.ly = 1e20; grid.lz = 1e-300;
  const auto balanced = tpic::config::prepare_domain(grid, topology, 0, 1);
  require(std::isfinite(balanced.cell_volume) && balanced.cell_volume > 0,
          "representable volume rejected because of intermediate arithmetic");
}

} // namespace

int main() {
  try {
    nondivisible_partition();
    single_cell_axes();
    invalid_geometry();
    representability();
    std::cout << "TPIC domain checks passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
