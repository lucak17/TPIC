// The kernel-argument records ParticleCellMap and MassMatrixLayout are copied
// from the prepared domain's checked counts; a domain without counts is
// rejected, and the test helper for hand-built domains reproduces the counts.
#include "../support/test_builders.hpp"

#include <tpic/config/domain.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/data/mass_matrix/mass_matrix_layout.hpp>
#include <tpic/data/particles/particle_cell.hpp>

#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tpic;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Decomposition {
  std::array<int, 3> cells, ranks;
  bool periodic;
};

void check(const Decomposition& decomposition) {
  config::GridConfig grid;
  grid.nxc = decomposition.cells[0]; grid.nyc = decomposition.cells[1]; grid.nzc = decomposition.cells[2];
  config::TopologyConfig topology;
  topology.xlen = decomposition.ranks[0]; topology.ylen = decomposition.ranks[1]; topology.zlen = decomposition.ranks[2];
  topology.periodic_x = topology.periodic_y = topology.periodic_z = decomposition.periodic;
  const int size = decomposition.ranks[0] * decomposition.ranks[1] * decomposition.ranks[2];
  for (int rank = 0; rank < size; ++rank) {
    const auto domain = config::prepare_domain(grid, topology, rank, size);
    const auto map = data::prepare_particle_cell_map(domain);
    require(map.nx == domain.local_cells[0] && map.ny == domain.local_cells[1] &&
                map.nz == domain.local_cells[2] &&
                map.count == domain.particle_cell_count_with_halo,
            "particle cell map differs from the prepared domain");
    const auto layout = data::prepare_mass_matrix_layout(domain);
    require(layout.interior_count == domain.interior_node_count &&
                layout.vertex_count == domain.field_node_count_with_halo &&
                layout.extent_x == domain.local_cells[0] + 3 &&
                layout.extent_y == domain.local_cells[1] + 3 &&
                layout.extent_z == domain.local_cells[2] + 3,
            "mass-matrix layout differs from the prepared domain");

    // The helper for domains built by hand reproduces the prepared counts.
    config::PreparedDomain hand_built{};
    hand_built.local_cells = domain.local_cells;
    test::fill_domain_counts(hand_built);
    require(hand_built.interior_cell_count == domain.interior_cell_count &&
                hand_built.interior_node_count == domain.interior_node_count &&
                hand_built.field_node_count_with_halo == domain.field_node_count_with_halo &&
                hand_built.particle_cell_count_with_halo == domain.particle_cell_count_with_halo,
            "hand-built domain counts differ from prepare_domain");
  }
}

template <class F> void rejects(F action, const char* message) {
  try { (void)action(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error(message);
}

} // namespace

int main() {
  try {
    for (const Decomposition& decomposition : {
             Decomposition{{1, 1, 1}, {1, 1, 1}, true},
             Decomposition{{6, 5, 4}, {1, 1, 1}, true},
             Decomposition{{6, 5, 4}, {1, 1, 1}, false},
             Decomposition{{7, 3, 2}, {2, 1, 1}, true},
             Decomposition{{9, 7, 5}, {2, 2, 2}, false},
             Decomposition{{13, 6, 4}, {3, 2, 1}, true},
             Decomposition{{64, 48, 32}, {2, 1, 1}, true}})
      check(decomposition);

    // A domain built by hand without its counts is rejected once, by the builder.
    config::PreparedDomain uncounted{};
    uncounted.local_cells = {4, 2, 2};
    rejects([&] { return data::prepare_particle_cell_map(uncounted); },
            "particle cell map accepted a domain without counts");
    rejects([&] { return data::prepare_mass_matrix_layout(uncounted); },
            "mass-matrix layout accepted a domain without counts");
    std::cout << "TPIC kernel record checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
