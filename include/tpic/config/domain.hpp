#ifndef TPIC_CONFIG_DOMAIN_HPP
#define TPIC_CONFIG_DOMAIN_HPP

#include <array>
#include <cstdint>

#include <lpfx/domain/cartesian/partition.hpp>
#include <lpfx/domain/cartesian/partition_neighborhood.hpp>
#include <lpfx/domain/cartesian/index_policy.hpp>

namespace tpic::config {

// Checked host geometry. Fields carry one halo layer and the extended
// particle-cell grid two (see ParticleCellMap); a particle cell key is not an
// LPFx field allocation index.
struct PreparedDomain {
  using IndexPolicy = lpfx::LogicalIndexPolicy32;
  using Rank = std::uint32_t;
  using TopologyInput = lpfx::CanonicalCartesianTopologyInput3D<Rank>;
  using Topology = lpfx::PreparedCanonicalCartesianTopology3D<Rank>;
  using Partition = lpfx::CartesianPartition3D<IndexPolicy, Rank>;
  using PartitionNeighborhood =
      lpfx::PreparedCartesianPartitionNeighborhood3D<IndexPolicy, Rank>;

  static constexpr int field_halo = 1;
  static constexpr int particle_halo = 2;

  Topology particle_topology;
  PartitionNeighborhood field_neighborhood;
  std::array<int, 3> global_cells{};
  std::array<int, 3> local_cells{};
  std::array<int, 3> global_offset{};
  std::array<double, 3> lengths{};
  std::array<double, 3> spacing{};
  std::array<double, 3> origin{};
  double cell_volume{};
  double inverse_cell_volume{};

  // Logical sample counts exclude LPFx pitch/alignment padding. All counts fit
  // the signed int indexing of the kernels.
  int interior_cell_count{};
  int interior_node_count{};
  int field_node_count_with_halo{};
  int particle_cell_count_with_halo{};
};

struct GridConfig;
struct TopologyConfig;

// Checked topology size, also used by input checking before MPI exists.
[[nodiscard]] int configured_process_count(const TopologyConfig &topology);

// No MPI initialization, device allocation or output side effects. Invalid
// geometry throws invalid_argument; unrepresentable adapter sizes or physical
// arithmetic throw overflow_error. Rank order is x fastest, with low-coordinate
// ranks receiving remainder cells independently on each axis.
[[nodiscard]] PreparedDomain prepare_domain(
    const GridConfig &grid, const TopologyConfig &topology,
    int rank, int communicator_size);

// The domain's global cells, local cells and offset are those of its prepared
// field partition. Throws std::invalid_argument prefixed with context.
void require_canonical_partition(const PreparedDomain &domain, const char *context);


} // namespace tpic::config

#endif // TPIC_CONFIG_DOMAIN_HPP
