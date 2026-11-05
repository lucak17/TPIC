#ifndef TPIC_COMMUNICATION_PARTICLES_PARTICLE_GEOMETRY_HPP
#define TPIC_COMMUNICATION_PARTICLES_PARTICLE_GEOMETRY_HPP

#include <lpfx/core/attributes.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/config/physics.hpp>
#include <tpic/data/particles/particle_cell.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tpic::communication {

// Small, immutable host-prepared parameters copied into kernels and codecs.
// Integer cells stay separate from fractions; global positions never pass
// through a floating-point absolute coordinate.
struct ParticleMigrationGeometry {
  data::ParticleCellMap cells{};
  int global_cells[3]{};
  int global_offset[3]{};
  std::uint32_t owner_coordinate[3]{};
  int partition_base[3]{};
  int partition_remainder[3]{};
  int enlarged_partition_end[3]{};
  bool periodic[3]{};
  unsigned owner_x_bits{};
  unsigned owner_y_bits{};
};

// A prepared whole-domain owner has no possible remote destination under the
// supported periodic/mirror boundaries. Check the partition metadata as well
// as the local extent so a malformed/subdomain geometry cannot take this path.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool is_local_particle_geometry(
    const ParticleMigrationGeometry& geometry) noexcept {
  if (geometry.owner_x_bits != 0 || geometry.owner_y_bits != 0) return false;
  for (int axis = 0; axis < 3; ++axis)
    if (geometry.cells.extent(axis) <= 0 ||
        geometry.cells.extent(axis) != geometry.global_cells[axis] ||
        geometry.global_offset[axis] != 0 || geometry.owner_coordinate[axis] != 0 ||
        geometry.partition_base[axis] != geometry.cells.extent(axis) ||
        geometry.partition_remainder[axis] != 0 ||
        geometry.enlarged_partition_end[axis] != 0)
      return false;
  return true;
}

[[nodiscard]] inline ParticleMigrationGeometry prepare_particle_migration_geometry(
    const config::PreparedDomain& domain, const config::PreparedPhysics& physics) {
  if (!domain.particle_topology.is_valid())
    throw std::invalid_argument("TPIC particle migration: prepared particle domain required");
  config::require_canonical_partition(domain, "TPIC particle migration");
  const auto &field_input = domain.field_neighborhood.topology().input();
  const auto &particle_input = domain.particle_topology.input();
  if (particle_input.rank() != field_input.rank() ||
      particle_input.communicator_size() != field_input.communicator_size() ||
      particle_input.process_shape() != field_input.process_shape())
    throw std::invalid_argument("TPIC particle migration: inconsistent particle partition");
  const auto &partition = domain.field_neighborhood.local_partition();
  const auto shape = particle_input.process_shape();
  const auto coordinate = partition.process_coordinate();
  const std::uint32_t owner_coordinate[3]{coordinate.x(), coordinate.y(), coordinate.z()};
  const auto periodic = particle_input.periodicity();
  if (particle_input.communicator_size() >
      static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("TPIC particle migration: process count exceeds signed int");
  const int extents[3]{static_cast<int>(shape.x()), static_cast<int>(shape.y()),
                       static_cast<int>(shape.z())};
  const bool periodic_axes[3]{periodic.x, periodic.y, periodic.z};
  ParticleMigrationGeometry result;
  result.cells = data::prepare_particle_cell_map(domain);
  for (int axis = 0; axis < 3; ++axis) {
    const auto boundary = periodic_axes[axis] ? config::ParticleBoundary::periodic
                                             : config::ParticleBoundary::mirror;
    if (physics.particle_boundary[2 * axis] != boundary ||
        physics.particle_boundary[2 * axis + 1] != boundary)
      throw std::invalid_argument("TPIC particle migration: particle boundaries disagree with topology");
    result.global_cells[axis] = domain.global_cells[axis];
    result.global_offset[axis] = domain.global_offset[axis];
    result.owner_coordinate[axis] = owner_coordinate[axis];
    result.partition_base[axis] = domain.global_cells[axis] / extents[axis];
    result.partition_remainder[axis] = domain.global_cells[axis] % extents[axis];
    result.enlarged_partition_end[axis] = result.partition_remainder[axis] *
        (result.partition_base[axis] + 1);
    result.periodic[axis] = periodic_axes[axis];
  }
  const auto coordinate_bits = [](unsigned extent) {
    unsigned bits = 0;
    for (--extent; extent != 0; extent >>= 1) ++bits;
    return bits;
  };
  result.owner_x_bits = coordinate_bits(shape.x());
  result.owner_y_bits = coordinate_bits(shape.y());
  return result;
}

static_assert(std::is_trivially_copyable_v<ParticleMigrationGeometry>);

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_PARTICLES_PARTICLE_GEOMETRY_HPP
