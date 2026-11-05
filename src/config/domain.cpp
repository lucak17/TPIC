#include <tpic/config/domain.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/config/detail/errors.hpp>

#include <lpfx/core/error_reporting.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tpic::config {
namespace {

constexpr detail::Errors errors{"domain"};

void check_lpfx(lpfx::ErrorCode error, const char *operation) {
  if (error == lpfx::ErrorCode::success) return;
  const auto context = errors.text(operation);
  if (error == lpfx::ErrorCode::overflow)
    lpfx::check_error<std::overflow_error>(error, context.c_str());
  lpfx::check_error<std::invalid_argument>(error, context.c_str());
}

int checked_product(const std::array<int, 3> &extents, int increment,
                    int components, const char *description) {
  constexpr int maximum = std::numeric_limits<int>::max();
  int product = components;
  for (const int extent : extents) {
    if (extent > maximum - increment) errors.overflow(description);
    const int expanded = extent + increment;
    if (expanded <= 0 || product > maximum / expanded) errors.overflow(description);
    product *= expanded;
  }
  return product;
}

void require_finite_positive(double value, const char *description) {
  if (!std::isfinite(value) || value <= 0.0) errors.overflow(description);
}

} // namespace

int configured_process_count(const config::TopologyConfig &topology) {
  const std::array<int, 3> processes{topology.xlen, topology.ylen, topology.zlen};
  for (int extent : processes)
    if (extent <= 0) errors.invalid("process extents must be positive");
  return checked_product(processes, 0, 1, "process-count product exceeds signed int");
}

PreparedDomain prepare_domain(const config::GridConfig &grid,
                              const config::TopologyConfig &topology,
                              int rank, int communicator_size) {
  PreparedDomain domain;
  domain.global_cells = {grid.nxc, grid.nyc, grid.nzc};
  domain.lengths = {grid.lx, grid.ly, grid.lz};
  const std::array<int, 3> processes{
      topology.xlen, topology.ylen, topology.zlen};
  const std::array<bool, 3> field_periodicity{
      topology.periodic_x, topology.periodic_y, topology.periodic_z};
  constexpr const char *axes[] = {"x", "y", "z"};

  if (communicator_size <= 0 || rank < 0 || rank >= communicator_size)
    errors.invalid("rank must be in [0, communicator_size) with a positive size");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    if (processes[axis] <= 0)
      errors.invalid(std::string(axes[axis]) + " process extent must be positive");
    if (domain.global_cells[axis] <= 0)
      errors.invalid(std::string(axes[axis]) + " global cell count must be positive");
    if (!std::isfinite(domain.lengths[axis]) || domain.lengths[axis] <= 0.0)
      errors.invalid(std::string(axes[axis]) + " physical length must be finite and positive");
    const int minimum = processes[axis] == 1 && field_periodicity[axis] ? 1 : 2;
    // Division avoids overflow in minimum * process extent.
    if (domain.global_cells[axis] / processes[axis] < minimum)
      errors.invalid(std::string(axes[axis]) + " requires at least " +
              std::to_string(minimum) + " cells per rank");
  }
  const int process_count = configured_process_count(topology);
  if (process_count != communicator_size)
    errors.invalid("process-count product differs from communicator_size");

  PreparedDomain::TopologyInput field_input;
  PreparedDomain::TopologyInput particle_input;
  check_lpfx(lpfx::try_make_canonical_cartesian_topology_input(
      topology.xlen, topology.ylen, topology.zlen, rank, communicator_size,
      lpfx::CartesianPeriodicity3{topology.periodic_x, topology.periodic_y,
                                  topology.periodic_z}, field_input),
      "field topology preparation");
  check_lpfx(lpfx::try_make_canonical_cartesian_topology_input(
      topology.xlen, topology.ylen, topology.zlen, rank, communicator_size,
      lpfx::CartesianPeriodicity3{topology.periodic_x_particles,
                                  topology.periodic_y_particles,
                                  topology.periodic_z_particles}, particle_input),
      "particle topology preparation");
  PreparedDomain::Topology field_topology;
  check_lpfx(lpfx::try_prepare_canonical_cartesian_topology(
      field_input, field_topology), "field neighbor preparation");
  check_lpfx(lpfx::try_prepare_canonical_cartesian_topology(
      particle_input, domain.particle_topology),
      "particle neighbor preparation");
  check_lpfx(lpfx::try_prepare_cartesian_partition_neighborhood3d<
                 PreparedDomain::IndexPolicy>(
      grid.nxc, grid.nyc, grid.nzc, field_topology,
      domain.field_neighborhood),
      "Cartesian partition-neighborhood preparation");

  const auto local = domain.field_neighborhood.local_partition().local_cells();
  const auto offset = domain.field_neighborhood.local_partition().global_offset();
  domain.local_cells = {static_cast<int>(local.x()), static_cast<int>(local.y()),
                        static_cast<int>(local.z())};
  domain.global_offset = {offset.x(), offset.y(), offset.z()};
  domain.interior_cell_count = checked_product(domain.local_cells, 0, 1,
                                                "local cell count exceeds signed int");
  domain.interior_node_count = checked_product(domain.local_cells, 1, 1,
                                                "local node count exceeds signed int");
  // The field cell and compact solver counts are not stored; computing them
  // proves that the halo-extended and three-component counts fit a signed int.
  (void)checked_product(
      domain.local_cells, 2 * PreparedDomain::field_halo, 1,
      "field cell extent with halos exceeds signed int");
  domain.field_node_count_with_halo = checked_product(
      domain.local_cells, 1 + 2 * PreparedDomain::field_halo, 1,
      "field node extent with halos exceeds signed int");
  domain.particle_cell_count_with_halo = checked_product(
      domain.local_cells, 2 * PreparedDomain::particle_halo, 1,
      "particle cell extent with halos exceeds signed int");
  (void)checked_product(
      domain.local_cells, 1, 3, "compact solver count exceeds signed int");

  for (std::size_t axis = 0; axis < 3; ++axis) {
    domain.spacing[axis] = domain.lengths[axis] / domain.global_cells[axis];
    require_finite_positive(domain.spacing[axis], "cell spacing is not representable");
    require_finite_positive(1.0 / domain.spacing[axis],
                            "inverse cell spacing is not representable");
    domain.origin[axis] = domain.spacing[axis] * domain.global_offset[axis];
    if (!std::isfinite(domain.origin[axis])) errors.overflow("physical origin is not representable");
  }
  // Evaluate volume with additional exponent range so mixed extreme spacings
  // do not spuriously fail due to the order of double intermediate products.
  const long double volume = static_cast<long double>(domain.spacing[0]) *
                             domain.spacing[1] * domain.spacing[2];
  domain.cell_volume = static_cast<double>(volume);
  require_finite_positive(domain.cell_volume, "cell volume is not representable");
  domain.inverse_cell_volume = 1.0 / domain.cell_volume;
  require_finite_positive(domain.inverse_cell_volume,
                          "inverse cell volume is not representable");
  return domain;
}

void require_canonical_partition(const PreparedDomain &domain, const char *context) {
  const auto &neighborhood = domain.field_neighborhood;
  if (!neighborhood.is_valid())
    throw std::invalid_argument(std::string(context) + ": prepared field partition required");
  const auto global = neighborhood.global_cells();
  const auto &partition = neighborhood.local_partition();
  const auto local = partition.local_cells();
  const auto offset = partition.global_offset();
  if (!std::cmp_equal(global.x(), domain.global_cells[0]) ||
      !std::cmp_equal(global.y(), domain.global_cells[1]) ||
      !std::cmp_equal(global.z(), domain.global_cells[2]) ||
      domain.local_cells != std::array<int, 3>{static_cast<int>(local.x()),
          static_cast<int>(local.y()), static_cast<int>(local.z())} ||
      domain.global_offset != std::array<int, 3>{offset.x(), offset.y(), offset.z()})
    throw std::invalid_argument(std::string(context) + ": inconsistent canonical partition geometry");
}

} // namespace tpic::config
