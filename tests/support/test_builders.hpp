#ifndef TPIC_TESTS_SUPPORT_TEST_BUILDERS_HPP
#define TPIC_TESTS_SUPPORT_TEST_BUILDERS_HPP

// Test-only builders. Production prepares every record once through
// simulation::RunPlan::prepare and trusts it by construction; these helpers
// build the same records from raw parameters and re-check hand-edited
// aggregates, which only tests construct.

#include <lpfx/core/error_reporting.hpp>
#include <lpfx/distributed/plan/collective.hpp>

#include <tpic/config/domain.hpp>
#include <tpic/config/particle_plan.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/data/output/output_snapshot.hpp>
#include <tpic/data/particles/particle_cell.hpp>
#include <tpic/initialization/cases/initial_case.hpp>
#include <tpic/initialization/fields/field_image.hpp>
#include <tpic/initialization/prepared_initialization.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace tpic::test {

// Append one owned field array to a snapshot, as a capture would.
inline void append_array(data::OwnedOutputSnapshot& snapshot, std::string_view name,
                         int components, std::span<const double> values) {
  snapshot.arrays.push_back({std::string(name), components,
                             std::vector<double>(values.begin(), values.end())});
}

// Stored particle records of one species for engine tests. The vectors back
// the spans of records(), so the storage outlives every snapshot built from it.
struct ParticleRecordStorage {
  std::array<std::vector<double>, 3> fraction, velocity;
  std::vector<double> charge;
  std::vector<std::int32_t> cell;
  std::vector<std::uint64_t> id;

  // Appends one record in the interior cell cell_index of domain.
  void add(const config::PreparedDomain& domain, std::array<int, 3> cell_index,
           std::array<double, 3> fractions, std::array<double, 3> velocities, double q,
           std::uint64_t identifier) {
    cell.push_back(data::prepare_particle_cell_map(domain).encode(cell_index[0], cell_index[1], cell_index[2]));
    for (std::size_t axis = 0; axis < 3; ++axis) {
      fraction[axis].push_back(fractions[axis]);
      velocity[axis].push_back(velocities[axis]);
    }
    charge.push_back(q);
    id.push_back(identifier);
  }
  [[nodiscard]] data::ParticleRecords records(std::size_t species) const {
    return data::ParticleRecords{
        .species = species,
        .fraction = {std::span<const double>(fraction[0]), std::span<const double>(fraction[1]),
                     std::span<const double>(fraction[2])},
        .velocity = {std::span<const double>(velocity[0]), std::span<const double>(velocity[1]),
                     std::span<const double>(velocity[2])},
        .charge = std::span<const double>(charge), .cell = cell, .id = id};
  }
};

// The arrays every engine writes for stored records: interleaved positions
// (offset + cell + fraction) * spacing, interleaved velocities, charges, ids.
struct ParticleOutputArrays {
  std::vector<double> position, velocity, charge;
  std::vector<std::uint64_t> id;
};

[[nodiscard]] inline ParticleOutputArrays particle_output_arrays(
    const data::ParticleRecords& records, const config::PreparedDomain& domain) {
  const auto cells = data::prepare_particle_cell_map(domain);
  const auto value = [](const data::ParticleValues& values, std::size_t index) {
    return std::visit([&](auto stored) { return double(stored[index]); }, values);
  };
  ParticleOutputArrays arrays;
  for (std::size_t index = 0; index < records.id.size(); ++index) {
    const auto cell = cells.decode(records.cell[index]);
    const int coordinate[3]{cell.i, cell.j, cell.k};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      arrays.position.push_back((domain.global_offset[axis] + coordinate[axis] +
                                 value(records.fraction[axis], index)) * domain.spacing[axis]);
      arrays.velocity.push_back(value(records.velocity[axis], index));
    }
    arrays.charge.push_back(value(records.charge, index));
    arrays.id.push_back(records.id[index]);
  }
  return arrays;
}

// Fill the checked counts of a domain built by hand from its local cells, with
// the formulas of config::prepare_domain. Test extents are small; no overflow.
inline void fill_domain_counts(config::PreparedDomain& domain) {
  const auto product = [&](int increment) {
    int count = 1;
    for (const int cells : domain.local_cells) count *= cells + increment;
    return count;
  };
  domain.interior_cell_count = product(0);
  domain.interior_node_count = product(1);
  domain.field_node_count_with_halo = product(1 + 2 * config::PreparedDomain::field_halo);
  domain.particle_cell_count_with_halo = product(2 * config::PreparedDomain::particle_halo);
}

// Re-prepare the domain from the parameters and reject a hand-edited one.
// Returns the canonical prepared geometry.
[[nodiscard]] inline config::PreparedDomain checked_domain(
    const config::SimulationParameters& parameters, const config::PreparedDomain& domain) {
  if (!domain.field_neighborhood.is_valid() || !domain.particle_topology.is_valid())
    throw std::invalid_argument("TPIC domain: unprepared domain geometry");
  const auto rank = domain.field_neighborhood.topology().input().rank();
  if (rank > static_cast<unsigned>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("TPIC domain: rank exceeds signed int");
  const auto checked = config::prepare_domain(parameters.grid, parameters.topology,
      static_cast<int>(rank), config::configured_process_count(parameters.topology));
  const auto& field = domain.field_neighborhood;
  const auto& expected_field = checked.field_neighborhood;
  const auto& partition = field.local_partition();
  const auto& expected_partition = expected_field.local_partition();
  if (field.topology().input() != expected_field.topology().input() ||
      field.global_cells() != expected_field.global_cells() ||
      partition.rank() != expected_partition.rank() ||
      partition.process_shape() != expected_partition.process_shape() ||
      partition.process_coordinate() != expected_partition.process_coordinate() ||
      partition.global_cell_box() != expected_partition.global_cell_box() ||
      domain.particle_topology.input() != checked.particle_topology.input() ||
      domain.global_cells != checked.global_cells ||
      domain.local_cells != checked.local_cells ||
      domain.global_offset != checked.global_offset ||
      domain.lengths != checked.lengths || domain.spacing != checked.spacing ||
      domain.origin != checked.origin || domain.cell_volume != checked.cell_volume ||
      domain.inverse_cell_volume != checked.inverse_cell_volume ||
      domain.interior_cell_count != checked.interior_cell_count ||
      domain.interior_node_count != checked.interior_node_count ||
      domain.field_node_count_with_halo != checked.field_node_count_with_halo ||
      domain.particle_cell_count_with_halo != checked.particle_cell_count_with_halo)
    throw std::invalid_argument("TPIC domain: domain does not match input geometry");
  return checked;
}

// Field image from raw parameters, with the domain checked against them.
template <class Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>
[[nodiscard]] initialization::HostFieldImage<Precision> make_host_field_image(
    const config::SimulationParameters& parameters, const config::PreparedDomain& domain) {
  return initialization::make_host_field_image<Precision>(
      parameters, initialization::prepare_initial_case(parameters), checked_domain(parameters, domain));
}

// Particle load plans of a possibly hand-edited PreparedInput: the domain,
// particle plan and initial case must equal those prepared from its parameters.
[[nodiscard]] inline std::vector<initialization::ParticleLoadPlan> prepare_particle_loads(
    const simulation::PreparedInput& input) {
  (void)checked_domain(input.parameters, input.domain);
  if (initialization::prepare_initial_case(input.parameters) != input.initial_case)
    throw std::invalid_argument("TPIC initialization: initial case does not match input");
  if (initialization::prepare_bounded_particle_plan(
          config::prepare_particle_plan(input.parameters, input.domain, input.particles.order),
          input.initial_case, input.parameters, input.domain) != input.particles)
    throw std::invalid_argument("TPIC initialization: particle plan does not match input");
  return initialization::prepare_particle_loads(input.parameters, input.domain, input.particles);
}

[[nodiscard]] inline initialization::ParticleLoadPlan prepare_particle_load(
    const simulation::PreparedInput& input, std::size_t species) {
  if (species >= input.particles.species.size())
    throw std::invalid_argument("TPIC particle loading: species index out of range");
  return prepare_particle_loads(input).at(species);
}

// Named four-value SUM collective plan for the velocity-centering operand;
// production registers the operand through the transport registry instead.
template <std::uint64_t ChannelIdentity, class Precision>
using VelocityCenteringDefinition = lpfx::distributed::CollectivePlan<
    ChannelIdentity, typename Precision::Compute::Reduction,
    lpfx::distributed::all_reduce_descriptor<
        lpfx::distributed::reduction_operation::sum>>;

template <std::uint64_t ChannelIdentity, class Precision>
[[nodiscard]] auto prepare_velocity_centering_plan(std::uint32_t ranks) {
  using Definition = VelocityCenteringDefinition<ChannelIdentity, Precision>;
  lpfx::distributed::PreparedCollectivePlan<Definition> plan;
  lpfx::check_error<std::invalid_argument>(
      lpfx::distributed::try_prepare_collective_plan(Definition{}, 4, 4, ranks, plan),
      "TPIC velocity centering: invalid collective rank count");
  return plan;
}

// The bytes of an output file the test expects.
inline std::string file_contents(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.good()) throw std::runtime_error("expected output file is missing: " + path.string());
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// One array of a raw appended VTK XML file with UInt64 headers.
template <class T> std::vector<T> vtk_appended_array(const std::string& file, const std::string& name) {
  const auto require = [](bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
  };
  constexpr std::string_view marker = "<AppendedData encoding=\"raw\">\n_";
  const auto raw = file.find(marker);
  require(raw != std::string::npos, "VTK appended payload is missing");
  const auto start = raw + marker.size();
  const auto descriptor = file.find("Name=\"" + name + "\"");
  require(descriptor < raw, "VTK array descriptor is missing");
  const auto offset_begin = file.find("offset=\"", descriptor);
  require(offset_begin < raw, "VTK array offset is missing");
  const auto offset = std::stoull(file.substr(offset_begin + 8));
  require(start + offset + sizeof(std::uint64_t) <= file.size(), "VTK array offset out of bounds");
  std::uint64_t bytes;
  std::memcpy(&bytes, file.data() + start + offset, sizeof(bytes));
  require(bytes % sizeof(T) == 0 && bytes <= file.size() - start - offset - sizeof(bytes),
          "VTK array size is invalid");
  std::vector<T> result(bytes / sizeof(T));
  if (bytes != 0) std::memcpy(result.data(), file.data() + start + offset + sizeof(bytes), bytes);
  return result;
}

} // namespace tpic::test

#endif // TPIC_TESTS_SUPPORT_TEST_BUILDERS_HPP
