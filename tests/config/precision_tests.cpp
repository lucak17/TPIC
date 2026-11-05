#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/io/diagnostics_writer.hpp>
#include <tpic/io/async_output_writer.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "../support/mpi_field_files.hpp"
#include "../support/test_builders.hpp"

#include <lpfx/data/particle/device/classification.hpp>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class F> void rejects(F&& f) {
  try { f(); }
  catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid precision/plan accepted");
}

tpic::config::SimulationParameters parameters() {
  tpic::config::SimulationParameters p;
  p.case_name = "Uniform";
  p.dt = .125;
  p.ncycles = 1;
  p.ns = 1;
  p.grid.nxc = 4;
  p.grid.nyc = 3;
  p.grid.nzc = 2;
  p.species.resize(1);
  auto& species = p.species.front();
  species.q_over_m = -1;
  species.rho_init = 1;
  species.npcelx = species.npcely = species.npcelz = 2;
  p.output.restart_cycle = 0;
  p.output.field_cycle = p.output.diagnostics_cycle = 1;
  p.output.field_tag = "E";
  return p;
}

// The --check-input byte lines: the record bytes of the storage roles (68, 56
// or 44), and the logical owner (record plus one 8-byte auxiliary), sort and
// classification bytes.
void byte_report(const tpic::simulation::PreparedInput& input) {
  const auto report = tpic::simulation::report_particle_bytes(input);
  const std::size_t expected_record[] = {68, 56, 44, 44, 44};
  const std::size_t record = expected_record[static_cast<std::size_t>(input.precision.storage)];
  require(report.particle_record_bytes == record, "logical record bytes changed");
  const auto& species = input.particles.species.at(0);
  const std::size_t capacity = species.initial_capacity;
  std::size_t expected = (record + 8) * capacity;
  if (input.particles.order != tpic::config::ParticleOrder::unsorted)
    expected += 4 * capacity +
        8 * static_cast<std::size_t>(input.domain.particle_cell_count_with_halo) + 8;
  if (species.classification_capacity != 0)
    expected += 8 * static_cast<std::size_t>(species.classification_capacity) +
        sizeof(lpfx::DeviceParticleRegisteredClassificationData);
  require(report.device_bytes_lower_bound == expected, "particle device lower bound changed");
}

void profiles(const std::filesystem::path& directory, lpfx::distributed::MpiFileWriter& field_files) {
  using namespace tpic;
  require(simulation::InputOptions{}.precision == config::PrecisionSelection{}, "input omitted runtime default");
  rejects([] { (void)config::parse_storage_precision("mix"); });
  rejects([] { (void)config::parse_compute_precision("float"); });
  const config::PrecisionSelection invalid{static_cast<config::StoragePrecision>(99), config::ComputePrecision::double_precision};
  rejects([&] { (void)config::precision_metadata(invalid); });
  rejects([&] { (void)simulation::prepare_input(parameters(), {.precision = invalid}); });
  bool dispatched = false;
  rejects([&] { config::dispatch_precision(invalid, [&]<class> { dispatched = true; }); });
  require(!dispatched, "invalid selection entered numerical dispatch");

  auto distributed_parameters = parameters();
  distributed_parameters.topology.xlen = 2;
  auto rank_zero = simulation::prepare_input(
      distributed_parameters, {.rank = 0, .communicator_size = 2});
  const auto rank_one = simulation::prepare_input(
      distributed_parameters, {.rank = 1, .communicator_size = 2});
  rank_zero.domain.field_neighborhood = rank_one.domain.field_neighborhood;
  rejects([&] { (void)tpic::test::checked_domain(rank_zero.parameters, rank_zero.domain); });

  for (unsigned pair = 0; pair < config::precision_level_count * config::precision_level_count; ++pair) {
    const config::PrecisionSelection profile{static_cast<config::StoragePrecision>(pair / config::precision_level_count),
                                     static_cast<config::ComputePrecision>(pair % config::precision_level_count)};
    const auto metadata = config::precision_metadata(profile);
    require(metadata.name == config::precision_name(profile) && config::precision_pair_index(profile) == pair &&
                config::parse_storage_precision(config::storage_precision_name(profile.storage)) == profile.storage &&
                config::parse_compute_precision(config::compute_precision_name(profile.compute)) == profile.compute,
            "canonical precision names do not roundtrip");
    auto input = simulation::prepare_input(parameters(), {.precision = profile});
    input.output.save_dir = (directory / std::to_string(pair)).string();
    std::filesystem::create_directories(input.output.save_dir);
    for (const auto order : {config::ParticleOrder::full, config::ParticleOrder::partial,
                             config::ParticleOrder::unsorted}) {
      auto ordered = simulation::prepare_input(parameters(), {.order = order, .precision = profile});
      // A build may compile a subset of the pairs; the run plan admits only those.
      if (config::precision_pair_compiled(profile)) {
        const auto frozen = simulation::RunPlan::prepare(parameters(), {.order = order, .precision = profile},
                                                         simulation::RunMode::initialize_only);
        require(frozen.input.precision == profile && frozen.input.particles == ordered.particles,
                "frozen run plan changed precision or particle allocation");
      } else {
        rejects([&] {
          (void)simulation::RunPlan::prepare(parameters(), {.order = order, .precision = profile},
                                             simulation::RunMode::initialize_only);
        });
      }
      require(simulation::prepare_input(parameters(), {.order = order}).particles == ordered.particles,
              "the particle plan depends on the precision profile");
      byte_report(ordered);
    }
    unsigned calls = 0;
    config::dispatch_precision(profile, [&]<class Precision> {
      ++calls;
      static_assert(std::is_same_v<typename Precision::Storage::Weight, double>);
      static_assert(std::is_same_v<typename Precision::Compute::SolverReduction, double>);
      static_assert(std::is_same_v<typename Precision::Compute::Reduction, double>);
      require(Precision::Storage::id == profile.storage && Precision::Compute::id == profile.compute,
              "dispatch selected another pair");
      const auto bytes = 3 * sizeof(typename Precision::Storage::Position) +
          3 * sizeof(typename Precision::Storage::Velocity) + sizeof(typename Precision::Storage::Weight) + 12;
      require(simulation::report_particle_bytes(input).particle_record_bytes == bytes,
              "runtime record width differs from storage roles");
    });
    require(calls == 1, "precision dispatch allocated multiple numerical leaves");

    io::DiagnosticsWriter csv;
    csv.open(input.output, 1, metadata);
    csv.close();
    const auto csv_text = tpic::test::file_contents(csv.path());
    require(csv_text.find("# precision_profile = " + std::string(metadata.name)) != std::string::npos &&
                csv_text.find(metadata.roles) != std::string::npos,
            "CSV omitted configured profile metadata");

    io::AsyncOutputWriter vtk;
    const tpic::config::PreparedOutputSelection selection(input.output);
    tpic::test::configure(vtk, input.output, input.domain, field_files, metadata);
    data::OwnedOutputSnapshot snapshot{0, 0.0, true, {}};
    for (const auto& array : selection.arrays())
      snapshot.arrays.push_back({array.name, array.components, std::vector<double>(
          static_cast<std::size_t>(input.domain.interior_node_count) * array.components, .125)});
    vtk.submit(std::move(snapshot));
    vtk.host_wait_piece();
    (void)vtk.complete_snapshot();
    std::string encoded;
    for (const unsigned char character : metadata.name)
      encoded += std::to_string(character) + ' ';
    encoded += "0</Array>";
    require(tpic::test::file_contents(vtk.fields_path()).find(encoded) != std::string::npos,
            "VTK field file omitted configured precision metadata");
  }
}

} // namespace

int main(int argc, char** argv) {
  tpic::test::MpiFieldFiles files(argc, argv);
  std::string pattern = (std::filesystem::temp_directory_path() / "tpic-precision-XXXXXX").string();
  if (!::mkdtemp(pattern.data())) return 1;
  int result = 0;
  try {
    profiles(pattern, files.files);
    std::cout << "TPIC runtime precision planning, dispatch and metadata checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }
  std::filesystem::remove_all(pattern);
  return result;
}
