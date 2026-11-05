#include <tpic/io/async_output_writer.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "../support/mpi_field_files.hpp"
#include "../support/test_builders.hpp"

namespace {

namespace fs = std::filesystem;
using tpic::io::AsyncOutputWriter;
using tpic::data::OwnedOutputSnapshot;

tpic::test::MpiFieldFiles* mpi = nullptr;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class F> void rejects(F&& f, const char* message) {
  try { f(); } catch (const Exception&) { return; }
  throw std::runtime_error(message);
}

struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    auto pattern = (fs::temp_directory_path() / "tpic-async-output-XXXXXX").string();
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("cannot create test directory");
    path = pattern;
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
};

tpic::config::PreparedDomain domain(int rank = 0, int ranks = 1) {
  tpic::config::GridConfig grid;
  grid.nxc = 4;
  grid.nyc = grid.nzc = 2;
  tpic::config::TopologyConfig topology;
  topology.xlen = ranks;
  return tpic::config::prepare_domain(grid, topology, rank, ranks);
}

tpic::config::PreparedOutput output(const fs::path& path, bool fields = false) {
  tpic::config::PreparedOutput result;
  result.save_dir = path.string();
  result.output_e = fields;
  result.field_cycle = fields ? 2 : 0;
  result.particles_cycle = 3;
  result.species_selected.fill({false, false});
  return result;
}

void configure(AsyncOutputWriter& writer, const tpic::config::PreparedOutput& prepared,
               const tpic::config::PreparedDomain& geometry) {
  tpic::test::configure(writer, prepared, geometry, mpi->files);
}

// The stored particle records of a test domain. Snapshots borrow them, so each
// storage lives until the program ends.
const tpic::test::ParticleRecordStorage& stored_particles(const tpic::config::PreparedDomain& geometry) {
  static std::list<std::pair<std::array<int, 3>, tpic::test::ParticleRecordStorage>> storages;
  for (const auto& [cells, storage] : storages)
    if (cells == geometry.local_cells) return storage;
  auto& storage = storages.emplace_back(geometry.local_cells, tpic::test::ParticleRecordStorage{}).second;
  storage.add(geometry, {0, 0, 0}, {0.1, 0.2, 0.15}, {1., 2., 3.}, -0.1, 9007199254740993ULL);
  storage.add(geometry, {1, 1, 0}, {0.5, 0.25, 0.35}, {4., 5., 6.}, -0.2, 18446744073709551614ULL);
  return storage;
}

const double* first_fraction(const tpic::data::ParticleRecords& records) {
  return std::get<std::span<const double>>(records.fraction[0]).data();
}

OwnedOutputSnapshot capture(const tpic::config::PreparedDomain& geometry, int cycle,
                            bool fields = false, bool particles = true,
                            bool final = false) {
  OwnedOutputSnapshot result{cycle, cycle * 0.25, final, {}, fields, particles, {}};
  if (fields)
    result.arrays.push_back({"E", 3,
        std::vector<double>(static_cast<std::size_t>(geometry.interior_node_count) * 3, 0.5)});
  if (particles) {
    result.particles.push_back(stored_particles(geometry).records(0));
    // An empty species must be emitted too, including on otherwise empty ranks.
    result.particles.push_back(tpic::data::ParticleRecords{.species = 1});
  }
  return result;
}

using tpic::test::file_contents;
using tpic::test::vtk_appended_array;

void particle_roundtrip_and_independent_cadence(const fs::path& base) {
  const auto directory = base / "cadence";
  AsyncOutputWriter writer;
  configure(writer, output(directory, true), domain());
  require(!fs::exists(directory), "configure performed filesystem work");
  fs::create_directories(directory);
  auto snapshot = capture(domain(), 0, true);
  const auto* particles_storage = first_fraction(snapshot.particles[0]);
  const auto* fields_storage = snapshot.arrays[0].values.data();
  writer.submit(std::move(snapshot));
  writer.host_wait_piece();
  require(!fs::exists(directory / "particles_t000000.pvtp"), "worker published particle manifest");
  auto reusable = writer.complete_snapshot();
  require(first_fraction(reusable.particles[0]) == particles_storage, "particle records were copied");
  require(reusable.arrays[0].values.data() == fields_storage, "field ownership was copied");
  require(writer.fields_path() == (directory / "E_t000000.vti").string() &&
          writer.particles_path() == (directory / "particles_t000000.pvtp").string(),
          "combined VTK publication exposed wrong paths");
  const auto file = file_contents(directory / "particles_t000000_r000000.vtp");
  const auto expected = tpic::test::particle_output_arrays(reusable.particles[0], domain());
  require(vtk_appended_array<double>(file, "position") == expected.position &&
          vtk_appended_array<double>(file, "velocity") == expected.velocity &&
          vtk_appended_array<double>(file, "charge") == expected.charge,
          "VTK particle floating point payload changed");
  require(vtk_appended_array<std::uint64_t>(file, "id") == expected.id,
          "VTK particle IDs lost integer precision");
  require(vtk_appended_array<std::uint64_t>(file, "species") == std::vector<std::uint64_t>{0, 0} &&
          vtk_appended_array<std::int64_t>(file, "connectivity") == std::vector<std::int64_t>{0, 1} &&
          vtk_appended_array<std::int64_t>(file, "offsets") == std::vector<std::int64_t>{1, 2},
          "VTK particle vertices/species are incorrect");

  reusable.cycle = 2;
  reusable.time = 0.5;
  reusable.particles_present = false;
  // Inactive buffers stay owned for reuse and are neither validated nor written.
  writer.submit(std::move(reusable));
  writer.host_wait_piece();
  reusable = writer.complete_snapshot();
  require(fs::exists(directory / "E_t000002.vti") &&
          !fs::exists(directory / "particles_t000002_r000000.vtp"),
          "field-only epoch emitted particles");
  require(writer.fields_path() == (directory / "E_t000002.vti").string() &&
          writer.particles_path() == (directory / "particles_t000000.pvtp").string(),
          "field-only epoch changed the particle manifest path");

  reusable.cycle = 3;
  reusable.time = 0.75;
  reusable.fields_present = false;
  reusable.particles_present = true;
  reusable.final_state = true;
  writer.submit(std::move(reusable));
  writer.host_wait_piece();
  reusable = writer.complete_snapshot();
  require(fs::exists(directory / "particles_t000003.pvtp") &&
          !fs::exists(directory / "E_t000003.vti"),
          "particle-only epoch emitted fields");
  require(writer.fields_path() == (directory / "E_t000002.vti").string() &&
          writer.particles_path() == (directory / "particles_t000003.pvtp").string(),
          "particle-only epoch changed the field manifest path");
  require(first_fraction(reusable.particles[0]) == particles_storage,
          "alternating cadence replaced the borrowed particle records");
  require(reusable.arrays[0].values.data() == fields_storage,
          "alternating cadence replaced inactive field buffer allocations");
  require(file_contents(writer.particles_path()).find("Name=\"final_state\" NumberOfTuples=\"1\" format=\"ascii\">1") != std::string::npos,
          "final particle manifest lost epoch metadata");
}

void independent_channel_configuration(const fs::path& base) {
  for (const auto backend : {tpic::config::OutputEngineKind::Adios2,
                             tpic::config::OutputEngineKind::Hdf5}) {
    // An inactive channel's backend must not be instantiated, even when the
    // optional library is absent. Configuration itself must never create files.
    auto particles = output(base / ("particles-" + std::to_string(static_cast<int>(backend))));
    particles.field_engine = backend;
    particles.particle_engine = tpic::config::OutputEngineKind::Vtk;
    AsyncOutputWriter particle_writer;
    configure(particle_writer, particles, domain());
    require(!fs::exists(particles.save_dir), "inactive field backend performed filesystem work");
    fs::create_directories(particles.save_dir);
    particle_writer.submit(capture(domain(), 0));
    particle_writer.host_wait_piece();
    particle_writer.complete_snapshot();
    require(particle_writer.fields_path().empty() &&
            particle_writer.particles_path().ends_with("particles_t000000.pvtp"),
            "particle-only configuration exposed a field manifest");

    auto fields = output(base / ("fields-" + std::to_string(static_cast<int>(backend))), true);
    fields.field_engine = tpic::config::OutputEngineKind::Vtk;
    fields.particle_engine = backend;
    fields.particles_cycle = 0;
    AsyncOutputWriter field_writer;
    configure(field_writer, fields, domain());
    require(!fs::exists(fields.save_dir), "inactive particle backend performed filesystem work");
    fs::create_directories(fields.save_dir);
    field_writer.submit(capture(domain(), 0, true, false));
    field_writer.host_wait_piece();
    field_writer.complete_snapshot();
    require(field_writer.fields_path().ends_with("E_t000000.vti") &&
            field_writer.particles_path().empty(),
            "field-only configuration exposed a particle manifest");
  }
}

void channel_failure_blocks_publication(const fs::path& base) {
  const auto directory = base / "channel-failure";
  fs::create_directory(directory);
  const auto piece = directory / "particles_t000000_r000000.vtp";
  { std::ofstream file(piece); file << "existing particle data"; }
  AsyncOutputWriter writer;
  configure(writer, output(directory, true), domain());
  writer.submit(capture(domain(), 0, true));
  rejects<std::system_error>([&] { writer.host_wait_piece(); }, "second channel failure was lost");
  rejects<std::system_error>([&] { writer.complete_snapshot(); }, "partially failed epoch was published");
  require(fs::exists(directory / "E_t000000.vti") &&
          file_contents(piece) == "existing particle data" &&
          !fs::exists(directory / "particles_t000000.pvtp"),
          "failed particle channel published its manifest");
}

void validation_and_empty_rank(const fs::path& base) {
  AsyncOutputWriter writer;
  fs::create_directories(base / "validate");
  configure(writer, output(base / "validate", true), domain());
  auto missing = capture(domain(), 0);
  missing.particles.pop_back();
  rejects<std::invalid_argument>([&] { writer.submit(std::move(missing)); }, "missing species accepted");
  auto duplicate = capture(domain(), 0);
  duplicate.particles[1].species = 0;
  rejects<std::invalid_argument>([&] { writer.submit(std::move(duplicate)); }, "duplicate species accepted");
  auto invalid = capture(domain(), 0);
  invalid.particles[1].species = 2;
  rejects<std::invalid_argument>([&] { writer.submit(std::move(invalid)); }, "out of range species accepted");
  auto shape = capture(domain(), 0);
  shape.particles[0].velocity[2] = std::get<std::span<const double>>(shape.particles[0].velocity[2]).first(1);
  rejects<std::invalid_argument>([&] { writer.submit(std::move(shape)); }, "particle vector shape accepted");
  auto neither = capture(domain(), 0, false, false);
  rejects<std::invalid_argument>([&] { writer.submit(std::move(neither)); }, "empty epoch accepted");
  require(!writer.pending(), "validation failure occupied asynchronous slot");

  // Failed submissions must leave the reusable writer healthy. Species can
  // arrive in either order; their explicit IDs determine their interpretation.
  auto reordered = capture(domain(), 0, true);
  std::swap(reordered.particles[0], reordered.particles[1]);
  writer.submit(std::move(reordered));
  writer.host_wait_piece();
  auto accepted = writer.complete_snapshot();
  require(accepted.particles[0].species == 1 && accepted.particles[1].species == 0,
          "valid reordered species were not preserved after validation failure");

  AsyncOutputWriter nonroot;
  fs::create_directories(base / "nonroot");
  configure(nonroot, output(base / "nonroot"), domain(1, 2));
  auto empty = capture(domain(1, 2), 0);
  empty.particles[0] = tpic::data::ParticleRecords{};
  nonroot.submit(std::move(empty));
  nonroot.host_wait_piece();
  nonroot.complete_snapshot();
  require(file_contents(base / "nonroot" / "particles_t000000_r000001.vtp").find("NumberOfPoints=\"0\"") != std::string::npos &&
          !fs::exists(base / "nonroot" / "particles_t000000.pvtp"),
          "empty nonroot particle piece/publication is incorrect");
}

void particle_overwrite_and_failure(const fs::path& base) {
  const auto directory = base / "collision";
  fs::create_directory(directory);
  const auto piece = directory / "particles_t000000_r000000.vtp";
  { std::ofstream file(piece); file << "existing particle data"; }
  {
    AsyncOutputWriter writer;
    configure(writer, output(directory), domain());
    writer.submit(capture(domain(), 0));
    rejects<std::system_error>([&] { writer.host_wait_piece(); }, "particle overwrite was implicitly authorized");
    rejects<std::system_error>([&] { writer.complete_snapshot(); }, "failed particle capture was published");
    require(file_contents(piece) == "existing particle data", "particle failure changed existing output");
  }
  auto config = output(directory);
  config.overwrite_output = true;
  AsyncOutputWriter writer;
  configure(writer, config, domain());
  writer.submit(capture(domain(), 0));
  writer.host_wait_piece();
  writer.complete_snapshot();
  require(file_contents(piece).starts_with("<?xml"), "explicit particle overwrite failed");
}

void destructor_drain(const fs::path& base) {
  const auto directory = base / "destructor";
  {
    AsyncOutputWriter writer;
    fs::create_directories(directory);
    configure(writer, output(directory), domain());
    writer.submit(capture(domain(), 0, false, true, true));
  }
  const auto file = file_contents(directory / "particles_t000000_r000000.vtp");
  require(file.ends_with("</VTKFile>\n") &&
          vtk_appended_array<std::uint64_t>(file, "id") ==
              std::vector<std::uint64_t>{9007199254740993ULL, 18446744073709551614ULL},
          "destruction abandoned accepted particle output");
  require(!fs::exists(directory / "particles_t000000.pvtp"),
          "destruction published without rank agreement");
}

struct Probe {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false, released = false, published = false, fail = false;
  std::thread::id worker, publisher, setup;
  const double* particle_storage = nullptr;
};

class BlockingEngine final : public tpic::io::OutputEngine {
public:
  explicit BlockingEngine(Probe& probe) : probe_(probe) {}
  void write_piece(tpic::data::OutputSnapshotView snapshot) override {
    std::unique_lock lock(probe_.mutex);
    probe_.worker = std::this_thread::get_id();
    probe_.entered = true;
    probe_.changed.notify_all();
    probe_.changed.wait(lock, [&] { return probe_.released; });
    if (probe_.fail) throw std::runtime_error("injected worker failure");
    require(snapshot.particles[0].id[0] == 9007199254740993ULL,
            "worker lost the particle records");
    require(first_fraction(snapshot.particles[0]) == probe_.particle_storage &&
            snapshot.arrays.empty() && !snapshot.fields_present,
            "worker copied the particle records or received inactive field storage");
  }
  void publish() override {
    probe_.publisher = std::this_thread::get_id();
    probe_.published = true;
  }
  const std::string& manifest_path() const noexcept override { return path_; }
private:
  Probe& probe_;
  std::string path_ = "probe-manifest";
};

void deterministic_worker_handoff(const fs::path& base) {
  for (const bool failure : {false, true}) {
    // The writer drains its worker before the borrowed probe is destroyed.
    Probe probe;
    probe.fail = failure;
    AsyncOutputWriter writer;
    const tpic::config::PreparedOutputSelection selection(output(base / "injected"));
    writer.configure(selection, domain(), std::make_unique<BlockingEngine>(probe));
    writer.start([&probe]() noexcept { probe.setup = std::this_thread::get_id(); });
    auto snapshot = capture(domain(), 0);
    probe.particle_storage = first_fraction(snapshot.particles[0]);
    snapshot.arrays.push_back({"retained-inactive-field", 1, {0.0}});
    writer.submit(std::move(snapshot));
    bool entered;
    {
      std::unique_lock lock(probe.mutex);
      entered = probe.changed.wait_for(lock, std::chrono::seconds(5), [&] { return probe.entered; });
      // Always release before assertions so even a failed test can drain safely.
      probe.released = true;
    }
    probe.changed.notify_all();
    require(entered && probe.worker != std::this_thread::get_id(),
            "serialization did not execute independently on a CPU worker");
    require(probe.setup == probe.worker, "worker setup did not run on the worker before its piece");
    require(!probe.published, "worker published before controller agreement");
    if (failure) {
      rejects<std::runtime_error>([&] { writer.host_wait_piece(); }, "injected worker failure was lost");
      rejects<std::runtime_error>([&] { writer.complete_snapshot(); }, "failed epoch was published");
      require(!probe.published, "failed worker published metadata");
    } else {
      writer.host_wait_piece();
      require(!probe.published, "piece wait published metadata");
      writer.complete_snapshot();
      require(probe.published && probe.publisher == std::this_thread::get_id(),
              "publication did not execute on the controller thread");
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  tpic::test::MpiFieldFiles files(argc, argv);
  mpi = &files;
  return files.run("async_output_writer", [] {
    TemporaryDirectory temporary;
    particle_roundtrip_and_independent_cadence(temporary.path);
    independent_channel_configuration(temporary.path);
    channel_failure_blocks_publication(temporary.path);
    validation_and_empty_rank(temporary.path);
    particle_overwrite_and_failure(temporary.path);
    destructor_drain(temporary.path);
    deterministic_worker_handoff(temporary.path);
  });
}
