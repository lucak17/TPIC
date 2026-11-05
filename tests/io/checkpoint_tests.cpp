#include <tpic/config/boundary_conditions.hpp>
#include <tpic/io/checkpoint.hpp>
#include <tpic/io/diagnostics_writer.hpp>

#include "../support/test_builders.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
namespace fs = std::filesystem;
using tpic::config::OutputEngineKind;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory() {
    auto pattern = (fs::temp_directory_path() / "tpic-checkpoint-XXXXXX").string();
    require(::mkdtemp(pattern.data()) != nullptr, "cannot create test directory");
    path = pattern;
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
};

// The text written to std::cerr while it lives.
struct CapturedErrors {
  std::ostringstream text;
  std::streambuf* const previous = std::cerr.rdbuf(text.rdbuf());
  ~CapturedErrors() { std::cerr.rdbuf(previous); }
};

using Grid = std::array<int, 3>;
constexpr Grid cells{6, 6, 4};
constexpr int species = 2;

tpic::config::PreparedDomain domain(Grid processes, int rank) {
  tpic::config::GridConfig grid;
  grid.lx = 12;
  grid.ly = 6;
  grid.lz = 4;
  grid.nxc = cells[0];
  grid.nyc = cells[1];
  grid.nzc = cells[2];
  tpic::config::TopologyConfig topology;
  topology.xlen = processes[0];
  topology.ylen = processes[1];
  topology.zlen = processes[2];
  topology.periodic_y = topology.periodic_y_particles = false;
  return tpic::config::prepare_domain(grid, topology, rank, processes[0] * processes[1] * processes[2]);
}

int global_index(std::array<int, 3> cell) { return cell[0] + cells[0] * (cell[1] + cells[1] * cell[2]); }

// Species 0 holds 1 + g % 3 records in global cell g, species 1 none. A
// record's values derive from its id, which encodes its global cell.
std::uint64_t record_id(int cell, int record) { return std::uint64_t(cell) * 16 + std::uint64_t(record); }
std::array<double, 3> record_fraction(std::uint64_t id) { return {(id % 7) / 8.0, (id % 5) / 6.0, 0.25}; }

double field_value(std::size_t field, std::array<int, 3> global) {
  return double(field) * 1000 + global[0] + 10.0 * global[1] + 100.0 * global[2];
}

tpic::data::CheckpointHeader header(Grid processes, int cycle) {
  tpic::data::CheckpointHeader result;
  result.cycle = cycle;
  result.species = species;
  result.processes = processes;
  result.global_cells = cells;
  result.lengths = {12, 6, 4};
  result.dt = 0.125;
  result.c = 1;
  result.initial_energy = 42.5;
  result.faces.fill(std::int32_t(tpic::config::FieldBoundary::periodic));
  result.faces[2] = result.faces[3] = std::int32_t(tpic::config::FieldBoundary::perfect_electric_conductor);
  for (std::size_t face = 6; face < 12; ++face)
    result.faces[face] = std::int32_t(face == 8 || face == 9 ? tpic::config::ParticleBoundary::mirror
                                                              : tpic::config::ParticleBoundary::periodic);
  result.storage_bytes = {8, 8, 8, 8, 8};
  result.q_over_m = {-1, 1};
  return result;
}

// Writes the pieces of the ranks in `ranks` of a checkpoint on `processes`.
// The records of a piece are in cell order but for the last three, which
// form an unsorted tail.
void write_checkpoint(const fs::path& directory, OutputEngineKind engine, Grid processes, int cycle,
                      std::vector<int> ranks = {}) {
  const int count = processes[0] * processes[1] * processes[2];
  if (ranks.empty())
    for (int rank = 0; rank < count; ++rank) ranks.push_back(rank);
  auto checkpoint = header(processes, cycle);
  for (int g = 0; g < cells[0] * cells[1] * cells[2]; ++g) checkpoint.global_count[0] += 1 + g % 3;
  for (const int rank : ranks) {
    const auto piece = domain(processes, rank);
    tpic::config::PreparedOutput output;
    output.checkpoint_engine = engine;
    output.checkpoint_dir = directory.string();
    output.checkpoint_cycle = 1;
    tpic::data::OwnedOutputSnapshot snapshot;
    snapshot.cycle = cycle;
    snapshot.time = cycle * 0.125;
    snapshot.fields_present = false;
    snapshot.checkpoint_present = true;
    snapshot.checkpoint = checkpoint;
    for (std::size_t field = 0; field < 6; ++field) {
      const int nodes = field < 3 ? 1 : 0;
      std::vector<double> values;
      for (int k = 0; k < piece.local_cells[2] + nodes; ++k)
        for (int j = 0; j < piece.local_cells[1] + nodes; ++j)
          for (int i = 0; i < piece.local_cells[0] + nodes; ++i)
            values.push_back(field_value(field, {piece.global_offset[0] + i, piece.global_offset[1] + j,
                                                 piece.global_offset[2] + k}));
      snapshot.checkpoint_fields[field] = std::move(values);
    }
    tpic::test::ParticleRecordStorage records, empty;
    for (int k = 0; k < piece.local_cells[2]; ++k)
      for (int j = 0; j < piece.local_cells[1]; ++j)
        for (int i = 0; i < piece.local_cells[0]; ++i) {
          const int g = global_index({piece.global_offset[0] + i, piece.global_offset[1] + j,
                                      piece.global_offset[2] + k});
          for (int record = 0; record <= g % 3; ++record) {
            const auto id = record_id(g, record);
            records.add(piece, {i, j, k}, record_fraction(id), {double(id), -1.0 * id, 0.5}, -2.0, id);
          }
        }
    // Move the first three records behind the others: an unsorted tail.
    const auto rotate = [](auto& values) { std::rotate(values.begin(), values.begin() + 3, values.end()); };
    for (std::size_t axis = 0; axis < 3; ++axis) {
      rotate(records.fraction[axis]);
      rotate(records.velocity[axis]);
    }
    rotate(records.charge);
    rotate(records.cell);
    rotate(records.id);
    snapshot.particles = {records.records(0), empty.records(1)};
    tpic::io::make_checkpoint_engine(output, piece)->write_piece(tpic::data::OutputSnapshotView(snapshot));
  }
}

// Restarts every rank of `processes`: each record arrives once, in its own
// cell, with its values; every interior field value is the checkpoint's.
void check_restart(const fs::path& directory, Grid processes, int cycle, std::size_t staging_records) {
  const auto checkpoint = tpic::io::find_restart_checkpoint(directory.string());
  require(checkpoint.header.cycle == cycle && checkpoint.header.initial_energy == 42.5 &&
              checkpoint.header.q_over_m[1] == 1 && checkpoint.header.faces == header(processes, cycle).faces,
          "checkpoint header did not round-trip");
  std::set<std::uint64_t> seen;
  for (int rank = 0; rank < processes[0] * processes[1] * processes[2]; ++rank) {
    const auto local = domain(processes, rank);
    const auto restart = tpic::io::prepare_restart(checkpoint, local);
    // Fields: a padded host image, one ghost layer on each side.
    const std::size_t stride_y = std::size_t(local.local_cells[0]) + 3;
    const std::size_t stride_z = stride_y * (std::size_t(local.local_cells[1]) + 3);
    std::array<std::vector<double>, 6> images;
    std::array<tpic::io::FieldDestination, 6> destinations;
    for (std::size_t field = 0; field < 6; ++field) {
      images[field].assign(stride_z * (std::size_t(local.local_cells[2]) + 3), -1);
      destinations[field] = {images[field].data() + 1 + stride_y + stride_z, stride_y, stride_z};
    }
    tpic::io::read_restart_fields(restart, local, destinations);
    for (std::size_t field = 0; field < 6; ++field) {
      const int nodes = field < 3 ? 1 : 0;
      for (int k = 0; k < local.local_cells[2] + nodes; ++k)
        for (int j = 0; j < local.local_cells[1] + nodes; ++j)
          for (int i = 0; i < local.local_cells[0] + nodes; ++i)
            require(images[field][1 + i + (1 + j) * stride_y + (1 + k) * stride_z] ==
                        field_value(field, {local.global_offset[0] + i, local.global_offset[1] + j,
                                            local.global_offset[2] + k}),
                    "restored field value differs");
    }
    // Particles, through a small staging so that the chunking is exercised.
    std::array<std::vector<double>, 3> fraction, velocity;
    std::vector<double> charge;
    std::vector<std::int32_t> cell(staging_records);
    std::vector<std::uint64_t> id(staging_records);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      fraction[axis].resize(staging_records);
      velocity[axis].resize(staging_records);
    }
    charge.resize(staging_records);
    const tpic::io::ParticleStaging staging{
        .fraction = {std::span(fraction[0]), std::span(fraction[1]), std::span(fraction[2])},
        .velocity = {std::span(velocity[0]), std::span(velocity[1]), std::span(velocity[2])},
        .charge = std::span(charge), .cell = cell, .id = id};
    const auto map = tpic::data::prepare_particle_cell_map(local);
    for (std::size_t s = 0; s < species; ++s) {
      std::uint64_t delivered = 0;
      tpic::io::read_restart_particles(restart, local, s, staging, [&](std::size_t count) {
        require(s == 0, "an empty species delivered records");
        for (std::size_t r = 0; r < count; ++r) {
          const auto at = map.decode(cell[r]);
          require(map.contains_interior(at.i, at.j, at.k), "restored record outside the rank's cells");
          const int g = global_index({local.global_offset[0] + at.i, local.global_offset[1] + at.j,
                                      local.global_offset[2] + at.k});
          require(id[r] / 16 == std::uint64_t(g), "restored record in another cell");
          require(seen.insert(id[r]).second, "record restored twice");
          const auto expected = record_fraction(id[r]);
          for (std::size_t axis = 0; axis < 3; ++axis)
            require(fraction[axis][r] == expected[axis], "restored fraction differs");
          require(velocity[0][r] == double(id[r]) && velocity[1][r] == -1.0 * id[r] && velocity[2][r] == 0.5 &&
                      charge[r] == -2.0,
                  "restored record values differ");
        }
        delivered += count;
      });
      require(delivered == restart.local_count[s], "restored records differ from the prepared count");
    }
  }
  require(seen.size() == checkpoint.header.global_count[0], "records lost in the restart");
}

std::vector<std::string> listing(const fs::path& directory) {
  std::vector<std::string> names;
  for (const auto& entry : fs::directory_iterator(directory)) names.push_back(entry.path().filename().string());
  std::sort(names.begin(), names.end());
  return names;
}

void run(OutputEngineKind engine) {
  // Restarts on the writer's grid, with fewer, more and differently split ranks.
  {
    TemporaryDirectory directory;
    write_checkpoint(directory.path, engine, {2, 1, 1}, 5);
    for (const Grid processes : {Grid{2, 1, 1}, Grid{1, 1, 1}, Grid{3, 1, 1}, Grid{1, 3, 2}, Grid{3, 1, 2}})
      for (const std::size_t staging : {std::size_t{1}, std::size_t{4}, tpic::io::restart_staging_records})
        check_restart(directory.path, processes, 5, staging);
  }
  // The first checkpoint creates RestartDirName; its retention warns of nothing.
  {
    TemporaryDirectory directory;
    const CapturedErrors errors;
    write_checkpoint(directory.path / "restart", engine, {1, 1, 1}, 5);
    require(errors.text.str().empty(), "the first checkpoint warned: " + errors.text.str());
  }
  // Discovery, set-aside and retention.
  TemporaryDirectory directory;
  write_checkpoint(directory.path, engine, {2, 1, 1}, 5);
  write_checkpoint(directory.path, engine, {2, 1, 1}, 10, {1});
  tpic::io::prepare_checkpoint_directory((directory.path / "missing").string(), nullptr, false);
  bool refused = false;
  try { tpic::io::prepare_checkpoint_directory(directory.path.string(), nullptr, false); }
  catch (const std::runtime_error&) { refused = true; }
  require(refused, "a fresh run accepted the checkpoints of an earlier run");
  check_restart(directory.path, {1, 2, 1}, 5, 2);
  const auto restart = tpic::io::find_restart_checkpoint(directory.path.string());
  tpic::io::prepare_checkpoint_directory(directory.path.string(), &restart, false);
  require(listing(directory.path) == std::vector<std::string>{"abandoned_checkpoint_t000010_n000002",
                                                              "checkpoint_t000005_n000002"},
          "the incomplete newer checkpoint was not set aside");
  write_checkpoint(directory.path, engine, {1, 1, 1}, 15);
  require(listing(directory.path) == std::vector<std::string>{"checkpoint_t000005_n000002",
                                                              "checkpoint_t000015_n000001"},
          "the root did not remove the set-aside checkpoint");
  write_checkpoint(directory.path, engine, {1, 1, 1}, 20);
  require(listing(directory.path) == std::vector<std::string>{"checkpoint_t000015_n000001",
                                                              "checkpoint_t000020_n000001"},
          "retention did not keep the newest complete and the new checkpoint");
  tpic::io::prepare_checkpoint_directory(directory.path.string(), nullptr, true);
  require(listing(directory.path) == std::vector<std::string>{"abandoned_checkpoint_t000015_n000001",
                                                              "abandoned_checkpoint_t000020_n000001"},
          "--overwrite-output did not set the checkpoints aside");
  refused = false;
  try { (void)tpic::io::find_restart_checkpoint(directory.path.string()); }
  catch (const std::runtime_error&) { refused = true; }
  require(refused, "a restart found a set-aside checkpoint");
}

// diagnostics.csv continues at the restart cycle: earlier rows stay, later
// ones go, and the energy change keeps the checkpoint's baseline.
void check_csv_resume() {
  TemporaryDirectory directory;
  tpic::config::PreparedOutput output;
  output.save_dir = directory.path.string();
  output.diagnostics_cycle = 1;
  const tpic::config::PrecisionMetadata precision{"double", "all double"};
  const auto sample = [](int cycle, double energy) {
    tpic::data::ScalarDiagnosticSample result;
    result.cycle = cycle;
    result.time = cycle;
    result.species_count = 1;
    result.values[0] = energy;
    return result;
  };
  const auto rows = [&] {
    std::ifstream csv(directory.path / "diagnostics.csv");
    std::vector<std::string> result;
    for (std::string line; std::getline(csv, line);)
      if (!line.empty() && line.front() != '#' && !line.starts_with("cycle,")) result.push_back(line);
    return result;
  };
  {
    tpic::io::DiagnosticsWriter writer;
    writer.open(output, 1, precision);
    for (int cycle = 0; cycle < 7; ++cycle) writer.write(sample(cycle, 10 + cycle), 10);
    writer.close();
  }
  // An interrupted last row is dropped; the rows before the restart stay.
  std::ofstream(directory.path / "diagnostics.csv", std::ios::app) << "7,7,pre_s";
  for (const int cycle : {9, 4}) {
    tpic::io::DiagnosticsWriter writer;
    writer.open(output, 1, precision, cycle);
    writer.write(sample(cycle, 20), 8);
    writer.close();
    const auto kept = rows();
    require(kept.size() == std::size_t(std::min(cycle, 7) + 1) && kept[kept.size() - 2].starts_with(
                std::to_string(std::min(cycle, 7) - 1) + ",") &&
                kept.back().starts_with(std::to_string(cycle) + "," + std::to_string(cycle) +
                                        ",pre_solve,20,0,0,20,12,"),
            "diagnostics.csv did not continue at the restart cycle");
  }
  std::ostringstream text;
  text << std::ifstream(directory.path / "diagnostics.csv").rdbuf();
  require(text.str().find("# restart at cycle 4: precision_profile = double") != std::string::npos,
          "the restart is not marked in diagnostics.csv");

  // gauss.csv: rows of samples begun before the restart stay, later ones go,
  // and a row must follow the last one in cycle and time.
  {
    tpic::io::GaussWriter writer;
    writer.open(output, precision);
    for (const int cycle : {0, 3, 6, 9}) writer.write({cycle, cycle + .5, {1, 2, 3, 4}});
    bool refused = false;
    try { writer.write({9, 9.5, {}}); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused, "gauss.csv accepted a repeated cycle");
    writer.close();
  }
  {
    tpic::io::GaussWriter writer;
    writer.open(output, precision, 6);
    writer.write({6, 6.5, {5, 6, 7, 8}});
    writer.close();
  }
  std::ifstream gauss(directory.path / "gauss.csv");
  std::vector<std::string> gauss_rows;
  for (std::string line; std::getline(gauss, line);)
    if (!line.empty() && line.front() != '#') gauss_rows.push_back(line);
  require(gauss_rows == std::vector<std::string>{"cycle,time,residual_rms,drift_rms,wall_residual_rms,charge_rms",
                                                 "0,0.5,1,2,3,4", "3,3.5,1,2,3,4", "6,6.5,5,6,7,8"},
          "gauss.csv did not continue at the restart cycle");
}

} // namespace

int main() {
  try {
#if TPIC_HAS_ADIOS2
    run(OutputEngineKind::Adios2);
#endif
#if TPIC_HAS_HDF5
    run(OutputEngineKind::Hdf5);
#endif
    check_csv_resume();
    std::cout << "TPIC checkpoint checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
