#include <tpic/config/domain.hpp>
#include <tpic/io/async_output_writer.hpp>
#include <tpic/data/output/output_snapshot.hpp>

#include "../support/mpi_field_files.hpp"
#include "../support/test_builders.hpp"

#include <lpfx/distributed/runtime/host_control.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tpic::io::AsyncOutputWriter;
using tpic::data::OwnedOutputSnapshot;

tpic::test::MpiFieldFiles* mpi = nullptr;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Exception, class Function>
void rejects(Function&& function, const char* message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void barrier() {
  require(lpfx::distributed::host_barrier(mpi->domain).succeeded(), "test barrier failed");
}

// Rank 0 makes an empty directory that every rank sees after the barrier.
fs::path fresh_directory(const fs::path& path) {
  if (mpi->rank() == 0) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  barrier();
  return path;
}

// 7 x 6 x 2 cells; three ranks split x into 3, 2 and 2 cells or y into 2 each.
constexpr std::array<int, 3> cells{7, 6, 2};

tpic::config::PreparedDomain domain(const std::array<int, 3>& processes) {
  tpic::config::GridConfig grid;
  grid.nxc = cells[0];
  grid.nyc = cells[1];
  grid.nzc = cells[2];
  tpic::config::TopologyConfig topology;
  topology.xlen = processes[0];
  topology.ylen = processes[1];
  topology.zlen = processes[2];
  return tpic::config::prepare_domain(grid, topology, mpi->rank(), mpi->ranks());
}

tpic::config::PreparedOutput output(const fs::path& directory, bool overwrite = false) {
  tpic::config::PreparedOutput result;
  result.save_dir = directory.string();
  result.overwrite_output = overwrite;
  result.field_cycle = 1;
  result.output_e = true;
  result.output_gauss = true;
  result.total_selected[tpic::config::output_rho_quantity] = true;
  result.species_selected.fill({false});
  return result;
}

void configure(AsyncOutputWriter& writer, const tpic::config::PreparedOutput& prepared,
               const tpic::config::PreparedDomain& geometry) {
  tpic::test::configure(writer, prepared, geometry, mpi->files);
}

// Nonzero everywhere, so an unwritten node of a fresh file shows.
double value(int cycle, int z, int y, int x, int component) {
  return 1 + cycle * 1000 + z * 100 + y * 10 + x + component * 0.25;
}

// The closed local node box. Nodes a rank shares with its upper neighbour
// carry a rank-local offset, so a file holding them shows the wrong owner. The
// cell array (Gauss) holds the local cells and is absent at cycle 0.
OwnedOutputSnapshot capture(const tpic::config::PreparedDomain& geometry, int cycle,
                            bool final_state = false) {
  OwnedOutputSnapshot snapshot{cycle, cycle * 0.25, final_state, {}};
  const auto& local = geometry.local_cells;
  const auto& offset = geometry.global_offset;
  const tpic::config::PreparedOutputSelection selection(output({}));
  for (const auto& array : selection.arrays()) {
    std::vector<double> values;
    const int node = array.cell ? 0 : 1;
    for (int z = 0; z < local[2] + node && !(array.cell && cycle == 0); ++z)
      for (int y = 0; y < local[1] + node; ++y)
        for (int x = 0; x < local[0] + node; ++x) {
          const bool shared = (x == local[0] && offset[0] + x != cells[0]) ||
                              (y == local[1] && offset[1] + y != cells[1]) ||
                              (z == local[2] && offset[2] + z != cells[2]);
          for (int component = 0; component < array.components; ++component)
            values.push_back(value(cycle, offset[2] + z, offset[1] + y, offset[0] + x, component) +
                             (shared ? 1.0e6 * (mpi->rank() + 1) : 0.0));
        }
    snapshot.arrays.push_back({array.name, array.components, std::move(values), array.cell});
  }
  return snapshot;
}

std::string file_name(const std::string& array, int cycle) {
  const auto digits = std::to_string(cycle);
  return array + "_t" + std::string(6 - digits.size(), '0') + digits + ".vti";
}

// The whole-grid piece of one array, every node (or cell) once with its owner's value.
void check_file(const fs::path& directory, const std::string& name, int cycle, bool final_state,
                int components, bool cell = false) {
  const auto data = tpic::test::file_contents(directory / file_name(name, cycle));
  const auto global = std::to_string(cells[0]) + " 0 " + std::to_string(cells[1]) + " 0 " +
                      std::to_string(cells[2]);
  require(data.find("WholeExtent=\"0 " + global + "\"") != std::string::npos &&
              data.find("<Piece Extent=\"0 " + global + "\">") != std::string::npos,
          "field file is not one whole-grid piece");
  require(data.find("NumberOfComponents=\"" + std::to_string(components) + "\" format=\"appended\" offset=\"0\"") !=
              std::string::npos &&
              data.find(cell ? "<PointData/>\n      <CellData>" : "<PointData>\n") != std::string::npos,
          "field file array descriptor");
  require(data.find("Name=\"cycle\" NumberOfTuples=\"1\" format=\"ascii\">" + std::to_string(cycle) + "<") !=
                  std::string::npos &&
              data.find(std::string("Name=\"final_state\" NumberOfTuples=\"1\" format=\"ascii\">") +
                        (final_state ? "1" : "0")) != std::string::npos,
          "field file epoch metadata");
  std::vector<double> expected;
  const int node = cell ? 0 : 1;
  for (int z = 0; z < cells[2] + node; ++z)
    for (int y = 0; y < cells[1] + node; ++y)
      for (int x = 0; x < cells[0] + node; ++x)
        for (int component = 0; component < components; ++component)
          expected.push_back(value(cycle, z, y, x, component));
  require(tpic::test::vtk_appended_array<double>(data, name) == expected,
          "field file holds a node not written by its owner");
  require(data.ends_with("\n  </AppendedData>\n</VTKFile>\n"), "field file is not closed after its array");
}

// Every cycle writes one file per array, for a split along x and one along y.
void writes_one_file_per_array(const fs::path& directory) {
  for (const auto& [name, processes] : {std::pair{"x", std::array{mpi->ranks(), 1, 1}},
                                        std::pair{"y", std::array{1, mpi->ranks(), 1}}}) {
    const auto geometry = domain(processes);
    const auto target = directory / name;
    AsyncOutputWriter writer;
    configure(writer, output(fresh_directory(target)), geometry);
    rejects<std::logic_error>([&] { writer.host_wait_piece(); }, "wait accepted an idle writer");
    rejects<std::logic_error>([&] { writer.complete_snapshot(); }, "completion accepted an idle writer");
    for (int cycle = 0; cycle < 3; ++cycle) {
      auto snapshot = capture(geometry, cycle, cycle == 2);
      const auto* array_storage = snapshot.arrays.data();
      const auto* value_storage = snapshot.arrays.front().values.data();
      writer.submit(std::move(snapshot));
      require(writer.pending(), "submitted snapshot did not occupy its slot");
      auto next = capture(geometry, cycle + 1);
      rejects<std::logic_error>([&] { writer.submit(std::move(next)); },
                               "writer accepted a second pending snapshot");
      rejects<std::logic_error>([&] { (void)writer.fields_path(); },
                               "pending writer exposed mutable path storage");
      writer.host_wait_piece();
      require(writer.pending(), "local completion released the snapshot early");
      auto reusable = writer.complete_snapshot();
      require(reusable.arrays.data() == array_storage &&
                  reusable.arrays.front().values.data() == value_storage && reusable.cycle == cycle &&
                  reusable.arrays.front().values == capture(geometry, cycle).arrays.front().values,
              "writer completion did not return the original owned capture for reuse");
      require(!writer.pending() && writer.fields_path() == (target / file_name("E", cycle)).string(),
              "completion kept the slot or named another file");
      if (mpi->rank() == 0) {
        check_file(target, "E", cycle, cycle == 2, 3);
        check_file(target, "rho", cycle, cycle == 2, 1);
        if (cycle == 0) require(!fs::exists(target / file_name("Gauss", 0)), "an absent array wrote a file");
        else check_file(target, "Gauss", cycle, cycle == 2, 1, true);
      }
    }
    writer.submit(capture(geometry, 3));
    rejects<std::invalid_argument>([&] { writer.host_wait_piece(); },
                                  "writer accepted another epoch after final output");
    if (mpi->rank() == 0)
      require(std::distance(fs::directory_iterator(target), fs::directory_iterator{}) == 8,
              "field output wrote other files than one per present array and cycle");
  }
}

// Exclusive creation fails alike on every rank and keeps the file; overwriting
// replaces a longer stale file with the exact one.
void exclusive_and_replace(const fs::path& directory) {
  const auto geometry = domain({mpi->ranks(), 1, 1});
  const auto existing = directory / "exclusive";
  fresh_directory(existing);
  if (mpi->rank() == 0) std::ofstream(existing / file_name("E", 0)) << "preserve existing output";
  barrier();
  {
    AsyncOutputWriter writer;
    configure(writer, output(existing), geometry);
    writer.submit(capture(geometry, 0));
    try {
      writer.host_wait_piece();
      throw std::logic_error("an existing field file was overwritten");
    } catch (const std::runtime_error& failure) {
      require(std::string(failure.what()).find("--overwrite-output") != std::string::npos,
              "refusal does not name --overwrite-output");
    }
    rejects<std::runtime_error>([&] { writer.complete_snapshot(); }, "a failed piece was released");
  }
  if (mpi->rank() == 0)
    require(tpic::test::file_contents(existing / file_name("E", 0)) == "preserve existing output",
            "the writer changed an existing file");

  const auto stale = directory / "replace";
  fresh_directory(stale);
  if (mpi->rank() == 0) std::ofstream(stale / file_name("E", 0)) << std::string(1 << 16, 'x');
  barrier();
  AsyncOutputWriter writer;
  configure(writer, output(stale, true), geometry);
  writer.submit(capture(geometry, 0));
  writer.host_wait_piece();
  (void)writer.complete_snapshot();
  if (mpi->rank() == 0) check_file(stale, "E", 0, false, 3);
}

void destructor_drain(const fs::path& directory) {
  const auto geometry = domain({mpi->ranks(), 1, 1});
  {
    AsyncOutputWriter writer;
    configure(writer, output(fresh_directory(directory / "destructor")), geometry);
    writer.submit(capture(geometry, 0, true));
    // Destruction joins the worker, which completes the accepted collective write.
  }
  if (mpi->rank() == 0) check_file(directory / "destructor", "rho", 0, true, 1);
}

void capture_validation(const fs::path& directory) {
  const auto geometry = domain({mpi->ranks(), 1, 1});
  AsyncOutputWriter writer;
  configure(writer, output(directory / "validation"), geometry);
  auto malformed = capture(geometry, 0);
  malformed.arrays[0].values.pop_back();
  rejects<std::invalid_argument>([&] { writer.submit(std::move(malformed)); },
                                "invalid host capture was accepted");
  require(!writer.pending(), "capture validation failure occupied the slot");
}

} // namespace

int main(int argc, char** argv) {
  tpic::test::MpiFieldFiles files(argc, argv);
  mpi = &files;
  return files.run("vtk_field_output", [] {
    // In the working directory (the build tree), shared by the ranks of this case only.
    const auto directory = fresh_directory("tpic-vtk-fields-" + std::to_string(mpi->ranks()));
    writes_one_file_per_array(directory);
    exclusive_and_replace(directory);
    destructor_drain(directory);
    capture_validation(directory);
    barrier();
    if (mpi->rank() == 0) fs::remove_all(directory);
  });
}
