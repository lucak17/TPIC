#include <tpic/config/domain.hpp>
#include <tpic/io/async_output_writer.hpp>

#include <filesystem>
#include <utility>
#include <vector>

#include "../support/mpi_field_files.hpp"

// Exercise the production writer and its collective field files without a GPU.
// Every rank writes its nodes of each cycle's files; a snapshot of fields alone
// publishes nothing and is complete when the writer returns it.
void write(tpic::test::MpiFieldFiles& mpi, const std::filesystem::path& directory, int cycles,
           bool selected) {
  tpic::config::PreparedOutput output;
  output.save_dir = directory.string();
  output.field_cycle = 1;
  output.output_e = true;
  output.output_b = !selected;
  output.total_selected[tpic::config::output_rho_quantity] = true;
  output.species_selected.fill({false});
  const tpic::config::PreparedOutputSelection selection(output);
  tpic::config::GridConfig grid;
  grid.nxc = 7; grid.nyc = grid.nzc = 2;
  tpic::config::TopologyConfig topology;
  topology.xlen = mpi.ranks();
  const auto domain = tpic::config::prepare_domain(grid, topology, mpi.rank(), mpi.ranks());
  // The run creates SaveDirName before any rank writes; the first collective
  // call of the first write orders this creation before every open.
  if (mpi.rank() == 0) std::filesystem::create_directories(directory);
  tpic::io::AsyncOutputWriter writer;
  tpic::test::configure(writer, output, domain, mpi.files);
  for (int cycle = 0; cycle <= cycles; ++cycle) {
    tpic::data::OwnedOutputSnapshot snapshot;
    snapshot.cycle = cycle;
    snapshot.time = 0.125 * cycle;
    snapshot.final_state = cycle == cycles;
    for (const auto& array : selection.arrays()) {
      std::vector<double> values;
      for (int z = 0; z <= domain.local_cells[2]; ++z)
        for (int y = 0; y <= domain.local_cells[1]; ++y)
          for (int x = 0; x <= domain.local_cells[0]; ++x)
            for (int component = 0; component < array.components; ++component)
              values.push_back(cycle * 1000 + z * 100 + y * 10 +
                               x + domain.global_offset[0] + component * 0.25);
      snapshot.arrays.push_back({array.name, array.components, std::move(values)});
    }
    writer.submit(std::move(snapshot));
    writer.host_wait_piece();
    (void)writer.complete_snapshot();
  }
}

// Writes <root>/r1, zero and selected.
int main(int argc, char** argv) {
  tpic::test::MpiFieldFiles mpi(argc, argv);
  if (argc != 2) return 1;
  return mpi.run("write_field_fixture", [&] {
    const std::filesystem::path root(argv[1]);
    write(mpi, root / "r1", 2, false);
    write(mpi, root / "zero", 0, false);
    write(mpi, root / "selected", 1, true);
  });
}
