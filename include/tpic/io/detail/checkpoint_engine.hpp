#ifndef TPIC_IO_DETAIL_CHECKPOINT_ENGINE_HPP
#define TPIC_IO_DETAIL_CHECKPOINT_ENGINE_HPP

#include <tpic/config/domain.hpp>
#include <tpic/config/output.hpp>
#include <tpic/data/output/checkpoint.hpp>
#include <tpic/data/particles/particle_cell.hpp>
#include <tpic/io/output_engine.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace tpic::io::detail {

// The layout of a checkpoint piece, shared by the writers and the readers of
// both backends. Root attributes: schema and the header (visit_checkpoint_header).
// Datasets, C order: fields/<name> of data::checkpoint_field_names; per species
// species_<s>/ the records rx, ry, rz, u, v, w, q, id in their stored types, in
// the order of the device owner, with the attributes count and sorted_prefix.
// Records [0, sorted_prefix) are in cell order: those of local cell (i,j,k)
// are [prefix_start[k][j][i], prefix_start[k][j][i + 1]), and entry nx of a row
// is its end (uint32, shape nz, ny, nx + 1). tail_cell holds the rank-local
// cell keys (data::ParticleCellMap, particle halo 2) of records
// [sorted_prefix, count). An empty array is not written.
inline constexpr const char* checkpoint_schema = "tpic-checkpoint-v1";

[[nodiscard]] inline std::string checkpoint_species_path(std::size_t species) {
  return "species_" + std::to_string(species);
}

// The header attributes, in one order for the writers and the readers.
template <class Header, class Visitor> void visit_checkpoint_header(Header& header, Visitor&& visit) {
  visit("cycle", header.cycle);
  visit("species", header.species);
  visit("processes", header.processes);
  visit("global_cells", header.global_cells);
  visit("lengths", header.lengths);
  visit("dt", header.dt);
  visit("c", header.c);
  visit("initial_energy", header.initial_energy);
  visit("faces", header.faces);
  visit("storage_bytes", header.storage_bytes);
  visit("q_over_m", header.q_over_m);
  visit("global_count", header.global_count);
}

// visit(name, span) for each payload array of records (stored or restored),
// a stored-type variant as the span it holds.
template <class Records, class Visitor> void visit_checkpoint_records(Records& records, Visitor&& visit) {
  constexpr std::array<const char*, 8> names{"rx", "ry", "rz", "u", "v", "w", "q", "id"};
  std::size_t name = 0;
  const auto each = [&](const auto& values) {
    if constexpr (requires { values.index(); })
      std::visit([&](auto span) { visit(names[name], span); }, values);
    else
      visit(names[name], values);
    ++name;
  };
  for (const auto& values : records.fraction) each(values);
  for (const auto& values : records.velocity) each(values);
  each(records.charge);
  each(records.id);
}

// One read of a checkpoint dataset: the box [start, start + count) of its
// first `dimensions` axes, slowest first, into destination, which holds the
// product of the counts.
using CheckpointDestination = std::variant<std::span<float>, std::span<double>, std::span<std::uint32_t>,
                                           std::span<std::int32_t>, std::span<std::uint64_t>>;
struct CheckpointRead {
  std::string dataset;
  std::array<std::size_t, 3> start{}, count{};
  int dimensions = 1;
  CheckpointDestination destination;
};

// An open checkpoint piece, read on the calling thread at startup.
class CheckpointReader {
public:
  virtual ~CheckpointReader() = default;
  // Checks the schema.
  [[nodiscard]] virtual data::CheckpointHeader header() = 0;
  // The records of a species in this piece and their sorted prefix.
  [[nodiscard]] virtual std::array<std::uint64_t, 2> records(std::size_t species) = 0;
  virtual void read(std::span<const CheckpointRead> reads) = 0;
};

// The writer of one rank's checkpoint pieces.
class CheckpointEngine : public OutputEngine {
public:
  CheckpointEngine(const config::PreparedOutput& output, const config::PreparedDomain& domain,
                   std::string extension);
  void write_piece(data::OutputSnapshotView snapshot) final;
  void publish() final {}
  // The directory of the last checkpoint.
  [[nodiscard]] const std::string& manifest_path() const noexcept final { return checkpoint_; }

protected:
  // Writes the complete piece at path.
  virtual void write_file(const std::string& path, data::OutputSnapshotView snapshot) = 0;

  // visit(name, shape, span) for every array of the piece of snapshot.
  template <class Visitor> void visit_arrays(data::OutputSnapshotView snapshot, Visitor&& visit) const {
    const std::array nodes{cells_[0] + 1, cells_[1] + 1, cells_[2] + 1};
    for (std::size_t field = 0; field < snapshot.checkpoint_fields.size(); ++field)
      std::visit([&]<class T>(const std::vector<T>& values) {
        visit(std::string("fields/") + data::checkpoint_field_names[field],
              std::span<const std::size_t>(field < data::checkpoint_node_fields ? nodes : cells_),
              std::span<const T>(values));
      }, snapshot.checkpoint_fields[field]);
    const std::array index_shape{cells_[0], cells_[1], cells_[2] + 1};
    for (std::size_t s = 0; s < snapshot.particles.size(); ++s) {
      const auto& records = snapshot.particles[s];
      const auto path = checkpoint_species_path(s) + "/";
      const std::array shape{records.id.size()};
      visit_checkpoint_records(records, [&](const char* name, auto values) { visit(path + name, shape, values); });
      visit(path + "prefix_start", index_shape, std::span<const std::uint32_t>(index_[s].prefix_start));
      const auto tail = records.cell.subspan(index_[s].sorted_prefix);
      visit(path + "tail_cell", std::array{tail.size()}, tail);
    }
  }

  struct SpeciesIndex {
    std::uint64_t sorted_prefix{};
    std::vector<std::uint32_t> prefix_start;
  };
  std::array<std::size_t, 3> cells_{}; // local cells, z,y,x
  std::vector<SpeciesIndex> index_;

private:
  void index_species(const data::ParticleRecords& records, SpeciesIndex& index) const;

  std::string directory_, extension_, checkpoint_;
  data::ParticleCellMap map_;
  int rank_{}, ranks_{};
};

std::unique_ptr<CheckpointEngine> make_adios2_checkpoint_engine(const config::PreparedOutput& output,
                                                                const config::PreparedDomain& domain);
std::unique_ptr<CheckpointEngine> make_hdf5_checkpoint_engine(const config::PreparedOutput& output,
                                                              const config::PreparedDomain& domain);
std::unique_ptr<CheckpointReader> open_adios2_checkpoint(const std::string& path);
std::unique_ptr<CheckpointReader> open_hdf5_checkpoint(const std::string& path);

} // namespace tpic::io::detail

#endif // TPIC_IO_DETAIL_CHECKPOINT_ENGINE_HPP
