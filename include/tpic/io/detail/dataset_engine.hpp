#ifndef TPIC_IO_DETAIL_DATASET_ENGINE_HPP
#define TPIC_IO_DETAIL_DATASET_ENGINE_HPP

#include <tpic/io/output_engine.hpp>
#include <tpic/io/detail/output_layout.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace tpic::io::detail {

// Backend libraries write only inside a private sibling directory. Commit never
// follows the destination symlink and never replaces an existing output unless
// the caller explicitly enabled overwrite_output.
class StagedOutput {
public:
  StagedOutput(std::filesystem::path destination, bool overwrite);
  ~StagedOutput() noexcept;
  StagedOutput(const StagedOutput&) = delete;
  StagedOutput& operator=(const StagedOutput&) = delete;
  [[nodiscard]] const std::string& temporary_path() const noexcept { return temporary_; }
  void commit();
private:
  std::filesystem::path directory_, destination_;
  std::string temporary_;
  bool overwrite_;
};

// One channel of ADIOS2 or HDF5 output: rank pieces written by the adapter's
// write_piece and a JSON manifest published by the root.
class DatasetOutputEngine : public OutputEngine {
public:
  DatasetOutputEngine(data::OutputChannel channel,
                      const config::PreparedOutputSelection& selection,
                      const config::PreparedDomain& domain,
                      config::PrecisionMetadata precision, std::string backend,
                      std::string extension);
  void publish() override;
  [[nodiscard]] const std::string& manifest_path() const noexcept override { return manifest_; }
protected:
  struct FieldLayout {
    std::string path;
    std::array<std::size_t, 4> dimensions;
  };
  struct SpeciesLayout {
    std::string path, count_attribute;
    std::array<std::string, particle_array_count> arrays;
  };

  // Metadata policy and typed host-array traversal live here; adapters only
  // translate these values into their library's attribute and dataset APIs.
  template<class Visitor>
  void visit_attributes(data::OutputSnapshotView snapshot, const char* schema,
                        Visitor&& visit) const {
    visit("schema", schema);
    visit("channel", channel_);
    visit("precision_profile", precision_name_);
    visit("precision_roles", precision_roles_);
    visit("state", snapshot.final_state ? "final" : "pre_solve");
    visit("field_order", "z,y,x,component");
    visit("cycle", static_cast<std::int32_t>(snapshot.cycle));
    visit("time", snapshot.time);
    visit("final_state", static_cast<std::uint8_t>(snapshot.final_state));
    visit("fields_present", static_cast<std::uint8_t>(fields_));
    visit("particles_present", static_cast<std::uint8_t>(!fields_));
    visit("rank", static_cast<std::int32_t>(rank_));
    visit("ranks", static_cast<std::int32_t>(ranks_));
    visit("global_cells", domain_.global_cells);
    visit("local_cells", domain_.local_cells);
    visit("global_offset", domain_.global_offset);
    visit("spacing", domain_.spacing);
    visit("local_origin", domain_.origin);
    visit("lengths", domain_.lengths);
    visit("particle_species_count", static_cast<std::uint64_t>(snapshot.particles.size()));
  }

  template<class Visitor>
  void visit_species(data::OutputSnapshotView snapshot, Visitor&& visit) const {
    for (const auto& species : snapshot.particles)
      visit(species_layouts_.at(species.species), static_cast<std::uint64_t>(species.id.size()));
  }

  template<class Visitor>
  void visit_datasets(data::OutputSnapshotView snapshot, Visitor&& visit) const {
    for (std::size_t i = 0; i < snapshot.arrays.size(); ++i) {
      if (snapshot.arrays[i].values.empty()) continue;
      const auto& layout = field_layouts_[i];
      visit(layout.path, std::span<const std::size_t>(layout.dimensions),
            std::span<const double>(snapshot.arrays[i].values));
    }
  }

  // Defines each particle array at its full shape, then writes it in chunks of
  // particle_chunk_records records: ids as stored, the other arrays converted
  // into the reusable scratch. define(path, shape, std::type_identity<T>)
  // returns the adapter's dataset; write(dataset, first record, values).
  template<class Define, class Write>
  void write_particle_datasets(data::OutputSnapshotView snapshot, Define&& define, Write&& write) {
    for (const auto& records : snapshot.particles) {
      const auto& layout = species_layouts_.at(records.species);
      const std::size_t count = records.id.size();
      std::size_t index = 0;
      visit_particle_array_descriptors([&](const ParticleArray& array) {
        const auto components = static_cast<std::size_t>(array.components);
        const std::array dimensions{count, components};
        const auto shape = std::span<const std::size_t>(dimensions).first(components == 1 ? 1 : 2);
        const auto& path = layout.arrays[index++];
        if (array.quantity == ParticleQuantity::id) {
          auto dataset = define(path, shape, std::type_identity<std::uint64_t>{});
          for (std::size_t first = 0; first < count; first += particle_chunk_records)
            write(dataset, first, records.id.subspan(first, std::min(particle_chunk_records, count - first)));
          return;
        }
        auto dataset = define(path, shape, std::type_identity<double>{});
        scratch_.resize(3 * particle_chunk_records);
        for (std::size_t first = 0; first < count; first += particle_chunk_records) {
          const auto values = std::span<double>(scratch_).first(
              std::min(particle_chunk_records, count - first) * components);
          convert_particles(array, records, geometry_, first, values);
          write(dataset, first, std::span<const double>(values));
        }
      });
    }
  }

  // Records the epoch and the piece and manifest paths of a new piece.
  void begin_piece(data::OutputSnapshotView snapshot);
  const config::PreparedDomain domain_;
  std::array<std::uint64_t, 3> nodes_{};
  int rank_{}, ranks_{};
  std::vector<FieldLayout> field_layouts_;
  std::vector<SpeciesLayout> species_layouts_;
  ParticleGeometry geometry_;
  std::vector<double> scratch_;
  std::string directory_, precision_name_, precision_roles_, piece_, manifest_;
  std::string backend_, extension_, channel_;
  int cycle_{};
  double time_{};
  bool overwrite_{}, final_{}, fields_{};
};

} // namespace tpic::io::detail
#endif // TPIC_IO_DETAIL_DATASET_ENGINE_HPP
