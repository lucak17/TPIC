#include <tpic/io/output_engine.hpp>
#include <tpic/io/detail/output_file.hpp>
#include <tpic/io/detail/output_layout.hpp>
#include <tpic/io/detail/vtk_output.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tpic::io {
namespace {

constexpr std::string_view particle_type(const detail::ParticleArray& array) {
  return array.quantity == detail::ParticleQuantity::id ? "UInt64" : "Float64";
}

// The particle adapter streams the species consecutively into one rank piece,
// converting the stored records chunk by chunk, without a second particle
// image. Species is a point attribute; individual particles are VTK vertices
// with exact integer IDs.
class VtkParticleEngine final : public OutputEngine {
public:
  VtkParticleEngine(const config::PreparedOutputSelection& selection,
                    const config::PreparedDomain& domain, config::PrecisionMetadata precision)
      : geometry_(detail::prepare_particle_geometry(domain)),
        directory_(selection.output().save_dir), precision_(precision),
        overwrite_(selection.output().overwrite_output) {
    const auto& topology = domain.field_neighborhood.topology().input();
    rank_ = static_cast<int>(topology.rank());
    ranks_ = static_cast<int>(topology.communicator_size());
  }

  void write_piece(data::OutputSnapshotView snapshot) override {
    cycle_ = snapshot.cycle;
    time_ = snapshot.time;
    final_ = snapshot.final_state;
    write_particles(snapshot);
    manifest_ = (std::filesystem::path(directory_) /
                 (detail::output_stem("particles", cycle_) + ".pvtp")).string();
  }

  void publish() override {
    auto xml = detail::vtk_xml_stream();
    detail::vtk_file_open(xml, "PPolyData");
    xml << "  <PPolyData GhostLevel=\"0\">\n";
    detail::vtk_field_data(xml, cycle_, time_, final_, precision_);
    const auto particle_array = [&](const detail::ParticleArray& descriptor) {
      detail::vtk_parallel_array(xml, particle_type(descriptor), descriptor.name, descriptor.components);
    };
    xml << "    <PPointData>\n";
    particle_array(detail::particle_velocity);
    particle_array(detail::particle_charge);
    particle_array(detail::particle_id);
    detail::vtk_parallel_array(xml, "UInt64", "species", 1);
    xml << "    </PPointData>\n    <PCellData/>\n    <PPoints>\n";
    particle_array(detail::particle_position);
    xml << "    </PPoints>\n";
    for (int rank = 0; rank < ranks_; ++rank)
      xml << "    <Piece Source=\""
          << detail::output_piece_name("particles", cycle_, rank, ".vtp") << "\"/>\n";
    xml << "  </PPolyData>\n</VTKFile>\n";
    detail::write_text_file(manifest_, overwrite_, "particle output", xml.view());
  }

  const std::string& manifest_path() const noexcept override { return manifest_; }

private:
  void write_particles(data::OutputSnapshotView snapshot) {
    std::uint64_t count = 0;
    for (const auto& records : snapshot.particles) {
      if (records.id.size() > std::numeric_limits<std::uint64_t>::max() - count)
        throw std::overflow_error("TPIC particle output: point count exceeds UInt64");
      count += records.id.size();
    }
    // Four snapshot arrays plus generated species, connectivity and offsets.
    constexpr std::uint64_t stride = (detail::particle_position.components +
        detail::particle_velocity.components + detail::particle_charge.components +
        detail::particle_id.components + 3) * sizeof(std::uint64_t);
    if (count > (std::numeric_limits<std::uint64_t>::max() - 7 * sizeof(std::uint64_t)) / stride ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      throw std::overflow_error("TPIC particle output: appended offsets exceed UInt64");
    const auto scalar_bytes = count * sizeof(std::uint64_t);
    auto xml = detail::vtk_xml_stream();
    detail::vtk_file_open(xml, "PolyData");
    xml << "  <PolyData>\n";
    detail::vtk_field_data(xml, cycle_, time_, final_, precision_);
    xml << "    <Piece NumberOfPoints=\"" << count << "\" NumberOfVerts=\"" << count
        << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n"
        << "      <PointData>\n";
    std::uint64_t offset = 0;
    const auto array = [&](std::string_view type, std::string_view name, int components) {
      detail::vtk_data_array(xml, type, name, components, offset);
      offset += sizeof(std::uint64_t) + scalar_bytes * components;
    };
    const auto particle_array = [&](const detail::ParticleArray& descriptor) {
      array(particle_type(descriptor), descriptor.name, descriptor.components);
    };
    particle_array(detail::particle_velocity);
    particle_array(detail::particle_charge);
    particle_array(detail::particle_id);
    array("UInt64", "species", 1);
    xml << "      </PointData>\n      <CellData/>\n      <Points>\n";
    particle_array(detail::particle_position);
    xml << "      </Points>\n      <Verts>\n";
    array("Int64", "connectivity", 1);
    array("Int64", "offsets", 1);
    xml << "      </Verts>\n    </Piece>\n  </PolyData>\n"
        << detail::vtk_appended_data_open;
    detail::OutputFile file((std::filesystem::path(directory_) /
        detail::output_piece_name("particles", cycle_, rank_, ".vtp")).string(),
        overwrite_, "particle output");
    file.write(xml.view());
    const auto species_array = [&](const detail::ParticleArray& descriptor) {
      file.value(scalar_bytes * descriptor.components);
      for (const auto& records : snapshot.particles) {
        if (descriptor.quantity == detail::ParticleQuantity::id) {
          file.write(records.id.data(), records.id.size_bytes());
          continue;
        }
        values_.resize(3 * detail::particle_chunk_records);
        for (std::size_t first = 0; first < records.id.size(); first += detail::particle_chunk_records) {
          const auto chunk = std::span<double>(values_).first(
              std::min(detail::particle_chunk_records, records.id.size() - first) *
              static_cast<std::size_t>(descriptor.components));
          detail::convert_particles(descriptor, records, geometry_, first, chunk);
          file.write(chunk.data(), chunk.size_bytes());
        }
      }
    };
    species_array(detail::particle_velocity);
    species_array(detail::particle_charge);
    species_array(detail::particle_id);
    file.value(scalar_bytes);
    auto& scratch = scratch_;
    for (const auto& records : snapshot.particles) {
      std::fill_n(scratch.begin(), std::min(scratch.size(), records.id.size()),
                  static_cast<std::uint64_t>(records.species));
      for (std::size_t start = 0; start < records.id.size(); start += scratch.size())
        file.write(scratch.data(), std::min(scratch.size(), records.id.size() - start) * sizeof(scratch[0]));
    }
    species_array(detail::particle_position);
    for (std::uint64_t first : {std::uint64_t{0}, std::uint64_t{1}}) {
      file.value(scalar_bytes);
      for (std::uint64_t start = 0; start < count; start += scratch.size()) {
        const auto size = std::min<std::uint64_t>(scratch.size(), count - start);
        for (std::uint64_t i = 0; i < size; ++i) scratch[i] = start + i + first;
        file.write(scratch.data(), size * sizeof(scratch[0]));
      }
    }
    file.write(detail::vtk_appended_data_close);
    file.close();
  }

  detail::ParticleGeometry geometry_;
  // Bounded generated-array workspace (species, connectivity and offsets) and
  // conversion chunk, reused between snapshots.
  std::array<std::uint64_t, 4096> scratch_{};
  std::vector<double> values_;
  std::string directory_;
  std::string manifest_;
  config::PrecisionMetadata precision_;
  int rank_ = 0, ranks_ = 1, cycle_ = -1;
  double time_ = 0;
  bool overwrite_ = false;
  bool final_ = false;
};

} // namespace

std::unique_ptr<OutputEngine> make_vtk_particle_engine(
    const config::PreparedOutputSelection& selection,
    const config::PreparedDomain& domain, config::PrecisionMetadata precision) {
  return std::make_unique<VtkParticleEngine>(selection, domain, precision);
}

} // namespace tpic::io
