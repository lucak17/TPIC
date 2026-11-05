#include <lpfx/distributed/runtime/error_reporting.hpp>
#include <lpfx/distributed/runtime/mpi_file_writer.hpp>
#include <tpic/io/output_engine.hpp>
#include <tpic/io/detail/output_file.hpp>
#include <tpic/io/detail/vtk_output.hpp>
#include <tpic/config/precision.hpp>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace tpic::io {
namespace {

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

// Host-only Float64 VTK XML ImageData, one file per array and epoch, written by
// every rank together: one piece over the global node grid in x-fastest order
// with vector components interleaved per node. A rank writes the lower node of
// each of its cells and, where its box reaches the upper end of an axis, the
// last node too, so every node is written once. A cell array is CellData over
// the global cells, each rank writing its own. An absent array writes no file.
// Unit conversions belong to the capture.
class VtkFieldEngine final : public OutputEngine {
public:
  VtkFieldEngine(const config::PreparedOutputSelection& selection,
                 const config::PreparedDomain& domain, config::PrecisionMetadata precision,
                 lpfx::distributed::MpiFileWriter& files)
      : directory_(selection.output().save_dir), precision_(std::move(precision)), files_(files),
        creation_(selection.output().overwrite_output ? lpfx::distributed::MpiFileCreation::replace
                                                      : lpfx::distributed::MpiFileCreation::exclusive) {
    const auto global = domain.field_neighborhood.global_cells();
    const auto box = domain.field_neighborhood.local_partition().global_cell_box();
    // Slowest axis first; the closed local node box is the snapshot's array.
    const std::array<std::uint64_t, 3> cells{global.z(), global.y(), global.x()},
        lower{static_cast<std::uint64_t>(box.lower().z()), static_cast<std::uint64_t>(box.lower().y()),
              static_cast<std::uint64_t>(box.lower().x())},
        upper{static_cast<std::uint64_t>(box.upper().z()), static_cast<std::uint64_t>(box.upper().y()),
              static_cast<std::uint64_t>(box.upper().x())};
    for (std::size_t axis = 0; axis != cells.size(); ++axis) {
      block_.file_shape[axis] = cells[axis] + 1;
      block_.file_offset[axis] = cell_block_.file_offset[axis] = lower[axis];
      block_.extent[axis] = upper[axis] - lower[axis] + (upper[axis] == cells[axis] ? 1 : 0);
      block_.memory_shape[axis] = upper[axis] - lower[axis] + 1;
      cell_block_.file_shape[axis] = cells[axis];
      cell_block_.extent[axis] = cell_block_.memory_shape[axis] = upper[axis] - lower[axis];
    }
    whole_extent_ = "0 " + std::to_string(cells[2]) + " 0 " + std::to_string(cells[1]) + " 0 " +
                    std::to_string(cells[0]);
    spacing_ = domain.spacing;
  }

  void write_piece(data::OutputSnapshotView snapshot) override {
    const std::filesystem::path directory(directory_);
    bool named = false;
    for (const auto& array : snapshot.arrays) {
      if (array.values.empty()) continue;
      auto& block = array.cell ? cell_block_ : block_;
      auto xml = detail::vtk_xml_stream();
      detail::vtk_file_open(xml, "ImageData");
      xml << "  <ImageData WholeExtent=\"" << whole_extent_ << "\" Origin=\"0 0 0\" Spacing=\""
          << spacing_[0] << ' ' << spacing_[1] << ' ' << spacing_[2] << "\">\n";
      detail::vtk_field_data(xml, snapshot.cycle, snapshot.time, snapshot.final_state, precision_);
      xml << "    <Piece Extent=\"" << whole_extent_ << "\">\n"
          << (array.cell ? "      <PointData/>\n      <CellData>\n" : "      <PointData>\n");
      detail::vtk_data_array(xml, "Float64", array.name, array.components, 0);
      xml << (array.cell ? "      </CellData>\n" : "      </PointData>\n      <CellData/>\n")
          << "    </Piece>\n  </ImageData>\n" << detail::vtk_appended_data_open;
      const std::uint64_t bytes = block.file_shape[0] * block.file_shape[1] * block.file_shape[2] *
                                  static_cast<std::uint64_t>(array.components) * sizeof(double);
      auto head = std::move(xml).str();
      head.append(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
      block.width = static_cast<std::uint64_t>(array.components);
      const auto path = (directory / (detail::output_stem(array.name, snapshot.cycle) + ".vti")).string();
      const auto written = files_.host_write_array<double, 3>(
          path.c_str(), creation_, std::as_bytes(std::span(head)),
          std::as_bytes(std::span(detail::vtk_appended_data_close)), block, std::span(array.values));
      if (written.native_result() == MPI_ERR_FILE_EXISTS)
        detail::output_error("field output", "create", path, EEXIST);
      lpfx::check_mpi_error(written, ("TPIC field output: write " + path).c_str());
      if (!std::exchange(named, true)) manifest_ = path;
    }
  }

  void publish() override {}

  const std::string& manifest_path() const noexcept override { return manifest_; }

private:
  std::string directory_;
  std::string manifest_;
  config::PrecisionMetadata precision_;
  lpfx::distributed::MpiFileWriter& files_;
  lpfx::distributed::MpiFileCreation creation_;
  lpfx::distributed::MpiArrayBlock<3> block_{}, cell_block_{};
  std::string whole_extent_;
  std::array<double, 3> spacing_{};
};

} // namespace

std::unique_ptr<OutputEngine> make_vtk_field_engine(
    const config::PreparedOutputSelection& selection, const config::PreparedDomain& domain,
    config::PrecisionMetadata precision, lpfx::distributed::MpiFileWriter& field_files) {
  return std::make_unique<VtkFieldEngine>(selection, domain, std::move(precision), field_files);
}

} // namespace tpic::io
