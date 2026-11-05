#include <tpic/io/detail/dataset_engine.hpp>
#include <tpic/io/detail/output_file.hpp>

#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace tpic::io::detail {
namespace fs = std::filesystem;
namespace {
void check_destination(const fs::path& destination, bool overwrite) {
  std::error_code error;
  const auto status = fs::symlink_status(destination, error);
  if (error && error != std::errc::no_such_file_or_directory)
    throw std::system_error(error, "TPIC dataset output: inspect " + destination.string());
  if (fs::is_symlink(status))
    throw std::runtime_error("TPIC dataset output: refusing output symlink: " + destination.string());
  if (fs::exists(status) && !overwrite)
    output_error("dataset output", "create", destination.string(), EEXIST);
  if (fs::exists(status) && !fs::is_regular_file(status) && !fs::is_directory(status))
    throw std::runtime_error("TPIC dataset output: refusing special output file: " + destination.string());
}
} // namespace

StagedOutput::StagedOutput(fs::path destination, bool overwrite)
    : destination_(std::move(destination)), overwrite_(overwrite) {
  check_destination(destination_, overwrite_);
  auto pattern = (destination_.parent_path() / ".tpic-output-XXXXXX").string();
  if (!::mkdtemp(pattern.data()))
    output_error("dataset output", "create temporary directory", pattern, errno);
  directory_ = pattern;
  temporary_ = (directory_ / destination_.filename()).string();
}

StagedOutput::~StagedOutput() noexcept {
  std::error_code ignored;
  fs::remove_all(directory_, ignored);
}

void StagedOutput::commit() {
  check_destination(destination_, overwrite_);
  if (fs::exists(destination_) &&
      fs::is_directory(destination_) != fs::is_directory(temporary_))
    throw std::runtime_error("TPIC dataset output: refusing to replace an output with a different file type: " +
                             destination_.string());
#if defined(__linux__) && defined(SYS_renameat2)
  // RENAME_NOREPLACE keeps default preservation atomic for both files and BP
  // directories. RENAME_EXCHANGE replaces a prior BP directory atomically;
  // cleanup subsequently removes that old output from the private directory.
  constexpr unsigned no_replace = 1, exchange = 2;
  if (::syscall(SYS_renameat2, AT_FDCWD, temporary_.c_str(), AT_FDCWD,
                destination_.c_str(), no_replace) == 0) return;
  if (errno != EEXIST || !overwrite_)
    output_error("dataset output", "commit", destination_.string(), errno);
  check_destination(destination_, true);
  if (::syscall(SYS_renameat2, AT_FDCWD, temporary_.c_str(), AT_FDCWD,
                destination_.c_str(), exchange) != 0)
    output_error("dataset output", "replace", destination_.string(), errno);
#else
  // POSIX hard links provide exclusive publication for regular files. A BP
  // directory instead reserves its final name exclusively before rename.
  if (!overwrite_) {
    if (fs::is_directory(temporary_)) {
      if (!fs::create_directory(destination_))
        output_error("dataset output", "reserve", destination_.string(), EEXIST);
      fs::rename(temporary_, destination_);
    } else {
      fs::create_hard_link(temporary_, destination_);
    }
  } else {
    if (fs::exists(destination_)) fs::rename(destination_, directory_ / "previous");
    fs::rename(temporary_, destination_);
  }
#endif
}

DatasetOutputEngine::DatasetOutputEngine(data::OutputChannel channel,
    const config::PreparedOutputSelection& selection,
    const config::PreparedDomain& domain, config::PrecisionMetadata precision,
    std::string backend, std::string extension)
    : domain_(domain), directory_(selection.output().save_dir),
      precision_name_(precision.name), precision_roles_(precision.roles),
      backend_(std::move(backend)), extension_(std::move(extension)),
      overwrite_(selection.output().overwrite_output),
      fields_(channel == data::OutputChannel::fields) {
  channel_ = fields_ ? "fields" : "particles";
  const auto& topology = domain.field_neighborhood.topology().input();
  rank_ = static_cast<int>(topology.rank());
  ranks_ = static_cast<int>(topology.communicator_size());
  for (std::size_t axis = 0; axis < 3; ++axis)
    nodes_[axis] = static_cast<std::uint64_t>(domain.local_cells[axis]) + 1;
  if (fields_) {
    field_layouts_.reserve(selection.arrays().size());
    for (const auto& field : selection.arrays()) {
      const std::size_t cell = field.cell;
      field_layouts_.push_back({"fields/" + field.name,
          {nodes_[2] - cell, nodes_[1] - cell, nodes_[0] - cell, static_cast<std::size_t>(field.components)}});
    }
  } else {
    geometry_ = prepare_particle_geometry(domain);
    species_layouts_.reserve(selection.species_count());
    for (std::size_t species = 0; species < selection.species_count(); ++species) {
      SpeciesLayout layout;
      layout.path = "particles/species_" + std::to_string(species);
      layout.count_attribute = layout.path + "/count";
      std::size_t index = 0;
      visit_particle_array_descriptors([&](const auto& array) {
        layout.arrays[index++] = layout.path + "/" + array.name;
      });
      species_layouts_.push_back(std::move(layout));
    }
  }
}

void DatasetOutputEngine::begin_piece(data::OutputSnapshotView snapshot) {
  cycle_ = snapshot.cycle;
  time_ = snapshot.time;
  final_ = snapshot.final_state;
  piece_ = (fs::path(directory_) / output_piece_name(channel_, cycle_, rank_, extension_)).string();
  manifest_ = (fs::path(directory_) /
               (output_stem(channel_, cycle_) + "." + backend_ + ".json")).string();
}

void DatasetOutputEngine::publish() {
  std::ostringstream json;
  json.imbue(std::locale::classic());
  json << std::setprecision(std::numeric_limits<double>::max_digits10)
       << "{\n  \"schema\": \"tpic-output-v1\",\n  \"backend\": \"" << backend_
       << "\",\n  \"channel\": \"" << channel_
       << "\",\n  \"cycle\": " << cycle_ << ",\n  \"time\": " << time_
       << ",\n  \"final_state\": " << (final_ ? "true" : "false")
       << ",\n  \"fields_present\": " << (fields_ ? "true" : "false")
       << ",\n  \"particles_present\": " << (fields_ ? "false" : "true")
       << ",\n  \"pieces\": [";
  for (int rank = 0; rank < ranks_; ++rank) {
    if (rank != 0) json << ',';
    json << "\n    {\"rank\": " << rank << ", \"file\": \""
         << output_piece_name(channel_, cycle_, rank, extension_) << "\"}";
  }
  json << "\n  ]\n}\n";
  StagedOutput staged(manifest_, overwrite_);
  write_text_file(staged.temporary_path(), false, "dataset output manifest", json.view());
  staged.commit();
}

} // namespace tpic::io::detail
