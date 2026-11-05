#include <tpic/io/checkpoint.hpp>

#include <tpic/config/boundary_conditions.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/io/detail/checkpoint_engine.hpp>
#include <tpic/io/detail/output_file.hpp>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace tpic::io {
namespace fs = std::filesystem;
namespace {

constexpr std::string_view checkpoint_channel = "checkpoint";
constexpr std::string_view set_aside_prefix = "abandoned_";

// A checkpoint directory of a listing: checkpoint_t<cycle>_n<ranks>.
struct Listed {
  std::string name;
  std::uint64_t cycle{}, ranks{};
};

bool parse_checkpoint_name(const std::string& name, Listed& listed) {
  const auto prefix = std::string(checkpoint_channel) + "_t";
  const auto separator = name.find("_n", prefix.size());
  if (!name.starts_with(prefix) || separator == std::string::npos) return false;
  const auto number = [&](std::size_t first, std::size_t last, std::uint64_t& value) {
    const auto [end, error] = std::from_chars(name.data() + first, name.data() + last, value);
    return first != last && error == std::errc{} && end == name.data() + last;
  };
  listed.name = name;
  return number(prefix.size(), separator, listed.cycle) && number(separator + 2, name.size(), listed.ranks) &&
         listed.ranks != 0;
}

// The checkpoint directories of directory, newest first. A missing directory
// has none; any other listing error throws.
std::vector<Listed> list_checkpoints(const fs::path& directory) {
  std::vector<Listed> found;
  std::error_code error;
  for (fs::directory_iterator entry(directory, error), end; !error && entry != end; entry.increment(error)) {
    Listed listed;
    if (entry->is_directory() && parse_checkpoint_name(entry->path().filename().string(), listed))
      found.push_back(std::move(listed));
  }
  if (error && error != std::errc::no_such_file_or_directory)
    throw std::system_error(error, "TPIC checkpoint: cannot list " + directory.string());
  std::sort(found.begin(), found.end(), [](const Listed& a, const Listed& b) { return a.cycle > b.cycle; });
  return found;
}

// The piece extension of a complete checkpoint, or empty: complete means the
// final names of all its ranks' pieces, one extension, are listed.
std::string complete_extension(const fs::path& directory, const Listed& listed) {
  for (const std::string_view extension : {".bp", ".h5"}) {
    bool complete = true;
    for (std::uint64_t rank = 0; complete && rank < listed.ranks; ++rank) {
      std::error_code error;
      complete = fs::exists(directory / listed.name /
                                detail::output_piece_name(checkpoint_channel, listed.cycle, rank, extension),
                            error);
      if (error) throw std::system_error(error, "TPIC checkpoint: cannot list " + listed.name);
    }
    if (complete) return std::string(extension);
  }
  return {};
}

// Renames a checkpoint directory to abandoned_<name>; the root removes it at
// its next checkpoint.
void set_aside(const fs::path& directory, const std::string& name) {
  const auto target = directory / (std::string(set_aside_prefix) + name);
  fs::remove_all(target);
  fs::rename(directory / name, target);
}

// Root, at the start of its piece, once directory holds it: removes every
// checkpoint older than the newest complete one, and the set-aside ones. A
// rank writes its pieces in order, so nothing a slower rank still writes is
// touched.
void remove_old_checkpoints(const fs::path& directory) noexcept {
  try {
    const auto found = list_checkpoints(directory);
    const auto newest = std::find_if(found.begin(), found.end(), [&](const Listed& listed) {
      return !complete_extension(directory, listed).empty();
    });
    if (newest != found.end())
      for (const auto& old : found)
        if (old.cycle < newest->cycle) fs::remove_all(directory / old.name);
    for (const auto& entry : fs::directory_iterator(directory))
      if (entry.is_directory() && entry.path().filename().string().starts_with(set_aside_prefix))
        fs::remove_all(entry.path());
  } catch (const std::exception& error) {
    std::cerr << "TPIC checkpoint: warning: old checkpoints were not removed: " << error.what() << '\n';
  }
}

// Syncs one file or directory to stable storage.
void sync_path(const fs::path& path, bool directory) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
  if (fd < 0) detail::output_error("checkpoint", "open for sync", path.string(), errno);
  const int synced = ::fsync(fd), error = errno;
  ::close(fd);
  if (synced != 0) detail::output_error("checkpoint", "sync", path.string(), error);
}

std::string piece_path(const data::CheckpointLocation& checkpoint, int rank) {
  return (fs::path(checkpoint.directory) /
          detail::output_piece_name(checkpoint_channel, static_cast<std::uint64_t>(checkpoint.header.cycle),
                                    static_cast<std::uint64_t>(rank), checkpoint.extension)).string();
}

std::unique_ptr<detail::CheckpointReader> open_checkpoint(const std::string& path) {
#if TPIC_HAS_ADIOS2
  if (path.ends_with(".bp")) return detail::open_adios2_checkpoint(path);
#endif
#if TPIC_HAS_HDF5
  if (path.ends_with(".h5")) return detail::open_hdf5_checkpoint(path);
#endif
  throw std::invalid_argument("TPIC restart: no reader for checkpoint piece " + path +
                              " in this build (TPIC_ENABLE_ADIOS2 or TPIC_ENABLE_HDF5)");
}

// The domain of writer rank `rank` of a checkpoint: the partition is a pure
// function of the mesh and the process grid; the field periodicity enters the
// admission of the partition.
config::PreparedDomain piece_domain(const data::CheckpointHeader& header, int rank) {
  const config::GridConfig grid{header.lengths[0], header.lengths[1], header.lengths[2],
                                header.global_cells[0], header.global_cells[1], header.global_cells[2]};
  constexpr auto periodic = static_cast<std::int32_t>(config::FieldBoundary::periodic);
  config::TopologyConfig topology;
  topology.xlen = header.processes[0];
  topology.ylen = header.processes[1];
  topology.zlen = header.processes[2];
  topology.periodic_x = header.faces[0] == periodic;
  topology.periodic_y = header.faces[2] == periodic;
  topology.periodic_z = header.faces[4] == periodic;
  return config::prepare_domain(grid, topology, rank,
                                header.processes[0] * header.processes[1] * header.processes[2]);
}

// The overlap of a piece's cells with this rank's, per axis [first, last) in
// the piece's local cells; empty on some axis when they do not overlap.
struct Overlap {
  std::array<int, 3> first{}, last{};
  [[nodiscard]] bool empty() const noexcept {
    return first[0] >= last[0] || first[1] >= last[1] || first[2] >= last[2];
  }
};
Overlap overlap(const data::CheckpointSource& source, const config::PreparedDomain& domain) {
  Overlap result;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    result.first[axis] = std::max(source.offset[axis], domain.global_offset[axis]) - source.offset[axis];
    result.last[axis] = std::min(source.offset[axis] + source.cells[axis],
                                 domain.global_offset[axis] + domain.local_cells[axis]) - source.offset[axis];
  }
  return result;
}

// The records of species s of one piece, whose cell keys old_map decodes, that
// lie in the cells of domain; returns their number. With staging, they are read
// chunk by chunk into it, with their new cell keys, and delivered. The sorted
// prefix is read by its index in exact spans; the tail is read whole and
// filtered by its keys.
std::uint64_t restore_records(detail::CheckpointReader& reader, const data::ParticleCellMap& old_map,
                              const data::CheckpointSource& source, const config::PreparedDomain& domain,
                              std::size_t s, const ParticleStaging* staging,
                              const std::function<void(std::size_t)>& deliver) {
  const auto [count, sorted_prefix] = reader.records(s);
  const auto box = overlap(source, domain);
  if (count == 0 || box.empty()) return 0;
  const auto path = detail::checkpoint_species_path(s) + "/";
  const auto new_map = data::prepare_particle_cell_map(domain);
  const std::size_t capacity = staging ? staging->id.size() : restart_staging_records;
  // The new local cell of an old one, and its key.
  const auto key = [&](int i, int j, int k) {
    return new_map.encode(source.offset[0] + i - domain.global_offset[0],
                          source.offset[1] + j - domain.global_offset[1],
                          source.offset[2] + k - domain.global_offset[2]);
  };
  std::uint64_t restored = 0;
  std::size_t staged = 0;
  // Merged reads: piece records [first, first + count) to staging offset.
  struct Span {
    std::uint64_t first;
    std::size_t count, offset;
  };
  std::vector<Span> spans;
  std::vector<detail::CheckpointRead> reads;
  const auto read_records = [&](std::uint64_t first, std::size_t count, std::size_t offset) {
    detail::visit_checkpoint_records(*staging, [&](const char* name, auto values) {
      reads.push_back({path + name, {first}, {count}, 1, values.subspan(offset, count)});
    });
  };
  const auto flush = [&] {
    if (staged == 0) return;
    reads.clear();
    for (const auto& span : spans) read_records(span.first, span.count, span.offset);
    reader.read(reads);
    deliver(staged);
    staged = 0;
    spans.clear();
  };

  // The index sub-box: rows (k, j) of the overlap, x from first to last inclusive.
  const auto width = static_cast<std::size_t>(box.last[0] - box.first[0] + 1);
  const auto rows_y = static_cast<std::size_t>(box.last[1] - box.first[1]);
  const auto rows_z = static_cast<std::size_t>(box.last[2] - box.first[2]);
  std::vector<std::uint32_t> index(rows_z * rows_y * width);
  const std::array<detail::CheckpointRead, 1> index_read{{{path + "prefix_start",
      {std::size_t(box.first[2]), std::size_t(box.first[1]), std::size_t(box.first[0])},
      {rows_z, rows_y, width}, 3, std::span(index)}}};
  reader.read(index_read);
  for (std::size_t row = 0; row < rows_z * rows_y; ++row) {
    const auto* starts = index.data() + row * width;
    if (starts[width - 1] < starts[0] || starts[width - 1] > sorted_prefix)
      throw std::runtime_error("TPIC restart: invalid particle index in checkpoint");
    if (!staging) {
      restored += starts[width - 1] - starts[0];
      continue;
    }
    const int j = box.first[1] + int(row % rows_y), k = box.first[2] + int(row / rows_y);
    for (std::size_t x = 0; x + 1 < width; ++x) {
      std::uint64_t first = starts[x];
      const std::uint64_t last = starts[x + 1];
      const auto cell = key(box.first[0] + int(x), j, k);
      while (first < last) {
        const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(last - first, capacity - staged));
        if (!spans.empty() && spans.back().first + spans.back().count == first)
          spans.back().count += n;
        else
          spans.push_back({first, n, staged});
        std::fill_n(staging->cell.begin() + staged, n, cell);
        staged += n;
        first += n;
        restored += n;
        if (staged == capacity) flush();
      }
    }
  }
  flush();

  std::vector<std::int32_t> keys(staging ? 0 : capacity);
  for (std::uint64_t first = sorted_prefix; first < count; first += capacity) {
    const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(count - first, capacity));
    const auto tail = staging ? staging->cell.first(n) : std::span<std::int32_t>(keys).first(n);
    reads.clear();
    reads.push_back({path + "tail_cell", {std::size_t(first - sorted_prefix)}, {n}, 1, tail});
    if (staging) read_records(first, n, 0);
    reader.read(reads);
    std::size_t kept = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (tail[i] < 0 || tail[i] >= old_map.count)
        throw std::runtime_error("TPIC restart: invalid particle cell key in checkpoint");
      const auto cell = old_map.decode(tail[i]);
      if (!old_map.contains_interior(cell.i, cell.j, cell.k))
        throw std::runtime_error("TPIC restart: particle outside its piece in checkpoint");
      if (cell.i < box.first[0] || cell.i >= box.last[0] || cell.j < box.first[1] || cell.j >= box.last[1] ||
          cell.k < box.first[2] || cell.k >= box.last[2])
        continue;
      if (staging) {
        if (kept != i)
          detail::visit_checkpoint_records(*staging, [&](const char*, auto values) { values[kept] = values[i]; });
        tail[kept] = key(cell.i, cell.j, cell.k);
      }
      ++kept;
    }
    restored += kept;
    if (staging && kept != 0) deliver(kept);
  }
  return restored;
}

} // namespace

namespace detail {

CheckpointEngine::CheckpointEngine(const config::PreparedOutput& output, const config::PreparedDomain& domain,
                                   std::string extension)
    : cells_{std::size_t(domain.local_cells[2]), std::size_t(domain.local_cells[1]),
             std::size_t(domain.local_cells[0])},
      directory_(output.checkpoint_dir), extension_(std::move(extension)),
      map_(data::prepare_particle_cell_map(domain)),
      rank_(static_cast<int>(domain.field_neighborhood.topology().input().rank())),
      ranks_(static_cast<int>(domain.field_neighborhood.topology().input().communicator_size())) {}

// The longest prefix of non-decreasing keys and the start of every cell in it.
// The key is monotonic in (k, j, i), so key order is cell order. Every key
// must name an interior cell.
void CheckpointEngine::index_species(const data::ParticleRecords& records, SpeciesIndex& index) const {
  const auto keys = records.cell;
  const auto prefix = static_cast<std::size_t>(std::is_sorted_until(keys.begin(), keys.end()) - keys.begin());
  const auto slot = [&](std::int32_t key) {
    const auto cell = key < 0 || key >= map_.count ? data::GridIndex{-1, -1, -1} : map_.decode(key);
    if (!map_.contains_interior(cell.i, cell.j, cell.k))
      throw std::invalid_argument("TPIC checkpoint: particle cell key outside the rank's cells");
    return std::size_t(cell.i) + (cells_[2] + 1) * (std::size_t(cell.j) + cells_[1] * std::size_t(cell.k));
  };
  index.sorted_prefix = prefix;
  index.prefix_start.resize(cells_[0] * cells_[1] * (cells_[2] + 1));
  std::size_t next = 0;
  for (std::size_t record = 0; record < prefix; ++record)
    if (record == 0 || keys[record] != keys[record - 1])
      for (const auto last = slot(keys[record]); next <= last; ++next)
        index.prefix_start[next] = static_cast<std::uint32_t>(record);
  std::fill(index.prefix_start.begin() + next, index.prefix_start.end(), static_cast<std::uint32_t>(prefix));
  for (std::size_t record = prefix; record < keys.size(); ++record) (void)slot(keys[record]);
}

void CheckpointEngine::write_piece(data::OutputSnapshotView snapshot) {
  const fs::path directory(directory_);
  const auto cycle = static_cast<std::uint64_t>(snapshot.cycle);
  const auto checkpoint = directory / (output_stem(checkpoint_channel, cycle) + "_n" +
                                       output_padded(static_cast<std::uint64_t>(ranks_)));
  fs::create_directories(checkpoint);
  sync_path(directory, true);
  if (rank_ == 0) remove_old_checkpoints(directory);
  index_.resize(snapshot.particles.size());
  for (std::size_t s = 0; s < snapshot.particles.size(); ++s) index_species(snapshot.particles[s], index_[s]);
  // A BP5 piece is a directory of files.
  const auto name = output_piece_name(checkpoint_channel, cycle, static_cast<std::uint64_t>(rank_), extension_);
  const auto part = checkpoint / ("." + name);
  fs::remove_all(part);
  write_file(part.string(), snapshot);
  const bool directory_piece = fs::is_directory(part);
  if (directory_piece)
    for (const auto& entry : fs::directory_iterator(part)) sync_path(entry.path(), false);
  sync_path(part, directory_piece);
  fs::rename(part, checkpoint / name);
  sync_path(checkpoint, true);
  checkpoint_ = checkpoint.string();
}

} // namespace detail

std::unique_ptr<OutputEngine> make_checkpoint_engine(const config::PreparedOutput& output,
                                                     const config::PreparedDomain& domain) {
#if TPIC_HAS_ADIOS2
  if (output.checkpoint_engine == config::OutputEngineKind::Adios2)
    return detail::make_adios2_checkpoint_engine(output, domain);
#endif
#if TPIC_HAS_HDF5
  if (output.checkpoint_engine == config::OutputEngineKind::Hdf5)
    return detail::make_hdf5_checkpoint_engine(output, domain);
#endif
  throw std::invalid_argument("TPIC checkpoint: RestartWriteMethod '" +
                              std::string(config::output_engine_name(output.checkpoint_engine)) +
                              "' is not a checkpoint backend of this build (adios2 or hdf5)");
}

data::CheckpointLocation find_restart_checkpoint(const std::string& directory) {
  for (const auto& listed : list_checkpoints(directory)) {
    const auto extension = complete_extension(directory, listed);
    if (extension.empty()) continue;
    const auto path = fs::path(directory) / listed.name;
    data::CheckpointLocation checkpoint{path.string(), extension, open_checkpoint(
        (path / detail::output_piece_name(checkpoint_channel, listed.cycle, 0, extension)).string())->header()};
    const auto& header = checkpoint.header;
    if (std::uint64_t(header.cycle) != listed.cycle ||
        std::uint64_t(header.processes[0]) * header.processes[1] * header.processes[2] != listed.ranks)
      throw std::runtime_error("TPIC restart: checkpoint " + listed.name + " disagrees with its header");
    return checkpoint;
  }
  throw std::runtime_error("TPIC restart: no complete checkpoint in RestartDirName '" + directory + "'");
}

void prepare_checkpoint_directory(const std::string& directory, const data::CheckpointLocation* restart,
                                  bool overwrite) {
  const auto found = list_checkpoints(directory);
  if (!restart && !found.empty() && !overwrite)
    throw std::runtime_error("TPIC checkpoint: RestartDirName '" + directory + "' holds checkpoints of an "
                             "earlier run; continue it with --restart, or set them aside with --overwrite-output");
  for (const auto& listed : found)
    if (!restart || listed.cycle > std::uint64_t(restart->header.cycle)) set_aside(directory, listed.name);
}

data::PreparedRestart prepare_restart(data::CheckpointLocation checkpoint, const config::PreparedDomain& domain) {
  data::PreparedRestart restart{std::move(checkpoint), {}, {}};
  const auto& header = restart.checkpoint.header;
  const int ranks = header.processes[0] * header.processes[1] * header.processes[2];
  for (int rank = 0; rank < ranks; ++rank) {
    const auto piece = piece_domain(header, rank);
    const data::CheckpointSource source{rank, piece.global_offset, piece.local_cells};
    if (overlap(source, domain).empty()) continue;
    restart.sources.push_back(source);
    const auto reader = open_checkpoint(piece_path(restart.checkpoint, rank));
    const auto old_map = data::prepare_particle_cell_map(piece);
    for (std::size_t s = 0; s < std::size_t(header.species); ++s)
      restart.local_count[s] += restore_records(*reader, old_map, source, domain, s, nullptr, {});
  }
  return restart;
}

void read_restart_fields(const data::PreparedRestart& restart, const config::PreparedDomain& domain,
                         const std::array<FieldDestination, data::checkpoint_field_names.size()>& fields) {
  std::tuple<std::vector<float>, std::vector<double>> buffers;
  for (const auto& source : restart.sources) {
    const auto reader = open_checkpoint(piece_path(restart.checkpoint, source.rank));
    const auto box = overlap(source, domain);
    for (std::size_t field = 0; field < fields.size(); ++field) {
      // E closes every axis with its last node; B ends with its last cell.
      const int nodes = field < data::checkpoint_node_fields ? 1 : 0;
      std::array<std::size_t, 3> start{}, count{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        start[2 - axis] = std::size_t(box.first[axis]);
        count[2 - axis] = std::size_t(box.last[axis] + nodes - box.first[axis]);
      }
      std::visit([&]<class T>(T* origin) {
        auto& buffer = std::get<std::vector<T>>(buffers);
        buffer.resize(count[0] * count[1] * count[2]);
        const std::array<detail::CheckpointRead, 1> read{{{
            std::string("fields/") + data::checkpoint_field_names[field], start, count, 3,
            std::span<T>(buffer)}}};
        reader->read(read);
        const auto stride_y = fields[field].stride_y, stride_z = fields[field].stride_z;
        auto* base = origin + std::size_t(source.offset[0] + box.first[0] - domain.global_offset[0]) +
                     std::size_t(source.offset[1] + box.first[1] - domain.global_offset[1]) * stride_y +
                     std::size_t(source.offset[2] + box.first[2] - domain.global_offset[2]) * stride_z;
        for (std::size_t z = 0; z < count[0]; ++z)
          for (std::size_t y = 0; y < count[1]; ++y)
            std::copy_n(buffer.begin() + std::ptrdiff_t((z * count[1] + y) * count[2]), count[2],
                        base + y * stride_y + z * stride_z);
      }, fields[field].origin);
    }
  }
}

void read_restart_particles(const data::PreparedRestart& restart, const config::PreparedDomain& domain,
                            std::size_t species, const ParticleStaging& staging,
                            const std::function<void(std::size_t)>& deliver) {
  for (const auto& source : restart.sources) {
    const auto reader = open_checkpoint(piece_path(restart.checkpoint, source.rank));
    const auto old_map = data::prepare_particle_cell_map(piece_domain(restart.checkpoint.header, source.rank));
    (void)restore_records(*reader, old_map, source, domain, species, &staging, deliver);
  }
}

} // namespace tpic::io
