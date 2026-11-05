#include <tpic/io/detail/checkpoint_engine.hpp>
#include <tpic/io/detail/dataset_engine.hpp>

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tpic::io {
namespace {

// Every HDF5 call made by TPIC, including handle cleanup, occurs under this
// process-local lock. This also supports HDF5 builds without thread safety.
// An output piece announces itself in hdf5_waiting before it locks.
std::mutex hdf5_mutex;
std::atomic<int> hdf5_waiting{0};

void check(herr_t result, const char* operation) {
  if (result < 0) throw std::runtime_error(std::string("TPIC HDF5 output: ") + operation);
}

class Handle {
public:
  Handle(hid_t value, herr_t (*close)(hid_t)) : value_(value), close_(close) {
    if (value < 0) throw std::runtime_error("TPIC HDF5 output: cannot create object");
  }
  ~Handle() noexcept { if (value_ >= 0) close_(value_); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  operator hid_t() const noexcept { return value_; }
  void close() {
    const auto value = value_;
    value_ = -1;
    check(close_(value), "close object");
  }
private:
  hid_t value_;
  herr_t (*close_)(hid_t);
};

void attribute(hid_t owner, const char* name, hid_t file_type, hid_t memory_type,
               const void* data, hsize_t count = 0) {
  Handle space(count == 0 ? H5Screate(H5S_SCALAR) : H5Screate_simple(1, &count, nullptr), H5Sclose);
  Handle value(H5Acreate2(owner, name, file_type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
  check(H5Awrite(value, memory_type, data), "write attribute");
}

void attribute(hid_t owner, const char* name, const char* value) {
  Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
  check(H5Tset_size(type, std::strlen(value) + 1), "set string size");
  check(H5Tset_strpad(type, H5T_STR_NULLTERM), "set string padding");
  check(H5Tset_cset(type, H5T_CSET_UTF8), "set string encoding");
  attribute(owner, name, type, type, value);
}

void attribute(hid_t owner, const char* name, const std::string& value) {
  attribute(owner, name, value.c_str());
}

template<class T>
std::pair<hid_t, hid_t> types() {
  if constexpr (std::is_same_v<T, double>) return {H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE};
  else if constexpr (std::is_same_v<T, float>) return {H5T_IEEE_F32LE, H5T_NATIVE_FLOAT};
  else if constexpr (std::is_same_v<T, std::int32_t>) return {H5T_STD_I32LE, H5T_NATIVE_INT32};
  else if constexpr (std::is_same_v<T, std::uint32_t>) return {H5T_STD_U32LE, H5T_NATIVE_UINT32};
  else if constexpr (std::is_same_v<T, std::uint64_t>) return {H5T_STD_U64LE, H5T_NATIVE_UINT64};
  else if constexpr (std::is_same_v<T, std::uint8_t>) return {H5T_STD_U8LE, H5T_NATIVE_UINT8};
  else static_assert(!sizeof(T), "unsupported HDF5 output type");
}

template<class T> requires std::is_arithmetic_v<T>
void attribute(hid_t owner, const char* name, T value) {
  const auto [file_type, memory_type] = types<T>();
  attribute(owner, name, file_type, memory_type, &value);
}

template<class T, std::size_t N>
void attribute(hid_t owner, const char* name, const std::array<T, N>& values) {
  const auto [file_type, memory_type] = types<T>();
  attribute(owner, name, file_type, memory_type, values.data(), N);
}

template<class T>
Handle define(hid_t owner, const std::string& name, std::span<const std::size_t> shape) {
  std::array<hsize_t, 4> dimensions{};
  if (shape.size() > dimensions.size())
    throw std::logic_error("TPIC HDF5 output: dataset has too many dimensions");
  for (std::size_t axis = 0; axis < shape.size(); ++axis) dimensions[axis] = shape[axis];
  Handle space(H5Screate_simple(static_cast<int>(shape.size()), dimensions.data(), nullptr), H5Sclose);
  return Handle(H5Dcreate2(owner, name.c_str(), types<T>().first, space, H5P_DEFAULT, H5P_DEFAULT,
                           H5P_DEFAULT), H5Dclose);
}

// Writes values into rows [first, first + rows) of a defined dataset; a row
// spans every dimension after the first. HDF5 keeps zero-length datasets, so an
// empty write writes nothing.
template<class T>
void write_rows(hid_t dataset, std::size_t first, std::span<const T> values) {
  if (values.empty()) return;
  Handle file_space(H5Dget_space(dataset), H5Sclose);
  std::array<hsize_t, 4> start{}, count{};
  const int rank = H5Sget_simple_extent_dims(file_space, count.data(), nullptr);
  if (rank < 1) throw std::runtime_error("TPIC HDF5 output: cannot read dataset shape");
  hsize_t row = 1;
  for (int axis = 1; axis < rank; ++axis) row *= count[axis];
  start[0] = first;
  count[0] = values.size() / row;
  check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr),
        "select dataset rows");
  Handle memory_space(H5Screate_simple(rank, count.data(), nullptr), H5Sclose);
  check(H5Dwrite(dataset, types<T>().second, memory_space, file_space, H5P_DEFAULT, values.data()),
        "write dataset");
}

class Hdf5Engine final : public detail::DatasetOutputEngine {
public:
  Hdf5Engine(data::OutputChannel channel, const config::PreparedOutputSelection& selection,
             const config::PreparedDomain& domain, config::PrecisionMetadata precision)
      : DatasetOutputEngine(channel, selection, domain, precision, "hdf5", ".h5") {}

  void write_piece(data::OutputSnapshotView snapshot) override {
    begin_piece(snapshot);
    detail::StagedOutput staged(piece_, overwrite_);
    {
      hdf5_waiting.fetch_add(1);
      std::lock_guard lock(hdf5_mutex);
      hdf5_waiting.fetch_sub(1);
      Handle file(H5Fcreate(staged.temporary_path().c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
      visit_attributes(snapshot, "tpic-output-v1",
          [&](const char* name, const auto& value) { attribute(file, name, value); });
      {
        Handle group(H5Gcreate2(file, channel_.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
      }
      visit_species(snapshot, [&](const auto& layout, std::uint64_t count) {
        Handle group(H5Gcreate2(file, layout.path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
        attribute(group, "count", count);
      });
      visit_datasets(snapshot, [&](const auto& path, auto dimensions, auto values) {
        using T = typename decltype(values)::value_type;
        write_rows(define<T>(file, path, dimensions), 0, values);
      });
      write_particle_datasets(snapshot,
          [&]<class T>(const std::string& path, std::span<const std::size_t> shape, std::type_identity<T>) {
            return define<T>(file, path, shape);
          },
          [&]<class T>(const Handle& dataset, std::size_t first, std::span<const T> values) {
            write_rows(dataset, first, values);
          });
      // H5Fclose reports flush failures before the final filename is published.
      file.close();
    }
    staged.commit();
  }
};

// A checkpoint piece of contiguous datasets. A dataset is written in chunks of
// at most 2^23 values (one row when a row is longer), each under the lock taken
// after any waiting output piece, so an output piece waits for one chunk, never
// for a whole checkpoint.
class Hdf5CheckpointEngine final : public detail::CheckpointEngine {
public:
  Hdf5CheckpointEngine(const config::PreparedOutput& output, const config::PreparedDomain& domain)
      : CheckpointEngine(output, domain, ".h5") {}

private:
  static constexpr std::size_t chunk_values = std::size_t{1} << 23;

  // Writes values into a new dataset; returns with the lock held.
  template <class T>
  static void write(std::unique_lock<std::mutex>& lock, hid_t owner, const std::string& name,
                    std::span<const std::size_t> shape, std::span<const T> values) {
    if (values.empty()) return;
    Handle dataset = define<T>(owner, name, shape);
    std::size_t row = 1;
    for (std::size_t axis = 1; axis < shape.size(); ++axis) row *= shape[axis];
    const std::size_t rows = std::max<std::size_t>(1, chunk_values / row);
    for (std::size_t first = 0; first < shape[0]; first += rows) {
      if (first != 0) {
        lock.unlock();
        while (hdf5_waiting.load() != 0) std::this_thread::yield();
        lock.lock();
      }
      write_rows(dataset, first, values.subspan(first * row, std::min(rows, shape[0] - first) * row));
    }
  }

  void write_file(const std::string& path, data::OutputSnapshotView snapshot) override {
    std::unique_lock lock(hdf5_mutex);
    Handle file(H5Fcreate(path.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
    attribute(file, "schema", detail::checkpoint_schema);
    detail::visit_checkpoint_header(*snapshot.checkpoint,
        [&](const char* name, const auto& value) { attribute(file, name, value); });
    (void)Handle(H5Gcreate2(file, "fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    for (std::size_t s = 0; s < snapshot.particles.size(); ++s) {
      Handle species(H5Gcreate2(file, detail::checkpoint_species_path(s).c_str(), H5P_DEFAULT, H5P_DEFAULT,
                                H5P_DEFAULT), H5Gclose);
      attribute(species, "count", static_cast<std::uint64_t>(snapshot.particles[s].id.size()));
      attribute(species, "sorted_prefix", index_[s].sorted_prefix);
    }
    visit_arrays(snapshot, [&](const std::string& name, std::span<const std::size_t> shape, auto values) {
      write(lock, file, name, shape, values);
    });
    // H5Fclose reports flush failures before the piece is synced and renamed.
    file.close();
  }
};

template <class T> void read_attribute(hid_t owner, const char* name, T* values, hssize_t count) {
  Handle attribute(H5Aopen(owner, name, H5P_DEFAULT), H5Aclose);
  Handle space(H5Aget_space(attribute), H5Sclose);
  if (H5Sget_simple_extent_npoints(space) != count)
    throw std::runtime_error(std::string("TPIC restart: checkpoint attribute ") + name + " has another size");
  check(H5Aread(attribute, types<T>().second, values), "read attribute");
}
template <class T> void read_attribute(hid_t owner, const char* name, T& value) {
  read_attribute(owner, name, &value, 1);
}
template <class T, std::size_t N> void read_attribute(hid_t owner, const char* name, std::array<T, N>& values) {
  read_attribute(owner, name, values.data(), N);
}

class Hdf5CheckpointReader final : public detail::CheckpointReader {
public:
  explicit Hdf5CheckpointReader(const std::string& path) {
    std::lock_guard lock(hdf5_mutex);
    file_ = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_ < 0) throw std::runtime_error("TPIC restart: cannot open checkpoint piece " + path);
  }
  ~Hdf5CheckpointReader() override {
    std::lock_guard lock(hdf5_mutex);
    H5Fclose(file_);
  }
  Hdf5CheckpointReader(const Hdf5CheckpointReader&) = delete;
  Hdf5CheckpointReader& operator=(const Hdf5CheckpointReader&) = delete;

  data::CheckpointHeader header() override {
    std::lock_guard lock(hdf5_mutex);
    {
      Handle attribute(H5Aopen(file_, "schema", H5P_DEFAULT), H5Aclose);
      Handle type(H5Aget_type(attribute), H5Tclose);
      std::string schema(H5Tget_size(type), '\0');
      check(H5Aread(attribute, type, schema.data()), "read attribute");
      schema.resize(std::strlen(schema.c_str()));
      if (schema != detail::checkpoint_schema)
        throw std::runtime_error("TPIC restart: checkpoint schema '" + schema + "' is not " +
                                 detail::checkpoint_schema);
    }
    data::CheckpointHeader header;
    detail::visit_checkpoint_header(header,
        [&](const char* name, auto& value) { read_attribute(file_, name, value); });
    return header;
  }
  std::array<std::uint64_t, 2> records(std::size_t species) override {
    std::lock_guard lock(hdf5_mutex);
    Handle group(H5Gopen2(file_, detail::checkpoint_species_path(species).c_str(), H5P_DEFAULT), H5Gclose);
    std::array<std::uint64_t, 2> records{};
    read_attribute(group, "count", records[0]);
    read_attribute(group, "sorted_prefix", records[1]);
    return records;
  }
  void read(std::span<const detail::CheckpointRead> reads) override {
    std::lock_guard lock(hdf5_mutex);
    for (const auto& read : reads)
      std::visit([&]<class T>(std::span<T> destination) {
        if (destination.empty()) return;
        Handle dataset(H5Dopen2(file_, read.dataset.c_str(), H5P_DEFAULT), H5Dclose);
        Handle file_space(H5Dget_space(dataset), H5Sclose);
        std::array<hsize_t, 3> start{}, count{};
        std::copy(read.start.begin(), read.start.end(), start.begin());
        std::copy(read.count.begin(), read.count.end(), count.begin());
        check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr),
              "select checkpoint box");
        Handle memory_space(H5Screate_simple(read.dimensions, count.data(), nullptr), H5Sclose);
        check(H5Dread(dataset, types<T>().second, memory_space, file_space, H5P_DEFAULT, destination.data()),
              "read checkpoint array");
      }, read.destination);
  }

private:
  hid_t file_ = -1;
};

} // namespace

std::unique_ptr<OutputEngine> make_hdf5_engine(
    data::OutputChannel channel, const config::PreparedOutputSelection& selection,
    const config::PreparedDomain& domain, config::PrecisionMetadata precision) {
  return std::make_unique<Hdf5Engine>(channel, selection, domain, precision);
}

namespace detail {
std::unique_ptr<CheckpointEngine> make_hdf5_checkpoint_engine(const config::PreparedOutput& output,
                                                              const config::PreparedDomain& domain) {
  return std::make_unique<Hdf5CheckpointEngine>(output, domain);
}
std::unique_ptr<CheckpointReader> open_hdf5_checkpoint(const std::string& path) {
  return std::make_unique<Hdf5CheckpointReader>(path);
}
} // namespace detail

} // namespace tpic::io
