#include <tpic/io/detail/checkpoint_engine.hpp>
#include <tpic/io/detail/dataset_engine.hpp>

#include <adios2.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace tpic::io {
namespace {

template<class T, std::size_t N>
void attribute(adios2::IO& io, const std::string& name, const std::array<T, N>& values) {
  io.DefineAttribute<T>(name, values.data(), values.size());
}

template<class T> requires std::is_arithmetic_v<T>
void attribute(adios2::IO& io, const std::string& name, T value) {
  io.DefineAttribute<T>(name, value);
}

void attribute(adios2::IO& io, const std::string& name, const std::string& value) {
  io.DefineAttribute<std::string>(name, value);
}

void attribute(adios2::IO& io, const std::string& name, const char* value) {
  io.DefineAttribute<std::string>(name, value);
}

// Defines a constant-shape array of one block and puts it deferred: BP5 may
// read the source as late as EndStep, so it must stay valid until then.
// BP5 cannot represent an empty array block: an empty array is not written.
template<class Values>
void put(adios2::IO& io, adios2::Engine& engine, const std::string& name,
         std::span<const std::size_t> shape, const Values& values) {
  if (values.empty()) return;
  const adios2::Dims dimensions(shape.begin(), shape.end()), starts(shape.size(), 0);
  auto variable = io.DefineVariable<typename Values::value_type>(name, dimensions, starts, dimensions,
                                                                 adios2::ConstantDims);
  variable.SetMemorySpace(adios2::MemorySpace::Host);
  engine.Put(variable, values.data(), adios2::Mode::Deferred);
}

// Each engine owns a dedicated serial context. Clear file-specific definitions
// even when Open/Put/Close fails, after the local IO/Engine handles are gone and
// before staged files and borrowed snapshot buffers can be released.
struct ScopedIOs {
  adios2::ADIOS& context;
  ~ScopedIOs() noexcept { context.RemoveAllIOs(); }
};

// The arrays of one piece, each with one or three components; absent ones are skipped.
std::string fides_field_schema(std::span<const data::OwnedFieldArray> arrays) {
  // Grid geometry is independent of field selection and component count.
  // Fides requires all three geometry values to use array_variable together
  // so their deferred reads finish before it constructs the uniform grid.
  constexpr auto dimensions = "{\"source\":\"array_variable\",\"data_source\":\"source\","
      "\"variable\":\"mesh/dimensions\"}";
  std::ostringstream json;
  json << "{\"tpic_fields\":{\"data_sources\":[{\"name\":\"source\",\"filename_mode\":\"input\"}],"
       << "\"coordinate_system\":{\"array\":{\"array_type\":\"uniform_point_coordinates\","
       << "\"dimensions\":" << dimensions
       << ",\"origin\":{\"source\":\"array_variable\",\"data_source\":\"source\",\"variable\":\"mesh/origin\"},"
       << "\"spacing\":{\"source\":\"array_variable\",\"data_source\":\"source\",\"variable\":\"mesh/spacing\"}}},"
       << "\"cell_set\":{\"cell_set_type\":\"structured\",\"dimensions\":" << dimensions
       << "},\"fields\":[";
  bool first = true;
  for (const auto& field : arrays) {
    if (field.values.empty()) continue;
    if (!first) json << ',';
    first = false;
    // Like VTK, each field stores complete tuples from the shared host snapshot.
    // Fides consumes vectors directly without assembling separate components.
    json << "{\"name\":\"" << field.name << "\",\"association\":\""
         << (field.cell ? "cell_set" : "points") << "\",\"array\":"
         << "{\"array_type\":\"basic\",\"data_source\":\"source\",\"variable\":\"fields/"
         << field.name << "\",\"is_vector\":\"" << (field.components == 3 ? "true" : "false")
         << "\"}}";
  }
  json << "],\"step_information\":{\"data_source\":\"source\",\"variable\":\"time\"}}}";
  return json.str();
}

class Adios2Engine final : public detail::DatasetOutputEngine {
public:
  Adios2Engine(data::OutputChannel channel, const config::PreparedOutputSelection& selection,
               const config::PreparedDomain& domain, config::PrecisionMetadata precision)
      : DatasetOutputEngine(channel, selection, domain, precision, "adios2", ".bp") {
    dimensions_.reserve(4);
    starts_.reserve(4);
  }

  void write_piece(data::OutputSnapshotView snapshot) override {
    begin_piece(snapshot);
    detail::StagedOutput staged(piece_, overwrite_);
    // Lazily created on the writer worker. This serial context survives epochs
    // and never requires MPI_THREAD_MULTIPLE or accesses simulation storage.
    if (!adios_) adios_ = std::make_unique<adios2::ADIOS>();
    ScopedIOs cleanup{*adios_};
    auto io = adios_->DeclareIO("tpic-output");
    io.SetEngine("BP5");
    visit_attributes(snapshot, fields_ ? "tpic-output-v3" : "tpic-output-v1",
        [&](const char* name, const auto& value) { attribute(io, name, value); });
    visit_species(snapshot, [&](const auto& layout, std::uint64_t count) {
      attribute(io, layout.count_attribute, count);
    });
    visit_datasets(snapshot, [&](const auto& path, auto dimensions, auto) {
      attribute(io, path + "/components", static_cast<std::int32_t>(dimensions[3]));
    });
    if (fields_) attribute(io, "fides/schema", fides_field_schema(snapshot.arrays));

    auto engine = io.Open(staged.temporary_path(), adios2::Mode::Write);
    engine.BeginStep();
    if (fields_) {
      auto time = io.DefineVariable<double>("time");
      time.SetMemorySpace(adios2::MemorySpace::Host);
      engine.Put(time, &time_, adios2::Mode::Deferred);
      constexpr std::array<std::size_t, 1> shape{3};
      put(io, engine, "mesh/dimensions", shape, nodes_);
      put(io, engine, "mesh/origin", shape, domain_.origin);
      put(io, engine, "mesh/spacing", shape, domain_.spacing);
    }
    visit_datasets(snapshot, [&](const auto& path, auto dimensions, auto values) {
      put(io, engine, path, dimensions, values);
    });
    write_particle_datasets(snapshot,
        [&]<class T>(const std::string& path, std::span<const std::size_t> shape, std::type_identity<T>) {
          // BP5 cannot represent an empty array block: an empty species keeps
          // only its count attribute.
          if (shape[0] == 0) return adios2::Variable<T>{};
          dimensions_.assign(shape.begin(), shape.end());
          starts_.assign(shape.size(), 0);
          auto variable = io.DefineVariable<T>(path, dimensions_, starts_, dimensions_);
          variable.SetMemorySpace(adios2::MemorySpace::Host);
          return variable;
        },
        [&]<class T>(adios2::Variable<T>& variable, std::size_t first, std::span<const T> values) {
          dimensions_ = variable.Shape();
          dimensions_[0] = values.size() / (dimensions_.size() == 2 ? dimensions_[1] : 1);
          starts_.assign(dimensions_.size(), 0);
          starts_[0] = first;
          variable.SetSelection({starts_, dimensions_});
          // BP5 writes the deferred chunk before the scratch is refilled, so its
          // buffer never holds the whole population.
          engine.Put(variable, values.data(), adios2::Mode::Deferred);
          engine.PerformDataWrite();
        });
    // The remaining deferred sources are engine members and the snapshot's
    // field arrays, which the caller retains through EndStep and Close.
    engine.EndStep();
    engine.Close();
    staged.commit();
  }

private:
  std::unique_ptr<adios2::ADIOS> adios_;
  adios2::Dims dimensions_, starts_;
};

// A checkpoint piece: one step, every array put deferred straight from the
// snapshot and the particle images, no profile and no statistics pass.
class Adios2CheckpointEngine final : public detail::CheckpointEngine {
public:
  Adios2CheckpointEngine(const config::PreparedOutput& output, const config::PreparedDomain& domain)
      : CheckpointEngine(output, domain, ".bp") {}

private:
  void write_file(const std::string& path, data::OutputSnapshotView snapshot) override {
    if (!adios_) adios_ = std::make_unique<adios2::ADIOS>();
    ScopedIOs cleanup{*adios_};
    auto io = adios_->DeclareIO("tpic-checkpoint");
    io.SetEngine("BP5");
    io.SetParameters({{"Profile", "Off"}, {"StatsLevel", "0"}});
    attribute(io, "schema", detail::checkpoint_schema);
    detail::visit_checkpoint_header(*snapshot.checkpoint,
        [&](const char* name, const auto& value) { attribute(io, name, value); });
    for (std::size_t s = 0; s < snapshot.particles.size(); ++s) {
      const auto path = detail::checkpoint_species_path(s);
      attribute(io, path + "/count", static_cast<std::uint64_t>(snapshot.particles[s].id.size()));
      attribute(io, path + "/sorted_prefix", index_[s].sorted_prefix);
    }
    auto engine = io.Open(path, adios2::Mode::Write);
    engine.BeginStep();
    visit_arrays(snapshot, [&](const std::string& name, std::span<const std::size_t> shape, auto values) {
      put(io, engine, name, shape, values);
    });
    engine.EndStep();
    engine.Close();
  }

  std::unique_ptr<adios2::ADIOS> adios_;
};

template<class T> void read_attribute(adios2::IO& io, const std::string& name, T* values, std::size_t count) {
  const auto attribute = io.InquireAttribute<T>(name);
  const auto data = attribute ? attribute.Data() : std::vector<T>{};
  if (data.size() != count) throw std::runtime_error("TPIC restart: missing or resized checkpoint attribute " + name);
  std::copy(data.begin(), data.end(), values);
}
template<class T> void read_attribute(adios2::IO& io, const std::string& name, T& value) {
  read_attribute(io, name, &value, 1);
}
template<class T, std::size_t N>
void read_attribute(adios2::IO& io, const std::string& name, std::array<T, N>& values) {
  read_attribute(io, name, values.data(), N);
}

// The engine closes when the context, declared first, is destroyed.
class Adios2CheckpointReader final : public detail::CheckpointReader {
public:
  explicit Adios2CheckpointReader(const std::string& path) : io_(adios_.DeclareIO("tpic-restart")) {
    io_.SetEngine("BP5");
    engine_ = io_.Open(path, adios2::Mode::ReadRandomAccess);
  }

  data::CheckpointHeader header() override {
    std::string schema;
    read_attribute(io_, "schema", schema);
    if (schema != detail::checkpoint_schema)
      throw std::runtime_error("TPIC restart: checkpoint schema '" + schema + "' is not " +
                               detail::checkpoint_schema);
    data::CheckpointHeader header;
    detail::visit_checkpoint_header(header, [&](const char* name, auto& value) { read_attribute(io_, name, value); });
    return header;
  }
  std::array<std::uint64_t, 2> records(std::size_t species) override {
    const auto path = detail::checkpoint_species_path(species);
    std::array<std::uint64_t, 2> records{};
    read_attribute(io_, path + "/count", records[0]);
    read_attribute(io_, path + "/sorted_prefix", records[1]);
    return records;
  }
  void read(std::span<const detail::CheckpointRead> reads) override {
    for (const auto& read : reads)
      std::visit([&]<class T>(std::span<T> destination) {
        if (destination.empty()) return;
        auto variable = io_.InquireVariable<T>(read.dataset);
        if (!variable) throw std::runtime_error("TPIC restart: missing checkpoint array " + read.dataset);
        const auto dimensions = static_cast<std::size_t>(read.dimensions);
        variable.SetSelection({adios2::Dims(read.start.begin(), read.start.begin() + dimensions),
                               adios2::Dims(read.count.begin(), read.count.begin() + dimensions)});
        engine_.Get(variable, destination.data(), adios2::Mode::Deferred);
      }, read.destination);
    engine_.PerformGets();
  }

private:
  adios2::ADIOS adios_;
  adios2::IO io_;
  adios2::Engine engine_;
};

} // namespace

std::unique_ptr<OutputEngine> make_adios2_engine(
    data::OutputChannel channel, const config::PreparedOutputSelection& selection,
    const config::PreparedDomain& domain, config::PrecisionMetadata precision) {
  return std::make_unique<Adios2Engine>(channel, selection, domain, precision);
}

namespace detail {
std::unique_ptr<CheckpointEngine> make_adios2_checkpoint_engine(const config::PreparedOutput& output,
                                                                const config::PreparedDomain& domain) {
  return std::make_unique<Adios2CheckpointEngine>(output, domain);
}
std::unique_ptr<CheckpointReader> open_adios2_checkpoint(const std::string& path) {
  return std::make_unique<Adios2CheckpointReader>(path);
}
} // namespace detail

} // namespace tpic::io
