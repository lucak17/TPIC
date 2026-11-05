#include <tpic/io/async_output_writer.hpp>
#include <tpic/io/detail/output_layout.hpp>

#include "../support/mpi_field_files.hpp"
#include "../support/test_builders.hpp"

#if TPIC_HAS_ADIOS2
#include <adios2.h>
#endif
#if TPIC_HAS_HDF5
#include <hdf5.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {
namespace fs = std::filesystem;
using tpic::data::OwnedOutputSnapshot;
using tpic::data::OutputChannel;
using tpic::data::OutputSnapshotView;
using tpic::config::OutputEngineKind;
using tpic::io::AsyncOutputWriter;

// VTK fields are written collectively (vtk_field_output covers them); no
// backend here writes them, so the field file writer stays closed.
lpfx::distributed::MpiFileWriter field_files;

void configure(AsyncOutputWriter& writer, const tpic::config::PreparedOutput& prepared,
               const tpic::config::PreparedDomain& geometry, tpic::config::PrecisionMetadata precision) {
  tpic::test::configure(writer, prepared, geometry, field_files, std::move(precision));
}

using tpic::test::file_contents;
using tpic::test::vtk_appended_array;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function&& function, const char* message) {
  try { function(); } catch (const std::exception&) { return; }
  throw std::runtime_error(message);
}

struct TemporaryDirectory {
  fs::path path;
  bool keep = false;
  explicit TemporaryDirectory(const fs::path& retained = {}) {
    if (!retained.empty()) {
      require(!fs::exists(retained), "retained fixture directory already exists");
      fs::create_directories(retained);
      path = retained;
      keep = true;
      return;
    }
    auto pattern = (fs::temp_directory_path() / "tpic-dataset-output-XXXXXX").string();
    require(::mkdtemp(pattern.data()) != nullptr, "cannot create test directory");
    path = pattern;
  }
  ~TemporaryDirectory() {
    if (!keep) { std::error_code ignored; fs::remove_all(path, ignored); }
  }
};

tpic::config::PreparedDomain domain(int rank) {
  tpic::config::GridConfig grid;
  grid.nxc = 5;
  grid.nyc = grid.nzc = 2;
  grid.lx = 10;
  grid.ly = 4;
  grid.lz = 6;
  tpic::config::TopologyConfig topology;
  topology.xlen = 2;
  return tpic::config::prepare_domain(grid, topology, rank, 2);
}

tpic::config::PreparedOutput output(const fs::path& path, OutputEngineKind field_engine,
                                   OutputEngineKind particle_engine, bool overwrite = false) {
  tpic::config::PreparedOutput result;
  result.save_dir = path.string();
  result.field_engine = field_engine;
  result.particle_engine = particle_engine;
  result.overwrite_output = overwrite;
  result.field_cycle = 2;
  result.output_e = result.total_selected[tpic::config::output_rho_quantity] = true;
  result.particles_cycle = 1;
  result.species_selected.fill({false, false});
  return result;
}

OwnedOutputSnapshot snapshot(const tpic::config::PreparedOutput& prepared, int cycle, int rank) {
  const tpic::config::PreparedOutputSelection selection(prepared);
  const auto nodes = static_cast<std::size_t>(domain(rank).interior_node_count);
  OwnedOutputSnapshot result;
  result.cycle = cycle;
  result.time = cycle * 0.125;
  result.final_state = cycle == 4;
  result.fields_present = cycle % 2 == 0;
  result.particles_present = cycle != 2;
  if (result.fields_present) {
    for (const auto& field : selection.arrays()) {
      std::vector<double> values(nodes * field.components);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<double>(i) * 0.25 - 1 + cycle + rank * 100;
      tpic::test::append_array(result, field.name, field.components, values);
    }
  }
  if (result.particles_present) {
    // Snapshots borrow their records: each storage lives until the program ends.
    static std::list<tpic::test::ParticleRecordStorage> storages;
    for (std::size_t species_index = 0; species_index < 2; ++species_index) {
      auto& storage = storages.emplace_back();
      const std::size_t count = species_index == 1 || cycle == 3 ? 0 : 2 + cycle + rank;
      for (std::size_t i = 0; i < count; ++i)
        storage.add(domain(rank), {int(i % 2), 1, int(i % 2)}, {0.0625 * double(i + 1), 0.5, 0.25 + 0.125 * cycle},
                    {-1.25, 0.0, 8.125 + double(i)}, -0.25 * double(i + 1), (std::uint64_t{1} << 60) + 17 + i);
      result.particles.push_back(storage.records(species_index));
    }
  }
  return result;
}

std::string channel_name(OutputChannel channel) {
  return channel == OutputChannel::fields ? "fields" : "particles";
}

std::string epoch(OutputChannel channel, int cycle) {
  return channel_name(channel) + "_t00000" + std::to_string(cycle);
}

struct Backend {
  OutputEngineKind engine;
  std::string name, extension;

  std::string piece(OutputChannel channel, int cycle, int rank) const {
    const auto suffix = engine == OutputEngineKind::Vtk ? ".vtp" : extension;
    return epoch(channel, cycle) + "_r00000" + std::to_string(rank) + suffix;
  }
  std::string manifest(OutputChannel channel, int cycle) const {
    const auto suffix = engine == OutputEngineKind::Vtk ? ".pvtp" : "." + name + ".json";
    return epoch(channel, cycle) + suffix;
  }
};

constexpr std::array channels{OutputChannel::fields, OutputChannel::particles};

const double* first_fraction(const tpic::data::ParticleRecords& records) {
  return std::get<std::span<const double>>(records.fraction[0]).data();
}

bool present(OutputSnapshotView snapshot, OutputChannel channel) {
  return channel == OutputChannel::fields ? snapshot.fields_present : snapshot.particles_present;
}

#if TPIC_HAS_ADIOS2
template<class T>
std::vector<T> adios_values(adios2::IO& io, adios2::Engine& reader, const std::string& name,
                          const adios2::Dims& shape) {
  auto variable = io.InquireVariable<T>(name);
  require(static_cast<bool>(variable), "ADIOS2 variable is missing or incorrectly typed");
  require(variable.Shape() == shape, "ADIOS2 variable shape is incorrect");
  std::size_t count = 1;
  for (const auto dimension : shape) count *= dimension;
  std::vector<T> result(count);
  reader.Get(variable, result.data(), adios2::Mode::Sync);
  return result;
}

template<class T>
T adios_attribute(adios2::IO& io, const std::string& name) {
  const auto attribute = io.InquireAttribute<T>(name);
  require(static_cast<bool>(attribute), "ADIOS2 attribute is missing or incorrectly typed");
  require(attribute.Data().size() == 1, "ADIOS2 scalar attribute has incorrect size");
  return attribute.Data().front();
}

void read_adios(const fs::path& file, OutputSnapshotView expected, int rank) {
  adios2::ADIOS adios;
  auto io = adios.DeclareIO("readback");
  auto reader = io.Open(file.string(), adios2::Mode::ReadRandomAccess);
  require(adios_attribute<std::string>(io, "schema") ==
              (expected.fields_present ? "tpic-output-v3" : "tpic-output-v1"), "ADIOS2 schema mismatch");
  require(adios_attribute<std::string>(io, "channel") ==
              (expected.fields_present ? "fields" : "particles"), "ADIOS2 channel mismatch");
  require(adios_attribute<std::string>(io, "precision_profile") == "single-particles-moments-fields/double", "ADIOS2 precision missing");
  require(adios_attribute<std::string>(io, "field_order") == "z,y,x,component",
          "ADIOS2 order mismatch");
  require(adios_attribute<std::int32_t>(io, "cycle") == expected.cycle, "ADIOS2 cycle mismatch");
  require(adios_attribute<double>(io, "time") == expected.time, "ADIOS2 time mismatch");
  require(adios_attribute<std::uint8_t>(io, "final_state") == expected.final_state, "ADIOS2 final mismatch");
  require(adios_attribute<std::uint8_t>(io, "fields_present") == expected.fields_present, "ADIOS2 field cadence mismatch");
  require(adios_attribute<std::uint8_t>(io, "particles_present") == expected.particles_present, "ADIOS2 particle cadence mismatch");
  require(adios_attribute<std::int32_t>(io, "rank") == rank, "ADIOS2 rank mismatch");
  require(adios_attribute<std::int32_t>(io, "ranks") == 2, "ADIOS2 rank count mismatch");
  const auto geometry = domain(rank);
  require(io.InquireAttribute<int>("global_offset").Data() ==
      std::vector<int>(geometry.global_offset.begin(), geometry.global_offset.end()), "ADIOS2 global offset mismatch");
  require(io.InquireAttribute<double>("spacing").Data() ==
      std::vector<double>(geometry.spacing.begin(), geometry.spacing.end()), "ADIOS2 spacing mismatch");
  require(io.InquireAttribute<double>("local_origin").Data() ==
      std::vector<double>(geometry.origin.begin(), geometry.origin.end()), "ADIOS2 local origin mismatch");
  if (expected.fields_present) {
    require(!io.AvailableAttributes().contains("vtk.xml"),
            "ADIOS2 vector fields incorrectly advertise the scalar-only VTX reader");
    require(!adios_attribute<std::string>(io, "fides/schema").empty(),
            "ADIOS2 field output lacks its vector-aware Fides schema");
    require(adios_values<double>(io, reader, "time", {}) == std::vector{expected.time},
            "ADIOS2 time variable differs from physical time");
    const std::vector<std::uint64_t> dimensions{
        static_cast<std::uint64_t>(geometry.local_cells[0]) + 1,
        static_cast<std::uint64_t>(geometry.local_cells[1]) + 1,
        static_cast<std::uint64_t>(geometry.local_cells[2]) + 1};
    require(adios_values<std::uint64_t>(io, reader, "mesh/dimensions", {3}) == dimensions,
            "ADIOS2 mesh dimensions differ from the VTK point dimensions");
    require(adios_values<double>(io, reader, "mesh/origin", {3}) ==
                std::vector<double>(geometry.origin.begin(), geometry.origin.end()),
            "ADIOS2 mesh origin differs from the VTK origin");
    require(adios_values<double>(io, reader, "mesh/spacing", {3}) ==
                std::vector<double>(geometry.spacing.begin(), geometry.spacing.end()),
            "ADIOS2 mesh spacing differs from the VTK spacing");
    const adios2::Dims nodes{static_cast<std::size_t>(geometry.local_cells[2]) + 1,
                           static_cast<std::size_t>(geometry.local_cells[1]) + 1,
                           static_cast<std::size_t>(geometry.local_cells[0]) + 1};
    for (const auto& field : expected.arrays) {
      auto shape = nodes;
      shape.push_back(static_cast<std::size_t>(field.components));
      require(adios_values<double>(io, reader, "fields/" + field.name, shape) == field.values,
              "ADIOS2 interleaved field values differ from the shared VTK snapshot");
    }
    require(io.AvailableVariables().size() == expected.arrays.size() + 4,
            "ADIOS2 field file has extra variables or duplicate field payload");
  }
  if (!expected.fields_present) {
    require(!io.AvailableAttributes().contains("vtk.xml"),
            "ADIOS2 particle file incorrectly declares an ImageData schema");
    require(!io.AvailableAttributes().contains("fides/schema"),
            "ADIOS2 particle file incorrectly declares a field Fides schema");
    for (const auto& [name, metadata] : io.AvailableVariables()) {
      require(!name.starts_with("fields/"), "ADIOS2 particle file contains field payload");
      require(!name.starts_with("mesh/"), "ADIOS2 particle file contains field mesh geometry");
    }
    for (const auto& [name, metadata] : io.AvailableAttributes())
      require(!name.starts_with("fields/"), "ADIOS2 particle file contains field array metadata");
  }
  if (!expected.particles_present) {
    for (const auto& [name, metadata] : io.AvailableVariables())
      require(!name.starts_with("particles/"), "ADIOS2 field file contains particle payload");
    for (const auto& [name, metadata] : io.AvailableAttributes())
      require(!name.starts_with("particles/"), "ADIOS2 field file contains species metadata");
  }
  require(adios_attribute<std::uint64_t>(io, "particle_species_count") ==
              (expected.particles_present ? expected.particles.size() : 0),
          "ADIOS2 species count mismatch");
  if (expected.particles_present) for (const auto& species : expected.particles) {
    const auto prefix = "particles/species_" + std::to_string(species.species) + "/";
    const auto count = species.id.size();
    require(adios_attribute<std::uint64_t>(io, prefix + "count") == count, "ADIOS2 particle count mismatch");
    if (count == 0) {
      require(!io.AvailableVariables().contains(prefix + "id"), "ADIOS2 empty species contains payload");
      continue;
    }
    const auto arrays = tpic::test::particle_output_arrays(species, domain(rank));
    require(adios_values<double>(io, reader, prefix + "position", {count, 3}) == arrays.position, "ADIOS2 positions mismatch");
    require(adios_values<double>(io, reader, prefix + "velocity", {count, 3}) == arrays.velocity, "ADIOS2 velocities mismatch");
    require(adios_values<double>(io, reader, prefix + "charge", {count}) == arrays.charge, "ADIOS2 charges mismatch");
    require(adios_values<std::uint64_t>(io, reader, prefix + "id", {count}) == arrays.id, "ADIOS2 uint64 IDs mismatch");
  }
  reader.Close();
}
#endif

#if TPIC_HAS_HDF5
template<class T>
T hdf_attribute(hid_t owner, const char* name, hid_t type) {
  const auto attribute = H5Aopen(owner, name, H5P_DEFAULT);
  require(attribute >= 0, "HDF5 attribute missing");
  T result{};
  require(H5Aread(attribute, type, &result) >= 0, "HDF5 attribute read failed");
  H5Aclose(attribute);
  return result;
}

std::string hdf_string(hid_t file, const char* name) {
  const auto attribute = H5Aopen(file, name, H5P_DEFAULT);
  require(attribute >= 0, "HDF5 string attribute missing");
  const auto type = H5Aget_type(attribute);
  std::vector<char> data(H5Tget_size(type));
  require(H5Aread(attribute, type, data.data()) >= 0, "HDF5 string read failed");
  H5Tclose(type);
  H5Aclose(attribute);
  return data.data();
}

template<class T>
std::vector<T> hdf_values(hid_t file, const std::string& name, hid_t type,
                        const std::vector<hsize_t>& shape) {
  const auto dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  require(dataset >= 0, "HDF5 dataset missing");
  const auto space = H5Dget_space(dataset);
  require(H5Sget_simple_extent_ndims(space) == static_cast<int>(shape.size()), "HDF5 dataset rank mismatch");
  std::vector<hsize_t> actual_shape(shape.size());
  require(H5Sget_simple_extent_dims(space, actual_shape.data(), nullptr) >= 0 && actual_shape == shape,
          "HDF5 dataset shape mismatch");
  if constexpr (std::is_same_v<T, std::uint64_t>) {
    const auto actual_type = H5Dget_type(dataset);
    require(H5Tget_class(actual_type) == H5T_INTEGER && H5Tget_size(actual_type) == 8 &&
            H5Tget_sign(actual_type) == H5T_SGN_NONE, "HDF5 particle IDs are not unsigned 64-bit integers");
    H5Tclose(actual_type);
  }
  std::size_t count = 1;
  for (const auto extent : shape) count *= extent;
  std::vector<T> result(count);
  if (count != 0)
    require(H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data()) >= 0, "HDF5 dataset read failed");
  H5Sclose(space);
  H5Dclose(dataset);
  return result;
}

void read_hdf(const fs::path& path, OutputSnapshotView expected, int rank) {
  const auto file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  require(file >= 0, "HDF5 output cannot be reopened");
  require(hdf_string(file, "schema") == "tpic-output-v1", "HDF5 schema missing");
  require(hdf_string(file, "channel") ==
              (expected.fields_present ? "fields" : "particles"), "HDF5 channel mismatch");
  require(hdf_string(file, "precision_profile") == "single-particles-moments-fields/double", "HDF5 precision missing");
  require(hdf_string(file, "field_order") == "z,y,x,component", "HDF5 order missing");
  require(hdf_attribute<int>(file, "cycle", H5T_NATIVE_INT) == expected.cycle, "HDF5 cycle mismatch");
  require(hdf_attribute<double>(file, "time", H5T_NATIVE_DOUBLE) == expected.time, "HDF5 time mismatch");
  require(hdf_attribute<std::uint8_t>(file, "final_state", H5T_NATIVE_UINT8) == expected.final_state, "HDF5 final mismatch");
  require(hdf_attribute<std::uint8_t>(file, "fields_present", H5T_NATIVE_UINT8) == expected.fields_present, "HDF5 field cadence mismatch");
  require(hdf_attribute<std::uint8_t>(file, "particles_present", H5T_NATIVE_UINT8) == expected.particles_present, "HDF5 particle cadence mismatch");
  require(hdf_attribute<int>(file, "rank", H5T_NATIVE_INT) == rank, "HDF5 rank mismatch");
  require(hdf_attribute<int>(file, "ranks", H5T_NATIVE_INT) == 2, "HDF5 rank count mismatch");
  const auto geometry = domain(rank);
  require(hdf_attribute<std::array<int, 3>>(file, "global_offset", H5T_NATIVE_INT) == geometry.global_offset,
          "HDF5 global offset mismatch");
  require(hdf_attribute<std::array<double, 3>>(file, "spacing", H5T_NATIVE_DOUBLE) == geometry.spacing,
          "HDF5 spacing mismatch");
  if (expected.fields_present) for (const auto& field : expected.arrays) {
    const std::vector<hsize_t> shape{3, 3, static_cast<hsize_t>(geometry.local_cells[0]) + 1,
                                   static_cast<hsize_t>(field.components)};
    require(hdf_values<double>(file, "fields/" + field.name, H5T_NATIVE_DOUBLE, shape) == field.values,
            "HDF5 field values mismatch");
  }
  require((H5Lexists(file, "fields", H5P_DEFAULT) > 0) == expected.fields_present, "HDF5 included unscheduled fields");
  require((H5Lexists(file, "particles", H5P_DEFAULT) > 0) == expected.particles_present, "HDF5 included unscheduled particles");
  require(hdf_attribute<std::uint64_t>(file, "particle_species_count", H5T_NATIVE_UINT64) ==
              (expected.particles_present ? expected.particles.size() : 0),
          "HDF5 species count mismatch");
  if (expected.particles_present) for (const auto& species : expected.particles) {
    const auto prefix = "particles/species_" + std::to_string(species.species);
    const auto count = static_cast<hsize_t>(species.id.size());
    const auto group = H5Gopen2(file, prefix.c_str(), H5P_DEFAULT);
    require(group >= 0, "HDF5 species group missing");
    require(hdf_attribute<std::uint64_t>(group, "count", H5T_NATIVE_UINT64) == count, "HDF5 particle count mismatch");
    H5Gclose(group);
    const auto arrays = tpic::test::particle_output_arrays(species, domain(rank));
    require(hdf_values<double>(file, prefix + "/position", H5T_NATIVE_DOUBLE, {count, 3}) == arrays.position, "HDF5 positions mismatch");
    require(hdf_values<double>(file, prefix + "/velocity", H5T_NATIVE_DOUBLE, {count, 3}) == arrays.velocity, "HDF5 velocities mismatch");
    require(hdf_values<double>(file, prefix + "/charge", H5T_NATIVE_DOUBLE, {count}) == arrays.charge, "HDF5 charges mismatch");
    require(hdf_values<std::uint64_t>(file, prefix + "/id", H5T_NATIVE_UINT64, {count}) == arrays.id, "HDF5 uint64 IDs mismatch");
  }
  require(H5Fclose(file) >= 0, "HDF5 readback close failed");
}
#endif

void read_vtk(const fs::path& path, OutputSnapshotView expected, int rank) {
  const auto file = file_contents(path);
  const auto raw = file.find("<AppendedData");
  const auto metadata = file.substr(0, raw);
  const auto ascii = [&](const std::string& name) {
    const auto descriptor = metadata.find("Name=\"" + name + "\"");
    require(descriptor != std::string::npos, "VTK metadata missing");
    const auto begin = metadata.find('>', descriptor) + 1;
    return metadata.substr(begin, metadata.find('<', begin) - begin);
  };
  std::istringstream precision_bytes(ascii("precision_profile"));
  std::string precision;
  for (int value; precision_bytes >> value && value != 0; )
    precision.push_back(static_cast<char>(value));
  require(precision == "single-particles-moments-fields/double", "VTK precision mismatch");
  require(std::stoi(ascii("cycle")) == expected.cycle, "VTK cycle mismatch");
  require(std::stod(ascii("TimeValue")) == expected.time, "VTK time mismatch");
  require(std::stoi(ascii("final_state")) == expected.final_state, "VTK final state mismatch");
  require(!expected.fields_present && metadata.find("Name=\"E\"") == std::string::npos &&
          metadata.find("Name=\"rho\"") == std::string::npos, "VTK particle file contains fields");
  std::vector<double> position, velocity, charge;
  std::vector<std::uint64_t> ids, species_ids;
  for (const auto& species : expected.particles) {
    const auto arrays = tpic::test::particle_output_arrays(species, domain(rank));
    position.insert(position.end(), arrays.position.begin(), arrays.position.end());
    velocity.insert(velocity.end(), arrays.velocity.begin(), arrays.velocity.end());
    charge.insert(charge.end(), arrays.charge.begin(), arrays.charge.end());
    ids.insert(ids.end(), arrays.id.begin(), arrays.id.end());
    species_ids.insert(species_ids.end(), species.id.size(), species.species);
  }
  require(vtk_appended_array<double>(file, "position") == position, "VTK particle positions mismatch");
  require(vtk_appended_array<double>(file, "velocity") == velocity, "VTK particle velocities mismatch");
  require(vtk_appended_array<double>(file, "charge") == charge, "VTK particle charges mismatch");
  require(vtk_appended_array<std::uint64_t>(file, "id") == ids, "VTK uint64 particle IDs mismatch");
  require(vtk_appended_array<std::uint64_t>(file, "species") == species_ids, "VTK particle species mismatch");
}

void read(const Backend& backend, const fs::path& file, OutputSnapshotView expected, int rank) {
  require(expected.fields_present != expected.particles_present, "readback must select one channel");
  switch (backend.engine) {
    case OutputEngineKind::Vtk: read_vtk(file, expected, rank); return;
#if TPIC_HAS_ADIOS2
    case OutputEngineKind::Adios2: read_adios(file, expected, rank); return;
#endif
#if TPIC_HAS_HDF5
    case OutputEngineKind::Hdf5: read_hdf(file, expected, rank); return;
#endif
    default: throw std::logic_error("readback backend unavailable");
  }
}

void exercise(const fs::path& directory, const Backend& fields, const Backend& particles) {
  const auto precision = tpic::config::precision_metadata(tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_particles_moments_fields, tpic::config::ComputePrecision::double_precision});
  AsyncOutputWriter root, peer;
  configure(root, output(directory, fields.engine, particles.engine), domain(0), precision);
  configure(peer, output(directory, fields.engine, particles.engine), domain(1), precision);
  require(!fs::exists(directory), "configure created output directory");
  fs::create_directories(directory);
  require(root.fields_path().empty() && root.particles_path().empty(), "unwritten channels have a path");
  OwnedOutputSnapshot reusable_root, reusable_peer;
  std::array<std::string, 2> latest_manifest;
  for (int cycle = 0; cycle < 5; ++cycle) {
    auto root_snapshot = snapshot(output(directory, fields.engine, particles.engine), cycle, 0);
    auto peer_snapshot = snapshot(output(directory, fields.engine, particles.engine), cycle, 1);
    // Retained inactive buffers must never reach the other channel's backend.
    if (!root_snapshot.fields_present) {
      root_snapshot.arrays = std::move(reusable_root.arrays);
      peer_snapshot.arrays = std::move(reusable_peer.arrays);
    }
    if (!root_snapshot.particles_present) {
      root_snapshot.particles = std::move(reusable_root.particles);
      peer_snapshot.particles = std::move(reusable_peer.particles);
    }
    const auto expected_root = root_snapshot;
    const auto expected_peer = peer_snapshot;
    const auto* particle_storage = first_fraction(root_snapshot.particles.front());
    root.submit(std::move(root_snapshot));
    peer.submit(std::move(peer_snapshot));
    root.host_wait_piece();
    peer.host_wait_piece();
    for (const auto channel : channels) {
      const auto& backend = channel == OutputChannel::fields ? fields : particles;
      const auto manifest = directory / backend.manifest(channel, cycle);
      require(!fs::exists(manifest), "worker published a manifest before rank agreement");
      const bool active = present(OutputSnapshotView(expected_root), channel);
      for (int rank = 0; rank < 2; ++rank) {
        const auto file = directory / backend.piece(channel, cycle, rank);
        require(fs::exists(file) == active, "channel piece ignored independent cadence");
        if (active)
          read(backend, file, OutputSnapshotView(rank == 0 ? expected_root : expected_peer).select(channel), rank);
      }
    }
    reusable_peer = peer.complete_snapshot();
    for (const auto channel : channels) {
      const auto& backend = channel == OutputChannel::fields ? fields : particles;
      require(!fs::exists(directory / backend.manifest(channel, cycle)), "nonroot published a manifest");
    }
    reusable_root = root.complete_snapshot();
    require(first_fraction(reusable_root.particles.front()) == particle_storage,
            "routing copied the borrowed particle records");
    for (std::size_t index = 0; index < channels.size(); ++index) {
      const auto channel = channels[index];
      const auto& backend = channel == OutputChannel::fields ? fields : particles;
      const auto manifest = directory / backend.manifest(channel, cycle);
      if (present(OutputSnapshotView(expected_root), channel)) {
        latest_manifest[index] = manifest.string();
        const auto metadata = file_contents(manifest);
        require(metadata.find(backend.piece(channel, cycle, 0)) != std::string::npos &&
                metadata.find(backend.piece(channel, cycle, 1)) != std::string::npos,
                "manifest omitted a rank");
        const auto opposite = channel == OutputChannel::fields ? OutputChannel::particles : OutputChannel::fields;
        require(metadata.find(epoch(opposite, cycle)) == std::string::npos, "manifest mixes output channels");
        if (backend.engine != OutputEngineKind::Vtk) {
          require(metadata.find("\"channel\": \"" + channel_name(channel) + "\"") != std::string::npos,
                  "dataset manifest has incorrect channel");
          require(metadata.find("\"fields_present\": " + std::string(channel == OutputChannel::fields ? "true" : "false")) != std::string::npos &&
                  metadata.find("\"particles_present\": " + std::string(channel == OutputChannel::particles ? "true" : "false")) != std::string::npos,
                  "dataset manifest mixes channel presence flags");
        }
      } else {
        require(!fs::exists(manifest), "inactive channel published a manifest");
      }
    }
    require(root.fields_path() == latest_manifest[0] && root.particles_path() == latest_manifest[1],
            "inactive channel lost its independently stable manifest path");
  }
  for (const auto& entry : fs::directory_iterator(directory)) {
    require(!entry.path().filename().string().starts_with("output_t"), "combined output file was created");
    require(!entry.path().filename().string().starts_with(".tpic-output-"), "dataset staging leaked a directory");
  }
}

#if TPIC_HAS_ADIOS2
void adios_field_selection(const fs::path& directory, bool vector) {
  const auto precision = tpic::config::precision_metadata(tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_particles_moments_fields, tpic::config::ComputePrecision::double_precision});
  const Backend backend{OutputEngineKind::Adios2, "adios2", ".bp"};
  auto selection = output(directory, backend.engine, backend.engine);
  selection.output_e = vector;
  selection.total_selected[tpic::config::output_rho_quantity] = !vector;
  selection.particles_cycle = 0;
  // Keep the uneven rank decomposition: rank 1 has three nodes on its fastest
  // axis, which must not make a scalar array look like a three-component vector.
  fs::create_directories(directory);
  for (const int rank : {1, 0}) {
    AsyncOutputWriter writer;
    configure(writer, selection, domain(rank), precision);
    auto values = snapshot(selection, 0, rank);
    values.particles_present = false;
    values.particles.clear();
    const auto expected = values;
    writer.submit(std::move(values));
    writer.host_wait_piece();
    read(backend, directory / backend.piece(OutputChannel::fields, 0, rank),
         OutputSnapshotView(expected).select(OutputChannel::fields), rank);
    writer.complete_snapshot();
  }
}

void adios_deferred_payloads() {
  // Exceed BP5's default 4 MiB deferred threshold with the fields, and write
  // more than one particle chunk: tiny reader fixtures only exercise small
  // arrays and a single chunk.
  constexpr std::size_t minimum_deferred_bytes = 4 * 1024 * 1024;
  constexpr std::size_t initial_particles = tpic::io::detail::particle_chunk_records + 3;
  const auto particle_count = [](int cycle) -> std::size_t {
    return cycle == 1 ? 0 : initial_particles + 17 * cycle;
  };
  const auto field_value = [](int cycle, std::size_t i) { return 0.125 + 1000 * cycle + i * 0.25; };
  const auto cell_of = [](std::size_t i) {
    return std::array<int, 3>{int(i % 64), int(i / 64 % 64), int(i / 4096 % 48)};
  };
  const auto fraction_value = [](int cycle, std::size_t i, std::size_t axis) {
    return double((i + axis + cycle) % 7 + 1) / 8;
  };
  const auto velocity_value = [](int cycle, std::size_t i) { return -0.5 - 3 * cycle - i * 0.25; };
  const auto id_value = [](int cycle, std::size_t i) {
    return (std::uint64_t{1} << 60) + initial_particles * cycle + i;
  };

  // These larger files are private to this check, never retained with the
  // small ParaView fixtures. One writer and one reusable snapshot span epochs.
  TemporaryDirectory temporary;
  const Backend backend{OutputEngineKind::Adios2, "adios2", ".bp"};
  tpic::config::GridConfig grid;
  grid.nxc = grid.nyc = 64;
  grid.nzc = 48;
  const auto geometry = tpic::config::prepare_domain(grid, {}, 0, 1);
  const auto position_value = [&](int cycle, std::size_t element) {
    const auto record = element / 3, axis = element % 3;
    return (geometry.global_offset[axis] + cell_of(record)[axis] + fraction_value(cycle, record, axis)) *
           geometry.spacing[axis];
  };
  auto selection = output(temporary.path, backend.engine, backend.engine);
  selection.field_cycle = 1;
  selection.total_selected[tpic::config::output_rho_quantity] = false;
  const tpic::config::PreparedOutputSelection prepared(selection);
  const auto nodes = static_cast<std::size_t>(geometry.interior_node_count);
  AsyncOutputWriter writer;
  configure(writer, selection, geometry, tpic::config::precision_metadata(tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_particles_moments_fields, tpic::config::ComputePrecision::double_precision}));
  require(prepared.arrays().size() == 1 && prepared.arrays().front().name == "E",
          "large ADIOS2 check requires one vector field");
  require(nodes * 3 * sizeof(double) > minimum_deferred_bytes,
          "ADIOS2 vector fixture does not reach the deferred path");
  OwnedOutputSnapshot reusable;
  reusable.arrays.push_back({"E", 3, std::vector<double>(nodes * 3)});
  tpic::test::ParticleRecordStorage storage;
  for (int cycle = 0; cycle < 3; ++cycle) {
    reusable.cycle = cycle;
    reusable.time = cycle * 0.125;
    reusable.final_state = cycle == 2;
    reusable.fields_present = reusable.particles_present = true;
    auto& field = reusable.arrays.front().values;
    for (std::size_t i = 0; i < field.size(); ++i) field[i] = field_value(cycle, i);
    storage = {};
    for (std::size_t i = 0; i < particle_count(cycle); ++i)
      storage.add(geometry, cell_of(i), {fraction_value(cycle, i, 0), fraction_value(cycle, i, 1), fraction_value(cycle, i, 2)},
                  {velocity_value(cycle, 3 * i), velocity_value(cycle, 3 * i + 1), velocity_value(cycle, 3 * i + 2)},
                  -0.5 - cycle, id_value(cycle, i));
    reusable.particles = {storage.records(0), tpic::data::ParticleRecords{.species = 1}};
    writer.submit(std::move(reusable));
    writer.host_wait_piece();
    reusable = writer.complete_snapshot();
    // Reclaiming the snapshot must end every backend borrow. Overwrite all
    // returned payloads before readback, and reuse their storage next epoch.
    std::fill(reusable.arrays.front().values.begin(), reusable.arrays.front().values.end(), 0.0);
    for (auto* values : {&storage.fraction[0], &storage.fraction[1], &storage.fraction[2], &storage.velocity[0],
                         &storage.velocity[1], &storage.velocity[2], &storage.charge})
      std::fill(values->begin(), values->end(), 0.0);
    std::fill(storage.id.begin(), storage.id.end(), 0);
  }
  for (int cycle = 0; cycle < 3; ++cycle) for (const auto channel : channels) {
    adios2::ADIOS adios;
    auto io = adios.DeclareIO("large-deferred-readback");
    auto reader = io.Open((temporary.path / backend.piece(channel, cycle, 0)).string(),
                          adios2::Mode::ReadRandomAccess);
    require(adios_attribute<std::int32_t>(io, "cycle") == cycle,
            "ADIOS2 reused stale cycle metadata");
    require(adios_attribute<double>(io, "time") == cycle * 0.125,
            "ADIOS2 reused stale time metadata");
    require(adios_attribute<std::uint8_t>(io, "final_state") == (cycle == 2),
            "ADIOS2 reused stale final-state metadata");
    const auto check_array = [&](const std::string& name, const adios2::Dims& shape, auto expected) {
      const auto values = adios_values<decltype(expected(std::size_t{}))>(io, reader, name, shape);
      for (std::size_t i = 0; i < values.size(); ++i)
        if (values[i] != expected(i))
          throw std::runtime_error("ADIOS2 deferred payload changed after reclamation: " + name);
    };
    if (channel == OutputChannel::fields) {
      check_array("fields/E", {49, 65, 65, 3}, [&](std::size_t i) { return field_value(cycle, i); });
      require(io.AvailableVariables().size() == 5, "ADIOS2 retained stale field variables");
    } else {
      const auto count = particle_count(cycle);
      const std::string prefix = "particles/species_0/";
      require(adios_attribute<std::uint64_t>(io, prefix + "count") == count &&
              adios_attribute<std::uint64_t>(io, "particles/species_1/count") == 0,
              "ADIOS2 reused stale species counts");
      require(io.AvailableVariables().size() == (count == 0 ? 0 : 4),
              "ADIOS2 retained payload variables for an empty species");
      if (count != 0) {
        check_array(prefix + "position", {count, 3}, [&](std::size_t i) { return position_value(cycle, i); });
        check_array(prefix + "velocity", {count, 3}, [&](std::size_t i) { return velocity_value(cycle, i); });
        check_array(prefix + "charge", {count}, [&](std::size_t) { return -0.5 - cycle; });
        check_array(prefix + "id", {count}, [&](std::size_t i) { return id_value(cycle, i); });
      }
    }
    reader.Close();
  }
}
#endif

// Exercise replacement/preservation for each dataset channel independently.
void preservation(const fs::path& directory, const Backend& backend, OutputChannel channel) {
  const auto precision = tpic::config::precision_metadata(tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_particles_moments_fields, tpic::config::ComputePrecision::double_precision});
  const auto configure = [&](AsyncOutputWriter& writer, const fs::path& path, bool overwrite = false) {
    auto selection = output(path, backend.engine, backend.engine, overwrite);
    if (channel == OutputChannel::fields) selection.particles_cycle = 0;
    else selection.field_cycle = 0;
    ::configure(writer, selection, domain(0), precision);
  };
  const auto capture = [&] {
    auto values = snapshot(output(directory, backend.engine, backend.engine), 0, 0);
    if (channel != OutputChannel::fields) values.arrays.clear();
    values.fields_present = channel == OutputChannel::fields;
    values.particles_present = channel == OutputChannel::particles;
    return values;
  };
  OwnedOutputSnapshot original;
  fs::create_directories(directory);
  {
    AsyncOutputWriter initial;
    configure(initial, directory);
    original = capture();
    auto values = original;
    initial.submit(std::move(values));
    initial.host_wait_piece();
    initial.complete_snapshot();
    require((channel == OutputChannel::fields ? initial.particles_path() : initial.fields_path()).empty(),
            "disabled channel unexpectedly exposes a path");
  }
  {
    AsyncOutputWriter occupied;
    configure(occupied, directory);
    occupied.submit(capture());
    rejects([&] { occupied.host_wait_piece(); }, "dataset writer overwrote an existing piece without permission");
  }
  read(backend, directory / backend.piece(channel, 0, 0), OutputSnapshotView(original).select(channel), 0);
  {
    AsyncOutputWriter replacement;
    configure(replacement, directory, true);
    auto values = capture();
    if (channel == OutputChannel::fields) values.arrays.front().values.front() = 123456.25;
    else {
      // The replacement borrows records with a distinct charge.
      static tpic::test::ParticleRecordStorage replaced;
      replaced = {};
      replaced.add(domain(0), {0, 1, 0}, {0.5, 0.5, 0.5}, {1.0, 2.0, 3.0}, 123456.25, 42);
      values.particles.front() = replaced.records(0);
    }
    const auto expected = values;
    replacement.submit(std::move(values));
    replacement.host_wait_piece();
    replacement.complete_snapshot();
    read(backend, directory / backend.piece(channel, 0, 0), OutputSnapshotView(expected).select(channel), 0);
  }
  const auto link_directory = directory / "symlink";
  fs::create_directory(link_directory);
  const auto protected_file = link_directory / "protected";
  { std::ofstream stream(protected_file); stream << "preserved"; }
  fs::create_symlink(protected_file, link_directory / backend.piece(channel, 0, 0));
  {
    AsyncOutputWriter writer;
    configure(writer, link_directory, true);
    writer.submit(capture());
    rejects([&] { writer.host_wait_piece(); }, "dataset overwrite followed a symlink");
  }
  require(file_contents(protected_file) == "preserved", "dataset overwrite modified a symlink target");
  const auto manifest_directory = directory / "occupied-manifest";
  fs::create_directory(manifest_directory);
  const auto protected_manifest = manifest_directory / backend.manifest(channel, 0);
  { std::ofstream stream(protected_manifest); stream << "preserved manifest"; }
  {
    AsyncOutputWriter writer;
    configure(writer, manifest_directory);
    writer.submit(capture());
    writer.host_wait_piece();
    rejects([&] { writer.complete_snapshot(); }, "dataset publication overwrote an existing manifest");
  }
  require(file_contents(protected_manifest) == "preserved manifest", "dataset publication modified an existing manifest");
  const auto wrong_type_directory = directory / "wrong-type";
  fs::create_directory(wrong_type_directory);
  const auto wrong_type_piece = wrong_type_directory / backend.piece(channel, 0, 0);
  fs::path wrong_type_marker;
  if (backend.extension == ".bp") wrong_type_marker = wrong_type_piece;
  else { fs::create_directory(wrong_type_piece); wrong_type_marker = wrong_type_piece / "protected"; }
  { std::ofstream stream(wrong_type_marker); stream << "preserved file type"; }
  {
    AsyncOutputWriter writer;
    configure(writer, wrong_type_directory, true);
    writer.submit(capture());
    rejects([&] { writer.host_wait_piece(); }, "dataset overwrite replaced a different file type");
  }
  require(file_contents(wrong_type_marker) == "preserved file type", "dataset overwrite destroyed a different file type");
  for (const auto& entry : fs::recursive_directory_iterator(directory))
    require(!entry.path().filename().string().starts_with(".tpic-output-"), "dataset staging leaked a directory");
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc <= 2, "usage: dataset_engine_tests [new-output-directory]");
    TemporaryDirectory temporary(argc == 2 ? fs::path(argv[1]) : fs::path{});
    std::vector<Backend> backends{{OutputEngineKind::Vtk, "vtk", ""}};
#if TPIC_HAS_ADIOS2
    backends.push_back({OutputEngineKind::Adios2, "adios2", ".bp"});
#endif
#if TPIC_HAS_HDF5
    backends.push_back({OutputEngineKind::Hdf5, "hdf5", ".h5"});
#endif
    for (const auto& fields : backends)
      if (fields.engine != OutputEngineKind::Vtk)
        for (const auto& particles : backends)
          exercise(temporary.path / (fields.name + "-" + particles.name), fields, particles);
#if TPIC_HAS_ADIOS2
    adios_field_selection(temporary.path / "adios2-vector", true);
    adios_field_selection(temporary.path / "adios2-scalar", false);
    if (argc == 1) adios_deferred_payloads();
#endif
    for (const auto& backend : backends)
      if (backend.engine != OutputEngineKind::Vtk)
        for (const auto channel : channels)
          preservation(temporary.path / (backend.name + "-preservation-" + channel_name(channel)), backend, channel);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
