// Native startup qualification using the build-selected LPFx device route.
#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/config/domain.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/api/distributed/communication.hpp>
#include <tpic/communication/transport_routes.hpp>
#include <tpic/initialization/fields/field_image.hpp>
#include <tpic/simulation/actions/magnetic_startup_device.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../support/test_builders.hpp"

namespace {

namespace distributed = lpfx::distributed;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Domain = tpic::config::PreparedDomain;
using Index = std::array<int, 3>;
int rank = 0, ranks = 1;

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
         error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<class Field, class Function>
void for_each_sample(Field& field, Function function) {
  const auto visit = [&]<std::size_t I>() {
    const auto component = field.view().template component<I>();
    const auto halo = component.layout().halo();
    const auto interior = component.layout().interior_extents();
    for (int k = -static_cast<int>(halo.z_minus);
         k < static_cast<int>(interior.z() + halo.z_plus); ++k)
      for (int j = -static_cast<int>(halo.y_minus);
           j < static_cast<int>(interior.y() + halo.y_plus); ++j)
        for (int i = -static_cast<int>(halo.x_minus);
             i < static_cast<int>(interior.x() + halo.x_plus); ++i)
          function(component(i, j, k), I, Index{i, j, k});
  };
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (visit.template operator()<I>(), ...);
  }(std::make_index_sequence<Field::component_count>{});
}

// Independently model the 26 disjoint halo regions. A region with
// any missing physical neighbour stays unchanged, including physical corners.
// No LPFx communication descriptors or TPIC stencil helpers enter this oracle.
struct CopySource {
  int peer;
  Index index;
  bool exterior;
  bool available;
};

CopySource copy_source(const std::vector<Domain>& domains,
                       const std::array<bool, 3>& periodic,
                       int owner, Index index, bool nodal) {
  const auto& domain = domains[owner];
  const Index processes{ranks, 1, 1};
  const auto owner_coordinate =
      domain.field_neighborhood.local_partition().process_coordinate();
  Index peer_coordinate{static_cast<int>(owner_coordinate.x()),
                        static_cast<int>(owner_coordinate.y()),
                        static_cast<int>(owner_coordinate.z())};
  Index global{};
  bool exterior = false, available = true;
  for (std::size_t axis = 0; axis != 3; ++axis) {
    global[axis] = domain.global_offset[axis] + index[axis];
    const int extent = domain.local_cells[axis] + static_cast<int>(nodal);
    const int direction = index[axis] < 0 ? -1 : index[axis] >= extent ? 1 : 0;
    exterior |= direction != 0;
    peer_coordinate[axis] += direction;
    if (peer_coordinate[axis] < 0 || peer_coordinate[axis] >= processes[axis]) {
      if (!periodic[axis]) {
        available = false;
      } else {
        peer_coordinate[axis] = (peer_coordinate[axis] + processes[axis]) % processes[axis];
        global[axis] -= direction * domain.global_cells[axis];
      }
    }
  }
  if (!available) return {owner, index, exterior, false};
  const int peer = peer_coordinate[0]; // These fixtures decompose only x.
  for (std::size_t axis = 0; axis != 3; ++axis)
    index[axis] = global[axis] - domains[peer].global_offset[axis];
  return {peer, index, exterior, true};
}

template<class Value>
Value initial_sample(const Domain& domain, int epoch,
                     std::size_t component, Index index) {
  const double x = domain.global_offset[0] + index[0] + 0.5;
  const double y = domain.global_offset[1] + index[1] + 0.5;
  const double z = domain.global_offset[2] + index[2] + 0.5;
  // Dyadic values make exact comparison valid for every tested storage type.
  // Rank stamps expose wrong peer selection; curvature exposes shifted stencils.
  return static_cast<Value>(1000 * epoch + 10000 * component +
      x * x + 3 * y * y + 5 * z * z + 7 * x * y +
      static_cast<double>(domain.field_neighborhood.topology().input().rank()) * 0.0625);
}

// After the halo copy, a closed run sets a ghost cell on a physical face to the
// adjacent cell, negated for PEC normal and PMC tangential B (upper faces PMC,
// lower faces PEC); an edge or corner ghost takes the product of its signs.
template<class Precision>
auto expected_cell(const std::vector<Domain>& domains, const std::array<bool, 3>& periodic, bool closed,
                   int owner, int epoch, std::size_t component, Index index) {
  const auto& domain = domains[owner];
  bool negate = false;
  for (std::size_t axis = 0; closed && axis != 3; ++axis) {
    const bool lower = index[axis] == -1 && domain.global_offset[axis] == 0;
    const bool upper = index[axis] == domain.local_cells[axis] &&
                       domain.global_offset[axis] + domain.local_cells[axis] == domain.global_cells[axis];
    if (periodic[axis] || !(lower || upper)) continue;
    index[axis] = lower ? 0 : index[axis] - 1;
    negate ^= (component == axis) != upper;
  }
  const auto source = copy_source(domains, periodic, owner, index, false);
  const auto value = initial_sample<typename Precision::Storage::Magnetic>(
      domains[source.peer], epoch, component, source.index);
  return negate ? -value : value;
}

template<class Precision>
auto expected_node(const std::vector<Domain>& domains,
                   const std::array<bool, 3>& periodic, bool closed,
                   int owner, int epoch, std::size_t component, Index index) {
  using Value = typename Precision::Storage::Magnetic;
  const auto source = copy_source(domains, periodic, owner, index, true);
  if (source.exterior && !source.available) return Value{0};
  double sum = 0;
  for (int dz = -1; dz != 1; ++dz)
    for (int dy = -1; dy != 1; ++dy)
      for (int dx = -1; dx != 1; ++dx)
        sum += expected_cell<Precision>(domains, periodic, closed, source.peer, epoch,
            component, {source.index[0] + dx, source.index[1] + dy,
                        source.index[2] + dz});
  return static_cast<Value>(sum / 8.0);
}

template<class Host, class Device, class Expected>
void download_and_compare(Host& host, Device& device, Executor executor,
                          Expected expected, const char* name) {
  for_each_sample(host, [](auto& value, auto...) { value = -99999; });
  auto prepared = lpfx::make_device_field_transfer(host.view(), device);
  require(prepared.has_value(), "magnetic startup: download preparation failed");
  auto transfer = std::move(prepared).value();
  const auto submitted = transfer.download(executor);
  require(submitted.is_success() && submitted.requires_conclusion(),
          "magnetic startup: download submission failed");
  require(transfer.wait().is_success() && !transfer.pending(),
          "magnetic startup: download completion failed");
  for_each_sample(host, [&](const auto& actual, std::size_t component, Index index) {
    const auto wanted = expected(component, index);
    if (actual != wanted) {
      std::ostringstream message;
      message << "magnetic startup: " << name << " rank " << rank
              << " component " << component << " sample (" << index[0] << ','
              << index[1] << ',' << index[2] << ") got " << actual
              << ", expected " << wanted;
      throw std::runtime_error(message.str());
    }
  });
}



// closed: the restart startup, which closes the B-cell wall ghosts before C2N.
template<class Precision>
void run(Executor executor, const lpfx::device::DeviceProperties& properties,
         std::array<bool, 3> periodic, bool closed) {
  using Fields = tpic::data::FieldSchema<Precision>;
  tpic::config::SimulationParameters parameters;
  parameters.case_name = "Uniform";
  parameters.grid.nxc = 3 * ranks + 1; // Unequal x partitions on multiple ranks.
  parameters.grid.nyc = 3;
  parameters.grid.nzc = 2;
  parameters.topology.xlen = ranks;
  parameters.topology.periodic_x = periodic[0];
  parameters.topology.periodic_y = periodic[1];
  parameters.topology.periodic_z = periodic[2];
  parameters.ns = 1;
  parameters.species.resize(1);
  parameters.species[0].q_over_m = -1;
  parameters.species[0].rho_init = 1;
  std::vector<Domain> domains;
  for (int peer = 0; peer != ranks; ++peer)
    domains.push_back(tpic::config::prepare_domain(
        parameters.grid, parameters.topology, peer, ranks));
  const auto& domain = domains[rank];
  tpic::config::PreparedPhysics physics; // Upper faces PMC, lower faces PEC.
  for (std::size_t axis = 0; axis != 3; ++axis)
    physics.field_boundary[2 * axis] = tpic::config::FieldBoundary::perfect_magnetic_conductor;
  const auto plans = tpic::ecsim::prepare_field_boundary_plans(domain, physics);
  auto host = tpic::test::make_host_field_image<Precision>(parameters, domain);
  auto cell = tpic::data::upload_field(host.magnetic_cell, executor);
  auto node = tpic::data::make_zero_field<typename Fields::BNode>(domain, executor);
  auto host_node_result = lpfx::field::make<typename Fields::BNode>(lpfx::field::options{
      .cells = {domain.local_cells[0], domain.local_cells[1], domain.local_cells[2]},
      .halo = Domain::field_halo});
  require(host_node_result.has_value(), "magnetic startup: host node allocation failed");
  auto host_node = std::move(host_node_result).value();
  require(cell.initialized() && node.initialized(),
          "magnetic startup: fields must be initialized before registration");

  lpfx::MpiCallDomain mpi_domain;
  require(lpfx::try_make_mpi_call_domain(MPI_COMM_WORLD,
      lpfx::MpiProcessThreadUse::single_process_thread, mpi_domain).succeeded(),
      "magnetic startup: MPI call domain failed");
  distributed::CommunicationSetup setup;
  require(distributed::try_make_communication_setup(
      mpi_domain, properties, executor, setup).succeeded(),
      "magnetic startup: communication setup failed");
  auto fields = distributed::fields(domain.field_neighborhood, distributed::uniform_halos)
                    .template on<tpic::communication::DeviceRoute>(executor);
  auto requests = distributed::operations(
      distributed::copy_ghosts<"B.cell">(cell, fields),
      distributed::copy_ghosts<"B.node">(node, fields));
  auto communication = distributed::make_communication(requests);
  require(distributed::prepare(communication, setup, std::move(requests),
              distributed::PreparationLimits{16 * 1024 * 1024, 16 * 1024 * 1024}) == lpfx::ErrorCode::success,
          "magnetic startup: registration failed");
  auto& cell_exchange = communication.template get<"B.cell">();
  auto& node_exchange = communication.template get<"B.node">();

  for (int epoch = 0; epoch != 2; ++epoch) {
    for_each_sample(host.magnetic_cell,
        [&](auto& value, std::size_t component, Index index) {
          value = initial_sample<typename Precision::Storage::Magnetic>(domain, epoch, component, index);
        });
    {
      auto prepared = lpfx::make_device_field_transfer(host.magnetic_cell.view(), cell);
      require(prepared.has_value(), "magnetic startup: stamped upload preparation failed");
      auto transfer = std::move(prepared).value();
      const auto submitted = transfer.upload(executor);
      require(submitted.is_success() && submitted.requires_conclusion(),
              "magnetic startup: stamped upload submission failed");
      require(transfer.wait().is_success() && !transfer.pending(),
              "magnetic startup: stamped upload completion failed");
    }
    tpic::simulation::complete_initial_magnetic_field<Precision>(
        cell, node, cell_exchange, node_exchange, executor, closed ? &plans : nullptr);
    require(Traits::succeeded(Traits::synchronize_stream(executor.stream())),
            "magnetic startup: device completion failed");
    require(cell_exchange.reusable() && node_exchange.reusable() &&
        cell_exchange.generation() == static_cast<unsigned>(epoch + 1) &&
        node_exchange.generation() == static_cast<unsigned>(epoch + 1),
        "magnetic startup: halo operations did not conclude reusable epochs");
    download_and_compare(host.magnetic_cell, cell, executor,
        [&](std::size_t component, Index index) {
          return expected_cell<Precision>(domains, periodic, closed, rank, epoch, component, index);
        }, "B-cell");
    download_and_compare(host_node, node, executor,
        [&](std::size_t component, Index index) {
          return expected_node<Precision>(domains, periodic, closed, rank, epoch, component, index);
        }, "B-node");
  }
  require(communication.close().is_success(), "magnetic startup: communication close failed");
}

template<class Precision>
void all_boundaries(Executor executor, const lpfx::device::DeviceProperties& properties) {
  run<Precision>(executor, properties, {true, true, true}, false);
  for (const bool closed : {false, true}) {
    run<Precision>(executor, properties, {true, false, true}, closed);
    run<Precision>(executor, properties, {false, false, false}, closed);
  }
}

} // namespace

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  try {
    require(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS,
            "magnetic startup: MPI error handler failed");
    require(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS &&
            MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS,
            "magnetic startup: MPI rank query failed");
    lpfx::device::DeviceProperties properties{};
    const auto queried = Traits::query_current_device_properties(properties);
    int local = Traits::succeeded(queried) ? 0 : runtime_unavailable(queried) ? 1 : 2;
    int collective = 0;
    require(MPI_Allreduce(&local, &collective, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS,
            "magnetic startup: device availability agreement failed");
    if (collective != 0) {
      if (rank == 0)
        std::cout << "TPIC native magnetic startup "
                  << (collective == 1 ? "skipped: no usable device/driver\n"
                                      : "failed: device query error\n");
      return MPI_Finalize() == MPI_SUCCESS ? (collective == 1 ? 77 : 1) : 1;
    }
    require(properties.device_ordinal >= 0, "magnetic startup: invalid device ordinal");
    auto stream = Traits::null_stream;
    require(Traits::succeeded(Traits::create_stream(stream)) && stream != Traits::null_stream,
            "magnetic startup: stream creation failed");
    const Executor executor{stream};
    all_boundaries<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor, properties);
    all_boundaries<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor, properties);
    all_boundaries<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor, properties);
    require(Traits::succeeded(Traits::destroy_stream(stream)),
            "magnetic startup: stream destruction failed");
    if (rank == 0)
      std::cout << "TPIC native magnetic startup passed on " << ranks << " rank(s)\n";
    return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "rank " << rank << ": " << failure.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
