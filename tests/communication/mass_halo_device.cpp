#include <tpic/config/simulation_parameters.hpp>
#include <tpic/config/domain.hpp>
// Compact-record integration against a global-coordinate host oracle. Every
// rank deposits the corner pairs of its own cells only, as the deposition
// does, so the additive phase may drop the stencils a rank never deposits.
// The topologies cover remote, coalesced, self and absent directions of every
// kind on 8 and 12 ranks; other counts run an x decomposition.
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <tpic/communication/mass_halo/mass_halo_device.hpp>
#include <tpic/communication/transport_routes.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_stencil.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>

#include <tpic/config/precision.hpp>

#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
namespace d = lpfx::distributed;
namespace vc = tpic::communication;
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Event = lpfx::device::DeviceEvent<Backend>;
using Domain = tpic::config::PreparedDomain;
using Layout = tpic::data::MassMatrixLayout;
using Node = std::array<int, 3>;
constexpr int tensor = tpic::data::mass_tensor_entries;
int rank = 0, ranks = 1;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Topology {
  std::array<int, 3> ranks;
  std::array<bool, 3> periodic;
};

std::vector<Topology> topologies() {
  constexpr bool T = true, F = false;
  switch (ranks) {
  case 8: return {{{2, 2, 2}, {T, T, T}}, {{2, 2, 2}, {F, F, F}}, {{2, 2, 2}, {T, F, T}}};
  case 12: return {{{3, 2, 2}, {T, T, T}}, {{3, 2, 2}, {F, T, F}}, {{1, 3, 4}, {T, T, T}},
                   {{2, 3, 2}, {F, F, F}}};
  default: return {{{ranks, 1, 1}, {T, T, T}}, {{ranks, 1, 1}, {F, F, F}}};
  }
}

// Two or three cells per rank and axis: every split is uneven.
Domain fixture(int source_rank, const Topology& t) {
  const auto cells = [&](int axis) { return 2 * t.ranks[axis] + 1; };
  tpic::config::GridConfig grid{double(cells(0)), double(cells(1)), double(cells(2)),
                                cells(0), cells(1), cells(2)};
  tpic::config::TopologyConfig topology;
  topology.xlen = t.ranks[0];
  topology.ylen = t.ranks[1];
  topology.zlen = t.ranks[2];
  topology.periodic_x = t.periodic[0];
  topology.periodic_y = t.periodic[1];
  topology.periodic_z = t.periodic[2];
  // FIELD routing must be independent from particle boundary configuration.
  topology.periodic_x_particles = !t.periodic[0];
  topology.periodic_y_particles = !t.periodic[1];
  topology.periodic_z_particles = !t.periodic[2];
  return tpic::config::prepare_domain(grid, topology, source_rank, ranks);
}

// One corner pair of one global cell. Small integers keep every sum exact in
// float and double, whatever its order.
template <class Scalar> Scalar contribution(const Node& cell, int pair, int component, int epoch) {
  return Scalar(1 + (cell[0] + 3 * cell[1] + 7 * cell[2] + 11 * pair + 13 * component + 17 * epoch) % 29);
}

// Every corner pair a cell deposits: the canonical owner corner and stencil.
template <class Visit> void for_each_pair(Visit visit) {
  for (int a = 0; a != tpic::ecsim::cic_corner_count; ++a)
    for (int b = 0; b != tpic::ecsim::cic_corner_count; ++b)
      if (const int stencil = tpic::ecsim::mass_pair_stencil(a, b); stencil >= 0)
        visit(tpic::ecsim::cic_corner({}, a), a * tpic::ecsim::cic_corner_count + b, stencil);
}

// A local node in global coordinates; false outside a physical wall.
bool global_node(const Domain& domain, const Topology& t, Node local, Node& global) {
  for (int axis = 0; axis != 3; ++axis) {
    const int n = domain.global_cells[axis];
    global[axis] = local[axis] + domain.global_offset[axis];
    if (t.periodic[axis]) global[axis] = (global[axis] % n + n) % n;
    else if (global[axis] < 0 || global[axis] > n) return false;
  }
  return true;
}

constexpr int poison = -1;

template <class Scalar>
std::vector<Scalar> initial_mass(const Domain& domain, int epoch) {
  const auto layout = tpic::data::prepare_mass_matrix_layout(domain);
  std::vector<Scalar> result(layout.storage_size(), Scalar(poison));
  for (int k = 0; k != layout.nz; ++k)
    for (int j = 0; j != layout.ny; ++j)
      for (int i = 0; i != layout.nx; ++i)
        for (int s = 0; s != tpic::data::mass_stencil_count; ++s)
          for (int c = 0; c != tensor; ++c)
            result[layout.entry_index(layout.vertex_index(i, j, k), s) + c] = Scalar(0);
  for (int k = 0; k + 1 != layout.nz; ++k)
    for (int j = 0; j + 1 != layout.ny; ++j)
      for (int i = 0; i + 1 != layout.nx; ++i) {
        const Node cell{i + domain.global_offset[0], j + domain.global_offset[1], k + domain.global_offset[2]};
        for_each_pair([&](const tpic::data::GridIndex corner, int pair, int stencil) {
          for (int c = 0; c != tensor; ++c)
            result[layout.entry_index(layout.vertex_index(i + corner.i, j + corner.j, k + corner.k), stencil) + c] +=
                contribution<Scalar>(cell, pair, c, epoch);
        });
      }
  return result;
}

// The oracle uses no halo direction, buffer or LPFx matching: it sums every
// global cell's pairs at their global nodes, then reads that sum at each local
// interior record and at each ghost record a neighbour sends. Everything else
// keeps its uploaded value.
template <class Scalar>
std::vector<Scalar> expected_mass(const Domain& local, const Topology& t, int epoch) {
  std::map<std::array<int, 5>, Scalar> assembled;
  for (int k = 0; k != local.global_cells[2]; ++k)
    for (int j = 0; j != local.global_cells[1]; ++j)
      for (int i = 0; i != local.global_cells[0]; ++i)
        for_each_pair([&](const tpic::data::GridIndex corner, int pair, int stencil) {
          Node node{i + corner.i, j + corner.j, k + corner.k};
          for (int axis = 0; axis != 3; ++axis)
            if (t.periodic[axis]) node[axis] %= local.global_cells[axis];
          for (int c = 0; c != tensor; ++c)
            assembled[{node[0], node[1], node[2], stencil, c}] += contribution<Scalar>({i, j, k}, pair, c, epoch);
        });
  const auto layout = tpic::data::prepare_mass_matrix_layout(local);
  const auto& topology = local.field_neighborhood.topology();
  const int n[]{layout.nx, layout.ny, layout.nz};
  const auto side = [&](int at, int axis) { return std::int8_t(at < 0 ? -1 : at >= n[axis] ? 1 : 0); };
  auto result = initial_mass<Scalar>(local, epoch);
  for (int k = -1; k <= layout.nz; ++k)
    for (int j = -1; j <= layout.ny; ++j)
      for (int i = -1; i <= layout.nx; ++i) {
        Node global;
        if (!global_node(local, t, {i, j, k}, global)) continue;
        const lpfx::CartesianDirection3 toward{side(i, 0), side(j, 1), side(k, 2)};
        const bool ghost = toward.x || toward.y || toward.z;
        if (ghost && !topology.neighbor_by_index(lpfx::cartesian_neighbor_index(toward)).has_peer()) continue;
        const auto sent = ghost ? vc::mass_halo_stencil_mask(false, lpfx::opposite_cartesian_direction(toward))
                                : std::uint16_t((1U << tpic::data::mass_stencil_count) - 1U);
        for (int s = 0; s != tpic::data::mass_stencil_count; ++s) {
          if (!(sent >> s & 1U)) continue;
          for (int c = 0; c != tensor; ++c) {
            const auto found = assembled.find({global[0], global[1], global[2], s, c});
            result[layout.entry_index(layout.vertex_index(i, j, k), s) + c] =
                found == assembled.end() ? Scalar(0) : found->second;
          }
        }
      }
  return result;
}

bool interior(const Layout& layout, std::size_t entry) {
  const auto vertex = entry / tpic::data::mass_entries_per_node;
  const int i = int(vertex % layout.extent_x) - 1, j = int(vertex / layout.extent_x % layout.extent_y) - 1,
            k = int(vertex / layout.stride_z) - 1;
  return i >= 0 && i < layout.nx && j >= 0 && j < layout.ny && k >= 0 && k < layout.nz;
}

template<class Precision>
void run(const d::CommunicationSetup& setup,
         lpfx::HostMemoryBudget& registration_budget,
         lpfx::HostMemoryBudget& operation_budget, Executor executor,
         Executor consumer, const Topology& t) {
  using Scalar = typename Precision::Storage::Mass;
  using Storage = vc::MassHaloStorage<Precision, Backend>;
  using Plan = typename Storage::plan_type;
  const auto domain = fixture(rank, t);
  tpic::data::CompactMassMatrix<Precision, Backend> mass(domain);
  const auto initial_registration_bytes = registration_budget.used();
  const auto initial_operation_bytes = operation_budget.used();
  Storage storage(domain, mass);
  storage.prepare(setup, executor);

  auto add_binding = d::bind<vc::DeviceRoute>(
      storage.plan().template phase<true>().message_plan(), storage.template buffers<true>(), executor);
  auto ghost_binding = d::bind<vc::DeviceRoute>(
      storage.plan().template phase<false>().message_plan(), storage.template buffers<false>(), executor);
  d::Communication<decltype(add_binding), decltype(ghost_binding)> communication;
  require(d::try_prepare_communication(
              communication, setup, registration_budget, operation_budget,
              std::move(add_binding), std::move(ghost_binding)) == lpfx::ErrorCode::success,
          "mass halo test: registered communication preparation failed");
  auto& add = communication.template get<typename Plan::template phase_type<true>::definition_type>();
  auto& ghosts = communication.template get<typename Plan::template phase_type<false>::definition_type>();
  require(storage.quiescent() && add.reusable() && ghosts.reusable(),
          "mass halo test: prepared port is not reusable");

  using Staging = lpfx::device::AlignedPinnedHostAllocation<Backend, alignof(Scalar)>;
  const auto layout = storage.plan().layout();
  const auto values = layout.storage_size();
  const auto bytes = values * sizeof(Scalar);
  Staging staging;
  require(Staging::create(3 * bytes, staging) == lpfx::ErrorCode::success,
          "mass halo test: pinned transfer staging allocation failed");
  auto* upload = reinterpret_cast<Scalar*>(staging.raw_base());
  auto* actual = upload + values;
  auto* at_interior = actual + values;
  Event completed, interior_ready;
  require(Traits::succeeded(Event::create(completed)) && Traits::succeeded(Event::create(interior_ready)),
          "mass halo test: event creation failed");
  // Assembles once and returns how often progress reported the interior.
  const auto assemble = [&] {
    const auto add_generation = add.generation(), ghost_generation = ghosts.generation();
    storage.begin(add);
    int interiors = 0;
    for (vc::HaloProgress progress; (progress = storage.progress(add, ghosts)) != vc::HaloProgress::complete;) {
      if (progress != vc::HaloProgress::interior) {
        std::this_thread::yield();
        continue;
      }
      // The interior records are final in stream order from here on, while
      // the ghost phase may still be in flight.
      ++interiors;
      require(Traits::succeeded(interior_ready.record(executor)) &&
                  Traits::succeeded(interior_ready.wait_on(consumer)) &&
                  Traits::succeeded(Traits::copy_device_to_host_async(at_interior, mass.data(), bytes,
                                                                      consumer.stream())),
              "mass halo test: interior snapshot failed");
    }
    storage.conclude();
    require(add.generation() == add_generation + 1 && ghosts.generation() == ghost_generation + 1,
            "mass halo test: one assembly did not consume one generation per phase");
    require(Traits::succeeded(completed.record(executor)) &&
                Traits::succeeded(completed.wait_on(consumer)) &&
                Traits::succeeded(Traits::copy_device_to_host_async(actual, mass.data(), bytes, consumer.stream())) &&
                Traits::succeeded(Traits::synchronize_stream(consumer.stream())),
            "mass halo test: consumer completion failed");
    return interiors;
  };
  const auto retained_bytes = storage.storage_bytes();
  const auto retained_registration_bytes = registration_budget.used();
  const auto retained_operation_bytes = operation_budget.used();
  require(retained_bytes == storage.plan().device_bytes(), "mass halo test: device storage differs from the plan");
  for (int epoch = 0; epoch != 3; ++epoch) {
    const auto initial = initial_mass<Scalar>(domain, epoch);
    const auto expected = expected_mass<Scalar>(domain, t, epoch);
    std::memcpy(upload, initial.data(), bytes);
    require(Traits::succeeded(Traits::copy_host_to_device_async(mass.data(), upload,
        bytes, executor.stream())), "mass halo test: upload failed");
    // No producer host fence: each registered generation must order its pack,
    // and the test publishes the ghost scatter to its consumer explicitly.
    require(assemble() == 1, "mass halo test: the interior was not reported once");
    for (std::size_t i = 0; i != values; ++i) {
      const bool inside = interior(layout, i);
      if (actual[i] != expected[i] || (inside && at_interior[i] != expected[i])) {
        std::cerr << "rank " << rank << ", epoch " << epoch << ", entry " << i << (inside ? " (interior)" : " (ghost)")
                  << ": got " << actual[i] << " (" << at_interior[i] << " at the interior), expected "
                  << expected[i] << '\n';
        throw std::runtime_error("mass halo test: additive/ghost oracle mismatch");
      }
    }
    require(storage.storage_bytes() == retained_bytes &&
                registration_budget.used() == retained_registration_bytes &&
                operation_budget.used() == retained_operation_bytes,
            "mass halo test: repeated exchange changed retained storage");
  }

  // At rank zero's lower corner the canonical fold is L + 1 - L: the first
  // direction holding the corner adds +1, the second -L; on two ranks the
  // second is a self direction. Folding in another order yields 1. Each value
  // sits at stencil zero, which every additive side carries first, at the
  // start of its sender's opposite send region; a self direction's sender is
  // rank zero itself.
  const auto root_domain = fixture(0, t);
  const auto root = Plan::prepare(root_domain);
  const auto root_directions = root.template phase<true>().directions();
  std::vector<std::uint32_t> holding;
  for (std::uint32_t ordinal = 0; ordinal != root_directions.size(); ++ordinal) {
    const auto& start = root_directions[ordinal].receive.start;
    if (start[0] == 0 && start[1] == 0 && start[2] == 0) holding.push_back(ordinal);
  }
  if (t.periodic == std::array{true, true, true}) {
    std::memset(upload, 0, bytes);
    const Scalar large = std::ldexp(Scalar{1}, std::numeric_limits<Scalar>::digits);
    const auto place = [&](std::uint32_t ordinal, Scalar value) {
      const auto& direction = root_directions[ordinal];
      const int sender = direction.self ? 0
          : int(root_domain.field_neighborhood.topology().neighbor_by_index(direction.canonical_index).peer_rank());
      if (sender != rank) return;
      for (const auto& own : storage.plan().template phase<true>().directions())
        if (own.canonical_index == lpfx::opposite_cartesian_neighbor_index(direction.canonical_index))
          upload[layout.entry_index(layout.vertex_index(own.send.start[0], own.send.start[1], own.send.start[2]), 0)] =
              value;
    };
    const auto target = layout.entry_index(layout.vertex_index(0, 0, 0), 0);
    if (rank == 0) upload[target] = large;
    place(holding[0], Scalar{1});
    place(holding[1], -large);
    require(Traits::succeeded(Traits::copy_host_to_device_async(mass.data(), upload, bytes, executor.stream())),
            "mass halo test: order-sensitive upload failed");
    require(assemble() == 1, "mass halo test: order-sensitive exchange failed");
    if (rank == 0) {
      Scalar canonical = large;
      canonical += Scalar{1};
      canonical -= large;
      Scalar reordered = large;
      reordered -= large;
      reordered += Scalar{1};
      require(std::memcmp(actual + target, &canonical, sizeof(Scalar)) == 0 &&
                  std::memcmp(actual + target, &reordered, sizeof(Scalar)) != 0,
              "mass halo test: additive scatter changed canonical bit order");
    }
    require(storage.storage_bytes() == retained_bytes &&
                registration_budget.used() == retained_registration_bytes &&
                operation_budget.used() == retained_operation_bytes,
            "mass halo test: order-sensitive exchange changed retained storage");
  }
  require(storage.quiescent() && add.reusable() && ghosts.reusable() && communication.close().is_success(),
          "mass halo test: registered communication close failed");
  storage.close();
  require(!storage.quiescent() && storage.storage_bytes() == 0,
          "mass halo test: close retained device storage");
  require(registration_budget.used() == initial_registration_bytes &&
              operation_budget.used() == initial_operation_bytes,
          "mass halo test: close retained host storage");
  require(Traits::succeeded(completed.reset()) && Traits::succeeded(interior_ready.reset()),
          "mass halo test: event release failed");
}

template <class Precision>
void run_case(const lpfx::device::DeviceProperties &properties,
              const Executor executor, const Executor consumer,
              const Topology& topology) {
  // LPFx seals one registration composition per domain. Give every distinct
  // precision/topology fixture its own registration epoch.
  lpfx::MpiCallDomain mpi_domain;
  require(lpfx::try_make_mpi_call_domain(
              MPI_COMM_WORLD, lpfx::MpiProcessThreadUse::single_process_thread,
              mpi_domain)
              .succeeded(),
          "mass halo test: MPI call domain failed");
  d::CommunicationSetup setup;
  require(d::try_make_communication_setup(
              mpi_domain, properties, executor, setup)
              .succeeded(),
          "mass halo test: setup failed");
  lpfx::HostMemoryBudget registration_budget, operation_budget;
  require(registration_budget.try_set_limit(4 * 1024 * 1024) &&
              registration_budget.seal() &&
              operation_budget.try_set_limit(4 * 1024 * 1024) &&
              operation_budget.seal(),
          "mass halo test: budget failed");
  run<Precision>(setup, registration_budget, operation_budget, executor,
                 consumer, topology);
  require(registration_budget.used() == 0 && operation_budget.used() == 0,
          "mass halo test: transport retained host storage");
}
} // namespace

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  try {
    require(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS &&
        MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS &&
        MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS,
        "mass halo test: process setup failed");
    const int result = [&] {
      int visible = 0;
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
      const auto queried = cudaGetDeviceCount(&visible);
      const bool unavailable = queried == cudaErrorNoDevice || queried == cudaErrorInsufficientDriver ||
                               queried == cudaErrorSystemDriverMismatch;
#else
      const auto queried = hipGetDeviceCount(&visible);
      const bool unavailable = queried == hipErrorNoDevice || queried == hipErrorInsufficientDriver;
#endif
      const int local = Traits::succeeded(queried) && visible > 0 ? 0 :
          unavailable || (Traits::succeeded(queried) && visible == 0) ? 1 : 2;
      int global = 0;
      require(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX,
                            MPI_COMM_WORLD) == MPI_SUCCESS,
          "mass halo test: device agreement failed");
      if (global != 0) {
        if (rank == 0) std::cout << "TPIC compact mass halo GPU runtime unavailable\n";
        return global == 1 ? 77 : 1;
      }
      MPI_Comm local_communicator = MPI_COMM_NULL;
      int local_rank = 0, local_ranks = 0;
      require(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
          MPI_INFO_NULL, &local_communicator) == MPI_SUCCESS &&
          MPI_Comm_rank(local_communicator, &local_rank) == MPI_SUCCESS &&
          MPI_Comm_size(local_communicator, &local_ranks) == MPI_SUCCESS,
          "mass halo test: local rank query failed");
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
      require(cudaSetDevice(local_rank % visible) == cudaSuccess, "mass halo test: selection failed");
#else
      require(hipSetDevice(local_rank % visible) == hipSuccess, "mass halo test: selection failed");
#endif
      int duplicate = 0, any_duplicate = 0;
      if constexpr (vc::payload_backend_id != 0) {
        // Qualify this test's hardware before LPFx preparation. Per-rank
        // visibility masks may all expose ordinal zero on different GPUs;
        // only equal physical identities on the same node justify a skip.
        lpfx::device::DeviceIdentity identity;
        std::int32_t ordinal = -1;
        require(Traits::succeeded(Traits::query_current_device_identity(identity, ordinal)) &&
            identity.valid(), "mass halo test: physical GPU identity query failed");
        std::vector<lpfx::device::DeviceIdentity> identities(local_ranks);
        require(MPI_Allgather(&identity, int(sizeof(identity)), MPI_BYTE,
            identities.data(), int(sizeof(identity)), MPI_BYTE, local_communicator) == MPI_SUCCESS,
            "mass halo test: physical GPU identity collection failed");
        for (int i = 0; i != local_ranks; ++i)
          for (int j = i + 1; j != local_ranks; ++j)
            duplicate = duplicate || identities[i] == identities[j];
        require(MPI_Allreduce(&duplicate, &any_duplicate, 1, MPI_INT, MPI_MAX,
                              MPI_COMM_WORLD) == MPI_SUCCESS,
            "mass halo test: distinct GPU agreement failed");
      }
      require(MPI_Comm_free(&local_communicator) == MPI_SUCCESS,
              "mass halo test: local communicator close failed");
      if (any_duplicate) {
        if (rank == 0) std::cout << "TPIC compact mass halo requires distinct physical GPUs for "
                                << vc::payload_backend_name << '\n';
        return 77;
      }
      lpfx::device::DeviceProperties properties{};
      require(Traits::succeeded(Traits::query_current_device_properties(properties)),
              "mass halo test: device properties failed");
      auto stream = Traits::null_stream, consumer_stream = Traits::null_stream;
      require(Traits::succeeded(Traits::create_stream(stream)) &&
          Traits::succeeded(Traits::create_stream(consumer_stream)),
          "mass halo test: stream creation failed");
      const Executor executor{stream}, consumer{consumer_stream};
      for (const auto& topology : topologies()) {
        run_case<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(properties, executor, consumer,
                                        topology);
        run_case<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(properties, executor, consumer,
                                              topology);
      }
      require(Traits::succeeded(Traits::destroy_stream(consumer_stream)) &&
          Traits::succeeded(Traits::destroy_stream(stream)), "mass halo test: stream destruction failed");
      return 0;
    }();
    if (result == 0 && rank == 0)
      std::cout << "TPIC compact mass halo passed on " << ranks << " rank(s) via "
                << vc::payload_backend_name << '\n';
    return MPI_Finalize() == MPI_SUCCESS ? result : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
