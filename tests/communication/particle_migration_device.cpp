// GPU/MPI integration: real loading and boundary classification, unsorted and
// resident-sort migration, reader-protected append/growth, all-attribute
// preservation, split J/M numerical equivalence and repeated lease/event reuse.
#include <tpic/communication/particles/initial_particle_classification_device.hpp>
#include <tpic/communication/particles/particle_migration_policy.hpp>
#include <tpic/communication/transport_routes.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/mass_matrix/compact_mass_matrix_device.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/initialization/particles/particle_loading_device.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/ecsim/deposition/current_deposition_device.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp>
#include <tpic/ecsim/deposition/moment_field_device.hpp>
#include <tpic/ecsim/particles/particle_sort_device.hpp>
#include <tpic/device/device_validation.hpp>
#include <tpic/simulation/execution/execution_streams.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include "../support/test_builders.hpp"

namespace {
namespace d = lpfx::distributed;
namespace vc = tpic::communication;
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
int rank = 0, ranks = 1;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

tpic::simulation::PreparedInput fixture(int source_rank) {
  tpic::config::SimulationParameters parameters;
  parameters.case_name = "Uniform";
  parameters.dt = 0.1;
  parameters.ncycles = 1;
  parameters.grid = {double(2 * ranks + 1), 2, 2, 2 * ranks + 1, 2, 2};
  parameters.topology.xlen = ranks;
  parameters.topology.periodic_y_particles = false;
  parameters.topology.periodic_z_particles = false;
  parameters.output = {.field_cycle = 0, .particles_cycle = 0,
      .test_particles_cycle = 0, .restart_cycle = 0, .diagnostics_cycle = 0};
  parameters.ns = 1;
  parameters.species.resize(1);
  auto& species = parameters.species[0];
  species.q_over_m = -1;
  species.rho_init = 1;
  species.npcelx = species.npcely = species.npcelz = 2;
  return tpic::simulation::prepare_input(std::move(parameters),
      {.rank = source_rank, .communicator_size = ranks,
       .order = tpic::config::ParticleOrder::unsorted});
}

// Controlled post-load state. Most rank-zero records migrate to its neighbours;
// on two ranks the receiving owner must grow beyond its loading capacity.
template <class Precision, class View>
__global__ void set_particle_crossings_kernel(View particles, tpic::data::ParticleCellMap cells,
                             int source_rank) {
  using S = tpic::data::ParticleSchema<Precision>;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= particles.size()) return;
  const auto id = particles.template get<typename S::Id>()[index];
  const int x = source_rank == 0 && index != 0
      ? (index % 2 ? cells.nx : -1) : int(index % cells.nx);
  const int y = id % 2 == 0 ? -1 : 0;
  const int z = id % 3 == 0 ? cells.nz : 0;
  particles.template get<typename S::Cell>()[index] = cells.encode(x, y, z);
  particles.template get<typename S::Rx>()[index] = 0.125;
  particles.template get<typename S::Ry>()[index] = 0.25;
  particles.template get<typename S::Rz>()[index] = 0.375;
  particles.template get<typename S::U>()[index] = double(id % 17) * 0.125;
  particles.template get<typename S::V>()[index] = 0.5;
  particles.template get<typename S::W>()[index] = -0.75;
}

struct Expected {
  int cell;
  double ry, rz, u, v, w, q;
};

auto expected_records(const tpic::simulation::PreparedInput& local) {
  std::map<std::uint64_t, Expected> expected;
  const auto cells = tpic::data::prepare_particle_cell_map(local.domain);
  for (int source_rank = 0; source_rank != ranks; ++source_rank) {
    const auto input = fixture(source_rank);
    const auto plan = tpic::test::prepare_particle_load(input, 0);
    for (unsigned index = 0; index != plan.count; ++index) {
      const auto id = plan.first_id + index;
      const int x = source_rank == 0 && index != 0
          ? (index % 2 ? plan.cells.nx : -1) : int(index % plan.cells.nx);
      const int global_x = (input.domain.global_offset[0] + x +
          input.domain.global_cells[0]) % input.domain.global_cells[0];
      const int local_x = global_x - local.domain.global_offset[0];
      if (local_x < 0 || local_x >= cells.nx) continue;
      expected.emplace(id, Expected{
          cells.encode(local_x, 0, id % 3 == 0 ? cells.nz - 1 : 0),
          id % 2 == 0 ? 0.75 : 0.25, id % 3 == 0 ? 0.625 : 0.375,
          double(id % 17) * 0.125, id % 2 == 0 ? -0.5 : 0.5,
          id % 3 == 0 ? 0.75 : -0.75, plan.q_factor * plan.density});
    }
  }
  return expected;
}

template<class Attribute, class Owner>
auto download(const Owner& owner, Executor executor) {
  std::vector<typename Attribute::value_type> result(owner.size());
  if (!result.empty()) {
    require(Traits::succeeded(Traits::copy_device_to_host_async(result.data(),
        owner.cview().template get<Attribute>(), result.size() * sizeof(result[0]),
        executor.stream())), "migration smoke: validation download failed");
    require(Traits::succeeded(Traits::synchronize_stream(executor.stream())),
            "migration smoke: validation download completion failed");
  }
  return result;
}

template<class Precision, class Owner>
void verify(const Owner& owner, const tpic::simulation::PreparedInput& input, Executor executor) {
  using S = tpic::data::ParticleSchema<Precision>;
  auto expected = expected_records(input);
  require(owner.size() == expected.size(), "migration smoke: wrong local population");
  const auto id = download<typename S::Id>(owner, executor);
  const auto cell = download<typename S::Cell>(owner, executor);
  const auto rx = download<typename S::Rx>(owner, executor);
  const auto ry = download<typename S::Ry>(owner, executor);
  const auto rz = download<typename S::Rz>(owner, executor);
  const auto u = download<typename S::U>(owner, executor);
  const auto v = download<typename S::V>(owner, executor);
  const auto w = download<typename S::W>(owner, executor);
  const auto q = download<typename S::Charge>(owner, executor);
  for (unsigned i = 0; i != owner.size(); ++i) {
    const auto found = expected.find(id[i]);
    require(found != expected.end(), "migration smoke: lost, duplicated or misrouted identity");
    const auto& e = found->second;
    require(cell[i] == e.cell && rx[i] == 0.125 && ry[i] == e.ry && rz[i] == e.rz &&
            u[i] == e.u && v[i] == e.v && w[i] == e.w && q[i] == e.q,
            "migration smoke: incorrect rebuilt cell, reflection or persistent attribute");
    expected.erase(found);
  }
  require(expected.empty(), "migration smoke: missing records");
}

template<class Precision>
void run(Executor executor, const lpfx::device::DeviceProperties& properties) {
  const auto input = fixture(rank);
  require(input.host_admitted(), "migration smoke: input rejected");
  const auto load = tpic::test::prepare_particle_load(input, 0);
  auto particles = tpic::initialization::make_initial_particles<Precision>(load, input.domain, input.initial_case, 0, executor);
  const auto initial_capacity = particles.capacity();
  const auto geometry = vc::prepare_particle_migration_geometry(input.domain, input.physics);
  set_particle_crossings_kernel<Precision><<<(particles.size() + 255) / 256, 256, 0, executor.stream()>>>(
      particles.view(), load.cells, rank);
  require(Traits::succeeded(Traits::take_last_launch_error()) &&
          Traits::succeeded(Traits::synchronize_stream(executor.stream())),
          "migration smoke: crossing fixture failed");
  lpfx::MpiCallDomain mpi_domain;
  require(lpfx::try_make_mpi_call_domain(MPI_COMM_WORLD,
      lpfx::MpiProcessThreadUse::single_process_thread, mpi_domain).succeeded(),
      "migration smoke: MPI domain failed");
  d::CommunicationSetup setup;
  require(d::try_make_communication_setup(mpi_domain,
      properties, executor, setup).succeeded(), "migration smoke: setup failed");
  lpfx::DeviceParticleClassificationSlots<Backend> slots;
  auto requests = d::operations(d::migrate<"particles", 0>(
      particles, input.domain.particle_topology, vc::particle_migration_policy<Precision>(geometry),
      d::on<vc::DeviceRoute>(executor), d::using_flags(slots),
      d::traffic_seed(1, d::arrivals_up_to_owner), d::resident_access<false>));
  auto communication = d::make_communication(requests);
  require(d::prepare(communication, setup, std::move(requests),
              d::PreparationLimits{16 * 1024 * 1024, 16 * 1024 * 1024}) == lpfx::ErrorCode::success,
          "migration smoke: registration failed");
  auto& migration = communication.template get<"particles", 0>();
  for (int epoch = 0; epoch != 2; ++epoch) {
    require(slots.reserve(particles.capacity()) == lpfx::ErrorCode::success,
            "migration smoke: classification reservation failed");
    require(vc::enqueue_initial_particle_classification<Precision>(
        migration, particles, geometry, executor) == lpfx::ErrorCode::success,
        "migration smoke: producer publication failed");
    require(migration.begin(executor).requires_conclusion() &&
            migration.template wait<lpfx::distributed::ParticleMigrationStage::prepare>().is_success() &&
            migration.template wait<lpfx::distributed::ParticleMigrationStage::resident>().is_success() &&
            migration.template wait<lpfx::distributed::ParticleMigrationStage::route>().is_success() &&
            migration.template wait<lpfx::distributed::ParticleMigrationStage::finalize>().is_success() &&
            migration.conclude().is_success(), "migration smoke: stages failed");
    require(slots.reusable() && migration.reusable(), "migration smoke: handoff not retired");
    verify<Precision>(particles, input, executor);
  }
  if (ranks == 2 && rank == 1)
    require(particles.capacity() > initial_capacity, "migration smoke: growth was not exercised");
  require(communication.close().is_success(),
          "migration smoke: close retained operation storage");
}

// Hold one real quantity reader after routing has finished. The callback never
// calls a device API. Finalize must query its registered event and return pending
// before considering either a fitting append or allocation replacement.
class HeldMassReader {
public:
  ~HeldMassReader() {
    release();
    if (executor_.bound() && !Traits::succeeded(Traits::synchronize_stream(executor_.stream())))
      std::abort();
  }
  void block(Executor executor) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
    const auto error = cudaLaunchHostFunc(executor.stream(), wait, &released_);
#else
    const auto error = hipLaunchHostFunc(executor.stream(), wait, &released_);
#endif
    require(Traits::succeeded(error), "resident deposition: mass hold submission failed");
    executor_ = executor;
  }
  void release() noexcept { released_.store(true, std::memory_order_release); }
private:
  static void wait(void* state) {
    const auto& released = *static_cast<std::atomic<bool>*>(state);
    while (!released.load(std::memory_order_acquire)) std::this_thread::yield();
  }
  std::atomic<bool> released_{};
  Executor executor_{Traits::null_stream};
};

template <class Precision, class View>
__global__ void empty_rank_zero_residents_kernel(View particles, tpic::data::ParticleCellMap cells) {
  using S = tpic::data::ParticleSchema<Precision>;
  if (threadIdx.x == 0 && blockIdx.x == 0 && particles.size() != 0)
    particles.template get<typename S::Cell>()[0] = cells.encode(cells.nx, 0, 0);
}

template <class View>
__global__ void initialize_resident_magnetic_kernel(View field, int nx, int ny, int nz) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= unsigned((nx + 1) * (ny + 1) * (nz + 1))) return;
  const int i = index % (nx + 1), j = (index / (nx + 1)) % (ny + 1),
            k = index / ((nx + 1) * (ny + 1));
  field.template component<0>()(i,j,k) = .25 + .03125 * (i + j);
  field.template component<1>()(i,j,k) = -.125 + .0625 * (j + k);
  field.template component<2>()(i,j,k) = .5 + .015625 * (i + k);
}

template <class View>
__global__ void compare_resident_currents_kernel(View unsorted, View split, int nx, int ny, int nz,
                                          unsigned* invalid, double tolerance) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= unsigned((nx + 1) * (ny + 1) * (nz + 1))) return;
  const int i = index % (nx + 1), j = (index / (nx + 1)) % (ny + 1),
            k = index / ((nx + 1) * (ny + 1));
  [&]<std::size_t... Axis>(std::index_sequence<Axis...>) {
    const auto compare = [&](auto axis) {
      const double a = unsorted.template component<decltype(axis)::value>()(i,j,k);
      const double b = split.template component<decltype(axis)::value>()(i,j,k);
      if (!::isfinite(a) || !::isfinite(b) || ::fabs(a - b) > tolerance * (1 + ::fabs(a)))
        atomicExch(invalid, 1U);
    };
    (compare(std::integral_constant<std::size_t, Axis>{}), ...);
  }(std::make_index_sequence<3>{});
}

template <class T>
auto download_values(const T* source, std::size_t size, Executor executor) {
  std::vector<T> values(size);
  require(Traits::succeeded(Traits::copy_device_to_host_async(
              values.data(), source, size * sizeof(T), executor.stream())) &&
              Traits::succeeded(Traits::synchronize_stream(executor.stream())),
          "resident deposition: numerical download failed");
  return values;
}

template <class Precision>
void run_resident_deposition(Executor main, const lpfx::device::DeviceProperties& properties,
                             bool reserve_fitting, bool no_rank_zero_residents) {
  namespace ecsim = tpic::ecsim;
  using F = tpic::data::FieldSchema<Precision>;
  using Event = lpfx::device::DeviceEvent<Backend>;
  using Stage = d::ParticleMigrationStage;
  const auto input = fixture(rank);
  auto load = tpic::test::prepare_particle_load(input, 0);
  if (reserve_fitting) load.capacity *= 4;
  auto particles = tpic::initialization::make_initial_particles<Precision>(load, input.domain, input.initial_case, 0, main);
  const auto initial_capacity = particles.capacity();
  const auto geometry = vc::prepare_particle_migration_geometry(input.domain, input.physics);
  set_particle_crossings_kernel<Precision><<<(particles.size() + 255) / 256, 256, 0, main.stream()>>>(
      particles.view(), load.cells, rank);
  if (rank == 0 && no_rank_zero_residents)
    empty_rank_zero_residents_kernel<Precision><<<1, 1, 0, main.stream()>>>(particles.view(), load.cells);
  require(Traits::succeeded(Traits::take_last_launch_error()) &&
              Traits::succeeded(Traits::synchronize_stream(main.stream())),
          "resident deposition: crossing fixture failed");

  tpic::simulation::ExecutionStreams<Backend> streams(properties.device_ordinal);
  (void)streams.admit(main, "main");
  const auto local = streams.create("local");
  const auto sort = streams.create("sort");
  const auto route = streams.create("route", true);
  const auto current = streams.create("current");
  const auto mass = streams.create("mass");
  streams.seal();
  ecsim::PartialParticleSort<Precision, Backend> ordering;
  ordering.initialize(particles, input.domain, properties, sort);
  auto magnetic = tpic::data::make_zero_field<typename F::BNode>(input.domain, main);
  auto split_current = tpic::data::make_zero_field<typename F::Current>(input.domain, main);
  auto unsorted_current = tpic::data::make_zero_field<typename F::Current>(input.domain, main);
  tpic::data::CompactMassMatrix<Precision, Backend> split_mass(input.domain), unsorted_mass(input.domain);
  const auto cells = load.cells;
  const auto nodes = unsigned((cells.nx + 1) * (cells.ny + 1) * (cells.nz + 1));
  initialize_resident_magnetic_kernel<<<(nodes + 127) / 128, 128, 0, main.stream()>>>(
      magnetic.view(), cells.nx, cells.ny, cells.nz);
  require(Traits::succeeded(Traits::take_last_launch_error()),
          "resident deposition: magnetic fixture launch failed");
  Event initialized, current_reader, mass_reader, current_done, mass_done, tail_ready;
  for (auto* event : {&initialized, &current_reader, &mass_reader, &current_done, &mass_done, &tail_ready})
    require(Traits::succeeded(Event::create(*event)), "resident deposition: event creation failed");
  require(Traits::succeeded(initialized.record(main)) &&
              Traits::succeeded(initialized.wait_on(current)) && Traits::succeeded(initialized.wait_on(mass)),
          "resident deposition: initialization dependency failed");
  tpic::device::DeviceValidation<Backend> current_validation, mass_validation, unsorted_validation;
  const auto deposition_cells = tpic::data::prepare_particle_cell_map(input.domain);

  lpfx::MpiCallDomain mpi_domain;
  require(lpfx::try_make_mpi_call_domain(MPI_COMM_WORLD,
              lpfx::MpiProcessThreadUse::single_process_thread, mpi_domain).succeeded(),
          "resident deposition: MPI domain failed");
  d::CommunicationSetup setup;
  require(d::try_make_communication_setup(mpi_domain,
              properties, main, setup).succeeded(), "resident deposition: communication setup failed");
  lpfx::DeviceParticleClassificationSlots<Backend> slots;
  auto request = d::operations(d::migrate<"particles", 0>(particles, input.domain.particle_topology,
      vc::particle_migration_policy<Precision>(geometry),
      d::on<vc::DeviceRoute>(route), d::using_flags(slots),
      d::traffic_seed(1, d::arrivals_up_to_owner), d::resident_access<true>));
  auto communication = d::make_communication(request);
  require(d::prepare(communication, setup, std::move(request),
              d::PreparationLimits{16 * 1024 * 1024, 16 * 1024 * 1024}) == lpfx::ErrorCode::success,
          "resident deposition: registration failed");
  auto& migration = communication.template get<"particles", 0>();
  const double tolerance = std::is_same_v<typename Precision::Storage::Mass, float> ? 5.e-5 : 3.e-12;
  for (int epoch = 0; epoch != 2; ++epoch) {
    current_validation.clear(current);
    mass_validation.clear(mass);
    require(ecsim::enqueue_clear_moment_field(split_current, current) == lpfx::ErrorCode::success &&
                ecsim::enqueue_clear_mass_matrix(split_mass, mass) == lpfx::ErrorCode::success,
            "resident deposition: accumulator clear failed");
    require(slots.reserve(particles.capacity()) == lpfx::ErrorCode::success &&
                vc::enqueue_initial_particle_classification<Precision>(
                    migration, particles, geometry, local) == lpfx::ErrorCode::success,
            "resident deposition: classification publication failed");
    require(migration.begin(local).requires_conclusion(), "resident deposition: migration begin failed");
    bool sorting = false;
    for (;;) {
      const auto prepared = migration.template progress<Stage::prepare>();
      require(!prepared.is_failure(), "resident deposition: preparation failed");
      if (!sorting && migration.resident_access_ready()) {
        auto access = migration.acquire_resident_access();
        require(access.has_value() && Traits::succeeded(access.value().ready->wait_on(sort)),
                "resident deposition: resident dependency failed");
        ordering.enqueue(std::move(access).value(), sort);
        sorting = true;
      }
      if (prepared.is_success()) break;
    }
    require(sorting && migration.template wait<Stage::resident>().is_success(),
            "resident deposition: resident preparation failed");
    ordering.host_wait_complete();
    auto residents = ordering.resident_read();
    require(residents.has_value(), "resident deposition: real resident capability missing");
    const auto prefix = residents->resident_particle_view().size();
    if (ranks == 2 && rank == 0 && no_rank_zero_residents)
      require(prefix == 0, "resident deposition: zero resident case was not exercised");
    require(ecsim::enqueue_resident_current<Precision>(*residents, magnetic, split_current,
                deposition_cells, input.domain.inverse_cell_volume, .125, current_validation.data(), current) == lpfx::ErrorCode::success &&
                Traits::succeeded(current_reader.record(current)),
            "resident deposition: early current submission failed");

    // Keep routing independent of the retained current reader. Delay mass
    // only after route releases allocation/native transport ownership.
    require(migration.template wait<Stage::route>().is_success(), "resident deposition: routing failed");
    HeldMassReader hold_mass;
    hold_mass.block(mass);
    require(ecsim::enqueue_resident_mass_matrix<Precision>(*residents, magnetic, split_mass,
                deposition_cells, .125, mass_validation.data(), mass) == lpfx::ErrorCode::success &&
                Traits::succeeded(mass_reader.record(mass)),
            "resident deposition: early mass submission failed");
    require(Traits::succeeded(current_reader.wait_on(local)) &&
                Traits::succeeded(mass_reader.wait_on(local)) &&
                Traits::succeeded(tail_ready.record(local)),
            "resident deposition: two-reader handoff failed");
    residents.reset(); // No capability or ordinary-owner view survives append.
    require(migration.release_resident_access(tail_ready).is_pending() &&
                migration.template progress<Stage::finalize>().is_pending(),
            "resident deposition: finalize ignored the delayed mass reader");
    hold_mass.release();
    for (;;) {
      const auto release = migration.release_resident_access(tail_ready);
      require(!release.is_failure(), "resident deposition: resident release failed");
      if (release.is_success()) break;
    }
    const auto requirements = migration.finalization_requirements();
    require(requirements.has_value(), "resident deposition: finalization facts unavailable");
    if (requirements.value().requires_growth) ordering.release_growth_scratch();
    require(migration.template wait<Stage::finalize>().is_success() && migration.conclude().is_success(),
            "resident deposition: reader join or finalize failed");
    if (migration.storage_changed()) ordering.rebind_after_growth(particles, properties);
    const auto partition = migration.completed_partition();
    require(partition.has_value() && partition.value().resident_prefix == prefix &&
                partition.value().total_size == particles.size(),
            "resident deposition: completed partition differs from resident prefix");
    const auto arrivals = particles.size() - prefix;
    if (epoch == 1) require(arrivals == 0, "resident deposition: no-arrival reuse case failed");
    if (epoch == 0 && ranks == 2 && rank == 1) {
      require(arrivals != 0, "resident deposition: arriving tail was not exercised");
      require(reserve_fitting ? particles.capacity() == initial_capacity
                              : particles.capacity() > initial_capacity,
              "resident deposition: expected fitting/growing append was not exercised");
    }
    require(Traits::succeeded(tail_ready.record(local)) &&
                Traits::succeeded(tail_ready.wait_on(current)) && Traits::succeeded(tail_ready.wait_on(mass)),
            "resident deposition: fresh tail dependency failed");
    require(ecsim::enqueue_unsorted_current<Precision>(particles, magnetic, split_current,
                deposition_cells, input.domain.inverse_cell_volume, .125, prefix, arrivals, current_validation.data(), current) == lpfx::ErrorCode::success &&
            ecsim::enqueue_unsorted_mass_matrix<Precision>(particles, magnetic, split_mass,
                deposition_cells, .125, prefix, arrivals, mass_validation.data(), mass) == lpfx::ErrorCode::success,
            "resident deposition: fresh tail submission failed");
    current_validation.capture(current);
    mass_validation.capture(mass);
    require(Traits::succeeded(current_done.record(current)) && Traits::succeeded(mass_done.record(mass)) &&
                Traits::succeeded(current_done.wait_on(main)) && Traits::succeeded(mass_done.wait_on(main)),
            "resident deposition: final quantity join failed");
    if (!no_rank_zero_residents)
      verify<Precision>(particles, input, main);
    else {
      const auto total = tpic::test::prepare_particle_load(fixture(0), 0).count +
                         tpic::test::prepare_particle_load(fixture(1), 0).count;
      require(particles.size() == (rank == 0 ? 0U : total),
              "resident deposition: empty sender lost or duplicated population");
    }
    unsorted_validation.clear(main);
    require(ecsim::enqueue_clear_moment_field(unsorted_current, main) == lpfx::ErrorCode::success &&
                ecsim::enqueue_clear_mass_matrix(unsorted_mass, main) == lpfx::ErrorCode::success &&
            ecsim::enqueue_unsorted_current<Precision>(particles, magnetic, unsorted_current,
                deposition_cells, input.domain.inverse_cell_volume, .125, 0, particles.size(), unsorted_validation.data(), main) == lpfx::ErrorCode::success &&
            ecsim::enqueue_unsorted_mass_matrix<Precision>(particles, magnetic, unsorted_mass,
                deposition_cells, .125, 0, particles.size(), unsorted_validation.data(), main) == lpfx::ErrorCode::success,
            "resident deposition: completed scalar oracle failed");
    compare_resident_currents_kernel<<<(nodes + 127) / 128, 128, 0, main.stream()>>>(
        unsorted_current.view(), split_current.view(), cells.nx, cells.ny, cells.nz,
        unsorted_validation.data(), tolerance);
    require(Traits::succeeded(Traits::take_last_launch_error()), "resident deposition: current comparison launch failed");
    unsorted_validation.capture(main);
    current_validation.host_wait_check("resident deposition: current numerical validation failed");
    mass_validation.host_wait_check("resident deposition: mass numerical validation failed");
    unsorted_validation.host_wait_check("resident deposition: current differs from completed unsorted oracle");
    const auto expected = download_values(unsorted_mass.data(), unsorted_mass.layout().storage_size(), main);
    const auto actual = download_values(split_mass.data(), split_mass.layout().storage_size(), main);
    for (std::size_t n = 0; n < actual.size(); ++n)
      require(std::isfinite(actual[n]) &&
                  std::abs(double(actual[n]) - expected[n]) <= tolerance * (1 + std::abs(double(expected[n]))),
              "resident deposition: mass differs from completed unsorted oracle");
    ordering.host_wait_retired(main);
    require(slots.reusable() && migration.reusable(), "resident deposition: reader lease did not retire");
  }
  require(communication.close().is_success(),
          "resident deposition: close retained communication storage");
}

template <class Precision, class View>
__global__ void set_local_particle_boundary_cases_kernel(View particles, tpic::data::ParticleCellMap cells,
                                        unsigned invalid_case) {
  using S = tpic::data::ParticleSchema<Precision>;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= particles.size()) return;
  const int xs[]{-2, -1, 0, cells.nx - 1, cells.nx, cells.nx + 1};
  const int ys[]{-2, -1, 0, cells.ny - 1, cells.ny, cells.ny + 1};
  const int zs[]{-2, -1, 0, cells.nz - 1, cells.nz, cells.nz + 1};
  particles.template get<typename S::Cell>()[index] =
      cells.encode(xs[index % 6], ys[(index / 2) % 6], zs[(index / 3) % 6]);
  particles.template get<typename S::Rx>()[index] = (index % 3) * .25;
  particles.template get<typename S::Ry>()[index] = (index % 4) * .25;
  particles.template get<typename S::Rz>()[index] = (index % 2) * .5;
  particles.template get<typename S::U>()[index] = .125 * (1 + index % 11);
  particles.template get<typename S::V>()[index] = -.25 * (1 + index % 7);
  particles.template get<typename S::W>()[index] = .5 * (1 + index % 5);
  particles.template get<typename S::Charge>()[index] = -1.0;
  if (index != 0) return;
  if (invalid_case == 1) particles.template get<typename S::Cell>()[0] = -1;
  if (invalid_case == 2) particles.template get<typename S::Rx>()[0] = 1;
  if (invalid_case == 3)
    particles.template get<typename S::V>()[0] = std::numeric_limits<typename Precision::Storage::Velocity>::quiet_NaN();
  if (invalid_case == 4)
    particles.template get<typename S::Charge>()[0] = std::numeric_limits<typename Precision::Storage::Weight>::infinity();
}

template <class Attribute, class Owner>
void compare_attribute(const Owner& local, const Owner& oracle, Executor executor) {
  const auto a = download<Attribute>(local, executor);
  const auto b = download<Attribute>(oracle, executor);
  require(a.size() == b.size(), "local startup changed population");
  for (std::size_t i = 0; i < a.size(); ++i) {
    bool same = a[i] == b[i];
    if constexpr (std::is_floating_point_v<typename Attribute::value_type>)
      same = same || (std::isnan(a[i]) && std::isnan(b[i]));
    require(same, "local startup differs from registered boundary classifier");
  }
}

template <class Precision>
void run_local_boundaries(Executor executor, const lpfx::device::DeviceProperties& properties,
                          bool periodic, bool thin) {
  using S = tpic::data::ParticleSchema<Precision>;
  using Owner = lpfx::Particles<typename S::type, lpfx::DeviceSpace<Backend>>;
  auto parameters = fixture(0).parameters;
  parameters.topology.periodic_x_particles = periodic;
  parameters.topology.periodic_y_particles = periodic;
  parameters.topology.periodic_z_particles = periodic;
  if (thin) parameters.grid = {1, 1, 1, 1, 1, 1};
  const auto input = tpic::simulation::prepare_input(std::move(parameters),
      {.rank = 0, .communicator_size = 1, .order = tpic::config::ParticleOrder::full});
  const auto load = tpic::test::prepare_particle_load(input, 0);
  auto local = tpic::initialization::make_initial_particles<Precision>(load, input.domain, input.initial_case, 0, executor);
  auto oracle = tpic::initialization::make_initial_particles<Precision>(load, input.domain, input.initial_case, 0, executor);
  const auto geometry = vc::prepare_particle_migration_geometry(input.domain, input.physics);
  const auto identities = download<typename S::Id>(local, executor);
  tpic::device::DeviceValidation<Backend> validation;

  lpfx::MpiCallDomain mpi_domain;
  require(lpfx::try_make_mpi_call_domain(MPI_COMM_WORLD,
      lpfx::MpiProcessThreadUse::single_process_thread, mpi_domain).succeeded(),
      "local startup: MPI domain failed");
  d::CommunicationSetup setup;
  require(d::try_make_communication_setup(mpi_domain,
      properties, executor, setup).succeeded(), "local startup: setup failed");
  lpfx::DeviceParticleClassificationSlots<Backend> oracle_slots, unused_slots;
  auto requests = d::operations(
      d::migrate<"oracle">(oracle, input.domain.particle_topology,
          vc::particle_migration_policy<Precision>(geometry), d::on<vc::DeviceRoute>(executor),
          d::using_flags(oracle_slots), d::traffic_seed(1, d::arrivals_up_to_owner, false),
          d::resident_access<false>),
      d::migrate<"unused">(local, input.domain.particle_topology,
          vc::particle_migration_policy<Precision>(geometry), d::on<vc::DeviceRoute>(executor),
          d::using_flags(unused_slots), d::traffic_seed(1, d::arrivals_up_to_owner, false),
          d::resident_access<false>));
  auto communication = d::make_communication(requests);
  require(d::prepare(communication, setup, std::move(requests),
              d::PreparationLimits{16 * 1024 * 1024, 16 * 1024 * 1024}) == lpfx::ErrorCode::success,
          "local startup: registration failed");
  auto& migration = communication.template get<"oracle">();
  auto& unused_migration = communication.template get<"unused">();
  require(oracle_slots.workspace_bytes() == 0 && unused_slots.workspace_bytes() == 0,
          "local startup: registration eagerly allocated classification slots");
  require(oracle_slots.reserve(oracle.capacity()) == lpfx::ErrorCode::success,
          "local startup: oracle classification reservation failed");

  auto nonlocal = geometry;
  nonlocal.global_offset[0] = 1;
  Owner invalid_owner;
  require(vc::enqueue_local_initial_particle_boundaries<Precision>(
              local, nonlocal, validation.data(), executor) == lpfx::ErrorCode::invalid_argument &&
          vc::enqueue_local_initial_particle_boundaries<Precision>(
              invalid_owner, geometry, validation.data(), executor) == lpfx::ErrorCode::invalid_state &&
          vc::enqueue_local_initial_particle_boundaries<Precision>(
              local, geometry, nullptr, executor) == lpfx::ErrorCode::invalid_argument &&
          vc::enqueue_local_initial_particle_boundaries<Precision>(
              local, geometry, validation.data(), Executor{Traits::null_stream}) == lpfx::ErrorCode::invalid_argument,
          "local startup: invalid launch admission accepted");

  for (unsigned invalid_case = 0; invalid_case <= 4; ++invalid_case) {
    for (auto* owner : {&local, &oracle})
      set_local_particle_boundary_cases_kernel<Precision><<<(owner->size() + 255) / 256, 256, 0, executor.stream()>>>(
          owner->view(), load.cells, invalid_case);
    require(Traits::succeeded(Traits::take_last_launch_error()), "local startup: fixture launch failed");
    validation.clear(executor);
    require(vc::enqueue_local_initial_particle_boundaries<Precision>(
        local, geometry, validation.data(), executor) == lpfx::ErrorCode::success,
        "local startup: boundary submission failed");
    validation.capture(executor);
    bool rejected = false;
    try { validation.host_wait_check("local startup: expected invalid boundary record"); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected == (invalid_case != 0), "local startup: boundary failure status differs");
    require(vc::enqueue_initial_particle_classification<Precision>(
        migration, oracle, geometry, executor) == lpfx::ErrorCode::success,
        "local startup: oracle classifier failed");
    if (invalid_case == 0) {
      require(migration.begin(executor).requires_conclusion() &&
              migration.template wait<lpfx::distributed::ParticleMigrationStage::prepare>().is_success() &&
              migration.template wait<lpfx::distributed::ParticleMigrationStage::resident>().is_success() &&
              migration.template wait<lpfx::distributed::ParticleMigrationStage::route>().is_success() &&
              migration.template wait<lpfx::distributed::ParticleMigrationStage::finalize>().is_success() &&
              migration.conclude().is_success(), "local startup: oracle migration stages failed");
    } else {
      unsigned failure = 0;
      require(Traits::succeeded(Traits::copy_device_to_host_async(&failure,
                  &oracle_slots.registered_summary()->failure, sizeof(failure), executor.stream())) &&
                  Traits::succeeded(Traits::synchronize_stream(executor.stream())) && failure != 0,
              "local startup: oracle classifier accepted invalid record");
      require(oracle_slots.cancel(executor) == lpfx::ErrorCode::success,
              "local startup: invalid oracle classification could not cancel");
    }
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (compare_attribute<std::tuple_element_t<I, typename S::type::attributes_type>>(
          local, oracle, executor), ...);
    }(std::make_index_sequence<Owner::attribute_count>{});
    require(local.size() == load.count && download<typename S::Id>(local, executor) == identities &&
                unused_slots.workspace_bytes() == 0 && unused_slots.reusable() && unused_migration.reusable(),
            "local startup: count, IDs or idle migration state changed");
  }

  lpfx::PreparedDeviceParticleStorage<typename S::type, Backend> empty_storage;
  require(lpfx::try_prepare_device_particle_storage(0, empty_storage) == lpfx::ErrorCode::success,
          "local startup: empty owner preparation failed");
  auto made_empty = lpfx::make_device_particles(empty_storage);
  require(made_empty.has_value(), "local startup: empty owner allocation failed");
  auto empty = std::move(made_empty).value();
  validation.clear(executor);
  require(vc::enqueue_local_initial_particle_boundaries<Precision>(
      empty, geometry, validation.data(), executor) == lpfx::ErrorCode::success,
      "local startup: empty owner rejected");
  validation.capture(executor);
  validation.host_wait_check("local startup: empty owner changed validation");
  require(empty.size() == 0 && communication.close().is_success() &&
              unused_slots.workspace_bytes() == 0 && unused_slots.reusable(),
          "local startup: closing never-started migration retained storage or classification");
}
} // namespace

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  try {
    require(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS &&
            MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS &&
            MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS,
            "migration smoke: MPI setup failed");
    lpfx::device::DeviceProperties properties{};
    const auto queried = Traits::query_current_device_properties(properties);
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
    const bool unavailable = queried == cudaErrorNoDevice || queried == cudaErrorInsufficientDriver ||
                             queried == cudaErrorSystemDriverMismatch;
#else
    const bool unavailable = queried == hipErrorNoDevice || queried == hipErrorInsufficientDriver;
#endif
    const int local = Traits::succeeded(queried) ? 0 : unavailable ? 1 : 2;
    int collective = 0;
    require(MPI_Allreduce(&local, &collective, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS,
            "migration smoke: device agreement failed");
    if (collective != 0) {
      if (rank == 0) std::cout << "TPIC native migration unavailable\n";
      return MPI_Finalize() == MPI_SUCCESS ? (collective == 1 ? 77 : 1) : 1;
    }
    auto stream = Traits::null_stream;
    require(Traits::succeeded(Traits::create_stream(stream)) && stream != Traits::null_stream,
            "migration smoke: stream creation failed");
    const Executor executor{stream};
    run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor, properties);
    run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor, properties);
    for (bool fitting : {false, true}) {
      run_resident_deposition<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor, properties, fitting, false);
      run_resident_deposition<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor, properties, fitting, false);
      run_resident_deposition<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor, properties, fitting, false);
    }
    if (ranks == 2)
      run_resident_deposition<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor, properties, true, true);
    if (ranks == 1) {
      for (bool periodic : {false, true})
        for (bool thin : {false, true}) {
          run_local_boundaries<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor, properties, periodic, thin);
          run_local_boundaries<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor, properties, periodic, thin);
          run_local_boundaries<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor, properties, periodic, thin);
        }
    }
    require(Traits::succeeded(Traits::destroy_stream(stream)), "migration smoke: stream destroy failed");
    if (rank == 0) std::cout << "TPIC native particle migration passed on " << ranks << " rank(s)\n";
    return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
