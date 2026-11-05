// Native startup qualification using the build-selected LPFx device route.
#include <lpfx/backend/backend.hpp>
#include "../support/pinned_transfer.hpp"
#include <lpfx/data/particle/device/initialization.hpp>
#include <tpic/communication/transport_routes.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/initialization/particles/particle_centering_device.hpp>
#include <tpic/initialization/particles/particle_loading_device.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "../support/test_builders.hpp"

namespace {
namespace d = lpfx::distributed;
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
int rank = 0, ranks = 1;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
         error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}

// A drift that differs between particles: along z it reverses across y = 1.
// The seed along y has no part in centering: an evaluation would fail the oracle.
struct ReversingProfile {
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(tpic::initialization::ParticleProfileSample& sample,
      const tpic::initialization::ParticleLoadPlan&, double, double y, double, const int*) const noexcept {
    if (y > 1.0) sample.drift[2] = -sample.drift[2];
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double, double, double z, double&, double& v,
                                                             double&) const noexcept {
    v += 0.25 * z;
  }
  [[nodiscard]] tpic::initialization::ParticleSupport support() const noexcept { return {}; }
};

// The loaded drift of a stored particle, written out independently of the device helper.
template<class Loading>
std::array<double, 3> oracle_drift(const tpic::initialization::ParticleLoadPlan& plan, std::int32_t key,
                                   double ry) {
  std::array<double, 3> drift{plan.drift[0], plan.drift[1], plan.drift[2]};
  if constexpr (std::is_same_v<Loading, ReversingProfile>) {
    if (plan.origin[1] + (plan.cells.decode(key).j + ry) * plan.spacing[1] > 1.0) drift[2] = -drift[2];
  }
  return drift;
}

template<class Precision, class Range>
__global__ void fill_centering_particles_kernel(Range range, int peer, int epoch, int cells) {
  using S = tpic::data::ParticleSchema<Precision>;
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range.size()) return;
  range.template store<typename S::Rx>(i, 0.125);
  range.template store<typename S::Ry>(i, 0.375);
  range.template store<typename S::Rz>(i, 0.625);
  range.template store<typename S::U>(i, 0.125 * (i % 17) - 0.0625 * (peer + epoch));
  range.template store<typename S::V>(i, 0.25 * (i % 11) - 0.5 * peer + 0.375 * epoch);
  range.template store<typename S::W>(i, -0.375 * (i % 13) + 0.125 * peer - 0.25 * epoch);
  // Mixed signs distinguish signed weighting from absolute-charge weighting.
  const double q = epoch == 3 ? (i % 2 == 0 ? 1.0 : -1.0)
                              : (i % 3 == 0 ? 1.0 : -double(i % 3 + 1));
  range.template store<typename S::Charge>(i, q);
  range.template store<typename S::Cell>(i, (i + 12) % cells);
  range.template store<typename S::Id>(i, std::uint64_t(peer) * 10000 + i);
}

template<class Precision>
auto make_fixture(unsigned capacity, int epoch, int cells, Executor executor) {
  using Schema = typename tpic::data::ParticleSchema<Precision>::type;
  lpfx::PreparedDeviceParticleStorage<Schema, Backend> storage;
  require(lpfx::try_prepare_device_particle_storage(capacity, storage) == lpfx::ErrorCode::success,
          "centering: particle storage preparation failed");
  auto made = lpfx::make_device_particles(storage);
  require(made.has_value(), "centering: particle allocation failed");
  auto owner = std::move(made).value();
  const unsigned count = epoch == 2 && ranks > 1 && rank == 0 ? 0
      : epoch == 3 ? 514 : 513 + rank % 3;
  if (count != 0) {
    auto prepared = lpfx::make_device_particle_initializer(owner);
    require(prepared.has_value(), "centering: fixture initializer failed");
    auto initializer = std::move(prepared).value();
    auto admitted = initializer.begin_append(count, executor);
    require(admitted.has_value(), "centering: fixture admission failed");
    fill_centering_particles_kernel<Precision><<<(count + 255) / 256, 256, 0, executor.stream()>>>(
        admitted.value(), rank, epoch, cells);
    require(Traits::succeeded(Traits::take_last_launch_error()) &&
            initializer.publish(executor).is_success() && initializer.wait().is_success(),
            "centering: fixture publication failed");
  }
  return owner;
}

tpic::initialization::ParticleLoadPlan loader_fixture(std::array<double, 3> drift,
                                                     tpic::config::PrecisionSelection profile) {
  tpic::config::SimulationParameters parameters;
  parameters.case_name = "Uniform";
  parameters.dt = 0.1;
  parameters.ncycles = 1;
  parameters.grid = {double(2 * ranks), 2, 2, 2 * ranks, 2, 2};
  parameters.topology.xlen = ranks;
  parameters.output = {.field_cycle = 0, .particles_cycle = 0,
      .test_particles_cycle = 0, .restart_cycle = 0, .diagnostics_cycle = 0};
  parameters.ns = 1;
  parameters.species.resize(1);
  auto& species = parameters.species[0];
  species.q_over_m = -1;
  species.rho_init = 1;
  species.npcelx = species.npcely = species.npcelz = 2;
  species.uth = 0.25; species.vth = 0.5; species.wth = 0.375;
  species.u0 = drift[0]; species.v0 = drift[1]; species.w0 = drift[2];
  species.center_initial_velocity = true;
  const auto input = tpic::simulation::prepare_input(std::move(parameters),
      {.rank = rank, .communicator_size = ranks, .precision = profile});
  const auto plan = tpic::test::prepare_particle_load(input, 0);
  require(input.host_admitted() && plan.center_velocity && plan.drift[0] == drift[0] &&
              plan.drift[1] == drift[1] && plan.drift[2] == drift[2],
          "centering: loader lost the request or the configured drift");
  return plan;
}

template<class Precision> struct Snapshot {
  std::vector<typename Precision::Storage::Position> rx, ry, rz;
  std::array<std::vector<typename Precision::Storage::Velocity>, 3> velocity;
  std::vector<typename Precision::Storage::Weight> q;
  std::vector<std::int32_t> cell;
  std::vector<std::uint64_t> id;
};

// Particle payload readback is confined to validation.
template<class Precision, class Owner>
Snapshot<Precision> download(const Owner& owner, Executor executor) {
  using S = tpic::data::ParticleSchema<Precision>;
  Snapshot<Precision> values;
  const auto copy = [&]<class Attribute>(auto& destination) {
    destination.resize(owner.size());
    if (destination.empty()) return;
    tpic::test::download_bytes(
        destination.data(), owner.cview().template get<Attribute>(),
        destination.size() * sizeof(typename Attribute::value_type), executor);
  };
  copy.template operator()<typename S::Rx>(values.rx);
  copy.template operator()<typename S::Ry>(values.ry);
  copy.template operator()<typename S::Rz>(values.rz);
  copy.template operator()<typename S::U>(values.velocity[0]);
  copy.template operator()<typename S::V>(values.velocity[1]);
  copy.template operator()<typename S::W>(values.velocity[2]);
  copy.template operator()<typename S::Charge>(values.q);
  copy.template operator()<typename S::Cell>(values.cell);
  copy.template operator()<typename S::Id>(values.id);
  return values;
}

// The global charge-weighted sums of value(axis, i) and, last, the charge sum.
template<class Precision, class Value>
std::array<double, 4> global_moments(const Snapshot<Precision>& values, Value value) {
  std::array<double, 4> local{}, global{};
  for (std::size_t i = 0; i != values.q.size(); ++i) {
    local[3] += values.q[i];
    for (unsigned axis = 0; axis != 3; ++axis)
      local[axis] += double(values.q[i]) * value(axis, i);
  }
  require(MPI_Allreduce(local.data(), global.data(), 4, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD) == MPI_SUCCESS,
          "centering: validation moment reduction failed");
  return global;
}

template<class Precision>
void run(Executor executor, const lpfx::device::DeviceProperties& properties,
         tpic::config::PrecisionSelection profile) {
  constexpr unsigned capacity = 640;
  using Definition = tpic::test::VelocityCenteringDefinition<0x564301, Precision>;
  tpic::initialization::VelocityCentering<Precision, Backend> workspace(capacity, properties, executor);
  auto plan = tpic::test::prepare_velocity_centering_plan<0x564301, Precision>(ranks);
  auto operand = workspace.collective_operand();
  auto binding = d::bind<tpic::communication::DeviceRoute>(
      plan, operand, executor);
  lpfx::MpiCallDomain mpi_domain;
  require(lpfx::try_make_mpi_call_domain(MPI_COMM_WORLD,
      lpfx::MpiProcessThreadUse::single_process_thread, mpi_domain).succeeded(),
      "centering: MPI call domain failed");
  d::CommunicationSetup setup;
  require(d::try_make_communication_setup(mpi_domain,
      properties, executor, setup).succeeded(), "centering: communication setup failed");
  lpfx::HostMemoryBudget registration_budget, operation_budget;
  for (auto* budget : {&registration_budget, &operation_budget})
    require(budget->try_set_limit(1024 * 1024) && budget->seal(),
            "centering: communication budget failed");
  d::Communication<decltype(binding)> communication;
  require(d::try_prepare_communication(communication, setup, registration_budget,
      operation_budget, std::move(binding)) == lpfx::ErrorCode::success,
      "centering: collective registration failed");
  auto& operation = communication.template get<Definition>();
  constexpr double tolerance = std::is_same_v<typename Precision::Storage::Velocity, float> ? 2e-6 : 2e-13;
  // Epoch 1 centers a real uniform load; the other epochs center fixture
  // particles whose drift differs from particle to particle.
  const auto center_and_check = [&]<class Loading>(int epoch, const Loading& loading) {
    const auto load_plan = loader_fixture({0.35 + 0.125 * epoch, -0.2, 0.125}, profile);
    auto owner = epoch == 1
        ? tpic::initialization::make_initial_particles<Precision>(load_plan, {}, loading, executor)
        : make_fixture<Precision>(capacity, epoch, load_plan.cells.count, executor);
    const auto before = download<Precision>(owner, executor);
    std::vector<std::array<double, 3>> drift(before.q.size());
    for (std::size_t i = 0; i != drift.size(); ++i)
      drift[i] = oracle_drift<Loading>(load_plan, before.cell[i], double(before.ry[i]));
    // The velocities without their loaded drift, and the loaded drift.
    const auto moments = global_moments(before,
        [&](unsigned axis, std::size_t i) { return double(before.velocity[axis][i]) - drift[i][axis]; });
    const auto loaded = global_moments(before, [&](unsigned axis, std::size_t i) { return drift[i][axis]; });
    if (epoch == 1) workspace.host_blocking_center_if_requested(load_plan, loading, owner, operation);
    else workspace.host_blocking_center(owner, load_plan, loading, operation);
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
    require(Traits::succeeded(cudaStreamQuery(executor.stream())),
#else
    require(Traits::succeeded(hipStreamQuery(executor.stream())),
#endif
            "centering: helper returned before stream completion");
    require(operation.reusable() && operation.generation() == unsigned(epoch + 1),
            "centering: collective epoch did not conclude reusable");
    const auto after = download<Precision>(owner, executor);
    require(before.rx == after.rx && before.ry == after.ry && before.rz == after.rz &&
            before.q == after.q && before.cell == after.cell && before.id == after.id,
            "centering: position, charge, cell or identity changed");
    require((moments[3] == 0) == (epoch == 3), "centering: invalid fixture charge sum");
    // A zero charge sum has no mean: the loaded velocities return to rounding.
    for (unsigned axis = 0; axis != 3; ++axis) {
      const double mean = moments[3] == 0 ? 0 : moments[axis] / moments[3];
      for (std::size_t i = 0; i != before.q.size(); ++i)
        require(std::abs(double(after.velocity[axis][i]) - (double(before.velocity[axis][i]) - mean)) < tolerance,
                "centering: per-particle velocity shift differs from the signed-charge oracle");
    }
    if (moments[3] != 0) {
      const auto centered = global_moments(after,
          [&](unsigned axis, std::size_t i) { return double(after.velocity[axis][i]); });
      for (unsigned axis = 0; axis != 3; ++axis)
        require(std::abs(centered[axis] / centered[3] - loaded[axis] / loaded[3]) < tolerance,
                "centering: global charge-weighted velocity missed the mean of the loaded drift");
    }
  };
  for (int epoch = 0; epoch != 4; ++epoch) {
    if (epoch == 1) center_and_check(epoch, tpic::initialization::UniformProfile{});
    else center_and_check(epoch, ReversingProfile{});
  }
  require(communication.close().is_success() && operation_budget.used() == 0,
          "centering: communication close retained operation storage");
}
} // namespace

int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
  try {
    require(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS &&
            MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS &&
            MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS,
            "centering: MPI setup failed");
    lpfx::device::DeviceProperties properties{};
    const auto queried = Traits::query_current_device_properties(properties);
    const int local = Traits::succeeded(queried) ? 0 : runtime_unavailable(queried) ? 1 : 2;
    int collective = 0;
    require(MPI_Allreduce(&local, &collective, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS,
            "centering: device availability agreement failed");
    if (collective != 0) {
      if (rank == 0) std::cout << "TPIC native particle centering "
          << (collective == 1 ? "skipped: no usable device/driver\n" : "failed: device query error\n");
      return MPI_Finalize() == MPI_SUCCESS ? (collective == 1 ? 77 : 1) : 1;
    }
    auto stream = Traits::null_stream;
    require(Traits::succeeded(Traits::create_stream(stream)) && stream != Traits::null_stream,
            "centering: stream creation failed");
    const Executor executor{stream};
    // The matched pairs: every build compiles them, so the input is admitted.
    for (unsigned level = 0; level < tpic::config::precision_level_count; ++level) {
      const auto profile = tpic::config::matched_precision(static_cast<tpic::config::ComputePrecision>(level));
      tpic::config::dispatch_precision(profile, [&]<class Precision> {
        run<Precision>(executor, properties, profile);
      });
    }
    require(Traits::succeeded(Traits::destroy_stream(stream)), "centering: stream destruction failed");
    if (rank == 0) std::cout << "TPIC native particle centering passed on " << ranks << " rank(s)\n";
    return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
  } catch (const std::exception& failure) {
    std::cerr << "rank " << rank << ": " << failure.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
