#include <tpic/config/domain.hpp>
#include <tpic/config/simulation_parameters.hpp>
// Differential checks for fused scalar reductions, rank-one collective identity,
// output selection, odd particle tails, empty populations, strided fields and
// changing producer epochs consumed on two bounded diagnostics lanes, and the
// Gauss residual against an independent host evaluation.
#include <tpic/config/precision.hpp>
#include <tpic/diagnostics/gauss_residual_device.hpp>
#include <tpic/diagnostics/scalar_diagnostics_device.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <lpfx/backend/cuda_hip/kernel.hpp>
#include "../support/pinned_transfer.hpp"
#include <tpic/diagnostics/output_moment_deposition_device.hpp>
#include <lpfx/memory/space.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Allocation = lpfx::device::DeviceAllocation<Backend>;
using Cells = decltype(std::declval<tpic::config::PreparedDomain>().field_neighborhood.local_partition().local_cells());
namespace diag = tpic::diagnostics;
constexpr unsigned capacity = 4099;
constexpr int nx = 17, ny = 5, nz = 3, pitch_x = 21, pitch_y = 8;
constexpr int nodes = pitch_x * pitch_y * (nz + 2);
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const char* message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <=
      2e-11 * std::max(1., std::abs(expected)), message);
}
template <class Function> void rejects(Function&& function, std::string_view reason) {
  bool rejected = false;
  try {
    function();
  } catch (const std::runtime_error& error) {
    rejected = std::string_view(error.what()).find(reason) != std::string_view::npos;
  }
  require(rejected, "diagnostics accepted invalid producer lifecycle");
}

template <class P> struct ParticleState {
  typename P::Storage::Position rx[capacity], ry[capacity], rz[capacity];
  typename P::Storage::Velocity u[capacity], v[capacity], w[capacity];
  typename P::Storage::Weight q[capacity];
  int cell[capacity];
};
template <class P> struct ParticleView {
  ParticleState<P>* state;
  unsigned count;
  __host__ __device__ unsigned size() const { return count; }
  template <class Tag> __device__ auto get() const {
    using S = tpic::data::ParticleSchema<P>;
    if constexpr (std::is_same_v<Tag, typename S::Rx>) return state->rx;
    else if constexpr (std::is_same_v<Tag, typename S::Ry>) return state->ry;
    else if constexpr (std::is_same_v<Tag, typename S::Rz>) return state->rz;
    else if constexpr (std::is_same_v<Tag, typename S::U>) return state->u;
    else if constexpr (std::is_same_v<Tag, typename S::V>) return state->v;
    else if constexpr (std::is_same_v<Tag, typename S::W>) return state->w;
    else if constexpr (std::is_same_v<Tag, typename S::Charge>) return state->q;
    else return state->cell;
  }
};
template <class P> struct ParticleOwner {
  using backend_type = Backend;
  ParticleView<P> data;
  bool valid() const { return true; }
  unsigned size() const { return data.size(); }
  auto cview() const { return data; }
};
template <class Scalar> struct Component {
  using location_type = lpfx::Node;
  Scalar* data;
  __host__ __device__ Scalar& operator()(int i, int j, int k) const {
    return data[i + pitch_x * (j + pitch_y * k)];
  }
};
template <class Scalar, unsigned Components> struct FieldView {
  static constexpr std::size_t component_count = Components;
  Scalar* data;
  template <std::size_t Index> __host__ __device__ Component<Scalar> component() const {
    static_assert(Index < component_count);
    return {data + Index * nodes};
  }
};
template <class Scalar, unsigned Components> struct FieldOwner {
  using space_type = lpfx::DeviceSpace<Backend>;
  static constexpr unsigned component_count = Components;
  FieldView<Scalar, component_count> data;
  Cells extent;
  auto cells() const { return extent; }
  bool initialized() const { return true; }
  auto view() const { return data; }
};
struct IdentityCollective {
  lpfx::Status begin() { throw std::runtime_error("rank-one diagnostics called MPI"); }
  lpfx::Status wait() { throw std::runtime_error("rank-one diagnostics waited for MPI"); }
  lpfx::Status conclude() { throw std::runtime_error("rank-one diagnostics concluded MPI"); }
};
template <class P> struct Fixture {
  ParticleState<P> particles{};
  typename P::Storage::Electric electric[3 * nodes]{};
  typename P::Storage::Magnetic magnetic[3 * nodes]{};
  double moments[10 * nodes]{};
  double partials[5 * tpic::data::contribution_count]{}, sums[5]{};
  unsigned invalid{};
};
template <class T> void upload(Allocation& allocation, const T& source, Executor executor) {
  tpic::test::upload_bytes(allocation.raw_base(), &source, sizeof(T), executor);
}
template <class T> void download(T& destination, Allocation& allocation, Executor executor) {
  require(Traits::succeeded(Traits::take_last_launch_error()), "diagnostics execution failed");
  tpic::test::download_bytes(&destination, allocation.raw_base(), sizeof(T), executor);
}

template <class P> void run(Executor executor, Executor diagnostic_executor,
                            Executor helper_executor,
                            const lpfx::device::DeviceProperties& properties) {
  auto input = std::make_unique<Fixture<P>>(), result = std::make_unique<Fixture<P>>();
  tpic::config::GridConfig grid{.lx = nx, .ly = ny, .lz = nz, .nxc = nx, .nyc = ny, .nzc = nz};
  tpic::config::TopologyConfig topology;
  topology.periodic_y = false; // Exercise physical dual-volume weighting.
  const auto domain = tpic::config::prepare_domain(grid, topology, 0, 1);
  const auto cells = tpic::data::prepare_particle_cell_map(domain);
  for (unsigned i = 0; i < capacity; ++i) {
    auto& p = input->particles;
    p.rx[i] = (i % 16) / 16.; p.ry[i] = (i % 8) / 8.; p.rz[i] = (i % 4) / 4.;
    p.cell[i] = cells.encode(i % nx, (i / nx) % ny, (i / nx / ny) % nz);
    p.q[i] = -double(1 + i % 7) / 8.;
    p.u[i] = (int(i % 13) - 6) / 8.; p.v[i] = (int(i % 11) - 5) / 16.;
    p.w[i] = (int(i % 17) - 8) / 32.;
  }
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k <= nz; ++k)
      for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
          const int index = c * nodes + i + pitch_x * (j + pitch_y * k);
          input->electric[index] = (c + 1) * (1 + i + 2*j - 3*k) / 16.;
          input->magnetic[index] = (c + 2) * (1 + 3*i - j + 2*k) / 32.;
        }
  Allocation allocation;
  require(Traits::succeeded(Allocation::create(sizeof(Fixture<P>), allocation)), "diagnostics allocation failed");
  auto* device = reinterpret_cast<Fixture<P>*>(allocation.raw_base());
  FieldOwner<typename P::Storage::Electric, 3> electric{{device->electric}, domain.field_neighborhood.local_partition().local_cells()};
  FieldOwner<typename P::Storage::Magnetic, 3> magnetic{{device->magnetic}, domain.field_neighborhood.local_partition().local_cells()};
  FieldOwner<double, 10> moments{{device->moments}, domain.field_neighborhood.local_partition().local_cells()};
  ParticleOwner<P> particles{{&device->particles, capacity}};
  diag::ScalarDiagnostics<P, Backend> workspace(domain, 1, true, properties, diagnostic_executor,
      diagnostic_executor);
  lpfx::device::DeviceEvent<Backend> input_ready;
  require(Traits::succeeded(lpfx::device::DeviceEvent<Backend>::create(input_ready)),
      "diagnostics input event allocation failed");
  IdentityCollective collective;
  tpic::ecsim::WallFaces volume;
  require(tpic::ecsim::try_prepare_wall_faces(domain, volume) == lpfx::ErrorCode::success,
      "diagnostics quadrature preparation failed");
  int epoch = 0;
  for (const unsigned count : {0U, 1U, 31U, 32U, 33U, 257U, capacity}) {
    particles.data.count = count;
    // Publish changing fields/velocities from the producer stream every epoch,
    // including the empty initial state. Reads may overlap independent output
    // work, but scalar collection must finish before the next state upload.
    input->particles.u[0] = (epoch + 1) / 8.;
    input->electric[0] = (epoch + 1) / 16.;
    upload(allocation, *input, executor);
    require(Traits::succeeded(input_ready.record(executor)) &&
        Traits::succeeded(input_ready.wait_on(diagnostic_executor)),
        "diagnostics input epoch dependency failed");
    workspace.enqueue_field_energy(electric, magnetic);
    workspace.enqueue_species_scalars(0, particles, -2.);
    require(diag::enqueue_deposit_output_moments<P>(particles, moments, tpic::data::prepare_particle_cell_map(domain),
        domain.inverse_cell_volume, -2., 1, &device->invalid,
        executor) == lpfx::ErrorCode::success, "concurrent output launch rejected");
    const bool final_state = count == capacity;
    const auto sample = workspace.host_blocking_finish(collective, epoch, epoch * .25, final_state);
    require(sample.cycle == epoch && sample.time == epoch * .25 &&
        sample.final_state == final_state, "diagnostics returned the wrong epoch");
    double expected[7]{};
    for (unsigned index = 0; index < count; ++index) {
      const auto& p = input->particles;
      const double q = p.q[index], mass = q / -2.;
      const double u = p.u[index], v = p.v[index], w = p.w[index];
      expected[2] += .5 * mass * (u*u + v*v + w*w);
      expected[3] += q; expected[4] += mass*u; expected[5] += mass*v; expected[6] += mass*w;
    }
    for (int kind = 0; kind < 2; ++kind)
      for (int k = 0; k < nz + (kind == 0 && volume.upper[2]); ++k)
        for (int j = 0; j < ny + (kind == 0 && volume.upper[1]); ++j)
          for (int i = 0; i < nx + (kind == 0 && volume.upper[0]); ++i)
            for (int c = 0; c < 3; ++c) {
              const int index = c * nodes + i + pitch_x * (j + pitch_y * k);
              const double f = kind == 0 ? double(input->electric[index]) : double(input->magnetic[index]);
              expected[kind] += f*f / (kind == 0 ? volume.dual_volume_factor(i,j,k) : 1) /
                  (8 * std::numbers::pi_v<double>);
            }
    for (int component = 0; component < 7; ++component)
      near(sample.values[component], expected[component], "fused scalar result differs from host");

    // Deliberately few blocks exercise each thread's grid-stride loop as well
    // as block tails independently of the production occupancy heuristic.
    lpfx::device::launch_or_throw<diag::kernels::species_scalar_partials_kernel<
        P, decltype(particles.cview())>>("grid-stride scalar partials launch", executor,
        lpfx::device::LaunchGrid{2, lpfx::device::default_block_threads},
        particles.cview(), device->partials, -.5);
    lpfx::device::launch_or_throw<tpic::device::kernels::reduce_block_partials_kernel<double>>(
        "grid-stride scalar reduction launch", executor,
        lpfx::device::LaunchGrid{5, lpfx::device::default_block_threads},
        device->partials, device->sums, unsigned(tpic::data::contribution_count), 2U, 5U);
    download(*result, allocation, executor);
    for (int component = 0; component < 5; ++component)
      near(result->sums[component], expected[component + 2], "grid-stride scalar sum differs");
    require(result->invalid == 0, "concurrent output produced invalid flag");
    ++epoch;
  }
  // A bounded helper must support every admitted species count, reuse scratch
  // across species/epochs, and preserve the serial reduction arithmetic. The
  // helper gets a separate input dependency, and finish is the only host join.
  {
    diag::ScalarDiagnostics<P, Backend> disabled(domain, tpic::config::max_real_species, false,
        properties, diagnostic_executor, helper_executor);
    require(!disabled.enabled() && !disabled.parallel_reductions() &&
        disabled.scalar_count() == 0 &&
        disabled.species_executor(0).stream() == diagnostic_executor.stream(),
        "disabled diagnostics activated a helper lane");
    diag::ScalarDiagnostics<P, Backend> same_lane(domain, 1, true,
        properties, diagnostic_executor, diagnostic_executor);
    require(!same_lane.parallel_reductions() &&
        same_lane.species_executor(0).stream() == diagnostic_executor.stream(),
        "same-executor diagnostics activated a helper lane");
    rejects([&] { (void)same_lane.species_executor(1); }, "invalid species executor index");
  }
  const unsigned population_sizes[]{0, 1, 31, 32, 33, 257, capacity};
  for (std::size_t ns = 1; ns <= static_cast<std::size_t>(tpic::config::max_real_species); ++ns) {
    diag::ScalarDiagnostics<P, Backend> parallel(domain, ns, true, properties,
        diagnostic_executor, helper_executor);
    diag::ScalarDiagnostics<P, Backend> serial(domain, ns, true, properties,
        diagnostic_executor, diagnostic_executor);
    require(parallel.parallel_reductions() && !serial.parallel_reductions() &&
        parallel.scalar_count() == tpic::data::field_scalar_count + tpic::data::species_scalar_count * ns,
        "diagnostics lane/count admission mismatch");
    for (int cycle = 0; cycle < 3; ++cycle) {
      input->particles.u[0] = (cycle + 1) / 16.;
      input->electric[0] = (ns + cycle + 1) / 32.;
      upload(allocation, *input, executor);
      require(Traits::succeeded(input_ready.record(executor)) &&
          Traits::succeeded(input_ready.wait_on(diagnostic_executor)) &&
          Traits::succeeded(input_ready.wait_on(helper_executor)),
          "two-lane diagnostics input epoch dependency failed");
      parallel.enqueue_field_energy(electric, magnetic);
      serial.enqueue_field_energy(electric, magnetic);
      rejects([&] { parallel.enqueue_field_energy(electric, magnetic); }, "field producer already queued");
      rejects([&] { (void)parallel.host_blocking_finish(collective, cycle, cycle * .25, false); },
          "incomplete scalar producers");
      std::array<double, tpic::data::max_scalar_count> expected{};
      for (int kind = 0; kind < 2; ++kind)
        for (int k = 0; k < nz + (kind == 0 && volume.upper[2]); ++k)
          for (int j = 0; j < ny + (kind == 0 && volume.upper[1]); ++j)
            for (int i = 0; i < nx + (kind == 0 && volume.upper[0]); ++i)
              for (int c = 0; c < 3; ++c) {
                const int index = c * nodes + i + pitch_x * (j + pitch_y * k);
                const double f = kind == 0 ? double(input->electric[index]) : double(input->magnetic[index]);
                expected[kind] += f*f / (kind == 0 ? volume.dual_volume_factor(i,j,k) : 1) /
                    (8 * std::numbers::pi_v<double>);
              }
      for (std::size_t ordinal = 0; ordinal < ns; ++ordinal) {
        // Reverse producer order on alternate epochs; output slots and scratch
        // ownership depend on species identity, never submission order.
        const std::size_t index = cycle % 2 ? ns - ordinal - 1 : ordinal;
        particles.data.count = population_sizes[(index + cycle) % std::size(population_sizes)];
        const double qom = -2. * (index + 1);
        require(parallel.species_executor(index).stream() ==
            (index % 2 == 0 ? helper_executor.stream() : diagnostic_executor.stream()),
            "diagnostics species assigned to wrong executor");
        parallel.enqueue_species_scalars(index, particles, qom);
        serial.enqueue_species_scalars(index, particles, qom);
        rejects([&] { parallel.enqueue_species_scalars(index, particles, qom); }, "species producer already queued");
        const auto offset = tpic::data::field_scalar_count + tpic::data::species_scalar_count * index;
        for (unsigned particle = 0; particle < particles.size(); ++particle) {
          const auto& p = input->particles;
          const double q = p.q[particle], mass = q / qom;
          const double u = p.u[particle], v = p.v[particle], w = p.w[particle];
          expected[offset] += .5 * mass * (u*u + v*v + w*w);
          expected[offset + 1] += q;
          expected[offset + 2] += mass*u;
          expected[offset + 3] += mass*v;
          expected[offset + 4] += mass*w;
        }
      }
      rejects([&] { (void)parallel.host_blocking_finish(collective, -1, 0., false); }, "invalid epoch");
      const auto sample = parallel.host_blocking_finish(collective, cycle, cycle * .25, cycle == 2);
      const auto serial_values = serial.host_blocking_finish(collective, cycle, cycle * .25, cycle == 2);
      require(sample.species_count == ns && sample.cycle == cycle &&
          sample.time == cycle * .25 && sample.final_state == (cycle == 2),
          "parallel diagnostics returned the wrong epoch or species count");
      for (std::size_t component = 0; component < parallel.scalar_count(); ++component) {
        near(sample.values[component], expected[component], "parallel scalar result differs from host");
        require(sample.values[component] == serial_values.values[component],
            "parallel scalar result differs from serial reduction arithmetic");
      }
      rejects([&] { (void)parallel.host_blocking_finish(collective, cycle, cycle * .25, false); },
          "incomplete scalar producers");
    }
  }
  particles.data.count = capacity;
  for (const std::uint16_t mask : {std::uint16_t(0), std::uint16_t(1), std::uint16_t(15),
       std::uint16_t(1U << 6), tpic::data::all_output_moments}) {
    upload(allocation, *input, executor);
    require(diag::enqueue_deposit_output_moments<P>(particles, moments, tpic::data::prepare_particle_cell_map(domain),
        domain.inverse_cell_volume, -2., mask, &device->invalid,
        executor) == lpfx::ErrorCode::success, "selected output launch rejected");
    download(*result, allocation, executor);
    require(result->invalid == 0, "selected output produced invalid flag");
    std::array<double, 10 * nodes> expected{};
    for (unsigned i = 0; i < capacity; ++i) {
      const auto& p = input->particles;
      const double q = p.q[i], m = q / -2., u = p.u[i], v = p.v[i], w = p.w[i];
      const double amount[10]{q, q*u, q*v, q*w, m*u*u, m*u*v, m*u*w, m*v*v, m*v*w, m*w*w};
      const auto cell = cells.decode(p.cell[i]);
      const auto weights = tpic::ecsim::make_cic_weights<double>(p.rx[i], p.ry[i], p.rz[i]);
      for (int corner = 0; corner < 8; ++corner) {
        const auto node = tpic::ecsim::cic_corner(cell, corner);
        const int index = node.i + pitch_x * (node.j + pitch_y * node.k);
        for (int c = 0; c < 10; ++c)
          if (mask & (1U << c)) expected[c * nodes + index] += amount[c] * weights.value[corner];
      }
    }
    for (int index = 0; index < 10 * nodes; ++index)
      near(result->moments[index], expected[index], "selected output CIC node or padding mismatch");
  }
  // A velocity outside a rho-only consumer is not evaluated. Requested J
  // retains validation; invalid fractions remain errors for every selection.
  input->particles.u[0] = std::numeric_limits<typename P::Storage::Velocity>::quiet_NaN();
  {
    diag::ScalarDiagnostics<P, Backend> failed(domain, 1, true, properties,
        diagnostic_executor, helper_executor);
    upload(allocation, *input, executor);
    require(Traits::succeeded(input_ready.record(executor)) &&
        Traits::succeeded(input_ready.wait_on(diagnostic_executor)) &&
        Traits::succeeded(input_ready.wait_on(helper_executor)),
        "failed diagnostics input dependency failed");
    failed.enqueue_field_energy(electric, magnetic);
    failed.enqueue_species_scalars(0, particles, -2.);
    rejects([&] { (void)failed.host_blocking_finish(collective, 0, 0., false); }, "nonfinite global scalar");
    // Failed collection must not release its producer slots for a new epoch.
    rejects([&] { failed.enqueue_field_energy(electric, magnetic); }, "field producer already queued");
    rejects([&] { failed.enqueue_species_scalars(0, particles, -2.); }, "species producer already queued");
  }
  for (const std::uint16_t mask : {std::uint16_t(1), std::uint16_t(2)}) {
    upload(allocation, *input, executor);
    require(diag::enqueue_deposit_output_moments<P>(particles, moments, tpic::data::prepare_particle_cell_map(domain),
        domain.inverse_cell_volume, -2., mask, &device->invalid,
        executor) == lpfx::ErrorCode::success, "invalid-output launch rejected");
    download(*result, allocation, executor);
    require(bool(result->invalid) == (mask == 2), "requested-output validation mismatch");
  }
  // J remains finite when unrequested pressure would overflow. Its absence
  // must remove both the work and that irrelevant overflow failure.
  input->particles.u[0] = 1e5;
  input->particles.q[0] = 1e300;
  for (const std::uint16_t mask : {std::uint16_t(15), std::uint16_t(16)}) {
    upload(allocation, *input, executor);
    (void)diag::enqueue_deposit_output_moments<P>(particles, moments, tpic::data::prepare_particle_cell_map(domain),
        domain.inverse_cell_volume, -2., mask, &device->invalid, executor);
    download(*result, allocation, executor);
    require(bool(result->invalid) == (mask == 16), "unselected pressure overflow was evaluated");
  }
  input->particles.rx[0] = 1;
  upload(allocation, *input, executor);
  (void)diag::enqueue_deposit_output_moments<P>(particles, moments, tpic::data::prepare_particle_cell_map(domain),
        domain.inverse_cell_volume, -2., 1, &device->invalid, executor);
  download(*result, allocation, executor);
  require(result->invalid != 0, "rho-only output accepted invalid coordinate");
  require(diag::enqueue_deposit_output_moments<P>(particles, moments, tpic::data::prepare_particle_cell_map(domain),
        domain.inverse_cell_volume, -2., 1U << 10, &device->invalid,
      executor) == lpfx::ErrorCode::invalid_argument, "invalid selection accepted");
}
bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}

// One rank with walls on every face: no exchange has a peer.
struct NoPeerExchange {
  lpfx::Status begin() { return lpfx::Status::accepted(); }
  lpfx::Status wait() { return lpfx::Status::success(); }
  lpfx::Status conclude() { return lpfx::Status::success(); }
};
enum class GaussWalls { pec, pmc, mixed };
double uniform(std::uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return double(state >> 11) * 0x1.0p-53;
}

// The host evaluation of the residual (docs/user/gauss_law.md),
// written independently of the kernels: CIC density, dual volume, zero on PEC
// wall nodes, the deck's smoothing with signed wall ghosts, the cell mean and the
// node-to-cell divergence.
struct GaussHost {
  std::array<int, 3> n{};
  std::array<double, 3> d{};
  double inverse_volume{};
  std::array<bool, 3> pec_lower{}, pec_upper{};
  int passes{};
  bool binomial{};
  double alpha{};

  [[nodiscard]] int node(int i, int j, int k) const { return (i + 1) + (n[0] + 3) * ((j + 1) + (n[1] + 3) * (k + 1)); }
  [[nodiscard]] int cell(int i, int j, int k) const { return i + n[0] * (j + n[1] * k); }
  [[nodiscard]] std::size_t nodes() const { return std::size_t(n[0] + 3) * (n[1] + 3) * (n[2] + 3); }
  [[nodiscard]] std::size_t cells() const { return std::size_t(n[0]) * n[1] * n[2]; }
  template <class F> void each_node(F&& f) const {
    for (int k = 0; k <= n[2]; ++k) for (int j = 0; j <= n[1]; ++j) for (int i = 0; i <= n[0]; ++i) f(i, j, k);
  }
  template <class F> void each_cell(F&& f) const {
    for (int k = 0; k < n[2]; ++k) for (int j = 0; j < n[1]; ++j) for (int i = 0; i < n[0]; ++i) f(i, j, k);
  }
  void rows(std::vector<double>& a) const {
    each_node([&](int i, int j, int k) {
      const int at[3] = {i, j, k};
      for (int axis = 0; axis < 3; ++axis)
        if ((pec_lower[axis] && at[axis] == 0) || (pec_upper[axis] && at[axis] == n[axis])) a[node(i, j, k)] = 0;
    });
  }
  void ghosts(std::vector<double>& a) const {
    for (int axis = 0; axis < 3; ++axis) {
      const int u = axis == 0 ? 1 : 0, v = axis == 2 ? 1 : 2;
      for (int b = -1; b <= n[v] + 1; ++b)
        for (int c = -1; c <= n[u] + 1; ++c) {
          int ghost[3], source[3];
          ghost[u] = source[u] = c;
          ghost[v] = source[v] = b;
          ghost[axis] = -1; source[axis] = 1;
          a[node(ghost[0], ghost[1], ghost[2])] =
              (pec_lower[axis] ? -1 : 1) * a[node(source[0], source[1], source[2])];
          ghost[axis] = n[axis] + 1; source[axis] = n[axis] - 1;
          a[node(ghost[0], ghost[1], ghost[2])] =
              (pec_upper[axis] ? -1 : 1) * a[node(source[0], source[1], source[2])];
        }
    }
  }
  // 4*pi times the cell mean of the processed density of (cell, fractions, q) records.
  [[nodiscard]] std::vector<double> charge(const std::vector<std::array<double, 7>>& particles) const {
    std::vector<double> rho(nodes()), next(nodes());
    for (const auto& p : particles)
      for (int corner = 0; corner < 8; ++corner) {
        const int a = corner & 1, b = (corner >> 1) & 1, c = (corner >> 2) & 1;
        rho[node(int(p[0]) + a, int(p[1]) + b, int(p[2]) + c)] += p[6] * inverse_volume *
            (a ? p[3] : 1 - p[3]) * (b ? p[4] : 1 - p[4]) * (c ? p[5] : 1 - p[5]);
      }
    each_node([&](int i, int j, int k) {
      rho[node(i, j, k)] *= double(1 << ((i == 0) + (i == n[0]) + (j == 0) + (j == n[1]) + (k == 0) + (k == n[2])));
    });
    for (int pass = 0; pass < passes; ++pass) {
      rows(rho);
      ghosts(rho);
      rows(rho);
      each_node([&](int i, int j, int k) {
        double faces = 0, edges = 0, corners = 0;
        for (int dk = -1; dk <= 1; ++dk) for (int dj = -1; dj <= 1; ++dj) for (int di = -1; di <= 1; ++di) {
          const int order = std::abs(di) + std::abs(dj) + std::abs(dk);
          const double value = rho[node(i + di, j + dj, k + dk)];
          if (order == 1) faces += value; else if (order == 2) edges += value; else if (order == 3) corners += value;
        }
        const double center = rho[node(i, j, k)];
        next[node(i, j, k)] = binomial ? (8 * center + 4 * faces + 2 * edges + corners) / 64
                                       : alpha * center + (1 - alpha) / 6 * faces;
      });
      std::swap(rho, next);
    }
    rows(rho);
    std::vector<double> c(cells());
    each_cell([&](int i, int j, int k) {
      double sum = 0;
      for (int corner = 0; corner < 8; ++corner)
        sum += rho[node(i + (corner & 1), j + ((corner >> 1) & 1), k + ((corner >> 2) & 1))];
      c[cell(i, j, k)] = std::numbers::pi / 2 * sum;
    });
    return c;
  }
  // divC of a nodal vector stored per component as [component][node(i, j, k)].
  [[nodiscard]] std::vector<double> divergence(const std::array<std::vector<double>, 3>& e) const {
    std::vector<double> div(cells());
    each_cell([&](int i, int j, int k) {
      double sum = 0;
      for (int axis = 0; axis < 3; ++axis) {
        const int u = axis == 0 ? 1 : 0, v = axis == 2 ? 1 : 2;
        double difference = 0;
        for (int b = 0; b < 2; ++b)
          for (int c = 0; c < 2; ++c) {
            int lo[3] = {i, j, k};
            lo[u] += b;
            lo[v] += c;
            int hi[3] = {lo[0], lo[1], lo[2]};
            hi[axis] += 1;
            difference += e[axis][node(hi[0], hi[1], hi[2])] - e[axis][node(lo[0], lo[1], lo[2])];
          }
        sum += difference / (4 * d[axis]);
      }
      div[cell(i, j, k)] = sum;
    });
    return div;
  }
  [[nodiscard]] bool wall_cell(int i, int j, int k) const {
    return i == 0 || i == n[0] - 1 || j == 0 || j == n[1] - 1 || k == 0 || k == n[2] - 1;
  }
};

template <class P> void gauss(Executor executor, const lpfx::device::DeviceProperties& properties,
                              GaussWalls walls, int smoothing) {
  using Fields = tpic::data::FieldSchema<P>;
  tpic::config::SimulationParameters parameters;
  parameters.case_name = "Uniform";
  parameters.dt = .1;
  parameters.ncycles = 1;
  parameters.grid.nxc = 6; parameters.grid.nyc = 5; parameters.grid.nzc = 4;
  parameters.grid.lx = 3; parameters.grid.ly = 2; parameters.grid.lz = 1.6;
  parameters.topology.periodic_x = parameters.topology.periodic_y = parameters.topology.periodic_z = false;
  parameters.topology.periodic_x_particles = parameters.topology.periodic_y_particles =
      parameters.topology.periodic_z_particles = false;
  parameters.output.field_cycle = parameters.output.restart_cycle = 0;
  parameters.ns = 1;
  parameters.species.resize(1);
  parameters.species[0].q_over_m = -1;
  parameters.species[0].rho_init = 1;
  parameters.species[0].npcelx = parameters.species[0].npcely = parameters.species[0].npcelz = 1;
  if (walls == GaussWalls::pmc)
    parameters.bc.em_x_left = parameters.bc.em_x_right = parameters.bc.em_y_left = parameters.bc.em_y_right =
        parameters.bc.em_z_left = parameters.bc.em_z_right = 1;
  else if (walls == GaussWalls::mixed)
    parameters.bc.em_x_left = parameters.bc.em_y_right = parameters.bc.em_z_left = 1;
  if (smoothing) {
    parameters.ecsim_smoothing_mode = tpic::config::EcsimSmoothingMode::EnergyConserving;
    parameters.ecsim_smoothing_operator = smoothing == 1 ? tpic::config::EcsimSmoothingOperator::Face7
                                                         : tpic::config::EcsimSmoothingOperator::Binomial27;
    parameters.smooth = .5;
    parameters.smooth_niter = 2;
  }
  const auto input = tpic::simulation::prepare_input(std::move(parameters));
  require(input.host_admitted(), "Gauss fixture not admitted");
  const auto& domain = input.domain;
  GaussHost host;
  for (int axis = 0; axis < 3; ++axis) {
    host.n[axis] = domain.local_cells[axis];
    host.d[axis] = domain.spacing[axis];
    // PreparedPhysics lists right, then left, per axis.
    host.pec_upper[axis] = input.physics.field_boundary[2 * axis] ==
                           tpic::config::FieldBoundary::perfect_electric_conductor;
    host.pec_lower[axis] = input.physics.field_boundary[2 * axis + 1] ==
                           tpic::config::FieldBoundary::perfect_electric_conductor;
  }
  host.inverse_volume = domain.inverse_cell_volume;
  host.passes = smoothing ? 2 : 0;
  host.binomial = smoothing == 2;
  host.alpha = .5;

  std::uint64_t state = 17 + 3 * unsigned(walls) + 7 * unsigned(smoothing);
  // Three committed E: the host keeps the stored values.
  std::array<std::array<std::vector<double>, 3>, 3> e;
  const auto electric = [&](std::array<std::vector<double>, 3>& values) {
    auto made = lpfx::field::make<typename Fields::E>(lpfx::field::options{
        .cells = {host.n[0], host.n[1], host.n[2]}, .halo = tpic::config::PreparedDomain::field_halo});
    require(made.has_value(), "Gauss field allocation failed");
    auto image = std::move(made).value();
    const auto fill = [&](auto view, std::vector<double>& stored) {
      stored.assign(host.nodes(), 0.0);
      host.each_node([&](int i, int j, int k) {
        using Value = std::remove_reference_t<decltype(view.core_data()[0])>;
        const Value value = Value(2 * uniform(state) - 1);
        view.core_data()[i + view.stride_y() * j + view.stride_z() * k] = value;
        stored[host.node(i, j, k)] = double(value);
      });
    };
    fill(image.template get<typename Fields::Ex>(), values[0]);
    fill(image.template get<typename Fields::Ey>(), values[1]);
    fill(image.template get<typename Fields::Ez>(), values[2]);
    return tpic::data::upload_field(image, executor);
  };
  auto e0 = electric(e[0]);
  auto e1 = electric(e[1]);
  auto e2 = electric(e[2]);

  // Two particle sets with their stored cell, fractions and charge.
  constexpr unsigned count = 300;
  const auto cell_map = tpic::data::prepare_particle_cell_map(domain);
  auto particle_state = std::make_unique<ParticleState<P>>();
  std::array<std::vector<std::array<double, 7>>, 2> records;
  Allocation particle_allocation;
  require(Traits::succeeded(Allocation::create(sizeof(ParticleState<P>), particle_allocation)),
          "Gauss particle allocation failed");
  ParticleOwner<P> particles{{reinterpret_cast<ParticleState<P>*>(particle_allocation.raw_base()), count}};
  const auto load = [&](std::vector<std::array<double, 7>>& stored) {
    auto& s = *particle_state;
    stored.clear();
    for (unsigned index = 0; index < count; ++index) {
      const int i = int(uniform(state) * host.n[0]), j = int(uniform(state) * host.n[1]),
                k = int(uniform(state) * host.n[2]);
      s.rx[index] = static_cast<std::remove_reference_t<decltype(s.rx[0])>>(uniform(state) * .999);
      s.ry[index] = static_cast<std::remove_reference_t<decltype(s.ry[0])>>(uniform(state) * .999);
      s.rz[index] = static_cast<std::remove_reference_t<decltype(s.rz[0])>>(uniform(state) * .999);
      s.q[index] = static_cast<std::remove_reference_t<decltype(s.q[0])>>(uniform(state) - .3);
      s.cell[index] = cell_map.encode(i, j, k);
      stored.push_back({double(i), double(j), double(k), double(s.rx[index]), double(s.ry[index]),
                        double(s.rz[index]), double(s.q[index])});
    }
    upload(particle_allocation, s, executor);
  };

  Allocation invalid_allocation;
  require(Traits::succeeded(Allocation::create(sizeof(unsigned), invalid_allocation)), "Gauss flag allocation failed");
  unsigned invalid = 0;
  upload(invalid_allocation, invalid, executor);
  auto* flag = reinterpret_cast<unsigned*>(invalid_allocation.raw_base());

  diag::GaussResidual<P, Backend> residual(domain, input.physics, input.parameters, true, properties, executor);
  // The stored per-cell residual of the last finished sample, x fastest.
  const auto stored = [&] {
    const auto view = residual.residual.view().template component<0>();
    std::vector<double> values(host.cells());
    for (int k = 0; k < host.n[2]; ++k)
      for (int j = 0; j < host.n[1]; ++j)
        tpic::test::download_bytes(values.data() + host.cell(0, j, k),
                                   view.core_data() + j * view.stride_y() + k * view.stride_z(),
                                   std::size_t(host.n[0]) * sizeof(double), executor);
    return values;
  };
  NoPeerExchange sum, charge_ghosts, smooth_ghosts;
  IdentityCollective collective;
  // The pipeline's order: the field kernel and the depositions read the state,
  // the finished sample is collected before the next one completes its charge.
  load(records[0]);
  residual.enqueue_field(e0, true, false);
  residual.enqueue_deposit(particles, -1., flag);
  residual.complete_charge(sum, charge_ghosts, smooth_ghosts);
  load(records[1]);
  residual.enqueue_field(e1, true, true);
  residual.enqueue_deposit(particles, -1., flag);
  const auto first = residual.host_blocking_finish(collective, 0);
  const auto first_cells = stored();
  residual.complete_charge(sum, charge_ghosts, smooth_ghosts);
  residual.enqueue_field(e2, false, true);
  const auto second = residual.host_blocking_finish(collective, 1);
  const auto second_cells = stored();
  download(invalid, invalid_allocation, executor);
  require(invalid == 0, "Gauss deposition flagged a valid particle");

  const auto div0 = host.divergence(e[0]), div1 = host.divergence(e[1]), div2 = host.divergence(e[2]);
  const auto c0 = host.charge(records[0]), c1 = host.charge(records[1]);
  std::array<double, 4> sums0{}, sums1{};
  double wall_cells = 0;
  host.each_cell([&](int i, int j, int k) {
    const int at = host.cell(i, j, k);
    const double r0 = (div0[at] + div1[at]) / 2 - c0[at], r1 = (div1[at] + div2[at]) / 2 - c1[at];
    near(first_cells[at], r0, "Gauss stored residual differs from the host evaluation");
    near(second_cells[at], r1, "Gauss stored residual differs from the host evaluation");
    sums0[0] += r0 * r0;
    sums1[0] += r1 * r1;
    sums1[1] += (r1 - r0) * (r1 - r0);
    if (host.wall_cell(i, j, k)) {
      sums0[2] += r0 * r0;
      sums1[2] += r1 * r1;
      ++wall_cells;
    }
    sums0[3] += c0[at] * c0[at];
    sums1[3] += c1[at] * c1[at];
  });
  const double cells = double(host.cells());
  for (std::size_t slot = 0; slot < 4; ++slot) {
    const double scale = slot == 2 ? wall_cells : cells;
    near(first.rms[slot], std::sqrt(sums0[slot] / scale), "Gauss first sample differs from the host evaluation");
    near(second.rms[slot], std::sqrt(sums1[slot] / scale), "Gauss second sample differs from the host evaluation");
  }
  require(first.cycle == 0 && second.cycle == 1 && first.time == .05 && std::abs(second.time - .15) < 1e-15,
          "Gauss sample cycles or times wrong");
}
} // namespace
int main() {
  lpfx::device::DeviceProperties properties{};
  const auto queried = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(queried)) {
    if (runtime_unavailable(queried)) {
      std::cout << "TPIC diagnostics skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto stream = Traits::null_stream, diagnostic_stream = Traits::null_stream,
      helper_stream = Traits::null_stream;
  if (!Traits::succeeded(Traits::create_stream(stream))) return 1;
  if (!Traits::succeeded(Traits::create_stream(diagnostic_stream))) {
    static_cast<void>(Traits::destroy_stream(stream));
    return 1;
  }
  if (!Traits::succeeded(Traits::create_stream(helper_stream))) {
    static_cast<void>(Traits::destroy_stream(diagnostic_stream));
    static_cast<void>(Traits::destroy_stream(stream));
    return 1;
  }
  int status = 0;
  try {
    const Executor executor{stream};
    const Executor diagnostic_executor{diagnostic_stream};
    const Executor helper_executor{helper_stream};
    run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor, diagnostic_executor, helper_executor, properties);
    run<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor, diagnostic_executor, helper_executor, properties);
    run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor, diagnostic_executor, helper_executor, properties);
    for (const auto walls : {GaussWalls::pec, GaussWalls::pmc, GaussWalls::mixed})
      for (const int smoothing : {0, 1, 2}) {
        gauss<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(diagnostic_executor, properties, walls, smoothing);
        gauss<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(
            diagnostic_executor, properties, walls, smoothing);
      }
    std::cout << "TPIC fused diagnostics and selected snapshot checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    static_cast<void>(Traits::synchronize_stream(stream));
    static_cast<void>(Traits::synchronize_stream(diagnostic_stream));
    static_cast<void>(Traits::synchronize_stream(helper_stream));
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(helper_stream))) status = 1;
  if (!Traits::succeeded(Traits::destroy_stream(diagnostic_stream))) status = 1;
  if (!Traits::succeeded(Traits::destroy_stream(stream))) status = 1;
  return status;
}
