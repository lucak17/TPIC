#include <tpic/simulation/planning/run_plan.hpp>
// Differential scheduling test: the same operator kernels on one stream and
// on main/mass/interior streams see repeated changing input/matrix epochs. Closed
// physical walls need no peer payload, so fake empty exchanges exercise the
// actual boundary kernels without requiring MPI in this test.
#include <tpic/communication/mass_halo/mass_halo_plan.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/ecsim/gmres/gmres_device.hpp>
#include <tpic/ecsim/maxwell/maxwell_device.hpp>
#include <tpic/device/device_validation.hpp>
#include <tpic/simulation/actions/field_advance_actions_device.hpp>
#include "../support/test_builders.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Schedule = tpic::ecsim::MaxwellApplySchedule;
using Smoother = tpic::config::EcsimSmoothingOperator;
using tpic::ecsim::TransferDirection;

void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }
void check(Traits::error_type error) { require(Traits::succeeded(error), "Maxwell overlap backend failure"); }

template <class T> struct Buffer {
  lpfx::device::DeviceAllocation<Backend> memory;
  std::size_t count;
  explicit Buffer(std::size_t n) : count(n) { check(decltype(memory)::create(n * sizeof(T), memory)); }
  T* data() { return reinterpret_cast<T*>(memory.raw_base()); }
  std::vector<T> download(Executor executor) {
    std::vector<T> result(count);
    check(Traits::copy_device_to_host_async(result.data(), data(), count * sizeof(T), executor.stream()));
    check(Traits::synchronize_stream(executor.stream()));
    return result;
  }
};
__global__ void delay_kernel(const long long ticks) {
  const auto started = clock64();
  while (clock64() - started < ticks) {}
}

struct EmptyExchange {
  unsigned begins{}, waits{}, concludes{};
  bool active{}, completed{}, polled{};
  Executor executor{Traits::null_stream};
  unsigned* active_exchanges{};

  EmptyExchange() = default;
  EmptyExchange(Executor queue, unsigned& gate) : executor(queue), active_exchanges(&gate) {}
  lpfx::Status begin() {
    if (active || (active_exchanges && *active_exchanges))
      return lpfx::Status::failure(lpfx::ErrorCode::invalid_state);
    if (executor.bound()) {
      // Hold back the physical ghost closure while helper kernels are eligible
      // to run. This models stream work, not a native communication benchmark.
      delay_kernel<<<1, 1, 0, executor.stream()>>>(12000);
      check(Traits::take_last_launch_error());
    }
    ++begins;
    active = true;
    completed = polled = false;
    if (active_exchanges) ++*active_exchanges;
    return lpfx::Status::accepted();
  }
  // The first poll finds the halo in flight, as an MPI halo still on the network.
  lpfx::Status progress() {
    if (!active) return lpfx::Status::failure(lpfx::ErrorCode::invalid_state);
    if (!std::exchange(polled, true)) return lpfx::Status::pending();
    return wait();
  }
  lpfx::Status wait() {
    if (!active) return lpfx::Status::failure(lpfx::ErrorCode::invalid_state);
    ++waits;
    completed = true;
    return lpfx::Status::success();
  }
  lpfx::Status conclude() {
    if (!active || !completed) return lpfx::Status::failure(lpfx::ErrorCode::invalid_state);
    ++concludes;
    active = false;
    if (active_exchanges) --*active_exchanges;
    return lpfx::Status::success();
  }
};

enum class Walls { pec, pmc, mixed };

auto fixture(bool smoothing, std::array<int, 3> cells = {8, 6, 4}, double theta = 1,
             Walls walls = Walls::pec, Smoother smoother = Smoother::Face7) {
  tpic::config::SimulationParameters p;
  p.case_name = "Uniform";
  p.dt = .1; p.ncycles = 1;
  p.th = theta;
  // Production admission requires two cells on a closed axis. Admit the
  // physical parameters on that supported geometry, then construct the exact
  // smaller LPFx geometry below solely to exercise low-level empty-core paths.
  // These fixtures never consume the admitted particle/initialization plans.
  p.grid.nxc = cells[0] > 1 ? cells[0] : 2;
  p.grid.nyc = cells[1] > 1 ? cells[1] : 2;
  p.grid.nzc = cells[2] > 1 ? cells[2] : 2;
  p.topology.periodic_x = p.topology.periodic_y = p.topology.periodic_z = false;
  p.topology.periodic_x_particles = p.topology.periodic_y_particles =
      p.topology.periodic_z_particles = false;
  p.output.field_cycle = p.output.restart_cycle = 0;
  p.ns = 1; p.species.resize(1);
  p.species[0].q_over_m = 1; p.species[0].rho_init = 1;
  p.species[0].npcelx = p.species[0].npcely = p.species[0].npcelz = 1;
  if (walls == Walls::pmc) {
    p.bc.em_x_left = p.bc.em_x_right = p.bc.em_y_left = p.bc.em_y_right =
        p.bc.em_z_left = p.bc.em_z_right = 1;
  } else if (walls == Walls::mixed) {
    p.bc.em_x_left = p.bc.em_y_right = p.bc.em_z_left = 1;
  }
  if (smoothing) {
    p.ecsim_smoothing_mode = tpic::config::EcsimSmoothingMode::EnergyConserving;
    p.ecsim_smoothing_operator = smoother;
    p.smooth = .5; p.smooth_niter = 2;
  }
  auto result = tpic::simulation::prepare_input(std::move(p));
  require(result.host_admitted(), "Maxwell overlap fixture not admitted");
  if (cells[0] == 1 || cells[1] == 1 || cells[2] == 1) {
    auto& domain = result.domain;
    tpic::config::PreparedDomain::PartitionNeighborhood neighborhood;
    require(lpfx::try_prepare_cartesian_partition_neighborhood3d<
                tpic::config::PreparedDomain::IndexPolicy>(
                cells[0], cells[1], cells[2],
                domain.field_neighborhood.topology(), neighborhood) ==
                lpfx::ErrorCode::success,
            "thin Maxwell fixture LPFx partition neighborhood failed");
    domain.field_neighborhood = neighborhood;
    result.parameters.grid.nxc = cells[0];
    result.parameters.grid.nyc = cells[1];
    result.parameters.grid.nzc = cells[2];
    domain.global_cells = domain.local_cells = cells;
    tpic::test::fill_domain_counts(domain);
    for (int axis = 0; axis < 3; ++axis) domain.spacing[axis] = domain.lengths[axis] / cells[axis];
    domain.cell_volume = static_cast<double>(static_cast<long double>(domain.spacing[0]) *
                                           domain.spacing[1] * domain.spacing[2]);
    domain.inverse_cell_volume = 1.0 / domain.cell_volume;
  }
  for (auto face : result.physics.field_boundary)
    require(face == tpic::config::FieldBoundary::perfect_electric_conductor ||
            face == tpic::config::FieldBoundary::perfect_magnetic_conductor,
            "Maxwell overlap fixture has a communicating face");
  return result;
}

tpic::ecsim::StencilRegionGeometry region_geometry(int nx, int ny, int nz) {
  tpic::ecsim::StencilRegionGeometry geometry;
  require(tpic::ecsim::try_make_stencil_region_geometry(nx, ny, nz, geometry) ==
          lpfx::ErrorCode::success, "Maxwell overlap region geometry failed");
  return geometry;
}

template <class A, class B>
__global__ void fill_maxwell_inputs_kernel(A first, B second, tpic::data::GridShape shape, int epoch) {
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < unsigned(shape.count);
       n += blockDim.x * gridDim.x) {
    const int i = n % shape.nx, j = (n / shape.nx) % shape.ny, k = n / shape.nx / shape.ny;
    [&]<std::size_t... Component>(std::index_sequence<Component...>) {
      ((first.template component<Component>()(i,j,k) =
        second.template component<Component>()(i,j,k) =
            (1 + .125 * epoch) * (.1 * (Component + 1) + .015625 * i * j + .03125 * k * k)), ...);
    }(std::make_index_sequence<3>{});
  }
}
template <class Scalar> __device__ Scalar maxwell_mass_coefficient(std::size_t n, int epoch) {
  return Scalar(.0001 * (1 + .0625 * epoch) * (1 + n % 11));
}
template <class Scalar>
__global__ void fill_maxwell_mass_coefficients_kernel(Scalar* mass, tpic::data::MassMatrixLayout layout, int epoch, int poison) {
  const int i = poison == 1 && layout.nx > 2 ? 1 : 0;
  const int j = poison == 1 && layout.ny > 2 ? 1 : 0;
  const int k = poison == 1 && layout.nz > 2 ? 1 : 0;
  const auto bad = layout.entry_index(layout.vertex_index(i, j, k), 0);
  for (std::size_t n = blockIdx.x * blockDim.x + threadIdx.x; n < layout.storage_size();
       n += blockDim.x * gridDim.x)
    mass[n] = poison && n == bad ? std::numeric_limits<Scalar>::quiet_NaN() :
        maxwell_mass_coefficient<Scalar>(n, epoch);
}
// The stencils the ghost phase sends into the ghost region of each canonical direction.
struct GhostStencils {
  std::uint16_t sent[tpic::communication::mass_halo_neighbour_count];
};
GhostStencils ghost_stencils() {
  GhostStencils result{};
  for (std::size_t c = 0; c != tpic::communication::mass_halo_neighbour_count; ++c)
    result.sent[c] = tpic::communication::mass_halo_stencil_mask(
        false, lpfx::opposite_cartesian_direction(lpfx::canonical_cartesian_neighbor_directions[c]));
  return result;
}
// Overwrites the ghost records with NaN, or restores the fill's coefficients
// of the stencils the ghost phase sends; the others stay NaN.
template <class Scalar>
__global__ void fill_maxwell_mass_ghosts_kernel(Scalar* mass, tpic::data::MassMatrixLayout layout, int epoch,
                                                bool poison, GhostStencils ghosts) {
  const auto side = [](int at, int n) { return std::int8_t(at < 0 ? -1 : at >= n ? 1 : 0); };
  for (std::size_t n = blockIdx.x * blockDim.x + threadIdx.x; n < layout.storage_size();
       n += blockDim.x * gridDim.x) {
    const auto vertex = n / tpic::data::mass_entries_per_node;
    const lpfx::CartesianDirection3 toward{side(int(vertex % layout.extent_x) - 1, layout.nx),
                                           side(int(vertex / layout.extent_x % layout.extent_y) - 1, layout.ny),
                                           side(int(vertex / layout.stride_z) - 1, layout.nz)};
    if (toward.x == 0 && toward.y == 0 && toward.z == 0) continue;
    const auto stencil = n % tpic::data::mass_entries_per_node / tpic::data::mass_tensor_entries;
    if (poison) mass[n] = std::numeric_limits<Scalar>::quiet_NaN();
    else if (ghosts.sent[lpfx::cartesian_neighbor_index(toward)] >> stencil & 1U)
      mass[n] = maxwell_mass_coefficient<Scalar>(n, epoch);
  }
}
__global__ void capture_maxwell_validation_flags_kernel(const unsigned* first, const unsigned* second, unsigned* result) {
  result[0] = *first; result[1] = *second;
}

template <class View>
__global__ void fill_maxwell_cell_field_kernel(View field, tpic::data::GridShape shape, bool poison_ghosts) {
  const int nx = shape.nx + 2, ny = shape.ny + 2, nz = shape.nz + 2;
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < unsigned(nx * ny * nz);
       n += blockDim.x * gridDim.x) {
    const int i = int(n % nx) - 1, j = int((n / nx) % ny) - 1, k = int(n / nx / ny) - 1;
    const bool ghost = i < 0 || i >= shape.nx || j < 0 || j >= shape.ny || k < 0 || k >= shape.nz;
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ((field.template component<C>()(i,j,k) = poison_ghosts && ghost
          ? std::numeric_limits<double>::quiet_NaN()
          : (C + 1) * (.125 * i * j + .25 * j * k + .5 * k * i)), ...);
    }(std::make_index_sequence<3>{});
  }
}

template <class View>
__global__ void poison_maxwell_node_field_kernel(View field, tpic::data::GridShape shape) {
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < unsigned(shape.count);
       n += blockDim.x * gridDim.x) {
    const int i = n % shape.nx, j = (n / shape.nx) % shape.ny, k = n / shape.nx / shape.ny;
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ((field.template component<C>()(i,j,k) = std::numeric_limits<double>::quiet_NaN()), ...);
    }(std::make_index_sequence<3>{});
  }
}

template <class View>
__global__ void fill_maxwell_node_ghosts_kernel(View field, tpic::data::GridShape shape, bool poison) {
  const int nx = shape.nx + 2, ny = shape.ny + 2, nz = shape.nz + 2;
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < unsigned(nx * ny * nz);
       n += blockDim.x * gridDim.x) {
    const int i = int(n % nx) - 1, j = int((n / nx) % ny) - 1, k = int(n / nx / ny) - 1;
    if (i >= 0 && i < shape.nx && j >= 0 && j < shape.ny && k >= 0 && k < shape.nz) continue;
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ((field.template component<C>()(i,j,k) = poison
          ? std::numeric_limits<double>::quiet_NaN() : 123.0 + C), ...);
    }(std::make_index_sequence<3>{});
  }
}

template <class View>
__global__ void capture_maxwell_field_with_halo_kernel(View field, tpic::data::GridShape shape,
                                      double* values) {
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < unsigned(shape.count);
       n += blockDim.x * gridDim.x) {
    const int i = int(n % shape.nx) - 1, j = int((n / shape.nx) % shape.ny) - 1;
    const int k = int(n / shape.nx / shape.ny) - 1;
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ((values[C * shape.count + n] = field.template component<C>()(i, j, k)), ...);
    }(std::make_index_sequence<3>{});
  }
}

template <class View, class Scalar>
__global__ void fill_constant_field_with_halo_kernel(View field, tpic::data::GridShape shape, Scalar value) {
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < unsigned(shape.count);
       n += blockDim.x * gridDim.x) {
    const int i = int(n % shape.nx) - 1, j = int((n / shape.nx) % shape.ny) - 1;
    const int k = int(n / shape.nx / shape.ny) - 1;
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ((field.template component<C>()(i, j, k) = value), ...);
    }(std::make_index_sequence<3>{});
  }
}

template <class Precision> void interpolation_validation(Executor executor) {
  using Fields = tpic::data::FieldSchema<Precision>;
  using Magnetic = typename Precision::Storage::Magnetic;
  const auto prepared = fixture(false, {2, 2, 2});
  auto cell = tpic::data::make_zero_field<typename Fields::BCell>(prepared.domain, executor);
  auto node = tpic::data::make_zero_field<typename Fields::BNode>(prepared.domain, executor);
  Buffer<unsigned> invalid(2);
  check(Traits::memset_device_async(invalid.data(), 0, 2 * sizeof(unsigned), executor.stream()));
  for (unsigned epoch = 0; epoch < 2; ++epoch) {
    // Double's eight-term sequential sum overflows; reduced magnetic storage
    // promotes before adding, and its finite maximum still interpolates safely.
    const auto value = epoch == 0 ? std::numeric_limits<Magnetic>::max() :
                                   std::numeric_limits<Magnetic>::quiet_NaN();
    fill_constant_field_with_halo_kernel<<<2, 128, 0, executor.stream()>>>(
        cell.view(), tpic::ecsim::field_vector_shape(cell, 1), value);
    check(Traits::take_last_launch_error());
    require(tpic::ecsim::enqueue_magnetic_cell_to_node<Precision>(
        cell, node, invalid.data() + epoch, executor) == lpfx::ErrorCode::success,
        "B interpolation validation launch failed");
  }
  const auto flags = invalid.download(executor);
  require((flags[0] != 0) == std::is_same_v<Magnetic, double> && flags[1] != 0,
          "B interpolation did not report its own nonfinite node values");
}

// The B facade must reproduce the sequential numerical schedule on its fixed
// helper stream without changing either current magnetic owner before commit.
// Compare complete logical halos as well as interiors over repeated epochs;
// main's next theta fill cannot overwrite theta before the field stream reads it.
template <class Precision>
void separate_field_update(Executor main, Executor field, double theta) {
  using Fields = tpic::data::FieldSchema<Precision>;
  using Maxwell = tpic::ecsim::MaxwellOperator<Precision, Backend>;
  auto prepared = fixture(false, {8, 6, 4}, theta);
  const auto make = [&]<class Schema>() {
    return tpic::data::make_zero_field<Schema>(prepared.domain, main);
  };
  Maxwell sequential(prepared, main, main, main), separated(prepared, main, main, main);
  // The sequential baseline uses the values MaxwellOperator derives.
  using Compute = typename Precision::Compute::Electric;
  const tpic::ecsim::MaxwellParameters maxwell = prepared;
  const Compute theta_value = maxwell.theta;
  const Compute faraday = -maxwell.c * maxwell.dt;
  tpic::ecsim::NodeCellSpacing<Compute> spacing;
  require(tpic::ecsim::try_make_node_cell_spacing(prepared.domain.spacing, spacing) ==
          lpfx::ErrorCode::success, "B facade sequential spacing failed");
  tpic::data::DeviceFieldImage<Precision, Backend> expected_fields{
      make.template operator()<typename Fields::E>(), make.template operator()<typename Fields::BCell>()};
  tpic::data::DeviceFieldImage<Precision, Backend> actual_fields{
      make.template operator()<typename Fields::E>(), make.template operator()<typename Fields::BCell>()};
  auto expected_node = make.template operator()<typename Fields::BNode>();
  auto actual_node = make.template operator()<typename Fields::BNode>();
  tpic::data::GridShape nodes;
  require(tpic::data::try_make_grid_shape(prepared.domain.field_neighborhood.local_partition().local_cells(), 1, {}, nodes) ==
          lpfx::ErrorCode::success, "B facade shape failure");
  const tpic::data::GridShape cells{nodes.nx - 1, nodes.ny - 1, nodes.nz - 1,
      (nodes.nx - 1) * (nodes.ny - 1) * (nodes.nz - 1)};
  const auto cell_values = 3 * tpic::ecsim::field_vector_shape(expected_fields.magnetic_cell, 1).count;
  const auto node_values = 3 * tpic::ecsim::field_vector_shape(expected_node, 1).count;
  const auto values_per_epoch = 2 * cell_values + 3 * node_values;
  constexpr unsigned epochs = 5;
  Buffer<double> baseline(std::size_t(epochs) * values_per_epoch), concurrent(baseline.count);
  Buffer<unsigned> invalid(2), flags(2 * epochs);
  lpfx::device::DeviceEvent<Backend> ready, done;
  check(decltype(ready)::create(ready)); check(decltype(done)::create(done));
  EmptyExchange expected_theta, actual_theta;
  const auto capture = [&](const auto& owner, double* values, Executor executor) {
    capture_maxwell_field_with_halo_kernel<<<2, 128, 0, executor.stream()>>>(
        owner.view(), tpic::ecsim::field_vector_shape(owner, 1), values);
  };
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    check(Traits::memset_device_async(invalid.data(), 0, 2 * sizeof(unsigned), main.stream()));
    fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(
        expected_fields.electric.view(), actual_fields.electric.view(), nodes, epoch + 1);
    fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(sequential.theta.view(), separated.theta.view(), nodes, epoch + 2);
    fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(expected_fields.magnetic_cell.view(), cells, false);
    fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(actual_fields.magnetic_cell.view(), cells, false);
    fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(expected_node.view(), actual_node.view(), nodes, epoch + 3);
    if (epoch + 1 == epochs) {
      poison_maxwell_node_field_kernel<<<2, 128, 0, main.stream()>>>(sequential.theta.view(), nodes);
      poison_maxwell_node_field_kernel<<<2, 128, 0, main.stream()>>>(separated.theta.view(), nodes);
    }
    check(Traits::take_last_launch_error());
    sequential.complete_electric_refresh(sequential.theta, expected_theta);
    separated.complete_electric_refresh(separated.theta, actual_theta);
    check(ready.record(main)); check(ready.wait_on(field));
    auto* expected = baseline.data() + std::size_t(epoch) * values_per_epoch;
    auto* actual = concurrent.data() + std::size_t(epoch) * values_per_epoch;

    // Sequential main-stream field construction and commit, with local closed
    // walls (no native halo payload). Capture old B between the two phases.
    tpic::ecsim::enqueue_next_fields<Precision>(expected_fields.electric, expected_fields.magnetic_cell,
        sequential.theta, sequential.next_cell, spacing, theta_value, faraday,
        invalid.data(), main);
    sequential.close_magnetic_ghosts(sequential.next_cell);
    require(tpic::ecsim::enqueue_magnetic_cell_to_node<Precision>(
        sequential.next_cell, sequential.next_node, invalid.data(), main) == lpfx::ErrorCode::success,
        "B facade sequential C2N failed");
    capture(expected_fields.magnetic_cell, expected, main);
    capture(expected_node, expected + cell_values, main);
    require(tpic::ecsim::enqueue_field_vector_copy(expected_fields.magnetic_cell, sequential.next_cell,
        true, nullptr, main) == lpfx::ErrorCode::success, "B facade sequential cell copy failed");
    require(tpic::ecsim::enqueue_field_vector_copy(expected_node, sequential.next_node, false,
        nullptr, main) == lpfx::ErrorCode::success, "B facade sequential node copy failed");
    require(tpic::ecsim::enqueue_constrained_rows(expected_fields.electric, tpic::ecsim::detail::NoRowInput{},
        tpic::ecsim::prepare_field_boundary_plans(prepared.domain, prepared.physics).constrained_rows, main) ==
        lpfx::ErrorCode::success, "B facade sequential projection failed");
    capture(expected_fields.magnetic_cell, expected + cell_values + node_values, main);
    capture(expected_node, expected + 2 * cell_values + node_values, main);
    capture(expected_fields.electric, expected + 2 * cell_values + 2 * node_values, main);

    separated.enqueue_next_e_and_cell(actual_fields, invalid.data() + 1, field);
    separated.enqueue_next_cell_closure_and_node_interpolation(invalid.data() + 1, field);
    capture(actual_fields.magnetic_cell, actual, field);
    capture(actual_node, actual + cell_values, field);
    separated.enqueue_commit_to_current(actual_fields, actual_node, field);
    capture(actual_fields.magnetic_cell, actual + cell_values + node_values, field);
    capture(actual_node, actual + 2 * cell_values + node_values, field);
    capture(actual_fields.electric, actual + 2 * cell_values + 2 * node_values, field);
    check(Traits::take_last_launch_error());
    check(done.record(field)); check(done.wait_on(main));
    capture_maxwell_validation_flags_kernel<<<1, 1, 0, main.stream()>>>(invalid.data(), invalid.data() + 1, flags.data() + 2 * epoch);
    check(Traits::take_last_launch_error());
  }
  const auto expected = baseline.download(main), actual = concurrent.download(main);
  const auto validation = flags.download(main);
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    require(validation[2 * epoch] == validation[2 * epoch + 1] &&
            (validation[2 * epoch] != 0) == (epoch + 1 == epochs),
            "B facade lost field validation");
    const auto checked_count = epoch + 1 == epochs ? cell_values + node_values : values_per_epoch;
    for (int n = 0; n < checked_count; ++n) {
      const auto index = std::size_t(epoch) * values_per_epoch + n;
      require(std::isfinite(actual[index]) && actual[index] == expected[index],
              "B facade changed an old epoch, field value or logical halo");
    }
  }
}

// Script communication latency independently of the GPU. Exercise the actual B
// action's submission handoff and pending conclusion with real numerical work.
struct ProgressExchange {
  unsigned progress_limit, submission_at, conclude_limit;
  bool no_native_work;
  unsigned progresses{}, conclusions{};
  bool active{};
  lpfx::Status begin() {
    require(!active, "B action restarted an active halo");
    progresses = conclusions = 0;
    active = true;
    return lpfx::Status::accepted();
  }
  lpfx::Status progress() {
    require(active && progresses < progress_limit, "B action repeated completed halo progress");
    return ++progresses == progress_limit ? lpfx::Status::success() : lpfx::Status::pending();
  }
  auto first_native_submission() const {
    using Submission = lpfx::distributed::FieldSubmission;
    require(active, "B action observed a retired submission");
    return lpfx::Result<Submission>::from_value(no_native_work ? Submission::no_native_work :
        progresses >= submission_at ? Submission::submitted : Submission::pending);
  }
  lpfx::Status conclude() {
    require(active && progresses == progress_limit, "B action concluded an incomplete halo");
    if (++conclusions < conclude_limit) return lpfx::Status::pending();
    active = false;
    return lpfx::Status::success();
  }
};

// The registered next-cell halo the B action resolves by key.
struct ProgressPorts {
  ProgressExchange cell;
  template <lpfx::field::fixed_string Key> ProgressExchange& get() { return cell; }
};

void field_action_handoffs(Executor main, Executor field) {
  using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
  using Fields = tpic::data::FieldSchema<Precision>;
  auto prepared = fixture(false, {4, 3, 2});
  tpic::ecsim::MaxwellOperator<Precision, Backend> maxwell(prepared, main, main, main);
  tpic::data::DeviceFieldImage<Precision, Backend> fields{
      tpic::data::make_zero_field<typename Fields::E>(prepared.domain, main),
      tpic::data::make_zero_field<typename Fields::BCell>(prepared.domain, main)};
  auto magnetic = tpic::data::make_zero_field<typename Fields::BNode>(prepared.domain, main);
  lpfx::device::DeviceEvent<Backend> ready, moved_event;
  check(decltype(ready)::create(ready));
  check(decltype(moved_event)::create(moved_event));
  check(ready.record(main));
  check(moved_event.record(main));
  tpic::device::DeviceValidation<Backend> advanced;
  ProgressPorts ports{};
  tpic::simulation::FieldAdvanceActions<Precision, ProgressPorts> actions(fields, magnetic, maxwell, ports,
                                                                          main, field);
  for (const auto script : {ProgressExchange{3, 2, 3, false},
                             ProgressExchange{1, 1, 3, false},
                             ProgressExchange{3, 1, 1, false},
                             ProgressExchange{1, 1, 2, true}}) {
    auto& cell = ports.cell;
    cell = script;
    check(ready.wait_on(field));
    advanced.clear(field);
    actions.advance(advanced.data());
    bool first = false, complete = false;
    for (unsigned poll = 0; poll < 10 && !complete; ++poll) {
      const auto status = actions.progress_cell();
      if (!first && status.first_submitted)
        require(!status.complete && cell.conclusions == 0,
                "B action failed to yield submission before conclusion");
      first = first || status.first_submitted;
      complete = status.complete;
    }
    require(first && complete && !cell.active, "B action lost the cell handoff or conclusion");
    require(cell.progresses == script.progress_limit && cell.conclusions == script.conclude_limit,
            "B action skipped cell progress or pending conclusion");
    require(actions.progress_cell().complete, "B action lost retired cell completion");
    actions.close_cell(advanced.data());
    advanced.capture(field);
    while (!advanced.ready("B action produced invalid next fields")) {}
    check(moved_event.wait_on(field));
    actions.commit();
  }
  check(Traits::synchronize_stream(field.stream()));
}

// Full N2C may consume an interior epoch while another stream changes every
// source ghost. Poisoning those ghosts also exposes any accidental halo read.
void n2c_during_ghost_update(Executor main, Executor interior, std::array<int, 3> cells) {
  using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
  using Schema = tpic::ecsim::SolverFieldSchema<Precision>;
  const auto make = [&]<class Field>() {
    auto made = lpfx::field::make<Field>(lpfx::field::options{
        .cells = {cells[0], cells[1], cells[2]}, .halo = 1});
    require(made.has_value(), "N2C ghost test field allocation failed");
    auto host = std::move(made).value();
    return tpic::data::upload_field(host, main);
  };
  auto input = make.template operator()<typename Schema::Node>();
  auto early = make.template operator()<typename Schema::Cell>();
  auto late = make.template operator()<typename Schema::Cell>();
  tpic::data::GridShape nodes;
  require(tpic::data::try_make_grid_shape(input.cells(), 1, {}, nodes) ==
          lpfx::ErrorCode::success, "N2C ghost test shape failure");
  const tpic::data::GridShape cell_shape{cells[0], cells[1], cells[2],
      cells[0] * cells[1] * cells[2]};
  const tpic::ecsim::NodeCellSpacing<double> spacing{2, 3, 4};
  Buffer<double> actual(3 * cell_shape.count), expected(3 * cell_shape.count);
  lpfx::device::DeviceEvent<Backend> ready, done;
  check(decltype(ready)::create(ready)); check(decltype(done)::create(done));
  fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(input.view(), input.view(), nodes, 1);
  fill_maxwell_node_ghosts_kernel<<<2, 128, 0, main.stream()>>>(input.view(), nodes, true);
  check(Traits::take_last_launch_error());
  check(ready.record(main)); check(ready.wait_on(interior));
  const auto cell_regions = region_geometry(cells[0], cells[1], cells[2]);
  require(tpic::ecsim::enqueue_node_cell_curl<Precision>(input, early, cell_regions,
          spacing, interior) == lpfx::ErrorCode::success, "N2C early dispatch failed");
  check(done.record(interior));
  fill_maxwell_node_ghosts_kernel<<<2, 128, 0, main.stream()>>>(input.view(), nodes, false);
  check(Traits::take_last_launch_error());
  require(tpic::ecsim::enqueue_node_cell_curl<Precision>(input, late, cell_regions,
          spacing, main) == lpfx::ErrorCode::success, "N2C late dispatch failed");
  check(done.wait_on(main));
  tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
      early.view(), actual.data(), cell_shape);
  tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
      late.view(), expected.data(), cell_shape);
  check(Traits::take_last_launch_error());
  const auto result = actual.download(main), baseline = expected.download(main);
  for (std::size_t n = 0; n < result.size(); ++n)
    require(std::isfinite(result[n]) && result[n] == baseline[n],
            "N2C read ghosts or changed during ghost update");
}

// Split old-field work from a deliberately later current publication. Reusing
// all scratch across epochs checks the staged API's actual dependency boundary.
template <class Precision>
void split_rhs(Executor main, Executor helper, bool smoothing) {
  auto prepared = fixture(smoothing);
  tpic::ecsim::MaxwellOperator<Precision, Backend> sequential(prepared, main, main, main);
  tpic::ecsim::MaxwellOperator<Precision, Backend> staged(prepared, main, main, main);
  tpic::data::GridShape shape;
  require(tpic::data::try_make_grid_shape(staged.input.cells(), 1, {}, shape) ==
          lpfx::ErrorCode::success, "RHS split shape failure");
  const tpic::data::GridShape cells{shape.nx - 1, shape.ny - 1, shape.nz - 1,
      (shape.nx - 1) * (shape.ny - 1) * (shape.nz - 1)};
  constexpr unsigned epochs = 5;
  const unsigned vector_count = 3 * shape.count;
  Buffer<double> expected(std::size_t(epochs) * 2 * vector_count), actual(expected.count);
  Buffer<unsigned> invalid(2);
  EmptyExchange sequential_a, sequential_b, staged_a, staged_b;
  lpfx::device::DeviceEvent<Backend> epoch_ready, current_ready;
  check(decltype(epoch_ready)::create(epoch_ready)); check(decltype(current_ready)::create(current_ready));
  check(Traits::memset_device_async(invalid.data(), 0, 2 * sizeof(unsigned), main.stream()));
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(sequential.input.view(), staged.input.view(), shape, epoch);
    fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(sequential.next_cell.view(), cells, false);
    fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(staged.next_cell.view(), cells, false);
    fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(sequential.mass_image.view(), sequential.mass_image.view(),
                                         shape, epoch + 3);
    sequential.enqueue_rhs_fields(sequential.input, sequential.next_cell);
    sequential.complete_rhs(sequential.input, sequential.mass_image, sequential_a, sequential_b,
                          invalid.data());
    check(epoch_ready.record(main)); check(epoch_ready.wait_on(helper));
    staged.enqueue_rhs_fields(staged.input, staged.next_cell);
    fill_maxwell_inputs_kernel<<<2, 128, 0, helper.stream()>>>(staged.mass_image.view(), staged.mass_image.view(),
                                           shape, epoch + 3);
    check(current_ready.record(helper)); check(current_ready.wait_on(main));
    staged.complete_rhs(staged.input, staged.mass_image, staged_a, staged_b, invalid.data() + 1);
    auto* baseline = expected.data() + std::size_t(epoch) * 2 * vector_count;
    auto* concurrent = actual.data() + std::size_t(epoch) * 2 * vector_count;
    tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
        sequential.rhs.view(), baseline, shape);
    tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
        sequential.theta.view(), baseline + vector_count, shape);
    tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
        staged.rhs.view(), concurrent, shape);
    tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
        staged.theta.view(), concurrent + vector_count, shape);
    check(Traits::take_last_launch_error());
  }
  const auto result = actual.download(main), baseline = expected.download(main);
  for (std::size_t n = 0; n < result.size(); ++n)
    require(std::isfinite(result[n]) && result[n] == baseline[n], "Split RHS changed RHS or theta");
  const auto validation = invalid.download(main);
  require(validation[0] == 0 && validation[1] == 0, "Split RHS validation failure");
  require(sequential_a.begins == staged_a.begins && sequential_b.begins == staged_b.begins,
          "Split RHS changed smoothing halo schedule");
}

// Poison source ghosts and target nodes to expose premature halo reads, writes
// outside a region and missing shell nodes, including one-cell local extents.
void split_stencil(Executor main, Executor interior, std::array<int, 3> cells) {
  using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
  using Schema = tpic::ecsim::SolverFieldSchema<Precision>;
  using Region = tpic::ecsim::StencilRegion;
  const auto make = [&]<class Field>() {
    auto made = lpfx::field::make<Field>(lpfx::field::options{
        .cells = {cells[0], cells[1], cells[2]}, .halo = 1});
    require(made.has_value(), "C2N region test field allocation failed");
    auto host = std::move(made).value();
    return tpic::data::upload_field(host, main);
  };
  auto source = make.template operator()<typename Schema::Cell>();
  auto full = make.template operator()<typename Schema::Node>();
  auto split = make.template operator()<typename Schema::Node>();
  tpic::data::GridShape shape;
  require(tpic::data::try_make_grid_shape(source.cells(), 1, {}, shape) == lpfx::ErrorCode::success,
          "C2N region test shape failure");
  const tpic::data::GridShape cell_shape{cells[0], cells[1], cells[2],
      cells[0] * cells[1] * cells[2]};
  const tpic::ecsim::NodeCellSpacing<double> spacing{2, 3, 4};
  Buffer<double> partial(3 * shape.count), actual(3 * shape.count), expected(3 * shape.count);
  lpfx::device::DeviceEvent<Backend> ready, done;
  check(decltype(ready)::create(ready)); check(decltype(done)::create(done));
  fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(source.view(), cell_shape, true);
  poison_maxwell_node_field_kernel<<<2, 128, 0, main.stream()>>>(split.view(), shape);
  check(Traits::take_last_launch_error());
  check(ready.record(main)); check(ready.wait_on(interior));
  const auto node_regions = region_geometry(cells[0] + 1, cells[1] + 1, cells[2] + 1);
  const auto cell_regions = region_geometry(cells[0], cells[1], cells[2]);
  require(tpic::ecsim::enqueue_node_cell_curl<Precision, Region::strict_interior>(source, split,
          node_regions, spacing, interior) == lpfx::ErrorCode::success,
          "C2N strict interior dispatch failed");
  check(done.record(interior)); check(done.wait_on(main));
  tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
      split.view(), partial.data(), shape);
  fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(source.view(), cell_shape, false);
  require(tpic::ecsim::enqueue_node_cell_curl<Precision, Region::shell>(source, split, node_regions,
          spacing, main) == lpfx::ErrorCode::success, "C2N shell dispatch failed");
  require(tpic::ecsim::enqueue_node_cell_curl<Precision>(source, full, node_regions, spacing, main) ==
          lpfx::ErrorCode::success, "C2N whole dispatch failed");
  require(tpic::ecsim::enqueue_node_cell_curl<Precision, Region::strict_interior>(full, source,
          cell_regions, spacing, main) == lpfx::ErrorCode::invalid_argument,
          "N2C accepted C2N split regions");
  require(tpic::ecsim::enqueue_node_cell_curl<Precision>(source, full, cell_regions, spacing, main) ==
          lpfx::ErrorCode::invalid_argument, "C2N accepted a geometry of the wrong location");
  tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
      split.view(), actual.data(), shape);
  tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
      full.view(), expected.data(), shape);
  check(Traits::take_last_launch_error());
  const auto first = partial.download(main), result = actual.download(main), baseline = expected.download(main);
  for (int n = 0; n < shape.count; ++n) {
    const int i = n % shape.nx, j = (n / shape.nx) % shape.ny, k = n / shape.nx / shape.ny;
    const bool inside = i > 0 && i < cells[0] && j > 0 && j < cells[1] && k > 0 && k < cells[2];
    for (int component = 0; component < 3; ++component) {
      const int index = component * shape.count + n;
      require(std::isfinite(first[index]) == inside, "C2N strict interior touched ghosts or shell");
      require(std::isfinite(result[index]) && result[index] == baseline[index],
              "C2N split differs from whole-domain curl");
      if (inside) require(first[index] == baseline[index], "C2N strict interior changed after ghost update");
    }
  }
}

// Freeze the serial algorithm independently of the production apply
// leaves. In particular complete_electric_refresh retains its second wall projection.
// Both leaves agreeing with each other would not detect a shared BC error.
template <class Precision>
const tpic::ecsim::SolverNodeOwner<Precision, Backend>& sequential_smooth(
    tpic::ecsim::MaxwellOperator<Precision, Backend>& op,
    const tpic::ecsim::SolverNodeOwner<Precision, Backend>& source,
    const tpic::simulation::PreparedInput& prepared, EmptyExchange& exchange_a,
    EmptyExchange& exchange_b, bool valid_halos, bool final_halos, Executor main, unsigned* invalid) {
  using Owner = tpic::ecsim::SolverNodeOwner<Precision, Backend>;
  Owner* current = &op.smooth_a;
  Owner* next = &op.smooth_b;
  const auto refresh = [&](Owner* field) {
    if (field == &op.smooth_a) op.complete_electric_refresh(*field, exchange_a);
    else op.complete_electric_refresh(*field, exchange_b);
  };
  const auto pass = [&](const auto& from, auto& to) {
    require(tpic::ecsim::enqueue_node_smoothing<Precision>(from, to,
        prepared.parameters.ecsim_smoothing_operator, prepared.parameters.smooth, invalid, main) ==
        lpfx::ErrorCode::success, "sequential smoother failed");
  };
  int completed = 0;
  if (valid_halos) {
    pass(source, *current);
    completed = 1;
  } else {
    require(tpic::ecsim::enqueue_field_vector_copy(*current, source, false, nullptr, main) ==
        lpfx::ErrorCode::success, "sequential smoothing copy failed");
  }
  for (; completed < prepared.parameters.smooth_niter; ++completed) {
    refresh(current);
    pass(*current, *next);
    std::swap(current, next);
  }
  if (final_halos) refresh(current);
  else require(tpic::ecsim::enqueue_constrained_rows(*current, tpic::ecsim::detail::NoRowInput{},
      tpic::ecsim::prepare_field_boundary_plans(prepared.domain, prepared.physics).constrained_rows, main) == lpfx::ErrorCode::success,
      "sequential smoothing projection failed");
  return *current;
}

template <class Precision>
void sequential_apply(tpic::ecsim::MaxwellOperator<Precision, Backend>& op,
    const tpic::data::CompactMassMatrix<Precision, Backend>& mass,
    const tpic::simulation::PreparedInput& prepared,
    const tpic::ecsim::WallFaces& dual_volume, EmptyExchange& input_exchange,
    EmptyExchange& curl_exchange, EmptyExchange& exchange_a, EmptyExchange& exchange_b,
    Executor main, unsigned* invalid) {
  using Compute = typename Precision::Compute::Electric;
  tpic::ecsim::NodeCellSpacing<Compute> spacing;
  require(tpic::ecsim::try_make_node_cell_spacing(prepared.domain.spacing, spacing) ==
          lpfx::ErrorCode::success, "sequential spacing failed");
  const auto plans = tpic::ecsim::prepare_field_operator_plans(prepared.domain, prepared.physics);
  op.complete_electric_refresh(op.input, input_exchange);
  require(tpic::ecsim::enqueue_node_cell_curl<Precision>(op.input, op.curl_cell, plans.cells, spacing,
                                                         main) ==
          lpfx::ErrorCode::success, "sequential N2C failed");
  tpic::communication::complete_field_exchange(curl_exchange,
      "TPIC Maxwell: halo exchange rejected", "TPIC Maxwell: halo exchange failed");
  op.close_magnetic_ghosts(op.curl_cell);
  require(tpic::ecsim::enqueue_node_cell_curl<Precision>(op.curl_cell, op.curl_node, plans.nodes,
                                                         spacing, main) ==
          lpfx::ErrorCode::success, "sequential C2N failed");
  const auto& mass_input = prepared.physics.smoothing_enabled
      ? sequential_smooth(op, op.input, prepared, exchange_a, exchange_b, true, true, main, invalid)
      : op.input;
  require(tpic::ecsim::enqueue_mass_matrix_apply<Precision>(mass, mass_input, op.mass_image,
      plans.nodes, dual_volume, static_cast<Compute>(prepared.domain.inverse_cell_volume), invalid,
      main) == lpfx::ErrorCode::success, "sequential mass failed");
  const auto& response = prepared.physics.smoothing_enabled
      ? sequential_smooth(op, op.mass_image, prepared, exchange_a, exchange_b, false, false, main, invalid)
      : op.mass_image;
  const Compute delta = prepared.physics.c_theta_dt;
  const Compute current_factor = 4 * std::numbers::pi_v<Compute> * Compute(prepared.physics.theta_dt);
  require(tpic::ecsim::enqueue_field_vector_linear_combination<Precision>(op.image, nullptr, main,
      tpic::ecsim::scaled(Compute(1), op.input), tpic::ecsim::scaled(delta * delta, op.curl_node),
      tpic::ecsim::scaled(current_factor, response)) ==
      lpfx::ErrorCode::success, "sequential operator combination failed");
  require(tpic::ecsim::enqueue_constrained_rows(op.image, op.input, tpic::ecsim::prepare_field_boundary_plans(prepared.domain,
      prepared.physics).constrained_rows, main) == lpfx::ErrorCode::success, "sequential boundary rows failed");
}

template <class Precision, Schedule Selected> void run(Executor main, Executor helper, Executor interior,
    std::array<int, 3> cells = {8, 6, 4}, Walls walls = Walls::pec,
    Smoother smoother = Smoother::Face7) {
  constexpr bool smoothing = Selected == Schedule::smoothed;
  auto prepared = fixture(smoothing, cells, 1, walls, smoother);
  tpic::ecsim::MaxwellOperator<Precision, Backend> sequential(prepared, main, main, main);
  tpic::ecsim::MaxwellOperator<Precision, Backend> overlapped(prepared, main, helper, interior);
  tpic::data::CompactMassMatrix<Precision, Backend> mass(prepared.domain);
  tpic::ecsim::WallFaces dual_volume;
  require(tpic::ecsim::try_prepare_wall_faces(prepared.domain, dual_volume) ==
              lpfx::ErrorCode::success,
          "Maxwell overlap dual-volume preparation failed");
  tpic::data::GridShape shape;
  require(tpic::data::try_make_grid_shape(prepared.domain.field_neighborhood.local_partition().local_cells(), 1, {}, shape) ==
          lpfx::ErrorCode::success, "Maxwell overlap shape failure");
  constexpr unsigned epochs = 8;
  const unsigned vector_count = 3 * shape.count;
  const auto halo_shape = tpic::ecsim::field_vector_shape(sequential.input, 1);
  const unsigned snapshot_count = vector_count + 3 * halo_shape.count;
  Buffer<double> baseline(std::size_t(epochs) * snapshot_count), concurrent(baseline.count);
  Buffer<unsigned> invalid(2), flags(2 * epochs);
  // The curl halo may overlap the input halo; no other field epochs overlap.
  unsigned sequential_gate = 0, actual_gate = 0, curl_gate = 0;
  EmptyExchange sequential_input(main, sequential_gate), sequential_curl(main, curl_gate),
      sequential_a(main, sequential_gate), sequential_b(main, sequential_gate);
  EmptyExchange parallel_input(main, actual_gate), parallel_curl(main, curl_gate),
      parallel_a(main, actual_gate), parallel_b(main, actual_gate);
  // The ghost records of mass arrive late on their own stream, and only the
  // stencils the ghost phase sends: the apply reads no other ghost entry, and
  // only the hook orders its first ghost read after them, once per apply, on
  // the executor of the shell or full mass kernel.
  auto late_stream = Traits::null_stream;
  check(Traits::create_stream(late_stream));
  const Executor late{late_stream};
  lpfx::device::DeviceEvent<Backend> poisoned, ghosts_ready;
  check(decltype(poisoned)::create(poisoned)); check(decltype(ghosts_ready)::create(ghosts_ready));
  const auto ghost_reader = Selected == Schedule::smoothed ? main : helper;
  const auto ghosts = ghost_stencils();
  unsigned ghost_waits = 0;
  const auto host_wait_mass_ghosts = [&](const Executor executor) {
    require(executor.stream() == ghost_reader.stream(), "mass ghosts ordered on another executor");
    ++ghost_waits;
    check(ghosts_ready.wait_on(executor));
  };
  // No host fence between epochs: reuse of the events must bind each wait
  // to the corresponding input/mass generation before the next fill begins.
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    const int poison = epoch + 3 == epochs ? 1 : epoch + 2 == epochs ? 2 : 0;
    check(Traits::memset_device_async(invalid.data(), 0, 2 * sizeof(unsigned), main.stream()));
    fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(sequential.input.view(), overlapped.input.view(), shape, epoch);
    fill_maxwell_node_ghosts_kernel<<<2, 128, 0, main.stream()>>>(sequential.input.view(), shape, true);
    fill_maxwell_node_ghosts_kernel<<<2, 128, 0, main.stream()>>>(overlapped.input.view(), shape, true);
    poison_maxwell_node_field_kernel<<<2, 128, 0, main.stream()>>>(overlapped.image.view(), shape);
    poison_maxwell_node_field_kernel<<<2, 128, 0, main.stream()>>>(overlapped.mass_image.view(), shape);
    poison_maxwell_node_field_kernel<<<2, 128, 0, main.stream()>>>(overlapped.curl_node.view(), shape);
    fill_maxwell_mass_coefficients_kernel<<<3, 128, 0, main.stream()>>>(mass.data(), mass.layout(), epoch, poison);
    check(Traits::take_last_launch_error());
    sequential_apply(sequential, mass, prepared, dual_volume, sequential_input, sequential_curl,
                    sequential_a, sequential_b, main, invalid.data());
    auto* expected_epoch = baseline.data() + std::size_t(epoch) * snapshot_count;
    auto* actual_epoch = concurrent.data() + std::size_t(epoch) * snapshot_count;
    tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
        sequential.image.view(), expected_epoch, shape);
    capture_maxwell_field_with_halo_kernel<<<2, 128, 0, main.stream()>>>(
        sequential.input.view(), halo_shape, expected_epoch + vector_count);
    fill_maxwell_mass_ghosts_kernel<<<3, 128, 0, main.stream()>>>(mass.data(), mass.layout(), epoch, true, ghosts);
    check(poisoned.record(main)); check(poisoned.wait_on(late));
    delay_kernel<<<1, 1, 0, late.stream()>>>(2000000);
    fill_maxwell_mass_ghosts_kernel<<<3, 128, 0, late.stream()>>>(mass.data(), mass.layout(), epoch, false, ghosts);
    check(Traits::take_last_launch_error());
    check(ghosts_ready.record(late));
    overlapped.template complete_apply<Selected>(mass, parallel_input, parallel_curl,
                                        parallel_a, parallel_b, invalid.data() + 1, host_wait_mass_ghosts);
    tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
        overlapped.image.view(), actual_epoch, shape);
    capture_maxwell_field_with_halo_kernel<<<2, 128, 0, main.stream()>>>(
        overlapped.input.view(), halo_shape, actual_epoch + vector_count);
    capture_maxwell_validation_flags_kernel<<<1, 1, 0, main.stream()>>>(invalid.data(), invalid.data() + 1, flags.data() + 2 * epoch);
    check(Traits::take_last_launch_error());
  }
  const auto expected = baseline.download(main), actual = concurrent.download(main);
  const auto validation = flags.download(main);
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    const bool bad = epoch + 3 == epochs || epoch + 2 == epochs;
    require(validation[2 * epoch] == validation[2 * epoch + 1], "Maxwell overlap lost validation writes");
    require((validation[2 * epoch] != 0) == bad, "Maxwell overlap validation mismatch");
    // Input closure remains finite even on an invalid-matrix epoch. A valid
    // epoch after both core/shell failures also checks validation reset/reuse.
    for (unsigned n = bad ? vector_count : 0; n < snapshot_count; ++n) {
      const auto index = std::size_t(epoch) * snapshot_count + n;
      require(std::isfinite(actual[index]) && actual[index] == expected[index],
              "Maxwell schedule changed the original image or input closure");
    }
  }
  require(parallel_input.begins == sequential_input.begins &&
          parallel_curl.begins == sequential_curl.begins &&
          parallel_a.begins == sequential_a.begins && parallel_b.begins == sequential_b.begins,
          "Maxwell overlap changed the halo exchange schedule");
  for (const auto* exchange : {&sequential_input, &sequential_curl, &sequential_a, &sequential_b,
                               &parallel_input, &parallel_curl, &parallel_a, &parallel_b})
    require(!exchange->active && exchange->begins == exchange->waits &&
            exchange->begins == exchange->concludes, "Maxwell schedule left a live field epoch");
  require(sequential_gate == 0 && actual_gate == 0 && curl_gate == 0, "Maxwell schedule overlapped field epochs");
  require(ghost_waits == epochs, "Maxwell apply did not wait the mass ghosts once");
  check(Traits::synchronize_stream(helper.stream()));
  check(Traits::synchronize_stream(interior.stream()));
  check(Traits::synchronize_stream(late_stream));
  check(poisoned.reset()); check(ghosts_ready.reset());
  check(Traits::destroy_stream(late_stream));
}

template <class Precision>
void all_schedules(Executor main, Executor mass, Executor interior,
                   std::array<int, 3> cells = {8, 6, 4}, Walls walls = Walls::pec) {
  run<Precision, Schedule::full>(main, mass, interior, cells, walls);
  run<Precision, Schedule::split>(main, mass, interior, cells, walls);
  run<Precision, Schedule::smoothed>(main, mass, interior, cells, walls);
}

template <Schedule Selected>
void failed_apply(Executor main, Executor helper, Executor interior) {
  using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
  const auto prepared = fixture(false);
  tpic::ecsim::MaxwellOperator<Precision, Backend> op(prepared, main, helper, interior);
  tpic::data::CompactMassMatrix<Precision, Backend> mass(prepared.domain);
  tpic::data::GridShape shape;
  require(tpic::data::try_make_grid_shape(op.input.cells(), 1, {}, shape) == lpfx::ErrorCode::success,
          "apply failure shape failed");
  Buffer<unsigned> invalid(1);
  Buffer<double> image(3 * shape.count);
  unsigned gate = 0, curl_gate = 0;
  EmptyExchange input(main, gate), curl(main, curl_gate), smooth_a(main, gate), smooth_b(main, gate);
  fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(op.input.view(), op.input.view(), shape, 0);
  fill_maxwell_mass_coefficients_kernel<<<3, 128, 0, main.stream()>>>(mass.data(), mass.layout(), 0, 0);
  check(Traits::take_last_launch_error());
  bool caught = false;
  try {
    op.template complete_apply<Selected>(mass, input, curl, smooth_a, smooth_b, nullptr, [](Executor) {});
  } catch (const std::runtime_error&) {
    caught = true;
  }
  const unsigned expected_curl = Selected == Schedule::full ? 1 : 0;
  require(caught && input.begins == 1 && curl.begins == expected_curl,
          "apply did not fail in the expected accepted input/curl interval");
  for (const auto* exchange : {&input, &curl, &smooth_a, &smooth_b})
    require(!exchange->active && exchange->begins == exchange->waits &&
            exchange->begins == exchange->concludes, "failed apply leaked an accepted exchange");
  require(gate == 0 && curl_gate == 0, "failed apply did not release the field epoch gate");

  // The catch must drain helper readers before these input/matrix writes.
  // No caller drain is inserted before reusing the exact same operator/events.
  check(Traits::memset_device_async(invalid.data(), 0, sizeof(unsigned), main.stream()));
  fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(op.input.view(), op.input.view(), shape, 1);
  fill_maxwell_mass_coefficients_kernel<<<3, 128, 0, main.stream()>>>(mass.data(), mass.layout(), 1, 0);
  check(Traits::take_last_launch_error());
  op.template complete_apply<Selected>(mass, input, curl, smooth_a, smooth_b, invalid.data(), [](Executor) {});
  tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
      op.image.view(), image.data(), shape);
  check(Traits::take_last_launch_error());
  for (double value : image.download(main)) require(std::isfinite(value), "apply reuse produced nonfinite output");
  require(invalid.download(main)[0] == 0 && input.begins == 2 && curl.begins == expected_curl + 1 && gate == 0,
          "apply was not reusable after retiring a healthy failed submission");
}

// Energy conservation of the field equations: with the wall ghosts of B and
// the zeroed wall rows of E, the node curl of B is the adjoint of the cell curl
// of E in the dual-volume inner product, for every wall kind and every mix.
void curl_adjointness(Executor main, Walls walls) {
  using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
  const auto prepared = fixture(false, {4, 3, 2}, 1, walls);
  tpic::ecsim::MaxwellOperator<Precision, Backend> op(prepared, main, main, main);
  const auto plans = tpic::ecsim::prepare_field_operator_plans(prepared.domain, prepared.physics);
  tpic::ecsim::NodeCellSpacing<double> spacing;
  require(tpic::ecsim::try_make_node_cell_spacing(prepared.domain.spacing, spacing) ==
          lpfx::ErrorCode::success, "adjointness spacing failed");
  const auto& cells = prepared.domain.local_cells;
  const tpic::data::GridShape node_shape{cells[0] + 1, cells[1] + 1, cells[2] + 1,
                                         (cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1)};
  const tpic::data::GridShape cell_shape{cells[0], cells[1], cells[2], cells[0] * cells[1] * cells[2]};
  const auto with_halo = [](tpic::data::GridShape s) {
    return tpic::data::GridShape{s.nx + 2, s.ny + 2, s.nz + 2, (s.nx + 2) * (s.ny + 2) * (s.nz + 2)};
  };
  EmptyExchange exchange;
  fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(op.input.view(), op.input.view(), node_shape, 0);
  op.complete_electric_refresh(op.input, exchange);
  fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(op.next_cell.view(), cell_shape, false);
  op.close_magnetic_ghosts(op.next_cell);
  require(tpic::ecsim::enqueue_node_cell_curl<Precision>(op.next_cell, op.curl_node, plans.nodes, spacing,
                                                         main) == lpfx::ErrorCode::success &&
              tpic::ecsim::enqueue_node_cell_curl<Precision>(op.input, op.curl_cell, plans.cells, spacing,
                                                             main) == lpfx::ErrorCode::success,
          "adjointness curls failed");
  const auto capture = [&](const auto& owner, tpic::data::GridShape shape) {
    Buffer<double> values(3 * static_cast<std::size_t>(shape.count));
    capture_maxwell_field_with_halo_kernel<<<2, 128, 0, main.stream()>>>(owner.view(), shape, values.data());
    check(Traits::take_last_launch_error());
    return values.download(main);
  };
  const auto node_halo = with_halo(node_shape), cell_halo = with_halo(cell_shape);
  const auto electric = capture(op.input, node_halo), node_curl = capture(op.curl_node, node_halo);
  const auto magnetic = capture(op.next_cell, cell_halo), cell_curl = capture(op.curl_cell, cell_halo);
  const auto dot = [](const std::vector<double>& a, const std::vector<double>& b,
                      const tpic::data::GridShape& s, int i, int j, int k) {
    const auto n = static_cast<std::size_t>((i + 1) + s.nx * ((j + 1) + s.ny * (k + 1)));
    const auto count = static_cast<std::size_t>(s.count);
    return a[n] * b[n] + a[count + n] * b[count + n] + a[2 * count + n] * b[2 * count + n];
  };
  double node_sum = 0, cell_sum = 0, scale = 0;
  for (int k = 0; k <= cells[2]; ++k)
    for (int j = 0; j <= cells[1]; ++j)
      for (int i = 0; i <= cells[0]; ++i)
        node_sum += dot(electric, node_curl, node_halo, i, j, k) /
            plans.boundary.node_ghosts.faces.dual_volume_factor(i, j, k);
  for (int k = 0; k < cells[2]; ++k)
    for (int j = 0; j < cells[1]; ++j)
      for (int i = 0; i < cells[0]; ++i) {
        const double term = dot(magnetic, cell_curl, cell_halo, i, j, k);
        cell_sum += term;
        scale += std::abs(term);
      }
  require(scale > 0 && std::abs(node_sum - cell_sum) <= 1e-12 * scale,
          "the wall closure broke the adjointness of the two curls");
}

// Energy conservation with smoothing: the smoothing of MaxwellOperator, with
// its wall ghosts and its projection after every pass, is self-adjoint in the
// dual-volume inner product for arbitrary node vectors.
void smoothing_adjointness(Executor main, Walls walls, Smoother smoother) {
  using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
  const auto prepared = fixture(true, {4, 3, 2}, 1, walls, smoother);
  tpic::ecsim::MaxwellOperator<Precision, Backend> op(prepared, main, main, main);
  const auto plans = tpic::ecsim::prepare_field_operator_plans(prepared.domain, prepared.physics);
  const auto& cells = prepared.domain.local_cells;
  const tpic::data::GridShape shape{cells[0] + 1, cells[1] + 1, cells[2] + 1,
                                    (cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1)};
  const tpic::data::GridShape halo{shape.nx + 2, shape.ny + 2, shape.nz + 2,
                                   (shape.nx + 2) * (shape.ny + 2) * (shape.nz + 2)};
  Buffer<unsigned> invalid(1);
  check(Traits::memset_device_async(invalid.data(), 0, sizeof(unsigned), main.stream()));
  EmptyExchange exchange_a, exchange_b;
  // Two linearly independent node vectors that satisfy no wall condition.
  fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(op.rhs.view(), op.rhs.view(), shape, 0);
  fill_maxwell_cell_field_kernel<<<2, 128, 0, main.stream()>>>(op.image.view(), shape, false);
  const auto capture = [&](const auto& owner) {
    Buffer<double> values(3 * static_cast<std::size_t>(halo.count));
    capture_maxwell_field_with_halo_kernel<<<2, 128, 0, main.stream()>>>(owner.view(), halo, values.data());
    check(Traits::take_last_launch_error());
    return values.download(main);
  };
  const auto first = capture(op.rhs), second = capture(op.image);
  const auto smoothed_second = capture(sequential_smooth(op, op.image, prepared, exchange_a, exchange_b,
                                                         false, false, main, invalid.data()));
  const auto smoothed_first = capture(sequential_smooth(op, op.rhs, prepared, exchange_a, exchange_b,
                                                        false, false, main, invalid.data()));
  double left = 0, right = 0, scale = 0;
  const auto count = static_cast<std::size_t>(halo.count);
  for (int k = 0; k <= cells[2]; ++k)
    for (int j = 0; j <= cells[1]; ++j)
      for (int i = 0; i <= cells[0]; ++i) {
        const auto n = static_cast<std::size_t>((i + 1) + halo.nx * ((j + 1) + halo.ny * (k + 1)));
        const double weight = 1.0 / plans.boundary.node_ghosts.faces.dual_volume_factor(i, j, k);
        for (std::size_t c = 0; c < 3; ++c) {
          left += weight * first[c * count + n] * smoothed_second[c * count + n];
          right += weight * smoothed_first[c * count + n] * second[c * count + n];
          scale += weight * std::abs(first[c * count + n] * smoothed_second[c * count + n]);
        }
      }
  require(invalid.download(main)[0] == 0 && scale > 0 && std::abs(left - right) <= 1e-12 * scale,
          "the wall closure broke the self-adjointness of the smoothing");
}

void final_theta_projection(Executor main, Walls walls) {
  using Precision = tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>;
  const auto prepared = fixture(false, {4, 3, 2}, 1, walls);
  tpic::ecsim::MaxwellOperator<Precision, Backend> op(prepared, main, main, main);
  tpic::data::GridShape shape;
  require(tpic::data::try_make_grid_shape(op.theta.cells(), 1, {}, shape) == lpfx::ErrorCode::success,
          "final theta shape failed");
  Buffer<double> captured(3 * shape.count);
  EmptyExchange exchange;
  fill_maxwell_inputs_kernel<<<2, 128, 0, main.stream()>>>(op.theta.view(), op.theta.view(), shape, 0);
  op.complete_electric_refresh(op.theta, exchange);
  tpic::ecsim::kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack><<<2, 128, 0, main.stream()>>>(
      op.theta.view(), captured.data(), shape);
  check(Traits::take_last_launch_error());
  const auto values = captured.download(main);
  for (int n = 0; n < shape.count; ++n) {
    const int q[3] = {n % shape.nx, (n / shape.nx) % shape.ny, n / shape.nx / shape.ny};
    for (int component = 0; component < 3; ++component) {
      bool constrained = false;
      for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
          if (q[axis] != (side ? prepared.domain.local_cells[axis] : 0)) continue;
          const auto face = prepared.physics.field_boundary[2 * axis + (side == 0)];
          constrained |= (face == tpic::config::FieldBoundary::perfect_electric_conductor && component != axis) ||
                         (face == tpic::config::FieldBoundary::perfect_magnetic_conductor && component == axis);
        }
      }
      const double value = values[component * shape.count + n];
      require(std::isfinite(value) && ((value == 0) == constrained),
              "final theta lost a wall constraint or projected an unconstrained component");
    }
  }
}

bool unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
         error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}
} // namespace

int main() {
  lpfx::device::DeviceProperties properties;
  auto error = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(error)) {
    if (unavailable(error)) {
      std::cout << "TPIC Maxwell overlap tests skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto main_stream = Traits::null_stream, helper_stream = Traits::null_stream,
       interior_stream = Traits::null_stream;
  error = Traits::create_stream(main_stream);
  if (!Traits::succeeded(error)) return unavailable(error) ? 77 : 1;
  error = Traits::create_stream(helper_stream);
  if (!Traits::succeeded(error)) { static_cast<void>(Traits::destroy_stream(main_stream)); return 1; }
  error = Traits::create_stream(interior_stream);
  if (!Traits::succeeded(error)) {
    static_cast<void>(Traits::destroy_stream(helper_stream));
    static_cast<void>(Traits::destroy_stream(main_stream)); return 1;
  }
  int status = 0;
  try {
    field_action_handoffs(Executor{main_stream}, Executor{helper_stream});
    interpolation_validation<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{helper_stream});
    interpolation_validation<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(Executor{helper_stream});
    for (const auto theta : {.5, 1.0}) {
      separate_field_update<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, theta);
      separate_field_update<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, theta);
      separate_field_update<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, theta);
    }
    all_schedules<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, Executor{interior_stream});
    all_schedules<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, Executor{interior_stream});
    all_schedules<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, Executor{interior_stream});
    for (const auto streams : {std::array{helper_stream, helper_stream},
                               std::array{main_stream, interior_stream},
                               std::array{helper_stream, main_stream},
                               std::array{main_stream, main_stream}})
      all_schedules<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{streams[0]}, Executor{streams[1]});
    for (const auto walls : {Walls::pec, Walls::pmc, Walls::mixed}) {
      final_theta_projection(Executor{main_stream}, walls);
      curl_adjointness(Executor{main_stream}, walls);
      for (const auto smoother : {Smoother::Face7, Smoother::Binomial27})
        smoothing_adjointness(Executor{main_stream}, walls, smoother);
      all_schedules<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream},
                                           Executor{interior_stream}, {4, 3, 2}, walls);
      run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>, Schedule::smoothed>(Executor{main_stream}, Executor{helper_stream},
          Executor{interior_stream}, {4, 3, 2}, walls, Smoother::Binomial27);
    }
    run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>, Schedule::smoothed>(Executor{main_stream}, Executor{helper_stream},
        Executor{interior_stream}, {8, 6, 4}, Walls::pec, Smoother::Binomial27);
    run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>, Schedule::smoothed>(Executor{main_stream}, Executor{helper_stream},
        Executor{interior_stream}, {8, 6, 4}, Walls::pec, Smoother::Binomial27);
    failed_apply<Schedule::full>(Executor{main_stream}, Executor{helper_stream}, Executor{interior_stream});
    failed_apply<Schedule::split>(Executor{main_stream}, Executor{helper_stream}, Executor{interior_stream});
    for (bool smoothing : {false, true}) {
      split_rhs<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, smoothing);
      split_rhs<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, smoothing);
    }
    for (const auto cells : {std::array{8, 6, 4}, std::array{2, 2, 2}, std::array{1, 3, 2},
                            std::array{4, 1, 2}, std::array{3, 2, 1}, std::array{1, 1, 1}}) {
      split_stencil(Executor{main_stream}, Executor{interior_stream}, cells);
      n2c_during_ghost_update(Executor{main_stream}, Executor{interior_stream}, cells);
      if (cells != std::array{8, 6, 4})
        all_schedules<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream},
                                             Executor{interior_stream}, cells);
    }
    std::cout << "TPIC Maxwell overlap tests passed\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    static_cast<void>(Traits::synchronize_stream(main_stream));
    static_cast<void>(Traits::synchronize_stream(helper_stream));
    static_cast<void>(Traits::synchronize_stream(interior_stream));
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(interior_stream))) status = 1;
  if (!Traits::succeeded(Traits::destroy_stream(helper_stream))) status = 1;
  if (!Traits::succeeded(Traits::destroy_stream(main_stream))) status = 1;
  return status;
}
