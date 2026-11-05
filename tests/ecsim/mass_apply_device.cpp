// Verify row partitioning independently from Maxwell and its halo transport.
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_apply_device.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include "../support/test_builders.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Region = tpic::ecsim::StencilRegion;
using Geometry = tpic::ecsim::StencilRegionGeometry;

void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }
void check(Traits::error_type error) {
  require(Traits::succeeded(error), "mass apply test backend failure");
}

template<class T> struct Buffer {
  lpfx::device::DeviceAllocation<Backend> memory;
  lpfx::device::AlignedPinnedHostAllocation<Backend, alignof(T)> staging;
  std::size_t count;
  explicit Buffer(std::size_t n) : count(n) {
    check(decltype(memory)::create(n * sizeof(T), memory));
    require(decltype(staging)::create(n * sizeof(T), staging) == lpfx::ErrorCode::success,
            "mass apply test pinned allocation failed");
  }
  T* data() { return reinterpret_cast<T*>(memory.raw_base()); }
  std::vector<T> download(Executor executor) {
    auto* host = reinterpret_cast<T*>(staging.raw_base());
    check(Traits::copy_device_to_host_async(host, data(), count * sizeof(T), executor.stream()));
    check(Traits::synchronize_stream(executor.stream()));
    return std::vector<T>(host, host + count);
  }
};

tpic::config::PreparedDomain fixture(std::array<int, 3> cells, bool periodic) {
  // These low-level kernels support a one-cell extent independently of the
  // production input policy, which requires at least two cells per rank on
  // every axis, one on a field-periodic axis with a single rank.
  // Prepare consistent LPFx geometry directly; do not weaken domain admission.
  tpic::config::PreparedDomain domain;
  tpic::config::PreparedDomain::TopologyInput input;
  require(lpfx::try_make_canonical_cartesian_topology_input(1, 1, 1, 0, 1,
      lpfx::CartesianPeriodicity3{periodic, periodic, periodic}, input) ==
      lpfx::ErrorCode::success, "mass apply fixture topology input failed");
  tpic::config::PreparedDomain::Topology topology;
  require(lpfx::try_prepare_canonical_cartesian_topology(input, topology) ==
      lpfx::ErrorCode::success, "mass apply fixture field topology failed");
  domain.particle_topology = topology;
  require(lpfx::try_prepare_cartesian_partition_neighborhood3d<
              tpic::config::PreparedDomain::IndexPolicy>(
              cells[0], cells[1], cells[2], topology,
              domain.field_neighborhood) == lpfx::ErrorCode::success,
          "mass apply fixture partition neighborhood failed");
  domain.local_cells = domain.global_cells = cells;
  domain.spacing = {.5, .25, 2.0};
  for (unsigned axis = 0; axis < 3; ++axis)
    domain.lengths[axis] = domain.spacing[axis] * cells[axis];
  domain.cell_volume = .25;
  domain.inverse_cell_volume = 4;
  tpic::test::fill_domain_counts(domain);
  return domain;
}

template<class View>
__global__ void initialize_mass_apply_field_kernel(View field, Geometry nodes, int epoch,
                                 bool poison_all, bool ghosts_only) {
  const int nx = nodes.nx + 2, ny = nodes.ny + 2, nz = nodes.nz + 2;
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < unsigned(nx * ny * nz);
       n += blockDim.x * gridDim.x) {
    const int i = int(n % nx) - 1, j = int((n / nx) % ny) - 1, k = int(n / nx / ny) - 1;
    const bool ghost = i < 0 || i >= nodes.nx || j < 0 || j >= nodes.ny || k < 0 || k >= nodes.nz;
    if (ghosts_only && !ghost) continue;
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ((field.template component<C>()(i,j,k) = poison_all || (ghost && !ghosts_only)
          ? std::numeric_limits<double>::quiet_NaN()
          : (C + 1) * (.125 * i * j + .25 * j * k + .5 * k * i + epoch + 1)), ...);
    }(std::make_index_sequence<3>{});
  }
}

template<class Scalar>
__global__ void initialize_mass_apply_coefficients_kernel(Scalar* mass, tpic::data::MassMatrixLayout layout,
                                int epoch, std::size_t invalid_entry) {
  for (std::size_t n = blockIdx.x * blockDim.x + threadIdx.x; n < layout.storage_size();
       n += blockDim.x * gridDim.x)
    mass[n] = n == invalid_entry ? std::numeric_limits<Scalar>::quiet_NaN()
        : Scalar(.0001 * (1 + .0625 * epoch) * (1 + n % 17));
}

template<class View>
__global__ void capture_mass_apply_field_kernel(View field, Geometry nodes, double* output) {
  const unsigned nx = unsigned(nodes.nx + 2), ny = unsigned(nodes.ny + 2);
  const unsigned count = nx * ny * unsigned(nodes.nz + 2);
  for (unsigned n = blockIdx.x * blockDim.x + threadIdx.x; n < count;
       n += blockDim.x * gridDim.x) {
    const int i = int(n % nx) - 1, j = int((n / nx) % ny) - 1, k = int(n / nx / ny) - 1;
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ((output[C * count + n] = field.template component<C>()(i,j,k)), ...);
    }(std::make_index_sequence<3>{});
  }
}

__global__ void capture_mass_apply_flags_kernel(const unsigned* source, unsigned* output) {
  for (int i = 0; i < 3; ++i) output[i] = source[i];
}

template<class Precision>
void run(Executor main, Executor helper, std::array<int, 3> cells, bool periodic) {
  using Fields = tpic::data::FieldSchema<Precision>;
  const auto domain = fixture(cells, periodic);
  tpic::ecsim::WallFaces dual_volume;
  require(tpic::ecsim::try_prepare_wall_faces(domain, dual_volume) ==
              lpfx::ErrorCode::success,
          "mass apply dual-volume preparation failed");
  auto source = tpic::data::make_zero_field<typename Fields::E>(domain, main);
  auto split = tpic::data::make_zero_field<typename Fields::E>(domain, main);
  auto full = tpic::data::make_zero_field<typename Fields::E>(domain, main);
  tpic::data::CompactMassMatrix<Precision, Backend> mass(domain);
  Geometry nodes;
  require(tpic::ecsim::try_make_stencil_region_geometry(cells[0] + 1, cells[1] + 1, cells[2] + 1,
          nodes) == lpfx::ErrorCode::success, "mass apply test geometry failed");
  const auto inverse_volume = static_cast<typename Precision::Compute::Electric>(domain.inverse_cell_volume);
  const unsigned halo_count = unsigned((nodes.nx + 2) * (nodes.ny + 2) * (nodes.nz + 2));
  Buffer<double> partial(3 * halo_count), actual(3 * halo_count), expected(3 * halo_count);
  constexpr unsigned epochs = 4;
  Buffer<double> concurrent_values(epochs * 3 * halo_count), sequential_values(epochs * 3 * halo_count);
  Buffer<unsigned> invalid(3), epoch_flags(epochs * 3);
  lpfx::device::DeviceEvent<Backend> ready, done;
  check(decltype(ready)::create(ready)); check(decltype(done)::create(done));
  const auto launch = [&]<Region R>(auto& destination, unsigned* flag, Executor executor) {
    require(tpic::ecsim::enqueue_mass_matrix_apply<Precision, R>(
        mass, source, destination, nodes, dual_volume, inverse_volume, flag, executor) ==
            lpfx::ErrorCode::success,
        "mass apply region dispatch failed");
  };
  const auto initialize = [&](int epoch, std::size_t bad = std::numeric_limits<std::size_t>::max()) {
    check(Traits::memset_device_async(invalid.data(), 0, 3 * sizeof(unsigned), main.stream()));
    initialize_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(source.view(), nodes, epoch, false, false);
    initialize_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(split.view(), nodes, epoch, true, false);
    initialize_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(full.view(), nodes, epoch, true, false);
    initialize_mass_apply_coefficients_kernel<<<3, 128, 0, main.stream()>>>(mass.data(), mass.layout(), epoch, bad);
    check(Traits::take_last_launch_error());
  };

  // First prove that the core cannot read even a single poisoned input ghost
  // and cannot modify any shell/output ghost. Preserve its values for equality.
  initialize(0);
  check(ready.record(main)); check(ready.wait_on(helper));
  launch.template operator()<Region::strict_interior>(split, invalid.data(), helper);
  capture_mass_apply_field_kernel<<<2, 128, 0, helper.stream()>>>(split.view(), nodes, partial.data());
  check(Traits::take_last_launch_error());
  check(done.record(helper)); check(done.wait_on(main));
  const auto first = partial.download(main);
  require(invalid.download(main)[0] == 0, "mass core read poisoned source ghosts");
  initialize_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(source.view(), nodes, 0, false, true);
  launch.template operator()<Region::shell>(split, invalid.data() + 1, main);
  launch.template operator()<Region::full>(full, invalid.data() + 2, main);
  capture_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(split.view(), nodes, actual.data());
  capture_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(full.view(), nodes, expected.data());
  check(Traits::take_last_launch_error());
  const auto result = actual.download(main), baseline = expected.download(main);
  for (unsigned n = 0; n < halo_count; ++n) {
    const int i = int(n % unsigned(nodes.nx + 2)) - 1;
    const int j = int((n / unsigned(nodes.nx + 2)) % unsigned(nodes.ny + 2)) - 1;
    const int k = int(n / unsigned(nodes.nx + 2) / unsigned(nodes.ny + 2)) - 1;
    const bool ghost = i < 0 || i >= nodes.nx || j < 0 || j >= nodes.ny || k < 0 || k >= nodes.nz;
    const bool core = i > 0 && i < cells[0] && j > 0 && j < cells[1] && k > 0 && k < cells[2];
    for (unsigned c = 0; c < 3; ++c) {
      const auto index = c * halo_count + n;
      require(std::isfinite(first[index]) == core, "mass core modified shell or output ghosts");
      if (ghost) require(std::isnan(result[index]) && std::isnan(baseline[index]),
                         "mass apply modified output ghosts");
      else require(std::isfinite(result[index]) && result[index] == baseline[index],
                   "composed mass regions differ from full application");
      if (core) require(first[index] == baseline[index], "mass core changed after input ghost refresh");
    }
  }
  for (auto flag : invalid.download(main)) require(flag == 0, "valid split mass apply raised invalid");

  // Reuse events without host fences between changing epochs. Ghost writes and
  // shell computation may execute concurrently with the core's local readers.
  for (unsigned epoch = 0; epoch < epochs; ++epoch) {
    initialize(int(epoch + 1));
    check(ready.record(main)); check(ready.wait_on(helper));
    launch.template operator()<Region::strict_interior>(split, invalid.data(), helper);
    check(done.record(helper));
    initialize_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(source.view(), nodes, int(epoch + 1), false, true);
    launch.template operator()<Region::shell>(split, invalid.data(), main);
    launch.template operator()<Region::full>(full, invalid.data() + 1, main);
    check(done.wait_on(main));
    const auto offset = epoch * 3 * halo_count;
    capture_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(split.view(), nodes, concurrent_values.data() + offset);
    capture_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(full.view(), nodes, sequential_values.data() + offset);
    capture_mass_apply_flags_kernel<<<1, 1, 0, main.stream()>>>(invalid.data(), epoch_flags.data() + 3 * epoch);
    check(Traits::take_last_launch_error());
  }
  const auto concurrent = concurrent_values.download(main), sequential = sequential_values.download(main);
  for (unsigned n = 0; n < concurrent.size(); ++n)
    require((std::isnan(concurrent[n]) && std::isnan(sequential[n])) || concurrent[n] == sequential[n],
            "concurrent mass regions differ from full application");
  for (auto flag : epoch_flags.download(main)) require(flag == 0, "concurrent mass apply raised invalid");

  // A center coefficient is consumed only by its own row. Isolate invalid
  // reporting in each region, including the empty-core fallback.
  for (bool bad_core : {false, true}) {
    if (bad_core && nodes.interior_count == 0) continue;
    const int coordinate = bad_core ? 1 : 0;
    const auto bad = mass.layout().entry_index(mass.layout().vertex_index(coordinate, coordinate, coordinate), 0);
    initialize(5, bad);
    initialize_mass_apply_field_kernel<<<2, 128, 0, main.stream()>>>(source.view(), nodes, 5, false, true);
    launch.template operator()<Region::strict_interior>(split, invalid.data(), main);
    launch.template operator()<Region::shell>(split, invalid.data() + 1, main);
    launch.template operator()<Region::full>(full, invalid.data() + 2, main);
    const auto flags = invalid.download(main);
    require(flags[0] == unsigned(bad_core) && flags[1] == unsigned(!bad_core) && flags[2] == 1,
            "mass region lost or misplaced coefficient validation");
  }
  require(tpic::ecsim::enqueue_mass_matrix_apply<Precision, Region::strict_interior>(
      mass, source, source, nodes, dual_volume, inverse_volume, invalid.data(), main) ==
          lpfx::ErrorCode::invalid_argument,
      "mass core accepted aliasing, including empty work");
  require(tpic::ecsim::enqueue_mass_matrix_apply<Precision, Region::strict_interior>(
      mass, source, split, nodes, dual_volume, inverse_volume, nullptr, main) ==
          lpfx::ErrorCode::invalid_argument,
      "mass core accepted missing validation, including empty work");
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
      std::cout << "TPIC mass apply tests skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto main_stream = Traits::null_stream, helper_stream = Traits::null_stream;
  error = Traits::create_stream(main_stream);
  if (!Traits::succeeded(error)) return unavailable(error) ? 77 : 1;
  error = Traits::create_stream(helper_stream);
  if (!Traits::succeeded(error)) {
    static_cast<void>(Traits::destroy_stream(main_stream));
    return unavailable(error) ? 77 : 1;
  }
  int status = 0;
  try {
    for (bool periodic : {false, true})
      for (const auto cells : {std::array{8, 6, 4}, std::array{2, 2, 2}, std::array{1, 3, 2},
                              std::array{4, 1, 2}, std::array{3, 2, 1}, std::array{1, 1, 1}}) {
        run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, cells, periodic);
        run<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, cells, periodic);
        run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(Executor{main_stream}, Executor{helper_stream}, cells, periodic);
      }
    std::cout << "TPIC mass apply tests passed\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    static_cast<void>(Traits::synchronize_stream(main_stream));
    static_cast<void>(Traits::synchronize_stream(helper_stream));
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(helper_stream))) status = 1;
  if (!Traits::succeeded(Traits::destroy_stream(main_stream))) status = 1;
  return status;
}
