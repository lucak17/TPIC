// Differential device checks for shared cell gathers, cell current reductions,
// partial-sort suffixes and fused mover classification. No MPI runtime needed.
#include <tpic/config/precision.hpp>
#include <tpic/ecsim/deposition/current_deposition_device.hpp>
#include <tpic/ecsim/particles/particle_mover_device.hpp>
#include <lpfx/backend/cuda_hip/kernel.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Allocation = lpfx::device::DeviceAllocation<Backend>;
namespace ecsim = tpic::ecsim;
namespace kernels = tpic::ecsim::kernels;
namespace vc = tpic::communication;
namespace device = tpic::device;
constexpr unsigned capacity = 1024;
constexpr int nx = 8, ny = 2, nz = 1;
// Include untouched row/plane padding to catch flattened field-cache indexing.
constexpr int pitch_x = nx + 3, pitch_y = ny + 2;
constexpr int nodes = pitch_x * pitch_y * (nz + 1);
constexpr tpic::data::ParticleCellMap cells{nx, ny, nz, nx + 4, ny + 4,
    (nx + 4) * (ny + 4) * (nz + 4)};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Precision> struct ParticleState {
  typename Precision::Storage::Position rx[capacity], ry[capacity], rz[capacity];
  typename Precision::Storage::Velocity u[capacity], v[capacity], w[capacity];
  typename Precision::Storage::Weight q[capacity];
  std::int32_t cell[capacity];
  std::uint64_t id[capacity];
};

template <class Precision> struct ParticleView {
  using S = tpic::data::ParticleSchema<Precision>;
  ParticleState<Precision>* values;
  template <class Tag> __device__ auto get() const {
    if constexpr (std::is_same_v<Tag, typename S::Rx>) return values->rx;
    else if constexpr (std::is_same_v<Tag, typename S::Ry>) return values->ry;
    else if constexpr (std::is_same_v<Tag, typename S::Rz>) return values->rz;
    else if constexpr (std::is_same_v<Tag, typename S::U>) return values->u;
    else if constexpr (std::is_same_v<Tag, typename S::V>) return values->v;
    else if constexpr (std::is_same_v<Tag, typename S::W>) return values->w;
    else if constexpr (std::is_same_v<Tag, typename S::Charge>) return values->q;
    else if constexpr (std::is_same_v<Tag, typename S::Cell>) return values->cell;
    else return values->id;
  }
};

template <class Scalar> struct Component {
  using location_type = lpfx::Node;
  Scalar* values;
  __host__ __device__ Scalar& operator()(int i, int j, int k) const {
    return values[i + pitch_x * (j + pitch_y * k)];
  }
};

template <class Scalar, class X, class Y, class Z> struct FieldView {
  Scalar* values;
  template <std::size_t Axis> __host__ __device__ Component<Scalar> component() const {
    return {values + Axis * nodes};
  }
  template <class Tag> __device__ Component<Scalar> get() const {
    static_assert(std::is_same_v<Tag, X> || std::is_same_v<Tag, Y> || std::is_same_v<Tag, Z>);
    return component<std::is_same_v<Tag, X> ? 0 : std::is_same_v<Tag, Y> ? 1 : 2>();
  }
};

struct Producer {
  std::uint64_t* flags;
  unsigned* registrations;
  __device__ bool register_flag(unsigned index, std::uint64_t flag) const {
    atomicExch(reinterpret_cast<unsigned long long*>(flags + index),
        static_cast<unsigned long long>(flag));
    atomicAdd(registrations + index, 1U);
    return true;
  }
};

template <class Precision> struct Fixture {
  ParticleState<Precision> unsorted{}, sorted{};
  typename Precision::Storage::Magnetic magnetic[3 * nodes]{};
  typename Precision::Storage::Electric electric[3 * nodes]{};
  typename Precision::Storage::Current unsorted_current[3 * nodes]{}, sorted_current[3 * nodes]{};
  unsigned counts[cells.count]{}, offsets[cells.count + 1]{}, failure{};
  unsigned invalid[3]{};
  std::uint64_t unsorted_flags[capacity]{}, sorted_flags[capacity]{};
  unsigned unsorted_registered[capacity]{}, sorted_registered[capacity]{};
};

vc::ParticleMigrationGeometry geometry() {
  vc::ParticleMigrationGeometry result;
  result.cells = cells;
  const int extents[]{nx, ny, nz};
  for (int axis = 0; axis < 3; ++axis) {
    result.global_cells[axis] = extents[axis];
    result.partition_base[axis] = extents[axis];
    result.periodic[axis] = axis != 1; // y uses reflecting boundaries.
  }
  return result;
}

template <class Precision> unsigned fill(Fixture<Precision>& input, unsigned& prefix) {
  const unsigned populations[]{1, 31, 32, 33, 129, 257, 0, 3};
  for (int i = 0; i < nx; ++i) input.counts[cells.encode(i, i % 2, 0)] = populations[i];
  unsigned count = 0;
  for (int bin = 0; bin < cells.count; ++bin) {
    input.offsets[bin] = count;
    for (unsigned p = 0; p < input.counts[bin]; ++p) input.unsorted.cell[count++] = bin;
  }
  input.offsets[cells.count] = count;
  prefix = count;
  // Unsorted arrival suffix deliberately reverses keys and crosses neighbours.
  for (int i = 0; i < 5; ++i) input.unsorted.cell[count++] = cells.encode(7 - i, i % 2, 0);
  for (unsigned i = 0; i < count; ++i) {
    auto& p = input.unsorted;
    p.rx[i] = (i % 16) / 16.; p.ry[i] = (i % 8) / 8.; p.rz[i] = (i % 4) / 4.;
    p.u[i] = 0.05 * (int(i % 17) - 8); p.v[i] = 0.07 * (int(i % 13) - 6);
    p.w[i] = 0.09 * (int(i % 11) - 5);
    p.q[i] = (i % 3 == 0 ? -1. : 0.5) * (1 + i % 7);
    p.id[i] = i + 100;
  }
  // Sub-ULP negative crossing exercises canonical fraction endpoint repair.
  input.unsorted.rx[0] = 0;
  input.unsorted.u[0] = -1.e-30;
  input.unsorted.v[0] = input.unsorted.w[0] = 0;
  input.sorted = input.unsorted;
  for (int c = 0; c < 3; ++c)
    for (int k = 0; k <= nz; ++k)
      for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
          const auto node = i + pitch_x * (j + pitch_y * k);
          input.magnetic[c * nodes + node] = (c + 1) * (0.02 + 0.03*i - 0.04*j + 0.05*k);
          input.electric[c * nodes + node] = (c + 1) * (-0.03 + 0.015*i + 0.02*j - 0.025*k);
        }
  return count;
}

template <class T> void upload(Allocation& allocation, const T& source, Executor executor) {
  require(Traits::succeeded(Traits::copy_host_to_device_async(allocation.raw_base(), &source,
      sizeof(T), executor.stream())), "sorted kernels: upload failed");
}

template <class T> void download(T& result, Allocation& allocation, Executor executor) {
  require(Traits::succeeded(Traits::take_last_launch_error()), "sorted kernels: launch failed");
  require(Traits::succeeded(Traits::copy_device_to_host_async(&result, allocation.raw_base(),
      sizeof(T), executor.stream())) && Traits::succeeded(Traits::synchronize_stream(executor.stream())),
      "sorted kernels: execution/download failed");
}

template <class Precision> void run(Executor executor) {
  using F = tpic::data::FieldSchema<Precision>;
  using B = FieldView<typename Precision::Storage::Magnetic, typename F::Bnx, typename F::Bny, typename F::Bnz>;
  using E = FieldView<typename Precision::Storage::Electric, typename F::Ex, typename F::Ey, typename F::Ez>;
  using J = FieldView<typename Precision::Storage::Current, typename F::Jx, typename F::Jy, typename F::Jz>;
  using Particles = ParticleView<Precision>;
  constexpr auto unsorted_current = kernels::unsorted_current_kernel<Precision, Particles, B, J>;
  constexpr auto sorted_current = kernels::sorted_current_kernel<Precision, Particles, B, J>;
  auto input = std::make_unique<Fixture<Precision>>();
  auto result = std::make_unique<Fixture<Precision>>();
  unsigned prefix;
  const unsigned count = fill(*input, prefix);
  Allocation allocation;
  require(Traits::succeeded(Allocation::create(sizeof(*input), allocation)), "sorted kernels: allocation failed");
  auto* device = reinterpret_cast<Fixture<Precision>*>(allocation.raw_base());
  const tpic::ecsim::ParticleBins bins{{device->counts, device->offsets, unsigned(cells.count)}, &device->failure, prefix};
  const ParticleView<Precision> unsorted{&device->unsorted}, sorted{&device->sorted};
  const B magnetic{device->magnetic};
  const E electric{device->electric};
  // The sorted deposit reports through invalid[2]; the unsorted deposit and
  // mover share [0]; the sorted mover uses [1].
  const auto deposit = [&] {
    lpfx::device::launch_or_throw<unsorted_current>("sorted kernels: launch", executor, lpfx::device::LaunchGrid{3, 128},
        unsorted, magnetic, J{device->unsorted_current}, cells,
        1.25, -0.17, 0, count, device->invalid);
    // One block forces many grid-stride cache reuses, even on wave64 devices.
    lpfx::device::launch_or_throw<sorted_current>("sorted kernels: launch", executor, lpfx::device::LaunchGrid{1, 128},
        sorted, magnetic, J{device->sorted_current}, cells,
        1.25, -0.17, bins, device->invalid + 2);
    lpfx::device::launch_or_throw<unsorted_current>("sorted kernels: launch", executor, lpfx::device::LaunchGrid{1, 128},
        sorted, magnetic, J{device->sorted_current}, cells,
        1.25, -0.17, prefix, count - prefix, device->invalid + 2);
  };
  upload(allocation, *input, executor);
  deposit();
  download(*result, allocation, executor);
  require(result->invalid[0] == 0 && result->invalid[1] == 0 && result->invalid[2] == 0,
      "sorted kernels: valid deposition rejected");
  const double tolerance = std::is_same_v<typename Precision::Storage::Current, float> ? 3.e-5 : 2.e-12;
  const auto near = [=](double a, double b) { return std::abs(a-b) <= tolerance * (1 + std::max(std::abs(a),std::abs(b))); };
  for (int i = 0; i < 3 * nodes; ++i)
    require(near(result->unsorted_current[i], result->sorted_current[i]), "sorted kernels: cell current differs");

  const ecsim::ParticleMoverParameters<Precision> parameters{0.03, -0.17, 0.25, {1, 1, 1}};
  const auto move = [&](ecsim::ParticleMoverParameters<Precision> mover_parameters,
                        bool local = false) {
    lpfx::device::launch_or_throw<kernels::unsorted_particle_mover_kernel<Precision, Particles, B, E, Producer>>(
        "sorted kernels: launch", executor, lpfx::device::LaunchGrid{3, 128}, magnetic, electric, unsorted,
        Producer{device->unsorted_flags, device->unsorted_registered}, geometry(), mover_parameters,
        0, count, device->invalid);
    const auto sorted_move = [&](auto producer) {
      using Sink = decltype(producer);
      lpfx::device::launch_or_throw<kernels::sorted_particle_mover_kernel<Precision, Particles, B, E, Sink>>(
          "sorted kernels: launch", executor, lpfx::device::LaunchGrid{1, 128}, magnetic, electric, sorted,
          producer, geometry(), mover_parameters, bins, device->invalid + 1);
      lpfx::device::launch_or_throw<kernels::unsorted_particle_mover_kernel<Precision, Particles, B, E, Sink>>(
          "sorted kernels: launch", executor, lpfx::device::LaunchGrid{1, 128}, magnetic, electric, sorted,
          producer, geometry(), mover_parameters, prefix, count - prefix, device->invalid + 1);
    };
    if (local) sorted_move(tpic::communication::LocalParticleDestinationSink{device->invalid + 1, count});
    else sorted_move(Producer{device->sorted_flags, device->sorted_registered});
  };
  upload(allocation, *input, executor);
  move(parameters);
  download(*result, allocation, executor);
  require(result->invalid[0] == 0 && result->invalid[1] == 0, "sorted kernels: valid mover rejected");
  for (unsigned i = 0; i < count; ++i) {
    const auto& a = result->unsorted;
    const auto& b = result->sorted;
    require(a.cell[i] == b.cell[i] && a.rx[i] == b.rx[i] && a.ry[i] == b.ry[i] && a.rz[i] == b.rz[i] &&
        a.u[i] == b.u[i] && a.v[i] == b.v[i] && a.w[i] == b.w[i] && a.q[i] == b.q[i] && a.id[i] == b.id[i],
        "sorted kernels: cached mover changed particle state");
    require(result->unsorted_flags[i] == result->sorted_flags[i] &&
        result->unsorted_registered[i] == 1 && result->sorted_registered[i] == 1,
        "sorted kernels: mover classification missing/duplicated");
  }

  // The local path must preserve particle state exactly across periodic and
  // mirror crossings without writing any classification slot or counter.
  upload(allocation, *input, executor);
  move(parameters, true);
  download(*result, allocation, executor);
  require(result->invalid[0] == 0 && result->invalid[1] == 0,
      "sorted kernels: valid local mover rejected");
  for (unsigned i = 0; i < count; ++i) {
    const auto& a = result->unsorted;
    const auto& b = result->sorted;
    require(a.cell[i] == b.cell[i] && a.rx[i] == b.rx[i] && a.ry[i] == b.ry[i] && a.rz[i] == b.rz[i] &&
        a.u[i] == b.u[i] && a.v[i] == b.v[i] && a.w[i] == b.w[i] && a.q[i] == b.q[i] && a.id[i] == b.id[i],
        "sorted kernels: local mover changed particle state");
    require(result->sorted_flags[i] == 0 && result->sorted_registered[i] == 0,
        "sorted kernels: local mover wrote classification storage");
  }

  // Whole-domain owners fold finite flights before applying the migration
  // halo limit. Exercise multiple periodic and mirror crossings on device.
  for (auto* state : {&input->unsorted, &input->sorted}) {
    state->cell[0] = cells.encode(0, 0, 0);
    state->rx[0] = state->ry[0] = state->rz[0] = 0.25;
    state->u[0] = 2 * nx + 0.5;
    state->v[0] = -(2 * ny + 1.5);
    state->w[0] = -(2 * nz + 0.5);
  }
  const ecsim::ParticleMoverParameters<Precision> long_flight{0, 0, 1, {1, 1, 1}};
  upload(allocation, *input, executor);
  move(long_flight, true);
  download(*result, allocation, executor);
  require(result->invalid[0] == 0 && result->invalid[1] == 0,
      "sorted kernels: valid long local flight rejected");
  for (const auto* state : {&result->unsorted, &result->sorted})
    require(state->cell[0] == cells.encode(0, 1, 0) &&
            state->rx[0] == 0.75 && state->ry[0] == 0.25 && state->rz[0] == 0.75 &&
            state->u[0] == 2 * nx + 0.5 && state->v[0] == 2 * ny + 1.5 &&
            state->w[0] == -(2 * nz + 0.5),
        "sorted kernels: long local flight was not folded canonically");
  (void)fill(*input, prefix);

  // Invalid field gathers must fail without modifying the rejected record.
  input->magnetic[0] = std::numeric_limits<typename Precision::Storage::Magnetic>::quiet_NaN();
  upload(allocation, *input, executor);
  deposit();
  move(parameters, true);
  download(*result, allocation, executor);
  require(result->invalid[0] && result->invalid[1] && result->invalid[2],
      "sorted kernels: nonfinite cached field accepted");
  require(result->sorted.u[0] == input->sorted.u[0] && result->sorted.rx[0] == input->sorted.rx[0] &&
      result->sorted.cell[0] == input->sorted.cell[0], "sorted kernels: invalid mover record modified");
  input->magnetic[0] = 0.02;

  // A mismatched particle key must fail before indexing its cached cell.
  input->sorted.cell[0] = cells.encode(-1, 0, 0);
  upload(allocation, *input, executor);
  deposit();
  move(parameters);
  download(*result, allocation, executor);
  require(result->invalid[1] && result->invalid[2], "sorted kernels: mismatched sorted key accepted");
  input->sorted.cell[0] = input->unsorted.cell[0];
  // Move the singleton bin into a halo cell with consistent offset metadata;
  // cached gathers must not attempt field reads for the unowned cell.
  const int owned = cells.encode(0, 0, 0), halo = cells.encode(-1, 0, 0);
  input->counts[owned] = 0;
  input->counts[halo] = 1;
  input->offsets[owned] = 1;
  input->sorted.cell[0] = halo;
  upload(allocation, *input, executor);
  deposit();
  move(parameters);
  download(*result, allocation, executor);
  require(result->invalid[1] && result->invalid[2], "sorted kernels: occupied halo bin accepted");
  input->counts[owned] = 1;
  input->counts[halo] = 0;
  input->sorted.cell[0] = input->unsorted.cell[0];
  input->offsets[owned] = prefix + 1;
  upload(allocation, *input, executor);
  deposit();
  move(parameters);
  download(*result, allocation, executor);
  require(result->invalid[1] && result->invalid[2], "sorted kernels: invalid bins accepted");

  // The sorted mover visits interior bins only: inconsistent metadata of an
  // empty halo bin touches no particle, raises no flag and changes nothing.
  (void)fill(*input, prefix);
  input->counts[halo] = 3;
  upload(allocation, *input, executor);
  move(parameters);
  download(*result, allocation, executor);
  require(result->invalid[0] == 0 && result->invalid[1] == 0,
      "sorted kernels: empty halo bin metadata rejected by the mover");
  for (unsigned i = 0; i < count; ++i) {
    const auto& a = result->unsorted;
    const auto& b = result->sorted;
    require(a.cell[i] == b.cell[i] && a.rx[i] == b.rx[i] && a.ry[i] == b.ry[i] && a.rz[i] == b.rz[i] &&
        a.u[i] == b.u[i] && a.v[i] == b.v[i] && a.w[i] == b.w[i] &&
        result->unsorted_flags[i] == result->sorted_flags[i] && result->sorted_registered[i] == 1,
        "sorted kernels: empty halo bin metadata changed the mover");
  }
  input->counts[halo] = 0;

  // A bin-level error in the bin holding particle 0 is reported from lane 0.
  // Its unreadable range registers no slot and leaves particle 0 unchanged.
  input->counts[owned] = 2;
  upload(allocation, *input, executor);
  move(parameters);
  download(*result, allocation, executor);
  require(result->invalid[1] && result->sorted_registered[0] == 0 &&
      result->sorted.cell[0] == input->sorted.cell[0] && result->sorted.u[0] == input->sorted.u[0],
      "sorted kernels: unreadable bin of particle 0 accepted");
  input->counts[owned] = 1;

  // A failed sort producer must reject before any bin offset is accessed.
  input->failure = 1;
  upload(allocation, *input, executor);
  tpic::ecsim::ParticleBins failed{{nullptr, nullptr, unsigned(cells.count)}, &device->failure, prefix};
  lpfx::device::launch_or_throw<sorted_current>("sorted kernels: launch", executor, lpfx::device::LaunchGrid{1, 128},
      sorted, magnetic, J{device->sorted_current}, cells,
      1.25, -0.17, failed, device->invalid + 1);
  lpfx::device::launch_or_throw<kernels::sorted_particle_mover_kernel<Precision, Particles, B, E,
      tpic::communication::LocalParticleDestinationSink>>("sorted kernels: launch", executor, lpfx::device::LaunchGrid{1, 128},
      magnetic, electric, sorted, tpic::communication::LocalParticleDestinationSink{device->invalid + 1, count},
      geometry(), parameters, failed, device->invalid + 1);
  download(*result, allocation, executor);
  require(result->invalid[1], "sorted kernels: failed producer accepted");
}

// A zero-B oracle is independent of the device gather, response, shuffle and
// reduction implementations: J at a corner is simply sum(q*v*weight). Dyadic
// fractions make each weight exact while signed velocities exercise reduction
// cancellation. Check tails on both sides of wave32 and wave64 boundaries.
template <class Precision> void check_current_corners(Executor executor) {
  using F = tpic::data::FieldSchema<Precision>;
  using Current = typename Precision::Storage::Current;
  using B = FieldView<typename Precision::Storage::Magnetic, typename F::Bnx, typename F::Bny, typename F::Bnz>;
  using J = FieldView<Current, typename F::Jx, typename F::Jy, typename F::Jz>;
  using Particles = ParticleView<Precision>;
  auto input = std::make_unique<Fixture<Precision>>();
  auto result = std::make_unique<Fixture<Precision>>();
  Allocation allocation;
  require(Traits::succeeded(Allocation::create(sizeof(*input), allocation)), "current corners: allocation failed");
  auto* device = reinterpret_cast<Fixture<Precision>*>(allocation.raw_base());
  const int bin = cells.encode(2, 1, 0);
  const auto initialize = [&](unsigned count) {
    *input = {};
    input->counts[bin] = count;
    for (int b = bin + 1; b <= cells.count; ++b) input->offsets[b] = count;
    for (unsigned p = 0; p < count; ++p) {
      input->sorted.cell[p] = bin;
      input->sorted.rx[p] = (p % 8) / 8.;
      input->sorted.ry[p] = ((p / 3) % 8) / 8.;
      input->sorted.rz[p] = ((p / 5) % 8) / 8.;
      input->sorted.q[p] = (p % 3 == 0 ? -1. : 0.5);
      input->sorted.u[p] = (int(p % 17) - 8) * 0.25;
      input->sorted.v[p] = (int(p % 13) - 6) * 0.125;
      input->sorted.w[p] = (int(p % 11) - 5) * 0.5;
    }
  };
  const auto deposit = [&](unsigned count, bool use_unsorted = false) {
    upload(allocation, *input, executor);
    const tpic::ecsim::ParticleBins bins{{device->counts, device->offsets, unsigned(cells.count)},
        &device->failure, count};
    const Particles particles{&device->sorted};
    if (use_unsorted)
      lpfx::device::launch_or_throw<kernels::unsorted_current_kernel<Precision, Particles, B, J>>(
          "sorted kernels: launch", executor, lpfx::device::LaunchGrid{2, 128},
          particles, B{device->magnetic}, J{device->unsorted_current},
          cells, 1., 0., 0, count, device->invalid);
    lpfx::device::launch_or_throw<kernels::sorted_current_kernel<Precision, Particles, B, J>>(
        "sorted kernels: launch", executor, lpfx::device::LaunchGrid{1, 128},
        particles, B{device->magnetic}, J{device->sorted_current},
        cells, 1., 0., bins, device->invalid + 1);
    download(*result, allocation, executor);
  };
  const unsigned populations[]{0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
      95, 96, 97, 127, 128, 129, 257, 1023};
  for (const unsigned count : populations) {
    initialize(count);
    deposit(count, true);
    require(!result->invalid[0] && !result->invalid[1],
        "current corners: valid occupancy rejected");
    long double expected[3 * nodes]{}, magnitude[3 * nodes]{};
    for (unsigned p = 0; p < count; ++p) {
      const auto& state = input->sorted;
      for (int corner = 0; corner < 8; ++corner) {
        const long double wx = (corner & 1) ? state.rx[p] : 1.L - state.rx[p];
        const long double wy = (corner & 2) ? state.ry[p] : 1.L - state.ry[p];
        const long double wz = (corner & 4) ? state.rz[p] : 1.L - state.rz[p];
        const long double weight = (wx * wy) * wz;
        const int node = 2 + (corner & 1) + pitch_x * (1 + ((corner >> 1) & 1) + pitch_y * (corner >> 2));
        const long double values[]{state.u[p], state.v[p], state.w[p]};
        for (int axis = 0; axis < 3; ++axis) {
          const long double product = state.q[p] * values[axis] * weight;
          const long double contribution = static_cast<Current>(product);
          expected[axis * nodes + node] += contribution;
          magnitude[axis * nodes + node] += std::abs(contribution);
        }
      }
    }
    for (int index = 0; index < 3 * nodes; ++index) {
      const long double epsilon = std::numeric_limits<Current>::epsilon();
      const long double tolerance = 16 * epsilon * (1 + magnitude[index]);
      const long double value = result->sorted_current[index];
      const long double unsorted_value = result->unsorted_current[index];
      require(std::abs(value - expected[index]) <= tolerance && std::abs(unsorted_value - expected[index]) <= tolerance,
          "current corners: occupancy/cancellation differs from independent oracle");
    }
  }

  const auto check_invalid = [&](bool current_invalid) {
    deposit(33);
    require(bool(result->invalid[1]) == current_invalid,
        "current corners: invalid input classification differs");
    for (const Current value : result->sorted_current)
      require(std::isfinite(value), "current corners: rejected contribution contaminated current");
  };
  // Poison the lone particle in the second wave32 tile: a rejected record must
  // stage zero weights and currents and contaminate no corner.
  initialize(33);
  input->sorted.rx[32] = std::numeric_limits<typename Precision::Storage::Position>::quiet_NaN();
  check_invalid(true);
  initialize(33);
  input->sorted.ry[32] = 1;
  check_invalid(true);
  initialize(33);
  input->sorted.u[32] = std::numeric_limits<typename Precision::Storage::Velocity>::infinity();
  check_invalid(true);
  initialize(33);
  input->sorted.q[32] = std::numeric_limits<typename Precision::Storage::Weight>::max();
  input->sorted.u[32] = 2;
  check_invalid(true); // Finite input can overflow the scaled response.

  // All particle products are finite, but the corner-group sum overflows.
  initialize(33);
  for (unsigned p = 0; p < 33; ++p) {
    input->sorted.rx[p] = input->sorted.ry[p] = input->sorted.rz[p] = 0;
    input->sorted.q[p] = 1;
    input->sorted.u[p] = std::numeric_limits<Current>::max() * 0.75;
    input->sorted.v[p] = input->sorted.w[p] = 0;
  }
  check_invalid(true);

  // A subgroup shuffle-down may overflow a nonleader's intermediate when it
  // reads itself beyond the group boundary. Only the leader owns the result.
  initialize(4);
  for (unsigned p = 0; p < 4; ++p) {
    input->sorted.rx[p] = input->sorted.ry[p] = input->sorted.rz[p] = 0;
    input->sorted.q[p] = p == 3 ? double(std::numeric_limits<Current>::max()) * 0.75 : 0;
    input->sorted.u[p] = 1;
    input->sorted.v[p] = input->sorted.w[p] = 0;
  }
  deposit(4);
  require(!result->invalid[1],
      "current corners: finite group-leader result rejected");
  const int sole_node = 2 + pitch_x;
  const Current sole_current = static_cast<Current>(input->sorted.q[3]);
  require(result->sorted_current[sole_node] == sole_current,
      "current corners: finite group-leader contribution lost");

  if constexpr (std::is_same_v<Current, float>) {
    const double large = double(std::numeric_limits<float>::max()) * 2;
    initialize(1);
    input->sorted.rx[0] = input->sorted.ry[0] = input->sorted.rz[0] = 0.5;
    input->sorted.q[0] = large; // Weight is FP64; single velocity storage is FP32.
    input->sorted.u[0] = 1;
    input->sorted.v[0] = input->sorted.w[0] = 0;
    deposit(1);
    require(!result->invalid[1],
        "current corners: finite split FP32 contribution rejected");
    initialize(33);
    input->sorted.rx[32] = input->sorted.ry[32] = input->sorted.rz[32] = 0;
    input->sorted.q[32] = large;
    input->sorted.u[32] = 1;
    input->sorted.v[32] = input->sorted.w[32] = 0;
    check_invalid(true); // FP64 product is finite; storage conversion is not.
    initialize(33);
    input->sorted.rx[32] = input->sorted.ry[32] = input->sorted.rz[32] = 0;
    input->sorted.q[32] = large;
    input->sorted.u[32] = input->sorted.v[32] = input->sorted.w[32] = 0;
    check_invalid(false); // A charge that overflows only as a density must not reject J.
  }
}

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}
} // namespace

int main() {
  lpfx::device::DeviceProperties properties{};
  const auto queried = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(queried)) {
    if (runtime_unavailable(queried)) {
      std::cout << "TPIC sorted current/mover skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto stream = Traits::null_stream;
  const auto created = Traits::create_stream(stream);
  if (!Traits::succeeded(created)) return runtime_unavailable(created) ? 77 : 1;
  int status = 0;
  try {
    const Executor executor{stream};
    run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor);
    run<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor);
    run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor);
    check_current_corners<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor);
    check_current_corners<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor);
    check_current_corners<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor);
    std::cout << "TPIC sorted current/mover differential checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(stream))) status = 1;
  return status;
}
