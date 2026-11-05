#ifndef TPIC_INITIALIZATION_PARTICLES_PARTICLE_LOADING_DEVICE_HPP
#define TPIC_INITIALIZATION_PARTICLES_PARTICLE_LOADING_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/data/particle/device/initialization.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/initialization/cases/initial_case.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
#include <curand_kernel.h>
#else
#include <hiprand_kernel.h>
#endif

namespace tpic::initialization {

namespace detail {

// The backend's default XORWOW generator, with sequence = load index. Each
// particle draws three uniform offsets x, y, z (none under QuietStart), one
// normal pair for u and v, then one normal for w. Generator and draw order
// are part of the initial state.
// State lives in the initializing thread; no persistent particle RNG array.
struct ParticleRandom {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  curandState state;
  __device__ LPFX_ALWAYS_INLINE ParticleRandom(std::uint64_t seed, std::uint64_t sequence) {
    curand_init(seed, sequence, 0, &state);
  }
  __device__ LPFX_ALWAYS_INLINE double uniform() { return curand_uniform_double(&state); }
  __device__ LPFX_ALWAYS_INLINE double2 normal_pair() { return curand_normal2_double(&state); }
  __device__ LPFX_ALWAYS_INLINE double normal() { return curand_normal_double(&state); }
#else
  hiprandState state;
  __device__ LPFX_ALWAYS_INLINE ParticleRandom(std::uint64_t seed, std::uint64_t sequence) {
    hiprand_init(seed, sequence, 0, &state);
  }
  __device__ LPFX_ALWAYS_INLINE double uniform() { return hiprand_uniform_double(&state); }
  __device__ LPFX_ALWAYS_INLINE double2 normal_pair() { return hiprand_normal2_double(&state); }
  __device__ LPFX_ALWAYS_INLINE double normal() { return hiprand_normal_double(&state); }
#endif
};

// A bounded load's rows (visit_support_rows) in device memory: first subcell
// and running point count of each; no rows load every subcell.
struct LoadRows {
  const std::uint32_t* first{};
  const std::uint32_t* end{};
  std::uint32_t count{};
};

} // namespace detail

namespace kernels {

// Two instantiations per case profile, Quiet = QuietStart: the profile shapes
// the species Maxwellian of the plan (case_interface.hpp). Load index i is
// subcell i % ppc of local cell i / ppc (both x fastest), or the i-th point of
// the rows; Quiet places it at the subcell centre, otherwise at three uniform
// offsets.
template <class Precision, class Profile, class Range, bool Quiet>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void initial_particle_load_kernel(
    Range range, ParticleLoadPlan plan, detail::LoadRows rows, Profile profile, unsigned* invalid) {
  static_assert(std::is_trivially_copyable_v<Profile>);
  using Schema = data::ParticleSchema<Precision>;
  using Position = typename Precision::Storage::Position;
  using Velocity = typename Precision::Storage::Velocity;
  using Weight = typename Precision::Storage::Weight;
  lpfx::device::for_each_grid_index(range.size(), [&](const unsigned index) {
    int i, j, k, sub_x, sub_y, sub_z;
    if (rows.count == 0) {
      const int cell = index / plan.particles_per_cell;
      const int sub = index % plan.particles_per_cell;
      i = cell % plan.cells.nx;
      j = (cell / plan.cells.nx) % plan.cells.ny;
      k = cell / plan.cells.nx / plan.cells.ny;
      sub_x = sub % plan.ppc[0];
      sub_y = (sub / plan.ppc[0]) % plan.ppc[1];
      sub_z = sub / plan.ppc[0] / plan.ppc[1];
    } else {
      // The first row whose running count exceeds index holds the point.
      std::uint32_t at = 0;
      for (std::uint32_t upper = rows.count - 1; at < upper;) {
        const std::uint32_t middle = at + (upper - at) / 2;
        if (rows.end[middle] > index) upper = middle;
        else at = middle + 1;
      }
      const int s = static_cast<int>(rows.first[at] + (index - (at == 0 ? 0 : rows.end[at - 1])));
      const int row_x = plan.cells.nx * plan.ppc[0], row_y = plan.cells.ny * plan.ppc[1];
      const int x = s % row_x, y = s / row_x % row_y, z = s / row_x / row_y;
      i = x / plan.ppc[0];
      j = y / plan.ppc[1];
      k = z / plan.ppc[2];
      sub_x = x % plan.ppc[0];
      sub_y = y % plan.ppc[1];
      sub_z = z % plan.ppc[2];
    }
    detail::ParticleRandom random(plan.seed, index);
    const double offset_x = Quiet ? 0.5 : random.uniform();
    const double offset_y = Quiet ? 0.5 : random.uniform();
    const double offset_z = Quiet ? 0.5 : random.uniform();
    const Position rx = data::canonicalize_particle_fraction<Position>(i,
        (sub_x + offset_x) * plan.inverse_ppc[0]);
    const Position ry = data::canonicalize_particle_fraction<Position>(j,
        (sub_y + offset_y) * plan.inverse_ppc[1]);
    const Position rz = data::canonicalize_particle_fraction<Position>(k,
        (sub_z + offset_z) * plan.inverse_ppc[2]);
    // Every profile samples the stored (possibly float) position and cell, after
    // the endpoint carry, as velocity centering does: a random offset that rounds
    // to one carries the particle into the next cell, whose nodes then give the
    // Harris density. Subcell centres never carry.
    const int cell[3]{i, j, k};
    const double x = plan.origin[0] + (i + static_cast<double>(rx)) * plan.spacing[0];
    const double y = plan.origin[1] + (j + static_cast<double>(ry)) * plan.spacing[1];
    const double z = plan.origin[2] + (k + static_cast<double>(rz)) * plan.spacing[2];
    ParticleProfileSample sample{plan.density, 1.0,
        {plan.drift[0], plan.drift[1], plan.drift[2]},
        {plan.thermal[0], plan.thermal[1], plan.thermal[2]}};
    profile.evaluate(sample, plan, x, y, z, cell);
    const auto xy = random.normal_pair();
    double u = sample.drift[0] + sample.thermal[0] * xy.x;
    double v = sample.drift[1] + sample.thermal[1] * xy.y;
    const double normal_z = random.normal();
    double w = sample.drift[2] + sample.thermal[2] * normal_z;
    profile.add_velocity_seed(x, y, z, u, v, w);
    const Velocity stored_u = config::convert<Velocity>(u);
    const Velocity stored_v = config::convert<Velocity>(v);
    const Velocity stored_w = config::convert<Velocity>(w);
    const Weight charge = config::convert<Weight>(
        (plan.q_factor * sample.density) * sample.density_multiplier);
    if (!::isfinite(static_cast<double>(stored_u)) ||
        !::isfinite(static_cast<double>(stored_v)) ||
        !::isfinite(static_cast<double>(stored_w)) ||
        !::isfinite(static_cast<double>(charge)))
      atomicExch(invalid, 1U);
    range.template store<typename Schema::Rx>(index, rx);
    range.template store<typename Schema::Ry>(index, ry);
    range.template store<typename Schema::Rz>(index, rz);
    range.template store<typename Schema::U>(index, stored_u);
    range.template store<typename Schema::V>(index, stored_v);
    range.template store<typename Schema::W>(index, stored_w);
    range.template store<typename Schema::Charge>(index, charge);
    range.template store<typename Schema::Cell>(index, plan.cells.encode(i, j, k));
    range.template store<typename Schema::Id>(index, plan.first_id + index);
  });
}

} // namespace kernels

// Startup only, called by make_initial_particles with the whole initial
// population at offset zero on a bound executor; plan.quiet_start selects the
// instantiation. invalid is a caller-owned, zeroed device unsigned on the
// producer stream. It must remain live through completion. No allocation or
// synchronization here.
template <class Precision, class Profile, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_initial_particle_load(
    const ParticleLoadPlan& plan, detail::LoadRows rows, const Profile& profile,
    lpfx::DeviceParticleInitializationRange<typename data::ParticleSchema<Precision>::type,
                                             Backend> range,
    unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  if (invalid == nullptr) return lpfx::ErrorCode::invalid_argument;
  const auto launch = [&]<bool Quiet>() {
    return lpfx::device::launch<lpfx::device::KernelLaunch<>,
        kernels::initial_particle_load_kernel<Precision, Profile, decltype(range), Quiet>,
        lpfx::device::EmptyWork::skip>(executor, range.size(), range, plan, rows, profile, invalid);
  };
  return plan.quiet_start ? launch.template operator()<true>() : launch.template operator()<false>();
}

// Cold-start factory: allocates an address-stable, empty LPFx owner with the
// already-rounded capacity (LPFx's rotating attribute storage); fill(range)
// constructs every attribute of the count records of range in its final device
// slot on executor, before checked size publication. The analytic load and a
// restart's restored records share it.
template <class Precision, class Backend, class Fill>
[[nodiscard]] auto make_initial_particles(std::uint32_t capacity, std::uint32_t count,
    lpfx::device::Executor<Backend> executor, Fill&& fill) {
  if (!executor.bound())
    throw std::invalid_argument("TPIC particle loading: bound executor required");
  using Schema = typename data::ParticleSchema<Precision>::type;
  lpfx::PreparedDeviceParticleStorage<Schema, Backend> storage;
  lpfx::check_error(lpfx::try_prepare_device_particle_storage(capacity, storage),
                    "TPIC particle loading: LPFx storage preparation failed");
  auto made = lpfx::make_device_particles(storage);
  lpfx::check_error(made, "TPIC particle loading: LPFx allocation failed");
  auto owner = std::move(made).value();
  // The initializer borrower dies before the return.
  {
    using Traits = lpfx::device::BackendTraits<Backend>;
    auto prepared = lpfx::make_device_particle_initializer(owner);
    lpfx::check_error(prepared, "TPIC particle loading: LPFx initializer preparation failed");
    auto initializer = std::move(prepared).value();
    const auto admitted = initializer.begin_append(count, executor);
    lpfx::check_error(admitted, "TPIC particle loading: LPFx range admission failed");
    try {
      fill(admitted.value());
      lpfx::check_error(initializer.publish(executor),
                        "TPIC particle loading: checked publication failed");
      lpfx::check_error(initializer.wait(), "TPIC particle loading: checked publication failed");
    } catch (...) {
      // A failed copy submission may still have accepted work. Retire its host
      // source or destination before cancellation or destruction can release it.
      if (!Traits::succeeded(Traits::synchronize_stream(executor.stream()))) std::abort();
      if (initializer.producing() || initializer.pending()) {
        if (!initializer.cancel(executor).is_success()) std::abort();
      }
      throw;
    }
  }
  return owner;
}

// The analytic load of a plan on the domain it was prepared from: every
// subcell, or the subcells of a bounded support. Only a four-byte validation
// flag is read back; the support's rows are the only upload.
template <class Precision, class Profile, class Backend>
[[nodiscard]] auto make_initial_particles(const ParticleLoadPlan& plan, const config::PreparedDomain& domain,
    const Profile& profile, lpfx::device::Executor<Backend> executor) {
  using Traits = lpfx::device::BackendTraits<Backend>;
  lpfx::device::DeviceAllocation<Backend> validation, row_storage;
  lpfx::device::check_device_error<Backend>(
      decltype(validation)::create(sizeof(unsigned), validation),
      "TPIC particle loading: status allocation failed");
  lpfx::device::PinnedHostAllocation<Backend> status, host_rows;
  lpfx::device::check_device_error<Backend>(
      decltype(status)::create(sizeof(unsigned), status),
      "TPIC particle loading: pinned status allocation failed");
  // The rows of a bounded support, written once into one pinned block (first,
  // then end) and uploaded with the load.
  detail::LoadRows rows{};
  std::size_t bytes = 0;
  if (const auto support = profile.support(); support.bounded()) {
    std::uint32_t count = 0;
    visit_support_rows(support, domain, plan.ppc, false, [&](std::int64_t, std::int64_t) { ++count; });
    bytes = 2 * std::size_t{count} * sizeof(std::uint32_t);
    lpfx::device::check_device_error<Backend>(decltype(host_rows)::create(bytes, host_rows),
                                              "TPIC particle loading: pinned row allocation failed");
    lpfx::device::check_device_error<Backend>(decltype(row_storage)::create(bytes, row_storage),
                                              "TPIC particle loading: row allocation failed");
    auto* first = reinterpret_cast<std::uint32_t*>(host_rows.raw_base());
    std::uint32_t at = 0, points = 0;
    visit_support_rows(support, domain, plan.ppc, false, [&](std::int64_t start, std::int64_t length) {
      first[at] = static_cast<std::uint32_t>(start);
      first[count + at++] = points += static_cast<std::uint32_t>(length);
    });
    const auto* device = reinterpret_cast<const std::uint32_t*>(row_storage.raw_base());
    rows = {device, device + count, count};
  }
  return make_initial_particles<Precision>(plan.capacity, plan.count, executor, [&](auto range) {
    auto* invalid = reinterpret_cast<unsigned*>(validation.raw_base());
    lpfx::device::check_device_error<Backend>(
        Traits::memset_device_async(invalid, 0, sizeof(unsigned), executor.stream()),
        "TPIC particle loading: kernel submission failed");
    if (bytes != 0)
      lpfx::device::check_device_error<Backend>(
          Traits::copy_host_to_device_async(row_storage.raw_base(), host_rows.raw_base(), bytes, executor.stream()),
          "TPIC particle loading: row upload failed");
    lpfx::check_error(
        enqueue_initial_particle_load<Precision>(plan, rows, profile, range, invalid, executor),
        "TPIC particle loading: kernel submission failed");
    lpfx::device::check_device_error<Backend>(
        Traits::copy_device_to_host_async(status.raw_base(), invalid, sizeof(unsigned),
                                          executor.stream()),
        "TPIC particle loading: validation readback failed");
    lpfx::device::check_device_error<Backend>(
        Traits::synchronize_stream(executor.stream()),
        "TPIC particle loading: validation readback failed");
    unsigned failed;
    std::memcpy(&failed, status.raw_base(), sizeof(failed));
    if (failed != 0)
      throw std::overflow_error("TPIC particle loading: nonfinite sampled velocity or charge in selected storage precision");
  });
}

// The initial particles of one species of the selected case.
template <class Precision, class Backend>
[[nodiscard]] auto make_initial_particles(const ParticleLoadPlan& plan, const config::PreparedDomain& domain,
    const InitialCase& initial_case, std::size_t species, lpfx::device::Executor<Backend> executor) {
  return std::visit([&](const auto& selected) {
    return make_initial_particles<Precision>(plan, domain, selected.profile(species), executor);
  }, initial_case);
}

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_PARTICLES_PARTICLE_LOADING_DEVICE_HPP
