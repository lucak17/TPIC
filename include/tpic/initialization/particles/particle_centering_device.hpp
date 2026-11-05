#ifndef TPIC_INITIALIZATION_PARTICLES_PARTICLE_CENTERING_DEVICE_HPP
#define TPIC_INITIALIZATION_PARTICLES_PARTICLE_CENTERING_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/algorithm/reduce/device/reduction.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/distributed/collective_operand.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/initialization/cases/case_interface.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tpic::initialization {

namespace detail {

// The drift the loader gave a particle: that of its profile at the stored
// position, with the position expressions of initial_particle_load_kernel
// (whose arithmetic is part of the initial state and stays there). The loader
// evaluates the profile at the stored position and cell, after the endpoint
// carry, so the drift found here is the loaded one under either QuietStart.
template <class Profile, class Position>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void loaded_drift(const ParticleLoadPlan& plan, const Profile& profile,
    std::int32_t key, Position rx, Position ry, Position rz, double* drift) noexcept {
  const auto at = plan.cells.decode(key);
  const int cell[3]{at.i, at.j, at.k};
  const double x = plan.origin[0] + (at.i + static_cast<double>(rx)) * plan.spacing[0];
  const double y = plan.origin[1] + (at.j + static_cast<double>(ry)) * plan.spacing[1];
  const double z = plan.origin[2] + (at.k + static_cast<double>(rz)) * plan.spacing[2];
  ParticleProfileSample sample{plan.density, 1.0,
      {plan.drift[0], plan.drift[1], plan.drift[2]},
      {plan.thermal[0], plan.thermal[1], plan.thermal[2]}};
  profile.evaluate(sample, plan, x, y, z, cell);
  drift[0] = sample.drift[0];
  drift[1] = sample.drift[1];
  drift[2] = sample.drift[2];
}

} // namespace detail

namespace kernels {

// One instantiation pair per case profile, around the four global sums.
// Restore = false removes the loaded drift of every particle, so the sums hold
// the rest of the velocities alone: the sampled part and the velocity seed.
// Restore = true puts the drift back and subtracts the charge-weighted mean of
// that rest, sums[1..3] / sums[0]. Each stored velocity is formed in double
// and narrowed once. Grid-stride over the particles with at least one block;
// *invalid flags a non-finite sum, mean or stored velocity.
template <class Precision, class Profile, class View, bool Restore>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void center_particle_velocity_kernel(
    const typename Precision::Compute::Reduction* sums, View particles, ParticleLoadPlan plan,
    Profile profile, unsigned* invalid) {
  static_assert(std::is_trivially_copyable_v<Profile>);
  using Schema = data::ParticleSchema<Precision>;
  using Velocity = typename Precision::Storage::Velocity;
  using Reduction = typename Precision::Compute::Reduction;
  double mean[3]{};
  if constexpr (Restore) {
    // Launch even for an empty local population: every rank validates the same
    // global result. Empty ranks must also have participated in all four sums.
    for (int component = 0; component < velocity_centering_reductions; ++component) {
      if (!::isfinite(sums[component])) {
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(invalid, 1U);
        return;
      }
    }
    // A zero charge sum has no mean: the drift returns without a shift.
    if (sums[0] != Reduction{0}) {
      for (int axis = 0; axis < 3; ++axis) mean[axis] = config::convert<double>(sums[axis + 1] / sums[0]);
    }
    if (!::isfinite(mean[0]) || !::isfinite(mean[1]) || !::isfinite(mean[2])) {
      if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(invalid, 1U);
      return;
    }
  }
  constexpr double sign = Restore ? 1.0 : -1.0;
  auto* u = particles.template get<typename Schema::U>();
  auto* v = particles.template get<typename Schema::V>();
  auto* w = particles.template get<typename Schema::W>();
  const auto* rx = particles.template get<typename Schema::Rx>();
  const auto* ry = particles.template get<typename Schema::Ry>();
  const auto* rz = particles.template get<typename Schema::Rz>();
  const auto* cell = particles.template get<typename Schema::Cell>();
  lpfx::device::for_each_grid_index(particles.size(), [&](const unsigned index) {
    double drift[3];
    detail::loaded_drift(plan, profile, cell[index], rx[index], ry[index], rz[index], drift);
    const Velocity next_u = config::convert<Velocity>((config::convert<double>(u[index]) - mean[0]) + sign * drift[0]);
    const Velocity next_v = config::convert<Velocity>((config::convert<double>(v[index]) - mean[1]) + sign * drift[1]);
    const Velocity next_w = config::convert<Velocity>((config::convert<double>(w[index]) - mean[2]) + sign * drift[2]);
    if (!::isfinite(config::convert<Reduction>(next_u)) ||
        !::isfinite(config::convert<Reduction>(next_v)) ||
        !::isfinite(config::convert<Reduction>(next_w))) {
      atomicExch(invalid, 1U);
    } else {
      u[index] = next_u;
      v[index] = next_v;
      w[index] = next_w;
    }
  });
}

} // namespace kernels

// Cold startup workspace, reusable sequentially across species. Construct with
// the maximum local population before the ONE complete communication registration;
// the transport registry registers collective_operand() as a four-value SUM
// all-reduce. Keep this workspace alive until communication closes. LPFx owns the
// reduction hierarchy and publishes directly into the borrowed device scalars;
// no particle-sized temporary, pinned reduction result or payload transfer exists.
template <class Precision, class Backend> class VelocityCentering {
  using Schema = data::ParticleSchema<Precision>;
  using Reduction = typename Precision::Compute::Reduction;
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Executor = lpfx::device::Executor<Backend>;
  using Allocation = lpfx::device::DeviceAllocation<Backend>;
  using Prepared = lpfx::PreparedDeviceReduction<Reduction>;
  template <lpfx::ReductionOperation Operation>
  using Reducer = lpfx::DeviceReduction<Operation, Reduction, Backend,
      lpfx::DefaultDeviceReductionLaunchPolicy, lpfx::DeviceReductionOutput::device>;
  using Sum = Reducer<lpfx::ReductionOperation::sum>;
  using Dot = Reducer<lpfx::ReductionOperation::dot>;

public:
  VelocityCentering(std::uint32_t capacity,
                   const lpfx::device::DeviceProperties& properties,
                   Executor executor)
      : executor_(executor), prepared_(prepare(capacity, properties, executor)),
        values_(allocate(velocity_centering_reductions * sizeof(Reduction))),
        validation_(allocate(sizeof(unsigned))),
        charge_(make_reducer<lpfx::ReductionOperation::sum>(0)),
        weighted_velocity_{make_reducer<lpfx::ReductionOperation::dot>(1),
                           make_reducer<lpfx::ReductionOperation::dot>(2),
                           make_reducer<lpfx::ReductionOperation::dot>(3)} {}

  VelocityCentering(const VelocityCentering&) = delete;
  VelocityCentering& operator=(const VelocityCentering&) = delete;

  [[nodiscard]] auto collective_operand() {
    return lpfx::distributed::device_collective_in_place<Backend>(
        values(), velocity_centering_reductions);
  }

  // All ranks call in the same species order, including empty local owners,
  // with the particles as plan and profile loaded them: before any migration
  // or sort. The operation MUST bind collective_operand() on this executor.
  // Admission and all producer/reader dependencies are caller-owned; failures
  // must enter the application's collective fatal path. This completes the
  // cold startup correction before returning, reading back only one flag. The
  // charge-weighted mean velocity of the species becomes that of its loaded
  // drift: the mean of the sampled part and of the velocity seed is removed.
  template <class Profile, class Collective>
  void host_blocking_center(lpfx::Particles<typename Schema::type, lpfx::DeviceSpace<Backend>>& particles,
              const ParticleLoadPlan& plan, const Profile& profile, Collective& collective) {
    if (!particles.valid() || particles.size() > prepared_.capacity())
      throw std::invalid_argument("TPIC velocity centering: invalid particle owner or population exceeds workspace");
    lpfx::device::PinnedHostAllocation<Backend> status;
    lpfx::device::check_device_error<Backend>(
        decltype(status)::create(sizeof(unsigned), status),
        "TPIC velocity centering: pinned status allocation failed");
    try {
      auto* flag = reinterpret_cast<unsigned*>(validation_.raw_base());
      lpfx::device::check_device_error<Backend>(
          Traits::memset_device_async(flag, 0, sizeof(unsigned), executor_.stream()),
          "TPIC velocity centering: validation submission failed");
      enqueue_step<false>(particles, plan, profile, flag);
      const auto view = particles.cview();
      const auto* q = view.template get<typename Schema::Charge>();
      lpfx::check_error(charge_.begin(q, particles.size(), executor_),
                        "TPIC velocity centering: LPFx operation failed");
      lpfx::check_error(weighted_velocity_[0].begin(q, view.template get<typename Schema::U>(),
                                                    particles.size(), executor_),
                        "TPIC velocity centering: LPFx operation failed");
      lpfx::check_error(weighted_velocity_[1].begin(q, view.template get<typename Schema::V>(),
                                                    particles.size(), executor_),
                        "TPIC velocity centering: LPFx operation failed");
      lpfx::check_error(weighted_velocity_[2].begin(q, view.template get<typename Schema::W>(),
                                                    particles.size(), executor_),
                        "TPIC velocity centering: LPFx operation failed");
      // Producers and the registered operand use this same stream. The LPFx
      // collective records readiness after the reductions before native MPI.
      const auto accepted = collective.begin();
      lpfx::check_acceptance(accepted, "TPIC velocity centering: global sum rejected");
      lpfx::check_error(collective.wait(), "TPIC velocity centering: LPFx operation failed");
      lpfx::check_error(collective.conclude(), "TPIC velocity centering: LPFx operation failed");
      enqueue_step<true>(particles, plan, profile, flag);
      lpfx::device::check_device_error<Backend>(
          Traits::copy_device_to_host_async(status.raw_base(), flag, sizeof(unsigned),
                                            executor_.stream()),
          "TPIC velocity centering: correction submission failed");
      // Retire AFTER collective writes, correction reads and validation copy.
      // This also finishes the copy before inspecting its host destination.
      host_wait_reductions();
      unsigned invalid;
      std::memcpy(&invalid, status.raw_base(), sizeof(invalid));
      if (invalid != 0)
        throw std::overflow_error("TPIC velocity centering: nonfinite global sum, mean or corrected velocity");
    } catch (...) {
      // LPFx prohibits destruction of pending reducers. Drain every accepted
      // producer and its ordered readers, including partially submitted work.
      if (!Traits::succeeded(Traits::synchronize_stream(executor_.stream()))) std::abort();
      host_wait_reductions();
      throw;
    }
  }

  template <class Profile, class Collective>
  void host_blocking_center_if_requested(
      const ParticleLoadPlan& plan, const Profile& profile,
      lpfx::Particles<typename Schema::type, lpfx::DeviceSpace<Backend>>& particles,
      Collective& collective) {
    // The request must agree across ranks; a zero local size never skips work.
    if (plan.center_velocity) host_blocking_center(particles, plan, profile, collective);
  }

private:
  // Stream: executor_. One block runs even for an empty population.
  template <bool Restore, class Profile>
  void enqueue_step(lpfx::Particles<typename Schema::type, lpfx::DeviceSpace<Backend>>& particles,
                    const ParticleLoadPlan& plan, const Profile& profile, unsigned* flag) {
    using Launch = lpfx::device::KernelLaunch<>;
    lpfx::device::launch_or_throw<kernels::center_particle_velocity_kernel<
        Precision, Profile, decltype(particles.view()), Restore>>(
        "TPIC velocity centering: correction submission failed", executor_,
        lpfx::device::LaunchGrid{Launch::blocks_at_least_one(particles.size()), Launch::threads},
        values(), particles.view(), plan, profile, flag);
  }

  static Prepared prepare(std::uint32_t capacity,
                          const lpfx::device::DeviceProperties& properties,
                          Executor executor) {
    Prepared result;
    if (!executor.bound() || capacity > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        lpfx::try_prepare_device_reduction(capacity, properties, result) != lpfx::ErrorCode::success)
      throw std::invalid_argument("TPIC velocity centering: invalid capacity, device or executor");
    return result;
  }
  static Allocation allocate(std::size_t bytes) {
    Allocation result;
    lpfx::device::check_device_error<Backend>(Allocation::create(bytes, result),
                                              "TPIC velocity centering: device allocation failed");
    return result;
  }
  Reduction* values() { return reinterpret_cast<Reduction*>(values_.raw_base()); }

  template <lpfx::ReductionOperation Operation>
  auto make_reducer(int component) {
    auto result = lpfx::make_device_reduction<Operation, Reduction, Backend>(
        prepared_, values() + component);
    lpfx::check_error(result, "TPIC velocity centering: LPFx reduction allocation failed");
    return std::move(result).value();
  }
  void host_wait_reductions() noexcept {
    const auto finish = [&](auto& reduction) {
      if (reduction.pending() &&
          (!reduction.retire_on(executor_).is_success() || !reduction.wait().is_success()))
        std::abort();
    };
    finish(charge_);
    for (auto& reduction : weighted_velocity_) finish(reduction);
  }

  Executor executor_;
  Prepared prepared_;
  Allocation values_, validation_; // Outlive the reducers borrowing their scalars.
  Sum charge_;
  std::array<Dot, 3> weighted_velocity_;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_PARTICLES_PARTICLE_CENTERING_DEVICE_HPP
