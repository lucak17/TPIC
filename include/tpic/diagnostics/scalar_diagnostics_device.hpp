#ifndef TPIC_DIAGNOSTICS_SCALAR_DIAGNOSTICS_DEVICE_HPP
#define TPIC_DIAGNOSTICS_SCALAR_DIAGNOSTICS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/distributed/collective_operand.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/block_reduction_device.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/data/output/scalar_sample.hpp>
#include <tpic/ecsim/fields/wall_faces.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <utility>

namespace tpic::diagnostics {

namespace detail {

using ScalarLaunch = lpfx::device::KernelLaunch<>;

// Reduce all observables of a block together (one barrier) into one partial
// per component and block. The fixed scratch bound is independent of the
// population; launch enough blocks to occupy the device.
template <std::size_t Components, class Reduction>
__device__ LPFX_ALWAYS_INLINE void write_scalar_block_partials(Reduction (&value)[Components],
    Reduction* contributions) {
  __shared__ Reduction scratch[Components][lpfx::device::subgroup::block_warp_count<ScalarLaunch::threads>];
  lpfx::device::subgroup::block_reduce_sum<ScalarLaunch::threads>(value, scratch);
  if (threadIdx.x == 0) {
    #pragma unroll
    for (unsigned component = 0; component < Components; ++component)
      contributions[component * data::contribution_count + blockIdx.x] = value[component];
  }
}

template <bool Nodes, class Reduction, class View>
__device__ LPFX_ALWAYS_INLINE Reduction field_energy_thread_contribution(View field, ecsim::WallFaces volume,
    Reduction normalization) {
  const unsigned nx = volume.cells[0] + (Nodes && volume.upper[0]);
  const unsigned ny = volume.cells[1] + (Nodes && volume.upper[1]);
  const unsigned nz = volume.cells[2] + (Nodes && volume.upper[2]);
  const unsigned count = nx * ny * nz;
  const auto x = field.template component<0>();
  const auto y = field.template component<1>();
  const auto z = field.template component<2>();
  Reduction energy{};
  lpfx::device::for_each_grid_index(count, [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, nx, ny);
    const int i = at.i, j = at.j, k = at.k;
    const Reduction fx = config::convert<Reduction>(x(i,j,k)), fy = config::convert<Reduction>(y(i,j,k)),
                    fz = config::convert<Reduction>(z(i,j,k));
    const Reduction weight = Nodes ? volume.inverse_dual_volume_factor<Reduction>(i,j,k)
                                   : Reduction(1);
    energy += weight * (fx * fx + fy * fy + fz * fz);
  });
  return normalization * energy;
}

} // namespace detail

namespace kernels {

template <class Reduction, class Electric, class Magnetic>
__global__ __launch_bounds__(detail::ScalarLaunch::threads) void field_energy_partials_kernel(
    Electric electric, Magnetic magnetic, Reduction* contributions, ecsim::WallFaces volume,
    Reduction normalization) {
  Reduction energy[data::field_scalar_count] = {
      detail::field_energy_thread_contribution<true>(electric, volume, normalization),
      detail::field_energy_thread_contribution<false>(magnetic, volume, normalization)};
  detail::write_scalar_block_partials(energy, contributions);
}

template <class Precision, class View>
__global__ __launch_bounds__(detail::ScalarLaunch::threads) void species_scalar_partials_kernel(
    View particles, typename Precision::Compute::Reduction* contributions,
    typename Precision::Compute::Reduction inverse_qom) {
  using Schema = data::ParticleSchema<Precision>;
  using Reduction = typename Precision::Compute::Reduction;
  using Index = data::SpeciesScalarIndex;
  const auto* q = particles.template get<typename Schema::Charge>();
  const auto* u = particles.template get<typename Schema::U>();
  const auto* v = particles.template get<typename Schema::V>();
  const auto* w = particles.template get<typename Schema::W>();
  Reduction value[data::species_scalar_count]{};
  lpfx::device::for_each_grid_index<unsigned>(particles.size(), [&](const unsigned index) {
    const Reduction weight = config::convert<Reduction>(q[index]), mass = weight * inverse_qom;
    const Reduction vx = config::convert<Reduction>(u[index]), vy = config::convert<Reduction>(v[index]),
                    vz = config::convert<Reduction>(w[index]);
    value[Index::kinetic_energy] += Reduction(0.5) * mass * (vx * vx + vy * vy + vz * vz);
    value[Index::charge] += weight;
    value[Index::momentum_x] += mass * vx;
    value[Index::momentum_y] += mass * vy;
    value[Index::momentum_z] += mass * vz;
  });
  detail::write_scalar_block_partials(value, contributions);
}

} // namespace kernels

// Each active lane reuses independent block partials and writes disjoint slots
// of one packed collective operand. Both executors must share a selected device;
// callers publish input readiness to them before queuing any producers.
// Register the operand once, including its zero-count form when disabled.
// This object, its executors and scalar addresses must outlive communication
// close. Both lanes belong to diagnostics for the complete scalar epoch.
template <class Precision, class Backend> class ScalarDiagnostics {
  using Reduction = typename Precision::Compute::Reduction;
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Allocation = lpfx::device::DeviceAllocation<Backend>;
  using Pinned = lpfx::device::PinnedHostAllocation<Backend>;
  using Executor = lpfx::device::Executor<Backend>;

public:
  ScalarDiagnostics(const config::PreparedDomain& domain, std::size_t species,
                    bool enabled, const lpfx::device::DeviceProperties& properties,
                    Executor executor, Executor helper_executor)
      : executor_(executor), helper_executor_(helper_executor), species_count_(species),
        scalar_count_(enabled ? data::scalar_count(species) : 0),
        ranks_(domain.field_neighborhood.topology().input().communicator_size()),
        normalization_(domain.cell_volume / (8 * std::numbers::pi_v<double>)) {
    lpfx::require(species > 0 && std::cmp_less_equal(species, config::max_real_species) && executor.bound(),
                  "TPIC scalar diagnostics: invalid species count or executor");
    if (!enabled) return;
    lpfx::require(helper_executor.bound() && ranks_ > 0 && properties.multiprocessor_count > 0 &&
                      properties.maximum_threads_per_block >= detail::ScalarLaunch::threads &&
                      properties.maximum_grid_size_x > 0,
                  "TPIC scalar diagnostics: invalid device properties or topology");
    max_blocks_ = static_cast<unsigned>(std::min<std::uint64_t>(data::contribution_count,
        std::min<std::uint64_t>(4ULL * properties.multiprocessor_count,
                               properties.maximum_grid_size_x)));
    node_count_ = domain.interior_node_count;
    lpfx::require(
        ecsim::try_prepare_wall_faces(domain, volume_) == lpfx::ErrorCode::success &&
            domain.interior_node_count > 0 && std::isfinite(normalization_) && normalization_ > 0,
        "TPIC scalar diagnostics: invalid energy quadrature");
    values_ = allocate(scalar_count_ * sizeof(Reduction));
    lpfx::device::check_device_error<Backend>(
        Pinned::create(scalar_count_ * sizeof(Reduction), host_values_),
        "TPIC scalar diagnostics: pinned scalar allocation failed");
    scratch_ = allocate(data::scalar_scratch_values * sizeof(Reduction));
    if (parallel_reductions()) {
      helper_scratch_ = allocate(data::scalar_scratch_values * sizeof(Reduction));
      lpfx::device::create_event(helper_ready_event_,
          "TPIC scalar diagnostics: helper completion event allocation failed");
    }
  }

  ScalarDiagnostics(const ScalarDiagnostics&) = delete;
  ScalarDiagnostics& operator=(const ScalarDiagnostics&) = delete;

  ~ScalarDiagnostics() noexcept {
    if (!work_pending_) return;
    // A producer or copy can fail after accepting work. Both private lanes
    // must retire before their scratch or pinned destination is released.
    if (!Traits::succeeded(Traits::synchronize_stream(executor_.stream())) ||
        (parallel_reductions() &&
         !Traits::succeeded(Traits::synchronize_stream(helper_executor_.stream()))))
      std::abort();
  }

  [[nodiscard]] bool enabled() const noexcept { return scalar_count_ != 0; }
  [[nodiscard]] std::size_t scalar_count() const noexcept { return scalar_count_; }
  [[nodiscard]] bool parallel_reductions() const noexcept {
    return enabled() && helper_executor_.stream() != executor_.stream();
  }
  [[nodiscard]] Executor species_executor(std::size_t index) const {
    lpfx::require(index < species_count_, "TPIC scalar diagnostics: invalid species executor index");
    // Start on the helper so even one species can overlap field energy.
    return parallel_reductions() && index % 2 == 0 ? helper_executor_ : executor_;
  }
  [[nodiscard]] auto collective_operand() {
    return lpfx::distributed::device_collective_in_place<Backend>(values(), scalar_count_);
  }

  template <class ElectricOwner, class MagneticOwner>
  // Enqueues the field-energy partials and their reduction on the primary
  // executor into the field slots. E and B must be readable on that executor;
  // host_blocking_finish() sums the slots. Once per epoch.
  void enqueue_field_energy(const ElectricOwner& electric, const MagneticOwner& magnetic) {
    lpfx::require(
        enabled() && electric.initialized() && magnetic.initialized() &&
            electric.cells() == magnetic.cells() && electric.cells().x() == volume_.cells[0] &&
            electric.cells().y() == volume_.cells[1] && electric.cells().z() == volume_.cells[2],
        "TPIC scalar diagnostics: invalid field owners");
    for (std::size_t component = 0; component < data::field_scalar_count; ++component)
      lpfx::require(!ready_[component], "TPIC scalar diagnostics: field producer already queued");
    const unsigned blocks = detail::ScalarLaunch::blocks_at_least_one(node_count_, max_blocks_);
    work_pending_ = true;
    lpfx::device::launch_or_throw<kernels::field_energy_partials_kernel<
        Reduction, decltype(electric.view()), decltype(magnetic.view())>>(
        "TPIC scalar diagnostics: contribution kernel failed", executor_,
        lpfx::device::LaunchGrid{blocks, detail::ScalarLaunch::threads},
        electric.view(), magnetic.view(), scratch(scratch_), volume_, normalization_);
    enqueue_scalar_reduction(data::field_scalar_count, blocks, 0, executor_, scratch(scratch_));
    for (std::size_t component = 0; component < data::field_scalar_count; ++component)
      ready_[component] = true;
  }

  template <class ParticleOwner>
  // Enqueues one species' partials and their reduction on species_executor(index)
  // into the species' slots. The owner must be readable on that executor and qom
  // finite and nonzero; host_blocking_finish() sums the slots. Once per species and epoch.
  void enqueue_species_scalars(std::size_t index, const ParticleOwner& particles, double qom) {
    lpfx::require(enabled() && index < species_count_ && particles.valid() &&
                      particles.size() <=
                          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) &&
                      std::isfinite(qom) && qom != 0 && std::isfinite(Reduction(1) / qom),
                  "TPIC scalar diagnostics: invalid particle owner or charge-to-mass ratio");
    const auto output = data::field_scalar_count + data::species_scalar_count * index;
    for (std::size_t component = 0; component < data::species_scalar_count; ++component)
      lpfx::require(!ready_[output + component],
                    "TPIC scalar diagnostics: species producer already queued");
    // FIFO order protects scratch reuse within each lane; the two lanes write
    // independent result slots without atomics or a population-sized workspace.
    const auto executor = species_executor(index);
    auto* partials = scratch(executor.stream() == executor_.stream() ? scratch_ : helper_scratch_);
    const unsigned blocks =
        detail::ScalarLaunch::blocks_at_least_one(unsigned(particles.size()), max_blocks_);
    work_pending_ = true;
    lpfx::device::launch_or_throw<kernels::species_scalar_partials_kernel<
        Precision, decltype(particles.cview())>>(
        "TPIC scalar diagnostics: contribution kernel failed", executor,
        lpfx::device::LaunchGrid{blocks, detail::ScalarLaunch::threads},
        particles.cview(), partials, Reduction(1) / Reduction(qom));
    enqueue_scalar_reduction(data::species_scalar_count, blocks, output, executor, partials);
    for (std::size_t component = 0; component < data::species_scalar_count; ++component)
      ready_[output + component] = true;
  }

  template <class Collective>
  [[nodiscard]] data::ScalarDiagnosticSample host_blocking_finish(Collective& collective, int cycle,
                                               double time, bool final_state) {
    lpfx::require(enabled() && cycle >= 0 && std::isfinite(time),
                  "TPIC scalar diagnostics: invalid epoch");
    for (std::size_t component = 0; component < scalar_count_; ++component)
      lpfx::require(ready_[component], "TPIC scalar diagnostics: incomplete scalar producers");
    // The collective keeps its primary executor and one in-place operand.
    // R1: the wait publishes every helper-written slot before either transport
    // or the rank-one download can read the packed array.
    if (parallel_reductions()) {
      lpfx::device::record_event(helper_ready_event_, helper_executor_,
          "TPIC scalar diagnostics: helper completion dependency failed");
      lpfx::device::wait_event(helper_ready_event_, executor_,
          "TPIC scalar diagnostics: helper completion dependency failed");
    }
    // An in-place sum on one rank is the identity. Avoid MPI's device/host
    // visibility fences; the download below is the sole required host join.
    if (ranks_ > 1) {
      lpfx::check_acceptance(collective.begin(), "TPIC scalar diagnostics: global sum rejected");
      lpfx::check_error(collective.wait(), "TPIC scalar diagnostics: LPFx reduction/collective failed");
      lpfx::check_error(collective.conclude(),
                        "TPIC scalar diagnostics: LPFx reduction/collective failed");
    }
    lpfx::device::check_device_error<Backend>(
        Traits::copy_device_to_host_async(host_values_.raw_base(), values(),
                                          scalar_count_ * sizeof(Reduction), executor_.stream()),
        "TPIC scalar diagnostics: scalar download failed");
    lpfx::device::check_device_error<Backend>(
        Traits::synchronize_stream(executor_.stream()),
        "TPIC scalar diagnostics: scalar download completion failed");
    work_pending_ = false;
    data::ScalarDiagnosticSample sample;
    sample.cycle = cycle;
    sample.time = time;
    sample.final_state = final_state;
    sample.species_count = species_count_;
    for (std::size_t component = 0; component < scalar_count_; ++component) {
      Reduction value;
      std::memcpy(&value, host_values_.raw_base() + component * sizeof(Reduction), sizeof(value));
      sample.values[component] = static_cast<double>(value);
      lpfx::require(std::isfinite(sample.values[component]),
                    "TPIC scalar diagnostics: nonfinite global scalar");
    }
    ready_.fill(false);
    return sample;
  }

private:
  static Allocation allocate(std::size_t bytes) {
    Allocation result;
    lpfx::device::check_device_error<Backend>(Allocation::create(bytes, result),
                                              "TPIC scalar diagnostics: device allocation failed");
    return result;
  }
  Reduction* values() { return reinterpret_cast<Reduction*>(values_.raw_base()); }
  static Reduction* scratch(Allocation& allocation) {
    return reinterpret_cast<Reduction*>(allocation.raw_base());
  }
  void enqueue_scalar_reduction(unsigned components, unsigned blocks, std::size_t output,
                Executor executor, Reduction* partials) {
    lpfx::device::launch_or_throw<device::kernels::reduce_block_partials_kernel<Reduction>>(
        "TPIC scalar diagnostics: contribution kernel failed", executor,
        lpfx::device::LaunchGrid{components, detail::ScalarLaunch::threads},
        partials, values() + output, unsigned(data::contribution_count), blocks, components);
  }

  Executor executor_, helper_executor_;
  std::size_t species_count_, scalar_count_;
  unsigned ranks_, max_blocks_ = 0, node_count_ = 0;
  Reduction normalization_;
  ecsim::WallFaces volume_;
  Allocation values_, scratch_, helper_scratch_; // Outlive the registered collective operand.
  lpfx::device::DeviceEvent<Backend> helper_ready_event_;
  Pinned host_values_;
  bool work_pending_ = false;
  std::array<bool, data::max_scalar_count> ready_{};
};

} // namespace tpic::diagnostics

#endif // TPIC_DIAGNOSTICS_SCALAR_DIAGNOSTICS_DEVICE_HPP
