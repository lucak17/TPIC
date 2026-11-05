#ifndef TPIC_ECSIM_DEPOSITION_CURRENT_DEPOSITION_DEVICE_HPP
#define TPIC_ECSIM_DEPOSITION_CURRENT_DEPOSITION_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/fields/grid_shape.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <lpfx/backend/launch.hpp>
#include <tpic/ecsim/particles/cic.hpp>
#include <tpic/ecsim/particles/particle_response.hpp>
#include <tpic/ecsim/particles/particle_sample_device.hpp>
#include <tpic/ecsim/deposition/deposition_admission_device.hpp>
#include <tpic/ecsim/particles/sorted_cell_device.hpp>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace tpic::ecsim {
namespace detail {

// One record's contribution, the same in the sorted and unsorted kernels.
template <class Compute> struct CurrentContribution {
  CicWeights<Compute> weights;
  Compute x, y, z;
};

// Rows of the sorted-current contribution tile: the eight CIC weights in rows
// 0-7, then the three current components. Staging the finished weights is
// measured faster than staging the fractions and rebuilding each corner
// weight from them (two FP64 multiplies per particle and corner).
struct SortedCurrentTile {
  static constexpr int jx = cic_corner_count;
  static constexpr int jy = jx + 1, jz = jx + 2, rows = jx + 3;
};

// One rounding policy for every current-deposition product: each product is
// rounded on its own before it is narrowed and accumulated, never contracted
// into the following addition. CUDA pins it with __dmul_rn or __fmul_rn,
// HIP with a contraction-free scope.
template <class Storage, class Compute>
[[nodiscard]] __device__ LPFX_ALWAYS_INLINE Storage deposit_product(
    Compute value, Compute weight) noexcept {
#if defined(__CUDA_ARCH__)
  static_assert(std::is_same_v<Compute, double> || std::is_same_v<Compute, float>);
  if constexpr (std::is_same_v<Compute, double>) return config::convert<Storage>(__dmul_rn(value, weight));
  else return config::convert<Storage>(__fmul_rn(value, weight));
#elif defined(__clang__)
#pragma clang fp contract(off)
  return config::convert<Storage>(value * weight);
#else
  // Host passes of other compilers only parse this device function.
  return config::convert<Storage>(value * weight);
#endif
}

// Key, fractions, gather and response of one record, then q*h. A non-finite B
// or charge propagates to every component of q*h (beta_over_c and the inverse
// volume are admitted finite on the host), so the three component checks are
// the only finiteness checks.
template <class Precision, class CellSource, class ParticleView, class MagneticView>
[[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool evaluate_current_particle(
    ParticleView particles, MagneticView magnetic, CellSource source,
    typename Precision::Compute::Current inverse_volume,
    typename Precision::Compute::Current beta_over_c, std::uint32_t index,
    data::GridIndex& cell,
    CurrentContribution<typename Precision::Compute::Current>& result,
    unsigned* invalid) {
  using Particles = data::ParticleSchema<Precision>;
  using Compute = typename Precision::Compute::Current;
  if (!source.resolve(particles.template get<typename Particles::Cell>()[index], cell)) {
    atomicExch(invalid, 1U);
    return false;
  }
  const auto fraction = data::load_fractions<Precision>(particles, index);
  const Compute rx = config::convert<Compute>(fraction.x), ry = config::convert<Compute>(fraction.y),
                rz = config::convert<Compute>(fraction.z);
  // The stored fractions are tested: a fraction below one can round to one in a
  // narrower compute type, which is still a valid weight.
  if (!data::in_unit_cell(fraction.x, fraction.y, fraction.z)) {
    atomicExch(invalid, 1U);
    return false;
  }
  const auto sample = sample_particle<Precision>(magnetic, cell, rx, ry, rz, beta_over_c);
  const Compute u = config::convert<Compute>(particles.template get<typename Particles::U>()[index]);
  const Compute v = config::convert<Compute>(particles.template get<typename Particles::V>()[index]);
  const Compute w = config::convert<Compute>(particles.template get<typename Particles::W>()[index]);
  const Compute charge = config::convert<Compute>(particles.template get<typename Particles::Charge>()[index]);
  Compute hx, hy, hz;
  sample.response.apply(u, v, w, hx, hy, hz);
  const Compute q = charge * inverse_volume;
  result.weights = sample.weights;
  result.x = q * hx;
  result.y = q * hy;
  result.z = q * hz;
  if (!::isfinite(result.x) || !::isfinite(result.y) || !::isfinite(result.z)) {
    atomicExch(invalid, 1U);
    return false;
  }
  return true;
}

template <class Precision, class ParticleView, class MagneticView, class CurrentView>
__device__ LPFX_ALWAYS_INLINE void deposit_current_particle(
    ParticleView particles, MagneticView magnetic, CurrentView current,
    data::ParticleCellMap cells,
    typename Precision::Compute::Current inverse_volume,
    typename Precision::Compute::Current beta_over_c,
    std::uint32_t index, unsigned* invalid) {
  using Fields = data::FieldSchema<Precision>;
  using Compute = typename Precision::Compute::Current;
  using Current = typename Precision::Storage::Current;
  data::GridIndex cell;
  CurrentContribution<Compute> value;
  if (!evaluate_current_particle<Precision>(particles, magnetic, DecodeKey{cells},
      inverse_volume, beta_over_c, index, cell, value, invalid)) return;
  const auto jx = current.template get<typename Fields::Jx>();
  const auto jy = current.template get<typename Fields::Jy>();
  const auto jz = current.template get<typename Fields::Jz>();
  #pragma unroll
  for (int corner = 0; corner < cic_corner_count; ++corner) {
    const Compute weight = value.weights.value[corner];
    if (weight == Compute(0)) continue;
    const Current x = deposit_product<Current>(value.x, weight);
    const Current y = deposit_product<Current>(value.y, weight);
    const Current z = deposit_product<Current>(value.z, weight);
    if constexpr (narrowing_may_overflow<Current, Compute>) {
      if (!::isfinite(x) || !::isfinite(y) || !::isfinite(z)) {
        atomicExch(invalid, 1U);
        continue;
      }
    }
    const auto node = cic_corner(cell, corner);
    atomicAdd(&jx(node.i, node.j, node.k), x);
    atomicAdd(&jy(node.i, node.j, node.k), y);
    atomicAdd(&jz(node.i, node.j, node.k), z);
  }
}

} // namespace detail

namespace kernels {

// Grid-stride over the records [first, first + count) with KernelLaunch<>
// blocks; each record adds its narrowed q*h atomically. *invalid flags an
// invalid or noninterior record or a non-finite contribution.
template <class Precision, class ParticleView, class MagneticView, class CurrentView>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void unsorted_current_kernel(
    ParticleView particles, MagneticView magnetic, CurrentView current,
    data::ParticleCellMap cells,
    typename Precision::Compute::Current inverse_volume,
    typename Precision::Compute::Current beta_over_c,
    std::uint32_t first, std::uint32_t count, unsigned* invalid) {
  lpfx::device::for_each_grid_index(count, [&](const unsigned offset) {
    detail::deposit_current_particle<Precision>(particles, magnetic, current, cells,
        inverse_volume, beta_over_c, first + offset, invalid);
  });
}

// One native warp per cell bin (SortedCellLaunch): the warp stages its
// particles' contributions in shared memory, reduces them per corner and adds
// once per node and component. *invalid as in unsorted_current_kernel, and
// for a non-finite corner sum of the cell.
template <class Precision, class ParticleView, class MagneticView, class CurrentView>
__global__ __launch_bounds__(ecsim::sorted_cell_threads) void sorted_current_kernel(
    ParticleView particles, MagneticView magnetic, CurrentView current,
    data::ParticleCellMap cells,
    typename Precision::Compute::Current inverse_volume,
    typename Precision::Compute::Current beta_over_c,
    ecsim::ParticleBins bins, unsigned* invalid) {
  namespace subgroup = lpfx::device::subgroup;
  using Fields = data::FieldSchema<Precision>;
  using Compute = typename Precision::Compute::Current;
  using Current = typename Precision::Storage::Current;
  using Tile = detail::SortedCurrentTile;
  // Host parsing reports width one. Its legal placeholder never executes;
  // each CUDA/HIP device image uses its own native wave32/wave64 width.
  constexpr unsigned lanes_per_corner = subgroup::warp_size() >= cic_corner_count ?
      subgroup::warp_size() / cic_corner_count : 1;
  __shared__ Compute cell_magnetic[ecsim::sorted_cell_max_warps][ecsim::cached_cell_values];
  // Preserve the weights already used by the magnetic gather. SoA staging
  // lets each corner load only its own weight and the three current values.
  // Padding separates the eight corner groups' rows across shared banks; an
  // unpadded 128-double row would alias their distinct reads.
  __shared__ Compute contribution_tile[Tile::rows]
      [ecsim::sorted_cell_threads + lanes_per_corner];
  const unsigned lane = subgroup::warp_lane();
  const unsigned corner = lane / lanes_per_corner;
  const unsigned slot = lane % lanes_per_corner;
  const unsigned warp = threadIdx.x / subgroup::warp_size();
  const unsigned warps = blockDim.x / subgroup::warp_size();
  // Every physical lane remains live through each collective, including empty
  // particle tails and invalid records. Empty/failed bins leave whole warps.
  if (!bins.ready(invalid)) return;
  for (unsigned bin = blockIdx.x * warps + warp; bin < bins.view.bin_count;
       bin += gridDim.x * warps) {
    unsigned first, count;
    data::GridIndex cell;
    if (!ecsim::open_sorted_cell(bins, cells, bin, invalid, first, count, cell)) continue;
    const auto cached_magnetic = ecsim::cache_cell_magnetic<Precision>(
        magnetic, cell, cell_magnetic[warp]);
    const ecsim::KnownBin source{static_cast<std::int32_t>(bin), cell};
    Current sum_x = 0, sum_y = 0, sum_z = 0;
    for (unsigned tile = 0; tile < count; tile += subgroup::warp_size()) {
      {
        detail::CurrentContribution<Compute> value{};
        data::GridIndex resolved;
        if (lane < count - tile &&
            !detail::evaluate_current_particle<Precision>(particles, cached_magnetic, source,
                inverse_volume, beta_over_c, first + tile + lane, resolved, value, invalid))
          value = {}; // Reset weights too after a partially failed evaluation.
        #pragma unroll
        for (unsigned c = 0; c < cic_corner_count; ++c)
          contribution_tile[c][threadIdx.x] = value.weights.value[c];
        contribution_tile[Tile::jx][threadIdx.x] = value.x;
        contribution_tile[Tile::jy][threadIdx.x] = value.y;
        contribution_tile[Tile::jz][threadIdx.x] = value.z;
      }
      subgroup::warp_synchronize();
      // Each particle reaches each corner exactly once. Every warp reads
      // only its own disjoint span of the shared tile.
      #pragma unroll
      for (unsigned batch = 0; batch < cic_corner_count; ++batch) {
        const unsigned source_lane = warp * subgroup::warp_size() + batch * lanes_per_corner + slot;
        const Compute weight = contribution_tile[corner][source_lane];
        const Compute jx = contribution_tile[Tile::jx][source_lane];
        const Compute jy = contribution_tile[Tile::jy][source_lane];
        const Compute jz = contribution_tile[Tile::jz][source_lane];
        if (weight == Compute(0)) continue;
        const Current x = detail::deposit_product<Current>(jx, weight);
        const Current y = detail::deposit_product<Current>(jy, weight);
        const Current z = detail::deposit_product<Current>(jz, weight);
        if constexpr (narrowing_may_overflow<Current, Compute>) {
          if (!::isfinite(x) || !::isfinite(y) || !::isfinite(z)) {
            atomicExch(invalid, 1U);
            continue;
          }
        }
        sum_x += x;
        sum_y += y;
        sum_z += z;
      }
      subgroup::warp_synchronize(); // All readers finish before tile reuse.
    }
    const Current x = subgroup::warp_reduce_sum<lanes_per_corner>(sum_x);
    const Current y = subgroup::warp_reduce_sum<lanes_per_corner>(sum_y);
    const Current z = subgroup::warp_reduce_sum<lanes_per_corner>(sum_z);
    if (slot == 0) {
      if (!::isfinite(x) || !::isfinite(y) || !::isfinite(z)) {
        atomicExch(invalid, 1U);
      } else {
        const auto node = cic_corner(cell, corner);
        atomicAdd(&current.template get<typename Fields::Jx>()(node.i, node.j, node.k), x);
        atomicAdd(&current.template get<typename Fields::Jy>()(node.i, node.j, node.k), y);
        atomicAdd(&current.template get<typename Fields::Jz>()(node.i, node.j, node.k), z);
      }
    }
    // All cache readers must finish before this warp reuses its cell storage.
    subgroup::warp_synchronize();
  }
}

} // namespace kernels

namespace detail {

// Admission shared by the three current entry points. It reads no particle
// owner, so it may run before migration publishes its mapping, and it consumes
// the launch-error slot last, so the launch that follows is admitted.
// Contract of every entry point: beta_over_c is (species q/m)*dt/(2*c); the
// macrocharge is signed and not multiplied by q/m again; arithmetic is
// Compute::Current. Stream: executor, on which migration has completed (not
// needed by the resident entry point), the nodal B replicas are coherent, and
// current and *invalid have been cleared.
// A nonzero *invalid (invalid or noninterior record, non-finite contribution of
// one particle or, sorted, of one cell corner) invalidates the whole deposit;
// overflow of the atomic node sums is left to the moment and solver checks.
// Only the inverse cell volume is applied: the caller completes the shared-node
// addition, the dual-volume factors and, with smoothing, the ghost refresh
// (CurrentAssembly) before Jhat is read. Owners outlive the caller's last-use
// event.
template <class Precision, class MagneticOwner, class CurrentOwner, class Backend,
          class Diagnostic>
[[nodiscard]] lpfx::ErrorCode admit_current_deposition(
    const MagneticOwner& magnetic, CurrentOwner& current,
    data::ParticleCellMap cells, double inverse_cell_volume, double beta_over_c,
    unsigned* invalid, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic) {
  static_assert(std::is_same_v<Backend, typename MagneticOwner::space_type::backend_type> &&
                std::is_same_v<Backend, typename CurrentOwner::space_type::backend_type>);
  static_assert(MagneticOwner::component_count == 3 && VectorFieldAt<CurrentOwner, lpfx::Node>);
  using Compute = typename Precision::Compute::Current;
  if (!executor.bound() || invalid == nullptr || cells.count <= 0)
    return lpfx::ErrorCode::invalid_argument;
  if (!magnetic.initialized() || !current.initialized()) return lpfx::ErrorCode::invalid_state;
  data::GridShape shape;
  const auto checked = data::try_make_grid_shape(current.cells(), 1, {}, shape);
  if (checked != lpfx::ErrorCode::success) return checked;
  if (magnetic.cells() != current.cells() ||
      !field_cells_equal(magnetic, cells.nx, cells.ny, cells.nz))
    return lpfx::ErrorCode::invalid_argument;
  if (!std::isfinite(inverse_cell_volume) || inverse_cell_volume <= 0 ||
      !std::isfinite(beta_over_c))
    return lpfx::ErrorCode::invalid_argument;
  const Compute inverse_volume = config::convert<Compute>(inverse_cell_volume);
  const Compute beta = config::convert<Compute>(beta_over_c);
  if (!std::isfinite(inverse_volume) || inverse_volume <= 0 || !std::isfinite(beta))
    return lpfx::ErrorCode::overflow;
  return lpfx::device::begin_device_submission<Backend>(diagnostic);
}

template <class Precision, class MagneticOwner, class CurrentOwner, class Backend,
          class Diagnostic>
[[nodiscard]] lpfx::ErrorCode launch_sorted_current(
    ConstParticleReadView<Precision, Backend> particles, ParticleBins bins,
    const MagneticOwner& magnetic, CurrentOwner& current,
    data::ParticleCellMap cells, double inverse_cell_volume, double beta_over_c,
    unsigned* invalid, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic) {
  using Compute = typename Precision::Compute::Current;
  return lpfx::device::launch<SortedCellLaunch,
      kernels::sorted_current_kernel<Precision, ConstParticleReadView<Precision, Backend>,
          decltype(magnetic.view()), decltype(current.view())>,
      lpfx::device::EmptyWork::launch, lpfx::device::LaunchPreflight::admitted>(
      executor, diagnostic, bins.view.bin_count,
      particles, magnetic.view(), current.view(),
      cells, config::convert<Compute>(inverse_cell_volume),
      config::convert<Compute>(beta_over_c), bins, invalid);
}

} // namespace detail

// Adds the records [first, first + count) of one species into Jhat; charge
// density for output comes from the output-moment kernel. Contract at
// detail::admit_current_deposition. The caller captures *invalid and checks
// it before Jhat is consumed.
template <class Precision, class ParticleOwner, class MagneticOwner,
          class CurrentOwner, class Backend, class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_unsorted_current(
    const ParticleOwner& particles, const MagneticOwner& magnetic,
    CurrentOwner& current, data::ParticleCellMap cells, double inverse_cell_volume, double beta_over_c,
    std::uint32_t first, std::uint32_t count, unsigned* invalid,
    lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  static_assert(std::is_same_v<Backend, typename ParticleOwner::backend_type>);
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  const auto range = ecsim::check_completed_particle_range(particles, first, count);
  if (range != lpfx::ErrorCode::success) return range;
  const auto checked = detail::admit_current_deposition<Precision>(
      magnetic, current, cells, inverse_cell_volume, beta_over_c, invalid, executor, diagnostic);
  if (checked != lpfx::ErrorCode::success) return checked;
  using Compute = typename Precision::Compute::Current;
  const auto inverse_volume = config::convert<Compute>(inverse_cell_volume);
  const auto beta = config::convert<Compute>(beta_over_c);
  return lpfx::device::launch<lpfx::device::KernelLaunch<>,
      kernels::unsorted_current_kernel<Precision, decltype(particles.cview()),
          decltype(magnetic.view()), decltype(current.view())>,
      lpfx::device::EmptyWork::skip, lpfx::device::LaunchPreflight::admitted>(
      executor, diagnostic, count, particles.cview(), magnetic.view(), current.view(),
      cells, inverse_volume, beta, first, count, invalid);
}

// Adds exactly the sorted prefix of a completed receipt; the caller adds the
// tail [receipt.prefix_size(), particles.size()) with enqueue_unsorted_current
// and keeps the receipt through the mover. The per-cell reduction changes the
// summation order, not the per-particle narrowing. Contract at
// detail::admit_current_deposition.
template <class Precision, class ParticleOwner, class MagneticOwner,
          class CurrentOwner, class Backend, class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_sorted_current(
    const ParticleOwner& particles,
    const lpfx::DeviceSortSubmission<typename data::ParticleSchema<Precision>::type, Backend>& receipt,
    const MagneticOwner& magnetic, CurrentOwner& current,
    data::ParticleCellMap cells, double inverse_cell_volume, double beta_over_c, unsigned* invalid,
    lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  static_assert(std::is_same_v<Backend, typename ParticleOwner::backend_type>);
  static_assert(std::is_same_v<decltype(receipt.resident_particle_view()),
                               ConstParticleReadView<Precision, Backend>>,
                "completed receipts and resident capabilities launch one read-view specialization");
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  const auto range = ecsim::check_completed_particle_range(particles, 0, 0);
  if (range != lpfx::ErrorCode::success) return range;
  const auto checked = detail::admit_current_deposition<Precision>(
      magnetic, current, cells, inverse_cell_volume, beta_over_c, invalid, executor, diagnostic);
  if (checked != lpfx::ErrorCode::success) return checked;
  ecsim::ParticleBins bins;
  const auto admitted = ecsim::device_wait_particle_bins(
      particles, receipt, cells, executor, bins);
  if (admitted != lpfx::ErrorCode::success) return admitted;
  return detail::launch_sorted_current<Precision>(
      receipt.resident_particle_view(), bins, magnetic, current,
      cells, inverse_cell_volume, beta_over_c, invalid, executor, diagnostic);
}

// Adds the retained prefix of a partial sort before migration concludes.
// residents and its receipt stay unchanged, and no append or growth is
// admitted, until the caller has handed its last-reader event to the sort
// owner. Contract at detail::admit_current_deposition.
template <class Precision, class MagneticOwner, class CurrentOwner, class Backend, class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_resident_current(
    const lpfx::ResidentParticleRead<typename data::ParticleSchema<Precision>::type, Backend>& residents,
    const MagneticOwner& next_b, CurrentOwner& current,
    data::ParticleCellMap cells, double inverse_cell_volume, double beta_over_c,
    unsigned* invalid, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  const auto checked = detail::admit_current_deposition<Precision>(
      next_b, current, cells, inverse_cell_volume, beta_over_c, invalid, executor, diagnostic);
  if (checked != lpfx::ErrorCode::success) return checked;
  ecsim::ParticleBins bins;
  const auto admitted = ecsim::device_wait_resident_particle_bins(
      residents, cells, executor, bins);
  if (admitted != lpfx::ErrorCode::success) return admitted;
  return detail::launch_sorted_current<Precision>(
      residents.resident_particle_view(), bins, next_b, current,
      cells, inverse_cell_volume, beta_over_c, invalid, executor, diagnostic);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_DEPOSITION_CURRENT_DEPOSITION_DEVICE_HPP
