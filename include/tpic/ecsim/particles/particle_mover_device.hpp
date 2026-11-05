#ifndef TPIC_ECSIM_PARTICLES_PARTICLE_MOVER_DEVICE_HPP
#define TPIC_ECSIM_PARTICLES_PARTICLE_MOVER_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/communication/particles/local_particle_destination_sink_device.hpp>
#include <tpic/communication/particles/particle_boundaries.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/particles/cic.hpp>
#include <tpic/ecsim/particles/particle_mover_parameters.hpp>
#include <tpic/ecsim/particles/particle_response.hpp>
#include <tpic/ecsim/particles/particle_sample_device.hpp>
#include <tpic/ecsim/particles/sorted_cell_device.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

namespace tpic::ecsim {
namespace detail {

// One ECSIM push of a stored record, then boundaries and owner
// classification. Returns the destination flag, or the invalid flag with the
// record unchanged. Checks that stay: the key (through CellSource), the
// charge, the fractions, the pushed velocity and the cell-shift range. A
// non-finite stored velocity, B or E propagates to the pushed velocity
// (+, -, * never turn inf or NaN finite; an infinite B makes the response
// inverse 0, and inf * 0 = NaN reaches every response row). The
// moved fraction is finite or infinite, never NaN; floor keeps the
// infinity, which the range test rejects together with NaN.
template <class Precision, class CellSource, class ParticleView, class MagneticView,
          class ElectricView>
[[nodiscard]] __device__ LPFX_ALWAYS_INLINE std::uint64_t push_and_classify_particle(
    const communication::ParticleMigrationGeometry& geometry,
    ParticleMoverParameters<Precision> parameters, ParticleView particles,
    MagneticView magnetic, ElectricView electric, CellSource source,
    unsigned index) noexcept {
  using Particles = data::ParticleSchema<Precision>;
  using Compute = typename Precision::Compute::Velocity;
  using PositionCompute = typename Precision::Compute::Position;
  using Position = typename Precision::Storage::Position;
  using Velocity = typename Precision::Storage::Velocity;
  constexpr auto invalid = communication::ParticleDestinationFlagDecoder::invalid_flag;
  data::GridIndex cell;
  // The mover's field gathers require completed ownership migration.
  if (!source.resolve(particles.template get<typename Particles::Cell>()[index], cell) ||
      !communication::finite_or_integral(
          particles.template get<typename Particles::Charge>()[index]))
    return invalid;
  const auto fraction = data::load_fractions<Precision>(particles, index);
  if (!data::in_unit_cell(fraction.x, fraction.y, fraction.z)) return invalid;

  // R1: B and E are gathered before the response is built, and the stored
  // velocity is read after both gathers, so neither the nine response entries
  // nor the velocity are live across the 48 field loads. This is the
  // sample_particle core in another order: through sample_particle the double
  // unsorted mover needs 104 registers (16 warps per SM) instead of 74 (24).
  const auto weights = make_cic_weights<Compute>(fraction.x, fraction.y, fraction.z);
  const auto b = gather_cic_magnetic<Precision>(magnetic, cell, weights);
  // Solver Etheta may have its own schema and scalar type; gather it directly.
  const Compute ex = interpolate_cic(electric.template component<0>(), cell, weights);
  const Compute ey = interpolate_cic(electric.template component<1>(), cell, weights);
  const Compute ez = interpolate_cic(electric.template component<2>(), cell, weights);
  const ParticleResponse<Compute> response(b.x, b.y, b.z, parameters.beta_over_c);
  const Compute u = config::convert<Compute>(particles.template get<typename Particles::U>()[index]);
  const Compute v = config::convert<Compute>(particles.template get<typename Particles::V>()[index]);
  const Compute w = config::convert<Compute>(particles.template get<typename Particles::W>()[index]);
  Compute hx, hy, hz, aex, aey, aez;
  response.apply(u, v, w, hx, hy, hz);
  response.apply(ex, ey, ez, aex, aey, aez);
  const Compute px = hx + parameters.beta * aex;
  const Compute py = hy + parameters.beta * aey;
  const Compute pz = hz + parameters.beta * aez;
  communication::ParticleMigrationPoint<Precision> point{{cell.i, cell.j, cell.k},
      {fraction.x, fraction.y, fraction.z}};
  // Two conversion points: the new velocity is converted to Velocity storage
  // before the drift uses it, and each displacement, formed in PositionCompute,
  // is converted to Position storage before it is added to the stored
  // cell-relative fraction.
  point.velocity[0] = config::convert<Velocity>(Compute(2) * px - u);
  point.velocity[1] = config::convert<Velocity>(Compute(2) * py - v);
  point.velocity[2] = config::convert<Velocity>(Compute(2) * pz - w);
  for (int axis = 0; axis < 3; ++axis) {
    if (!communication::finite_or_integral(point.velocity[axis]))
      return invalid;
    const Position displacement = config::convert<Position>(parameters.dt *
        config::convert<PositionCompute>(point.velocity[axis]) * parameters.inverse_spacing[axis]);
    const Position moved = point.fraction[axis] + displacement;
    // The cell shift is floor(double(moved)), never a narrowing float floor.
    // Integer cells remain separate; check before converting any floating cell shift.
    const double shift = ::floor(config::convert<double>(moved));
    if (!(shift >= double(std::numeric_limits<std::int32_t>::min()) &&
          shift <= double(std::numeric_limits<std::int32_t>::max()))) return invalid;
    point.cell[axis] += static_cast<std::int64_t>(shift);
    point.fraction[axis] = moved - config::convert<Position>(shift);
    // A tiny negative crossing can round its remainder up to one in storage.
    // Carry that rounded endpoint to keep fractions canonical and half-open.
    if (point.fraction[axis] == Position(1)) {
      ++point.cell[axis];
      point.fraction[axis] = Position(0);
    }
  }
  // Apply physical BC and produce owner semantics before encoding. Distributed
  // owners also validate the untruncated halo-2 flight. Invalid records retain
  // every original attribute.
  const auto flag = communication::classify_particle_point(geometry, point);
  if (flag == invalid) return flag;
  particles.template get<typename Particles::Cell>()[index] = geometry.cells.encode(
      static_cast<int>(point.cell[0]), static_cast<int>(point.cell[1]),
      static_cast<int>(point.cell[2]));
  particles.template get<typename Particles::Rx>()[index] = point.fraction[0];
  particles.template get<typename Particles::Ry>()[index] = point.fraction[1];
  particles.template get<typename Particles::Rz>()[index] = point.fraction[2];
  particles.template get<typename Particles::U>()[index] = point.velocity[0];
  particles.template get<typename Particles::V>()[index] = point.velocity[1];
  particles.template get<typename Particles::W>()[index] = point.velocity[2];
  return flag;
}

} // namespace detail

namespace kernels {

// Grid-stride over the records [first, first + count) with KernelLaunch<>
// blocks. An invalid record sets *invalid and registers the invalid flag.
template <class Precision, class ParticleView, class MagneticView,
          class ElectricView, class Producer>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void unsorted_particle_mover_kernel(
    MagneticView magnetic, ElectricView electric, ParticleView particles,
    Producer producer, communication::ParticleMigrationGeometry geometry,
    ParticleMoverParameters<Precision> parameters,
    unsigned first, unsigned count, unsigned* invalid) {
  lpfx::device::for_each_grid_index(count, [&](const unsigned offset) {
    const unsigned index = first + offset;
    const auto flag = detail::push_and_classify_particle<Precision>(
        geometry, parameters, particles, magnetic, electric,
        ecsim::DecodeKey{geometry.cells}, index);
    if (flag == communication::ParticleDestinationFlagDecoder::invalid_flag)
      atomicExch(invalid, 1U);
    (void)producer.register_flag(index, flag);
  });
}

// One warp per interior cell; halo bins, which must be empty after
// migration, get no warp. Bin-level errors (failed sort, unreadable range,
// occupied halo bin) are reported from lane 0 only, and register the invalid
// flag with the first index of the warp's own bin range when that range is
// readable and nonempty; after a failed sort no warp registers any particle
// and lane 0 of each warp registers the invalid flag with index 0. Both
// producers ignore the index of an invalid flag.
template <class Precision, class ParticleView, class MagneticView,
          class ElectricView, class Producer>
__global__ __launch_bounds__(ecsim::sorted_cell_threads) void sorted_particle_mover_kernel(
    MagneticView magnetic, ElectricView electric, ParticleView particles,
    Producer producer, communication::ParticleMigrationGeometry geometry,
    ParticleMoverParameters<Precision> parameters,
    ecsim::ParticleBins bins, unsigned* invalid) {
  namespace subgroup = lpfx::device::subgroup;
  using Compute = typename Precision::Compute::Velocity;
  constexpr auto invalid_flag = communication::ParticleDestinationFlagDecoder::invalid_flag;
  // The cell magnetic field, then the cell electric field.
  __shared__ Compute cell_fields[ecsim::sorted_cell_max_warps][2 * ecsim::cached_cell_values];
  const unsigned lane = subgroup::warp_lane();
  const unsigned warp = threadIdx.x / subgroup::warp_size();
  const unsigned warps = blockDim.x / subgroup::warp_size();
  if (bins.failed()) {
    if (lane == 0) {
      atomicExch(invalid, 1U);
      (void)producer.register_flag(0, invalid_flag);
    }
    return;
  }
  const unsigned interior_cells = ecsim::interior_cell_count(geometry.cells);
  for (unsigned interior = blockIdx.x * warps + warp; interior < interior_cells;
       interior += gridDim.x * warps) {
    data::GridIndex cell;
    unsigned bin, first, count;
    const auto status = bins.interior_range(geometry.cells, interior, cell, bin, first, count);
    if (status != ecsim::InteriorBinStatus::open) {
      if (lane == 0) {
        atomicExch(invalid, 1U);
        if (status == ecsim::InteriorBinStatus::halo_occupied && count != 0)
          (void)producer.register_flag(first, invalid_flag);
      }
      continue;
    }
    if (count == 0) continue;
    ecsim::fill_cell_magnetic<Precision>(magnetic, cell, cell_fields[warp]);
    ecsim::fill_cell_nodes(electric.template component<0>(), electric.template component<1>(),
        electric.template component<2>(), cell, cell_fields[warp] + ecsim::cached_cell_values);
    subgroup::warp_synchronize();
    const ecsim::CachedMagneticCell<Precision, Compute> cached_magnetic{{cell_fields[warp], cell}};
    const ecsim::CachedCellNodes<Compute> cached_electric{
        cell_fields[warp] + ecsim::cached_cell_values, cell};
    const ecsim::KnownBin source{static_cast<std::int32_t>(bin), cell};
    for (unsigned offset = lane; offset < count; offset += subgroup::warp_size()) {
      const unsigned index = first + offset;
      const auto flag = detail::push_and_classify_particle<Precision>(
          geometry, parameters, particles, cached_magnetic, cached_electric, source, index);
      if (flag == invalid_flag) atomicExch(invalid, 1U);
      (void)producer.register_flag(index, flag);
    }
    // Tail lanes must join before any lane overwrites this warp's next cache.
    subgroup::warp_synchronize();
  }
}

} // namespace kernels

// Shared body of both movers: v_new = 2*(alpha*v_old + beta*alpha*Etheta) - v_old,
// then x_new = x_dep + dt*v_new, with beta = (q/m)*dt/2 and beta_over_c = beta/c.
// B is the old nodal field that this cycle's deposition read. Stream: executor;
// the caller has cleared `invalid` on it and joined every earlier particle
// reader, output moments included. Fields, particles and parameters outlive the
// launch; nothing is allocated, synchronised or read back. A numerical or flight
// failure sets `invalid` and registers the invalid flag; the return value
// reports submission failures only. A failed step may already have moved valid
// records, so it is fatal and never retried.
namespace detail {

template <class Precision, class MagneticOwner,
          class ElectricOwner, class Backend, class Producer>
[[nodiscard]] lpfx::ErrorCode launch_particle_mover(
    const lpfx::DeviceSortSubmission<typename data::ParticleSchema<Precision>::type, Backend>* receipt,
    const MagneticOwner& magnetic, const ElectricOwner& electric,
    lpfx::Particles<typename data::ParticleSchema<Precision>::type,
                    lpfx::DeviceSpace<Backend>>& particles,
    Producer producer, const communication::ParticleMigrationGeometry& geometry,
    const ParticleMoverParameters<Precision>& parameters,
    unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename MagneticOwner::space_type::backend_type> &&
                std::is_same_v<Backend, typename ElectricOwner::space_type::backend_type>);
  static_assert(MagneticOwner::component_count == 3 && VectorFieldAt<ElectricOwner, lpfx::Node>);
  if (!executor.bound() || invalid == nullptr) return lpfx::ErrorCode::invalid_argument;
  if (!particles.valid() || !magnetic.initialized() || !electric.initialized())
    return lpfx::ErrorCode::invalid_state;
  if (particles.size() > static_cast<unsigned>(std::numeric_limits<int>::max()))
    return lpfx::ErrorCode::overflow;
  // The prepared parameters carry the domain checks; the field extents are
  // checked per call against the geometry they were prepared with.
  if (magnetic.cells() != electric.cells() ||
      !field_cells_equal(electric, geometry.cells.nx, geometry.cells.ny, geometry.cells.nz))
    return lpfx::ErrorCode::invalid_argument;
  const auto begun = lpfx::device::begin_device_submission<Backend>();
  if (begun != lpfx::ErrorCode::success) return begun;
  auto result = lpfx::ErrorCode::success;
  unsigned first = 0;
  using MagneticView = decltype(magnetic.view());
  using ElectricView = decltype(electric.view());
  // The slot was consumed above, so both launches are admitted.
  if (receipt != nullptr) {
    ecsim::ParticleBins bins;
    const auto checked = ecsim::device_wait_particle_bins(particles, *receipt, geometry.cells, executor, bins);
    if (checked != lpfx::ErrorCode::success) return checked;
    // One warp per interior cell; device_wait_particle_bins checked the bins.
    result = lpfx::device::launch<ecsim::SortedCellLaunch, kernels::sorted_particle_mover_kernel<
        Precision, decltype(receipt->pending_particle_view()), MagneticView, ElectricView, Producer>,
        lpfx::device::EmptyWork::launch, lpfx::device::LaunchPreflight::admitted>(
        executor, ecsim::interior_cell_count(geometry.cells), magnetic.view(), electric.view(),
        receipt->pending_particle_view(), producer, geometry, parameters,
        bins, invalid);
    first = bins.prefix;
  }
  const unsigned count = particles.size() - first;
  if (result == lpfx::ErrorCode::success && count != 0) {
    result = lpfx::device::launch<lpfx::device::KernelLaunch<>, kernels::unsorted_particle_mover_kernel<
        Precision, decltype(particles.view()), MagneticView, ElectricView, Producer>,
        lpfx::device::EmptyWork::launch, lpfx::device::LaunchPreflight::admitted>(
        executor, count, magnetic.view(), electric.view(),
        particles.view(), producer, geometry, parameters,
        first, count, invalid);
  }
  return result;
}

} // namespace detail

// Moves the sorted prefix (receipt) and the unsorted tail into one
// classification batch prepared and published on executor; the contract of
// detail::launch_particle_mover applies. On success the caller consumes the
// batch with its migration and concludes it; the bins are stale, so the
// receipt retires after the last reader and before that migration.
template <class Precision, class Migration, class MagneticOwner, class ElectricOwner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_migrating_particle_mover(
    const lpfx::DeviceSortSubmission<typename data::ParticleSchema<Precision>::type, Backend>* receipt,
    const MagneticOwner& magnetic, const ElectricOwner& electric,
    lpfx::Particles<typename data::ParticleSchema<Precision>::type, lpfx::DeviceSpace<Backend>>& particles,
    Migration& migration, const communication::ParticleMigrationGeometry& geometry,
    const ParticleMoverParameters<Precision>& parameters,
    unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  auto prepared = migration.prepare_classification(executor);
  if (!prepared.has_value()) return prepared.error();
  auto batch = std::move(prepared).value();
  auto result = detail::launch_particle_mover<Precision>(receipt, magnetic, electric,
      particles, batch.device_view(), geometry, parameters, invalid, executor);
  if (result == lpfx::ErrorCode::success) result = batch.publish();
  if (result != lpfx::ErrorCode::success && batch.cancel() != lpfx::ErrorCode::success)
    std::abort();
  return result;
}

// The mover of a one-rank particle topology with periodic or mirror boundaries:
// the validation of the migrating mover without a classification batch; the
// contract of detail::launch_particle_mover applies. Next the caller captures
// `invalid`, retires the stale sort receipt and sorts the new epoch.
template <class Precision, class MagneticOwner, class ElectricOwner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_local_particle_mover(
    const lpfx::DeviceSortSubmission<typename data::ParticleSchema<Precision>::type, Backend>* receipt,
    const MagneticOwner& magnetic, const ElectricOwner& electric,
    lpfx::Particles<typename data::ParticleSchema<Precision>::type,
                    lpfx::DeviceSpace<Backend>>& particles,
    const communication::ParticleMigrationGeometry& geometry,
    const ParticleMoverParameters<Precision>& parameters,
    unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  // A whole-domain geometry (one owner on every axis) implies a one-rank
  // particle topology: prepare_particle_migration_geometry ties the two.
  if (!communication::is_local_particle_geometry(geometry))
    return lpfx::ErrorCode::invalid_argument;
  // Boundary handling stays identical to migration mode; an unexpected
  // destination still poisons the caller's validation word.
  return detail::launch_particle_mover<Precision>(receipt, magnetic, electric, particles,
      communication::LocalParticleDestinationSink{invalid, static_cast<unsigned>(particles.size())},
      geometry, parameters, invalid, executor);
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_PARTICLES_PARTICLE_MOVER_DEVICE_HPP
