#ifndef TPIC_COMMUNICATION_PARTICLES_INITIAL_PARTICLE_CLASSIFICATION_DEVICE_HPP
#define TPIC_COMMUNICATION_PARTICLES_INITIAL_PARTICLE_CLASSIFICATION_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <cstdlib>
#include <limits>
#include <utility>

#include <lpfx/backend/backend.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <tpic/communication/particles/local_particle_destination_sink_device.hpp>
#include <tpic/communication/particles/particle_boundaries.hpp>

namespace tpic::communication {

namespace kernels {

// Classifies each stored record and registers its destination flag with the
// producer: the LPFx classification batch, or a LocalParticleDestinationSink
// whose invalid word is the validation flag. Grid-stride over producer.size()
// with KernelLaunch<> blocks.
template <class Precision, class View, class Producer>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void initial_particle_classification_kernel(
    View particles, Producer producer, ParticleMigrationGeometry geometry) {
  lpfx::device::for_each_grid_index(producer.size(), [&](const unsigned index) {
    const auto flag = classify_stored_particle<Precision>(
        geometry, particles, index);
    (void)producer.register_flag(index, flag);
  });
}

} // namespace kernels

// Startup ownership of a whole-domain population, with the boundary and
// canonicalization rules of migration but no classification slots or epoch.
// Stream: executor, with the records complete on it. Rewrites records in place
// (counts and identities unchanged) and sets *invalid for an invalid record.
// The caller has cleared *invalid, captures it next, then may sort the owner.
template <class Precision, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_local_initial_particle_boundaries(
    lpfx::Particles<typename data::ParticleSchema<Precision>::type,
                    lpfx::DeviceSpace<Backend>>& particles,
    const ParticleMigrationGeometry& geometry, unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  if (!executor.bound() || invalid == nullptr || !is_local_particle_geometry(geometry))
    return lpfx::ErrorCode::invalid_argument;
  if (!particles.valid()) return lpfx::ErrorCode::invalid_state;
  if (particles.size() > static_cast<unsigned>(std::numeric_limits<int>::max()))
    return lpfx::ErrorCode::invalid_argument;
  const auto count = static_cast<unsigned>(particles.size());
  return lpfx::device::launch<lpfx::device::KernelLaunch<>, kernels::initial_particle_classification_kernel<
      Precision, decltype(particles.view()), LocalParticleDestinationSink>,
      lpfx::device::EmptyWork::skip>(
      executor, particles.size(), particles.view(), LocalParticleDestinationSink{invalid, count},
      geometry);
}

// Startup ownership pass of a distributed population. Stream: executor, with
// the records complete on it; the batch is prepared and published on it, every index
// registered once, empty ranks included. A numerical failure becomes an LPFx
// admission error before particle traffic. On success the caller consumes the
// batch with its migration and concludes it before reuse.
template <class Precision, class Migration, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_initial_particle_classification(
    Migration& migration,
    lpfx::Particles<typename data::ParticleSchema<Precision>::type,
                    lpfx::DeviceSpace<Backend>>& particles,
    const ParticleMigrationGeometry& geometry, lpfx::device::Executor<Backend> executor) {
  if (!executor.bound() || !particles.valid() ||
      particles.size() > static_cast<unsigned>(std::numeric_limits<int>::max()))
    return lpfx::ErrorCode::invalid_argument;
  const auto begun = lpfx::device::begin_device_submission<Backend>();
  if (begun != lpfx::ErrorCode::success) return begun;
  auto prepared = migration.prepare_classification(executor);
  if (!prepared.has_value()) return prepared.error();
  auto batch = std::move(prepared).value();
  auto result = lpfx::device::launch<lpfx::device::KernelLaunch<>,
      kernels::initial_particle_classification_kernel<
          Precision, decltype(particles.view()), decltype(batch.device_view())>,
      lpfx::device::EmptyWork::skip, lpfx::device::LaunchPreflight::admitted>(
      executor, particles.size(), particles.view(), batch.device_view(), geometry);
  if (result == lpfx::ErrorCode::success) result = batch.publish();
  if (result != lpfx::ErrorCode::success &&
      batch.cancel() != lpfx::ErrorCode::success)
    std::abort(); // Never release storage with an unjoined producer.
  return result;
}

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_PARTICLES_INITIAL_PARTICLE_CLASSIFICATION_DEVICE_HPP
