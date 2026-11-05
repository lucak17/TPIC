#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_KERNEL_SUITE_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_KERNEL_SUITE_HPP

#include <cstddef>
#include <cstdint>

#include <lpfx/distributed/particle/limits.hpp>

namespace lpfx {

inline constexpr std::uint32_t device_particle_migration_block_threads = 256;
inline constexpr std::size_t device_particle_migration_bucket_count =
    particle_migration_maximum_peer_count + 1;
inline constexpr std::uint8_t device_particle_migration_resident_bucket =
    static_cast<std::uint8_t>(particle_migration_maximum_peer_count);

// Admission writes the classification counts and failure word as one complete
// native transfer image. One D2H copy covers the complete result.
struct DeviceParticleMigrationSummary final {
  std::uint32_t counts[device_particle_migration_bucket_count]{};
  std::uint32_t failure{};
};

namespace detail {

enum class DeviceParticleMigrationFailureWord : std::uint32_t {
  none = 0,
  invalid_record = 1,
  write_failure = 4,
};

// Launch seam of the device migration kernels; one static specialization per
// backend (cuda_hip/kernel_suite.hpp). Every enqueue orders its reads and
// writes after prior work on the supplied stream and before later work on it,
// and returns the native launch error. Members:
//   enqueue_registered_initial_admission(source_count, peer_count,
//       registration summary, sub_offsets, migration summary, stream)
//   enqueue_initial_classification<Schema>(source_count, const source view,
//       registered view, protocol, stream)
//   enqueue_received_classification<Contract>(round, source_count,
//       peer_count, const peer wire view, registered view, migration summary,
//       protocol, stream)
//   enqueue_registered_initial_pack<Contract>(source_count, decisions,
//       sub_offsets, const source view, peer wire view, failure_word,
//       codec_context, stream)
//   enqueue_received_processing<Contract>(source_count, decisions,
//       sub_offsets, const peer wire view, final destination view, peer wire
//       view, failure_word, codec_context, stream)
// Classification inputs are fresh slot leases whose registration counters were
// cleared on the stream. Admission writes the complete migration summary,
// including failure = none on success; initial packing and received processing
// follow successful admission and keep that zero failure until a writer
// reports an error.
template <class Backend> struct DeviceParticleKernelSuite;

} // namespace detail

} // namespace lpfx

#endif
