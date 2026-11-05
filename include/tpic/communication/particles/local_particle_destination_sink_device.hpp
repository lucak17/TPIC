#ifndef TPIC_COMMUNICATION_PARTICLES_LOCAL_PARTICLE_DESTINATION_SINK_DEVICE_HPP
#define TPIC_COMMUNICATION_PARTICLES_LOCAL_PARTICLE_DESTINATION_SINK_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <cstdint>

namespace tpic::communication {

// Destination sink of a whole-domain rank over count records, used by the
// local mover and the local startup ownership pass in place of an LPFx
// classification batch. It stores nothing on success: a valid record has
// zero owner-coordinate bits and destination zero; any other flag is an
// invalid record or an unexpected remote destination and marks the
// population invalid.
struct LocalParticleDestinationSink {
  unsigned* invalid;
  unsigned count;
  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE unsigned size() const noexcept { return count; }
  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool register_flag(unsigned, std::uint64_t flag) const noexcept {
    if (flag == 0) return true;
    atomicExch(invalid, 1U);
    return false;
  }
};

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_PARTICLES_LOCAL_PARTICLE_DESTINATION_SINK_DEVICE_HPP
