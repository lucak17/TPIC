#ifndef TPIC_ECSIM_DEPOSITION_DEPOSITION_ADMISSION_DEVICE_HPP
#define TPIC_ECSIM_DEPOSITION_DEPOSITION_ADMISSION_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/algorithm/particle/device/sort.hpp>
#include <tpic/data/particles/particle_schema.hpp>

#include <cstdint>
#include <limits>
#include <utility>

namespace tpic::ecsim {

// Completed receipts and early resident reads launch this one const
// specialization.
template <class Precision, class Backend>
using ConstParticleReadView = decltype(std::declval<const lpfx::ResidentParticleRead<
    typename data::ParticleSchema<Precision>::type, Backend>&>().resident_particle_view());

template <class ParticleOwner>
[[nodiscard]] lpfx::ErrorCode check_completed_particle_range(
    const ParticleOwner& particles, std::uint32_t first, std::uint32_t count) noexcept {
  if (!particles.valid()) return lpfx::ErrorCode::invalid_state;
  constexpr auto limit = std::uint32_t(std::numeric_limits<int>::max());
  if (particles.size() > limit || first > limit || count > limit)
    return lpfx::ErrorCode::overflow;
  if (first > particles.size() || count > particles.size() - first)
    return lpfx::ErrorCode::invalid_argument;
  return lpfx::ErrorCode::success;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_DEPOSITION_DEPOSITION_ADMISSION_DEVICE_HPP
