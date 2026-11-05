#ifndef LPFX_DATA_PARTICLE_DEVICE_CUDA_HIP_CLASSIFICATION_HPP
#define LPFX_DATA_PARTICLE_DEVICE_CUDA_HIP_CLASSIFICATION_HPP

#include <cstdint>
#include <type_traits>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/data/particle/device/classification.hpp>

#if (defined(__CUDACC__) || defined(__HIPCC__)) &&                             \
    !defined(LPFX_TEST_DEVICE_HOST_EMULATION)

namespace lpfx::detail {

template <device::DeviceBackend Backend>
struct DeviceParticleRegisteredClassificationAtomics final {
  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE static std::uint32_t
  fetch_add(std::uint32_t *const address,
            const std::uint32_t increment) noexcept {
    static_assert(std::is_same_v<std::uint32_t, unsigned int>);
    return atomicAdd(address, increment);
  }

  LPFX_DEVICE LPFX_ALWAYS_INLINE static void
  store_first(std::uint32_t *const address,
              const std::uint32_t value) noexcept {
    static_assert(std::is_same_v<std::uint32_t, unsigned int>);
    static_cast<void>(atomicCAS(address, 0U, value));
  }
};

} // namespace lpfx::detail

#endif

#endif
