#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MFMA_LAYOUT_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MFMA_LAYOUT_HPP

#include <tpic/ecsim/mass_matrix/mass_matrix_stencil.hpp>
#include <string_view>

// The FP64 MFMA4 architectures: gfx90a, gfx940, gfx941, gfx942 and gfx950.
// Keep this device-pass test and mass_mfma4_supported_architecture's names
// together. Host passes and other device images compile the flag-setting body.
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
     defined(__gfx942__) || defined(__gfx950__))
#define TPIC_MASS_MFMA4_DEVICE_TARGET 1
#else
#define TPIC_MASS_MFMA4_DEVICE_TARGET 0
#endif

namespace tpic::ecsim {

// MFMA4 layout with CBSZ/ABID/BLGP all zero. The four independent
// hardware blocks are interleaved across wave64, and hold the four 4x4
// quadrants of one 8x8 corner product. Input K is the particle within a group.
struct MassMfma4Layout {
  static constexpr unsigned width = 64;
  // One output pair per lane.
  static constexpr unsigned slots = 1;
  // Particles accumulated by one MFMA4 instruction: K of the 4x4x4 product.
  static constexpr int group_particles = 4;
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr unsigned particle(unsigned lane) { return lane / 16; }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr unsigned quadrant(unsigned lane) { return (lane / 4) % 4; }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr int input_corner(unsigned lane) {
    return 4 * (quadrant(lane) / 2) + lane % 4;
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr unsigned partner_source(unsigned lane) {
    return 16 * particle(lane) + 8 * (quadrant(lane) % 2) + lane % 4;
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr int output_a(unsigned lane) {
    return 4 * (quadrant(lane) / 2) + lane / 16;
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr int output_b(unsigned lane, unsigned /*slot*/) {
    return 4 * (quadrant(lane) % 2) + lane % 4;
  }
};

static_assert(valid_mass_lane_stencil_map<MassMfma4Layout>());

// Architecture names can include feature suffixes such as gfx90a:xnack-.
[[nodiscard]] constexpr bool mass_mfma4_supported_architecture(
    std::string_view name, int warp_size) noexcept {
  const auto arch = name.substr(0, name.find(':'));
  return warp_size == 64 && (arch == "gfx90a" || arch == "gfx940" ||
      arch == "gfx941" || arch == "gfx942" || arch == "gfx950");
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_MFMA_LAYOUT_HPP
