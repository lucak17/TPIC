#ifndef TPIC_ECSIM_PARTICLES_PARTICLE_RESPONSE_HPP
#define TPIC_ECSIM_PARTICLES_PARTICLE_RESPONSE_HPP

#include <lpfx/core/attributes.hpp>

namespace tpic::ecsim {

// Spatial pair symmetry in the mass matrix does not make this tensor
// symmetric. All nine row-major entries are stored, each with its own sign.
template <class Compute> struct ParticleResponse {
  Compute a[9];
  // The response is finite whenever this nonnegative denominator input is
  // finite. Callers that need validation can check it without retesting all
  // nine bounded response entries.
  Compute norm_square;

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE ParticleResponse(Compute bx, Compute by, Compute bz,
                                    Compute beta_over_c) noexcept {
    const Compute x = beta_over_c * bx;
    const Compute y = beta_over_c * by;
    const Compute z = beta_over_c * bz;
    norm_square = x*x + y*y + z*z;
    const Compute inverse = Compute(1) / (Compute(1) + norm_square);
    a[0] = (Compute(1) + x*x) * inverse;
    a[4] = (Compute(1) + y*y) * inverse;
    a[8] = (Compute(1) + z*z) * inverse;
    const Compute xy = x*y*inverse, xz = x*z*inverse, yz = y*z*inverse;
    const Compute sx = x*inverse, sy = y*inverse, sz = z*inverse;
    a[1] = xy + sz;
    a[2] = xz - sy;
    a[3] = xy - sz;
    a[5] = yz + sx;
    a[6] = xz + sy;
    a[7] = yz - sx;
  }

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void apply(Compute x, Compute y, Compute z,
                              Compute& out_x, Compute& out_y,
                              Compute& out_z) const noexcept {
    out_x = a[0]*x + a[1]*y + a[2]*z;
    out_y = a[3]*x + a[4]*y + a[5]*z;
    out_z = a[6]*x + a[7]*y + a[8]*z;
  }
};

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_PARTICLES_PARTICLE_RESPONSE_HPP
