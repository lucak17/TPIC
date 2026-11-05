#ifndef TPIC_INITIALIZATION_CASES_WHISTLER_TURBULENCE_HPP
#define TPIC_INITIALIZATION_CASES_WHISTLER_TURBULENCE_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace tpic::initialization {

// The seed of decaying whistler turbulence (S. P. Gary, O. Chang & J. Wang, ApJ 755, 142 (2012);
// Section 3 of docs/user/whistler_turbulence.md): helical whistlers of equal magnetic energy on
// B0 = B0z z.
struct WhistlerTurbulenceModes {
  static constexpr int max_index = 3;
  static constexpr int count = 150;
  double wavenumber[3]{};       // 2*pi/L per axis
  double amplitude{};           // |dB| of every mode
  double phase_speed[count]{};  // omega/(c*|k|) of the cold-plasma whistler
  double phase[count]{};

  // Calls visit(mode, nx, ny, nz) for mode = 0..count-1: the wavevectors
  // k = (nx, ny, nz)*wavenumber with nz != 0 and k_perp along x, y or a
  // diagonal, |nx|, |ny|, |nz| <= max_index.
  template <class Visit> LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static void for_each_index(Visit&& visit) noexcept {
    int mode = 0;
    for (int nz = -max_index; nz <= max_index; ++nz)
      for (int ny = -max_index; ny <= max_index; ++ny)
        for (int nx = -max_index; nx <= max_index; ++nx)
          if (nz != 0 && (nx == 0 || ny == 0 || nx == ny || nx == -ny)) visit(mode++, nx, ny, nz);
  }

  // Calls visit(curl, speed, b, kb) for every mode at (x, y, z). b is the unit
  // field e1*cos(psi) - sigma*e2*sin(psi), psi = k.x + phase, sigma =
  // sign(kz), e1 = z x k/|z x k| (x when k is along z) and e2 = k x e1/|k|;
  // kb = k x b/|k|; curl = sigma*|k| is the eigenvalue of curl b = curl*b, the
  // handedness of a whistler; speed = phase_speed.
  template <class Visit>
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void for_each_mode(double x, double y, double z, Visit&& visit) const noexcept {
    for_each_index([&](int mode, int nx, int ny, int nz) {
      const double kx = nx * wavenumber[0], ky = ny * wavenumber[1], kz = nz * wavenumber[2];
      const double perp2 = kx * kx + ky * ky;
      const double k = ::sqrt(perp2 + kz * kz);
      const double sigma = nz > 0 ? 1.0 : -1.0;
      double e1[3]{1.0, 0.0, 0.0}, e2[3]{0.0, sigma, 0.0};
      if (perp2 > 0.0) {
        const double inverse_perp = 1.0 / ::sqrt(perp2), inverse_k = 1.0 / k;
        e1[0] = -ky * inverse_perp;
        e1[1] = kx * inverse_perp;
        e1[2] = 0.0;
        e2[0] = -kz * kx * inverse_perp * inverse_k;
        e2[1] = -kz * ky * inverse_perp * inverse_k;
        e2[2] = perp2 * inverse_perp * inverse_k;
      }
      const double psi = (kx * x + ky * y + kz * z) + phase[mode];
      const double cosine = ::cos(psi), sine = sigma * ::sin(psi);
      const double b[3]{e1[0] * cosine - e2[0] * sine, e1[1] * cosine - e2[1] * sine, e1[2] * cosine - e2[2] * sine};
      const double kb[3]{e2[0] * cosine + e1[0] * sine, e2[1] * cosine + e1[1] * sine, e2[2] * cosine + e1[2] * sine};
      visit(sigma * k, phase_speed[mode], b, kb);
    });
  }
  bool operator==(const WhistlerTurbulenceModes&) const = default;
};

// Admission requires QuietStart and equal npcel: an electron on every ion, zero
// charge with the transverse E.
struct WhistlerTurbulenceProfile {
  // -c*amplitude/(electron rhoINIT sum) for an electron species, 0 otherwise.
  double flow_scale{};
  WhistlerTurbulenceModes modes;

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample&, const ParticleLoadPlan&, double, double,
                                                    double, const int*) const noexcept {}
  // The electron flow of the waves, from Ampere's law with the displacement
  // current and the ions at rest, added after the Maxwellian.
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double x, double y, double z, double& u, double& v,
                                                             double& w) const noexcept {
    if (flow_scale == 0.0) return;
    double flow[3]{};
    modes.for_each_mode(x, y, z, [&](double curl, double speed, const double (&b)[3], const double (&)[3]) {
      const double scale = flow_scale * curl * (1.0 - speed * speed);
      flow[0] += scale * b[0];
      flow[1] += scale * b[1];
      flow[2] += scale * b[2];
    });
    u += flow[0];
    v += flow[1];
    w += flow[2];
  }
  [[nodiscard]] ParticleSupport support() const noexcept { return {}; }
  bool operator==(const WhistlerTurbulenceProfile&) const = default;
};

// WhistlerTurbulence: decaying turbulence of the WhistlerTurbulenceModes on a
// uniform, pair-loaded plasma.
struct WhistlerTurbulenceCase {
  static constexpr std::array<std::string_view, 1> names{"WhistlerTurbulence"};
  explicit WhistlerTurbulenceCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] std::string_view particle_label() const { return "WhistlerTurbulence"; }
  using Profile = WhistlerTurbulenceProfile;
  [[nodiscard]] Profile profile(std::size_t index) const noexcept { return {flow_scale[index], modes}; }
  bool operator==(const WhistlerTurbulenceCase&) const = default;

  WhistlerTurbulenceModes modes;
  std::vector<double> flow_scale;  // per species
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_WHISTLER_TURBULENCE_HPP
