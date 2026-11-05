#ifndef TPIC_INITIALIZATION_CASES_TAYLOR_GREEN_VORTEX_HPP
#define TPIC_INITIALIZATION_CASES_TAYLOR_GREEN_VORTEX_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace tpic::initialization {

struct TaylorGreenVortexProfile {
  double wavenumber[3]{};
  double velocity_amplitude{}, magnetic_amplitude{}, current_split{};
  double current_carrier_density{};

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample&, const ParticleLoadPlan&, double, double,
                                 double, const int*) const noexcept {}
  // The bulk flow is added after the Maxwellian and before the current split;
  // this addition order is part of the initial state.
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double x, double y, double z, double& u, double& v,
                                          double& w) const noexcept {
    const double sin_x = ::sin(wavenumber[0] * x);
    const double cos_x = ::cos(wavenumber[0] * x);
    const double sin_y = ::sin(wavenumber[1] * y);
    const double cos_y = ::cos(wavenumber[1] * y);
    const double sin_z = ::sin(wavenumber[2] * z);
    const double cos_z = ::cos(wavenumber[2] * z);
    u += velocity_amplitude * sin_x * cos_y * cos_z;
    v -= velocity_amplitude * cos_x * sin_y * cos_z;
    // The Ampere drift c*curl(B)/rho, written for c = 1 (prepare_physics
    // rejects c != 1).
    if (magnetic_amplitude != 0.0 && current_split != 0.0) {
      const double inverse_density = 1.0 / current_carrier_density;
      const double scale = current_split * inverse_density;
      const double current_x = -magnetic_amplitude * wavenumber[2] * cos_x * sin_y * sin_z;
      const double current_y = -magnetic_amplitude * wavenumber[2] * sin_x * cos_y * sin_z;
      const double current_z = magnetic_amplitude *
          (wavenumber[0] + wavenumber[1]) * sin_x * sin_y * cos_z;
      u += scale * current_x;
      v += scale * current_y;
      w += scale * current_z;
    }
  }
  [[nodiscard]] ParticleSupport support() const noexcept { return {}; }
  bool operator==(const TaylorGreenVortexProfile&) const = default;
};

// TaylorGreenVortex: a seeded vortex velocity and magnetic field.
struct TaylorGreenVortexCase {
  static constexpr std::array<std::string_view, 1> names{"TaylorGreenVortex"};
  explicit TaylorGreenVortexCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] std::string_view particle_label() const { return "TaylorGreenVortex"; }
  using Profile = TaylorGreenVortexProfile;
  [[nodiscard]] Profile profile(std::size_t index) const noexcept { return species[index]; }
  bool operator==(const TaylorGreenVortexCase&) const = default;

  std::array<double, 3> wavenumber{};
  std::vector<Profile> species;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_TAYLOR_GREEN_VORTEX_HPP
