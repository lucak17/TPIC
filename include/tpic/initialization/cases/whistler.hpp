#ifndef TPIC_INITIALIZATION_CASES_WHISTLER_HPP
#define TPIC_INITIALIZATION_CASES_WHISTLER_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace tpic::initialization {

struct WhistlerNoiseMode {
  double wavenumber = 0.0;
  // Raw deterministic component phases, excluding WhistlerPhase. Sampling
  // evaluates (k*x + WhistlerPhase) + phase, in that grouping.
  double phase_y = 0.0;
  double phase_z = 0.0;

  bool operator==(const WhistlerNoiseMode&) const = default;
};

struct WhistlerProfile {
  bool electron_drift = false;
  double wavenumber{}, seed_phase{}, omega{}, time_offset{};
  double current_amplitude{}, current_carrier_density{};

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample&, const ParticleLoadPlan&, double, double,
                                 double, const int*) const noexcept {}
  // The electron flow of the wave is added after the Maxwellian; this addition
  // order is part of the initial state. Folding it into drift changes rounding.
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double x, double, double, double&, double& v,
                                          double& w) const noexcept {
    if (electron_drift) {
      const double phase = (wavenumber * x + seed_phase) - omega * time_offset;
      const double inverse_density = 1.0 / current_carrier_density;
      v += -current_amplitude * ::cos(phase) * inverse_density;
      w += current_amplitude * ::sin(phase) * inverse_density;
    }
  }
  [[nodiscard]] ParticleSupport support() const noexcept { return {}; }
  bool operator==(const WhistlerProfile&) const = default;
};

// Whistler: a parallel whistler eigenmode, magnetic seed or magnetic noise.
struct WhistlerCase {
  static constexpr std::array<std::string_view, 1> names{"Whistler"};
  explicit WhistlerCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] std::string_view particle_label() const { return "WhistlerUniform"; }
  using Profile = WhistlerProfile;
  [[nodiscard]] Profile profile(std::size_t index) const noexcept { return species[index]; }
  bool operator==(const WhistlerCase&) const = default;

  config::WhistlerSeedMode mode = config::WhistlerSeedMode::Eigenmode;
  double wavenumber = 0.0;
  double omega = 0.0;
  double delta_b = 0.0;
  double electric_amplitude = 0.0;
  // Sampling evaluates (k*x + WhistlerPhase) - omega*time_offset in that order.
  double current_amplitude = 0.0;
  int noise_mode_max = 0;
  double noise_amplitude = 0.0;
  std::vector<WhistlerNoiseMode> noise_modes;
  std::vector<Profile> species;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_WHISTLER_HPP
