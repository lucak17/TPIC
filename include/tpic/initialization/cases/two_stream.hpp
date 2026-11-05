#ifndef TPIC_INITIALIZATION_CASES_TWO_STREAM_HPP
#define TPIC_INITIALIZATION_CASES_TWO_STREAM_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace tpic::initialization {

struct TwoStreamProfile {
  bool density_seed = false;
  double wavenumber{}, seed_phase{}, density_amplitude{};

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample& sample, const ParticleLoadPlan&, double x,
                                 double, double, const int*) const noexcept {
    if (density_seed) {
      sample.density_multiplier = 1.0 + density_amplitude *
          ::cos(wavenumber * x + seed_phase);
    }
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double, double, double, double&, double&,
                                          double&) const noexcept {}
  [[nodiscard]] ParticleSupport support() const noexcept { return {}; }
  bool operator==(const TwoStreamProfile&) const = default;
};

// TwoStream: two counter-streaming electron beams on an ion background.
struct TwoStreamCase {
  static constexpr std::array<std::string_view, 1> names{"TwoStream"};
  explicit TwoStreamCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] std::string_view particle_label() const { return "TwoStream"; }
  using Profile = TwoStreamProfile;
  [[nodiscard]] Profile profile(std::size_t index) const noexcept { return species[index]; }
  bool operator==(const TwoStreamCase&) const = default;

  double wavenumber = 0.0;
  // Multiplier of sin(k*x + phase) for the seeded E component.
  double electric_amplitude = 0.0;
  std::vector<Profile> species;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_TWO_STREAM_HPP
