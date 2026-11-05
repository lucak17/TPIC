#ifndef TPIC_INITIALIZATION_CASES_FLUX_ROPE_HPP
#define TPIC_INITIALIZATION_CASES_FLUX_ROPE_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace tpic::initialization {

struct FluxRopeProfile {
  bool electron_species = false;
  double current_carrier_density{}, species_group_density{};
  double rope_b0{}, rope_width{}, rope_center_x{}, rope_center_y{};
  double rope_pressure_floor{}, rope_pressure_fraction{}, abs_q_over_m{};

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample& sample, const ParticleLoadPlan&, double x,
                                 double y, double, const int*) const noexcept {
    const double xc = x - rope_center_x;
    const double yc = y - rope_center_y;
    const double denominator = xc * xc + yc * yc + rope_width * rope_width;
    const double pressure = rope_pressure_floor +
        rope_b0 * rope_b0 * rope_width * rope_width /
            (2.0 * denominator * denominator);
    const double temperature = rope_pressure_fraction * pressure / species_group_density;
    const double speed = ::sqrt(temperature * abs_q_over_m);
    sample.thermal[0] = sample.thermal[1] = sample.thermal[2] = speed;
    // The electrons carry the Ampere current; the drift is written for c = 1
    // (prepare_physics rejects c != 1).
    if (electron_species) {
      const double current = 2.0 * rope_b0 * rope_width * rope_width /
          (denominator * denominator);
      sample.drift[2] -= current / current_carrier_density;
    }
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double, double, double, double&, double&,
                                          double&) const noexcept {}
  [[nodiscard]] ParticleSupport support() const noexcept { return {}; }
  bool operator==(const FluxRopeProfile&) const = default;
};

// fluxRope: a kink-unstable flux rope in pressure balance.
struct FluxRopeCase {
  static constexpr std::array<std::string_view, 1> names{"fluxRope"};
  explicit FluxRopeCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] std::string_view particle_label() const { return "fluxRope"; }
  using Profile = FluxRopeProfile;
  [[nodiscard]] Profile profile(std::size_t index) const noexcept { return species[index]; }
  bool operator==(const FluxRopeCase&) const = default;

  std::vector<Profile> species;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_FLUX_ROPE_HPP
