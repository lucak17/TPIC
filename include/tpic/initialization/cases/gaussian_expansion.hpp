#ifndef TPIC_INITIALIZATION_CASES_GAUSSIAN_EXPANSION_HPP
#define TPIC_INITIALIZATION_CASES_GAUSSIAN_EXPANSION_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cmath>
#include <cstddef>
#include <string_view>

namespace tpic::initialization {

// Every species carries the same Gaussian weight on the subcell centres of
// the same region (admission requires QuietStart), so every species shares the
// positions.
struct GaussianExpansionProfile {
  // The centre (a grid node) and the cutoff ellipsoid.
  ParticleSupport region;
  // 1/(2*sigma^2) per axis; 0 on a uniform axis.
  double inverse_two_variance[3]{};

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample& sample, const ParticleLoadPlan&,
                                 double x, double y, double z, const int*) const noexcept {
    const double dx = x - region.center[0];
    const double dy = y - region.center[1];
    const double dz = z - region.center[2];
    sample.density *= ::exp(-(dx * dx * inverse_two_variance[0] + dy * dy * inverse_two_variance[1] +
                              dz * dz * inverse_two_variance[2]));
  }
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double, double, double, double&, double&,
                                          double&) const noexcept {}
  [[nodiscard]] ParticleSupport support() const noexcept { return region; }
  bool operator==(const GaussianExpansionProfile&) const = default;
};

// GaussianExpansion: a neutral Gaussian plasma, pair loaded, released into
// vacuum with no field.
struct GaussianExpansionCase {
  static constexpr std::array<std::string_view, 1> names{"GaussianExpansion"};
  explicit GaussianExpansionCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] std::string_view particle_label() const { return "GaussianExpansion"; }
  using Profile = GaussianExpansionProfile;
  [[nodiscard]] Profile profile(std::size_t) const noexcept { return shape; }
  bool operator==(const GaussianExpansionCase&) const = default;

  Profile shape;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_GAUSSIAN_EXPANSION_HPP
