#ifndef TPIC_INITIALIZATION_CASES_WEIBEL_HPP
#define TPIC_INITIALIZATION_CASES_WEIBEL_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cstddef>
#include <string_view>

namespace tpic::initialization {

// Weibel: a seeded transverse magnetic mode with uniform loading.
struct WeibelCase {
  static constexpr std::array<std::string_view, 1> names{"Weibel"};
  explicit WeibelCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] std::string_view particle_label() const { return "uniform"; }
  using Profile = UniformProfile;
  [[nodiscard]] Profile profile(std::size_t) const noexcept { return {}; }
  bool operator==(const WeibelCase&) const = default;

  double wavenumber = 0.0;
  // Multiplier of sin(k*x) for the seeded E component.
  double electric_amplitude = 0.0;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_WEIBEL_HPP
