#include <tpic/initialization/cases/initial_case.hpp>

#include <cmath>

namespace tpic::initialization {

using config::SimulationParameters;

WeibelCase::WeibelCase(const SimulationParameters& params) {
  require_finite(params.weibel_seed_amplitude, "WeibelSeedAmplitude");
  require_finite(params.weibel_seed_growth_rate, "WeibelSeedGrowthRate");
  require_recipe(params.weibel_seed_mode_x >= 0, "WeibelSeedModeX must be non-negative");
  require_recipe(params.weibel_seed_amplitude == 0.0 || params.weibel_seed_mode_x > 0,
          "WeibelSeedModeX must be positive for nonzero seed amplitude");
  wavenumber = mode_wavenumber(params.weibel_seed_mode_x, params.grid.lx);
  // Faraday's law for the growing mode, written for c = 1 (prepare_physics
  // rejects c != 1): E_y = -(growth/(c*k)) * B_s.
  if (params.weibel_seed_amplitude != 0.0) {
    electric_amplitude = representable(
        -(params.weibel_seed_growth_rate / wavenumber) * params.weibel_seed_amplitude,
        "Weibel E amplitude");
  }
}

FieldVector WeibelCase::electric(const SimulationParameters& p, double x, double, double) const {
  FieldVector e{};
  if (p.weibel_seed_amplitude != 0.0 && p.weibel_seed_growth_rate != 0.0) {
    e[1] = electric_amplitude * std::sin(wavenumber * x);
  }
  return e;
}

FieldVector WeibelCase::magnetic(const SimulationParameters& p, double x, double, double) const {
  FieldVector b{};
  if (p.weibel_seed_amplitude != 0.0) {
    b[2] = p.weibel_seed_amplitude * std::cos(wavenumber * x);
  }
  return b;
}

} // namespace tpic::initialization
