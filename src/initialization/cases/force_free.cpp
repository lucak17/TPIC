#include <tpic/initialization/cases/initial_case.hpp>

#include <cmath>

namespace tpic::initialization {

using config::SimulationParameters;

ForceFreeCase::ForceFreeCase(const SimulationParameters& params) {
  require_finite(params.forcefree_b0x, "ForceFreeB0x");
  require_finite(params.forcefree_b0z, "ForceFreeB0z");
  require_recipe(std::isfinite(params.delta) && params.delta > 0.0, "ForceFree delta must be finite and positive");
}

FieldVector ForceFreeCase::magnetic(const SimulationParameters& p, double, double y, double) const {
  FieldVector b{};
  const double argument = (y - 0.5 * p.grid.ly) / p.delta;
  b[0] = p.forcefree_b0x * std::tanh(argument);
  b[2] = p.forcefree_b0z / std::cosh(argument);
  return b;
}

} // namespace tpic::initialization
