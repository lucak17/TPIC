#include <tpic/initialization/cases/initial_case.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tpic::initialization {

using config::SimulationParameters;

static bool nearly_equal(double a, double b) {
  return std::abs(a - b) <= 1.0e-10 * std::max({1.0, std::abs(a), std::abs(b)});
}

TwoStreamCase::TwoStreamCase(const SimulationParameters& params) {
  require_recipe(params.two_stream_seed_mode_x >= 0, "TwoStreamSeedModeX must be non-negative");
  // Widen before doubling: 2 * mode in int can overflow.
  require_recipe(params.two_stream_seed_mode_x == 0 ||
          2LL * params.two_stream_seed_mode_x < params.grid.nxc,
          "TwoStreamSeedModeX must be below the x-grid Nyquist mode");
  require_finite(params.two_stream_seed_density_amplitude, "TwoStreamSeedDensityAmplitude");
  require_finite(params.two_stream_seed_phase, "TwoStreamSeedPhase");
  require_recipe(std::abs(params.two_stream_seed_density_amplitude) < 1.0,
          "abs(TwoStreamSeedDensityAmplitude) must be less than one");
  require_recipe(params.two_stream_seed_density_amplitude == 0.0 || params.two_stream_seed_mode_x > 0,
          "TwoStreamSeedModeX must be positive for nonzero seed amplitude");
  require_recipe(params.topology.periodic_x && params.topology.periodic_x_particles,
          "TwoStream requires periodic x boundaries for fields and particles");
  std::vector<const config::SpeciesConfig*> electrons;
  const config::SpeciesConfig* ion = nullptr;
  for (const auto& species : params.species) {
    require_recipe(species.rho_init > 0.0, "TwoStream species rhoINIT must be positive");
    if (species.q_over_m < 0.0) electrons.push_back(&species);
    else ion = &species;
  }
  require_recipe(params.ns == 3 && electrons.size() == 2 && ion,
          "TwoStream requires exactly two electron species and one ion species");
  const auto& a = *electrons[0];
  const auto& b = *electrons[1];
  require_recipe(nearly_equal(a.rho_init, b.rho_init), "TwoStream electron rhoINIT must match");
  require_recipe(nearly_equal(a.q_over_m, b.q_over_m), "TwoStream electron qom must match");
  require_recipe(nearly_equal(a.uth, b.uth) && nearly_equal(a.vth, b.vth) && nearly_equal(a.wth, b.wth),
          "TwoStream electron thermal velocities must match");
  require_recipe(a.u0 != 0.0 && nearly_equal(a.u0, -b.u0),
          "TwoStream electron u0 must be equal and opposite and nonzero");
  const double electron_density = representable(a.rho_init + b.rho_init, "TwoStream electron density");
  require_recipe(nearly_equal(electron_density, ion->rho_init),
          "TwoStream ion rhoINIT must equal the sum of electron rhoINIT");
  require_recipe(nearly_equal(ion->u0, 0.0), "TwoStream ion drift must be zero");
  for (const auto& species : params.species) {
    require_recipe(nearly_equal(species.v0, 0.0) && nearly_equal(species.w0, 0.0),
            "TwoStream transverse drifts must be zero");
    require_recipe(species.npcelx == a.npcelx && species.npcely == a.npcely && species.npcelz == a.npcelz,
            "TwoStream requires identical particles-per-cell dimensions");
  }
  wavenumber = mode_wavenumber(params.two_stream_seed_mode_x, params.grid.lx);
  if (params.two_stream_seed_density_amplitude != 0.0) {
    electric_amplitude = representable(
        -(electron_density * params.two_stream_seed_density_amplitude / wavenumber),
        "TwoStream E amplitude");
  }
  for (const auto& loaded : params.species)
    species.push_back({loaded.q_over_m < 0.0 && params.two_stream_seed_density_amplitude != 0.0,
                       wavenumber, params.two_stream_seed_phase,
                       params.two_stream_seed_density_amplitude});
}

FieldVector TwoStreamCase::electric(const SimulationParameters& p, double x, double, double) const {
  FieldVector e{};
  if (p.two_stream_seed_density_amplitude != 0.0) {
    e[0] = electric_amplitude * std::sin(wavenumber * x + p.two_stream_seed_phase);
  }
  return e;
}

} // namespace tpic::initialization
