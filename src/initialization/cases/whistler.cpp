#include <tpic/initialization/cases/initial_case.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace tpic::initialization {

using config::SimulationParameters;
using config::WhistlerSeedMode;

namespace {

// A bounded preparation table, not a grid Nyquist restriction. A zero-amplitude
// noise seed resolves its mode count but needs no table and has no such limit.
constexpr std::size_t max_noise_modes = 65536;

// The root is bracketed by 4096 samples and refined by 100 bisection iterations.
// In particular, zero seed amplitude does not bypass eigenmode root admission.
double exact_frequency(double mu, double b0, double k, double c) {
  const auto residual = [=](double omega) {
    const double lhs = (c * c * k * k) / (omega * omega);
    const double rhs = 1.0 - mu / (omega * (omega - mu * b0))
                           - 1.0 / (omega * (omega + b0));
    return lhs - rhs;
  };
  const double lower = representable(b0 * (1.0 + 1.0e-8), "Whistler root lower bound");
  const double upper = representable(mu * b0 * (1.0 - 1.0e-8), "Whistler root upper bound");
  double left = lower;
  double value_left = residual(left);
  for (int sample = 1; sample <= 4096; ++sample) {
    const double right = lower + (static_cast<double>(sample) / 4096.0) * (upper - lower);
    const double value_right = residual(right);
    if (!std::isfinite(value_left) || !std::isfinite(value_right)) {
      left = right;
      value_left = value_right;
      continue;
    }
    if (value_left == 0.0) return left;
    if (value_left * value_right <= 0.0) {
      double bracket_left = left;
      double bracket_right = right;
      for (int iteration = 0; iteration < 100; ++iteration) {
        const double mid = 0.5 * (bracket_left + bracket_right);
        const double value_mid = residual(mid);
        require_recipe(std::isfinite(value_mid), "Whistler root residual became nonfinite");
        if (std::abs(value_mid) < 1.0e-13) return mid;
        if (value_left * value_mid <= 0.0) {
          bracket_right = mid;
        } else {
          bracket_left = mid;
          value_left = value_mid;
        }
      }
      return 0.5 * (bracket_left + bracket_right);
    }
    left = right;
    value_left = value_right;
  }
  throw std::invalid_argument("TPIC recipe: unable to find Whistler exact cold-plasma root");
}

// The wave coefficients; a magnetic or noise seed leaves the eigenmode members zero.
void prepare_wave(const SimulationParameters& params, WhistlerCase& result) {
  result.mode = params.whistler_seed_mode;
  require_recipe(result.mode == WhistlerSeedMode::Eigenmode ||
          result.mode == WhistlerSeedMode::MagneticOnly || result.mode == WhistlerSeedMode::Noise,
          "invalid Whistler seed mode");
  require_finite(params.whistler_seed_epsilon, "WhistlerSeedEpsilon");
  require_finite(params.whistler_phase, "WhistlerPhase");
  require_recipe(params.whistler_seed_epsilon >= 0.0, "WhistlerSeedEpsilon must be non-negative");
  require_recipe(params.b0x > 0.0, "Whistler requires B0x > 0");
  require_recipe(std::abs(params.b0y) <= 1.0e-10 && std::abs(params.b0z) <= 1.0e-10,
          "Whistler requires B0 parallel to x");
  require_recipe(std::abs(mode_wavenumber(params.whistler_mode_y, params.grid.ly)) <= 1.0e-10 &&
          std::abs(mode_wavenumber(params.whistler_mode_z, params.grid.lz)) <= 1.0e-10,
          "only parallel Whistler initialization is supported (ky=kz=0)");
  result.delta_b = representable(params.whistler_seed_epsilon * params.b0x, "Whistler delta B");

  if (result.mode == WhistlerSeedMode::Noise) {
    require_recipe(params.whistler_noise_mode_max >= 0, "WhistlerNoiseModeMax must be non-negative");
    result.noise_mode_max = params.whistler_noise_mode_max > 0 ? params.whistler_noise_mode_max
                         : params.whistler_mode_x > 0 ? params.whistler_mode_x : 16;
    result.wavenumber = mode_wavenumber(1, params.grid.lx);
    result.noise_amplitude = result.delta_b / std::sqrt(static_cast<double>(result.noise_mode_max));
    if (result.delta_b != 0.0) {
      if (static_cast<std::size_t>(result.noise_mode_max) > max_noise_modes)
        throw std::invalid_argument("TPIC recipe: active Whistler noise exceeds the " +
                                    std::to_string(max_noise_modes) + "-mode preparation-table limit");
      representable(result.wavenumber * result.noise_mode_max, "maximum Whistler noise wavenumber");
      result.noise_modes.reserve(static_cast<std::size_t>(result.noise_mode_max));
      for (int mode = 1; mode <= result.noise_mode_max; ++mode) {
        result.noise_modes.push_back({result.wavenumber * mode,
            seeded_phase(params.whistler_noise_seed, 0, mode),
            seeded_phase(params.whistler_noise_seed, 1, mode)});
      }
    }
    return;
  }

  require_recipe(params.whistler_mode_x > 0, "WhistlerModeX must be positive for a single-mode seed");
  result.wavenumber = mode_wavenumber(params.whistler_mode_x, params.grid.lx);
  if (result.mode == WhistlerSeedMode::MagneticOnly) return;

  require_finite(params.whistler_field_time_offset, "WhistlerFieldTimeOffset");
  require_finite(params.whistler_particle_time_offset, "WhistlerParticleTimeOffset");
  const auto qom = first_charge_magnitudes(params);
  require_recipe(qom[0] > 0.0 && qom[1] > 0.0, "Whistler eigenmode requires electron and ion qom");
  const double mass_ratio = representable(qom[0] / qom[1], "Whistler mass ratio");
  require_recipe(mass_ratio > 0.0, "Whistler mass ratio underflows");
  const double ck = representable(params.c * result.wavenumber, "Whistler c*k");
  require_recipe(ck > 0.0, "Whistler c*k underflows");
  if (params.whistler_use_exact_cold_root) {
    result.omega = exact_frequency(mass_ratio, params.b0x, result.wavenumber, params.c);
  } else {
    result.omega = representable(params.b0x * ck * ck / (1.0 + (ck * ck) / mass_ratio),
                               "approximate Whistler frequency");
  }
  require_recipe(std::isfinite(result.omega) && result.omega > 0.0, "Whistler frequency must be finite and positive");
  result.electric_amplitude = representable((result.omega / ck) * result.delta_b, "Whistler E amplitude");
  result.current_amplitude = representable(result.delta_b * (ck - result.omega * result.omega / ck),
                                         "Whistler current amplitude");
  representable(params.whistler_phase - result.omega * params.whistler_field_time_offset,
                "Whistler field phase");
  representable(params.whistler_phase - result.omega * params.whistler_particle_time_offset,
                "Whistler particle phase");
  for (const auto& species : params.species) {
    require_recipe(species.q_over_m > 0.0 || species.rho_init > 0.0,
            "Whistler eigenmode electron rhoINIT must be positive even for zero seed");
    if (species.q_over_m < 0.0) {
      // Loading forms 1/rho separately even when current_amplitude=0.
      // Checking only current_amplitude/rho would miss 0*infinity at zero seed.
      nonzero_representable(1.0 / species.rho_init, "Whistler inverse electron density");
      representable(result.current_amplitude / species.rho_init, "Whistler electron drift");
    }
  }
}

} // namespace

WhistlerCase::WhistlerCase(const SimulationParameters& params) {
  prepare_wave(params, *this);
  for (const auto& loaded : params.species) {
    if (params.whistler_seed_mode == WhistlerSeedMode::Eigenmode && loaded.q_over_m < 0.0)
      species.push_back({true, wavenumber, params.whistler_phase, omega,
                         params.whistler_particle_time_offset, current_amplitude, loaded.rho_init});
    else
      species.push_back({});
  }
}

FieldVector WhistlerCase::electric(const SimulationParameters& p, double x, double, double) const {
  FieldVector e{};
  const auto& w = *this;
  if (w.mode == config::WhistlerSeedMode::Eigenmode && w.delta_b != 0.0) {
    const double phase = (w.wavenumber * x + p.whistler_phase) -
        w.omega * p.whistler_field_time_offset;
    e[1] = -w.electric_amplitude * std::sin(phase);
    e[2] = -w.electric_amplitude * std::cos(phase);
  }
  return e;
}

FieldVector WhistlerCase::magnetic(const SimulationParameters& p, double x, double, double) const {
  FieldVector b{};
  const auto& w = *this;
  if (w.delta_b == 0.0) return b;
  if (w.mode == config::WhistlerSeedMode::Noise) {
    for (const auto& mode : w.noise_modes) {
      b[1] += w.noise_amplitude * std::cos((mode.wavenumber * x + p.whistler_phase) + mode.phase_y);
      b[2] += w.noise_amplitude * std::cos((mode.wavenumber * x + p.whistler_phase) + mode.phase_z);
    }
  } else {
    const double time_phase = w.mode == config::WhistlerSeedMode::Eigenmode
        ? w.omega * p.whistler_field_time_offset : 0.0;
    const double phase = (w.wavenumber * x + p.whistler_phase) - time_phase;
    b[1] = w.delta_b * std::cos(phase);
    b[2] = -w.delta_b * std::sin(phase);
  }
  return b;
}

} // namespace tpic::initialization
