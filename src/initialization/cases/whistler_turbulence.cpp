#include <tpic/initialization/cases/initial_case.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tpic::initialization {

using config::SimulationParameters;

namespace {

// A*n^4 - B*n^2 + C of the cold-plasma dispersion relation with every species
// (Stix S, D, P with omega_ps^2 = rhoINIT*|qom| and the signed Omega_s =
// qom*B0z/c), n = c*k/omega, at the propagation angle cos^2 = (kz/k)^2.
double cold_dispersion(const SimulationParameters& p, double omega, double k, double cos2) {
  double stix_s = 1.0, stix_d = 0.0, stix_p = 1.0;
  for (const auto& species : p.species) {
    const double plasma2 = species.rho_init * std::abs(species.q_over_m);
    const double cyclotron = species.q_over_m * p.b0z / p.c;
    const double resonance = omega * omega - cyclotron * cyclotron;
    stix_s -= plasma2 / resonance;
    stix_d += cyclotron * plasma2 / (omega * resonance);
    stix_p -= plasma2 / (omega * omega);
  }
  const double right_left = (stix_s + stix_d) * (stix_s - stix_d), sin2 = 1.0 - cos2;
  const double n2 = (p.c * k / omega) * (p.c * k / omega);
  return ((stix_s * sin2 + stix_p * cos2) * n2 - (right_left * sin2 + stix_p * stix_s * (1.0 + cos2))) * n2 +
      stix_p * right_left;
}

// The magnetosonic-whistler root nearest in log to the electron-MHD estimate:
// 4096 geometric samples of [emhd/8, min(8*emhd, |Omega_e|*cos(theta))]; each
// sign change that brackets no cyclotron resonance takes 100 bisections.
double cold_whistler_frequency(const SimulationParameters& p, double electron_cyclotron, double skin_depth2,
                               double k, double kz) {
  const double cos2 = (kz / k) * (kz / k);
  const double emhd = electron_cyclotron * std::abs(kz) * k * skin_depth2 / (1.0 + k * k * skin_depth2);
  const double lower = emhd / 8.0;
  const double upper = std::min(8.0 * emhd, (1.0 - 1.0e-9) * electron_cyclotron * std::sqrt(cos2));
  const auto residual = [&](double omega) { return cold_dispersion(p, omega, k, cos2); };
  const auto resonant = [&](double left, double right) {
    return std::ranges::any_of(p.species, [&](const auto& species) {
      const double cyclotron = std::abs(species.q_over_m * p.b0z / p.c);
      return left <= cyclotron && cyclotron <= right;
    });
  };
  double best = 0.0;
  double left = lower, value_left = residual(left);
  for (int sample = 1; sample <= 4096; ++sample) {
    const double right = lower * std::pow(upper / lower, sample / 4096.0);
    const double value_right = residual(right);
    if (value_left * value_right <= 0.0 && !resonant(left, right)) {
      double low = left, high = right, value_low = value_left;
      for (int iteration = 0; iteration < 100; ++iteration) {
        const double middle = 0.5 * (low + high);
        const double value = residual(middle);
        if (value_low * value <= 0.0) {
          high = middle;
        } else {
          low = middle;
          value_low = value;
        }
      }
      const double root = 0.5 * (low + high);
      if (best == 0.0 || std::abs(std::log(root / emhd)) < std::abs(std::log(best / emhd))) best = root;
    }
    left = right;
    value_left = value_right;
  }
  require_recipe(best > 0.0, "no cold-plasma whistler root for a WhistlerTurbulence mode");
  return best;
}

} // namespace

WhistlerTurbulenceCase::WhistlerTurbulenceCase(const SimulationParameters& params)
    : flow_scale(params.species.size()) {
  const double epsilon = params.whistler_turbulence_epsilon;
  require_finite(epsilon, "WhistlerTurbulenceEpsilon");
  require_recipe(epsilon >= 0.0, "WhistlerTurbulenceEpsilon must be non-negative");
  require_recipe(params.b0x == 0.0 && params.b0y == 0.0 && params.b0z > 0.0,
                 "WhistlerTurbulence requires B0 along z: B0x = B0y = 0 and B0z > 0");
  const auto& topology = params.topology;
  require_recipe(topology.periodic_x && topology.periodic_y && topology.periodic_z && topology.periodic_x_particles &&
                     topology.periodic_y_particles && topology.periodic_z_particles,
                 "WhistlerTurbulence requires periodic fields and particles");
  require_pair_loading(params);
  // The electrons carry the wave current with one velocity.
  double electron_qom = 0.0, electron_density = 0.0;
  for (const auto& species : params.species) {
    if (species.q_over_m >= 0.0) continue;
    require_recipe(electron_qom == 0.0 || species.q_over_m == electron_qom,
                   "WhistlerTurbulence electron species must share one qom");
    electron_qom = species.q_over_m;
    electron_density += species.rho_init;
  }
  require_recipe(electron_density > 0.0, "WhistlerTurbulence requires electrons with a positive rhoINIT sum");
  const double electron_cyclotron =
      nonzero_representable(-electron_qom * params.b0z / params.c, "WhistlerTurbulence electron cyclotron frequency");
  const double skin_depth2 = nonzero_representable(params.c * params.c / (-electron_qom * electron_density),
                                                   "WhistlerTurbulence electron skin depth squared");
  modes.amplitude = representable(params.b0z * std::sqrt(epsilon / WhistlerTurbulenceModes::count),
                                  "WhistlerTurbulence mode amplitude");
  modes.wavenumber[0] = mode_wavenumber(1, params.grid.lx);
  modes.wavenumber[1] = mode_wavenumber(1, params.grid.ly);
  modes.wavenumber[2] = mode_wavenumber(1, params.grid.lz);
  WhistlerTurbulenceModes::for_each_index([&](int mode, int nx, int ny, int nz) {
    const double kx = nx * modes.wavenumber[0], ky = ny * modes.wavenumber[1], kz = nz * modes.wavenumber[2];
    const double k = std::sqrt(kx * kx + ky * ky + kz * kz);
    modes.phase_speed[mode] =
        cold_whistler_frequency(params, electron_cyclotron, skin_depth2, k, kz) / (params.c * k);
    modes.phase[mode] = seeded_phase(params.whistler_turbulence_seed, 0, mode);
  });
  const double flow = representable(-params.c * modes.amplitude / electron_density, "WhistlerTurbulence electron flow");
  for (std::size_t index = 0; index < flow_scale.size(); ++index)
    flow_scale[index] = params.species[index].q_over_m < 0.0 ? flow : 0.0;
}

// Faraday's law: E = -(omega/(c*|k|)) k x dB/|k| for every mode.
FieldVector WhistlerTurbulenceCase::electric(const SimulationParameters&, double x, double y, double z) const {
  FieldVector e{};
  if (modes.amplitude == 0.0) return e;
  modes.for_each_mode(x, y, z, [&](double, double speed, const double (&)[3], const double (&kb)[3]) {
    e[0] -= speed * kb[0];
    e[1] -= speed * kb[1];
    e[2] -= speed * kb[2];
  });
  for (auto& component : e) component *= modes.amplitude;
  return e;
}

FieldVector WhistlerTurbulenceCase::magnetic(const SimulationParameters&, double x, double y, double z) const {
  FieldVector b{};
  if (modes.amplitude == 0.0) return b;
  modes.for_each_mode(x, y, z, [&](double, double, const double (&unit)[3], const double (&)[3]) {
    b[0] += unit[0];
    b[1] += unit[1];
    b[2] += unit[2];
  });
  for (auto& component : b) component *= modes.amplitude;
  return b;
}

} // namespace tpic::initialization
