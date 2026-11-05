#include <tpic/initialization/cases/initial_case.hpp>

#include <array>
#include <cmath>
#include <cstddef>

namespace tpic::initialization {

using config::SimulationParameters;

namespace {

void validate_tgv_coefficients(const SimulationParameters& params, const TaylorGreenVortexCase& result) {
  const double magnetic = params.taylor_green_vortex_magnetic_amplitude;
  const double velocity = std::abs(params.taylor_green_vortex_velocity_amplitude);
  const double bx_bound = representable(std::abs(magnetic) + std::abs(params.b0x), "TGV total Bx bound");
  const double by_bound = representable(std::abs(magnetic) + std::abs(params.b0y), "TGV total By bound");
  if (params.taylor_green_vortex_seed_motional_e && velocity != 0.0) {
    checked_product(velocity, std::abs(params.b0z), "TGV motional transverse E bound");
    // The seed-only cross product cancels algebraically, but the field profile
    // computes both products before subtracting. They must remain finite.
    const double ez_first = checked_product(velocity, by_bound, "TGV motional ux*By bound");
    const double ez_second = checked_product(velocity, bx_bound, "TGV motional uy*Bx bound");
    representable(ez_first + ez_second, "TGV motional Ez difference bound");
  }
  if (magnetic == 0.0) return;
  const double current_xy = checked_product(std::abs(magnetic), result.wavenumber[2], "TGV transverse current coefficient");
  const double transverse_k = representable(result.wavenumber[0] + result.wavenumber[1], "TGV kx+ky");
  const double current_z = checked_product(std::abs(magnetic), transverse_k, "TGV axial current coefficient");
  for (std::size_t index = 0; index < params.species.size(); ++index) {
    const auto& species = params.species[index];
    const double split = std::abs(result.species[index].current_split);
    if (split == 0.0) continue;
    const double inverse_density = nonzero_representable(1.0 / species.rho_init, "TGV inverse density");
    const double scale = checked_product(split, inverse_density, "TGV current scale");
    const double transverse_drift = checked_product(scale, current_xy, "TGV transverse current drift");
    const double axial_drift = checked_product(scale, current_z, "TGV axial current drift");
    representable(representable(velocity + transverse_drift, "TGV transverse drift bound") + std::abs(species.u0),
                "TGV x drift plus configured drift bound");
    representable(representable(velocity + transverse_drift, "TGV transverse drift bound") + std::abs(species.v0),
                "TGV y drift plus configured drift bound");
    representable(axial_drift + std::abs(species.w0), "TGV z drift plus configured drift bound");
  }
}

struct VortexFactors {
  double sin_x;
  double cos_x;
  double sin_y;
  double cos_y;
  double cos_z;
};

VortexFactors vortex_factors(const std::array<double, 3>& wavenumber,
                             double x, double y, double z) {
  return {std::sin(wavenumber[0] * x),
          std::cos(wavenumber[0] * x),
          std::sin(wavenumber[1] * y),
          std::cos(wavenumber[1] * y),
          std::cos(wavenumber[2] * z)};
}

} // namespace

TaylorGreenVortexCase::TaylorGreenVortexCase(const SimulationParameters& params)
    : species(params.species.size()) {
  require_finite(params.taylor_green_vortex_velocity_amplitude, "TaylorGreenVortexVelocityAmplitude");
  require_finite(params.taylor_green_vortex_magnetic_amplitude, "TaylorGreenVortexMagneticAmplitude");
  const bool active = params.taylor_green_vortex_velocity_amplitude != 0.0 ||
                      params.taylor_green_vortex_magnetic_amplitude != 0.0;
  const std::array<int, 3> modes{params.taylor_green_vortex_mode_x,
      params.taylor_green_vortex_mode_y, params.taylor_green_vortex_mode_z};
  const std::array<double, 3> lengths{params.grid.lx, params.grid.ly, params.grid.lz};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    require_recipe(modes[axis] >= 0 && (!active || modes[axis] > 0),
            "TaylorGreenVortex modes must be non-negative, and positive for active seeds");
    wavenumber[axis] = mode_wavenumber(modes[axis], lengths[axis]);
  }
  const double kx = wavenumber[0], ky = wavenumber[1];
  require_recipe(!active || std::abs(kx - ky) <= 1.0e-12 * (1.0 + std::abs(kx) + std::abs(ky)),
          "TaylorGreenVortex requires matching kx and ky");
  const auto qom = first_charge_magnitudes(params);
  require_recipe(params.taylor_green_vortex_magnetic_amplitude == 0.0 || (qom[0] > 0.0 && qom[1] > 0.0),
          "TaylorGreenVortex magnetic seed requires electron and ion species");
  double mass_ratio = 0.0;
  if (qom[0] > 0.0 && qom[1] > 0.0) mass_ratio = nonzero_representable(qom[0] / qom[1], "TGV mass ratio");
  for (std::size_t index = 0; index < params.species.size(); ++index) {
    const auto& loaded = params.species[index];
    auto& prepared = species[index];
    require_recipe(loaded.rho_init > 0.0, "TaylorGreenVortex rhoINIT must be positive");
    if (qom[0] > 0.0 && qom[1] > 0.0) {
      const double inverse = 1.0 / (mass_ratio + 1.0);
      prepared.current_split = loaded.q_over_m < 0.0 ? -mass_ratio * inverse : inverse;
      representable(prepared.current_split / loaded.rho_init, "TGV current/density");
    }
    for (std::size_t axis = 0; axis < 3; ++axis) prepared.wavenumber[axis] = wavenumber[axis];
    prepared.current_carrier_density = loaded.rho_init;
    prepared.velocity_amplitude = params.taylor_green_vortex_velocity_amplitude;
    prepared.magnetic_amplitude = params.taylor_green_vortex_magnetic_amplitude;
  }
  validate_tgv_coefficients(params, *this);
}

FieldVector TaylorGreenVortexCase::electric(const SimulationParameters& p, double x, double y,
                                            double z) const {
  FieldVector e{};
  if (p.taylor_green_vortex_seed_motional_e &&
      p.taylor_green_vortex_velocity_amplitude != 0.0) {
    const auto trig = vortex_factors(wavenumber, x, y, z);
    const double ux = p.taylor_green_vortex_velocity_amplitude * trig.sin_x * trig.cos_y * trig.cos_z;
    const double uy = -p.taylor_green_vortex_velocity_amplitude * trig.cos_x * trig.sin_y * trig.cos_z;
    const double uz = 0.0;
    // The motional seed is -u cross B without a 1/c factor,
    // and includes the uniform background as well as the vortex seed.
    auto b = magnetic(p, x, y, z);
    b[0] += p.b0x;
    b[1] += p.b0y;
    b[2] += p.b0z;
    e[0] = -(uy * b[2] - uz * b[1]);
    e[1] = -(uz * b[0] - ux * b[2]);
    e[2] = -(ux * b[1] - uy * b[0]);
  }
  return e;
}

FieldVector TaylorGreenVortexCase::magnetic(const SimulationParameters& p, double x, double y,
                                            double z) const {
  FieldVector b{};
  if (p.taylor_green_vortex_magnetic_amplitude != 0.0) {
    const auto trig = vortex_factors(wavenumber, x, y, z);
    b[0] = p.taylor_green_vortex_magnetic_amplitude * trig.sin_x * trig.cos_y * trig.cos_z;
    b[1] = -p.taylor_green_vortex_magnetic_amplitude * trig.cos_x * trig.sin_y * trig.cos_z;
  }
  return b;
}

} // namespace tpic::initialization
