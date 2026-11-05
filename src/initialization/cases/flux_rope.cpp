#include <tpic/initialization/cases/initial_case.hpp>

#include <cmath>
#include <cstddef>

namespace tpic::initialization {

using config::SimulationParameters;

namespace {

void validate_flux_rope_coefficients(const SimulationParameters& params, const FluxRopeCase& result) {
  const double width = params.flux_rope_width;
  const double b0 = std::abs(params.flux_rope_b0);
  const double width_squared = checked_product(width, width, "flux rope width squared");
  const double extent_x = coordinate_extent(params.grid.lx, params.grid.nxc, params.flux_rope_center_x);
  const double extent_y = coordinate_extent(params.grid.ly, params.grid.nyc, params.flux_rope_center_y);
  const double radius_squared = representable(
      checked_product(extent_x, extent_x, "flux rope x displacement squared") +
      checked_product(extent_y, extent_y, "flux rope y displacement squared"), "flux rope maximum radius squared");
  const double denominator_max = representable(radius_squared + width_squared, "flux rope maximum denominator");
  const double denominator_squared_min = checked_product(width_squared, width_squared, "flux rope minimum denominator squared");
  checked_product(denominator_max, denominator_max, "flux rope maximum denominator squared");
  const double pressure_denominator_min = checked_product(checked_product(2.0, width_squared, "flux rope doubled minimum denominator"),
                                                   width_squared, "flux rope minimum pressure denominator");
  checked_product(checked_product(2.0, denominator_max, "flux rope doubled maximum denominator"),
          denominator_max, "flux rope maximum pressure denominator");
  checked_product(b0, extent_x, "flux rope By numerator bound");
  checked_product(b0, extent_y, "flux rope Bx numerator bound");
  const double transverse_field = checked_quotient(b0, checked_product(2.0, width, "flux rope doubled width"),
                                           "flux rope transverse magnetic field bound");
  representable(transverse_field + std::abs(params.b0x), "flux rope Bx plus background bound");
  representable(transverse_field + std::abs(params.b0y), "flux rope By plus background bound");
  representable(params.flux_rope_b0 + params.b0z, "flux rope Bz plus background");
  // Check products in the field profile's order: mathematically finite final
  // ratios do not make b0*b0, 2*b0, or denom*denom safe intermediates.
  const double current_numerator = checked_product(checked_product(checked_product(2.0, b0, "flux rope doubled B0"),
      width, "flux rope current numerator times width"), width, "flux rope current numerator");
  const double current_max = checked_quotient(current_numerator, denominator_squared_min, "flux rope axial current bound");
  const double pressure_numerator = checked_product(checked_product(checked_product(b0, b0, "flux rope B0 squared"),
      width, "flux rope pressure numerator times width"), width, "flux rope pressure numerator");
  const double pressure_max = representable(params.flux_rope_pressure_floor +
      checked_quotient(pressure_numerator, pressure_denominator_min, "flux rope magnetic pressure bound"),
      "flux rope total pressure bound");
  for (std::size_t index = 0; index < params.species.size(); ++index) {
    const auto& species = params.species[index];
    const auto& prepared = result.species[index];
    const bool electron = species.q_over_m < 0.0;
    const double fraction = electron ? params.flux_rope_electron_pressure_fraction
                                    : 1.0 - params.flux_rope_electron_pressure_fraction;
    const double temperature = checked_quotient(checked_product(fraction, pressure_max, "flux rope species pressure bound"),
        prepared.species_group_density, "flux rope species temperature bound");
    checked_product(temperature, std::abs(species.q_over_m), "flux rope thermal variance bound");
    if (electron) {
      const double drift = checked_quotient(current_max, prepared.current_carrier_density, "flux rope electron current drift bound");
      representable(drift + std::abs(species.w0), "flux rope electron drift plus configured drift bound");
    }
  }
}

} // namespace

FluxRopeCase::FluxRopeCase(const SimulationParameters& params) {
  require_finite(params.flux_rope_b0, "FluxRopeB0");
  require_finite(params.flux_rope_center_x, "FluxRopeCenterX");
  require_finite(params.flux_rope_center_y, "FluxRopeCenterY");
  require_recipe(std::isfinite(params.flux_rope_width) && params.flux_rope_width > 0.0,
          "FluxRopeWidth must be finite and positive");
  require_recipe(std::isfinite(params.flux_rope_pressure_floor) && params.flux_rope_pressure_floor >= 0.0,
          "FluxRopePressureFloor must be finite and non-negative");
  require_recipe(std::isfinite(params.flux_rope_electron_pressure_fraction) &&
          params.flux_rope_electron_pressure_fraction >= 0.0 && params.flux_rope_electron_pressure_fraction <= 1.0,
          "FluxRopeElectronPressureFraction must be in [0,1]");
  double electron_density = 0.0, ion_density = 0.0;
  for (const auto& loaded : params.species) {
    if (loaded.q_over_m < 0.0) electron_density += loaded.rho_init;
    else ion_density += loaded.rho_init;
  }
  representable(electron_density, "flux rope electron density sum");
  representable(ion_density, "flux rope ion density sum");
  require_recipe(electron_density > 0.0 && ion_density > 0.0,
          "fluxRope requires positive electron and ion group densities");
  for (const auto& loaded : params.species) {
    const bool electron = loaded.q_over_m < 0.0;
    species.push_back({electron, electron_density, electron ? electron_density : ion_density,
        params.flux_rope_b0, params.flux_rope_width, params.flux_rope_center_x,
        params.flux_rope_center_y, params.flux_rope_pressure_floor,
        electron ? params.flux_rope_electron_pressure_fraction
                 : 1.0 - params.flux_rope_electron_pressure_fraction,
        std::abs(loaded.q_over_m)});
  }
  validate_flux_rope_coefficients(params, *this);
}

FieldVector FluxRopeCase::magnetic(const SimulationParameters& p, double x, double y, double) const {
  FieldVector b{};
  const double xc = x - p.flux_rope_center_x;
  const double yc = y - p.flux_rope_center_y;
  const double denominator = xc * xc + yc * yc + p.flux_rope_width * p.flux_rope_width;
  b[0] = -p.flux_rope_b0 * yc / denominator;
  b[1] = p.flux_rope_b0 * xc / denominator;
  b[2] = p.flux_rope_b0;
  return b;
}

} // namespace tpic::initialization
