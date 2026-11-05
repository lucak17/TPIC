#include <tpic/initialization/cases/initial_case.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace tpic::initialization {

using config::SimulationParameters;

GaussianExpansionCase::GaussianExpansionCase(const SimulationParameters& params) {
  const std::array<double, 3> center{params.gaussian_expansion_center_x, params.gaussian_expansion_center_y,
                                     params.gaussian_expansion_center_z};
  const std::array<double, 3> sigma{params.gaussian_expansion_sigma_x, params.gaussian_expansion_sigma_y,
                                    params.gaussian_expansion_sigma_z};
  const std::array<double, 3> lengths{params.grid.lx, params.grid.ly, params.grid.lz};
  const std::array<int, 3> cells{params.grid.nxc, params.grid.nyc, params.grid.nzc};
  constexpr std::array<const char*, 3> center_keys{"GaussianExpansionCenterX", "GaussianExpansionCenterY",
                                                   "GaussianExpansionCenterZ"};
  constexpr std::array<const char*, 3> sigma_keys{"GaussianExpansionSigmaX", "GaussianExpansionSigmaY",
                                                  "GaussianExpansionSigmaZ"};
  const double cutoff = params.gaussian_expansion_cutoff;
  require_finite(cutoff, "GaussianExpansionCutoff");
  require_recipe(cutoff >= 0.0, "GaussianExpansionCutoff must be non-negative");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    require_finite(sigma[axis], sigma_keys[axis]);
    require_recipe(sigma[axis] >= 0.0, "GaussianExpansion widths must be non-negative");
    if (sigma[axis] == 0.0) continue;
    require_finite(center[axis], center_keys[axis]);
    // The domain's spacing expression, so the node is the grid's.
    const double spacing = lengths[axis] / cells[axis];
    const double node = std::round(center[axis] / spacing);
    require_recipe(std::abs(center[axis] / spacing - node) <= 1.0e-9 * std::max(1.0, node) && node >= 0.0 &&
                       node <= cells[axis],
                   "GaussianExpansion centre must be a grid node on every axis with a positive width");
    shape.region.center[axis] = node * spacing;
    shape.region.radius[axis] = checked_product(cutoff, sigma[axis], "GaussianExpansion cutoff radius");
    shape.inverse_two_variance[axis] =
        nonzero_representable(1.0 / (2.0 * sigma[axis] * sigma[axis]), "GaussianExpansion 1/(2*sigma^2)");
  }
  require_recipe(sigma[0] > 0.0 || sigma[1] > 0.0 || sigma[2] > 0.0,
                 "GaussianExpansion requires a positive width on at least one axis");
  require_pair_loading(params);
  for (const auto& species : params.species) {
    require_recipe(species.rho_init > 0.0, "GaussianExpansion rhoINIT must be positive");
    require_recipe(species.u0 == 0.0 && species.v0 == 0.0 && species.w0 == 0.0,
                   "GaussianExpansion drifts must be zero");
  }
}

} // namespace tpic::initialization
