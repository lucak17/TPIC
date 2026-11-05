#ifndef TPIC_INITIALIZATION_CASES_INITIAL_CASE_HPP
#define TPIC_INITIALIZATION_CASES_INITIAL_CASE_HPP

#include <tpic/initialization/cases/flux_rope.hpp>
#include <tpic/initialization/cases/force_free.hpp>
#include <tpic/initialization/cases/gaussian_expansion.hpp>
#include <tpic/initialization/cases/harris_sheet.hpp>
#include <tpic/initialization/cases/taylor_green_vortex.hpp>
#include <tpic/initialization/cases/two_stream.hpp>
#include <tpic/initialization/cases/uniform.hpp>
#include <tpic/initialization/cases/weibel.hpp>
#include <tpic/initialization/cases/whistler.hpp>
#include <tpic/initialization/cases/whistler_turbulence.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace tpic::initialization {

// The interface of case_interface.hpp, checked for every case of the list.
template <class Case>
concept SimulationCase =
    std::is_trivially_copyable_v<typename Case::Profile> &&
    requires(const Case& c, const config::SimulationParameters& p, double x, std::size_t species) {
      Case{p};
      { Case::names.front() } -> std::convertible_to<std::string_view>;
      { c.electric(p, x, x, x) } -> std::same_as<FieldVector>;
      { c.magnetic(p, x, x, x) } -> std::same_as<FieldVector>;
      { c.particle_label() } -> std::same_as<std::string_view>;
      { c.profile(species) } noexcept -> std::same_as<typename Case::Profile>;
      { c == c } -> std::same_as<bool>;
    };

template <SimulationCase... Case> using CaseList = std::variant<Case...>;

// The one list of simulation cases. The value is the case selected by the
// deck, holding its coefficients; std::visit reaches its fields and profiles.
using InitialCase = CaseList<UniformCase, HarrisSheetCase, ForceFreeCase, WeibelCase, WhistlerCase,
                             TaylorGreenVortexCase, TwoStreamCase, FluxRopeCase, GaussianExpansionCase,
                             WhistlerTurbulenceCase>;

// Host admission and once-per-startup coefficients of the case named by
// params.case_name. Admission requires finite physical inputs and derived
// coefficients and positive Harris thickness. TGV/flux-rope
// coefficients and perturbed Harris arguments also pass conservative magnitude
// bounds (including one field halo) that reject unsafe double intermediates,
// and the Ampere drift of GEMHarris does not exceed c on any particle row.
// Precondition: params were admitted by config::require_species_count,
// config::prepare_physics, config::prepare_domain and config::prepare_particle_plan
// (prepare_input runs them first); only case-specific admission happens here.
// A case name outside config::case_names throws std::invalid_argument.
[[nodiscard]] InitialCase prepare_initial_case(const config::SimulationParameters& params);

// Analytic startup fields evaluated in double precision on the host from the
// parameters and the case prepared from them, including Whistler roots and
// noise tables. Storage adapters choose the node/cell locations and halos;
// nodal particle B must subsequently be derived from cell B. Coordinates are
// physical, including unwrapped halo coordinates. Magnetic samples include the
// uniform background exactly once. Nonfinite coordinates or sampled components
// throw std::overflow_error.
[[nodiscard]] FieldVector initial_electric(const config::SimulationParameters& parameters,
                                           const InitialCase& initial_case, double x, double y, double z);
[[nodiscard]] FieldVector initial_magnetic(const config::SimulationParameters& parameters,
                                           const InitialCase& initial_case, double x, double y, double z);

// Admission shared by the case sources (defined in initial_case.cpp). Every
// failure names its quantity: require_recipe and require_finite throw
// std::invalid_argument, the representable checks std::overflow_error.
inline constexpr double two_pi = 6.283185307179586476925286766559;
void require_recipe(bool condition, const char* message);
void require_finite(double value, const char* name);
// The value when finite.
double representable(double value, const char* name);
// The value when finite and nonzero.
double nonzero_representable(double value, const char* name);
// a*b, nonzero when both factors are.
double checked_product(double a, double b, const char* name);
// numerator/denominator, nonzero when the numerator is.
double checked_quotient(double numerator, double denominator, const char* name);
// Conservative absolute displacement from center on [-dx, L+dx], covering
// physical particles and the one-cell host field halo.
double coordinate_extent(double length, int cells, double center);
// 2*pi*mode/length; a nonzero mode may not underflow.
double mode_wavenumber(int mode, double length);
// |qom| of the first electron and the first ion species, zero when absent.
std::array<double, 2> first_charge_magnitudes(const config::SimulationParameters& params);
// A phase in [0, 2*pi) fixed by seed, component and index alone.
double seeded_phase(std::uint64_t seed, int component, int index);
// Pair loading on the quiet lattice: QuietStart is set and every species
// shares npcelx, npcely and npcelz, so the species sit on the same subcell
// centres, and the sum of sign(qom)*rhoINIT vanishes.
void require_pair_loading(const config::SimulationParameters& params);

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_INITIAL_CASE_HPP
