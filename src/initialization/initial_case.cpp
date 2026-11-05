#include <tpic/initialization/cases/initial_case.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tpic::initialization {

using config::SimulationParameters;

void require_recipe(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(std::string("TPIC recipe: ") + message);
}

void require_finite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("TPIC recipe: ") + name + " must be finite");
  }
}

double representable(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::overflow_error(std::string("TPIC recipe: ") + name + " is not representable");
  }
  return value;
}

double nonzero_representable(double value, const char* name) {
  representable(value, name);
  if (value == 0.0) {
    throw std::overflow_error(std::string("TPIC recipe: ") + name + " underflows");
  }
  return value;
}

double checked_product(double a, double b, const char* name) {
  return a != 0.0 && b != 0.0 ? nonzero_representable(a * b, name)
                             : representable(a * b, name);
}

double checked_quotient(double numerator, double denominator, const char* name) {
  return numerator != 0.0 ? nonzero_representable(numerator / denominator, name)
                          : representable(numerator / denominator, name);
}

// Checks every double intermediate operation as well as the final bound; no
// field sampling.
double coordinate_extent(double length, int cells, double center) {
  const double spacing = nonzero_representable(length / cells, "recipe grid spacing");
  const double upper = representable(length + spacing, "recipe upper halo coordinate");
  return std::max(std::abs(representable(-spacing - center, "recipe lower displacement")),
                  std::abs(representable(upper - center, "recipe upper displacement")));
}

double mode_wavenumber(int mode, double length) {
  const double value = representable(two_pi * static_cast<double>(mode) / length, "wavenumber");
  require_recipe(mode == 0 || value != 0.0, "nonzero-mode wavenumber underflows");
  return value;
}

std::array<double, 2> first_charge_magnitudes(const SimulationParameters& params) {
  std::array<double, 2> qom{}; // electron, ion
  for (const auto& species : params.species) {
    if (species.q_over_m < 0.0 && qom[0] == 0.0) qom[0] = -species.q_over_m;
    if (species.q_over_m > 0.0 && qom[1] == 0.0) qom[1] = species.q_over_m;
  }
  return qom;
}

double seeded_phase(std::uint64_t seed, int component, int index) {
  std::uint64_t value = seed ^
      (static_cast<std::uint64_t>(component + 1) * 0xd1b54a32d192ed03ULL) ^
      (static_cast<std::uint64_t>(index) * 0x94d049bb133111ebULL);
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  value ^= value >> 31U;
  return two_pi * (static_cast<double>(value >> 11U) / 9007199254740992.0);
}

void require_pair_loading(const SimulationParameters& params) {
  require_recipe(params.quiet_start, "pair loading requires QuietStart = true: only the subcell centres are shared");
  const auto& first = params.species.front();
  double net = 0.0, total = 0.0;
  for (const auto& species : params.species) {
    require_recipe(species.npcelx == first.npcelx && species.npcely == first.npcely &&
                       species.npcelz == first.npcelz,
                   "pair loading requires the same npcelx, npcely and npcelz for every species");
    net += species.q_over_m > 0.0 ? species.rho_init : -species.rho_init;
    total += species.rho_init;
  }
  require_recipe(std::abs(net) <= 1.0e-12 * total,
                 "pair loading requires a neutral plasma: sign(qom)*rhoINIT must sum to zero");
}

namespace {

template <std::size_t Index> using CaseAt = std::variant_alternative_t<Index, InitialCase>;
constexpr auto case_indices = std::make_index_sequence<std::variant_size_v<InitialCase>>{};

template <std::size_t... Index>
constexpr bool names_listed_once(std::index_sequence<Index...>) {
  if ((CaseAt<Index>::names.size() + ...) != config::case_names.size()) return false;
  for (const auto name : config::case_names)
    if ((std::ranges::count(CaseAt<Index>::names, name) + ...) != 1) return false;
  return true;
}
static_assert(names_listed_once(case_indices),
              "config::case_names holds the names of the cases, each implemented by one case");

void check_coordinates(double x, double y, double z) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    throw std::overflow_error("TPIC field profile: sample coordinates must be finite");
  }
}

FieldVector checked(FieldVector value) {
  for (double component : value) {
    if (!std::isfinite(component)) {
      throw std::overflow_error("TPIC field profile: sampled field is not representable");
    }
  }
  return value;
}

} // namespace

InitialCase prepare_initial_case(const SimulationParameters& params) {
  std::optional<InitialCase> selected;
  const auto select = [&]<std::size_t Index>() {
    if (std::ranges::count(CaseAt<Index>::names, params.case_name) == 0) return false;
    selected.emplace(std::in_place_index<Index>, params);
    return true;
  };
  [&]<std::size_t... Index>(std::index_sequence<Index...>) {
    (select.template operator()<Index>() || ...);
  }(case_indices);
  if (!selected) throw std::invalid_argument("TPIC recipe: unknown Case '" + params.case_name + "'");
  return std::move(*selected);
}

FieldVector initial_magnetic(const SimulationParameters& parameters, const InitialCase& initial_case,
                             double x, double y, double z) {
  check_coordinates(x, y, z);
  auto b = std::visit([&](const auto& selected) { return selected.magnetic(parameters, x, y, z); },
                      initial_case);
  b[0] += parameters.b0x;
  b[1] += parameters.b0y;
  b[2] += parameters.b0z;
  return checked(b);
}

FieldVector initial_electric(const SimulationParameters& parameters, const InitialCase& initial_case,
                             double x, double y, double z) {
  check_coordinates(x, y, z);
  return checked(std::visit(
      [&](const auto& selected) { return selected.electric(parameters, x, y, z); }, initial_case));
}

} // namespace tpic::initialization
