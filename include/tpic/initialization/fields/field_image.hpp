#ifndef TPIC_INITIALIZATION_FIELDS_FIELD_IMAGE_HPP
#define TPIC_INITIALIZATION_FIELDS_FIELD_IMAGE_HPP

#include <lpfx/api/field/field.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/initialization/cases/initial_case.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tpic::initialization {

// Setup images use LPFx owners and typed, pitched views. Their halos contain
// analytic samples at unwrapped coordinates: ghost positions are not wrapped
// into the domain before sampling.
// These are E at nodes and authoritative B at cells, NOT particle-facing B:
// startup must exchange B-cell halos, apply C2N, then exchange B-node halos.
template <class Precision> struct HostFieldImage {
  using Fields = data::FieldSchema<Precision>;
  lpfx::field::host<typename Fields::E> electric;
  lpfx::field::host<typename Fields::BCell> magnetic_cell;
};

namespace detail {

template <class Value> Value field_storage_value(double value) {
  static_assert(std::is_floating_point_v<Value>);
  // Check before narrowing; an out-of-range floating conversion is not a
  // portable way to detect overflow. Subnormal rounding is allowed.
  if (!std::isfinite(value) || std::abs(value) > config::convert<double>(std::numeric_limits<Value>::max()))
    throw std::overflow_error("TPIC field image: sample exceeds storage precision");
  return config::convert<Value>(value);
}

template <class Schema, class Sampler>
auto sample_field(const config::PreparedDomain& domain, bool nodal, Sampler sample) {
  auto made = lpfx::field::make<Schema>(lpfx::field::options{
      .cells = {domain.local_cells[0], domain.local_cells[1], domain.local_cells[2]},
      .halo = config::PreparedDomain::field_halo});
  lpfx::check_error(made, "TPIC field image: LPFx host allocation failed");
  auto field = std::move(made).value();
  using X = typename Schema::template component<0>;
  using Y = typename Schema::template component<1>;
  using Z = typename Schema::template component<2>;
  const auto fx = field.template get<X>();
  const auto fy = field.template get<Y>();
  const auto fz = field.template get<Z>();
  const double offset = nodal ? 0.0 : 0.5;
  const int extra = nodal ? 1 : 0;
  constexpr int halo = config::PreparedDomain::field_halo;
  for (int k = -halo; k < domain.local_cells[2] + extra + halo; ++k)
    for (int j = -halo; j < domain.local_cells[1] + extra + halo; ++j)
      for (int i = -halo; i < domain.local_cells[0] + extra + halo; ++i) {
        const auto value = sample(domain.origin[0] + (i + offset) * domain.spacing[0],
                                  domain.origin[1] + (j + offset) * domain.spacing[1],
                                  domain.origin[2] + (k + offset) * domain.spacing[2]);
        fx(i, j, k) = field_storage_value<typename X::value_type>(value[0]);
        fy(i, j, k) = field_storage_value<typename Y::value_type>(value[1]);
        fz(i, j, k) = field_storage_value<typename Z::value_type>(value[2]);
      }
  return field;
}

} // namespace detail

// Host field initialization. Precision controls storage only: profiles and
// coordinates are evaluated in double.
// The domain is trusted as prepared by prepare_domain from the parameters;
// only an unprepared domain is rejected.
template <class Precision>
[[nodiscard]] HostFieldImage<Precision> make_host_field_image(
    const config::SimulationParameters& parameters, const InitialCase& initial_case,
    const config::PreparedDomain& domain) {
  if (!domain.field_neighborhood.is_valid() || !domain.particle_topology.is_valid())
    throw std::invalid_argument("TPIC field image: unprepared domain geometry");
  using Fields = data::FieldSchema<Precision>;
  return {
      detail::sample_field<typename Fields::E>(domain, true,
          [&](double x, double y, double z) { return initial_electric(parameters, initial_case, x, y, z); }),
      detail::sample_field<typename Fields::BCell>(domain, false,
          [&](double x, double y, double z) { return initial_magnetic(parameters, initial_case, x, y, z); })};
}

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_FIELDS_FIELD_IMAGE_HPP
