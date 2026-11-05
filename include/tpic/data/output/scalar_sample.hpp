#ifndef TPIC_DATA_OUTPUT_SCALAR_SAMPLE_HPP
#define TPIC_DATA_OUTPUT_SCALAR_SAMPLE_HPP

#include <tpic/config/limits.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace tpic::data {

// Position of each scalar in a species block, in storage and column order.
struct SpeciesScalarIndex {
  static constexpr std::size_t kinetic_energy = 0;
  static constexpr std::size_t charge = 1;
  static constexpr std::size_t momentum_x = 2;
  static constexpr std::size_t momentum_y = 3;
  static constexpr std::size_t momentum_z = 4;
};
inline constexpr std::size_t species_scalar_count = SpeciesScalarIndex::momentum_z + 1;
inline constexpr std::size_t field_scalar_count = 2;
// Scalars of a run: the field scalars, then one block per species.
[[nodiscard]] constexpr std::size_t scalar_count(const std::size_t species) noexcept {
  return field_scalar_count + species_scalar_count * species;
}
inline constexpr std::size_t max_scalar_count =
    scalar_count(static_cast<std::size_t>(config::max_real_species));
// Maximum block partials per observable; runtime launches use up to four
// blocks per multiprocessor. This bound keeps scratch independent of migration.
inline constexpr std::uint32_t contribution_count = 1024;
// Block partials of one reduction lane.
inline constexpr std::size_t scalar_scratch_values = species_scalar_count * contribution_count;

// Globally summed native-state values: electric and magnetic energy, then
// [kinetic energy, charge, momentum x, momentum y, momentum z] per species.
// Energies and momenta use macro-particle mass q/qom; charge is not rescaled.
struct ScalarDiagnosticSample {
  int cycle = 0;
  double time = 0.0;
  bool final_state = false;
  std::size_t species_count = 0;
  std::array<double, max_scalar_count> values{};
};

// The system energy of a sample: electric, magnetic and the species' kinetic
// energies in species order, the sum every report of it uses.
[[nodiscard]] constexpr double total_energy(const ScalarDiagnosticSample& sample) noexcept {
  double kinetic = 0;
  for (std::size_t species = 0; species < sample.species_count; ++species)
    kinetic += sample.values[field_scalar_count + species_scalar_count * species];
  return sample.values[0] + sample.values[1] + kinetic;
}

} // namespace tpic::data

#endif // TPIC_DATA_OUTPUT_SCALAR_SAMPLE_HPP
