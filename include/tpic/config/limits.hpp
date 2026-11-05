#ifndef TPIC_CONFIG_LIMITS_HPP
#define TPIC_CONFIG_LIMITS_HPP

#include <cstddef>
#include <stdexcept>
#include <string>

namespace tpic::config {

// Compile-time real-species capacity (CMake option TPIC_MAX_SPECIES) shared by
// prepared input, diagnostics, per-species pipeline state, timing slots and result
// storage. A run uses the ns species of its input; the unused species of the
// capacity own no particle, stream or communication buffer.
inline constexpr int min_real_species = 1;
inline constexpr int max_real_species = TPIC_CONFIG_MAX_SPECIES;
static_assert(max_real_species >= min_real_species, "TPIC_MAX_SPECIES must be positive");

// The one check of the real-species count. The parser calls the range form
// before it allocates species records; prepare_input calls the full form.
inline void require_species_count(int ns) {
  if (ns < min_real_species || ns > max_real_species)
    throw std::invalid_argument("TPIC input: unsupported ns=" + std::to_string(ns) +
                                " (supported real species range: " +
                                std::to_string(min_real_species) + "-" +
                                std::to_string(max_real_species) + ")");
}

inline void require_species_count(int ns, std::size_t species_records) {
  require_species_count(ns);
  if (species_records != static_cast<std::size_t>(ns))
    throw std::invalid_argument("TPIC input: the species array must contain exactly ns records");
}

} // namespace tpic::config

#endif // TPIC_CONFIG_LIMITS_HPP
