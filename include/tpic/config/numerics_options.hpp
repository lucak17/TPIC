#ifndef TPIC_CONFIG_NUMERICS_OPTIONS_HPP
#define TPIC_CONFIG_NUMERICS_OPTIONS_HPP

#include <stdexcept>
#include <string_view>

namespace tpic::config {

enum class ParticleOrder { full, partial, unsorted };

// The order of every command that does not pass --order.
inline constexpr ParticleOrder default_particle_order = ParticleOrder::partial;

// Full: all resident particles grouped by cell after migration.
// Partial: grouped stayed prefix and arbitrary received suffix.
// Unsorted: all particles arbitrary; cell IDs still valid before deposition.
// All three modes share the same particle response and field solve.

[[nodiscard]] constexpr const char* particle_order_name(ParticleOrder order) noexcept {
  switch (order) {
  case ParticleOrder::full: return "full";
  case ParticleOrder::partial: return "partial";
  case ParticleOrder::unsorted: return "unsorted";
  }
  return "invalid";
}

// Host dispatch selects compile-time kernel instantiations; device kernels
// never inspect this enum.
// matrix_core selects CUDA FP64 m8n8k4 PTX or HIP FP64 MFMA4 quadrants.
enum class MassDepositionKernel { scalar, matrix_core };

[[nodiscard]] constexpr const char* mass_deposition_kernel_name(MassDepositionKernel kernel) noexcept {
  switch (kernel) {
  case MassDepositionKernel::scalar: return "scalar";
  case MassDepositionKernel::matrix_core: return "matrix-core";
  }
  return "invalid";
}

[[nodiscard]] constexpr MassDepositionKernel parse_mass_deposition_kernel(std::string_view name) {
  if (name == mass_deposition_kernel_name(MassDepositionKernel::scalar))
    return MassDepositionKernel::scalar;
  if (name == mass_deposition_kernel_name(MassDepositionKernel::matrix_core))
    return MassDepositionKernel::matrix_core;
  throw std::invalid_argument("TPIC options: --mass-matrix must be scalar or matrix-core");
}

// The public request is retained even when smoothing selects its own schedule.
enum class MassApplyMode { full, split };

[[nodiscard]] constexpr bool valid_mass_apply_mode(MassApplyMode mode) noexcept {
  return mode == MassApplyMode::full || mode == MassApplyMode::split;
}

[[nodiscard]] constexpr MassApplyMode parse_mass_apply_mode(std::string_view name) {
  if (name == "full") return MassApplyMode::full;
  if (name == "split") return MassApplyMode::split;
  throw std::invalid_argument("TPIC options: --mass-apply must be full or split");
}

[[nodiscard]] constexpr std::string_view mass_apply_mode_name(MassApplyMode mode) {
  switch (mode) {
  case MassApplyMode::full: return "full";
  case MassApplyMode::split: return "split";
  }
  throw std::invalid_argument("TPIC options: invalid mass apply mode");
}

} // namespace tpic::config

#endif // TPIC_CONFIG_NUMERICS_OPTIONS_HPP
