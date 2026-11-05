#ifndef TPIC_SIMULATION_EXECUTION_EXECUTION_LAYOUT_HPP
#define TPIC_SIMULATION_EXECUTION_EXECUTION_LAYOUT_HPP

#include <cstddef>
#include <cstdint>

namespace tpic::simulation {

// Every cardinality of a run. Species counts are admitted by
// config::require_species_count before these formulas are used.
struct ExecutionLayout {
  static constexpr std::size_t moment_quantities = 2;
  static constexpr std::size_t fixed_streams = 11, snapshot_streams = 2;

  // Validation slots outside the species: two for the moment deposition, one
  // each for the field solve, the field advance, the output moments and the
  // Gauss charge.
  static constexpr std::size_t shared_validations = 6;

  // One thread per shared worker (mover, field_advance, routing, deposition,
  // field_solve, diagnostics_output), then one local_species worker per species.
  static constexpr std::size_t shared_workers = 6;
  [[nodiscard]] static constexpr std::size_t workers(std::size_t species) noexcept {
    return species + shared_workers;
  }
  // StageTiming instances: one per worker and one for the main thread.
  [[nodiscard]] static constexpr std::size_t stage_timers(std::size_t species) noexcept {
    return workers(species) + 1;
  }
  [[nodiscard]] static constexpr std::size_t streams(std::size_t species, bool sorted,
                                                     bool snapshot_output) noexcept {
    return species * (sorted ? 2 : 1) + fixed_streams + (snapshot_output ? snapshot_streams : 0);
  }
  [[nodiscard]] static constexpr std::size_t validations(std::size_t species) noexcept {
    return species + shared_validations;
  }
  // Per species the moved and owner events, in partial order also one resident
  // event per moment quantity and the reader closure; new B, committed state,
  // one assembled event per quantity and the mass interior event; the state,
  // particle-field and deposition ready events; two diagnostics releases;
  // two events of an active field snapshot and two of an active particle snapshot.
  [[nodiscard]] static constexpr std::size_t dependency_events(
      std::size_t species, bool partial, bool field_snapshot, bool particle_snapshot) noexcept {
    return species * (2 + partial * (moment_quantities + 1)) +
        2 + moment_quantities + 1 + 3 + 2 + field_snapshot * 2 + particle_snapshot * 2;
  }
  // Host fences of the runtime: nine named, one per moment quantity, one per
  // worker and four per species.
  [[nodiscard]] static constexpr std::size_t host_fences(std::size_t species) noexcept {
    return 9 + moment_quantities + workers(species) + 4 * species;
  }
};

enum class DepositionPhase : unsigned { resident, tail, whole };
enum class MomentQuantity : unsigned { current, mass };
static_assert(static_cast<unsigned>(MomentQuantity::mass) + 1 == ExecutionLayout::moment_quantities,
              "the moment quantities are current and mass");

struct ParticleRange {
  std::uint32_t first{}, count{}, total{};
};

struct DepositionInput {
  std::uint64_t generation{};
  unsigned species{};
  DepositionPhase phase{};
  ParticleRange range{};
};

struct FieldAdvanceProgress {
  bool first_submitted{};
  bool complete{};
};

// One admission. cycle is the state index n it concerns: a solve reads state
// n, an advance writes it, and both share generation() = n + 1. bootstrap and
// moments apply to advances, solve, scalars, the Gauss flags and final_state to
// solves; an advance carries its solve's capture flags for its timing labels.
// A Gauss sample begins at one solve and finishes at the next.
struct Epoch {
  int cycle{};
  bool bootstrap{}, moments{}, solve{}, scalars{}, gauss_begin{}, gauss_finish{}, final_state{}, fields{},
      particles{}, checkpoint{};
  [[nodiscard]] constexpr std::uint64_t generation() const noexcept { return std::uint64_t(cycle) + 1; }
  // The diagnostics lanes read the committed state.
  [[nodiscard]] constexpr bool reads_state() const noexcept { return scalars || gauss_begin || gauss_finish; }
  // An output epoch.
  [[nodiscard]] constexpr bool snapshot() const noexcept { return fields || particles; }
  // The epoch captures the committed state, for output or a checkpoint.
  [[nodiscard]] constexpr bool captures() const noexcept { return snapshot() || checkpoint; }
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_EXECUTION_EXECUTION_LAYOUT_HPP
