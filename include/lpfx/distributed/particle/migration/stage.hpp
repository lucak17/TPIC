#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_STAGE_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_STAGE_HPP

#include <cstdint>

namespace lpfx {

// Execution stages the migration program completes; selected at compile time.
enum class ParticleExecutionStage : std::uint8_t {
  initial_classification,
  initial_pack,
  received_classification,
  received_processing
};

} // namespace lpfx

namespace lpfx::distributed {

// Preparation publishes two independent branches. Only finalization completes
// the migration epoch; routing never grows or appends the main particle owner.
enum class ParticleMigrationStage : std::uint8_t {
  prepare,
  resident,
  route,
  finalize
};

} // namespace lpfx::distributed

#endif
