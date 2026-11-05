#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_RESIDENT_REORDER_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_RESIDENT_REORDER_HPP

#include <cstdint>

namespace lpfx::distributed {

struct ParticleFinalizationRequirements final {
  std::uint32_t resident_count = 0;
  std::uint32_t final_count = 0;
  bool requires_growth = false;
};

struct CompletedParticlePartition final {
  std::uint32_t resident_prefix = 0;
  std::uint32_t total_size = 0;
  std::uint64_t generation = 0;
};

namespace detail {

template <class Semantics>
inline constexpr bool has_external_resident_access = [] {
  if constexpr (requires { Semantics::external_resident_access; })
    return Semantics::external_resident_access;
  else
    return false;
}();

} // namespace detail
} // namespace lpfx::distributed

#endif
