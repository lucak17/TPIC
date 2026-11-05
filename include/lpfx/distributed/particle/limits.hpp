#ifndef LPFX_DISTRIBUTED_PARTICLE_LIMITS_HPP
#define LPFX_DISTRIBUTED_PARTICLE_LIMITS_HPP

#include <cstddef>

namespace lpfx {

// One destination per Cartesian neighbor at most. Capacity admission and
// peer packet views share this bound without depending on each other.
inline constexpr std::size_t particle_migration_maximum_peer_count = 26;

} // namespace lpfx

#endif
