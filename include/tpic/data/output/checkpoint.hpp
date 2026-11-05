#ifndef TPIC_DATA_OUTPUT_CHECKPOINT_HPP
#define TPIC_DATA_OUTPUT_CHECKPOINT_HPP

#include <tpic/config/limits.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace tpic::data {

// What a checkpoint holds besides its fields and particles, the same in every
// piece. Trivially copyable: the root broadcasts the header it read.
struct CheckpointHeader {
  std::int32_t cycle{}, species{};
  // The writer's process grid and the mesh, from which every piece's cell box follows.
  std::array<std::int32_t, 3> processes{}, global_cells{};
  std::array<double, 3> lengths{};
  double dt{}, c{};
  // The total energy of the first state of the original run: the baseline of dE/E0.
  double initial_energy{};
  // config::FieldBoundary of the faces Xright..Zleft, then config::ParticleBoundary.
  std::array<std::int32_t, 12> faces{};
  // Bytes of the stored position, velocity, weight, electric and magnetic values.
  std::array<std::int32_t, 5> storage_bytes{};
  std::array<double, config::max_real_species> q_over_m{};
  std::array<std::uint64_t, config::max_real_species> global_count{};
};
static_assert(std::is_trivially_copyable_v<CheckpointHeader>);

// The stored values of one field component of a checkpoint, z,y,x order with
// x fastest: the first checkpoint_node_fields are E at the local nodes, both
// ends of every axis, the others B at the local cells.
using CheckpointValues = std::variant<std::vector<float>, std::vector<double>>;
inline constexpr std::array<const char*, 6> checkpoint_field_names{"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
inline constexpr std::size_t checkpoint_node_fields = 3;

// A checkpoint directory found for a restart: its pieces are
// <directory>/checkpoint_t<cycle>_r<rank><extension>.
struct CheckpointLocation {
  std::string directory, extension;
  CheckpointHeader header;
};

// A piece of the checkpoint whose cells overlap this rank's: the writer rank
// and its cell box in global cells.
struct CheckpointSource {
  int rank{};
  std::array<int, 3> offset{}, cells{};
};

// The restart of one rank: the checkpoint, its pieces that hold this rank's
// cells in ascending rank, and the records of each species in them.
struct PreparedRestart {
  CheckpointLocation checkpoint;
  std::vector<CheckpointSource> sources;
  std::array<std::uint64_t, config::max_real_species> local_count{};
};

} // namespace tpic::data

#endif // TPIC_DATA_OUTPUT_CHECKPOINT_HPP
