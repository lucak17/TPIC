#ifndef TPIC_DATA_OUTPUT_OUTPUT_SNAPSHOT_HPP
#define TPIC_DATA_OUTPUT_OUTPUT_SNAPSHOT_HPP

#include <lpfx/backend/traits.hpp>
#include <tpic/data/output/checkpoint.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace tpic::data {

// A species contribution to a field-snapshot total rounds scale * value to
// double before it is added. CUDA does both in one launch with a __dmul_rn
// product (no contraction) and needs no scratch. HIP packs into a scratch slab
// and adds in a second launch, because a contraction-free product is not
// verified in its ISA. Memory reports size the scratch through
// field_snapshot_scratch_components.
template <class Backend>
inline constexpr bool fused_field_snapshot_accumulation =
    std::is_same_v<Backend, lpfx::device::CudaBackend>;

// Scratch components of a field snapshot whose total-only contributions need
// selected_scratch_components of temporary width on the two-launch path.
template <class Backend>
[[nodiscard]] constexpr std::size_t field_snapshot_scratch_components(
    const std::size_t selected_scratch_components) noexcept {
  return fused_field_snapshot_accumulation<Backend> ? 0 : selected_scratch_components;
}

// An array lives on the nodes, or with cell on the cells (the Gauss residual).
struct OutputArrayDescriptor {
  std::string name;
  int components = 1;
  bool cell = false;
};

struct OwnedFieldArray {
  std::string name;
  int components = 1;
  // The rank's local nodes (or cells) in z,y,x order with x varying fastest;
  // components are interleaved within each sample. Empty: absent from this
  // snapshot. Every backend serializes these same immutable host tuples.
  std::vector<double> values;
  bool cell = false;
};

// The stored values of one particle attribute, in the storage type of its role.
using ParticleValues = std::variant<std::span<const float>, std::span<const double>>;

// One species' committed records [0, id.size()) exactly as the device stores
// them: cell fractions, velocities and signed macroparticle charge in their
// storage types, the rank-local cell key (ParticleCellMap) and the integer id.
// The spans borrow the particle capture's pinned image of the species. Engines
// convert the records to physical values while they serialize them.
struct ParticleRecords {
  std::size_t species = 0;
  std::array<ParticleValues, 3> fraction;
  std::array<ParticleValues, 3> velocity;
  ParticleValues charge;
  std::span<const std::int32_t> cell;
  std::span<const std::uint64_t> id;
};

// Host capture of one output or checkpoint epoch. It owns its field arrays,
// complete before they change owner. Its particle records borrow the particle
// capture's pinned images, which the capture rewrites only in its next
// capture, after every writer has returned the snapshots that borrow them.
// Presence flags let field and particle cadences share one epoch stream: only
// channels marked present are validated or serialized, and an absent channel
// may keep its previous buffers for reuse. A checkpoint snapshot carries only
// the checkpoint channel: its header, its fields and the particle records.
struct OwnedOutputSnapshot {
  int cycle = 0;
  double time = 0.0;
  bool final_state = false;
  std::vector<OwnedFieldArray> arrays;
  bool fields_present = true;
  bool particles_present = false;
  bool checkpoint_present = false;
  std::vector<ParticleRecords> particles;
  CheckpointHeader checkpoint;
  std::array<CheckpointValues, checkpoint_field_names.size()> checkpoint_fields;
};

enum class OutputChannel { fields, particles };

// Immutable view of an OwnedOutputSnapshot; the owner must outlive every
// write that uses the view. Selecting a channel copies no payload.
struct OutputSnapshotView {
  explicit OutputSnapshotView(const OwnedOutputSnapshot& snapshot) noexcept
      : cycle(snapshot.cycle), time(snapshot.time), final_state(snapshot.final_state),
        arrays(snapshot.fields_present ? std::span<const OwnedFieldArray>(snapshot.arrays)
                                       : std::span<const OwnedFieldArray>{}),
        fields_present(snapshot.fields_present), particles_present(snapshot.particles_present),
        particles(snapshot.particles_present || snapshot.checkpoint_present
                      ? std::span<const ParticleRecords>(snapshot.particles)
                      : std::span<const ParticleRecords>{}),
        checkpoint(snapshot.checkpoint_present ? &snapshot.checkpoint : nullptr),
        checkpoint_fields(snapshot.checkpoint_fields) {}

  [[nodiscard]] OutputSnapshotView select(OutputChannel channel) const noexcept {
    auto selected = *this;
    if (channel == OutputChannel::fields) {
      selected.particles_present = false;
      selected.particles = {};
    } else {
      selected.fields_present = false;
      selected.arrays = {};
    }
    return selected;
  }

  int cycle;
  double time;
  bool final_state;
  std::span<const OwnedFieldArray> arrays;
  bool fields_present;
  bool particles_present;
  std::span<const ParticleRecords> particles;
  // The header of a checkpoint snapshot, else null.
  const CheckpointHeader* checkpoint;
  std::span<const CheckpointValues, checkpoint_field_names.size()> checkpoint_fields;
};

} // namespace tpic::data

#endif // TPIC_DATA_OUTPUT_OUTPUT_SNAPSHOT_HPP
