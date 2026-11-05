#ifndef TPIC_COMMUNICATION_MASS_HALO_MASS_HALO_PLAN_HPP
#define TPIC_COMMUNICATION_MASS_HALO_MASS_HALO_PLAN_HPP

#include <lpfx/distributed/plan/message.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/data/mass_matrix/mass_matrix_layout.hpp>
#include <lpfx/backend/launch.hpp>
#include <tpic/device/block_threads.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace tpic::communication {

// A rank exchanges mass with every face, edge and corner neighbour.
inline constexpr int mass_halo_neighbour_count = 26;

// Block geometry of the mass-halo kernels. A side's block count is capped per
// side; its grid-stride loop covers the rest.
using MassHaloLaunch = lpfx::device::KernelLaunch<device::mass_matrix_block_threads>;

// How far one assembly of a two-phase moment halo has come: interior once the
// additive phase is scattered and the ghost phase issued, so the interior
// records are final in stream order; complete once the ghosts are too.
enum class HaloProgress : std::uint8_t { pending, interior, complete };

// The stencils one phase sends toward the neighbour in direction delta, one
// bit each. Additive: a rank deposits a pair only inside its own cells, so the
// steps that cross the shared plane toward the neighbour stay exact zeros in
// its shared records. Ghost: the neighbour's apply reads a ghost record only
// through steps equal to delta on every nonzero axis of delta.
[[nodiscard]] constexpr std::uint16_t mass_halo_stencil_mask(
    const bool additive, const lpfx::CartesianDirection3 delta) noexcept {
  std::uint16_t mask = 0;
  for (int stencil = 0; stencil != data::mass_stencil_count; ++stencil) {
    const auto offset = data::mass_stencil_offset(stencil);
    const int step[]{offset.i, offset.j, offset.k}, toward[]{delta.x, delta.y, delta.z};
    bool kept = true;
    for (int axis = 0; axis != 3; ++axis)
      if (toward[axis] != 0)
        kept = kept && (additive ? step[axis] != toward[axis] : step[axis] == toward[axis]);
    if (kept) mask = static_cast<std::uint16_t>(mask | 1U << stencil);
  }
  return mask;
}

// One direction's message in one phase, packed from or scattered into the
// region that begins at start. Its entries run over the region's nodes
// x-fastest, then the stencils of stencil_mask in ascending order (listed in
// stencils), then the nine tensor entries. offset counts scalars in the
// incoming image for a remote receive, otherwise in the outgoing image;
// first_block and block_count place the side in its kernel grid:
// MassHaloLaunch::blocks(entries), starting at the exclusive prefix sum of the
// preceding ordinals.
struct MassHaloSide final {
  int start[3]{};
  std::size_t offset{};
  std::uint32_t entries{};
  std::uint32_t first_block{};
  std::uint32_t block_count{};
  std::uint16_t stencil_mask{};
  std::uint8_t stencils[data::mass_stencil_count]{};
};

// Kernel-facing numerical record of one direction in one phase: send goes to
// the neighbour; receive is the neighbour's opposite send, or for a self
// direction the opposite ordinal's send snapshot. Peers and message keys
// remain cold host data.
struct MassHaloDirection final {
  int lengths[3]{};
  MassHaloSide send{};
  MassHaloSide receive{};
  std::uint32_t canonical_index{};
  bool self{};
};
static_assert(std::is_trivially_copyable_v<MassHaloDirection> &&
              std::is_standard_layout_v<MassHaloDirection>);

template <class Precision> class MassHaloPlan;

// The directions and message plan of one phase: Additive sums the shared
// records, otherwise the ghost phase copies the ghost records.
template <class Precision, bool Additive>
class MassHaloPhasePlan final {
  using Scalar = typename Precision::Storage::Mass;
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);

public:
  using rank_type = config::PreparedDomain::Rank;
  // One numeric channel per phase and one versioned wire identity per phase
  // and scalar width: VMASF002, VMASD002 (additive), VMAGF001, VMAGD001 (ghost).
  using definition_type = lpfx::distributed::MessagePlan<
      Additive ? 0x56450011 : 0x56450012,
      lpfx::distributed::FixedByteMessages<
          Additive ? (std::is_same_v<Scalar, float> ? UINT64_C(0x564d415346303032)
                                                    : UINT64_C(0x564d415344303032))
                   : (std::is_same_v<Scalar, float> ? UINT64_C(0x564d414746303031)
                                                    : UINT64_C(0x564d414744303031)),
          mass_halo_neighbour_count>>;
  using message_plan_type =
      lpfx::distributed::PreparedMessagePlan<definition_type, rank_type>;

  [[nodiscard]] constexpr const message_plan_type &message_plan() const noexcept {
    return message_plan_;
  }
  [[nodiscard]] constexpr std::span<const MassHaloDirection> directions() const noexcept {
    return {directions_.data(), direction_count_};
  }
  // Grids of the pack and of the scatter kernels: the sums of the send and of
  // the receive block counts.
  [[nodiscard]] constexpr std::uint32_t pack_blocks() const noexcept { return pack_blocks_; }
  [[nodiscard]] constexpr std::uint32_t unpack_blocks() const noexcept { return unpack_blocks_; }
  // The remote messages and the self snapshots after them.
  [[nodiscard]] constexpr std::size_t outgoing_storage_bytes() const noexcept {
    return outgoing_storage_bytes_;
  }

private:
  friend class MassHaloPlan<Precision>;
  [[nodiscard]] static MassHaloPhasePlan prepare(const config::PreparedDomain &domain,
                                                 const data::MassMatrixLayout &layout);

  std::array<MassHaloDirection, mass_halo_neighbour_count> directions_{};
  message_plan_type message_plan_{};
  std::size_t direction_count_{};
  std::size_t outgoing_storage_bytes_{};
  std::uint32_t pack_blocks_{};
  std::uint32_t unpack_blocks_{};
};

// Both phases over one layout. They run one after the other, so they share
// one outgoing and one incoming image sized for the larger phase.
template <class Precision>
class MassHaloPlan final {
public:
  template <bool Additive> using phase_type = MassHaloPhasePlan<Precision, Additive>;

  // Host, once per run. Throws std::invalid_argument for a domain without a
  // canonical partition or a reciprocal plan, and std::overflow_error for an
  // unrepresentable extent.
  [[nodiscard]] static MassHaloPlan prepare(const config::PreparedDomain &domain) {
    config::require_canonical_partition(domain, "TPIC mass halo");
    MassHaloPlan output{};
    output.layout_ = data::prepare_mass_matrix_layout(domain);
    output.add_ = phase_type<true>::prepare(domain, output.layout_);
    output.ghost_ = phase_type<false>::prepare(domain, output.layout_);
    return output;
  }

  template <bool Additive>
  [[nodiscard]] constexpr const phase_type<Additive> &phase() const noexcept {
    if constexpr (Additive) return add_;
    else return ghost_;
  }
  [[nodiscard]] constexpr const data::MassMatrixLayout &layout() const noexcept {
    return layout_;
  }
  [[nodiscard]] constexpr std::size_t outgoing_bytes() const noexcept {
    return std::max(add_.outgoing_storage_bytes(), ghost_.outgoing_storage_bytes());
  }
  [[nodiscard]] constexpr std::size_t incoming_bytes() const noexcept {
    return std::max(add_.message_plan().incoming_bytes(), ghost_.message_plan().incoming_bytes());
  }
  // The device tables of both phases' directions.
  [[nodiscard]] constexpr std::size_t direction_bytes() const noexcept {
    return (add_.directions().size() + ghost_.directions().size()) * sizeof(MassHaloDirection);
  }
  [[nodiscard]] constexpr std::size_t device_bytes() const noexcept {
    return outgoing_bytes() + incoming_bytes() + direction_bytes();
  }

private:
  data::MassMatrixLayout layout_{};
  phase_type<true> add_{};
  phase_type<false> ghost_{};
};

template <class Precision, bool Additive>
MassHaloPhasePlan<Precision, Additive> MassHaloPhasePlan<Precision, Additive>::prepare(
    const config::PreparedDomain &domain, const data::MassMatrixLayout &layout) {
  const auto &topology = domain.field_neighborhood.topology();
  const auto &input = topology.input();

  MassHaloPhasePlan output{};
  const int nodes[]{layout.nx, layout.ny, layout.nz};
  std::array<rank_type, mass_halo_neighbour_count> peers{};
  std::array<std::uint32_t, mass_halo_neighbour_count> canonical_to_ordinal{};
  canonical_to_ordinal.fill(mass_halo_neighbour_count);
  std::array<std::uint32_t, mass_halo_neighbour_count> remote_ordinals{};
  std::size_t remote_count = 0;

  // A side's kept stencils and entries, and its blocks after the preceding
  // sides of its grid. At most mass_halo_neighbour_count sides of at most
  // lpfx::device::kernel_grid_block_limit blocks: each grid fits uint32.
  const auto size_side = [](MassHaloSide &side, const std::size_t region_nodes,
                            const std::uint16_t mask, std::uint32_t &blocks) {
    std::size_t kept = 0;
    for (int stencil = 0; stencil != data::mass_stencil_count; ++stencil)
      if (mask >> stencil & 1U) side.stencils[kept++] = static_cast<std::uint8_t>(stencil);
    const auto record = kept * data::mass_tensor_entries;
    if (record != 0 && region_nodes > std::numeric_limits<std::uint32_t>::max() / record)
      throw std::overflow_error("TPIC mass halo: direction scalar extent is not representable");
    side.stencil_mask = mask;
    side.entries = static_cast<std::uint32_t>(region_nodes * record);
    side.block_count = MassHaloLaunch::blocks(side.entries);
    side.first_block = blocks;
    blocks += side.block_count;
  };

  // This is the numerical order and remains independent of peer packing.
  for (std::uint32_t canonical = 0; canonical != mass_halo_neighbour_count; ++canonical) {
    const auto &neighbor = topology.neighbor_by_index(canonical);
    if (!neighbor.has_peer()) continue;
    const auto vector = lpfx::canonical_cartesian_neighbor_directions[canonical];
    const int delta[]{vector.x, vector.y, vector.z};
    MassHaloDirection direction{};
    std::size_t region_nodes = 1;
    for (unsigned axis = 0; axis != 3; ++axis) {
      const int n = nodes[axis];
      direction.lengths[axis] = delta[axis] == 0 ? n : 1;
      region_nodes *= static_cast<std::size_t>(direction.lengths[axis]);
      // Additive: the shared plane both ways. Ghost: the plane inside it is
      // sent, the ghost plane beyond it is received.
      if constexpr (Additive) {
        direction.send.start[axis] = direction.receive.start[axis] = delta[axis] > 0 ? n - 1 : 0;
      } else {
        direction.send.start[axis] = delta[axis] > 0 ? n - 2 : (delta[axis] < 0 ? 1 : 0);
        direction.receive.start[axis] = delta[axis] > 0 ? n : (delta[axis] < 0 ? -1 : 0);
      }
    }
    size_side(direction.send, region_nodes, mass_halo_stencil_mask(Additive, vector), output.pack_blocks_);
    size_side(direction.receive, region_nodes,
              mass_halo_stencil_mask(Additive, lpfx::opposite_cartesian_direction(vector)),
              output.unpack_blocks_);
    direction.canonical_index = canonical;
    direction.self = neighbor.is_self_peer();
    const auto ordinal = static_cast<std::uint32_t>(output.direction_count_++);
    canonical_to_ordinal[canonical] = ordinal;
    peers[ordinal] = neighbor.peer_rank();
    output.directions_[ordinal] = direction;
    if (!direction.self) remote_ordinals[remote_count++] = ordinal;
  }

  const auto opposite_group = [&](const std::uint32_t ordinal) noexcept {
    return static_cast<std::uint32_t>(
        lpfx::opposite_cartesian_neighbor_index(output.directions_[ordinal].canonical_index));
  };
  // Lays out one remote image in peer order, outgoing by the receiver's
  // direction and incoming by this rank's, and returns its entries.
  const auto lay_out = [&](MassHaloSide MassHaloDirection::*const side, const bool outgoing) {
    const auto group = [&](const std::uint32_t ordinal) {
      return outgoing ? opposite_group(ordinal) : output.directions_[ordinal].canonical_index;
    };
    auto order = remote_ordinals;
    std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(remote_count),
              [&](const std::uint32_t left, const std::uint32_t right) noexcept {
                if (peers[left] != peers[right]) return peers[left] < peers[right];
                return group(left) < group(right);
              });
    std::size_t entries = 0;
    for (std::size_t index = 0; index != remote_count; ++index) {
      auto &laid = output.directions_[order[index]].*side;
      laid.offset = entries;
      entries += laid.entries;
    }
    return entries;
  };
  const auto outgoing_entries = lay_out(&MassHaloDirection::send, true);
  const auto incoming_entries = lay_out(&MassHaloDirection::receive, false);

  // Self snapshots occupy a send-only suffix and never enter the LPFx plan;
  // the opposite self direction receives each one.
  std::size_t outgoing_storage_entries = outgoing_entries;
  for (std::size_t ordinal = 0; ordinal != output.direction_count_; ++ordinal) {
    auto &direction = output.directions_[ordinal];
    if (!direction.self) continue;
    const auto opposite = canonical_to_ordinal[opposite_group(static_cast<std::uint32_t>(ordinal))];
    if (opposite >= output.direction_count_ || !output.directions_[opposite].self)
      throw std::invalid_argument("TPIC mass halo: missing opposite self direction");
    output.directions_[opposite].receive.offset = direction.send.offset = outgoing_storage_entries;
    outgoing_storage_entries += direction.send.entries;
  }

  output.outgoing_storage_bytes_ = outgoing_storage_entries * sizeof(Scalar);

  using Fragment = lpfx::distributed::FixedMessageFragment<rank_type>;
  std::array<Fragment, mass_halo_neighbour_count> fragments{};
  for (std::size_t index = 0; index != remote_count; ++index) {
    const auto ordinal = remote_ordinals[index];
    const auto &direction = output.directions_[ordinal];
    fragments[index] = {
        peers[ordinal], direction.canonical_index, opposite_group(ordinal),
        {direction.send.offset * sizeof(Scalar), direction.send.entries * sizeof(Scalar)},
        {direction.receive.offset * sizeof(Scalar), direction.receive.entries * sizeof(Scalar)}};
  }
  const auto plan_error = lpfx::distributed::try_prepare_message_plan(
      definition_type{}, input.rank(), input.communicator_size(),
      outgoing_entries * sizeof(Scalar), incoming_entries * sizeof(Scalar),
      std::span<const Fragment>{fragments.data(), remote_count},
      output.message_plan_);
  if (plan_error == lpfx::ErrorCode::overflow)
    throw std::overflow_error("TPIC mass halo: message plan extent overflow");
  if (plan_error != lpfx::ErrorCode::success)
    throw std::invalid_argument("TPIC mass halo: invalid reciprocal message plan");
  return output;
}

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_MASS_HALO_MASS_HALO_PLAN_HPP
