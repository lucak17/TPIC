#ifndef TPIC_COMMUNICATION_MASS_HALO_MASS_HALO_DEVICE_HPP
#define TPIC_COMMUNICATION_MASS_HALO_MASS_HALO_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/distributed/message/buffers.hpp>
#include <lpfx/distributed/transport/mpi/setup.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/memory/space.hpp>
#include <tpic/device/block_threads.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/communication/mass_halo/mass_halo_plan.hpp>
#include <tpic/data/mass_matrix/compact_mass_matrix_device.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tpic::communication {
namespace detail {
namespace sg = lpfx::device::subgroup;

// One entry of a side: its position in the side, its matrix index, and the
// vertex, stencil and tensor entry of its record.
struct MassHaloEntry final {
  std::uint32_t position;
  std::size_t index;
  device::GridCoordinates vertex;
  int stencil;
  std::uint32_t component;
};

// Grid-stride loop over the entries of one side from the block's place in the
// side's grid. The break before the increment keeps the uint32 entry counter
// from wrapping when entries lies within one stride of UINT32_MAX.
template <class Visit>
__device__ LPFX_ALWAYS_INLINE void for_each_mass_halo_entry(
    const data::MassMatrixLayout layout, const MassHaloDirection &direction,
    const MassHaloSide &side, Visit visit) {
  constexpr auto tensor = static_cast<std::uint32_t>(data::mass_tensor_entries);
  const auto record = sg::warp_population_count(side.stencil_mask) * tensor;
  const auto stride = static_cast<std::uint32_t>(blockDim.x * side.block_count);
  for (std::uint32_t entry = (blockIdx.x - side.first_block) * blockDim.x + threadIdx.x;
       entry < side.entries;) {
    const auto within = entry % record;
    const auto at = device::decode_x_fastest(entry / record,
        static_cast<std::uint32_t>(direction.lengths[0]), static_cast<std::uint32_t>(direction.lengths[1]));
    const device::GridCoordinates vertex{side.start[0] + at.i, side.start[1] + at.j, side.start[2] + at.k};
    const int stencil = side.stencils[within / tensor];
    visit(MassHaloEntry{entry, layout.entry_index(layout.vertex_index(vertex.i, vertex.j, vertex.k), stencil) +
                                   within % tensor,
                        vertex, stencil, within % tensor});
    if (side.entries - entry <= stride) break;
    entry += stride;
  }
}

// The one-dimensional grids of the pack and scatter kernels concatenate the
// blocks of every direction's send or receive side in ordinal order. Each block
// finds its direction, the last ordinal whose side starts at or below
// blockIdx.x (first_block ascends from zero; an empty side starts where the
// next side does and is never chosen), with one first-block load per lane and
// a ballot, so the lookup costs one load round trip whatever the ordinal.
// Every warp enters converged.
template <MassHaloSide MassHaloDirection::*Side>
[[nodiscard]] __device__ LPFX_ALWAYS_INLINE const MassHaloDirection &
block_mass_halo_direction(const MassHaloDirection *const directions,
                          const std::uint32_t direction_count) noexcept {
  static_assert(mass_halo_neighbour_count <= 32, "one lane per direction in a 32-wide warp");
  const std::uint32_t lane = sg::warp_lane();
  const bool started = lane < direction_count && (directions[lane].*Side).first_block <= blockIdx.x;
  return directions[sg::warp_population_count(sg::warp_ballot(started)) - 1];
}

} // namespace detail

namespace kernels {

// Packs the send side of every direction into outgoing. Grid: the phase's
// pack_blocks() blocks of MassHaloLaunch. No validation flag.
template <class Scalar>
__global__ __launch_bounds__(device::mass_matrix_block_threads) void mass_halo_pack_kernel(
    const Scalar *const mass, Scalar *const outgoing,
    const data::MassMatrixLayout layout,
    const communication::MassHaloDirection *const directions,
    const std::uint32_t direction_count) {
  const auto &direction =
      detail::block_mass_halo_direction<&MassHaloDirection::send>(directions, direction_count);
  detail::for_each_mass_halo_entry(layout, direction, direction.send, [&](const detail::MassHaloEntry &at) {
    outgoing[direction.send.offset + at.position] = mass[at.index];
  });
}

// Adds the receive side of every direction into its region. Grid: the phase's
// unpack_blocks(). An entry belongs to the lowest ordinal whose region holds
// its vertex and whose side carries its stencil: that thread adds the entry of
// every such direction and stores once. A side without the stencil would add
// the sender's exact zero, which changes no sum. No validation flag.
template <class Scalar>
__global__ __launch_bounds__(device::mass_matrix_block_threads) void mass_halo_unpack_add_kernel(
    Scalar *const mass, const Scalar *const outgoing,
    const Scalar *const incoming, const data::MassMatrixLayout layout,
    const communication::MassHaloDirection *const directions,
    const std::uint32_t direction_count) {
  const auto &direction =
      detail::block_mass_halo_direction<&MassHaloDirection::receive>(directions, direction_count);
  const auto ordinal = static_cast<std::uint32_t>(&direction - directions);
  constexpr auto tensor = static_cast<std::uint32_t>(data::mass_tensor_entries);
  detail::for_each_mass_halo_entry(layout, direction, direction.receive,
      [&](const detail::MassHaloEntry &at) {
        Scalar sum = mass[at.index];
        // R1: ascending canonical order fixes the sum, since boundary
        // destinations overlap and floating-point addition does not associate.
        for (std::uint32_t other = 0; other != direction_count; ++other) {
          const auto &region = directions[other];
          const auto &side = region.receive;
          const int x = at.vertex.i - side.start[0], y = at.vertex.j - side.start[1],
                    z = at.vertex.k - side.start[2];
          if (!(side.stencil_mask >> at.stencil & 1U) || x < 0 || y < 0 || z < 0 ||
              x >= region.lengths[0] || y >= region.lengths[1] || z >= region.lengths[2])
            continue;
          if (other < ordinal) return;
          const auto node = static_cast<std::uint32_t>(x + region.lengths[0] * (y + region.lengths[1] * z));
          const auto kept = detail::sg::warp_population_count(side.stencil_mask);
          const auto below = detail::sg::warp_population_count(side.stencil_mask & ((1U << at.stencil) - 1U));
          sum += (region.self ? outgoing : incoming)[side.offset + (node * kept + below) * tensor + at.component];
        }
        mass[at.index] = sum;
      });
}

// Overwrites the receive side of every direction's ghost region. Grid: the
// phase's unpack_blocks(). No validation flag.
template <class Scalar>
__global__ __launch_bounds__(device::mass_matrix_block_threads) void mass_halo_unpack_copy_kernel(
    Scalar *const mass, const Scalar *const outgoing,
    const Scalar *const incoming, const data::MassMatrixLayout layout,
    const communication::MassHaloDirection *const directions,
    const std::uint32_t direction_count) {
  const auto &direction =
      detail::block_mass_halo_direction<&MassHaloDirection::receive>(directions, direction_count);
  const auto *const source = (direction.self ? outgoing : incoming) + direction.receive.offset;
  detail::for_each_mass_halo_entry(layout, direction, direction.receive, [&](const detail::MassHaloEntry &at) {
    mass[at.index] = source[at.position];
  });
}

} // namespace kernels

// Address-stable cold state for both mass phases. It owns no native session:
// its plan and buffers are registered in the application's one Communication.
template <class Precision, class Backend>
class MassHaloStorage final {
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Allocation = lpfx::device::DeviceAllocation<Backend>;
  using PlanStaging =
      lpfx::device::AlignedPinnedHostAllocation<Backend, alignof(MassHaloDirection)>;

  static_assert(Traits::minimum_device_allocation_alignment >=
                alignof(typename Precision::Storage::Mass));

public:
  using executor_type = lpfx::device::Executor<Backend>;
  using plan_type = MassHaloPlan<Precision>;
  using buffers_type =
      lpfx::distributed::FixedMessageBuffers<lpfx::DeviceSpace<Backend>>;

  MassHaloStorage(const config::PreparedDomain &domain,
                  data::CompactMassMatrix<Precision, Backend> &mass)
      : plan_(plan_type::prepare(domain)), mass_(mass.data()) {
    if (!mass.initialized() || plan_.layout() != mass.layout())
      throw std::invalid_argument(
          "TPIC mass halo: matrix does not match the prepared domain");
  }
  MassHaloStorage(const MassHaloStorage &) = delete;
  MassHaloStorage &operator=(const MassHaloStorage &) = delete;
  ~MassHaloStorage() noexcept {
    if (initialized_) std::abort();
  }

  [[nodiscard]] bool quiescent() const noexcept {
    return initialized_ && phase_ == Phase::ready;
  }
  [[nodiscard]] const plan_type &plan() const noexcept { return plan_; }
  // One phase's images: prefixes of the two shared allocations.
  template <bool Additive> [[nodiscard]] buffers_type &buffers() noexcept {
    return buffers_[Additive ? 0 : 1];
  }
  [[nodiscard]] std::size_t storage_bytes() const noexcept {
    return outgoing_.capacity_bytes() + incoming_.capacity_bytes() +
           device_directions_.capacity_bytes();
  }

  void prepare(const lpfx::distributed::CommunicationSetup &setup,
               const executor_type executor) {
    lpfx::require(!initialized_ && phase_ == Phase::ready, "TPIC mass halo: storage prepared twice");
    lpfx::require(setup.is_valid() &&
                      setup.template try_verify_device_affinity<Backend>(executor) ==
                          lpfx::ErrorCode::success &&
                      setup.template try_verify_device_pointer_affinity<Backend>(mass_) ==
                          lpfx::ErrorCode::success,
                  "TPIC mass halo: storage requires its selected device and executor");
    const auto &message_plan = plan_.template phase<true>().message_plan();
    lpfx::require(std::cmp_equal(message_plan.rank(), setup.rank()) &&
                      std::cmp_equal(message_plan.rank_count(), setup.size()),
                  "TPIC mass halo: plan and communication setup disagree");

    executor_ = executor;
    device_identity_ = setup.device_identity();
    device_ordinal_ = setup.device_ordinal();
    lpfx::device::check_device_error<Backend>(
        Allocation::create(plan_.outgoing_bytes(), outgoing_),
        "TPIC mass halo: outgoing allocation failed");
    lpfx::device::check_device_error<Backend>(
        Allocation::create(plan_.incoming_bytes(), incoming_),
        "TPIC mass halo: incoming allocation failed");

    const auto direction_bytes = plan_.direction_bytes();
    lpfx::device::check_device_error<Backend>(
        Allocation::create(direction_bytes, device_directions_),
        "TPIC mass halo: direction allocation failed");
    if (direction_bytes != 0) {
      PlanStaging staging{};
      lpfx::check_error(PlanStaging::create(direction_bytes, staging),
                        "TPIC mass halo: direction pinned staging allocation failed");
      const auto add = plan_.template phase<true>().directions();
      const auto ghost = plan_.template phase<false>().directions();
      std::memcpy(staging.raw_base(), add.data(), add.size_bytes());
      std::memcpy(static_cast<std::byte *>(staging.raw_base()) + add.size_bytes(), ghost.data(),
                  ghost.size_bytes());
      const auto copied = Traits::copy_host_to_device_async(
          device_directions_.raw_base(), staging.raw_base(), direction_bytes,
          executor.stream());
      const auto completed = Traits::synchronize_stream(executor.stream());
      if (!Traits::succeeded(completed)) std::abort();
      lpfx::device::check_device_error<Backend>(copied, "TPIC mass halo: direction upload failed");
    }

    const auto images = [&](const auto &phase) {
      return buffers_type{outgoing_.raw_base(), phase.message_plan().outgoing_bytes(),
                          incoming_.raw_base(), phase.message_plan().incoming_bytes()};
    };
    buffers_ = {images(plan_.template phase<true>()), images(plan_.template phase<false>())};
    initialized_ = true;
  }

  void close() {
    lpfx::require(quiescent(), "TPIC mass halo: storage close requires quiescence");
    lpfx::device::DeviceIdentity identity{};
    std::int32_t ordinal = -1;
    const auto queried = Traits::query_current_device_identity(identity, ordinal);
    lpfx::device::check_device_error<Backend>(
        queried, "TPIC mass halo: storage close requires its selected device");
    lpfx::require(identity == device_identity_ && ordinal == device_ordinal_,
                  "TPIC mass halo: storage close requires its selected device");
    buffers_ = {};
    lpfx::device::check_device_error<Backend>(outgoing_.reset(), "TPIC mass halo: outgoing release failed");
    lpfx::device::check_device_error<Backend>(incoming_.reset(), "TPIC mass halo: incoming release failed");
    lpfx::device::check_device_error<Backend>(device_directions_.reset(),
                                              "TPIC mass halo: direction release failed");
    executor_ = executor_type{Traits::null_stream};
    mass_ = nullptr;
    initialized_ = false;
  }

  // TPIC numerical sequencing over the two registered LPFx operations, one
  // per phase; transport coalescing never changes the canonical additive fold.
  template <class AddOperation> void begin(AddOperation &add) {
    lpfx::require(quiescent() && add.reusable(), "TPIC mass halo: is not reusable");
    phase_ = Phase::add_active;
    lpfx::device::check_device_launch<Backend>("TPIC mass halo: prior device launch failed before the exchange");
    begin_phase<true>(add);
  }

  // Interior once, then complete when the ghost scatter is queued.
  template <class AddOperation, class GhostOperation>
  [[nodiscard]] HaloProgress progress(AddOperation &add, GhostOperation &ghosts) {
    lpfx::require(initialized_ && phase_ != Phase::ready,
                  "TPIC mass halo: progress requires an accepted exchange");
    if (phase_ == Phase::complete) return HaloProgress::complete;
    if (phase_ == Phase::add_active || phase_ == Phase::add_concluding) {
      if (!progress_phase(add, Phase::add_concluding)) return HaloProgress::pending;
      enqueue_unpack<true>();
      phase_ = Phase::ghost_active;
      begin_phase<false>(ghosts);
      return HaloProgress::interior;
    }
    if (!progress_phase(ghosts, Phase::ghost_concluding)) return HaloProgress::pending;
    enqueue_unpack<false>();
    phase_ = Phase::complete;
    return HaloProgress::complete;
  }

  void conclude() {
    lpfx::require(initialized_ && phase_ == Phase::complete,
                  "TPIC mass halo: conclusion requires completed communication");
    phase_ = Phase::ready;
  }

private:
  enum class Phase : std::uint8_t {
    ready,
    add_active,
    add_concluding,
    ghost_active,
    ghost_concluding,
    complete
  };

  template <bool Additive, class Operation> void begin_phase(Operation &operation) {
    enqueue_pack<Additive>();
    const auto status = operation.begin();
    lpfx::check_acceptance(status, "TPIC mass halo: registered message begin failed");
  }

  // Polls the active phase's operation once; once it has completed,
  // concludes it.
  template <class Operation> [[nodiscard]] bool progress_phase(Operation &operation, const Phase concluding) {
    if (phase_ != concluding) {
      if (!lpfx::check_progress(operation.progress(), "TPIC mass halo: message progress failed"))
        return false;
      phase_ = concluding;
    }
    return lpfx::check_progress(operation.conclude(), "TPIC mass halo: message conclusion failed");
  }

  // The device table of one phase; both phases have the same directions,
  // additive first.
  template <bool Additive> [[nodiscard]] const MassHaloDirection *directions() const noexcept {
    return reinterpret_cast<const MassHaloDirection *>(device_directions_.raw_base()) +
           (Additive ? 0 : direction_count());
  }
  [[nodiscard]] std::uint32_t direction_count() const noexcept {
    return static_cast<std::uint32_t>(plan_.template phase<true>().directions().size());
  }

  // Stream: the prepared executor. Reads the mass matrix, which the caller
  // has completed on that stream (deposition, or the additive scatter for the
  // ghost phase). Writes the outgoing buffer, which the registered LPFx
  // operation sends next. Throws on a launch error.
  template <bool Additive> void enqueue_pack() {
    const auto blocks = plan_.template phase<Additive>().pack_blocks();
    if (blocks == 0) return;
    lpfx::device::launch_or_throw<kernels::mass_halo_pack_kernel<Scalar>>(
        "TPIC mass halo: pack launch failed", executor_,
        lpfx::device::LaunchGrid{dim3(blocks), MassHaloLaunch::threads},
        mass_, reinterpret_cast<Scalar *>(outgoing_.raw_base()), plan_.layout(),
        directions<Additive>(), direction_count());
  }

  // Stream: the prepared executor. Reads the outgoing and incoming buffers of
  // the concluded phase and writes the matrix: the shared records for the
  // additive phase, the ghost records for the ghost phase. Throws on a launch
  // error.
  template <bool Additive> void enqueue_unpack() {
    const auto blocks = plan_.template phase<Additive>().unpack_blocks();
    if (blocks == 0) return;
    auto *const outgoing = reinterpret_cast<Scalar *>(outgoing_.raw_base());
    auto *const incoming = reinterpret_cast<Scalar *>(incoming_.raw_base());
    const lpfx::device::LaunchGrid grid{dim3(blocks), MassHaloLaunch::threads};
    if constexpr (Additive) {
      lpfx::device::launch_or_throw<kernels::mass_halo_unpack_add_kernel<Scalar>>(
          "TPIC mass halo: additive gather launch failed", executor_, grid,
          mass_, outgoing, incoming, plan_.layout(), directions<true>(), direction_count());
    } else {
      lpfx::device::launch_or_throw<kernels::mass_halo_unpack_copy_kernel<Scalar>>(
          "TPIC mass halo: copy scatter launch failed", executor_, grid,
          mass_, outgoing, incoming, plan_.layout(), directions<false>(), direction_count());
    }
  }

  using Scalar = typename Precision::Storage::Mass;

  plan_type plan_;
  std::array<buffers_type, 2> buffers_{};
  Scalar *mass_{};
  // Borrowed stream handle of the prepared executor; null before prepare.
  executor_type executor_{Traits::null_stream};
  lpfx::device::DeviceIdentity device_identity_{};
  std::int32_t device_ordinal_ = -1;
  Allocation outgoing_{};
  Allocation incoming_{};
  Allocation device_directions_{};
  Phase phase_ = Phase::ready;
  bool initialized_{};
};

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_MASS_HALO_MASS_HALO_DEVICE_HPP
