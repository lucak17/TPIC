#ifndef LPFX_DISTRIBUTED_PARTICLE_WIRE_STORAGE_HPP
#define LPFX_DISTRIBUTED_PARTICLE_WIRE_STORAGE_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/distributed/particle/migration/capacity.hpp>
#include <lpfx/distributed/particle/wire/image.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/allocator.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx {

enum class ParticleMigrationWireSlot : std::uint8_t { a, b };

namespace detail {

inline constexpr std::size_t particle_wire_slot_alignment = 64;
// Device codecs issue typed accesses that assume aligned image bases.
static_assert(particle_wire_slot_alignment %
                  particle_wire_maximum_alignment ==
              0);
using HostParticleWireAllocator =
    HostAlignedAllocator<particle_wire_slot_alignment>;

template <class Space> struct ParticleWireAllocation final {
  using type = RawHostAllocation<HostParticleWireAllocator>;
};

template <class Backend> struct ParticleWireAllocation<DeviceSpace<Backend>> final {
  using type = device::DeviceAllocation<Backend>;
};

} // namespace detail

// Geometry of the two reusable migration wire images. Slot A holds the
// initial outgoing image and, after each receive, the compacted transit image
// for the next round. Slot B holds the current round's incoming image. The
// 64-byte slot alignment keeps A and B from sharing a 64-byte line while
// send/receive or decode/repack work overlaps.
template <detail::ParticleWireContractType Contract>
class PreparedParticleMigrationWireStorage final {
public:
  static constexpr std::size_t slot_alignment =
      detail::particle_wire_slot_alignment;
  using contract_type = Contract;

  constexpr PreparedParticleMigrationWireStorage() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::size_t slot_a_bytes() const noexcept {
    return slot_a_bytes_;
  }
  [[nodiscard]] constexpr std::size_t slot_b_offset() const noexcept {
    return slot_b_offset_;
  }
  [[nodiscard]] constexpr std::size_t slot_b_bytes() const noexcept {
    return slot_b_bytes_;
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return total_bytes_;
  }

  friend constexpr bool
  operator==(const PreparedParticleMigrationWireStorage &,
             const PreparedParticleMigrationWireStorage &) noexcept = default;

private:
  constexpr PreparedParticleMigrationWireStorage(
      const std::size_t slot_a_bytes, const std::size_t slot_b_offset,
      const std::size_t slot_b_bytes, const std::size_t total_bytes) noexcept
      : slot_a_bytes_(slot_a_bytes), slot_b_offset_(slot_b_offset),
        slot_b_bytes_(slot_b_bytes), total_bytes_(total_bytes), valid_(true) {}

  std::size_t slot_a_bytes_{};
  std::size_t slot_b_offset_{};
  std::size_t slot_b_bytes_{};
  std::size_t total_bytes_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleWireContractType Contract>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_migration_wire_storage(
    const PreparedParticleMigrationCapacity<Contract> &capacity,
    PreparedParticleMigrationWireStorage<Contract> &output) noexcept {
  using Prepared = PreparedParticleMigrationWireStorage<Contract>;
  if (!capacity.valid()) {
    return ErrorCode::invalid_argument;
  }
  std::size_t outgoing_bytes{};
  std::size_t incoming_bytes{};
  std::size_t transit_bytes{};
  if (!detail::checked_integral_cast(capacity.maximum_outgoing_bytes(),
                                     outgoing_bytes) ||
      !detail::checked_integral_cast(capacity.maximum_incoming_bytes(),
                                     incoming_bytes) ||
      !detail::checked_integral_cast(capacity.maximum_transit_bytes(),
                                     transit_bytes)) {
    return ErrorCode::overflow;
  }
  if constexpr (Contract::wire_extent_slack != 0) {
    // The gaps between aligned packets add at most the slack to an image
    // (image.hpp).
    constexpr auto slack = static_cast<std::size_t>(Contract::wire_extent_slack);
    if ((outgoing_bytes != 0 &&
         !detail::checked_add(outgoing_bytes, slack, outgoing_bytes)) ||
        (incoming_bytes != 0 &&
         !detail::checked_add(incoming_bytes, slack, incoming_bytes)) ||
        (transit_bytes != 0 &&
         !detail::checked_add(transit_bytes, slack, transit_bytes))) {
      return ErrorCode::overflow;
    }
  }
  const std::size_t slot_a_bytes = std::max(outgoing_bytes, transit_bytes);
  std::size_t slot_b_offset{};
  std::size_t total_bytes = slot_a_bytes;
  if (incoming_bytes != 0 &&
      (!detail::checked_align_up(slot_a_bytes, Prepared::slot_alignment,
                                 slot_b_offset) ||
       !detail::checked_add(slot_b_offset, incoming_bytes, total_bytes))) {
    return ErrorCode::overflow;
  }
  output = detail::PrivateConstructionAccess::create<Prepared>(
      slot_a_bytes, slot_b_offset, incoming_bytes, total_bytes);
  return ErrorCode::success;
}

// One owner for the host and device wire slots of a prepared geometry.
template <detail::ParticleWireContractType Contract, class Space>
class ParticleMigrationWireStorage final {
public:
  using contract_type = Contract;
  using space_type = Space;
  using allocation_type = typename detail::ParticleWireAllocation<Space>::type;

  constexpr ParticleMigrationWireStorage() noexcept = default;
  ParticleMigrationWireStorage(ParticleMigrationWireStorage &&other) noexcept
      : allocation_(std::move(other.allocation_)),
        prepared_(std::exchange(other.prepared_, {})) {}
  ParticleMigrationWireStorage &
  operator=(ParticleMigrationWireStorage &&other) noexcept {
    if (this != &other) {
      allocation_ = std::move(other.allocation_);
      prepared_ = std::exchange(other.prepared_, {});
    }
    return *this;
  }

  [[nodiscard]] constexpr const PreparedParticleMigrationWireStorage<Contract> &
  prepared() const noexcept {
    return prepared_;
  }

  [[nodiscard]] constexpr BufferView<std::byte, Space>
  slot_a_capacity() noexcept {
    return view(allocation_.raw_base(), prepared_.slot_a_bytes());
  }
  [[nodiscard]] constexpr BufferView<const std::byte, Space>
  slot_a_capacity() const noexcept {
    return view(allocation_.raw_base(), prepared_.slot_a_bytes());
  }

  // The active bytes of the outgoing (slot A) or incoming (slot B) image.
  template <ParticlePeerPrefixKind Kind>
  [[nodiscard]] constexpr BufferView<std::byte, Space>
  bytes(const PreparedParticlePeerPrefixes<Contract, Kind> &prefixes) noexcept {
    return view(slot_base<Kind == ParticlePeerPrefixKind::outgoing
                              ? ParticleMigrationWireSlot::a
                              : ParticleMigrationWireSlot::b>(),
                static_cast<std::size_t>(prefixes.total_bytes()));
  }
  template <ParticlePeerPrefixKind Kind>
  [[nodiscard]] constexpr BufferView<const std::byte, Space>
  bytes(const PreparedParticlePeerPrefixes<Contract, Kind> &prefixes)
      const noexcept {
    return view(slot_base<Kind == ParticlePeerPrefixKind::outgoing
                              ? ParticleMigrationWireSlot::a
                              : ParticleMigrationWireSlot::b>(),
                static_cast<std::size_t>(prefixes.total_bytes()));
  }

  template <ParticleMigrationWireSlot Slot,
            detail::ParticleWireImageCount Count>
  [[nodiscard]] constexpr auto image(const Count count) noexcept {
    return BasicParticleWireImageView<Contract, std::byte, Count>{
        count == 0 ? nullptr : slot_base<Slot>(), count};
  }
  template <ParticleMigrationWireSlot Slot,
            detail::ParticleWireImageCount Count>
  [[nodiscard]] constexpr auto image(const Count count) const noexcept {
    return BasicParticleWireImageView<Contract, const std::byte, Count>{
        count == 0 ? nullptr : slot_base<Slot>(), count};
  }

  void reset_capacity() noexcept {
    if constexpr (std::same_as<Space, HostSpace>) {
      allocation_.reset_capacity();
    } else {
      static_cast<void>(allocation_.reset());
    }
    prepared_ = {};
  }

private:
  ParticleMigrationWireStorage(
      allocation_type &&allocation,
      const PreparedParticleMigrationWireStorage<Contract> prepared) noexcept
      : allocation_(std::move(allocation)), prepared_(prepared) {}

  template <class Byte>
  [[nodiscard]] static constexpr BufferView<Byte, Space>
  view(Byte *const base, const std::size_t bytes) noexcept {
    return detail::make_buffer_view<Byte, Space>(bytes == 0 ? nullptr : base,
                                                 bytes);
  }

  template <ParticleMigrationWireSlot Slot>
  [[nodiscard]] constexpr std::byte *slot_base() noexcept {
    std::byte *const base = allocation_.raw_base();
    return Slot == ParticleMigrationWireSlot::a || base == nullptr
               ? base
               : base + prepared_.slot_b_offset();
  }
  template <ParticleMigrationWireSlot Slot>
  [[nodiscard]] constexpr const std::byte *slot_base() const noexcept {
    const std::byte *const base = allocation_.raw_base();
    return Slot == ParticleMigrationWireSlot::a || base == nullptr
               ? base
               : base + prepared_.slot_b_offset();
  }

  allocation_type allocation_{};
  PreparedParticleMigrationWireStorage<Contract> prepared_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleWireContractType Contract>
using HostParticleMigrationWireStorage =
    ParticleMigrationWireStorage<Contract, HostSpace>;

template <detail::ParticleWireContractType Contract,
          device::DeviceAllocationBackend Backend>
using DeviceParticleMigrationWireStorage =
    ParticleMigrationWireStorage<Contract, DeviceSpace<Backend>>;

template <detail::ParticleWireContractType Contract>
[[nodiscard]] Result<HostParticleMigrationWireStorage<Contract>>
make_host_particle_migration_wire_storage(
    HostMemoryBudget &budget,
    const PreparedParticleMigrationWireStorage<Contract> &prepared) noexcept {
  using Owner = HostParticleMigrationWireStorage<Contract>;
  auto allocation =
      detail::allocate_single_host_layout<detail::HostParticleWireAllocator>(
          budget, prepared);
  LPFX_TRY_RESULT(allocation, Result<Owner>);
  return Result<Owner>::from_value(
      detail::PrivateConstructionAccess::create<Owner>(
          std::move(allocation).value(), prepared));
}

// Contracts with packet gaps zero their new slots, which needs a copy and
// stream backend.
template <device::DeviceAllocationBackend Backend,
          detail::ParticleWireContractType Contract>
  requires(Contract::wire_extent_slack == 0 ||
           (device::DeviceCopyBackend<Backend> &&
            device::DeviceStreamBackend<Backend>))
[[nodiscard]] Result<DeviceParticleMigrationWireStorage<Contract, Backend>>
make_device_particle_migration_wire_storage(
    const PreparedParticleMigrationWireStorage<Contract> &prepared) noexcept {
  using Owner = DeviceParticleMigrationWireStorage<Contract, Backend>;
  using Allocation = device::DeviceAllocation<Backend>;
  if (!prepared.valid() ||
      PreparedParticleMigrationWireStorage<Contract>::slot_alignment >
          device::BackendTraits<Backend>::minimum_device_allocation_alignment) {
    return Result<Owner>::from_error(ErrorCode::invalid_argument);
  }
  Allocation allocation{};
  if (!device::BackendTraits<Backend>::succeeded(
          Allocation::create(prepared.total_bytes(), allocation))) {
    return Result<Owner>::from_error(ErrorCode::allocation_failed);
  }
  if constexpr (Contract::wire_extent_slack != 0) {
    using Traits = device::BackendTraits<Backend>;
    // Whole-image copies (staged downloads, growth) span the unsent gaps
    // between aligned packets; zero the new slots once so no copied byte is
    // uninitialized. Nothing references them yet, and the stream
    // synchronization orders the fill before any later use.
    if (prepared.total_bytes() != 0 &&
        (!Traits::succeeded(Traits::memset_device_async(
             allocation.raw_base(), 0, prepared.total_bytes(),
             Traits::null_stream)) ||
         !Traits::succeeded(Traits::synchronize_stream(Traits::null_stream)))) {
      return Result<Owner>::from_error(ErrorCode::backend_failure);
    }
  }
  return Result<Owner>::from_value(
      detail::PrivateConstructionAccess::create<Owner>(std::move(allocation),
                                                       prepared));
}

} // namespace lpfx

#endif
