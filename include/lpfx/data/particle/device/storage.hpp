#ifndef LPFX_DATA_PARTICLE_DEVICE_STORAGE_HPP
#define LPFX_DATA_PARTICLE_DEVICE_STORAGE_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/fixed_array.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/device/view.hpp>
#include <lpfx/data/particle/layout.hpp>
#include <lpfx/data/particle/slots.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {

// Independent primary allocations plus one widest-attribute sort allocation.
template <detail::ParticleSchemaType Schema,
          device::DeviceAllocationBackend Backend>
class PreparedDeviceParticleStorage final {
  static_assert(Schema::alignment <=
                    device::BackendTraits<Backend>::
                        minimum_device_allocation_alignment,
                "particle attribute alignment exceeds device allocations");

public:
  using schema_type = Schema;
  using backend_type = Backend;

  constexpr PreparedDeviceParticleStorage() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::uint32_t capacity() const noexcept {
    return capacity_;
  }
  // The rotating sort scratch.
  [[nodiscard]] constexpr std::size_t auxiliary_bytes() const noexcept {
    return static_cast<std::size_t>(capacity_) * Schema::widest_width;
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return static_cast<std::size_t>(capacity_) *
           (Schema::record_width + Schema::widest_width);
  }

private:
  constexpr explicit PreparedDeviceParticleStorage(
      const std::uint32_t capacity) noexcept
      : capacity_(capacity), valid_(true) {}

  std::uint32_t capacity_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSchemaType Schema,
          device::DeviceAllocationBackend Backend,
          detail::checked_integer Capacity>
[[nodiscard]] constexpr ErrorCode try_prepare_device_particle_storage(
    const Capacity requested_capacity,
    PreparedDeviceParticleStorage<Schema, Backend> &output) noexcept {
  using Plan = PreparedDeviceParticleStorage<Schema, Backend>;
  if constexpr (std::signed_integral<Capacity>) {
    if (requested_capacity < 0) {
      return ErrorCode::invalid_argument;
    }
  }
  std::uint32_t capacity{};
  if (!detail::checked_integral_cast(requested_capacity, capacity)) {
    return ErrorCode::capacity_exceeded;
  }
  std::size_t total_bytes{};
  if (!detail::checked_multiply(static_cast<std::size_t>(capacity),
                                Schema::record_width + Schema::widest_width,
                                total_bytes)) {
    return ErrorCode::overflow;
  }
  output = detail::PrivateConstructionAccess::create<Plan>(capacity);
  return ErrorCode::success;
}

namespace detail {

struct DeviceParticlesAccess;

struct DeviceParticleRawSlot final {
  std::byte *raw_base{};
  std::size_t physical_bytes{};
  std::size_t physical_index{};
};

} // namespace detail

// Host-side identity of one device owner: its address, allocation
// incarnation, capacity and reorder epoch at capture time. The allocation
// incarnation prevents address ABA across reset, growth and move assignment,
// including null zero-capacity storage.
class DeviceParticleOwnerStamp final {
public:
  constexpr DeviceParticleOwnerStamp() noexcept = default;

  template <detail::ParticleSchemaType Schema,
            device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr DeviceParticleOwnerStamp
  capture(const Particles<Schema, DeviceSpace<Backend>> &owner) noexcept {
    DeviceParticleOwnerStamp stamp{};
    stamp.owner_ = &owner;
    stamp.allocation_ = owner.allocations_[0].raw_base();
    stamp.allocation_generation_ = owner.allocation_generation_;
    stamp.generation_ = owner.state_.generation;
    stamp.capacity_ = owner.state_.carousel.capacity;
    return stamp;
  }

  // owner is the captured object, is valid, and still holds the captured
  // allocation incarnation.
  template <detail::ParticleSchemaType Schema,
            device::DeviceAllocationBackend Backend>
  [[nodiscard]] constexpr bool
  matches(const Particles<Schema, DeviceSpace<Backend>> &owner) const noexcept {
    return owner_ == &owner && owner.valid() &&
           allocation_ == owner.allocations_[0].raw_base() &&
           allocation_generation_ == owner.allocation_generation_ &&
           capacity_ == owner.state_.carousel.capacity;
  }

  // matches() and no reorder or suffix publication advanced the epoch.
  template <detail::ParticleSchemaType Schema,
            device::DeviceAllocationBackend Backend>
  [[nodiscard]] constexpr bool matches_epoch(
      const Particles<Schema, DeviceSpace<Backend>> &owner) const noexcept {
    return matches(owner) && generation_ == owner.state_.generation;
  }

  // Opaque physical-slot anchor for identity and pointer affinity checks,
  // never an attribute base.
  [[nodiscard]] constexpr const std::byte *allocation() const noexcept {
    return allocation_;
  }
  [[nodiscard]] constexpr std::uint64_t
  allocation_generation() const noexcept {
    return allocation_generation_;
  }
  [[nodiscard]] constexpr std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] constexpr std::uint32_t capacity() const noexcept {
    return capacity_;
  }

private:
  const void *owner_ = nullptr;
  const std::byte *allocation_ = nullptr;
  std::uint64_t allocation_generation_ = 0;
  std::uint64_t generation_ = 0;
  std::uint32_t capacity_ = 0;
};

namespace detail {

// Host control metadata only. Kernel views remain N pointers plus a count.
// A planned mapping is unpublished until device validation has succeeded.
template <ParticleSchemaType Schema> struct DeviceParticleSlotMapping final {
  std::array<DeviceParticleRawSlot, Schema::attribute_count + 1> slots{};
  DeviceParticleOwnerStamp owner{};
  bool forward{true};
  bool rotates{};

  // The owner epoch once this mapping is committed.
  [[nodiscard]] constexpr std::uint64_t generation() const noexcept {
    return owner.generation() + (rotates ? 1 : 0);
  }
};

} // namespace detail

// Each physical attribute slot and the rotating sort scratch own an independent
// allocation. Logical slots may rotate without moving allocation ownership.
// Borrowed views must be retired before a reorder and refreshed after commit.
template <detail::ParticleSchemaType Schema,
          device::DeviceAllocationBackend Backend>
class Particles<Schema, DeviceSpace<Backend>> final {
public:
  using schema_type = Schema;
  using backend_type = Backend;
  using space_type = DeviceSpace<Backend>;
  using traits_type = device::BackendTraits<Backend>;
  using error_type = typename traits_type::error_type;
  using view_type = BasicParticleView<Schema, space_type, false>;
  using const_view_type = BasicParticleView<Schema, space_type, true>;
  static constexpr std::size_t attribute_count = Schema::attribute_count;

  constexpr Particles() noexcept = default;

  Particles(Particles &&other) noexcept
      : allocations_(std::move(other.allocations_)),
        allocation_generation_(other.allocation_generation_),
        state_(std::exchange(other.state_, {})) {}

  Particles &operator=(Particles &&other) noexcept {
    if (this != &other) {
      advance_allocation_generation();
      allocations_ = std::move(other.allocations_);
      state_ = std::exchange(other.state_, {});
    }
    return *this;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return state_.valid && !state_.growth.pending;
  }
  // A failed in-place growth retains ownership and its admitted resident data.
  // Retry the same operation; ordinary views remain unavailable until success.
  [[nodiscard]] constexpr bool growth_pending() const noexcept {
    return state_.growth.pending;
  }
  [[nodiscard]] constexpr std::uint32_t size() const noexcept {
    return state_.carousel.size;
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] constexpr std::uint32_t capacity() const noexcept {
    return state_.carousel.capacity;
  }

  [[nodiscard]] constexpr view_type view() noexcept {
    return make_view<view_type>(size());
  }

  [[nodiscard]] constexpr const_view_type view() const noexcept {
    return make_view<const_view_type>(size());
  }

  [[nodiscard]] constexpr const_view_type cview() const noexcept {
    return view();
  }

  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] constexpr auto get() noexcept {
    return view().template get<Name>();
  }

  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] constexpr auto get() const noexcept {
    return view().template get<Name>();
  }

  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    std::size_t bytes{};
    for (const auto &allocation : allocations_)
      bytes += allocation.capacity_bytes();
    return bytes;
  }

  // As with view(), the owner must have no pending operation when its active
  // prefix is changed. Shrinking never touches device memory.
  constexpr void clear() noexcept {
    require_accessible();
    state_.carousel.size = 0;
  }

  [[nodiscard]] constexpr bool
  try_shrink(const std::uint32_t new_size) noexcept {
    if (state_.growth.pending || new_size > size()) {
      return false;
    }
    state_.carousel.size = new_size;
    return true;
  }

  // Hard precondition: no submitted operation still references this storage.
  // The native allocation wrapper is consumed even when teardown reports an
  // asynchronous failure, so stale views are disarmed before returning it.
  [[nodiscard]] error_type reset_capacity() noexcept {
    advance_allocation_generation();
    auto error = traits_type::success_error;
    for (auto &allocation : allocations_) {
      const auto result = allocation.reset();
      if (traits_type::succeeded(error) && !traits_type::succeeded(result))
        error = result;
    }
    state_ = {};
    return error;
  }

  // Exclusive, sequential in-place growth. All readers and writers on every
  // stream must already be retired, and the current sorted mapping committed.
  // Preserve copies exactly the explicitly admitted prefix, even when it
  // differs from published size; published size becomes min(old size,
  // preserved_count). Allocation failure leaves the owner growth_pending(),
  // invalidates all old views, and retains the resident values. Retry with the
  // identical arguments; completed features are not allocated or copied again.
  // Backend synchronization or teardown failure is terminal, not a rollback or
  // recoverable allocation OOM.
  template <ParticleResizeMode Mode = ParticleResizeMode::preserve>
    requires(Mode == ParticleResizeMode::discard ||
             (device::DeviceCopyBackend<Backend> &&
              device::DeviceStreamBackend<Backend>))
  [[nodiscard]] ErrorCode
  grow_in_place(const std::uint32_t requested_capacity,
                const std::uint32_t preserved_count,
                const device::Executor<Backend> executor) noexcept {
    using Allocation = device::DeviceAllocation<Backend>;
    constexpr bool discard = Mode == ParticleResizeMode::discard;
    auto &growth = state_.growth;
    auto &carousel = state_.carousel;
    auto &slots = carousel.slots;
    if (!state_.valid || !executor.bound())
      return ErrorCode::invalid_argument;
    if (growth.pending) {
      if (growth.capacity != requested_capacity ||
          growth.preserved != preserved_count || growth.discard != discard)
        return ErrorCode::invalid_state;
    } else {
      if (requested_capacity < carousel.capacity ||
          preserved_count > carousel.capacity ||
          (discard && preserved_count != 0))
        return ErrorCode::invalid_argument;
      if (requested_capacity == carousel.capacity) {
        if constexpr (discard)
          carousel.size = 0;
        return ErrorCode::success;
      }
      // Invalidate every checked external access before changing any pointer.
      advance_allocation_generation();
      growth = {true, discard, requested_capacity, preserved_count, 0};
      const auto scratch = slots[attribute_count].physical_index;
      // The currently unused scratch can be a narrow original attribute slot.
      // Release its actual physical owner, never its canonical feature number.
      if (!traits_type::succeeded(allocations_[scratch].reset()))
        std::abort();
      slots[attribute_count] = {nullptr, 0, scratch};
    }
    for (; growth.stage != attribute_count; ++growth.stage) {
      const auto feature = Schema::forward_order[growth.stage];
      const auto physical = slots[feature].physical_index;
      const auto bytes = static_cast<std::size_t>(requested_capacity) *
                         Schema::widths[feature];
      Allocation replacement;
      if (!traits_type::succeeded(Allocation::create(bytes, replacement)))
        return ErrorCode::allocation_failed;
      if constexpr (!discard) {
        if (preserved_count != 0) {
          const auto error = traits_type::copy_device_to_device_async(
              replacement.raw_base(), slots[feature].raw_base,
              static_cast<std::size_t>(preserved_count) *
                  Schema::widths[feature],
              executor.stream());
          // Every feature copy completes before its source allocation dies.
          // A failed submission may still have queued work; drain it as well.
          if (!traits_type::succeeded(
                  traits_type::synchronize_stream(executor.stream())))
            std::abort();
          if (!traits_type::succeeded(error))
            return ErrorCode::backend_failure;
        }
      }
      Allocation previous = std::move(allocations_[physical]);
      allocations_[physical] = std::move(replacement);
      slots[feature] = {allocations_[physical].raw_base(), bytes, physical};
      if (!traits_type::succeeded(previous.reset()))
        std::abort();
    }
    const auto scratch = slots[attribute_count].physical_index;
    const auto scratch_bytes = static_cast<std::size_t>(requested_capacity) *
                               Schema::widest_width;
    if (!traits_type::succeeded(
            Allocation::create(scratch_bytes, allocations_[scratch])))
      return ErrorCode::allocation_failed;
    // Canonicalize physical ownership as well as the logical pointer mapping.
    // Mixed-width growth must restart with widest scratch and a forward sweep.
    allocations_type canonical;
    for (std::size_t feature = 0; feature != attribute_count + 1; ++feature) {
      canonical[feature] =
          std::move(allocations_[slots[feature].physical_index]);
      slots[feature] = {canonical[feature].raw_base(),
                        canonical[feature].capacity_bytes(), feature};
    }
    allocations_ = std::move(canonical);
    carousel.capacity = requested_capacity;
    if constexpr (discard)
      carousel.size = 0;
    else
      carousel.size = std::min(carousel.size, preserved_count);
    carousel.forward = true;
    state_.generation = 0;
    growth = {};
    return ErrorCode::success;
  }

private:
  using allocations_type =
      std::array<device::DeviceAllocation<Backend>, attribute_count + 1>;
  using carousel_type =
      detail::ParticleSlotCarousel<Schema, detail::DeviceParticleRawSlot,
                                   std::uint32_t>;

  // The resumable in-place growth: target, admitted prefix and next feature.
  struct Growth {
    bool pending{};
    bool discard{};
    std::uint32_t capacity{};
    std::uint32_t preserved{};
    std::size_t stage{};
  };

  struct State {
    carousel_type carousel{};
    std::uint64_t generation{};
    bool valid{};
    Growth growth{};
  };

  Particles(allocations_type &&allocations,
            const carousel_type &carousel) noexcept
      : allocations_(std::move(allocations)), allocation_generation_(1),
        state_{carousel, 0, true, {}} {}

  constexpr void require_accessible() const noexcept {
    if (state_.growth.pending)
      std::abort();
  }

  template <class View>
  [[nodiscard]] constexpr View make_view(const std::uint32_t count) const noexcept {
    require_accessible();
    return detail::PrivateConstructionAccess::create<View>(
        detail::DeviceFixedArray<std::byte *, attribute_count>{
            state_.carousel.bases()},
        count);
  }

  void advance_allocation_generation() noexcept {
    if (allocation_generation_ == std::numeric_limits<std::uint64_t>::max())
      std::abort();
    ++allocation_generation_;
  }

  allocations_type allocations_{};
  std::uint64_t allocation_generation_{};
  State state_{};

  friend class DeviceParticleOwnerStamp;
  friend struct detail::DeviceParticlesAccess;
  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

struct DeviceParticlesAccess final {
  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr auto
  capacity_view(Particles<Schema, DeviceSpace<Backend>> &particles) noexcept {
    return particles.template make_view<
        BasicParticleView<Schema, DeviceSpace<Backend>, false>>(
        particles.capacity());
  }

  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr auto
  const_prefix_view(const Particles<Schema, DeviceSpace<Backend>> &particles,
                    const std::uint32_t count) noexcept {
    return particles.template make_view<
        BasicParticleView<Schema, DeviceSpace<Backend>, true>>(count);
  }

  // Successful reorders or suffix publications invalidate prior
  // particle/bin epochs without changing slots or the next sweep. Callers
  // precheck overflow before submission and invoke this only for a nonempty
  // commit.
  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr ErrorCode advance_reorder_generation(
      Particles<Schema, DeviceSpace<Backend>> &particles) noexcept {
    if (!particles.valid())
      return ErrorCode::invalid_state;
    if (particles.state_.generation ==
        std::numeric_limits<std::uint64_t>::max())
      return ErrorCode::overflow;
    ++particles.state_.generation;
    return ErrorCode::success;
  }

  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr DeviceParticleSlotMapping<Schema>
  current_mapping(
      const Particles<Schema, DeviceSpace<Backend>> &particles) noexcept {
    return {particles.state_.carousel.slots,
            DeviceParticleOwnerStamp::capture(particles),
            particles.state_.carousel.forward, false};
  }

  template <ParticleSchemaType Schema, device::DeviceBackend Backend>
  [[nodiscard]] static constexpr auto
  pending_view(const DeviceParticleSlotMapping<Schema> &mapping,
               const std::uint32_t count) noexcept {
    DeviceFixedArray<std::byte *, Schema::attribute_count> bases{};
    for (std::size_t attribute = 0; attribute != Schema::attribute_count;
         ++attribute) {
      bases[attribute] = mapping.slots[attribute].raw_base;
    }
    return PrivateConstructionAccess::create<
        BasicParticleView<Schema, DeviceSpace<Backend>, false>>(bases, count);
  }

  // Plans a nonempty full-owner or admitted inactive-suffix reorder. This does
  // not inspect size(): a migration capability supplies its exact resident R.
  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr ErrorCode
  plan_reorder(const Particles<Schema, DeviceSpace<Backend>> &particles,
               DeviceParticleSlotMapping<Schema> &output) noexcept {
    if (!particles.valid()) {
      return ErrorCode::invalid_argument;
    }
    if (particles.state_.generation ==
        std::numeric_limits<std::uint64_t>::max()) {
      return ErrorCode::overflow;
    }
    auto carousel = particles.state_.carousel;
    const auto fits = [&](const auto attribute) noexcept {
      return carousel.slots[Schema::attribute_count].physical_bytes <
                     static_cast<std::size_t>(carousel.capacity) *
                         Schema::widths[decltype(attribute)::value]
                 ? ErrorCode::capacity_exceeded
                 : ErrorCode::success;
    };
    LPFX_TRY(carousel.forward ? carousel.template rotate<true>(fits)
                              : carousel.template rotate<false>(fits));
    output = {carousel.slots, DeviceParticleOwnerStamp::capture(particles),
              carousel.forward, true};
    return ErrorCode::success;
  }

  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr ErrorCode
  plan_reorder(const Particles<Schema, DeviceSpace<Backend>> &particles,
               const std::uint32_t count,
               DeviceParticleSlotMapping<Schema> &output) noexcept {
    if (!particles.valid()) {
      return ErrorCode::invalid_argument;
    }
    if (count > particles.capacity()) {
      return ErrorCode::capacity_exceeded;
    }
    if (count == 0) {
      output = current_mapping(particles);
      return ErrorCode::success;
    }
    return plan_reorder(particles, output);
  }

  // Internal checked publication: callers prove that all guarded scatters and
  // any suffix construction succeeded and every old view has been retired.
  // Size publication is separate because migration may append a live suffix.
  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  [[nodiscard]] static constexpr ErrorCode
  commit_reorder(Particles<Schema, DeviceSpace<Backend>> &particles,
                 const DeviceParticleSlotMapping<Schema> &mapping) noexcept {
    auto &state = particles.state_;
    if (!mapping.owner.matches_epoch(particles) ||
        state.carousel.forward != mapping.forward) {
      return ErrorCode::invalid_state;
    }
    if (mapping.rotates) {
      state.carousel.slots = mapping.slots;
      state.carousel.forward = !mapping.forward;
      ++state.generation;
    }
    return ErrorCode::success;
  }

  // Hard precondition: the producing operation has completed successfully and
  // new_size does not exceed capacity.
  template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
  static constexpr void
  publish_size(Particles<Schema, DeviceSpace<Backend>> &particles,
               const std::uint32_t new_size) noexcept {
    particles.require_accessible();
    particles.state_.carousel.size = new_size;
  }
};

} // namespace detail

template <detail::ParticleSchemaType Schema,
          device::DeviceAllocationBackend Backend>
[[nodiscard]] Result<Particles<Schema, DeviceSpace<Backend>>>
make_device_particles(
    const PreparedDeviceParticleStorage<Schema, Backend> &plan) noexcept {
  using Owner = Particles<Schema, DeviceSpace<Backend>>;
  using Traits = device::BackendTraits<Backend>;
  using Allocation = device::DeviceAllocation<Backend>;
  constexpr auto count = Schema::attribute_count;
  if (!plan.valid()) {
    return Result<Owner>::from_error(ErrorCode::invalid_argument);
  }
  std::array<Allocation, count + 1> allocations{};
  detail::ParticleSlotCarousel<Schema, detail::DeviceParticleRawSlot,
                               std::uint32_t>
      carousel{};
  carousel.capacity = plan.capacity();
  for (std::size_t slot = 0; slot != count + 1; ++slot) {
    const auto bytes =
        static_cast<std::size_t>(plan.capacity()) *
        (slot == count ? Schema::widest_width : Schema::widths[slot]);
    if (!Traits::succeeded(Allocation::create(bytes, allocations[slot])))
      return Result<Owner>::from_error(ErrorCode::allocation_failed);
    carousel.slots[slot] = {allocations[slot].raw_base(), bytes, slot};
  }
  return Result<Owner>::from_value(
      detail::PrivateConstructionAccess::create<Owner>(std::move(allocations),
                                                       carousel));
}

} // namespace lpfx

#endif
