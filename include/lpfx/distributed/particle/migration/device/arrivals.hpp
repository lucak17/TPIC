#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_ARRIVALS_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_ARRIVALS_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/device/view.hpp>
#include <lpfx/data/particle/layout.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx::detail {

// Final-only decoded chunks retain padded high-water allocations across epochs.
// Any fitting inactive chunk is moved into the active prefix, so reuse does not
// depend on the routing-round ordinal. Only active_count records belong to an
// allocation range; consecutive rounds share spare chunk capacity. No transit
// record or reused RX image is retained. Addresses stay stable until the final
// append event completes.
template <ParticleSchemaType Schema, device::DeviceAllocationBackend Backend>
class DeviceParticleArrivals final {
public:
  using view_type = BasicParticleView<Schema, DeviceSpace<Backend>, false>;
  using traits_type = device::BackendTraits<Backend>;

  DeviceParticleArrivals() noexcept = default;
  LPFX_NONMOVABLE(DeviceParticleArrivals);

  // A single fitting chunk covers any split of this cumulative final bound.
  // Cold reservation is exclusive; it never marks records active.
  [[nodiscard]] ErrorCode reserve(HostMemoryBudget &budget,
                                 const std::uint32_t count) noexcept {
    if (!empty())
      return ErrorCode::invalid_state;
    view_type ignored;
    const auto error = allocate<true>(budget, count, ignored);
    if (error == ErrorCode::success)
      clear();
    return error;
  }
  [[nodiscard]] std::uint32_t maximum_capacity() const noexcept {
    std::uint32_t capacity = 0;
    for (auto *chunk = head_.get(); chunk; chunk = chunk->next.get())
      if (chunk->capacity > capacity)
        capacity = chunk->capacity;
    return capacity;
  }
  // Publish a staged reservation only after the enclosing rank agreement.
  // Both pools are inactive; retain previous high-water chunks behind it.
  void prepend_reservation(DeviceParticleArrivals &&reserved) noexcept {
    if (!empty() || !reserved.empty())
      std::abort();
    if (!reserved.head_)
      return;
    auto *last = reserved.head_.get();
    while (last->next)
      last = last->next.get();
    last->next = std::move(head_);
    head_ = std::move(reserved.head_);
  }

  template <bool ExactCapacity = false>
  [[nodiscard]] ErrorCode allocate(HostMemoryBudget &budget,
                                  const std::uint32_t count,
                                  view_type &destination) noexcept {
    destination = {};
    if (count == 0)
      return ErrorCode::success;

    // Earlier decoded ranges must retain their addresses. Consume spare space
    // in the last active chunk, then use an inactive fitting chunk if necessary.
    if (active_tail_ && count <= active_tail_->capacity - active_tail_->active_count) {
      destination = activate(*active_tail_, count);
      return ErrorCode::success;
    }
    auto *inactive_link = active_tail_ ? &active_tail_->next : &head_;
    auto *fitting_link = inactive_link;
    while (*fitting_link && (*fitting_link)->capacity < count)
      fitting_link = &(*fitting_link)->next;
    if (*fitting_link && fitting_link != inactive_link) {
      auto fitting = std::move(*fitting_link);
      *fitting_link = std::move(fitting->next);
      fitting->next = std::move(*inactive_link);
      *inactive_link = std::move(fitting);
    }

    std::unique_ptr<Chunk> pending;
    Chunk *chunk = inactive_link->get();
    if (chunk == nullptr) {
      HostMemoryRequirement requirement;
      if (!requirement.try_add(sizeof(Chunk), alignof(Chunk)))
        return ErrorCode::overflow;
      auto admitted = budget.try_admit(requirement);
      LPFX_TRY(admitted);
      pending = std::unique_ptr<Chunk>{new (std::nothrow) Chunk};
      if (!pending)
        return ErrorCode::allocation_failed;
      pending->admission = std::move(admitted).value();
      chunk = pending.get();
    }
    if (count > chunk->capacity) {
      std::uint32_t capacity{};
      if constexpr (ExactCapacity) {
        capacity = count;
      } else {
        LPFX_TRY(lpfx::try_prepare_particle_capacity(count, capacity));
      }
      PreparedParticleLayout<Schema, DeviceSpace<Backend>> layout;
      LPFX_TRY(try_prepare_particle_layout(capacity, layout));
      device::DeviceAllocation<Backend> candidate;
      if (!traits_type::succeeded(device::DeviceAllocation<Backend>::create(
              layout.total_bytes(), candidate)))
        return ErrorCode::allocation_failed;
      const DeviceFixedArray<std::byte *, Schema::attribute_count> bases{
          layout.bases(candidate.raw_base())};
      // Publish only after native allocation succeeds. Replacement may release
      // the obsolete inactive image, whose values never need preservation.
      chunk->allocation = std::move(candidate);
      chunk->bases = bases;
      chunk->capacity = capacity;
    }
    if (pending)
      *inactive_link = std::move(pending);
    active_tail_ = chunk;
    destination = activate(*chunk, count);
    return ErrorCode::success;
  }

  [[nodiscard]] auto append(const view_type destination,
                            std::uint32_t offset,
                            const device::Executor<Backend> executor) noexcept
    requires device::DeviceCopyBackend<Backend>
  {
    for (auto *chunk = head_.get(); chunk && chunk->active_count != 0;
         chunk = chunk->next.get()) {
      auto error = traits_type::success_error;
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        ((traits_type::succeeded(error)
              ? static_cast<void>(error = traits_type::copy_device_to_device_async(
                    destination.template get<I>() + offset,
                    chunk->bases[I],
                    static_cast<std::size_t>(chunk->active_count) *
                        Schema::widths[I],
                    executor.stream()))
              : static_cast<void>(0)), ...);
      }(std::make_index_sequence<Schema::attribute_count>{});
      if (!traits_type::succeeded(error))
        return error;
      offset += chunk->active_count;
    }
    return traits_type::success_error;
  }

  [[nodiscard]] bool empty() const noexcept {
    return !head_ || head_->active_count == 0;
  }
  void clear() noexcept {
    for (auto *chunk = head_.get(); chunk && chunk->active_count != 0;
         chunk = chunk->next.get())
      chunk->active_count = 0;
    active_tail_ = nullptr;
  }
  ~DeviceParticleArrivals() noexcept { release(); }

private:
  void release() noexcept {
    // Iterative destruction also handles long routing paths without recursion.
    while (head_) {
      auto next = std::move(head_->next);
      head_ = std::move(next);
    }
    active_tail_ = nullptr;
  }
  struct Chunk final {
    HostMemoryLease admission;
    device::DeviceAllocation<Backend> allocation;
    DeviceFixedArray<std::byte *, Schema::attribute_count> bases;
    std::uint32_t capacity = 0;
    std::uint32_t active_count = 0;
    std::unique_ptr<Chunk> next;
  };
  [[nodiscard]] static view_type activate(Chunk &chunk,
                                          const std::uint32_t count) noexcept {
    const auto offset = chunk.active_count;
    DeviceFixedArray<std::byte *, Schema::attribute_count> bases{};
    for (std::size_t attribute = 0; attribute != Schema::attribute_count;
         ++attribute)
      bases[attribute] = chunk.bases[attribute] +
                         static_cast<std::size_t>(offset) *
                             Schema::widths[attribute];
    chunk.active_count += count;
    return PrivateConstructionAccess::create<view_type>(bases, count);
  }
  std::unique_ptr<Chunk> head_;
  Chunk *active_tail_ = nullptr;
};

} // namespace lpfx::detail
#endif
