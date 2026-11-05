#ifndef LPFX_DATA_PARTICLE_DEVICE_INITIALIZATION_HPP
#define LPFX_DATA_PARTICLE_DEVICE_INITIALIZATION_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/particle/device/storage.hpp>

namespace lpfx {

// Kernel-only access to an unpublished suffix. Indices passed to store() and
// get() are local to this range; first() gives the global particle offset.
// Unlike owner views, suffix pointers promise only their element alignment.
// Every attribute of every granted particle must be constructed before publish.
template <detail::ParticleSchemaType Schema,
          device::DeviceAllocationBackend Backend>
class DeviceParticleInitializationRange final {
public:
  using schema_type = Schema;
  using view_type = BasicParticleView<Schema, DeviceSpace<Backend>, false>;
  template <std::size_t I>
  using value_type = typename Schema::template value_type<I>;
  template <class Name>
  using named_value_type = value_type<Schema::template index_of<Name>>;

  LPFX_HOST_DEVICE constexpr DeviceParticleInitializationRange() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint32_t first() const noexcept {
    return first_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint32_t size() const noexcept {
    return count_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool empty() const noexcept {
    return count_ == 0;
  }

  template <std::size_t I>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto get() const noexcept {
    auto *const base = view_.template get<I>();
    return first_ == 0 ? base : base + first_;
  }
  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto get() const noexcept {
    return get<Schema::template index_of<Name>>();
  }

  // Explicit lifetime start without a preceding fill kernel or host record.
  // Hard precondition: index < size(). Raw get() users provide equivalent
  // device construction themselves; no host dereference is permitted.
  template <std::size_t I>
    requires std::is_nothrow_copy_constructible_v<value_type<I>>
  LPFX_DEVICE LPFX_ALWAYS_INLINE void
  store(const std::uint32_t index, const value_type<I> value) const noexcept {
    ::new (static_cast<void *>(get<I>() + index)) value_type<I>(value);
  }
  template <class Name>
    requires Schema::template contains<Name> &&
             std::is_nothrow_copy_constructible_v<named_value_type<Name>>
  LPFX_DEVICE LPFX_ALWAYS_INLINE void
  store(const std::uint32_t index,
        const named_value_type<Name> value) const noexcept {
    store<Schema::template index_of<Name>>(index, value);
  }

private:
  constexpr DeviceParticleInitializationRange(const view_type view,
                                              const std::uint32_t first,
                                              const std::uint32_t count) noexcept
      : view_(view), first_(first), count_(count) {}

  view_type view_{};
  std::uint32_t first_{};
  std::uint32_t count_{};

  friend struct detail::PrivateConstructionAccess;
};

// One allocation-bound reusable event; no staging, scratch, payload transfers,
// implicit fills or per-epoch allocations. Startup is append to an empty owner.
// The owner must stay alive at its original address/allocation. Retire borrowed
// views before begin_append; no overlapping initialization or structural owner
// mutation is allowed until completion/cancellation. Recreate after growth.
// Device/context affinity and successful kernel submission are caller duties.
template <detail::ParticleSchemaType Schema,
          device::DeviceAllocationBackend Backend>
  requires device::DeviceEventBackend<Backend>
class DeviceParticleInitializer final {
public:
  using owner_type = Particles<Schema, DeviceSpace<Backend>>;
  using range_type = DeviceParticleInitializationRange<Schema, Backend>;
  using executor_type = device::Executor<Backend>;
  using traits_type = device::BackendTraits<Backend>;

  constexpr DeviceParticleInitializer() noexcept = default;
  DeviceParticleInitializer &operator=(DeviceParticleInitializer &&) = delete;
  DeviceParticleInitializer(DeviceParticleInitializer &&other) noexcept
      : event_(std::move(other.event_)),
        state_(std::exchange(other.state_, {})) {}

  ~DeviceParticleInitializer() noexcept {
    if (producing() || pending()) std::abort();
  }

  [[nodiscard]] bool valid() const noexcept {
    return state_.phase != Phase::unprepared;
  }
  [[nodiscard]] bool reusable() const noexcept {
    return state_.phase == Phase::idle;
  }
  [[nodiscard]] bool producing() const noexcept {
    return state_.phase == Phase::producing;
  }
  [[nodiscard]] bool pending() const noexcept {
    return state_.phase == Phase::pending;
  }

  [[nodiscard]] Result<range_type>
  begin_append(const std::uint64_t count, const executor_type producer) noexcept {
    if (!reusable() || !state_.stamp.matches(*state_.owner))
      return Result<range_type>::from_error(ErrorCode::invalid_state);
    if (!producer.bound())
      return Result<range_type>::from_error(ErrorCode::invalid_argument);
    owner_type &owner = *state_.owner;
    if (count > owner.capacity() - owner.size())
      return Result<range_type>::from_error(ErrorCode::capacity_exceeded);
    const auto stamp = DeviceParticleOwnerStamp::capture(owner);
    if (count != 0 &&
        stamp.generation() == std::numeric_limits<std::uint64_t>::max())
      return Result<range_type>::from_error(ErrorCode::overflow);
    state_.stamp = stamp;
    state_.first = owner.size();
    state_.count = static_cast<std::uint32_t>(count);
    state_.producer = producer;
    state_.phase = Phase::producing;
    return Result<range_type>::from_value(
        detail::PrivateConstructionAccess::create<range_type>(
            detail::DeviceParticlesAccess::capacity_view(owner), state_.first,
            state_.count));
  }

  // Call after all successful initialization launches on the granted stream.
  // The caller must first join work on any other stream into this stream.
  // This only records completion, and never waits or changes owner.size().
  [[nodiscard]] Status publish(const executor_type producer) noexcept {
    if (!producing()) return Status::failure(ErrorCode::invalid_state);
    if (!same_producer(producer))
      return Status::failure(ErrorCode::invalid_argument);
    if (!epoch_matches()) return Status::failure(ErrorCode::invalid_state);
    record_or_terminate();
    state_.phase = Phase::pending;
    return Status::accepted();
  }

  [[nodiscard]] Status conclude() noexcept { return advance<false>(); }

  // Explicit setup/injection completion for the host-published owner prefix.
  [[nodiscard]] Status wait() noexcept { return advance<true>(); }

  // Drain all submitted suffix writers, then discard the range. The old live
  // prefix and its generation are preserved; the suffix may be overwritten by
  // a subsequent epoch. This also retires a producer whose publish was rejected.
  [[nodiscard]] Status cancel(const executor_type producer) noexcept {
    if (!producing() && !pending())
      return Status::failure(ErrorCode::invalid_state);
    if (!same_producer(producer))
      return Status::failure(ErrorCode::invalid_argument);
    if (producing()) record_or_terminate();
    if (!traits_type::succeeded(event_.synchronize())) std::abort();
    state_.phase = Phase::idle;
    return Status::success();
  }

private:
  enum class Phase : std::uint8_t { unprepared, idle, producing, pending };

  struct State {
    owner_type *owner = nullptr;
    // Allocation identity at creation; the epoch of the current append.
    DeviceParticleOwnerStamp stamp{};
    std::uint32_t first = 0;
    std::uint32_t count = 0;
    executor_type producer{traits_type::null_stream};
    Phase phase = Phase::unprepared;
  };

  DeviceParticleInitializer(owner_type &owner,
                            device::DeviceEvent<Backend> &&event) noexcept
      : event_(std::move(event)),
        state_{&owner, DeviceParticleOwnerStamp::capture(owner), 0, 0,
               executor_type{traits_type::null_stream}, Phase::idle} {}

  [[nodiscard]] bool epoch_matches() const noexcept {
    return state_.stamp.matches_epoch(*state_.owner) &&
           state_.owner->size() == state_.first;
  }
  [[nodiscard]] bool same_producer(const executor_type producer) const noexcept {
    return producer.bound() && producer.stream() == state_.producer.stream();
  }
  void record_or_terminate() noexcept {
    if (!traits_type::succeeded(event_.record(state_.producer))) std::abort();
  }

  template <bool Wait> [[nodiscard]] Status advance() noexcept {
    if (!pending()) return Status::failure(ErrorCode::invalid_state);
    const Status completed = event_.template advance<Wait>();
    if constexpr (!Wait) {
      if (completed.is_pending()) return completed;
    }
    if (completed.is_failure()) std::abort();
    state_.phase = Phase::idle; // Retired even on stale-owner rejection.
    if (!epoch_matches()) return Status::failure(ErrorCode::invalid_state);
    if (state_.count != 0) {
      owner_type &owner = *state_.owner;
      const auto error =
          detail::DeviceParticlesAccess::advance_reorder_generation(owner);
      if (error != ErrorCode::success) return Status::failure(error);
      detail::DeviceParticlesAccess::publish_size(owner,
                                                  state_.first + state_.count);
    }
    return Status::success();
  }

  device::DeviceEvent<Backend> event_{};
  State state_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSchemaType Schema,
          device::DeviceAllocationBackend Backend>
  requires device::DeviceEventBackend<Backend>
[[nodiscard]] Result<DeviceParticleInitializer<Schema, Backend>>
make_device_particle_initializer(
    Particles<Schema, DeviceSpace<Backend>> &owner) noexcept {
  using Initializer = DeviceParticleInitializer<Schema, Backend>;
  if (!owner.valid())
    return Result<Initializer>::from_error(ErrorCode::invalid_argument);
  device::DeviceEvent<Backend> event{};
  if (!device::BackendTraits<Backend>::succeeded(
          device::DeviceEvent<Backend>::create(event)))
    return Result<Initializer>::from_error(ErrorCode::backend_failure);
  return Result<Initializer>::from_value(
      detail::PrivateConstructionAccess::create<Initializer>(owner,
                                                             std::move(event)));
}

} // namespace lpfx

#endif
