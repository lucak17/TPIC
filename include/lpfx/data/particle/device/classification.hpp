#ifndef LPFX_DATA_PARTICLE_DEVICE_CLASSIFICATION_HPP
#define LPFX_DATA_PARTICLE_DEVICE_CLASSIFICATION_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {
namespace detail {
struct DeviceParticleClassificationSlotsAccess;
template <device::DeviceBackend Backend>
struct DeviceParticleRegisteredClassificationAtomics;
} // namespace detail

// Counter-registered slots pack bucket/sub-bucket/ordinal in 5/4/55 bits.
// Classification and registration happen together in the producer; later
// readers decode only this stored disposition and ordinal, without replacing
// the word.
struct PackedDeviceParticleClassificationSlot final {
  static constexpr std::uint32_t maximum_peer_count = 26;
  static constexpr std::uint32_t sub_bucket_count = 16;
  static constexpr std::uint32_t main_bucket_count = maximum_peer_count + 2;
  static constexpr std::uint32_t resident_bucket = maximum_peer_count;
  static constexpr std::uint32_t discard_bucket = maximum_peer_count + 1;

  // Preconditions: decision is resident (255), discard (254), or peer < 26;
  // sub_bucket < 16 and ordinal < 2^55. Device owners bound ordinals to uint32.
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr std::uint64_t pack(
      const std::uint32_t decision, const std::uint32_t sub_bucket,
      const std::uint64_t ordinal) noexcept {
    // Byte wrap maps discard 254 to 0, resident 255 to 1 and peers to [2,27].
    const std::uint32_t bucket = (decision + 2) & UINT8_MAX;
    const std::uint32_t low_bits = bucket | (sub_bucket << 5);
    return static_cast<std::uint64_t>(low_bits) | (ordinal << 9);
  }
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr std::uint8_t decision(
      const std::uint64_t word) noexcept {
    const auto bucket = static_cast<std::uint8_t>(word & UINT64_C(31));
    return bucket == 0 ? UINT8_MAX - 1 : bucket == 1 ? UINT8_MAX : bucket - 2;
  }
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr std::uint32_t sub_bucket(
      const std::uint64_t word) noexcept {
    return static_cast<std::uint32_t>((word >> 5) & UINT64_C(15));
  }
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr std::uint64_t ordinal(
      const std::uint64_t word) noexcept { return word >> 9; }
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr bool retained(
      const std::uint64_t word) noexcept { return (word & UINT64_C(31)) == 1; }
};

struct DeviceParticleRegisteredClassificationSummary final {
  std::uint32_t counts[PackedDeviceParticleClassificationSlot::main_bucket_count]
                      [PackedDeviceParticleClassificationSlot::sub_bucket_count]{};
  std::uint32_t failure{};
};

// Admission computes exclusive per-peer sub-bucket prefixes in this separate,
// fixed-size metadata image. Entry [bucket][16] is its admitted total; the
// resident row also supplies final received-record destinations.
struct DeviceParticleRegisteredClassificationData final {
  DeviceParticleRegisteredClassificationSummary summary{};
  std::uint32_t offsets[(PackedDeviceParticleClassificationSlot::maximum_peer_count + 1) *
                       (PackedDeviceParticleClassificationSlot::sub_bucket_count + 1)]{};
};

template <device::DeviceBackend Backend>
class DeviceParticleRegisteredClassificationView final {
public:
  using value_type = std::uint64_t;
  using backend_type = Backend;
  using slot_type = PackedDeviceParticleClassificationSlot;

  constexpr DeviceParticleRegisteredClassificationView() noexcept = default;
  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool valid() const noexcept {
    return summary_ != nullptr && (data_ != nullptr || size_ == 0) &&
           peer_count_ <= slot_type::maximum_peer_count;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint32_t size() const noexcept {
    return size_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t peer_count() const noexcept {
    return peer_count_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint64_t *data() const noexcept {
    return data_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr
  DeviceParticleRegisteredClassificationSummary *summary() const noexcept {
    return summary_;
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint32_t *offsets() const noexcept {
    return offsets_;
  }
  [[nodiscard]] LPFX_DEVICE constexpr std::uint64_t
  operator[](const std::uint32_t index) const noexcept { return data_[index]; }
  [[nodiscard]] LPFX_DEVICE bool invalidate() const noexcept {
    return valid() ? fail(1) : false;
  }

  // Invoke exactly once per source index, on the prepare_registered stream.
  // The caller supplies its final routing decision here, while mover state is
  // already in registers. A unique sub-bucket ordinal is assigned only once.
  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE bool register_particle(
      const std::uint32_t index, const std::uint32_t decision) const noexcept {
    if (!valid()) return false;
    if (index >= size_) return fail(1);
    const bool resident = decision == UINT8_MAX;
    const bool discarded = decision == UINT8_MAX - 1;
    if (!resident && !discarded && decision >= peer_count_) return fail(2);
    const std::uint32_t bucket = resident ? slot_type::resident_bucket :
        discarded ? slot_type::discard_bucket : decision;
    const auto sub = index & (slot_type::sub_bucket_count - 1);
#if LPFX_DEVICE_PASS && !defined(LPFX_TEST_DEVICE_HOST_EMULATION)
  const auto ordinal = detail::DeviceParticleRegisteredClassificationAtomics<
    Backend>::fetch_add(&summary_->counts[bucket][sub], 1U);
#else
    const auto ordinal = summary_->counts[bucket][sub]++;
#endif
    if (ordinal >= size_) return fail(3);
    data_[index] = slot_type::pack(decision, sub, ordinal);
    return true;
  }

private:
  constexpr DeviceParticleRegisteredClassificationView(
      std::uint64_t *data, const std::uint32_t size,
      const std::uint8_t peer_count,
      DeviceParticleRegisteredClassificationData *metadata) noexcept
      : data_(data), size_(size), peer_count_(peer_count),
        summary_(&metadata->summary), offsets_(metadata->offsets) {}

  [[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE bool
  fail(const std::uint32_t failure) const noexcept {
#if LPFX_DEVICE_PASS && !defined(LPFX_TEST_DEVICE_HOST_EMULATION)
  detail::DeviceParticleRegisteredClassificationAtomics<Backend>::
    store_first(&summary_->failure, failure);
#else
    if (summary_->failure == 0) summary_->failure = failure;
#endif
    return false;
  }

  std::uint64_t *data_ = nullptr;
  std::uint32_t size_ = 0;
  std::uint8_t peer_count_ = UINT8_MAX;
  DeviceParticleRegisteredClassificationSummary *summary_ = nullptr;
  std::uint32_t *offsets_ = nullptr;

  friend struct detail::PrivateConstructionAccess;
};

// Separate scratch: it never follows particle permutation or enters the schema.
// Keep this owner at its address while a binding borrows it. The producer must
// register each source once before publish(); no old values survive
// growth. conclude() of the consuming migration releases the reader lease.
template <device::DeviceBackend Backend>
class DeviceParticleClassificationSlots final {
public:
  using backend_type = Backend;
  using traits_type = device::BackendTraits<Backend>;
  using executor_type = device::Executor<Backend>;
  using registered_view_type = DeviceParticleRegisteredClassificationView<Backend>;
  static_assert(traits_type::minimum_device_allocation_alignment >=
                alignof(std::uint64_t));

  DeviceParticleClassificationSlots() noexcept = default;
  LPFX_NONMOVABLE(DeviceParticleClassificationSlots);
  ~DeviceParticleClassificationSlots() noexcept {
    if (!reusable())
      std::abort();
  }

  // Reserve the complete producer workspace while no epoch borrows it. Refresh
  // this reservation after the main particle capacity grows, before admitting
  // another mover. No existing classification values survive slot growth.
  [[nodiscard]] ErrorCode reserve(const std::uint32_t capacity) noexcept {
    LPFX_TRY(reserve_storage<false>(capacity));
    if (producer_ready_.empty() && !traits_type::succeeded(
            device::DeviceEvent<Backend>::create(producer_ready_)))
      return ErrorCode::backend_failure;
    return ErrorCode::success;
  }

  // Counter registration occurs in the mover itself. The metadata clear and
  // all register_particle calls must precede publish on this same stream. The
  // complete workspace must already be reserved: admission never allocates or
  // creates an event, including for an empty source.
  [[nodiscard]] Result<registered_view_type> prepare_registered(
      const std::uint32_t count, const std::uint8_t peer_count,
      const executor_type producer) noexcept {
    using Output = Result<registered_view_type>;
    if (!reusable())
      return Output::from_error(ErrorCode::invalid_state);
    LPFX_TRY_RESULT(begin(count, peer_count, producer, State::producing, true),
                    Output);
    return Output::from_value(view());
  }

  template <class Schema>
  [[nodiscard]] Result<registered_view_type> prepare_registered(
      const Particles<Schema, DeviceSpace<Backend>> &source,
      const std::uint8_t peer_count, const executor_type producer) noexcept {
    if (!source.valid())
      return Result<registered_view_type>::from_error(ErrorCode::invalid_argument);
    auto result = prepare_registered(source.size(), peer_count, producer);
    if (result.has_value())
      source_ = DeviceParticleOwnerStamp::capture(source);
    return result;
  }

  [[nodiscard]] ErrorCode publish(const executor_type producer) noexcept {
    if (state_ != State::producing)
      return ErrorCode::invalid_state;
    if (!producer.bound())
      return ErrorCode::invalid_argument;
    if (producer.stream() != registration_stream_)
      return ErrorCode::invalid_argument;
    if (!traits_type::succeeded(producer_ready_.record(producer)))
      std::abort();
    state_ = State::ready;
    return ErrorCode::success;
  }

  // Discard an unconsumed batch only after joining its producer. For an
  // unpublished batch, producer must be the stream that wrote the granted view.
  [[nodiscard]] ErrorCode cancel(const executor_type producer) noexcept {
    if (state_ != State::producing && state_ != State::ready)
      return ErrorCode::invalid_state;
    if (state_ == State::producing &&
        (!producer.bound() || producer.stream() != registration_stream_))
      return ErrorCode::invalid_argument;
    if (state_ == State::producing &&
        !traits_type::succeeded(producer_ready_.record(producer)))
      std::abort();
    if (!traits_type::succeeded(producer_ready_.synchronize()))
      std::abort();
    retire();
    return ErrorCode::success;
  }

  [[nodiscard]] bool reusable() const noexcept { return state_ == State::idle; }
  [[nodiscard]] std::uint32_t size() const noexcept { return size_; }
  [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t capacity_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return allocation_.capacity_bytes() + registration_allocation_.capacity_bytes();
  }
  // Every admitted producer or reader owns registered metadata. The lifecycle
  // state is the single authority for that lease, including empty batches.
  [[nodiscard]] bool registered() const noexcept { return state_ != State::idle; }
  // Borrowed device metadata; producer writes and admission prefix writes are
  // valid only under their leases, ordered by the publication event.
  [[nodiscard]] DeviceParticleRegisteredClassificationSummary *
  registered_summary() const noexcept {
    if (registration_allocation_.empty())
      return nullptr;
    return &reinterpret_cast<DeviceParticleRegisteredClassificationData *>(
                const_cast<std::byte *>(registration_allocation_.raw_base()))
                ->summary;
  }
  // Borrowed device pointer; mutation is allowed only under the granted lease.
  [[nodiscard]] std::uint64_t *data() const noexcept {
    return reinterpret_cast<std::uint64_t *>(
        const_cast<std::byte *>(allocation_.raw_base()));
  }

private:
  enum class State : std::uint8_t { idle, producing, ready, reading };

  // Registration metadata plus slots for required sources: the configured
  // particle capacity, or exactly required when ExactCapacity.
  template <bool ExactCapacity>
  [[nodiscard]] ErrorCode reserve_storage(const std::uint32_t required) noexcept {
    if (!reusable())
      return ErrorCode::invalid_state;
    if (registration_allocation_.empty() &&
        !traits_type::succeeded(device::DeviceAllocation<Backend>::create(
            sizeof(DeviceParticleRegisteredClassificationData),
            registration_allocation_)))
      return ErrorCode::allocation_failed;
    if (required <= capacity_)
      return ErrorCode::success;
    std::uint32_t capacity = required;
    if constexpr (!ExactCapacity) {
      LPFX_TRY(try_prepare_particle_capacity(required, capacity));
    }
    std::size_t bytes{};
    if (!detail::checked_multiply(static_cast<std::size_t>(capacity),
                                  sizeof(std::uint64_t), bytes))
      return ErrorCode::overflow;
    device::DeviceAllocation<Backend> candidate{};
    if (!traits_type::succeeded(
            device::DeviceAllocation<Backend>::create(bytes, candidate)))
      return ErrorCode::allocation_failed;
    allocation_ = std::move(candidate);
    capacity_ = capacity;
    return ErrorCode::success;
  }

  // Opens a producing or reading batch of count sources on the reserved,
  // reusable workspace. clear zeroes the summary counters on the stream; if
  // that submission fails the batch is retired again.
  [[nodiscard]] ErrorCode begin(const std::uint32_t count,
                                const std::uint8_t peers,
                                const executor_type executor, const State phase,
                                const bool clear) noexcept {
    if (!executor.bound() ||
        peers > PackedDeviceParticleClassificationSlot::maximum_peer_count)
      return ErrorCode::invalid_argument;
    if (registration_allocation_.empty() ||
        (phase == State::producing && producer_ready_.empty()))
      return ErrorCode::invalid_state;
    if (count > capacity_)
      return ErrorCode::capacity_exceeded;
    size_ = count;
    source_ = {};
    registration_context_ = nullptr;
    registration_peer_count_ = peers;
    registration_stream_ = executor.stream();
    state_ = phase;
    if (clear && !traits_type::succeeded(traits_type::memset_device_async(
                     &metadata()->summary, 0,
                     sizeof(DeviceParticleRegisteredClassificationSummary),
                     executor.stream()))) {
      if (phase == State::producing) {
        if (cancel(executor) != ErrorCode::success) std::abort();
      } else {
        retire();
      }
      return ErrorCode::backend_failure;
    }
    return ErrorCode::success;
  }

  void retire() noexcept {
    state_ = State::idle;
    size_ = 0;
    registration_context_ = nullptr;
  }

  [[nodiscard]] DeviceParticleRegisteredClassificationData *
  metadata() noexcept {
    return reinterpret_cast<DeviceParticleRegisteredClassificationData *>(
        registration_allocation_.raw_base());
  }

  [[nodiscard]] registered_view_type view() noexcept {
    return detail::PrivateConstructionAccess::create<registered_view_type>(
        data(), size_, registration_peer_count_, metadata());
  }

  device::DeviceAllocation<Backend> allocation_{};
  device::DeviceAllocation<Backend> registration_allocation_{};
  device::DeviceEvent<Backend> producer_ready_{};
  DeviceParticleOwnerStamp source_{};
  const void *registration_context_ = nullptr;
  typename traits_type::stream_type registration_stream_ = traits_type::null_stream;
  std::uint32_t capacity_ = 0;
  std::uint32_t size_ = 0;
  std::uint8_t registration_peer_count_ = UINT8_MAX;
  State state_ = State::idle;
  friend struct detail::DeviceParticleClassificationSlotsAccess;
};

namespace detail {
struct DeviceParticleClassificationSlotsAccess final {
  template <bool ExactCapacity = false, class Backend>
  [[nodiscard]] static ErrorCode reserve_internal(
      DeviceParticleClassificationSlots<Backend> &slots,
      const std::uint32_t capacity) noexcept {
    return slots.template reserve_storage<ExactCapacity>(capacity);
  }
  // Cold collective reservation publishes only after every rank admitted it.
  // These are idle receive-only slots: no producer event or external capability
  // may borrow either allocation. The temporary then destroys the old storage.
  template <class Backend>
  static void swap_received_reservation(
      DeviceParticleClassificationSlots<Backend> &slots,
      DeviceParticleClassificationSlots<Backend> &candidate) noexcept {
    if (!slots.reusable() || !candidate.reusable() ||
        !slots.producer_ready_.empty() ||
        !candidate.producer_ready_.empty())
      std::abort();
    using std::swap;
    swap(slots.allocation_, candidate.allocation_);
    swap(slots.registration_allocation_, candidate.registration_allocation_);
    swap(slots.capacity_, candidate.capacity_);
  }
  // Received routing batches have no main-particle owner. This internal path
  // is ordered by the routing stream and owns no producer-publication event.
  // Nonempty received batches initialize counters once. Empty received batches
  // publish a host summary without reading metadata; initial admission still
  // reads the counters for an empty source and requests the clear explicitly.
  template <bool ClearEmpty = false, class Backend>
  [[nodiscard]] static ErrorCode prepare_received(
      DeviceParticleClassificationSlots<Backend> &slots,
      const std::uint32_t count,
      const std::uint8_t peers, const device::Executor<Backend> executor) noexcept {
    using Slots = DeviceParticleClassificationSlots<Backend>;
    if (!executor.bound() ||
        peers > PackedDeviceParticleClassificationSlot::maximum_peer_count)
      return ErrorCode::invalid_argument;
    LPFX_TRY(slots.template reserve_storage<false>(count));
    return slots.begin(count, peers, executor, Slots::State::reading,
                       ClearEmpty || count != 0);
  }
  template <class Backend, class Schema>
  [[nodiscard]] static ErrorCode prepare_internal(
      DeviceParticleClassificationSlots<Backend> &slots,
      const Particles<Schema, DeviceSpace<Backend>> &source,
      const std::uint8_t peers, const device::Executor<Backend> executor) noexcept {
    if (!source.valid()) return ErrorCode::invalid_argument;
    LPFX_TRY(prepare_received<true>(slots, source.size(), peers, executor));
    slots.source_ = DeviceParticleOwnerStamp::capture(source);
    return ErrorCode::success;
  }
  template <class Backend>
  [[nodiscard]] static DeviceParticleRegisteredClassificationView<Backend>
  registered_view(DeviceParticleClassificationSlots<Backend> &slots) noexcept {
    using Slots = DeviceParticleClassificationSlots<Backend>;
    if (slots.state_ != Slots::State::producing &&
        slots.state_ != Slots::State::reading) std::abort();
    return slots.view();
  }

  template <class Backend>
  [[nodiscard]] static ErrorCode set_registration_context(
      DeviceParticleClassificationSlots<Backend> &slots,
      const void *const context) noexcept {
    using Slots = DeviceParticleClassificationSlots<Backend>;
    if (slots.state_ != Slots::State::producing ||
        context == nullptr)
      return ErrorCode::invalid_state;
    slots.registration_context_ = context;
    return ErrorCode::success;
  }
  template <class Backend>
  [[nodiscard]] static bool registration_matches(
      const DeviceParticleClassificationSlots<Backend> &slots,
      const void *const context, const std::uint8_t peer_count) noexcept {
    return slots.registered() && context != nullptr &&
           slots.registration_context_ == context &&
           slots.registration_peer_count_ == peer_count;
  }
  template <class Backend, class Schema>
  [[nodiscard]] static ErrorCode acquire(
      DeviceParticleClassificationSlots<Backend> &slots,
      const Particles<Schema, DeviceSpace<Backend>> &source,
      const device::Executor<Backend> executor) noexcept {
    if (!slots.source_.matches_epoch(source))
      return ErrorCode::invalid_state;
    return acquire(slots, source.size(), executor);
  }
  template <class Backend>
  [[nodiscard]] static ErrorCode acquire(
      DeviceParticleClassificationSlots<Backend> &slots,
      const std::uint32_t count,
      const device::Executor<Backend> executor) noexcept {
    using Slots = DeviceParticleClassificationSlots<Backend>;
    if (slots.state_ != Slots::State::ready || slots.size_ != count)
      return ErrorCode::invalid_state;
    if (!executor.bound())
      return ErrorCode::invalid_argument;
    if (!Slots::traits_type::succeeded(slots.producer_ready_.wait_on(executor)))
      std::abort();
    slots.state_ = Slots::State::reading;
    return ErrorCode::success;
  }
  template <class Backend>
  static void release(DeviceParticleClassificationSlots<Backend> &slots) noexcept {
    using Slots = DeviceParticleClassificationSlots<Backend>;
    if (slots.state_ != Slots::State::reading) std::abort();
    slots.retire();
  }
};
} // namespace detail
} // namespace lpfx
#endif
