#ifndef LPFX_DISTRIBUTED_TRANSPORT_FIXED_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_FIXED_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <type_traits>
#include <utility>

#include <lpfx/backend/launch.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/detail/fixed_byte.hpp>
#include <lpfx/distributed/transport/stage.hpp>
#include <lpfx/distributed/transport/workspace.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

// Native access to the family image without staging: host memory for MPI
// (Backend void) or device memory that the native library reads directly
// (XCCL, Backend is its device backend).
template <class Executor, class Backend = void> struct HostFixedStage {
  using executor_type = Executor;
  using device_backend_type = Backend;
  static constexpr bool has_pre_native = false;
  static constexpr bool has_post_native = false;

  [[nodiscard]] static constexpr ErrorCode try_prepare_resources(
      HostMemoryBudget &, const FixedByteExchange &, const Executor &) noexcept {
    return ErrorCode::success;
  }
  template <class Byte>
  [[nodiscard]] static constexpr Byte *native_address(Byte *data) noexcept {
    return data;
  }
  [[nodiscard]] static constexpr std::size_t workspace_bytes() noexcept {
    return 0;
  }
  static constexpr void reset() noexcept {}
};

// Device memory that the native library accesses on the executor's stream
// (XCCL), completed by one event recorded after the native work. An
// asynchronous apply on the same stream records its own event instead, which
// covers the transfer.
template <class Executor, class Backend>
class EventFixedStage final : public HostFixedStage<Executor, Backend>,
                              public EventCompletion<Backend> {
  using Completion = EventCompletion<Backend>;

public:
  using Completion::has_post_native;
  using Completion::reset;

  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &, const FixedByteExchange &, const Executor &) noexcept {
    return this->create_completion();
  }
  [[nodiscard]] static constexpr ErrorCode
  enqueue_post_on_apply_stream(const Executor &) noexcept {
    return ErrorCode::success;
  }
};

// Device memory handed to MPI. Pre-access records one event after the
// producers and polls it before the ordering hook. Sparse (fixed message)
// images may declare unused incoming capacity, so post-write ordering keys on
// the native receives; dense field images key on the same prepared image that
// pre-access tests.
template <class Ordering, class Backend,
          class Executor = device::Executor<Backend>, bool Sparse = false>
  requires device::DeviceStreamBackend<Backend> &&
           device::DeviceEventBackend<Backend>
class DirectFixedStage final {
  using ReadView = BufferView<const std::byte, DeviceSpace<Backend>>;
  using WriteView = BufferView<std::byte, DeviceSpace<Backend>>;
  static_assert(DirectMpiOrdering<Ordering, Executor, ReadView, WriteView>);

public:
  using executor_type = Executor;
  using device_backend_type = Backend;
  static constexpr bool has_pre_native = true;
  static constexpr bool has_post_native =
      MpiPostWriteOrdering<Ordering, Executor, WriteView>;

  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &, const FixedByteExchange &shape,
      const Executor &) noexcept {
    outgoing_ = shape.outgoing;
    incoming_ = shape.incoming;
    return device::to_error_code<Backend>(
        device::DeviceEvent<Backend>::create(ready_));
  }
  template <class Byte>
  [[nodiscard]] static constexpr Byte *native_address(Byte *data) noexcept {
    return data;
  }
  [[nodiscard]] Status start_pre(const Executor &executor) noexcept {
    if (outgoing_.empty() && incoming_.empty()) {
      order_access(executor);
      return Status::success();
    }
    if (!executor.bound() || ready_.empty()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if (!device::BackendTraits<Backend>::succeeded(ready_.record(executor))) {
      return Status::failure(ErrorCode::backend_failure);
    }
    return Status::pending();
  }
  // has_incoming_work: native receives wrote bytes.
  [[nodiscard]] Status start_post(const Executor &executor,
                                  bool has_incoming_work) noexcept {
    if constexpr (!Sparse) {
      has_incoming_work = !incoming_.empty();
    }
    const auto error = [&] {
      if constexpr (std::same_as<Ordering, StreamSynchronizedMpiOrdering>) {
        return Ordering::after_mpi_write(executor, has_incoming_work);
      } else {
        return has_incoming_work
                   ? Ordering::after_mpi_write(
                         executor, device_byte_view<Backend>(
                                       incoming_.data, incoming_.size_bytes))
                   : ErrorCode::success;
      }
    }();
    if (error != ErrorCode::success) {
      std::abort();
    }
    return Status::success();
  }
  // Only accepted generations advance; errors cannot leave peers waiting
  // while the owner silently retains an unfinished buffer lease.
  template <bool Wait>
  [[nodiscard]] Status advance_pre(const Executor &executor) noexcept {
    const auto ready = advance_or_abort<Wait>(ready_);
    if (ready.is_success()) {
      order_access(executor);
    }
    return ready;
  }
  template <bool Wait> [[nodiscard]] static Status advance_post() noexcept {
    static_cast<void>(Wait);
    std::abort();
  }
  [[nodiscard]] static constexpr std::size_t workspace_bytes() noexcept {
    return 0;
  }
  void reset() noexcept {
    ready_ = {};
    outgoing_ = {};
    incoming_ = {};
  }

private:
  void order_access(const Executor &executor) noexcept {
    if (Ordering::before_mpi_access(
            executor,
            device_byte_view<Backend>(outgoing_.data, outgoing_.size_bytes),
            device_byte_view<Backend>(incoming_.data, incoming_.size_bytes)) !=
        ErrorCode::success) {
      std::abort();
    }
  }

  ByteRegion<const std::byte> outgoing_{};
  ByteRegion<std::byte> incoming_{};
  device::DeviceEvent<Backend> ready_{};
};

struct FixedByteStagingInterval final {
  std::size_t device_offset = 0;
  std::size_t host_offset = 0;
  std::size_t bytes = 0;
};

// Only fixed messages admit sparse aggregate images. Fields use the empty
// specialization.
template <bool Sparse> struct FixedByteStagingIntervals final {};
template <> struct FixedByteStagingIntervals<true> final {
  HostWorkspace<FixedByteStagingInterval> storage{};
  std::span<const FixedByteStagingInterval> outgoing{}, incoming{};

  template <class Message, class Region>
  [[nodiscard]] static bool contiguous(const std::span<const Message> messages,
                                       const Region region,
                                       Region &compact) noexcept {
    compact = {};
    const auto base = reinterpret_cast<std::uintptr_t>(region.data);
    std::size_t offset = 0;
    for (const auto &message : messages) {
      if (message.size_bytes == 0) continue;
      if (compact.data == nullptr) {
        const auto first = reinterpret_cast<std::uintptr_t>(message.data);
        if (first < base || first - base > region.size_bytes) return false;
        offset = first - base;
        compact.data = message.data;
      }
      if (reinterpret_cast<std::uintptr_t>(message.data) != base + offset ||
          message.size_bytes > region.size_bytes - offset)
        return false;
      offset += message.size_bytes;
      compact.size_bytes += message.size_bytes;
    }
    return true;
  }

  template <class Message, class Region>
  [[nodiscard]] static ErrorCode coalesce(
      const std::span<const Message> messages, const Region region,
      const std::span<FixedByteStagingInterval> scratch, Region &compact,
      std::span<const FixedByteStagingInterval> &intervals) noexcept {
    const auto base = reinterpret_cast<std::uintptr_t>(region.data);
    std::size_t occupied = 0;
    for (const auto &message : messages) {
      if (message.size_bytes == 0) continue;
      const auto address = reinterpret_cast<std::uintptr_t>(message.data);
      if (address < base || address - base > region.size_bytes ||
          message.size_bytes > region.size_bytes - (address - base))
        return ErrorCode::invalid_argument;
      scratch[occupied++] = {address - base, 0, message.size_bytes};
    }
    const auto covered = scratch.first(occupied);
    std::sort(covered.begin(), covered.end(), [](const auto &left, const auto &right) {
      return left.device_offset < right.device_offset;
    });
    std::size_t count = 0;
    for (const auto interval : covered) {
      if (count != 0) {
        auto &previous = scratch[count - 1];
        const auto end = previous.device_offset + previous.bytes;
        if (interval.device_offset <= end) {
          previous.bytes = std::max(end, interval.device_offset + interval.bytes) -
                           previous.device_offset;
          continue;
        }
      }
      scratch[count++] = interval;
    }
    intervals = {};
    if (count == 0) {
      compact = {};
      return ErrorCode::success;
    }
    if (count == 1) {
      compact = {region.data + scratch[0].device_offset, scratch[0].bytes};
      return ErrorCode::success;
    }
    std::size_t bytes = 0;
    for (auto &interval : scratch.first(count)) {
      interval.host_offset = bytes;
      // Disjoint validated subranges fit inside the aggregate image.
      bytes += interval.bytes;
    }
    compact = {region.data, bytes};
    intervals = scratch.first(count);
    return ErrorCode::success;
  }

  [[nodiscard]] ErrorCode prepare(HostMemoryBudget &budget,
      const FixedByteExchange &shape, ByteRegion<const std::byte> &send,
      ByteRegion<std::byte> &receive) noexcept {
    const bool dense_send = contiguous(shape.sends, shape.outgoing, send);
    const bool dense_receive = contiguous(shape.receives, shape.incoming, receive);
    if (dense_send && dense_receive) return ErrorCode::success;
    const std::size_t sends = dense_send ? 0 : shape.sends.size();
    const std::size_t receives = dense_receive ? 0 : shape.receives.size();
    std::size_t count{};
    if (!lpfx::detail::checked_add(sends, receives, count))
      return ErrorCode::overflow;
    LPFX_TRY(storage.try_prepare(budget, {count}));
    const auto scratch = storage.span<0>();
    ErrorCode error = ErrorCode::success;
    if (!dense_send)
      error = coalesce(shape.sends, shape.outgoing, scratch.first(sends), send, outgoing);
    if (error == ErrorCode::success && !dense_receive)
      error = coalesce(shape.receives, shape.incoming, scratch.subspan(sends), receive, incoming);
    if (error != ErrorCode::success || (outgoing.empty() && incoming.empty())) {
      outgoing = {};
      incoming = {};
      storage.reset();
    }
    return error;
  }

  template <class Byte>
  [[nodiscard]] static Byte *native_address(Byte *data, Byte *device,
      Byte *host, const std::span<const FixedByteStagingInterval> intervals) noexcept {
    const auto offset = static_cast<std::size_t>(data - device);
    for (const auto &interval : intervals)
      if (offset >= interval.device_offset && offset - interval.device_offset < interval.bytes)
        return host + interval.host_offset + (offset - interval.device_offset);
    std::abort(); // Every native recipe was covered during cold preparation.
  }
};

// Device memory staged through pinned host images; the event completes the
// download before MPI and the upload after it.
template <class Backend, class Executor = device::Executor<Backend>,
          bool Sparse = false>
  requires device::PinnedHostAllocationBackend<Backend> &&
           device::DeviceCopyBackend<Backend> &&
           device::DeviceEventBackend<Backend>
class StagedFixedStage final {
  using Traits = device::BackendTraits<Backend>;

public:
  using executor_type = Executor;
  using device_backend_type = Backend;
  static constexpr bool has_pre_native = true;
  static constexpr bool has_post_native = true;

  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &budget, const FixedByteExchange &shape,
      const Executor &) noexcept {
    ByteRegion<const std::byte> send = shape.outgoing;
    ByteRegion<std::byte> receive = shape.incoming;
    if constexpr (Sparse) {
      LPFX_TRY(intervals_.prepare(budget, shape, send, receive));
    }
    const ErrorCode prepared = images_.try_prepare(
        budget, send.size_bytes, receive.size_bytes, alignof(std::byte));
    if (prepared != ErrorCode::success) {
      if constexpr (Sparse) intervals_ = {};
      return prepared;
    }
    static_cast<void>(lpfx::detail::start_host_array_lifetime<std::byte>(
        images_.first(), images_.first_bytes()));
    static_cast<void>(lpfx::detail::start_host_array_lifetime<std::byte>(
        images_.second(), images_.second_bytes()));
    outgoing_device_ = send.data;
    incoming_device_ = receive.data;
    return ErrorCode::success;
  }

  [[nodiscard]] const std::byte *
  native_address(const std::byte *data) const noexcept {
    if constexpr (Sparse)
      if (!intervals_.outgoing.empty())
        return intervals_.template native_address<const std::byte>(
            data, outgoing_device_, images_.first(), intervals_.outgoing);
    return images_.first() + (data - outgoing_device_);
  }
  [[nodiscard]] std::byte *native_address(std::byte *data) noexcept {
    if constexpr (Sparse)
      if (!intervals_.incoming.empty())
        return intervals_.template native_address<std::byte>(
            data, incoming_device_, images_.second(), intervals_.incoming);
    return images_.second() + (data - incoming_device_);
  }
  [[nodiscard]] Status start_pre(const Executor &executor) noexcept {
    if (images_.first_bytes() == 0) {
      return Status::success();
    }
    if constexpr (Sparse) {
      if (!intervals_.outgoing.empty()) {
        for (const auto &interval : intervals_.outgoing)
          if (!Traits::succeeded(Traits::copy_device_to_host_async(
                  images_.first() + interval.host_offset,
                  outgoing_device_ + interval.device_offset, interval.bytes,
                  executor.stream())))
            std::abort();
        if (!Traits::succeeded(images_.event().record(executor))) std::abort();
        return Status::pending();
      }
    }
    lpfx::detail::copy_and_record<lpfx::detail::DeviceCopyDirection::to_host>(
        images_.first(), outgoing_device_, images_.first_bytes(), executor,
        images_.event());
    return Status::pending();
  }
  [[nodiscard]] Status start_post(const Executor &executor, bool) noexcept {
    return submit_post<true>(executor);
  }
  // Caller must enqueue apply and its completion event on this executor and
  // retain this port's pinned storage until that event completes. Sparse fixed
  // messages have no apply owner and retain their ordinary completion path.
  [[nodiscard]] ErrorCode
  enqueue_post_on_apply_stream(const Executor &executor) noexcept
    requires(!Sparse)
  {
    return submit_post<false>(executor).code();
  }
  template <bool Wait>
  [[nodiscard]] Status advance_pre(const Executor &) noexcept {
    return images_.template advance<Wait>();
  }
  template <bool Wait> [[nodiscard]] Status advance_post() noexcept {
    return images_.template advance<Wait>();
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    if constexpr (Sparse)
      return images_.workspace_bytes() + intervals_.storage.capacity_bytes();
    else
      return images_.workspace_bytes();
  }
  void reset() noexcept {
    outgoing_device_ = nullptr;
    incoming_device_ = nullptr;
    images_.reset();
    if constexpr (Sparse) intervals_ = {};
  }

private:
  template <bool RecordCompletion>
  [[nodiscard]] Status submit_post(const Executor &executor) noexcept {
    static_assert(RecordCompletion || !Sparse);
    if (images_.second_bytes() == 0) {
      return Status::success();
    }
    if constexpr (Sparse) {
      if (!intervals_.incoming.empty()) {
        for (const auto &interval : intervals_.incoming)
          if (!Traits::succeeded(Traits::copy_host_to_device_async(
                  incoming_device_ + interval.device_offset,
                  images_.second() + interval.host_offset, interval.bytes,
                  executor.stream())))
            std::abort();
        if (!Traits::succeeded(images_.event().record(executor))) std::abort();
        return Status::pending();
      }
    }
    if (!Traits::succeeded(Traits::copy_host_to_device_async(
            incoming_device_, images_.second(), images_.second_bytes(),
            executor.stream())))
      std::abort();
    if constexpr (RecordCompletion) {
      if (!Traits::succeeded(images_.event().record(executor))) std::abort();
      return Status::pending();
    } else {
      return Status::success();
    }
  }

  [[no_unique_address]] FixedByteStagingIntervals<Sparse> intervals_{};
  PinnedStagingPair<Backend> images_{};
  const std::byte *outgoing_device_ = nullptr;
  std::byte *incoming_device_ = nullptr;
};

// Fixed-byte family port: Stage owns buffer access and staging, P2p the
// native exchange (MpiP2p or XcclP2p).
template <class Stage, class P2p> class FixedBytePort final {
public:
  using stage_type = Stage;
  using p2p_type = P2p;
  using executor_type = typename Stage::executor_type;
  using device_backend_type = typename Stage::device_backend_type;
  static constexpr bool has_pre_native = Stage::has_pre_native;
  static constexpr bool has_post_native = Stage::has_post_native;
  // A device exchange with no post-native step leaves its readers ordered on
  // the executor's stream (stream-ordered XCCL, StreamOrderedMpiOrdering).
  static constexpr bool completes_on_stream =
      !has_post_native && !std::is_void_v<device_backend_type>;
  // The native port observes its own completion (MPI requests).
  static constexpr bool has_native_completion =
      requires(P2p &p2p) { p2p.release(); };

  [[nodiscard]] bool empty() const noexcept { return p2p_.empty(); }
  [[nodiscard]] bool is_valid() const noexcept { return p2p_.is_valid(); }

  template <class Session, class Setup>
  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &budget, Session &session, const Setup &setup,
      const FixedByteExchange &shape, const executor_type &executor,
      const std::span<const int> route_tags) noexcept {
    if (!empty() || !shape.has_native_work() || !setup.is_valid()) {
      return ErrorCode::invalid_state;
    }
    if (shape.rank != static_cast<std::uint32_t>(setup.rank()) ||
        shape.rank_count != static_cast<std::uint32_t>(setup.size())) {
      return ErrorCode::invalid_argument;
    }
    if constexpr (!std::is_void_v<device_backend_type>) {
      LPFX_TRY(setup.template try_verify_device_affinity<device_backend_type>(
          executor));
    }
    LPFX_TRY(stage_.try_prepare_resources(budget, shape, executor));
    const ErrorCode error = p2p_.try_prepare(
        budget, session, shape, route_tags,
        [this](auto *data) noexcept { return stage_.native_address(data); });
    if (error != ErrorCode::success) {
      stage_.reset();
    }
    return error;
  }

  [[nodiscard]] Status start_pre(const executor_type &executor) noexcept
    requires(has_pre_native)
  {
    return stage_.start_pre(executor);
  }
  template <bool Wait>
  [[nodiscard]] Status advance_pre(const executor_type &executor) noexcept
    requires(has_pre_native)
  {
    return stage_.template advance_pre<Wait>(executor);
  }
  [[nodiscard]] Status start_post(const executor_type &executor) noexcept
    requires(has_post_native)
  {
    return stage_.start_post(executor, p2p_.has_incoming_work());
  }
  template <bool Wait>
  [[nodiscard]] Status advance_post() noexcept
    requires(has_post_native)
  {
    return stage_.template advance_post<Wait>();
  }
  [[nodiscard]] ErrorCode
  enqueue_post_on_apply_stream(const executor_type &executor) noexcept
    requires requires(Stage &stage) {
      stage.enqueue_post_on_apply_stream(executor);
    }
  {
    return stage_.enqueue_post_on_apply_stream(executor);
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  submit(const executor_type &executor) noexcept {
    return p2p_.submit(executor);
  }
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance_native() noexcept {
    return p2p_.template advance<Wait>();
  }
  [[nodiscard]] Status release_native() noexcept { return p2p_.release(); }
  [[nodiscard]] std::size_t work_count() const noexcept {
    return p2p_.work_count();
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return stage_.workspace_bytes() + p2p_.workspace_bytes();
  }
  void reset_resources() noexcept { reset(); }
  void reset() noexcept {
    p2p_.reset();
    stage_.reset();
  }

private:
  [[no_unique_address]] Stage stage_{};
  P2p p2p_{};
};

} // namespace lpfx::distributed::detail

#endif
