#ifndef LPFX_DISTRIBUTED_TRANSPORT_COLLECTIVE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_COLLECTIVE_HPP

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/collective/port.hpp>
#include <lpfx/distributed/transport/stage.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

// Native calls read and write the family operands in place.
template <class T> struct UnstagedCollectiveStage {
  using value_type = T;

  [[nodiscard]] static constexpr bool empty() noexcept { return true; }
  [[nodiscard]] static constexpr bool
  is_valid(const CollectivePreparedShape &,
           const CollectiveLocalAccess &access) noexcept {
    return access.read.valid() &&
           access.write.valid();
  }
  [[nodiscard]] static constexpr const T *
  native_input(const CollectiveBuffers<T> buffers) noexcept {
    return buffers.input;
  }
  [[nodiscard]] static constexpr T *
  native_output(const CollectiveBuffers<T> buffers) noexcept {
    return buffers.output;
  }
  static constexpr void reset() noexcept {}
};

// Host operands for MPI (Backend void), or device operands that the native
// library accesses directly (XCCL, Backend is its device backend).
template <class T, class Executor, class Backend = void>
struct HostCollectiveStage : UnstagedCollectiveStage<T> {
  using executor_type = Executor;
  using device_backend_type = Backend;
  static constexpr bool has_pre_native = false;
  static constexpr bool has_post_native = false;

  [[nodiscard]] static constexpr ErrorCode try_prepare_resources(
      HostMemoryBudget &, const CollectivePreparedShape &shape,
      const CollectiveLocalAccess &access, const Executor &) noexcept {
    return UnstagedCollectiveStage<T>::is_valid(shape, access)
               ? ErrorCode::success
               : ErrorCode::invalid_argument;
  }
};

// Device operands that the native library accesses on the executor's stream
// (XCCL), completed by one event recorded after the native call.
template <class T, class Executor, class Backend>
class EventCollectiveStage final
    : public HostCollectiveStage<T, Executor, Backend>,
      public EventCompletion<Backend> {
  using Host = HostCollectiveStage<T, Executor, Backend>;
  using Completion = EventCompletion<Backend>;

public:
  using Completion::empty;
  using Completion::has_post_native;
  using Completion::reset;

  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &budget, const CollectivePreparedShape &shape,
      const CollectiveLocalAccess &access, const Executor &executor) noexcept {
    LPFX_TRY(Host::try_prepare_resources(budget, shape, access, executor));
    return this->create_completion();
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_post(const CollectiveLocalAccess &, const CollectiveBuffers<T>,
             const Executor &executor) noexcept {
    return Completion::start_post(executor);
  }
};

// Device operands handed to MPI. Pre-access synchronizes the stream, then
// runs the ordering hook.
template <class Ordering, class T, class Backend, class Executor>
  requires device::DeviceStreamBackend<Backend>
struct DirectCollectiveStage final : UnstagedCollectiveStage<T> {
  using executor_type = Executor;
  using device_backend_type = Backend;
  static constexpr bool has_pre_native = true;
  static constexpr bool has_post_native =
      MpiPostWriteOrdering<Ordering, Executor,
                           BufferView<std::byte, DeviceSpace<Backend>>>;
  static_assert(
      DirectMpiOrdering<Ordering, Executor,
                        BufferView<const std::byte, DeviceSpace<Backend>>,
                        BufferView<std::byte, DeviceSpace<Backend>>>);

  [[nodiscard]] static ErrorCode try_prepare_resources(
      HostMemoryBudget &, const CollectivePreparedShape &shape,
      const CollectiveLocalAccess &access, const Executor &executor) noexcept {
    return UnstagedCollectiveStage<T>::is_valid(shape, access) &&
                   executor.bound()
               ? ErrorCode::success
               : ErrorCode::invalid_argument;
  }

  [[nodiscard]] static LPFX_ALWAYS_INLINE Status
  start_pre(const CollectiveLocalAccess &access, const CollectiveBuffers<T>,
            const Executor &executor) noexcept {
    // Prove producer/prior-reader completion before either the blocking or
    // nonblocking MPI submission and before the common access hook.
    // Receive-only ranks also need protection from earlier GPU buffer use.
    using Traits = device::BackendTraits<Backend>;
    const bool has_access = (access.read.size_bytes | access.write.size_bytes) != 0;
    if (has_access &&
        (!executor.bound() ||
         !Traits::succeeded(Traits::synchronize_stream(executor.stream())))) {
      std::abort();
    }
    if (Ordering::before_mpi_access(
            executor,
            device_byte_view<Backend>(access.read.data, access.read.size_bytes),
            device_byte_view<Backend>(access.write.data,
                                      access.write.size_bytes)) !=
        ErrorCode::success) {
      std::abort();
    }
    return Status::success();
  }

  template <bool Wait> [[nodiscard]] static Status advance_pre() noexcept {
    static_cast<void>(Wait);
    std::abort();
  }

  [[nodiscard]] static Status start_post(const CollectiveLocalAccess &access,
                                         const CollectiveBuffers<T>,
                                         const Executor &executor) noexcept {
    if (access.write.empty()) {
      return Status::success();
    }
    if (Ordering::after_mpi_write(
            executor, device_byte_view<Backend>(access.write.data,
                                                access.write.size_bytes)) !=
        ErrorCode::success) {
      std::abort();
    }
    return Status::success();
  }

  template <bool Wait> [[nodiscard]] static Status advance_post() noexcept {
    static_cast<void>(Wait);
    std::abort();
  }
};

// Device operands staged through distinct pinned input/output images, so the
// native call is statically out-of-place while the device-side read/write
// regions retain the semantic alias contract.
template <collective_kind Kind, class T, class Backend, class Executor>
  requires device::PinnedHostAllocationBackend<Backend> &&
           device::DeviceCopyBackend<Backend> &&
           device::DeviceEventBackend<Backend>
class StagedCollectiveStage final {
  static constexpr bool needs_input = Kind != collective_kind::broadcast;

public:
  using value_type = T;
  using executor_type = Executor;
  using device_backend_type = Backend;
  static constexpr bool has_pre_native = true;
  static constexpr bool has_post_native = true;

  [[nodiscard]] bool empty() const noexcept {
    return images_.empty() && input_host_ == nullptr && output_host_ == nullptr;
  }

  [[nodiscard]] bool
  is_valid(const CollectivePreparedShape &shape,
           const CollectiveLocalAccess &access) const noexcept {
    std::size_t input_bytes{};
    std::size_t output_bytes{};
    return shape.has_native_work() && access.read.valid() &&
           access.write.valid() &&
           try_image_bytes(shape, input_bytes, output_bytes) &&
           images_.admitted() && images_.first_bytes() == input_bytes &&
           images_.second_bytes() == output_bytes &&
           (input_host_ != nullptr) == (input_bytes != 0) &&
           (output_host_ != nullptr) == (output_bytes != 0);
  }

  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &budget, const CollectivePreparedShape &shape,
      const CollectiveLocalAccess &access, const Executor &executor) noexcept {
    if (!empty() || !shape.has_native_work() || !executor.bound() ||
        !access.read.valid() ||
        !access.write.valid()) {
      return ErrorCode::invalid_argument;
    }
    std::size_t input_bytes{};
    std::size_t output_bytes{};
    if (!try_image_bytes(shape, input_bytes, output_bytes)) {
      return ErrorCode::overflow;
    }
    LPFX_TRY(images_.try_prepare(budget, input_bytes, output_bytes, alignof(T)));
    input_host_ = lpfx::detail::start_host_array_lifetime<T>(
        images_.first(), input_bytes / sizeof(T));
    output_host_ = lpfx::detail::start_host_array_lifetime<T>(
        images_.second(), output_bytes / sizeof(T));
    return ErrorCode::success;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_pre(const CollectiveLocalAccess &access, const CollectiveBuffers<T>,
            const Executor &executor) noexcept {
    if (access.read.empty()) {
      return Status::success();
    }
    constexpr bool reads_output = Kind == collective_kind::broadcast;
    void *const destination = reads_output ? static_cast<void *>(output_host_)
                                           : static_cast<void *>(input_host_);
    assert(destination != nullptr);
    assert(access.read.size_bytes ==
           (reads_output ? images_.second_bytes() : images_.first_bytes()));
    assert(executor.bound());
    lpfx::detail::copy_and_record<lpfx::detail::DeviceCopyDirection::to_host>(
        destination, access.read.data, access.read.size_bytes, executor,
        images_.event());
    return Status::pending();
  }

  template <bool Wait> [[nodiscard]] Status advance_pre() noexcept {
    return images_.template advance<Wait>();
  }

  [[nodiscard]] const T *native_input(const CollectiveBuffers<T>) const noexcept {
    return input_host_;
  }
  [[nodiscard]] T *native_output(const CollectiveBuffers<T>) const noexcept {
    return output_host_;
  }

  // The epoch completes once the result copy is enqueued: a rank that writes
  // the output also reads its input first, so the next epoch's device-to-host
  // event, recorded after this copy on the same stream, orders the copy before
  // the next native write of the output image. A broadcast's receiving ranks
  // read nothing and wait for the copy.
  [[nodiscard]] Status start_post(const CollectiveLocalAccess &access,
                                  const CollectiveBuffers<T>,
                                  const Executor &executor) noexcept {
    if (access.write.empty()) {
      return Status::success();
    }
    assert(output_host_ != nullptr);
    assert(access.write.size_bytes == images_.second_bytes());
    assert(executor.bound());
    lpfx::detail::copy_and_record<lpfx::detail::DeviceCopyDirection::to_device>(
        access.write.data, output_host_, access.write.size_bytes, executor,
        images_.event());
    return Kind == collective_kind::broadcast ? Status::pending()
                                              : Status::success();
  }

  template <bool Wait> [[nodiscard]] Status advance_post() noexcept {
    return images_.template advance<Wait>();
  }

  // The result in the pinned output image, valid from completion until the
  // next native call.
  [[nodiscard]] const T *staged_output() const noexcept { return output_host_; }

  void reset() noexcept {
    // The last result copy may still be queued.
    if (!images_.empty()) {
      static_cast<void>(images_.template advance<true>());
    }
    input_host_ = nullptr;
    output_host_ = nullptr;
    images_.reset();
  }

private:
  [[nodiscard]] static bool try_image_bytes(const CollectivePreparedShape &shape,
                                            std::size_t &input_bytes,
                                            std::size_t &output_bytes) noexcept {
    const bool needs_output =
        Kind != collective_kind::reduce || shape.rank == shape.root;
    input_bytes = 0;
    output_bytes = 0;
    return (!needs_input ||
            lpfx::detail::checked_byte_count<T>(shape.input_count, input_bytes)) &&
           (!needs_output || lpfx::detail::checked_byte_count<T>(
                                 shape.output_count, output_bytes));
  }

  PinnedStagingPair<Backend> images_{};
  T *input_host_ = nullptr;
  T *output_host_ = nullptr;
};

// Operand projection for Call::submit, evaluated after thread admission and
// never stored.
template <class T> struct CollectiveNativeOperands final {
  const T *input;
  T *output;
  std::size_t count;
};

template <class Stage, class Buffers> struct CollectiveOperandProjection final {
  const Stage &stage;
  Buffers &get_buffers;
  const CollectiveEpoch &epoch;

  [[nodiscard]] LPFX_ALWAYS_INLINE auto operator()() const noexcept {
    using T = typename Stage::value_type;
    const CollectiveBuffers<T> buffers = get_buffers();
    return CollectiveNativeOperands<T>{stage.native_input(buffers),
                                       stage.native_output(buffers),
                                       epoch.native_count};
  }
};

// Collective family port: Stage owns operand access and staging, Call the
// native collective (MpiCollectiveCall or XcclCollectiveCall).
template <class Stage, class Call> class CollectivePort final {
public:
  using stage_type = Stage;
  using call_type = Call;
  using value_type = typename Call::value_type;
  using executor_type = typename Stage::executor_type;
  static_assert(std::same_as<typename Stage::value_type, value_type>);

  static constexpr bool has_pre_native = Stage::has_pre_native;
  static constexpr bool has_post_native = Stage::has_post_native;
  // The native call observes its own completion (MPI requests).
  static constexpr bool has_native_completion =
      requires(Call &call) { call.release(); };

  constexpr CollectivePort() noexcept = default;
  LPFX_NONMOVABLE(CollectivePort);
  ~CollectivePort() noexcept {
    if (!empty()) {
      std::abort();
    }
  }

  [[nodiscard]] bool empty() const noexcept {
    return call_.empty() && stage_.empty();
  }

  [[nodiscard]] bool
  is_valid(const CollectivePreparedShape &shape,
           const CollectiveLocalAccess &access) const noexcept {
    return call_.is_valid(shape) && stage_.is_valid(shape, access);
  }

  template <class Setup>
  [[nodiscard]] ErrorCode
  try_validate_setup(const Setup &setup, const CollectivePreparedShape &shape,
                     const CollectiveBuffers<value_type> buffers,
                     const executor_type &executor) const noexcept {
    if (!setup.is_valid() ||
        !Call::valid_recipe(shape, setup.rank(), setup.size(), false)) {
      return ErrorCode::invalid_argument;
    }
    using DeviceBackend = typename Stage::device_backend_type;
    if constexpr (std::is_void_v<DeviceBackend>) {
      static_cast<void>(buffers);
      static_cast<void>(executor);
      return ErrorCode::success;
    } else {
      return try_validate_device_collective_affinity<DeviceBackend>(
          setup, shape, buffers, executor);
    }
  }

  template <class Session, class Setup>
  [[nodiscard]] ErrorCode
  try_prepare_resources(HostMemoryBudget &budget, Session &session,
                        const Setup &setup, const CollectivePreparedShape &shape,
                        const CollectiveLocalAccess &access,
                        const executor_type &executor) noexcept {
    if (!empty() || !setup.is_valid()) {
      return ErrorCode::invalid_state;
    }
    if (!Call::valid_recipe(shape, setup.rank(), setup.size(), true)) {
      return ErrorCode::invalid_argument;
    }
    LPFX_TRY(call_.try_prepare(session, setup, executor));
    const ErrorCode error =
        stage_.try_prepare_resources(budget, shape, access, executor);
    if (error != ErrorCode::success) {
      call_.reset();
    }
    return error;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_pre(const CollectiveLocalAccess &access,
            const CollectiveBuffers<value_type> buffers,
            const executor_type &executor) noexcept
    requires(has_pre_native)
  {
    return stage_.start_pre(access, buffers, executor);
  }

  template <bool Wait>
  [[nodiscard]] Status advance_pre() noexcept
    requires(has_pre_native)
  {
    return stage_.template advance_pre<Wait>();
  }

  template <std::uint32_t Root, class Buffers>
    requires std::is_nothrow_invocable_r_v<CollectiveBuffers<value_type>,
                                           Buffers &>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  submit(const CollectiveEpoch &epoch, Buffers &&get_buffers,
         const executor_type &executor) noexcept {
    assert(epoch.native_count > 0);
    return call_.template submit<Root>(
        CollectiveOperandProjection<Stage, std::remove_reference_t<Buffers>>{
            stage_, get_buffers, epoch},
        executor);
  }

  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance_native() noexcept {
    return call_.template advance<Wait>();
  }

  [[nodiscard]] Status start_post(const CollectiveLocalAccess &access,
                                  const CollectiveBuffers<value_type> buffers,
                                  const executor_type &executor) noexcept
    requires(has_post_native)
  {
    return stage_.start_post(access, buffers, executor);
  }

  template <bool Wait>
  [[nodiscard]] Status advance_post() noexcept
    requires(has_post_native)
  {
    return stage_.template advance_post<Wait>();
  }

  [[nodiscard]] Status release_native() noexcept { return call_.release(); }

  [[nodiscard]] const value_type *staged_output() const noexcept
    requires requires(const Stage &stage) { stage.staged_output(); }
  {
    return stage_.staged_output();
  }

  void reset_resources() noexcept {
    call_.reset();
    stage_.reset();
  }
  void reset() noexcept { reset_resources(); }

private:
  [[no_unique_address]] Call call_{};
  [[no_unique_address]] Stage stage_{};
};

} // namespace lpfx::distributed::detail

#endif
