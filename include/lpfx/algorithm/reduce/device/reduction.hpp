#ifndef LPFX_ALGORITHM_REDUCE_DEVICE_REDUCTION_HPP
#define LPFX_ALGORITHM_REDUCE_DEVICE_REDUCTION_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <utility>

#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/memory/space.hpp>
#include <lpfx/algorithm/reduce/device/workspace.hpp>
#include <lpfx/algorithm/reduce/operation.hpp>

#if LPFX_NATIVE_DEVICE_OPERATIONS
#include <lpfx/algorithm/reduce/device/cuda_hip.hpp>
#endif

namespace lpfx {

// Host publication reads the result back into pinned memory. Device
// publication writes directly into a caller-owned scalar, suitable for fixed
// collective bindings.
enum class DeviceReductionOutput : std::uint8_t { host, device };

namespace detail {

template <class Component, class Backend>
concept DeviceReductionComponent =
    std::same_as<typename Component::backend_type, Backend> &&
    std::same_as<typename Component::space_type, DeviceSpace<Backend>> &&
    std::floating_point<typename Component::value_type> &&
    std::same_as<std::remove_const_t<typename Component::element_type>,
                 typename Component::value_type> &&
    CartesianIndexPolicy<typename Component::index_policy> &&
    CartesianLocation<typename Component::location_type> &&
    requires(
        const Component component,
        const typename Component::index_policy::coordinate_type coordinate) {
      {
        component.core_data()
      } -> std::convertible_to<const typename Component::value_type *>;
      {
        component.row_data(coordinate, coordinate)
      } -> std::convertible_to<const typename Component::value_type *>;
    };

// A contiguous operand is a const pointer; a field operand is a device
// component view matching the work's location, index and launch policy.
template <class Operand, class Work, class Backend>
concept DeviceReductionOperand =
    (std::same_as<Work, PreparedDeviceReductionWork> &&
     std::is_pointer_v<Operand> &&
     std::is_const_v<std::remove_pointer_t<Operand>> &&
     ReductionInput<std::remove_cv_t<std::remove_pointer_t<Operand>>>) ||
    (is_device_field_reduction_work<Work> &&
     DeviceReductionComponent<Operand, Backend> &&
     std::same_as<typename Work::location_type,
                  typename Operand::location_type> &&
     std::same_as<typename Work::index_policy, typename Operand::index_policy>);

// Contiguous pointers are submitted as pointers to const.
template <class Operand>
using device_reduction_operand_t =
    std::conditional_t<std::is_pointer_v<Operand>,
                       const std::remove_pointer_t<Operand> *, Operand>;

template <class Operand>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr const void *
device_reduction_operand_data(const Operand operand) noexcept {
  if constexpr (std::is_pointer_v<Operand>) {
    return operand;
  } else {
    return operand.core_data();
  }
}

template <class Backend, ReductionOperation Operation, class Accumulator,
          class LaunchPolicy, class Work, class... Operand>
concept DeviceReductionLaunchable =
    DeviceReductionWorkType<Work> &&
    (sizeof...(Operand) == (Operation == ReductionOperation::dot ? 2 : 1)) &&
    (DeviceReductionOperand<Operand, Work, Backend> && ...) &&
    device_reduction_work_policy_matches<Work, LaunchPolicy> &&
    requires(const Work work, std::byte *const scratch,
             const PreparedDeviceReduction<Accumulator, LaunchPolicy> &prepared,
             const typename device::BackendTraits<Backend>::stream_type stream,
             bool &accepted, Accumulator *const final_output,
             const Operand... operands) {
      {
        DeviceReductionLaunch<Backend>::template enqueue<Operation, Accumulator,
                                                         LaunchPolicy>(
            work, scratch, prepared, stream, accepted, final_output,
            operands...)
      } noexcept
          -> std::same_as<typename device::BackendTraits<Backend>::error_type>;
    };

template <class Backend>
concept DeviceReductionBackend = device::DeviceAllocationBackend<Backend> &&
                                 device::DeviceEventBackend<Backend>;

template <class Backend>
concept HostResultReductionBackend =
    DeviceReductionBackend<Backend> &&
    device::PinnedHostAllocationBackend<Backend> &&
    device::DeviceCopyBackend<Backend>;

template <class Accumulator, class Backend, DeviceReductionOutput Output>
struct DeviceReductionPublication;

template <class Accumulator, class Backend>
struct DeviceReductionPublication<Accumulator, Backend,
                                  DeviceReductionOutput::host> {
  HostMemoryLease admission{};
  device::PinnedHostAllocation<Backend> readback{};
  Accumulator *value = nullptr;
  ReductionResult<Accumulator> result{};
};

template <class Accumulator, class Backend>
struct DeviceReductionPublication<Accumulator, Backend,
                                  DeviceReductionOutput::device> {
  Accumulator *value = nullptr;
  typename device::BackendTraits<Backend>::stream_type completion_stream{};
};

enum class DeviceReductionState : std::uint8_t {
  unprepared = 0,
  idle = 1,
  pending = 2,
  retiring = 3,
};

} // namespace detail

// One reusable asynchronous reduction engine for contiguous device ranges and
// pitched 3-D field interiors. Operation, accumulator, backend, launch policy
// and publication are compile-time state. Setup owns the partial hierarchy and
// event; only host publication owns a pinned scalar. Device publication
// borrows one aligned writable scalar, disjoint from every input and scratch,
// which must remain alive at its fixed address through all producers and
// consumers. begin() never allocates, queries device properties or
// synchronizes the host.
template <ReductionOperation Operation,
          detail::DeviceReductionAccumulator Accumulator,
          detail::DeviceReductionBackend Backend,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy =
              DefaultDeviceReductionLaunchPolicy,
          DeviceReductionOutput Output = DeviceReductionOutput::host>
  requires(Operation >= ReductionOperation::sum &&
           Operation <= ReductionOperation::dot) &&
          (Output == DeviceReductionOutput::device ||
           detail::HostResultReductionBackend<Backend>)
class DeviceReduction final {
public:
  using accumulator_type = Accumulator;
  using backend_type = Backend;
  using launch_policy = LaunchPolicy;
  using traits_type = device::BackendTraits<Backend>;
  using prepared_type = PreparedDeviceReduction<Accumulator, LaunchPolicy>;
  static constexpr DeviceReductionOutput output_kind = Output;
  using publication_type =
      detail::DeviceReductionPublication<Accumulator, Backend, Output>;

  constexpr DeviceReduction() noexcept = default;

  DeviceReduction(DeviceReduction &&other) noexcept
      : scratch_(std::move(other.scratch_)), event_(std::move(other.event_)),
        prepared_(other.prepared_), publication_(std::move(other.publication_)),
        state_(other.state_) {
    other.disarm();
  }

  DeviceReduction &operator=(DeviceReduction &&) = delete;

  ~DeviceReduction() noexcept {
    if (pending()) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return state_ != detail::DeviceReductionState::unprepared;
  }
  [[nodiscard]] constexpr bool pending() const noexcept {
    return state_ == detail::DeviceReductionState::pending ||
           state_ == detail::DeviceReductionState::retiring;
  }
  [[nodiscard]] constexpr std::size_t scratch_capacity_bytes() const noexcept {
    return scratch_.capacity_bytes();
  }
  [[nodiscard]] constexpr std::size_t
  result_readback_capacity_bytes() const noexcept {
    if constexpr (Output == DeviceReductionOutput::host) {
      return publication_.readback.capacity_bytes();
    } else {
      return 0;
    }
  }
  [[nodiscard]] constexpr const prepared_type &prepared() const noexcept {
    return prepared_;
  }

  // Hard precondition: an epoch completed successfully and none is pending.
  // Returning by value keeps a published result stable across later reuse.
  [[nodiscard]] constexpr ReductionResult<Accumulator> result() const noexcept
    requires(Output == DeviceReductionOutput::host)
  {
    return publication_.result;
  }

  // Address is available at setup for collective binding. The value is valid
  // only after the current producer, and remains borrowed through its readers.
  [[nodiscard]] constexpr const Accumulator *device_result_data() const noexcept
    requires(Output == DeviceReductionOutput::device)
  {
    return publication_.value;
  }

  // Order a consuming stream before enqueueing any use of the scalar. This is
  // not a host wait and does not release the epoch or its borrowed input.
  [[nodiscard]] Status
  wait_on(const device::Executor<Backend> consumer) const noexcept
    requires(Output == DeviceReductionOutput::device)
  {
    if (state_ != detail::DeviceReductionState::pending || !consumer.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if (publication_.completion_stream != consumer.stream() &&
        !traits_type::succeeded(event_.wait_on(consumer))) {
      std::abort();
    }
    return Status::success();
  }

  // After ordering and enqueueing ALL consumers, record their final use on this
  // stream. Join other reader streams into it first. The next begin() waits for
  // this marker on its producer stream, permitting reuse without a host wait.
  // With no consumers use the producer stream, or order this stream with
  // wait_on() first. Retirement itself adds no redundant readiness wait.
  [[nodiscard]] Status
  retire_on(const device::Executor<Backend> consumer) noexcept
    requires(Output == DeviceReductionOutput::device)
  {
    if (state_ != detail::DeviceReductionState::pending || !consumer.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if (!traits_type::succeeded(event_.record(consumer))) {
      std::abort();
    }
    publication_.completion_stream = consumer.stream();
    state_ = detail::DeviceReductionState::retiring;
    return Status::success();
  }

  // Contiguous inputs of count elements. Dot takes (left, right, count).
  template <detail::ReductionInput T>
    requires detail::DeviceReductionLaunchable<
        Backend, Operation, Accumulator, LaunchPolicy,
        PreparedDeviceReductionWork, const T *>
  [[nodiscard]] Status
  begin(const T *const values, const std::uint64_t count,
        const device::Executor<Backend> executor) noexcept {
    PreparedDeviceReductionWork work{};
    const ErrorCode error =
        try_prepare_device_reduction_work(prepared_, count, work);
    return error == ErrorCode::success ? begin_impl(work, executor, values)
                                       : Status::failure(error);
  }

  template <detail::ReductionInput Left, detail::ReductionInput Right>
    requires detail::DeviceReductionLaunchable<
        Backend, Operation, Accumulator, LaunchPolicy,
        PreparedDeviceReductionWork, const Left *, const Right *>
  [[nodiscard]] Status
  begin(const Left *const left, const Right *const right,
        const std::uint64_t count,
        const device::Executor<Backend> executor) noexcept {
    PreparedDeviceReductionWork work{};
    const ErrorCode error =
        try_prepare_device_reduction_work(prepared_, count, work);
    return error == ErrorCode::success
               ? begin_impl(work, executor, left, right)
               : Status::failure(error);
  }

  // Prepared work: PreparedDeviceReductionWork with const pointers or
  // PreparedDeviceFieldReductionWork with component views. Hard precondition:
  // work was prepared from prepared().
  template <class Input, class Work>
    requires detail::DeviceReductionLaunchable<
        Backend, Operation, Accumulator, LaunchPolicy, Work,
        detail::device_reduction_operand_t<Input>>
  [[nodiscard]] Status
  begin(const Input input, const Work work,
        const device::Executor<Backend> executor) noexcept {
    return begin_impl(
        work, executor,
        static_cast<detail::device_reduction_operand_t<Input>>(input));
  }

  template <class Left, class Right, class Work>
    requires detail::DeviceReductionLaunchable<
        Backend, Operation, Accumulator, LaunchPolicy, Work,
        detail::device_reduction_operand_t<Left>,
        detail::device_reduction_operand_t<Right>>
  [[nodiscard]] Status
  begin(const Left left, const Right right, const Work work,
        const device::Executor<Backend> executor) noexcept {
    return begin_impl(
        work, executor,
        static_cast<detail::device_reduction_operand_t<Left>>(left),
        static_cast<detail::device_reduction_operand_t<Right>>(right));
  }

  // Poll producer/retirement completion. As with wait(), retire queued scalar
  // consumers first; producer completion alone does not finish their reads.
  [[nodiscard]] Status conclude() noexcept {
    if (!pending()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const Status completed = event_.template advance<false>();
    if (completed.is_pending()) return completed;
    if (completed.is_failure()) std::abort();
    return publish_completed();
  }

  // The only blocking operation in the lifecycle. Device callers that enqueue
  // scalar consumers must retire_on() first so this also joins their last use.
  [[nodiscard]] Status wait() noexcept {
    if (!pending()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if (event_.template advance<true>().is_failure()) std::abort();
    return publish_completed();
  }

private:
  DeviceReduction(device::DeviceAllocation<Backend> &&scratch,
                  device::DeviceEvent<Backend> &&event,
                  const prepared_type prepared,
                  publication_type &&publication) noexcept
      : scratch_(std::move(scratch)), event_(std::move(event)),
        prepared_(prepared), publication_(std::move(publication)),
        state_(detail::DeviceReductionState::idle) {}

  template <class Work, class... Operand>
  [[nodiscard]] Status begin_impl(const Work work,
                                  const device::Executor<Backend> executor,
                                  const Operand... operands) noexcept {
    if (state_ != detail::DeviceReductionState::idle &&
        !(Output == DeviceReductionOutput::device &&
          state_ == detail::DeviceReductionState::retiring)) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if constexpr (Output == DeviceReductionOutput::host) {
      if (work.empty()) {
        publication_.result = {};
        return Status::success();
      }
    }
    if (!executor.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if (!work.empty() &&
        ((detail::device_reduction_operand_data(operands) == nullptr) ||
         ...)) {
      return Status::failure(ErrorCode::invalid_argument);
    }
    bool accepted = false;
    typename traits_type::error_type launch_error{};
    if constexpr (Output == DeviceReductionOutput::device) {
      if (state_ == detail::DeviceReductionState::retiring &&
          publication_.completion_stream != executor.stream() &&
          !traits_type::succeeded(event_.wait_on(executor))) {
        std::abort();
      }
      launch_error =
          work.empty()
              ? detail::DeviceReductionLaunch<Backend>::template enqueue_identity<
                    Operation, Accumulator>(publication_.value,
                                            executor.stream(), accepted)
              : detail::DeviceReductionLaunch<Backend>::template enqueue<
                    Operation, Accumulator, LaunchPolicy>(
                    work, scratch_.raw_base(), prepared_, executor.stream(),
                    accepted, publication_.value, operands...);
    } else {
      launch_error =
          detail::DeviceReductionLaunch<Backend>::template enqueue<
              Operation, Accumulator, LaunchPolicy>(
              work, scratch_.raw_base(), prepared_, executor.stream(),
              accepted, static_cast<Accumulator *>(nullptr), operands...);
    }
    if (const ErrorCode error =
            device::finish_device_enqueue<Backend>(launch_error, accepted);
        error != ErrorCode::success) {
      return Status::failure(error);
    }

    if constexpr (Output == DeviceReductionOutput::host) {
      const std::byte *const device_value =
          scratch_.raw_base() +
          prepared_.level_byte_offset(
              static_cast<std::size_t>(work.level_count() - 1U));
      if (!traits_type::succeeded(traits_type::copy_device_to_host_async(
              publication_.value, device_value, sizeof(Accumulator),
              executor.stream()))) {
        std::abort();
      }
      publication_.result = {};
    } else {
      publication_.completion_stream = executor.stream();
    }
    if (!traits_type::succeeded(event_.record(executor))) {
      std::abort();
    }
    state_ = detail::DeviceReductionState::pending;
    return Status::accepted();
  }

  [[nodiscard]] Status publish_completed() noexcept {
    if constexpr (Output == DeviceReductionOutput::host) {
      publication_.result =
          ReductionResult<Accumulator>::from_value(*publication_.value);
    }
    state_ = detail::DeviceReductionState::idle;
    return Status::success();
  }

  constexpr void disarm() noexcept {
    prepared_ = {};
    publication_.value = nullptr;
    if constexpr (Output == DeviceReductionOutput::host)
      publication_.result = {};
    state_ = detail::DeviceReductionState::unprepared;
  }

  device::DeviceAllocation<Backend> scratch_{};
  device::DeviceEvent<Backend> event_{};
  prepared_type prepared_{};
  [[no_unique_address]] publication_type publication_{};
  detail::DeviceReductionState state_ =
      detail::DeviceReductionState::unprepared;

  friend struct detail::PrivateConstructionAccess;
};

template <ReductionOperation Operation,
          detail::DeviceReductionAccumulator Accumulator,
          detail::HostResultReductionBackend Backend,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy>
[[nodiscard]] Result<
    DeviceReduction<Operation, Accumulator, Backend, LaunchPolicy>>
make_device_reduction(HostMemoryBudget &host_budget,
                      const PreparedDeviceReduction<Accumulator, LaunchPolicy>
                          &prepared) noexcept {
  using Reduction =
      DeviceReduction<Operation, Accumulator, Backend, LaunchPolicy>;
  using Publication = typename Reduction::publication_type;
  using Traits = device::BackendTraits<Backend>;
  if (!prepared.valid()) {
    return Result<Reduction>::from_error(ErrorCode::invalid_argument);
  }

  HostMemoryRequirement requirement{};
  if (!prepared.empty() &&
      !requirement.try_add(sizeof(Accumulator), alignof(Accumulator))) {
    return Result<Reduction>::from_error(ErrorCode::overflow);
  }
  auto admission = host_budget.try_admit(requirement);
  LPFX_TRY_RESULT(admission, Result<Reduction>);

  device::DeviceAllocation<Backend> scratch{};
  if (!Traits::succeeded(device::DeviceAllocation<Backend>::create(
          prepared.total_bytes(), scratch))) {
    return Result<Reduction>::from_error(ErrorCode::allocation_failed);
  }

  device::PinnedHostAllocation<Backend> readback{};
  device::DeviceEvent<Backend> event{};
  if (!prepared.empty()) {
    if (!Traits::succeeded(device::PinnedHostAllocation<Backend>::create(
            sizeof(Accumulator), readback))) {
      return Result<Reduction>::from_error(ErrorCode::allocation_failed);
    }
    if (!Traits::succeeded(device::DeviceEvent<Backend>::create(event))) {
      return Result<Reduction>::from_error(ErrorCode::backend_failure);
    }
  }

  auto *const host_value = reinterpret_cast<Accumulator *>(readback.raw_base());
  if (host_value != nullptr) {
    std::construct_at(host_value, Accumulator{});
  }
  return Result<Reduction>::from_value(
      detail::PrivateConstructionAccess::create<Reduction>(
          std::move(scratch), std::move(event), prepared,
          Publication{std::move(admission).value(), std::move(readback),
                      host_value, {}}));
}

// Borrow a scalar at a fixed device address. It is not initialized at setup;
// every accepted begin, including an empty input, publishes the operation's
// identity/result asynchronously. No pinned memory or host budget is needed.
template <ReductionOperation Operation,
          detail::DeviceReductionAccumulator Accumulator,
          detail::DeviceReductionBackend Backend,
          detail::DeviceReductionLaunchPolicyType LaunchPolicy>
  requires requires(
      Accumulator *output,
      typename device::BackendTraits<Backend>::stream_type stream,
      bool &accepted) {
    {
      detail::DeviceReductionLaunch<Backend>::template enqueue_identity<
          Operation, Accumulator>(output, stream, accepted)
    } noexcept -> std::same_as<typename device::BackendTraits<Backend>::error_type>;
  }
[[nodiscard]] Result<
    DeviceReduction<Operation, Accumulator, Backend, LaunchPolicy,
                    DeviceReductionOutput::device>>
make_device_reduction(
    const PreparedDeviceReduction<Accumulator, LaunchPolicy> &prepared,
    Accumulator *const device_output) noexcept {
  using Reduction =
      DeviceReduction<Operation, Accumulator, Backend, LaunchPolicy,
                      DeviceReductionOutput::device>;
  using Publication = typename Reduction::publication_type;
  using Traits = device::BackendTraits<Backend>;
  if (!prepared.valid() || device_output == nullptr ||
      reinterpret_cast<std::uintptr_t>(device_output) % alignof(Accumulator) !=
          0) {
    return Result<Reduction>::from_error(ErrorCode::invalid_argument);
  }
  device::DeviceAllocation<Backend> scratch{};
  // The last level is written directly to the borrowed scalar. Earlier levels
  // keep their prepared offsets; omitting this final slot is safe for all work
  // admitted by this capacity, including smaller one-level epochs.
  const std::size_t bytes =
      prepared.empty() ? 0 : prepared.total_bytes() - sizeof(Accumulator);
  if (!Traits::succeeded(
          device::DeviceAllocation<Backend>::create(bytes, scratch))) {
    return Result<Reduction>::from_error(ErrorCode::allocation_failed);
  }
  device::DeviceEvent<Backend> event{};
  if (!Traits::succeeded(device::DeviceEvent<Backend>::create(event))) {
    return Result<Reduction>::from_error(ErrorCode::backend_failure);
  }
  return Result<Reduction>::from_value(
      detail::PrivateConstructionAccess::create<Reduction>(
          std::move(scratch), std::move(event), prepared,
          Publication{device_output, {}}));
}

} // namespace lpfx

#endif
