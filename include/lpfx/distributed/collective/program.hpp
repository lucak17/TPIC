#ifndef LPFX_DISTRIBUTED_COLLECTIVE_PROGRAM_HPP
#define LPFX_DISTRIBUTED_COLLECTIVE_PROGRAM_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/collective/port.hpp>
#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

struct PreparedOperationProgramAccess;

enum class CollectiveProgramPhase : std::uint8_t {
  empty = 0,
  ready = 1,
  pre_native = 2,
  native = 3,
  post_native = 4,
  complete = 5,
};
struct EmptyCollectiveOutput final {};

// The sole semantic/native sequence of every collective lowering. Equal
// algorithms and operands share one program type; registration identity
// belongs to cold preparation only.
template <CollectiveDescriptor Descriptor, class Operand, class Executor,
          class Port>
class CollectiveProgram final {
public:
  using port_type = Port;
  using descriptor_type = Descriptor;
  using operand_type = Operand;
  using value_type = typename operand_type::value_type;
  using executor_type = Executor;

  static constexpr bool exact_in_place = operand_type::exact_in_place;
  static constexpr bool needs_local_access =
      port_type::has_pre_native || port_type::has_post_native;

  constexpr CollectiveProgram() noexcept = default;
  LPFX_NONMOVABLE(CollectiveProgram);

  [[nodiscard]] bool empty() const noexcept {
    return phase_ == CollectiveProgramPhase::empty && input_ == nullptr &&
           port_.empty();
  }

  // Cold validation before registration: binding and plan/operand contract,
  // setup rank, and the port's setup and device-affinity checks.
  template <class Setup, class Binding>
  [[nodiscard]] static ErrorCode try_validate(const Setup &setup,
                                              const Binding &binding) noexcept {
    const auto &plan = binding.plan();
    const operand_type &operand = binding.resource();
    if (!binding.valid() || !contract_valid(plan, operand)) {
      return ErrorCode::invalid_argument;
    }
    std::size_t bytes{};
    if (!lpfx::detail::checked_byte_count<value_type>(plan.input_count(),
                                                      bytes) ||
        !lpfx::detail::checked_byte_count<value_type>(plan.output_count(),
                                                      bytes)) {
      return ErrorCode::overflow;
    }
    if (!setup.is_valid() || setup.size() <= 0 || setup.rank() < 0 ||
        setup.rank() >= setup.size() ||
        static_cast<std::uint64_t>(setup.size()) != plan.rank_count()) {
      return ErrorCode::invalid_argument;
    }
    const port_type port{};
    return port.try_validate_setup(
        setup, make_shape(plan, static_cast<std::uint32_t>(setup.rank())),
        buffers(operand), binding.executor());
  }

  // Local native work; the composition agrees the rank-global union.
  template <class Binding>
  [[nodiscard]] static bool needs_native_work(const Binding &binding) noexcept {
    return native_count(binding.plan()) != 0;
  }

  // Consumes a binding accepted by try_validate. The program keeps the
  // operand addresses and the executor; the plan is not retained.
  template <class Session, class Setup, class Binding>
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &budget,
                                      Session &session, const Setup &setup,
                                      Binding &&binding, int) noexcept {
    if (!empty()) {
      return ErrorCode::invalid_state;
    }
    const operand_type &operand = binding.resource();
    const CollectivePreparedShape shape =
        make_shape(binding.plan(), static_cast<std::uint32_t>(setup.rank()));
    const CollectiveLocalAccess access = make_local_access(shape, operand);
    executor_.emplace(std::move(binding).take_executor());
    if (shape.has_native_work()) {
      ErrorCode error = port_.try_prepare_resources(budget, session, setup,
                                                    shape, access, executor());
      if (error == ErrorCode::success && !port_.is_valid(shape, access)) {
        error = ErrorCode::invalid_state;
      }
      if (error != ErrorCode::success) {
        port_.reset_resources();
        executor_.reset();
        return error;
      }
    }
    if constexpr (needs_local_access) {
      access_ = access;
    }
    input_ = operand.input_data();
    if constexpr (!exact_in_place) {
      output_ = operand.output_data();
    }
    epoch_.native_count = shape.native_count;
    phase_ = CollectiveProgramPhase::ready;
    return ErrorCode::success;
  }

  // The ready transition seals port resources and access until release.
  [[nodiscard]] bool is_valid() const noexcept {
    return phase_ >= CollectiveProgramPhase::ready;
  }

  [[nodiscard]] const value_type *staged_output() const noexcept
    requires requires(const port_type &port) { port.staged_output(); }
  {
    return port_.staged_output();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start() noexcept {
    assert(phase_ == CollectiveProgramPhase::ready);
    if (!epoch_.has_native_work()) {
      phase_ = CollectiveProgramPhase::complete;
      return Status::success();
    }

    if constexpr (port_type::has_pre_native) {
      const Status pre = port_.start_pre(access_, buffers(), executor());
      require_internal_status(pre);
      if (pre.is_pending()) {
        phase_ = CollectiveProgramPhase::pre_native;
        return Status::pending();
      }
    }
    return start_native();
  }

private:
  friend struct PreparedOperationProgramAccess;

  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status drive() noexcept {
    // PreparedOperation handles completed epochs before calling this driver.
    if constexpr (port_type::has_pre_native) {
      if (phase_ == CollectiveProgramPhase::pre_native) {
        const Status pre = port_.template advance_pre<Wait>();
        require_internal_status(pre);
        if (pre.is_pending() || start_native().is_pending()) {
          return Status::pending();
        }
      }
    }

    if constexpr (port_type::has_native_completion) {
      if (phase_ == CollectiveProgramPhase::native) {
        const Status native = port_.template advance_native<Wait>();
        require_internal_status(native);
        if (native.is_pending() || start_post().is_pending()) {
          return Status::pending();
        }
      }
    }

    if constexpr (port_type::has_post_native) {
      if (phase_ == CollectiveProgramPhase::post_native) {
        const Status post = port_.template advance_post<Wait>();
        require_internal_status(post);
        if (post.is_pending()) {
          return Status::pending();
        }
        phase_ = CollectiveProgramPhase::complete;
      }
    }

    if (phase_ != CollectiveProgramPhase::complete) {
      std::abort();
    }
    return Status::success();
  }

public:
  [[nodiscard]] LPFX_ALWAYS_INLINE Status release() noexcept {
    // Public awaiting_conclusion admits release only after start/drive has
    // established completion; pending native release preserves that phase.
    assert(phase_ == CollectiveProgramPhase::complete);
    if constexpr (port_type::has_native_completion) {
      const Status released =
          epoch_.has_native_work() ? port_.release_native() : Status::success();
      require_internal_status(released);
      if (released.is_pending()) {
        return released;
      }
    }
    phase_ = CollectiveProgramPhase::ready;
    return Status::success();
  }

  void release_resources_for_close() noexcept {
    if (phase_ != CollectiveProgramPhase::ready) {
      std::abort();
    }
    port_.reset_resources();
    clear();
  }

  void reset() noexcept {
    if (phase_ > CollectiveProgramPhase::ready) {
      std::abort();
    }
    port_.reset();
    clear();
  }

private:
  template <class Plan>
  [[nodiscard]] static constexpr std::size_t
  native_count(const Plan &plan) noexcept {
    if constexpr (descriptor_type::kind == collective_kind::reduce_scatter) {
      return plan.output_count();
    } else {
      return plan.input_count();
    }
  }

  template <class Plan>
  [[nodiscard]] static CollectivePreparedShape
  make_shape(const Plan &plan, const std::uint32_t rank) noexcept {
    CollectivePreparedShape shape{};
    shape.input_count = plan.input_count();
    shape.output_count = plan.output_count();
    shape.native_count = native_count(plan);
    shape.rank = rank;
    shape.rank_count = plan.rank_count();
    shape.root = descriptor_type::root;
    return shape;
  }

  template <class Plan>
  [[nodiscard]] static bool contract_valid(const Plan &plan,
                                           const operand_type &operand) noexcept {
    if (plan.input_count() != operand.input_count() ||
        plan.output_count() != operand.output_count()) {
      return false;
    }
    if constexpr (!exact_in_place) {
      if (!operand.separate_buffers_disjoint()) {
        return false;
      }
    }
    if (Plan::requires_exact_in_place()) {
      return operand.in_place();
    }
    return Plan::permits_exact_in_place() || !operand.in_place();
  }

  [[nodiscard]] static CollectiveLocalAccess
  make_local_access(const CollectivePreparedShape &shape,
                    const operand_type &operand) noexcept {
    CollectiveLocalAccess access{};
    if (!shape.has_native_work()) {
      return access;
    }
    // Validation proved both byte counts representable.
    const std::size_t input_bytes = shape.input_count * sizeof(value_type);
    const std::size_t output_bytes = shape.output_count * sizeof(value_type);
    const ByteRegion<const std::byte> input{
        reinterpret_cast<const std::byte *>(operand.input_data()), input_bytes};
    const ByteRegion<const std::byte> output_read{
        reinterpret_cast<const std::byte *>(operand.output_data()),
        output_bytes};
    const ByteRegion<std::byte> output_write{
        reinterpret_cast<std::byte *>(operand.output_data()), output_bytes};
    assert(input.valid() && output_read.valid() && output_write.valid());

    if constexpr (descriptor_type::kind == collective_kind::broadcast) {
      if (shape.rank == shape.root) {
        access.read = output_read;
      } else {
        access.write = output_write;
      }
    } else if constexpr (descriptor_type::kind == collective_kind::reduce) {
      access.read =
          exact_in_place && shape.rank == shape.root ? output_read : input;
      if (shape.rank == shape.root) {
        access.write = output_write;
      }
    } else {
      access.read = exact_in_place ? output_read : input;
      access.write = output_write;
    }
    return access;
  }

  [[nodiscard]] static CollectiveBuffers<value_type>
  buffers(const operand_type &operand) noexcept {
    return {operand.input_data(), operand.output_data()};
  }

  [[nodiscard]] CollectiveBuffers<value_type> buffers() const noexcept {
    if constexpr (exact_in_place) {
      return {input_, const_cast<value_type *>(input_)};
    } else {
      return {input_, output_};
    }
  }

  [[nodiscard]] const executor_type &executor() const noexcept {
    return executor_.get();
  }

  static void require_internal_status(const Status status) noexcept {
    if (status.is_failure() || status.requires_conclusion()) {
      // Once a matching collective can proceed on another rank, no local
      // native or ordering failure is safely recoverable.
      std::abort();
    }
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_native() noexcept {
    const Status submitted = port_.template submit<descriptor_type::root>(
        epoch_, [this]() noexcept { return buffers(); }, executor());
    require_internal_status(submitted);
    if (submitted.is_pending()) {
      phase_ = CollectiveProgramPhase::native;
      return Status::pending();
    }
    return start_post();
  }

  [[nodiscard]] Status start_post() noexcept {
    if constexpr (port_type::has_post_native) {
      const Status post = port_.start_post(access_, buffers(), executor());
      require_internal_status(post);
      if (post.is_pending()) {
        phase_ = CollectiveProgramPhase::post_native;
        return Status::pending();
      }
    }
    phase_ = CollectiveProgramPhase::complete;
    return Status::success();
  }

  void clear() noexcept {
    if (!port_.empty()) {
      std::abort();
    }
    input_ = nullptr;
    if constexpr (!exact_in_place) {
      output_ = nullptr;
    }
    executor_.reset();
    epoch_ = CollectiveEpoch{};
    if constexpr (needs_local_access) {
      access_ = CollectiveLocalAccess{};
    }
    phase_ = CollectiveProgramPhase::empty;
  }

  // Exact-in-place execution needs only the input address.
  const value_type *input_ = nullptr;
  [[no_unique_address]] std::conditional_t<exact_in_place,
                                           EmptyCollectiveOutput, value_type *>
      output_{};
  [[no_unique_address]] ExecutorSlot<typename operand_type::space_type,
                                     executor_type> executor_{};
  CollectiveEpoch epoch_{};
  [[no_unique_address]] std::conditional_t<needs_local_access,
                                           CollectiveLocalAccess,
                                           EmptyCollectiveAccess> access_{};
  [[no_unique_address]] port_type port_{};
  CollectiveProgramPhase phase_ = CollectiveProgramPhase::empty;
};

} // namespace lpfx::distributed::detail

#endif
