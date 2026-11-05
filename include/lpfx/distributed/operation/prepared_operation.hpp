#ifndef LPFX_DISTRIBUTED_OPERATION_PREPARED_OPERATION_HPP
#define LPFX_DISTRIBUTED_OPERATION_PREPARED_OPERATION_HPP

#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/distributed/field/submission.hpp>
#include <lpfx/distributed/particle/migration/stage.hpp>
#include <lpfx/execution/host/prepared_plan.hpp>

namespace lpfx::distributed {

template <class... Bindings> class Communication;

namespace detail {

template <class Program>
inline constexpr bool staged_particle_program = [] {
  if constexpr (requires { Program::staged_migration; })
    return Program::staged_migration;
  else
    return false;
}();

// Family driving stays private to its program. This stateless access point is
// the only bridge from the public lifecycle to the family state machine.
struct PreparedOperationProgramAccess final {
  template <class Program>
  static LPFX_ALWAYS_INLINE void
  set_candidate_generation(Program &program,
                           const std::uint64_t generation) noexcept {
    if constexpr (requires { program.set_candidate_generation(generation); })
      program.set_candidate_generation(generation);
  }

  template <bool Wait, ParticleMigrationStage... Stages, class Program>
    requires ((sizeof...(Stages) == 0 && requires(Program &program) {
      { program.template drive<Wait>() } noexcept -> std::same_as<Status>;
    }) || (sizeof...(Stages) == 1 && requires(Program &program) {
      { program.template drive<Wait, Stages...>() } noexcept -> std::same_as<Status>;
    }))
  [[nodiscard]] static LPFX_ALWAYS_INLINE Status drive(Program &program) noexcept {
    if constexpr (sizeof...(Stages) == 0) return program.template drive<Wait>();
    else return program.template drive<Wait, Stages...>();
  }
};

template <class Program>
concept PreparedOperationProgram =
    std::default_initializable<Program> &&
    std::is_nothrow_default_constructible_v<Program> &&
    std::is_nothrow_destructible_v<Program> &&
    requires(Program &program, const Program &constant) {
      { constant.empty() } noexcept -> std::same_as<bool>;
      { constant.is_valid() } noexcept -> std::same_as<bool>;
      {
        PreparedOperationProgramAccess::template drive<false>(program)
      } noexcept -> std::same_as<Status>;
      {
        PreparedOperationProgramAccess::template drive<true>(program)
      } noexcept -> std::same_as<Status>;
      { program.release() } noexcept -> std::same_as<Status>;
      { program.release_resources_for_close() } noexcept -> std::same_as<void>;
      { program.reset() } noexcept -> std::same_as<void>;
    } && (requires(Program &program) {
      { program.start() } noexcept -> std::same_as<Status>;
    } || requires(Program &program, typename Program::executor_type executor) {
      { program.start(executor) } noexcept -> std::same_as<Status>;
    });

[[nodiscard]] constexpr bool
completed_without_obligation(const Status status) noexcept {
  return status.is_success() && !status.requires_conclusion();
}

} // namespace detail

// Sole public owner of operation readiness, phase, conclusion obligation, and
// generation. Program owns only its family algorithm and concrete static port.
template <detail::PreparedOperationProgram Program>
class PreparedOperation final {
public:
  using program_type = Program;
  using generation_type = lpfx::detail::PreparedPlanState::generation_type;

  constexpr PreparedOperation() noexcept = default;
  LPFX_NONMOVABLE(PreparedOperation);

  ~PreparedOperation() noexcept {
    if (state_.prepared()) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return !state_.prepared() && program_.empty();
  }

  // Successful publication establishes the program-validity invariant until
  // reset. Repeated lifecycle calls therefore inspect only the authoritative
  // FSM and never repeat a deep Program::is_valid() traversal.
  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return state_.prepared();
  }

  [[nodiscard]] constexpr bool reusable() const noexcept {
    if constexpr (requires { program_.auxiliary_reusable(); })
      return state_.reusable() && program_.auxiliary_reusable();
    return state_.reusable();
  }

  [[nodiscard]] constexpr bool quiescent_for_close() const noexcept {
    const lpfx::detail::PreparedPlanPhase phase = state_.phase();
    const bool quiescent = phase == lpfx::detail::PreparedPlanPhase::idle ||
                           phase == lpfx::detail::PreparedPlanPhase::exhausted;
    if constexpr (requires { program_.auxiliary_quiescent_for_close(); })
      return quiescent && program_.auxiliary_quiescent_for_close();
    return quiescent;
  }

  [[nodiscard]] constexpr generation_type generation() const noexcept {
    return state_.generation();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status begin() noexcept
    requires(!detail::staged_particle_program<Program>)
  {
    return begin_impl();
  }

  // The local preparation stream is supplied for this epoch. The binding's
  // executor remains the routing stream; local and routing stages do not
  // rebind transport resources or mutate each other's executor. The caller
  // supplies a stream on the binding's prepared physical device; worker
  // device selection and stream creation establish this before admission.
  template <class Executor>
    requires detail::staged_particle_program<Program> &&
             std::same_as<Executor, typename Program::executor_type>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  begin(const Executor executor) noexcept {
    return begin_impl(executor);
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status progress() noexcept
    requires(!detail::staged_particle_program<Program>)
  {
    return advance<false>();
  }

  // MPI and bare NCCL/RCCL routes block the calling thread until completion;
  // under StreamOrderedCompletion begin() completed the operation at enqueue
  // and this returns at once.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status wait() noexcept
    requires(!detail::staged_particle_program<Program>)
  {
    return advance<true>();
  }

  // Only the field owner observes this milestone after begin. It performs no
  // progress or synchronization and remains valid after completion until
  // conclude retires the generation. Publish the result through a host fence
  // rather than querying the same operation from a second CPU thread.
  [[nodiscard]] Result<FieldSubmission> first_native_submission() const noexcept
    requires requires(const Program &program) {
      { program.first_native_submission() } noexcept
          -> std::same_as<FieldSubmission>;
    }
  {
    if (!state_.conclusion_required())
      return Result<FieldSubmission>::from_error(ErrorCode::invalid_state);
    return Result<FieldSubmission>::from_value(
        program_.first_native_submission());
  }

  // A staged collective's result in its pinned host image, from the epoch's
  // completion until the next begin, without synchronization. Read it on the
  // thread that drives the operation.
  [[nodiscard]] auto staged_output() const noexcept
    requires requires(const Program &program) {
      { program.staged_output() } noexcept;
    }
  {
    using Output = Result<decltype(program_.staged_output())>;
    const lpfx::detail::PreparedPlanPhase phase = state_.phase();
    if (phase != lpfx::detail::PreparedPlanPhase::awaiting_conclusion &&
        (phase != lpfx::detail::PreparedPlanPhase::idle ||
         state_.generation() == 0))
      return Output::from_error(ErrorCode::invalid_state);
    return Output::from_value(program_.staged_output());
  }

  // Stage selection is static. Preparation and finalization have one local
  // owner; resident and routing may be progressed by independent CPU threads
  // after the completed preparation handoff. Only finalization retires work.
  template <ParticleMigrationStage Stage>
  [[nodiscard]] Status progress() noexcept
    requires detail::staged_particle_program<Program>
  {
    return advance<false, Stage>();
  }
  template <ParticleMigrationStage Stage>
  [[nodiscard]] Status wait() noexcept
    requires detail::staged_particle_program<Program>
  {
    return advance<true, Stage>();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status conclude() noexcept {
    switch (state_.phase()) {
    case lpfx::detail::PreparedPlanPhase::active:
      return Status::pending();
    case lpfx::detail::PreparedPlanPhase::awaiting_conclusion:
      break;
    case lpfx::detail::PreparedPlanPhase::exhausted:
      return Status::failure(ErrorCode::capacity_exceeded);
    case lpfx::detail::PreparedPlanPhase::unprepared:
    case lpfx::detail::PreparedPlanPhase::idle:
      return Status::failure(ErrorCode::invalid_state);
    }

    const Status released = program_.release();
    if (released.is_failure() || released.requires_conclusion()) {
      std::abort();
    }
    if (released.is_pending()) {
      return released;
    }
    if (!detail::completed_without_obligation(released)) {
      std::abort();
    }

    state_.retire();
    return Status::success();
  }

  [[nodiscard]] Status exchange() noexcept
    requires(!detail::staged_particle_program<Program>)
  {
    const Status begun = begin();
    if (begun.is_failure()) {
      return begun;
    }
    if (!begun.requires_conclusion()) {
      std::abort();
    }

    const Status completed = wait();
    if (completed.is_failure()) {
      std::abort();
    }

    Status concluded = conclude();
    while (concluded.is_pending()) {
      concluded = conclude();
    }
    if (concluded.is_failure()) {
      std::abort();
    }
    return concluded;
  }

  // Family extensions.
  template <class Executor>
  [[nodiscard]] auto prepare_classification(Executor executor) noexcept
    requires requires(Program &program) { program.prepare_classification(executor); }
  {
    using Output = decltype(program_.prepare_classification(executor));
    if (!state_.reusable()) return Output::from_error(ErrorCode::invalid_state);
    return program_.prepare_classification(executor);
  }
  [[nodiscard]] bool resident_access_ready() const noexcept
    requires requires(const Program &program) { program.resident_access_ready(); }
  {
    return active() && program_.resident_access_ready();
  }
  [[nodiscard]] auto acquire_resident_access() noexcept
    requires requires(Program &program) { program.acquire_resident_access(); }
  {
    using Output = decltype(program_.acquire_resident_access());
    if (!active()) return Output::from_error(ErrorCode::invalid_state);
    return program_.acquire_resident_access();
  }
  template <class Event>
  [[nodiscard]] Status release_resident_access(const Event &event) noexcept
    requires requires(Program &program) { program.release_resident_access(event); }
  {
    if (!active()) return Status::failure(ErrorCode::invalid_state);
    return program_.release_resident_access(event);
  }
  [[nodiscard]] auto completed_partition() const noexcept
    requires requires(const Program &program) { program.completed_partition(); }
  {
    return program_.completed_partition();
  }
  [[nodiscard]] auto finalization_requirements() const noexcept
    requires requires(const Program &program) { program.finalization_requirements(); }
  {
    return program_.finalization_requirements();
  }
  [[nodiscard]] bool storage_changed() const noexcept
    requires requires(const Program &program) { program.storage_changed(); }
  {
    return program_.storage_changed();
  }

private:
  template <class... Args>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status begin_impl(Args &&...args) noexcept {
    const lpfx::detail::PreparedPlanPhase phase = state_.phase();
    if (phase != lpfx::detail::PreparedPlanPhase::idle) {
      return Status::failure(phase == lpfx::detail::PreparedPlanPhase::exhausted
                                 ? ErrorCode::capacity_exceeded
                                 : ErrorCode::invalid_state);
    }
    if constexpr (requires { program_.auxiliary_reusable(); })
      if (!program_.auxiliary_reusable())
        return Status::failure(ErrorCode::invalid_state);
    // An empty/synchronous start may issue a resident capability immediately.
    // Centrally compute its candidate epoch before entering the program, while
    // preserving the prior committed generation on preacceptance failure.
    detail::PreparedOperationProgramAccess::set_candidate_generation(
        program_, state_.generation() + 1);
    const Status started = program_.start(std::forward<Args>(args)...);
    if (started.requires_conclusion()) {
      std::abort();
    }
    if (started.is_failure()) {
      // Program::start() may fail recoverably only before peer-visible work;
      // state and generation are still untouched here.
      return started;
    }

    const bool completed = detail::completed_without_obligation(started);
    if (!completed && !started.is_pending()) {
      std::abort();
    }

    state_.commit(completed);
    return Status::accepted();
  }

  template <bool Wait, ParticleMigrationStage... Stages>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() noexcept {
    static_assert(sizeof...(Stages) <= 1);
    constexpr bool staged = sizeof...(Stages) != 0;
    constexpr bool completes_epoch =
        ((Stages == ParticleMigrationStage::finalize) && ...);
    switch (state_.phase()) {
    case lpfx::detail::PreparedPlanPhase::awaiting_conclusion:
      if constexpr (completes_epoch)
        return Status::success();
      else
        return Status::failure(ErrorCode::invalid_state);
    case lpfx::detail::PreparedPlanPhase::active:
      break;
    case lpfx::detail::PreparedPlanPhase::exhausted:
      return Status::failure(ErrorCode::capacity_exceeded);
    case lpfx::detail::PreparedPlanPhase::unprepared:
    case lpfx::detail::PreparedPlanPhase::idle:
      return Status::failure(ErrorCode::invalid_state);
    }
    for (;;) {
      const auto advanced =
          detail::PreparedOperationProgramAccess::template drive<Wait,
                                                                 Stages...>(
              program_);
      if (advanced.is_pending()) {
        if constexpr (Wait)
          continue;
        return advanced;
      }
      // Stage preconditions can reject incorrect ownership/order without
      // altering the accepted epoch. Native post-acceptance failures remain
      // fail-stop in the family program, as for single-stage operations.
      if constexpr (staged)
        if (advanced.is_failure())
          return advanced;
      if (!detail::completed_without_obligation(advanced))
        std::abort();
      if constexpr (completes_epoch)
        state_.complete();
      return Status::success();
    }
  }

  [[nodiscard]] bool active() const noexcept {
    return state_.phase() == lpfx::detail::PreparedPlanPhase::active;
  }

  [[nodiscard]] bool ready_to_publish() const noexcept {
    return !state_.prepared() && program_.is_valid();
  }

  void publish() noexcept {
    if (!ready_to_publish()) {
      std::abort();
    }
    state_.prepare();
  }

  void reset_unpublished() noexcept {
    if (state_.prepared()) {
      std::abort();
    }
    program_.reset();
    if (!program_.empty()) {
      std::abort();
    }
    state_.reset();
  }

  // Called only after Communication has completed rank-symmetric quiescence
  // admission. Shared contexts are still alive throughout this call.
  void release_for_close() noexcept {
    if (!quiescent_for_close()) {
      std::abort();
    }
    program_.release_resources_for_close();
    if (!program_.empty()) {
      std::abort();
    }
    state_.reset();
  }

  Program program_{};
  lpfx::detail::PreparedPlanState state_{};

  template <class...> friend class Communication;
};

} // namespace lpfx::distributed

#endif
