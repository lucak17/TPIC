#ifndef LPFX_EXECUTION_HOST_PREPARED_PLAN_HPP
#define LPFX_EXECUTION_HOST_PREPARED_PLAN_HPP

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

namespace lpfx::detail {

enum class PreparedPlanPhase : std::uint8_t {
  unprepared = 0,
  idle = 1,
  active = 2,
  awaiting_conclusion = 3,
  exhausted = 4
};

// Generic host-side epoch state. Distributed admission and family semantics
// deliberately remain outside this type. Move construction transfers the
// state and leaves the source unprepared; there is no assignment.
class PreparedPlanState final {
public:
  using generation_type = std::uint64_t;

  constexpr PreparedPlanState() noexcept = default;

  constexpr PreparedPlanState(PreparedPlanState &&other) noexcept
      : generation_(std::exchange(other.generation_, 0)),
        phase_(std::exchange(other.phase_, PreparedPlanPhase::unprepared)) {}

  [[nodiscard]] constexpr PreparedPlanPhase phase() const noexcept {
    return phase_;
  }

  [[nodiscard]] constexpr generation_type generation() const noexcept {
    return generation_;
  }

  [[nodiscard]] constexpr bool prepared() const noexcept {
    return phase_ != PreparedPlanPhase::unprepared;
  }

  [[nodiscard]] constexpr bool reusable() const noexcept {
    return phase_ == PreparedPlanPhase::idle;
  }

  [[nodiscard]] constexpr bool conclusion_required() const noexcept {
    return phase_ == PreparedPlanPhase::active ||
           phase_ == PreparedPlanPhase::awaiting_conclusion;
  }

  // unprepared -> idle at generation 0.
  constexpr void prepare() noexcept {
    if (prepared()) {
      std::abort();
    }
    generation_ = 0;
    phase_ = PreparedPlanPhase::idle;
  }

  // Any phase without a pending conclusion -> unprepared.
  constexpr void reset() noexcept {
    if (conclusion_required()) {
      std::abort();
    }
    generation_ = 0;
    phase_ = PreparedPlanPhase::unprepared;
  }

  // idle -> active, or awaiting_conclusion when the epoch completed on
  // submission. Advances the generation.
  constexpr void commit(const bool completed) noexcept {
#ifndef NDEBUG
    if (phase_ != PreparedPlanPhase::idle ||
        generation_ == std::numeric_limits<generation_type>::max()) {
      std::abort();
    }
#endif
    ++generation_;
    phase_ = completed ? PreparedPlanPhase::awaiting_conclusion
                       : PreparedPlanPhase::active;
  }

  // active -> awaiting_conclusion.
  constexpr void complete() noexcept {
#ifndef NDEBUG
    if (phase_ != PreparedPlanPhase::active) {
      std::abort();
    }
#endif
    phase_ = PreparedPlanPhase::awaiting_conclusion;
  }

  // awaiting_conclusion -> idle, or exhausted after the last generation.
  constexpr void retire() noexcept {
#ifndef NDEBUG
    if (phase_ != PreparedPlanPhase::awaiting_conclusion) {
      std::abort();
    }
#endif
    phase_ = generation_ == std::numeric_limits<generation_type>::max()
                 ? PreparedPlanPhase::exhausted
                 : PreparedPlanPhase::idle;
  }

private:
  generation_type generation_ = 0;
  PreparedPlanPhase phase_ = PreparedPlanPhase::unprepared;
};

} // namespace lpfx::detail

#endif
