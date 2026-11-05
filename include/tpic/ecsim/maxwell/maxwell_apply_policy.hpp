#ifndef TPIC_ECSIM_MAXWELL_MAXWELL_APPLY_POLICY_HPP
#define TPIC_ECSIM_MAXWELL_MAXWELL_APPLY_POLICY_HPP

#include <tpic/config/numerics_options.hpp>

#include <stdexcept>
#include <string_view>

namespace tpic::ecsim {

enum class MaxwellApplySchedule { full, split, smoothed };

[[nodiscard]] constexpr std::string_view maxwell_apply_schedule_name(MaxwellApplySchedule schedule) {
  switch (schedule) {
  case MaxwellApplySchedule::full: return "full";
  case MaxwellApplySchedule::split: return "split";
  case MaxwellApplySchedule::smoothed: return "smoothed full-domain";
  }
  throw std::invalid_argument("TPIC Maxwell: invalid apply schedule");
}

[[nodiscard]] constexpr MaxwellApplySchedule resolve_maxwell_apply(
    config::MassApplyMode requested, bool smoothing_enabled) {
  if (!config::valid_mass_apply_mode(requested))
    throw std::invalid_argument("TPIC Maxwell: invalid mass apply mode");
  if (smoothing_enabled) return MaxwellApplySchedule::smoothed;
  return requested == config::MassApplyMode::full ? MaxwellApplySchedule::full : MaxwellApplySchedule::split;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MAXWELL_MAXWELL_APPLY_POLICY_HPP
