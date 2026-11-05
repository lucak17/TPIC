#ifndef TPIC_INITIALIZATION_CASES_FORCE_FREE_HPP
#define TPIC_INITIALIZATION_CASES_FORCE_FREE_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cstddef>
#include <string_view>

namespace tpic::initialization {

// ForceFree: a force-free current sheet with uniform loading.
struct ForceFreeCase {
  static constexpr std::array<std::string_view, 1> names{"ForceFree"};
  explicit ForceFreeCase(const config::SimulationParameters& params);
  [[nodiscard]] FieldVector electric(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
  [[nodiscard]] std::string_view particle_label() const { return "uniform"; }
  using Profile = UniformProfile;
  [[nodiscard]] Profile profile(std::size_t) const noexcept { return {}; }
  bool operator==(const ForceFreeCase&) const = default;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_FORCE_FREE_HPP
