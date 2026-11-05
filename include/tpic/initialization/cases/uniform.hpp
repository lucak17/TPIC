#ifndef TPIC_INITIALIZATION_CASES_UNIFORM_HPP
#define TPIC_INITIALIZATION_CASES_UNIFORM_HPP

#include <tpic/initialization/cases/case_interface.hpp>

#include <cstddef>
#include <string_view>

namespace tpic::initialization {

// Uniform: no seed field and uniform loading.
struct UniformCase {
  static constexpr std::array<std::string_view, 1> names{"Uniform"};
  UniformCase() = default;
  explicit UniformCase(const config::SimulationParameters&) {}
  [[nodiscard]] FieldVector electric(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] FieldVector magnetic(const config::SimulationParameters&, double, double, double) const { return {}; }
  [[nodiscard]] std::string_view particle_label() const { return "uniform"; }
  using Profile = UniformProfile;
  [[nodiscard]] Profile profile(std::size_t) const noexcept { return {}; }
  bool operator==(const UniformCase&) const = default;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_UNIFORM_HPP
