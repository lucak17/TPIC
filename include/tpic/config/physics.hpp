#ifndef TPIC_CONFIG_PHYSICS_HPP
#define TPIC_CONFIG_PHYSICS_HPP

#include <array>
#include <string>
#include <vector>

#include <tpic/config/boundary_conditions.hpp>
#include <tpic/config/simulation_parameters.hpp>

namespace tpic::config {

struct PreparedPhysics {
  bool operator==(const PreparedPhysics&) const = default;

  // Explicit face order: Xright, Xleft, Yright, Yleft, Zright, Zleft.
  std::array<FieldBoundary, 6> field_boundary{};
  std::array<ParticleBoundary, 6> particle_boundary{};
  bool smoothing_enabled = false;
  int gmres_max_iterations = 100;
  int gmres_restart = 30;
  double theta_dt = 0;
  double c_theta_dt = 0;
  double final_time = 0;
  std::vector<std::string> notes;
  std::vector<std::string> unsupported_requests;
};

// Resolve linear-ECSIM semantics without device/MPI/output side effects.
// Invalid physics throws. Unsupported execution requests are returned explicitly
// so a checker can show all of them without changing the parsed values.
[[nodiscard]] PreparedPhysics prepare_physics(const SimulationParameters &parameters);

} // namespace tpic::config

#endif // TPIC_CONFIG_PHYSICS_HPP
