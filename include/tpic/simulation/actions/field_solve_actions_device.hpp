#ifndef TPIC_SIMULATION_ACTIONS_FIELD_SOLVE_ACTIONS_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_FIELD_SOLVE_ACTIONS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/profiling.hpp>

#include <tpic/ecsim/gmres/gmres_device.hpp>
#include <tpic/ecsim/maxwell/maxwell_device.hpp>

#include <tpic/simulation/execution/active_backend_device.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include <utility>

namespace tpic::simulation {

// The implicit field solve of the committed state and the electric field the
// particles read. The solver streams are bound to Maxwell and GMRES.
template <class Precision, class Ports> class FieldSolveActions {
  using Maxwell = ecsim::MaxwellOperator<Precision, Backend>;
  using Gmres = ecsim::Gmres<Precision, Backend>;
public:
  FieldSolveActions(const simulation::PreparedInput &input,
                  const lpfx::device::DeviceProperties &properties, unsigned ranks,
                  data::DeviceFieldImage<Precision, Backend> &fields,
                  data::CurrentOwner<Precision, Backend> &current,
                  data::CompactMassMatrix<Precision, Backend> &mass, Ports &ports,
                  Executor solver_executor, Executor solver_mass_executor,
                  Executor solver_interior_executor)
      : input_(input), fields_(fields), current_(current), mass_(mass), ports_(ports),
        maxwell_(input, solver_executor, solver_mass_executor, solver_interior_executor),
        gmres_(input.domain.field_neighborhood.local_partition().local_cells(), input.physics.gmres_restart, properties,
               solver_executor, ranks) {}
  Maxwell &maxwell() noexcept { return maxwell_; }
  Gmres &gmres() noexcept { return gmres_; }
  [[nodiscard]] bool smoothed() const noexcept {
    return input_.maxwell_schedule == ecsim::MaxwellApplySchedule::smoothed;
  }
  void rhs_fields() {
    LPFX_PROFILE_SCOPE("TPIC::RhsFields");
    maxwell_.enqueue_rhs_fields(fields_.electric, fields_.magnetic_cell);
  }
  // RHS needs J, old E and curl(old B), but no mass coefficients.
  void complete_rhs(unsigned *invalid) {
    LPFX_PROFILE_SCOPE("TPIC::FinishRhs");
    maxwell_.complete_rhs(fields_.electric, current_, ports_.template get<"solver.smooth_a">(),
                        ports_.template get<"solver.smooth_b">(), invalid);
  }
  // The startup-resolved schedule enters a fully specialized solve. Every
  // Arnoldi/true-residual matvec has a direct compiled apply callback.
  // host_wait_mass_ghosts(executor) runs once, at the solve's first read of
  // mass ghost records.
  template <class HostWaitMassGhosts>
  auto host_blocking_solve(unsigned *invalid, const HostWaitMassGhosts &host_wait_mass_ghosts) {
    switch (input_.maxwell_schedule) {
    case ecsim::MaxwellApplySchedule::full:
      return host_blocking_solve_with<ecsim::MaxwellApplySchedule::full>(invalid, host_wait_mass_ghosts);
    case ecsim::MaxwellApplySchedule::split:
      return host_blocking_solve_with<ecsim::MaxwellApplySchedule::split>(invalid, host_wait_mass_ghosts);
    case ecsim::MaxwellApplySchedule::smoothed: break;
    }
    return host_blocking_solve_with<ecsim::MaxwellApplySchedule::smoothed>(invalid, host_wait_mass_ghosts);
  }
  const auto &complete_particle_field(unsigned *invalid) {
    LPFX_PROFILE_SCOPE("TPIC::ParticleField");
    return maxwell_.complete_particle_field(ports_.template get<"solver.theta">(),
                                           ports_.template get<"solver.smooth_a">(),
                                           ports_.template get<"solver.smooth_b">(), invalid);
  }

private:
  template <ecsim::MaxwellApplySchedule Schedule, class HostWaitMassGhosts>
  auto host_blocking_solve_with(unsigned *invalid, const HostWaitMassGhosts &host_wait_mass_ghosts) {
    LPFX_PROFILE_SCOPE("TPIC::SolveMaxwell");
    auto &input = ports_.template get<"solver.input">();
    auto &curl = ports_.template get<"solver.curl">();
    auto &smooth_a = ports_.template get<"solver.smooth_a">();
    auto &smooth_b = ports_.template get<"solver.smooth_b">();
    bool ghosts_pending = true;
    const auto host_wait_mass_ghosts_once = [&](const Executor executor) {
      if (std::exchange(ghosts_pending, false)) host_wait_mass_ghosts(executor);
    };
    return gmres_.host_blocking_solve(
        maxwell_.theta, maxwell_.rhs, maxwell_.input, maxwell_.image,
        [&](auto &, auto &) {
          maxwell_.template complete_apply<Schedule>(mass_, input, curl, smooth_a, smooth_b, invalid,
                                                     host_wait_mass_ghosts_once);
        },
        // The residual apply computes in the solver reduction type.
        [&](auto &, auto &) {
          maxwell_.template complete_apply<Schedule, typename Maxwell::Residual>(
              mass_, input, curl, smooth_a, smooth_b, invalid, host_wait_mass_ghosts_once);
        },
        ports_.template get<"gmres">(), input_.physics.gmres_max_iterations,
        input_.parameters.gmres_tol);
  }

  const simulation::PreparedInput &input_;
  data::DeviceFieldImage<Precision, Backend> &fields_;
  data::CurrentOwner<Precision, Backend> &current_;
  data::CompactMassMatrix<Precision, Backend> &mass_;
  Ports &ports_;
  Maxwell maxwell_;
  Gmres gmres_;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_FIELD_SOLVE_ACTIONS_DEVICE_HPP
