#ifndef TPIC_SIMULATION_ACTIONS_DEPOSITION_ACTIONS_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_DEPOSITION_ACTIONS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/profiling.hpp>

#include <tpic/communication/mass_halo/mass_halo_plan.hpp>
#include <tpic/config/numerics_options.hpp>

#include <tpic/ecsim/deposition/current_deposition_device.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp>
#include <tpic/ecsim/deposition/moment_field_device.hpp>
#include <tpic/ecsim/deposition/current_assembly_device.hpp>
#include <tpic/simulation/actions/moment_submission_error.hpp>
#include <tpic/simulation/execution/active_backend_device.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include <utility>

namespace tpic::simulation {

// Deposits immutable resident and completed ranges and assembles the moments;
// it never mutates migration. CurrentAssembly binds the current stream at
// construction and the mass halo at registration.
template <class Precision, bool Partial, config::MassDepositionKernel Kernel, class SpeciesPack, class Ports>
class DepositionActions {
public:
  DepositionActions(const simulation::PreparedInput& input,
      const SpeciesPack& species, const data::MagneticNodeOwner<Precision, Backend>& magnetic,
      const data::MagneticNodeOwner<Precision, Backend>& next_magnetic,
      data::CurrentOwner<Precision, Backend>& current, data::CompactMassMatrix<Precision, Backend>& mass,
      Ports& ports, Executor current_executor)
      : input_(input), species_(species), magnetic_node_(magnetic),
        next_node_(next_magnetic), current_(current), mass_(mass),
        mass_halo_(ports.mass_storage()), mass_add_(ports.template get<"mass.add">()),
        mass_ghosts_(ports.template get<"mass.ghosts">()),
        current_assembly_(current_, input.domain, input.physics.smoothing_enabled,
            ports.template get<"J.add">(), ports.template get<"J.ghosts">(), current_executor),
        deposition_cells_(data::prepare_particle_cell_map(input.domain)) {}

  template <bool Diagnose> void clear(MomentQuantity quantity, int cycle, bool bootstrap, Executor executor) {
    submit_moment<Diagnose>(
        [&](auto... diagnostic) {
          return quantity == MomentQuantity::current
              ? ecsim::enqueue_clear_moment_field(current_, executor, diagnostic...)
              : ecsim::enqueue_clear_mass_matrix(mass_, executor, diagnostic...);
        },
        "TPIC deposition: total moment clear failed", "clear", quantity, nullptr, bootstrap, cycle);
  }

  template <bool Diagnose, class Residents>
  void deposit_resident(MomentQuantity quantity, const DepositionInput& job, const Residents& read,
                        int cycle, unsigned* invalid, Executor executor) {
    const double beta_over_c = input_.parameters.species[job.species].q_over_m * input_.parameters.dt /
                               (2 * input_.parameters.c);
    submit_moment<Diagnose>(
        [&](auto... diagnostic) {
          if (quantity == MomentQuantity::current)
            return ecsim::enqueue_resident_current<Precision>(
                read, next_node_, current_, deposition_cells_,
                input_.domain.inverse_cell_volume, beta_over_c,
                invalid, executor, diagnostic...);
          return ecsim::enqueue_resident_mass_matrix<Precision, Kernel>(
              read, next_node_, mass_, deposition_cells_, beta_over_c,
              invalid, executor, diagnostic...);
        },
        "TPIC deposition: resident deposition submission failed",
        "resident", quantity, &job, false, cycle);
  }

  template <bool Diagnose>
  void deposit(MomentQuantity quantity, const DepositionInput& job, bool bootstrap, int cycle,
               unsigned* invalid, Executor executor) {
    // The pack is homogeneous: the job's species index selects the owner.
    const auto& species = species_[job.species];
    const double beta_over_c = input_.parameters.species[job.species].q_over_m * input_.parameters.dt /
                               (2 * input_.parameters.c);
    const auto& magnetic = bootstrap ? magnetic_node_ : next_node_;
    const auto* receipt = input_.particles.order != config::ParticleOrder::unsorted
        ? &species.ordering.receipt() : nullptr;
    const bool sorted = job.phase == DepositionPhase::resident ||
                        (job.phase == DepositionPhase::whole && receipt != nullptr);
    if (sorted) {
      lpfx::require(receipt && receipt->valid() && receipt->prefix_size() == job.range.count,
                    "TPIC deposition: completed sorted range lacks matching receipt");
      submit_moment<Diagnose>(
          [&](auto... diagnostic) {
            if (quantity == MomentQuantity::current)
              return ecsim::enqueue_sorted_current<Precision>(
                  species.particles, *receipt, magnetic, current_,
                  deposition_cells_, input_.domain.inverse_cell_volume, beta_over_c,
                  invalid, executor, diagnostic...);
            return ecsim::enqueue_sorted_mass_matrix<Precision, Kernel>(
                species.particles, *receipt, magnetic, mass_, deposition_cells_,
                beta_over_c, invalid, executor, diagnostic...);
          },
          "TPIC deposition: completed sorted deposition submission failed",
          "completed-sorted", quantity, &job, bootstrap, cycle);
    } else {
      submit_moment<Diagnose>(
          [&](auto... diagnostic) {
            return quantity == MomentQuantity::current
                ? ecsim::enqueue_unsorted_current<Precision>(
                      species.particles, magnetic, current_, deposition_cells_,
                      input_.domain.inverse_cell_volume, beta_over_c, job.range.first, job.range.count, invalid, executor,
                      diagnostic...)
                : ecsim::enqueue_unsorted_mass_matrix<Precision>(
                      species.particles, magnetic, mass_, deposition_cells_,
                      beta_over_c, job.range.first, job.range.count, invalid, executor,
                      diagnostic...);
          },
          "TPIC deposition: completed range deposition submission failed",
          "completed-unsorted", quantity, &job, bootstrap, cycle);
    }
  }
  void begin_assembly(MomentQuantity quantity) {
    if (quantity == MomentQuantity::current) current_assembly_.begin();
    else mass_halo_.begin(mass_add_);
  }
  // The mass reaches interior before complete.
  [[nodiscard]] communication::HaloProgress progress_assembly(MomentQuantity quantity) {
    LPFX_PROFILE_SCOPE(quantity == MomentQuantity::current
        ? "TPIC::CurrentAssembly" : "TPIC::MassAssembly");
    if (quantity == MomentQuantity::current)
      return current_assembly_.progress() ? communication::HaloProgress::complete
                                          : communication::HaloProgress::pending;
    const auto progress = mass_halo_.progress(mass_add_, mass_ghosts_);
    if (progress == communication::HaloProgress::complete) mass_halo_.conclude();
    return progress;
  }

private:
  template <bool Diagnose, class Submission>
  LPFX_ALWAYS_INLINE void submit_moment(
      Submission&& submission, const char* fast_failure, const char* operation,
      MomentQuantity quantity, const DepositionInput* input, bool bootstrap, int cycle) {
    if constexpr (Diagnose) {
      lpfx::device::DeviceSubmissionDiagnostic diagnostic;
      const auto result = std::forward<Submission>(submission)(&diagnostic);
      check_moment_submission(result, diagnostic, operation, quantity, input, bootstrap, cycle);
    } else {
      lpfx::check_error(std::forward<Submission>(submission)(), fast_failure);
    }
  }
  const simulation::PreparedInput& input_;
  const SpeciesPack& species_;
  const data::MagneticNodeOwner<Precision, Backend>& magnetic_node_;
  const data::MagneticNodeOwner<Precision, Backend>& next_node_;
  data::CurrentOwner<Precision, Backend>& current_;
  data::CompactMassMatrix<Precision, Backend>& mass_;
  typename Ports::MassStorage& mass_halo_;
  typename Ports::template Operation<"mass.add">& mass_add_;
  typename Ports::template Operation<"mass.ghosts">& mass_ghosts_;
  ecsim::CurrentAssembly<data::CurrentOwner<Precision, Backend>, Backend,
                         typename Ports::template Operation<"J.add">,
                         typename Ports::template Operation<"J.ghosts">>
      current_assembly_;
  const data::ParticleCellMap deposition_cells_;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_DEPOSITION_ACTIONS_DEVICE_HPP
