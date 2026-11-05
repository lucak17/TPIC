#ifndef TPIC_SIMULATION_ACTIONS_FIELD_ADVANCE_ACTIONS_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_FIELD_ADVANCE_ACTIONS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/distributed/field/submission.hpp>
#include <tpic/ecsim/maxwell/maxwell_device.hpp>
#include <tpic/simulation/actions/magnetic_startup_device.hpp>
#include <tpic/simulation/execution/active_backend_device.hpp>
#include <tpic/simulation/execution/execution_layout.hpp>

#include <utility>

namespace tpic::simulation {

// Advances the fields and commits them after the mover's old-B readers retire.
// Owns the next-cell halo operation and its phases; the field-update stream
// is bound at construction and shared with the registered halo.
template <class Precision, class Ports>
class FieldAdvanceActions {
  enum class HaloPhase : unsigned char { exchange, conclude, done };
  using Maxwell = ecsim::MaxwellOperator<Precision, Backend>;
  using CellPort = decltype(std::declval<Ports&>().template get<"B.next.cell">());
public:
  FieldAdvanceActions(data::DeviceFieldImage<Precision, Backend>& fields,
      data::MagneticNodeOwner<Precision, Backend>& magnetic, Maxwell& maxwell, Ports& ports,
      Executor control, Executor field_update)
      : fields_(fields), magnetic_(magnetic), maxwell_(maxwell), ports_(ports),
        cell_(ports.template get<"B.next.cell">()), control_executor_(control),
        executor_(field_update) {}
  // Startup: magnetic ghosts and the initial cell-to-node interpolation; a
  // restored B closes its wall ghosts first.
  void complete_initial_halos(bool restored) {
    complete_initial_magnetic_field<Precision>(fields_.magnetic_cell, magnetic_,
        ports_.template get<"B.cell">(), ports_.template get<"B.node">(), control_executor_,
        restored ? &maxwell_.boundary_plans() : nullptr);
  }
  // first_native_submission is observed before conclude retires the accepted
  // field generation.
  void advance(unsigned* invalid) {
    cell_phase_ = HaloPhase::exchange;
    cell_first_observed_ = false;
    maxwell_.enqueue_next_e_and_cell(fields_, invalid, executor_);
    lpfx::check_error(cell_.begin(), "TPIC field advance: next-cell halo begin failed");
  }
  [[nodiscard]] FieldAdvanceProgress progress_cell() {
    if (cell_phase_ == HaloPhase::done) return {true, true};
    if (cell_phase_ == HaloPhase::exchange) {
      if (lpfx::check_progress(cell_.progress(), "TPIC field advance: next-cell halo progress failed"))
        cell_phase_ = HaloPhase::conclude;
      const auto milestone = cell_.first_native_submission();
      lpfx::check_error(milestone, "TPIC field advance: next-cell submission milestone query failed");
      const bool observed = milestone.value() != lpfx::distributed::FieldSubmission::pending;
      if (!cell_first_observed_ && observed) {
        cell_first_observed_ = true;
        return {true, false}; // Signal the mover before a later conclude call.
      }
    }
    if (cell_phase_ == HaloPhase::exchange) return {cell_first_observed_, false};
    lpfx::require(cell_first_observed_,
                  "TPIC field advance: next-cell completion lacks submission milestone");
    if (lpfx::check_progress(cell_.conclude(), "TPIC field advance: next-cell halo conclusion failed"))
      cell_phase_ = HaloPhase::done;
    return {cell_first_observed_, cell_phase_ == HaloPhase::done};
  }
  void close_cell(unsigned* invalid) {
    maxwell_.enqueue_next_cell_closure_and_node_interpolation(invalid, executor_);
  }
  void commit() {
    maxwell_.enqueue_commit_to_current(fields_, magnetic_, executor_);
  }

private:
  data::DeviceFieldImage<Precision, Backend>& fields_;
  data::MagneticNodeOwner<Precision, Backend>& magnetic_;
  Maxwell& maxwell_;
  Ports& ports_;
  CellPort cell_;
  Executor control_executor_, executor_;
  HaloPhase cell_phase_{};
  bool cell_first_observed_{};
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_FIELD_ADVANCE_ACTIONS_DEVICE_HPP
