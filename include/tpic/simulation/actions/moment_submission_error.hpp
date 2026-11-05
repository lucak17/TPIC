#ifndef TPIC_SIMULATION_ACTIONS_MOMENT_SUBMISSION_ERROR_HPP
#define TPIC_SIMULATION_ACTIONS_MOMENT_SUBMISSION_ERROR_HPP

#include <lpfx/backend/launch.hpp>
#include <tpic/simulation/execution/execution_layout.hpp>

namespace tpic::simulation {

// Cold host formatting; native status capture remains at the submitting thread.
// Borrowed descriptor/native strings are consumed synchronously on failure.
void check_moment_submission(
    lpfx::ErrorCode code, const lpfx::device::DeviceSubmissionDiagnostic &diagnostic,
    const char *operation, MomentQuantity quantity, const DepositionInput *input,
    bool bootstrap, int destination_cycle);

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_MOMENT_SUBMISSION_ERROR_HPP
