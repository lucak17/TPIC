#include <tpic/simulation/actions/moment_submission_error.hpp>

#include <lpfx/backend/error_reporting.hpp>

#include <cstdio>

namespace tpic::simulation {
namespace {
const char *quantity_name(MomentQuantity quantity) noexcept {
  switch (quantity) {
  case MomentQuantity::current:
    return "current";
  case MomentQuantity::mass:
    return "mass";
  }
  return "unknown";
}
const char *deposition_phase_name(DepositionPhase phase) noexcept {
  switch (phase) {
  case DepositionPhase::resident:
    return "resident";
  case DepositionPhase::tail:
    return "tail";
  case DepositionPhase::whole:
    return "whole";
  }
  return "unknown";
}
} // namespace

void check_moment_submission(
    lpfx::ErrorCode code, const lpfx::device::DeviceSubmissionDiagnostic &diagnostic,
    const char *operation, MomentQuantity quantity, const DepositionInput *input,
    bool bootstrap, int destination_cycle) {
  if (code == lpfx::ErrorCode::success)
    return;

  char descriptor[160]{};
  if (input != nullptr) {
    std::snprintf(
        descriptor, sizeof(descriptor),
        "phase=%s descriptor_species=%u range=%u+%u/%u ",
        deposition_phase_name(input->phase),
        input->species, input->range.first, input->range.count,
        input->range.total);
  }
  char message[512];
  std::snprintf(message, sizeof(message),
                "TPIC deposition: moment submission failed operation=%s quantity=%s "
                "%sbootstrap=%u destination_cycle=%d diagnostic_sync=1",
                operation, quantity_name(quantity), descriptor, bootstrap ? 1U : 0U,
                destination_cycle);
  lpfx::device::check_device_submission(code, diagnostic, message);
}

} // namespace tpic::simulation
