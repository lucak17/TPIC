#include <tpic/simulation/actions/moment_submission_error.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void contains(const std::string &message, const char *text) {
  if (message.find(text) == std::string::npos)
    throw std::runtime_error(std::string("submission report omitted ") + text);
}
std::string failure(lpfx::ErrorCode code,
                    const lpfx::device::DeviceSubmissionDiagnostic &diagnostic,
                    const char *operation, tpic::simulation::MomentQuantity quantity,
                    const tpic::simulation::DepositionInput *input, bool bootstrap, int cycle) {
  try {
    tpic::simulation::check_moment_submission(code, diagnostic, operation, quantity, input,
                                              bootstrap, cycle);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  throw std::runtime_error("failed submission did not throw");
}
} // namespace
int main() {
  try {
    using namespace tpic;
    const simulation::DepositionInput input{
        11, 3, simulation::DepositionPhase::tail, {7, 4, 11}};
    lpfx::device::DeviceSubmissionDiagnostic native{
        lpfx::device::DeviceSubmissionStage::completion, 42, "native_failure",
        "stream error"};
    simulation::check_moment_submission(lpfx::ErrorCode::success, native, "deposit",
                                        simulation::MomentQuantity::mass, &input, false, 9);
    const auto message = failure(lpfx::ErrorCode::backend_failure, native, "deposit",
                                 simulation::MomentQuantity::mass, &input, false, 9);
    for (const auto text :
         {"operation=deposit", "quantity=mass", "phase=tail",
          "descriptor_species=3", "range=7+4/11", "bootstrap=0",
          "destination_cycle=9", "lpfx=backend_failure", "diagnostic_sync=1",
          "native_stage=completion", "native_code=42",
          "native_name=native_failure", "native_message=stream error"})
      contains(message, text);
    const auto portable = failure(
        lpfx::ErrorCode::invalid_argument, {},
        "clear", simulation::MomentQuantity::current, nullptr, true, 0);
    for (const auto text :
         {"operation=clear", "quantity=current", "bootstrap=1",
          "destination_cycle=0", "lpfx=invalid_argument", "native_stage=none"})
      contains(portable, text);
    native.native_name = nullptr;
    native.native_message = nullptr;
    const auto unnamed = failure(lpfx::ErrorCode::backend_failure, native, "deposit",
                                 simulation::MomentQuantity::mass, &input, false, 9);
    contains(unnamed, "native_name=unavailable");
    contains(unnamed, "native_message=unavailable");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
