#include <tpic/simulation/planning/run_plan.hpp>

#include <iostream>
#include <stdexcept>

int main() {
  try {
    using tpic::simulation::RuntimeDebugOptions;
    unsetenv("TPIC_PROFILE_STAGES");
    unsetenv("TPIC_DIAGNOSTIC_SYNC_DEPOSITION");
    const auto disabled = RuntimeDebugOptions::from_environment();
    if (disabled.profile_stages || disabled.synchronous_deposition)
      throw std::runtime_error("debug mode unexpectedly enabled by default");
    for (const auto value : {"", "0", "01", "true", "1 ", "11"}) {
      setenv("TPIC_PROFILE_STAGES", value, 1);
      setenv("TPIC_DIAGNOSTIC_SYNC_DEPOSITION", value, 1);
      const auto options = RuntimeDebugOptions::from_environment();
      if (options.profile_stages || options.synchronous_deposition)
        throw std::runtime_error("debug options must require exactly 1");
    }
    setenv("TPIC_PROFILE_STAGES", "1", 1);
    setenv("TPIC_DIAGNOSTIC_SYNC_DEPOSITION", "0", 1);
    const auto profiling = RuntimeDebugOptions::from_environment();
    if (!profiling.profile_stages || profiling.synchronous_deposition)
      throw std::runtime_error("profiling changed deposition policy");
    setenv("TPIC_PROFILE_STAGES", "0", 1);
    setenv("TPIC_DIAGNOSTIC_SYNC_DEPOSITION", "1", 1);
    const auto diagnostic = RuntimeDebugOptions::from_environment();
    if (diagnostic.profile_stages || !diagnostic.synchronous_deposition ||
        !profiling.profile_stages || profiling.synchronous_deposition)
      throw std::runtime_error(
          "options were not a frozen independent snapshot");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
