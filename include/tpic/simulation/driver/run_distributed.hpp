#ifndef TPIC_SIMULATION_DRIVER_RUN_DISTRIBUTED_HPP
#define TPIC_SIMULATION_DRIVER_RUN_DISTRIBUTED_HPP

#include <tpic/simulation/planning/run_plan.hpp>

#include <string_view>

namespace lpfx { class MpiCallDomain; }

namespace tpic::simulation {

// Owns the MPI/device bootstrap and the selected simulation's lifetime. The
// rank and communicator size of the options come from the runtime. All ranks
// must agree on the run mode, particle order, precision, mass kernel and
// application, output replacement, restart and the compiled payload backend.
int run_distributed(int argc, char** argv, std::string_view path, InputOptions options, RunMode mode);

// Collective informational sample after the owners and communication exist.
// A failed native memory query is reported as unavailable, never an admission
// failure. Free/used bytes are shared by all processes on the selected GPU.
void report_initialized_memory(const lpfx::MpiCallDomain& domain, int device);

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_DRIVER_RUN_DISTRIBUTED_HPP
