#include <tpic/simulation/driver/simulation_device.hpp>

// The Simulation instantiations of one precision pair: every order and every
// mass kernel the pair admits, at the species capacity config::max_real_species.
// Compiled once per compiled precision pair, with TPIC_DISPATCH_STORAGE
// and TPIC_DISPATCH_COMPUTE naming the pair's two structs.
namespace tpic::simulation {

template RunResult run_simulation<config::Precision<TPIC_DISPATCH_STORAGE, TPIC_DISPATCH_COMPUTE>>(
    const RunPlan& plan, const lpfx::distributed::CommunicationSetup& setup,
    lpfx::device::ActiveExecutor executor);

} // namespace tpic::simulation
