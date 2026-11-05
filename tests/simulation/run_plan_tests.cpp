#include <tpic/simulation/planning/run_plan.hpp>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
using tpic::simulation::RunMode;
using tpic::simulation::RunPlan;

auto parameters() {
  tpic::config::SimulationParameters p;
  p.dt = .125;
  p.ncycles = 2;
  p.ns = 2;
  p.grid = {.lx = 7, .ly = 3, .lz = 2, .nxc = 7, .nyc = 3, .nzc = 2};
  p.species.resize(2);
  for (auto& species : p.species) {
    species.rho_init = 1;
    species.npcelx = species.npcely = species.npcelz = 2;
    species.q_over_m = 1;
  }
  p.species[0].q_over_m = -25;
  return p;
}

void require(bool condition) {
  if (!condition) throw std::runtime_error("run-plan contract failed");
}

template <class F> void rejects(F action) {
  try { action(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid run plan request admitted");
}
}

int main() {
  static_assert(std::is_const_v<decltype(RunPlan::input)>);
  auto raw = parameters();
  const auto plan = RunPlan::prepare(raw, {}, RunMode::advance);
  raw.species[0].npcelx = 99;
  require(plan.input.parameters.species[0].npcelx == 2);
  require(plan.particle_loads[0].count == 7 * 3 * 2 * 8);
  require(plan.particle_loads[1].first_id == plan.particle_loads[0].count);
  const auto again = RunPlan::prepare(parameters(), {}, RunMode::advance);
  require(again.input.particles == plan.input.particles &&
          again.particle_loads[0].seed == plan.particle_loads[0].seed);

  // Output backend availability is checked for advancing runs only.
  raw = parameters();
  raw.output.field_cycle = 1;
  raw.write_method = "adios2";
  const auto host = tpic::simulation::prepare_input(raw);
  require(host.output.fields_enabled() && host.output.field_engine == tpic::config::OutputEngineKind::Adios2 &&
          host.output.particle_engine == tpic::config::OutputEngineKind::Adios2);
  {
    (void)RunPlan::prepare(raw, {}, RunMode::initialize_only);
    if (tpic::config::output_engine_available(tpic::config::OutputEngineKind::Adios2))
      (void)RunPlan::prepare(raw, {}, RunMode::advance);
    else
      rejects([&] { (void)RunPlan::prepare(raw, {}, RunMode::advance); });
  }
  raw.field_write_method = "vtk";
  raw.particles_write_method = "hdf5";
  // The unavailable particle engine is harmless while particle output is off.
  const auto mixed = RunPlan::prepare(raw, {}, RunMode::advance);
  require(mixed.input.output.field_engine == tpic::config::OutputEngineKind::Vtk &&
          mixed.input.output.particle_engine == tpic::config::OutputEngineKind::Hdf5);
}
