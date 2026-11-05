#include <tpic/simulation/planning/run_plan.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using tpic::simulation::RunMode;
using tpic::simulation::RunPlan;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class F> void rejects(F action, const char *expected) {
  try {
    action();
  } catch (const std::invalid_argument &error) {
    require(std::string(error.what()).find(expected) != std::string::npos,
            "admission lost its rejection context");
    return;
  }
  throw std::runtime_error("unsupported input was admitted");
}
tpic::config::SimulationParameters fixture() {
  tpic::config::SimulationParameters p;
  p.dt = .1;
  p.th = .5;
  p.ncycles = 2;
  p.ns = 1;
  p.grid.nxc = p.grid.nyc = p.grid.nzc = 2;
  p.grid.lx = p.grid.ly = p.grid.lz = 2;
  p.output.field_cycle = 0;
  p.output.restart_cycle = 0;
  p.species.resize(1);
  auto &species = p.species[0];
  species.q_over_m = -1;
  species.rho_init = 1;
  species.npcelx = species.npcely = species.npcelz = 1;
  return p;
}
void advance(const tpic::config::SimulationParameters &p,
             tpic::config::ParticleOrder order = tpic::config::ParticleOrder::full) {
  (void)RunPlan::prepare(p, {.order = order}, RunMode::advance);
}
} // namespace
int main() {
  try {
    using namespace tpic;
    for (const auto order :
         {config::ParticleOrder::full, config::ParticleOrder::partial,
          config::ParticleOrder::unsorted})
      advance(fixture(), order);
    auto p = fixture();
    p.output.test_particles_cycle = 1;
    rejects([&] { advance(p); }, "unsupported requests");
    p = fixture();
    p.topology.periodic_x = false;
    p.bc.em_x_right = 2;
    rejects([&] { advance(p); }, "open electromagnetic");
    rejects([&] { (void)RunPlan::prepare(p, {}, RunMode::initialize_only); }, "open electromagnetic");
    // A magnetic conductor is admitted with and without smoothing.
    p.bc.em_x_right = 1;
    advance(p);
    p.ecsim_smoothing_mode = config::EcsimSmoothingMode::EnergyConserving;
    p.smooth = .5;
    advance(p);
    p.ecsim_smoothing_mode = config::EcsimSmoothingMode::None;
    for (const auto *engine : {"adios2", "hdf5"}) {
      const auto kind = std::string(engine) == "adios2" ? config::OutputEngineKind::Adios2
                                                        : config::OutputEngineKind::Hdf5;
      p = fixture();
      p.field_write_method = engine;
      p.particles_write_method = "vtk";
      p.output.field_cycle = 1;
      const auto check = [&](const char *channel) {
        if (config::output_engine_available(kind)) advance(p);
        else rejects([&] { advance(p); }, channel);
      };
      check("field output backend");
      p.output.field_cycle = 0;
      p.output.particles_cycle = 3;
      advance(p); // Disabled field backend is irrelevant to VTK particles.
      p.field_write_method = "vtk";
      p.particles_write_method = engine;
      check("particle output backend");
      p.output.particles_cycle = 0;
      p.output.field_cycle = 1;
      advance(p); // Disabled particle backend is irrelevant to VTK fields.
      p.output.field_cycle = 0;
      p.field_write_method = engine;
      advance(p);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
