#include <tpic/simulation/planning/run_plan.hpp>

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using tpic::simulation::prepare_input;
using namespace tpic::config;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

SimulationParameters fixture(int count = 2) {
  SimulationParameters p;
  p.dt = .1;
  p.ncycles = 4;
  p.th = .5;
  p.ns = count;
  p.species.resize(count);
  p.grid.nxc = 7; p.grid.nyc = 3; p.grid.nzc = 2;
  p.grid.lx = 7; p.grid.ly = 3; p.grid.lz = 2;
  p.topology.xlen = 2;
  p.output.restart_cycle = 0;
  for (int index = 0; index < count; ++index) {
    auto &species = p.species[index];
    species.q_over_m = index % 2 ? 1 : -25;
    species.rho_init = 1;
    species.npcelx = index + 1;
    species.npcely = species.npcelz = 1;
  }
  return p;
}

template<class F> void rejects(F action, const char *expected) {
  try { action(); }
  catch (const std::exception &error) {
    require(std::string(error.what()).find(expected) != std::string::npos,
            "missing admission error context");
    return;
  }
  throw std::runtime_error("invalid input admitted");
}

void combined_admission() {
  for (int count = 1; count <= 10; ++count) {
    const auto p = fixture(count);
    for (const auto order : {tpic::config::ParticleOrder::full,
                            tpic::config::ParticleOrder::partial,
                            tpic::config::ParticleOrder::unsorted}) {
      const auto first = prepare_input(p, {.rank = 0, .order = order});
      const auto last = prepare_input(p, {.rank = 1, .communicator_size = 2, .order = order});
      require(first.host_admitted() && last.host_admitted(), "baseline must pass host admission");
      require(first.communicator_size == 2 && first.parameters.ns == count &&
                  first.particles.species.size() == static_cast<std::size_t>(count),
              "startup did not retain the selected species");
      require(first.particles.local_count + last.particles.local_count == first.particles.global_count,
              "nondivisible rank counts do not add to the global total");
      require(first.output.species_selected[tpic::config::output_rho_quantity].size() == static_cast<std::size_t>(count),
              "output mask and particle species disagree");
      require(first.particles.order == order, "particle order was lost");
    }
  }
  for (int count : {-1, 0, 11, std::numeric_limits<int>::max()}) {
    auto p = fixture();
    p.ns = count;
    rejects([&] { (void)prepare_input(p); }, "supported real species range: 1-10");
  }
  {
    auto p = fixture();
    p.species.clear();
    rejects([&] { (void)prepare_input(p); }, "exactly ns");
  }
  rejects([] { (void)prepare_input(fixture(), {.communicator_size = 3}); }, "communicator_size");
  rejects([] { (void)prepare_input(fixture(), {.rank = 2}); }, "rank");
  auto p = fixture();
  p.output.moments_tag = "rho9999999999999999999999";
  p.output.field_cycle = 0;
  rejects([&] { (void)prepare_input(p); }, "species index");
  p = fixture();
  p.case_name = "Whistler";
  rejects([&] { (void)prepare_input(p); }, "B0x");
}

void unsupported_requests() {
  auto p = fixture();
  p.output.particles_cycle = 100;
  p.output.test_particles_cycle = 2;
  const auto result = prepare_input(p);
  require(!result.host_admitted() && result.unsupported_requests.size() == 1 &&
              result.warnings.empty() && result.output.particles_enabled(),
          "test-particle output must fail");
  // The caller's parameters and their resolved values stay intact.
  require(p.output.test_particles_cycle == 2 && result.parameters.output.test_particles_cycle == 2,
          "unsupported output was silently disabled");
  p.output.test_particles_cycle = 0;
  const auto optional_output = prepare_input(p);
  require(optional_output.host_admitted() && optional_output.warnings.empty() &&
              optional_output.output.particles_enabled(),
          "particle output must be enabled");
  p.output.restart_cycle = 10;
  p.restart_write_method = "pvtk";
  require(!prepare_input(p).host_admitted(), "VTK checkpoints must be unsupported");
  p = fixture(3);
  p.case_name = "Whistler";
  p.b0x = .1;
  p.whistler_mode_x = 2;
  p.whistler_use_exact_cold_root = false;
  p.whistler_seed_epsilon = .01;
  require(!prepare_input(p).host_admitted(), "duplicate electron wave current admitted");
  p.whistler_seed_epsilon = 0;
  const auto unseeded = prepare_input(p);
  require(unseeded.host_admitted() && unseeded.warnings.empty(), "unseeded multi-species Whistler rejected or warned");
  auto single = p; // One electron species: an eigenmode seed is admitted.
  single.ns = 2;
  single.species.pop_back();
  single.whistler_seed_epsilon = .01;
  const auto exact = prepare_input(single);
  require(exact.host_admitted() && exact.warnings.empty(), "exact Whistler eigenmode rejected or warned");
  single.species[1].q_over_m = 2;
  require(prepare_input(single).warnings.size() == 1, "inexact Whistler eigenmode was not warned");
  auto gaussian = fixture(); // Centred on node 0, its subcell centres 0.5 away.
  gaussian.case_name = "GaussianExpansion";
  gaussian.quiet_start = true;
  gaussian.species[1].npcelx = 1;
  gaussian.gaussian_expansion_sigma_x = 1;
  gaussian.gaussian_expansion_cutoff = 4;
  require(prepare_input(gaussian).warnings.empty(), "loaded GaussianExpansion support warned");
  gaussian.gaussian_expansion_cutoff = .1;
  require(prepare_input(gaussian).warnings.size() == 2, "empty GaussianExpansion supports were not warned");
  p.whistler_seed_mode = WhistlerSeedMode::MagneticOnly;
  p.whistler_seed_epsilon = .01;
  require(prepare_input(p).host_admitted(), "magnetic-only seed has no repeated wave current");
  p.case_name = "TaylorGreenVortex";
  p.grid.lx = p.grid.ly = p.grid.lz = 4;
  p.taylor_green_vortex_magnetic_amplitude = .01;
  require(!prepare_input(p).host_admitted(), "repeated same-sign TGV current admitted");
  p.taylor_green_vortex_magnetic_amplitude = 0;
  require(prepare_input(p).host_admitted(), "zero magnetic TGV seed rejected");
}

void mass_matrix_selection() {
  using tpic::config::MassDepositionKernel;
  using tpic::config::ParticleOrder;
  require(prepare_input(fixture()).mass_matrix == MassDepositionKernel::scalar,
          "scalar mass must remain the default");
  for (const auto kernel : {MassDepositionKernel::matrix_core})
    for (const auto order : {ParticleOrder::full, ParticleOrder::partial, ParticleOrder::unsorted}) {
      const auto result = prepare_input(fixture(), {.order = order, .mass_matrix = kernel});
      require(result.mass_matrix == kernel, "mass kernel selection was lost");
      const bool supported = TPIC_CONFIG_MATRIX_MASS && order != ParticleOrder::unsorted;
      require(result.host_admitted() == supported,
          "matrix kernel backend/order admission mismatch");
    }
  rejects([] { (void)prepare_input(fixture(), {.mass_matrix = static_cast<MassDepositionKernel>(99)}); },
          "invalid kernel");
}

void mass_apply_selection() {
  using namespace tpic::ecsim;
  static_assert(resolve_maxwell_apply(MassApplyMode::full, false) == MaxwellApplySchedule::full);
  static_assert(resolve_maxwell_apply(MassApplyMode::split, false) == MaxwellApplySchedule::split);
  static_assert(resolve_maxwell_apply(MassApplyMode::full, true) == MaxwellApplySchedule::smoothed);
  static_assert(resolve_maxwell_apply(MassApplyMode::split, true) == MaxwellApplySchedule::smoothed);
  require(tpic::simulation::InputOptions{}.mass_apply == MassApplyMode::full &&
              prepare_input(fixture()).mass_apply == MassApplyMode::full,
          "full mass apply must remain the default");
  for (const auto mode : {MassApplyMode::full, MassApplyMode::split}) {
    require(valid_mass_apply_mode(mode) && parse_mass_apply_mode(mass_apply_mode_name(mode)) == mode,
            "mass apply mode does not roundtrip");
    for (bool smoothing : {false, true}) {
      auto p = fixture();
      if (smoothing) {
        p.ecsim_smoothing_mode = EcsimSmoothingMode::EnergyConserving;
        p.smooth = .5;
        p.smooth_niter = 2;
      }
      const auto input = prepare_input(p, {.mass_apply = mode});
      require(input.host_admitted() && input.mass_apply == mode &&
                  input.physics.smoothing_enabled == smoothing,
              "mass apply selection changed configuration admission");
      const auto effective = resolve_maxwell_apply(input.mass_apply, input.physics.smoothing_enabled);
      require(input.maxwell_schedule == effective, "prepared input stored another Maxwell schedule");
      require(effective == (smoothing ? MaxwellApplySchedule::smoothed :
                   mode == MassApplyMode::full ? MaxwellApplySchedule::full : MaxwellApplySchedule::split),
              "mass apply selection chose the wrong effective schedule");
      require(maxwell_apply_schedule_name(effective) ==
                  (smoothing ? "smoothed full-domain" : mass_apply_mode_name(mode)),
              "mass apply effective report differs from the selected schedule");
    }
    for (const auto kernel : {MassDepositionKernel::scalar, MassDepositionKernel::matrix_core}) {
      const auto input = prepare_input(fixture(), {.mass_matrix = kernel, .mass_apply = mode});
      require(input.mass_apply == mode && input.mass_matrix == kernel,
              "mass apply and deposition selection interfered");
    }
  }
  rejects([] { (void)parse_mass_apply_mode("smoothed"); }, "--mass-apply must be");
  const auto invalid = static_cast<MassApplyMode>(99);
  require(!valid_mass_apply_mode(invalid), "invalid mass apply mode passed validation");
  rejects([&] { (void)mass_apply_mode_name(invalid); }, "invalid mass apply mode");
  for (bool smoothing : {false, true}) {
    rejects([&] { (void)resolve_maxwell_apply(invalid, smoothing); }, "invalid mass apply mode");
    auto p = fixture();
    if (smoothing) {
      p.ecsim_smoothing_mode = EcsimSmoothingMode::EnergyConserving;
      p.smooth = .5;
      p.smooth_niter = 2;
    }
    rejects([&] { (void)prepare_input(p, {.mass_apply = invalid}); }, "invalid mass apply mode");
  }
  rejects([] { (void)maxwell_apply_schedule_name(static_cast<MaxwellApplySchedule>(99)); },
          "TPIC Maxwell: invalid apply schedule");
}

} // namespace

int main() {
  try {
    combined_admission();
    unsupported_requests();
    mass_matrix_selection();
    mass_apply_selection();
    std::cout << "TPIC combined host input checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
