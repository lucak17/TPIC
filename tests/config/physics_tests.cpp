#include <tpic/config/physics.hpp>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace tpic::config;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

SimulationParameters fixture() {
  SimulationParameters p;
  p.dt = .125;
  p.ncycles = 16;
  p.th = .5;
  p.ns = 1;
  p.species.resize(1);
  p.species[0].q_over_m = -1;
  p.species[0].rho_init = 1;
  return p;
}

template <class F> void rejects(F change, const char *expected) {
  auto p = fixture();
  change(p);
  try { (void)prepare_physics(p); }
  catch (const std::invalid_argument &error) {
    require(std::string(error.what()).find(expected) != std::string::npos,
            "missing physics error context");
    return;
  }
  throw std::runtime_error("invalid physics accepted");
}

void boundaries() {
  auto p = fixture();
  p.bc.em_x_right = 2;
  p.bc.particle_x_right = 4;
  auto resolved = prepare_physics(p);
  require(resolved.field_boundary[0] == FieldBoundary::periodic &&
              resolved.particle_boundary[0] == ParticleBoundary::periodic &&
              resolved.unsupported_requests.empty(), "periodicity must override codes");
  p.topology.periodic_x = false;
  p.bc.em_x_left = 1;
  p.topology.periodic_y = false;
  p.bc.em_y_left = 0;
  resolved = prepare_physics(p);
  require(resolved.field_boundary[0] == FieldBoundary::open &&
              resolved.field_boundary[1] == FieldBoundary::perfect_magnetic_conductor &&
              resolved.field_boundary[2] == FieldBoundary::perfect_electric_conductor &&
              resolved.field_boundary[3] == FieldBoundary::perfect_electric_conductor,
          "wrong face ordering");
  require(resolved.particle_boundary[0] == ParticleBoundary::periodic,
          "particle periodicity must remain independent");
  p.topology.periodic_x_particles = false;
  resolved = prepare_physics(p);
  require(resolved.particle_boundary[0] == ParticleBoundary::mirror &&
              resolved.unsupported_requests.size() == 1 &&
              resolved.unsupported_requests[0].find("injection is not implemented") != std::string::npos,
          "unwired particle inflow code must be reported");
  // An injection code counts only on a physical face.
  for (const int code : {3, 4}) {
    p = fixture();
    p.bc.particle_x_right = code;
    require(prepare_physics(p).unsupported_requests.empty(),
            "periodic particle faces must override injection codes");
    p.topology.periodic_x_particles = false;
    require(prepare_physics(p).unsupported_requests.size() == 1,
            "an injection face must be unsupported");
  }
  // An unknown EM face code is rejected, on periodic and physical faces alike.
  for (const int code : {-1, 3, 7}) {
    for (const bool periodic : {true, false}) {
      p = fixture();
      p.topology.periodic_x = periodic;
      p.bc.em_x_left = code;
      bool rejected = false;
      try {
        (void)prepare_physics(p);
      } catch (const std::invalid_argument &error) {
        rejected = std::string(error.what()).find("bcEMfaceXleft=" + std::to_string(code) +
                                                  ": unknown code") != std::string::npos;
      }
      require(rejected, "an unknown EM face code must be rejected");
    }
  }
}

void smoothing() {
  auto p = fixture();
  p.topology.periodic_x = false;
  p.bc.em_x_right = 2;
  p.smooth = -5;
  p.smooth_niter = -2;
  require(!prepare_physics(p).smoothing_enabled,
          "inactive smoothing controls must not affect physics");
  p.ecsim_smoothing_mode = EcsimSmoothingMode::EnergyConserving;
  p.smooth = 1;
  p.smooth_niter = 3;
  require(!prepare_physics(p).smoothing_enabled, "unit alpha is inactive");
  p.smooth = .5;
  p.smooth_niter = 0;
  require(!prepare_physics(p).smoothing_enabled, "zero iterations are inactive");
  p.smooth_niter = 3;
  p.bc.em_x_right = 1;
  require(prepare_physics(p).smoothing_enabled, "a magnetic conductor is admitted by the smoothing predicate");
  rejects([](auto &v) { v.ecsim_smoothing_mode = EcsimSmoothingMode::EnergyConserving;
    v.smooth = .5; v.topology.periodic_z = false; v.bc.em_z_left = 2;
  }, "effective open");
  rejects([](auto &v) { v.ecsim_smoothing_mode = EcsimSmoothingMode::EnergyConserving;
    v.smooth = std::numeric_limits<double>::quiet_NaN(); }, "Smooth");
  rejects([](auto &v) { v.ecsim_smoothing_mode = EcsimSmoothingMode::EnergyConserving;
    v.smooth_niter = -1; }, "SmoothNiter");
}

void physics_safety() {
  const auto p = fixture();
  const auto resolved = prepare_physics(p);
  require(resolved.theta_dt == .0625 && resolved.c_theta_dt == .0625 &&
              resolved.final_time == 2 && resolved.gmres_max_iterations == 100 &&
              resolved.gmres_restart == 30, "linear solver controls changed");
  rejects([](auto &v) { v.dt = 0; }, "dt");
  rejects([](auto &v) { v.dt = std::numeric_limits<double>::infinity(); }, "dt");
  rejects([](auto &v) { v.dt = std::numeric_limits<double>::denorm_min(); }, "th*dt");
  rejects([](auto &v) { v.dt = 1e308; }, "dt*ncycles");
  rejects([](auto &v) { v.th = .49; }, "th");
  rejects([](auto &v) { v.th = 1.01; }, "th");
  rejects([](auto &v) { v.ncycles = -1; }, "ncycles");
  rejects([](auto &v) { v.c = -1; }, "c must be finite and positive");
  rejects([](auto &v) { v.gmres_tol = 0; }, "GMREStol");
  rejects([](auto &v) { v.gmres_tol = 1; }, "GMREStol");
  rejects([](auto &v) { v.b0y = std::numeric_limits<double>::infinity(); }, "B0y");
  rejects([](auto &v) { v.species[0].q_over_m = 0; }, "neutral");
  rejects([](auto &v) { v.species[0].q_over_m = std::numeric_limits<double>::denorm_min(); }, "representable");
  rejects([](auto &v) { v.species[0].rho_init = -1; }, "rhoINIT");
  rejects([](auto &v) { v.species[0].wth = -1; }, "wth");
  rejects([](auto &v) { v.species[0].v0 = std::numeric_limits<double>::quiet_NaN(); }, "v0");
  auto unsupported = fixture();
  unsupported.c = 2;
  unsupported.ns_test_particles = 1;
  require(prepare_physics(unsupported).unsupported_requests.size() == 2,
          "unsupported physics requests were silently ignored");
}

} // namespace

int main() {
  try {
    boundaries();
    smoothing();
    physics_safety();
    std::cout << "TPIC physics admission checks passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
