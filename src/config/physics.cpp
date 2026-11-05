#include <tpic/config/physics.hpp>

#include <tpic/config/detail/errors.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>

namespace tpic::config {
namespace {

constexpr detail::Errors errors{"physics"};

void finite(double value, const std::string &name) {
  errors.require(std::isfinite(value), name + " must be finite");
}

void positive(double value, const std::string &name) {
  errors.require(std::isfinite(value) && value > 0, name + " must be finite and positive");
}

} // namespace

PreparedPhysics prepare_physics(const SimulationParameters &p) {
  PreparedPhysics result;
  positive(p.dt, "dt");
  positive(p.c, "c");
  errors.require(p.ncycles >= 0, "ncycles must be nonnegative");
  errors.require(std::isfinite(p.th) && p.th >= 0.5 && p.th <= 1,
          "th must be in [0.5, 1] for the admitted linear-ECSIM scheme");
  positive(p.gmres_tol, "GMREStol");
  errors.require(p.gmres_tol < 1, "GMREStol must be less than 1");
  result.theta_dt = p.th * p.dt;
  result.c_theta_dt = p.c * result.theta_dt;
  positive(result.theta_dt, "th*dt");
  positive(result.c_theta_dt, "c*th*dt");
  positive(1 / result.c_theta_dt, "1/(c*th*dt)");
  result.final_time = p.dt * p.ncycles;
  finite(result.final_time, "dt*ncycles");
  finite(p.b0x, "B0x");
  finite(p.b0y, "B0y");
  finite(p.b0z, "B0z");
  for (std::size_t index = 0; index < p.species.size(); ++index) {
    const auto &s = p.species[index];
    const std::string prefix = "species " + std::to_string(index) + ": ";
    finite(s.q_over_m, prefix + "qom");
    errors.require(s.q_over_m != 0, prefix + "qom=0 has no defined neutral-species semantics");
    for (const auto &[name, value] : {
             std::pair{"rhoINIT", s.rho_init},
             {"uth", s.uth}, {"vth", s.vth}, {"wth", s.wth}})
      errors.require(std::isfinite(value) && value >= 0,
              prefix + name + " must be finite and nonnegative");
    finite(s.u0, prefix + "u0");
    finite(s.v0, prefix + "v0");
    finite(s.w0, prefix + "w0");
    // A nonzero charge response may not silently underflow to a neutral one.
    const double beta = static_cast<double>(
        static_cast<long double>(s.q_over_m) * p.dt / (2.L * p.c));
    errors.require(std::isfinite(beta) && beta != 0,
            prefix + "qom*dt/(2*c) is not representable");
  }

  const auto mode = p.ecsim_smoothing_mode;
  if (mode == EcsimSmoothingMode::EnergyConserving) {
    errors.require(std::isfinite(p.smooth) && p.smooth >= 0 && p.smooth <= 1,
            "energy-conserving smoothing requires 0 <= Smooth <= 1");
    errors.require(p.smooth_niter >= 0, "energy-conserving smoothing requires SmoothNiter >= 0");
    result.smoothing_enabled = p.smooth_niter > 0 && p.smooth < 1;
  }

  const std::array<bool, 3> field_periodic{
      p.topology.periodic_x, p.topology.periodic_y, p.topology.periodic_z};
  const std::array<bool, 3> particle_periodic{p.topology.periodic_x_particles,
      p.topology.periodic_y_particles, p.topology.periodic_z_particles};
  const std::array<int, 6> em{p.bc.em_x_right, p.bc.em_x_left,
      p.bc.em_y_right, p.bc.em_y_left, p.bc.em_z_right, p.bc.em_z_left};
  const std::array<int, 6> particle{p.bc.particle_x_right, p.bc.particle_x_left,
      p.bc.particle_y_right, p.bc.particle_y_left,
      p.bc.particle_z_right, p.bc.particle_z_left};
  constexpr std::array<std::string_view, 6> faces{
      "Xright", "Xleft", "Yright", "Yleft", "Zright", "Zleft"};
  bool open_boundary = false;
  for (std::size_t face = 0; face < 6; ++face) {
    // The wall kinds precede periodic in the order of their deck codes.
    errors.require(em[face] >= 0 && em[face] < static_cast<int>(FieldBoundary::periodic),
            "bcEMface" + std::string(faces[face]) + "=" + std::to_string(em[face]) +
                ": unknown code (expected " + std::string(field_boundary_codes) + ")");
    auto &boundary = result.field_boundary[face];
    boundary = field_periodic[face / 2] ? FieldBoundary::periodic
                                        : static_cast<FieldBoundary>(em[face]);
    open_boundary |= boundary == FieldBoundary::open;
    result.particle_boundary[face] = particle_periodic[face / 2] ?
        ParticleBoundary::periodic : ParticleBoundary::mirror;
    if (!particle_periodic[face / 2] && particle[face] != particle_mirror_code) {
      const std::string request = "bcPface" + std::string(faces[face]) +
          "=" + std::to_string(particle[face]);
      if (particle[face] == 3 || particle[face] == 4) {
        result.unsupported_requests.push_back(request +
            ": boundary particle injection is not implemented; an effective open particle face requires injection");
      } else {
        result.unsupported_requests.push_back(request +
            ": only the mirror (1) is admitted on a physical particle face");
      }
    }
  }
  errors.require(!result.smoothing_enabled || !open_boundary,
          "active energy-conserving smoothing cannot use an effective open EM face");
  if (open_boundary)
    result.notes.emplace_back("Open EM boundaries are not implemented; --initialize and --run reject them");
  if (p.c != 1)
    result.unsupported_requests.emplace_back(
        "c != 1 is not supported: the mass-matrix and mover normalization is qualified for c=1 only");
  if (p.ns_test_particles != 0 || !p.test_species.empty())
    result.unsupported_requests.emplace_back("nsTestPart: test-particle initialization and evolution are not implemented");
  result.notes.emplace_back("Linear ECSIM uses GMRES restart=30 and maximum iterations=100");
  if (!p.output.particles_tag.empty())
    result.notes.emplace_back("ParticlesOutputTag has no effect; ParticlesOutputCycle selects particle output");
  return result;
}

} // namespace tpic::config
