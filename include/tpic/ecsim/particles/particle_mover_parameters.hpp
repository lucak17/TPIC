#ifndef TPIC_ECSIM_PARTICLES_PARTICLE_MOVER_PARAMETERS_HPP
#define TPIC_ECSIM_PARTICLES_PARTICLE_MOVER_PARAMETERS_HPP

#include <tpic/communication/particles/particle_geometry.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/config/precision.hpp>

#include <cmath>
#include <stdexcept>

namespace tpic::ecsim {

// Run-constant mover inputs of one species; the mover kernels take them by value.
template <class Precision> struct ParticleMoverParameters {
  typename Precision::Compute::Velocity beta, beta_over_c;
  typename Precision::Compute::Position dt, inverse_spacing[3];
};

// Built once per species at setup. beta = q_over_m * dt / 2, beta / c and
// 1.0 / spacing are evaluated in double and converted once each: the velocity
// update constants to Compute::Velocity, the drift constants to Compute::Position.
// The geometry must describe the same local and global cells as the domain.
template <class Precision>
[[nodiscard]] ParticleMoverParameters<Precision> prepare_particle_mover_parameters(
    const config::PreparedDomain& domain, const communication::ParticleMigrationGeometry& geometry,
    double q_over_m, double dt, double c) {
  using VelocityCompute = typename Precision::Compute::Velocity;
  using PositionCompute = typename Precision::Compute::Position;
  const double beta = q_over_m * dt / 2;
  const double beta_over_c = beta / c;
  if (geometry.cells.nx != domain.local_cells[0] ||
      geometry.cells.ny != domain.local_cells[1] ||
      geometry.cells.nz != domain.local_cells[2])
    throw std::invalid_argument("TPIC mover: geometry cells differ from the prepared domain");
  if (!std::isfinite(beta) || !std::isfinite(beta_over_c) || !std::isfinite(dt) || dt <= 0)
    throw std::invalid_argument("TPIC mover: q/m*dt/2, its ratio to c and dt must be finite");
  ParticleMoverParameters<Precision> parameters{
      config::convert<VelocityCompute>(beta), config::convert<VelocityCompute>(beta_over_c),
      config::convert<PositionCompute>(dt), {}};
  if (!std::isfinite(parameters.beta) || !std::isfinite(parameters.beta_over_c) ||
      !std::isfinite(parameters.dt) || parameters.dt <= 0)
    throw std::overflow_error("TPIC mover: parameters overflow the particle precision");
  for (int axis = 0; axis < 3; ++axis) {
    if (geometry.global_cells[axis] != domain.global_cells[axis] ||
        geometry.global_offset[axis] != domain.global_offset[axis] ||
        !std::isfinite(domain.spacing[axis]) || domain.spacing[axis] <= 0)
      throw std::invalid_argument("TPIC mover: geometry partition differs from the prepared domain");
    parameters.inverse_spacing[axis] = config::convert<PositionCompute>(1.0 / domain.spacing[axis]);
    if (!std::isfinite(parameters.inverse_spacing[axis]) || parameters.inverse_spacing[axis] <= 0)
      throw std::overflow_error("TPIC mover: inverse spacing overflows the particle precision");
  }
  return parameters;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_PARTICLES_PARTICLE_MOVER_PARAMETERS_HPP
