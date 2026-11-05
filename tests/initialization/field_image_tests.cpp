#include <tpic/config/domain.hpp>
#include <tpic/initialization/fields/field_image.hpp>
#include <tpic/initialization/prepared_initialization.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "../support/test_builders.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Error, class Function> void rejects(Function function) {
  try { function(); }
  catch (const Error&) { return; }
  throw std::runtime_error("invalid field image request was accepted");
}

tpic::config::SimulationParameters parameters() {
  tpic::config::SimulationParameters p;
  p.ns = 2;
  p.species.resize(2);
  p.species[0].q_over_m = -25.0;
  p.species[1].q_over_m = 1.0;
  for (auto& species : p.species) {
    species.rho_init = 1.0;
    species.npcelx = species.npcely = species.npcelz = 1;
  }
  p.grid = {.lx = 9.0, .ly = 7.0, .lz = 5.0, .nxc = 7, .nyc = 5, .nzc = 3};
  p.b0x = 0.3;
  p.b0y = -0.7;
  p.b0z = 1.1;
  return p;
}

template <class Precision> void uniform_storage() {
  using Fields = tpic::data::FieldSchema<Precision>;
  auto p = parameters();
  const auto domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  auto image = tpic::test::make_host_field_image<Precision>(p, domain);
  const auto ex = image.electric.template get<typename Fields::Ex>();
  const auto ey = image.electric.template get<typename Fields::Ey>();
  const auto ez = image.electric.template get<typename Fields::Ez>();
  const auto bx = image.magnetic_cell.template get<typename Fields::Bx>();
  const auto by = image.magnetic_cell.template get<typename Fields::By>();
  const auto bz = image.magnetic_cell.template get<typename Fields::Bz>();
  require(ex.extents().x() == 8 && ex.extents().y() == 6 && ex.extents().z() == 4,
          "E must be nodal, including the shared endpoint");
  require(bx.extents().x() == 7 && bx.extents().y() == 5 && bx.extents().z() == 3,
          "B must be cell centered");
  for (int k = -1; k <= 4; ++k)
    for (int j = -1; j <= 6; ++j)
      for (int i = -1; i <= 8; ++i) {
        require(ex(i,j,k) == 0 && ey(i,j,k) == 0 && ez(i,j,k) == 0,
                "electric image or halo not initialized");
        if (k <= 3 && j <= 5 && i <= 7)
          require(bx(i,j,k) == static_cast<typename Precision::Storage::Magnetic>(p.b0x) &&
                  by(i,j,k) == static_cast<typename Precision::Storage::Magnetic>(p.b0y) &&
                  bz(i,j,k) == static_cast<typename Precision::Storage::Magnetic>(p.b0z),
                  "magnetic background must appear exactly once, including halos");
      }
}

void analytic_halos_and_narrowing() {
  using Fields = tpic::data::FieldSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
  auto p = parameters();
  p.case_name = "ForceFree";
  p.forcefree_b0x = 0.9;
  p.delta = 2.3;
  const auto domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  auto image = tpic::test::make_host_field_image(p, domain);
  const auto bx = image.magnetic_cell.get<Fields::Bx>();
  const double y = -0.5 * domain.spacing[1];
  const double expected = p.b0x + p.forcefree_b0x * std::tanh((y - 0.5 * p.grid.ly) / p.delta);
  require(std::abs(bx(0,-1,0) - expected) < 2e-15, "cell halo must sample unwrapped physical coordinate");
  require(bx(0,-1,0) != bx(0,p.grid.nyc-1,0),
          "analytic image must not silently claim periodic halo synchronization");

  p = parameters();
  p.b0x = 2.0 * static_cast<double>(std::numeric_limits<float>::max());
  (void)tpic::test::make_host_field_image(p, domain);
  rejects<std::overflow_error>([&] {
    (void)tpic::test::make_host_field_image<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(p, domain);
  });
  p.b0x = std::numeric_limits<double>::denorm_min();
  image = tpic::test::make_host_field_image(p, domain);
  require(image.magnetic_cell.get<Fields::Bx>()(0,0,0) == p.b0x, "double subnormal sample lost");
}

void geometry_errors() {
  auto p = parameters();
  auto domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  ++domain.local_cells[0];
  rejects<std::invalid_argument>([&] { (void)tpic::test::make_host_field_image(p, domain); });
  domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  domain.origin[0] = std::numeric_limits<double>::infinity();
  rejects<std::invalid_argument>([&] { (void)tpic::test::make_host_field_image(p, domain); });
}

void prepared_initialization_boundary() {
  auto p = parameters();
  p.dt = .1;
  p.ncycles = 1;
  p.species[1].npcelx = 3;
  p.species[1].u0 = .25;
  auto input = tpic::simulation::prepare_input(p);
  const auto loads = tpic::test::prepare_particle_loads(input);
  require(loads.size() == 2 && loads[0].count == 7 * 5 * 3 &&
              loads[1].count == 3 * 7 * 5 * 3 &&
              loads[1].particles_per_cell == 3 &&
              loads[1].first_id == loads[0].count,
          "checked load plan changed species counts or disjoint IDs");
  require(loads[1].seed == 12445 &&
              loads[1].drift[0] == .25,
          "cached load plan changed the RNG seed or drift");
  auto image = tpic::initialization::make_host_field_image<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(
      input.parameters, input.initial_case, input.domain);
  using Fields = tpic::data::FieldSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
  require(image.magnetic_cell.get<Fields::Bx>()(0,0,0) == p.b0x,
          "prepared field coefficients were lost");

  ++input.particles.species[0].local_count;
  rejects<std::invalid_argument>([&] {
    (void)tpic::test::prepare_particle_loads(input);
  });
  input = tpic::simulation::prepare_input(p);
  ++input.particles.species[1].first_id;
  rejects<std::invalid_argument>([&] {
    (void)tpic::test::prepare_particle_load(input, 1);
  });
  input = tpic::simulation::prepare_input(p);
  input.initial_case.emplace<tpic::initialization::WeibelCase>(input.parameters);
  rejects<std::invalid_argument>([&] {
    (void)tpic::test::prepare_particle_loads(input);
  });
  input = tpic::simulation::prepare_input(p);
  input.domain.spacing[0] *= 2;
  rejects<std::invalid_argument>([&] {
    (void)tpic::test::prepare_particle_loads(input);
  });
  input = tpic::simulation::prepare_input(p);
  input.parameters.species[0].npcelx = std::numeric_limits<int>::max();
  input.parameters.species[0].npcely = 2;
  rejects<std::overflow_error>([&] {
    (void)tpic::test::prepare_particle_loads(input);
  });
}

} // namespace

int main() {
  try {
    uniform_storage<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>();
    uniform_storage<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>();
    uniform_storage<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>();
    analytic_halos_and_narrowing();
    geometry_errors();
    prepared_initialization_boundary();
    std::cout << "LPFx field images, halo coordinates and storage admission passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
