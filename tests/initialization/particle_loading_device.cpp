#include <lpfx/backend/backend.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/initialization/particles/particle_loading_device.hpp>
#include "../support/pinned_transfer.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "../support/test_builders.hpp"

namespace {
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
constexpr double pi = 3.1415926535897932384626433832795;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
         error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}

template<class Precision> struct Snapshot {
  std::vector<typename Precision::Storage::Position> rx, ry, rz;
  std::vector<typename Precision::Storage::Velocity> u, v, w;
  std::vector<typename Precision::Storage::Weight> q;
  std::vector<std::int32_t> cell;
  std::vector<std::uint64_t> id;
  bool operator==(const Snapshot&) const = default;
};

// Payload readback belongs exclusively to validation, after startup publication.
template<class Precision, class Owner>
Snapshot<Precision> download(const Owner& owner, Executor executor) {
  using S = tpic::data::ParticleSchema<Precision>;
  Snapshot<Precision> result;
  const auto copy = [&]<class Attribute>(auto& values) {
    values.resize(owner.size());
    tpic::test::download_bytes(
        values.data(), owner.cview().template get<Attribute>(),
        values.size() * sizeof(typename Attribute::value_type), executor);
  };
  copy.template operator()<typename S::Rx>(result.rx);
  copy.template operator()<typename S::Ry>(result.ry);
  copy.template operator()<typename S::Rz>(result.rz);
  copy.template operator()<typename S::U>(result.u);
  copy.template operator()<typename S::V>(result.v);
  copy.template operator()<typename S::W>(result.w);
  copy.template operator()<typename S::Charge>(result.q);
  copy.template operator()<typename S::Cell>(result.cell);
  copy.template operator()<typename S::Id>(result.id);
  return result;
}

tpic::config::SimulationParameters fixture(int species_count) {
  tpic::config::SimulationParameters p;
  p.case_name = species_count == 3 ? "TwoStream" : "Uniform";
  p.dt = 0.1; p.ncycles = 1;
  p.grid.nxc = 5; p.grid.nyc = 3; p.grid.nzc = 2;
  p.grid.lx = 5; p.grid.ly = 6; p.grid.lz = 4;
  p.topology.xlen = 2;
  p.output.restart_cycle = 0;
  p.ns = species_count;
  p.species.resize(species_count);
  for (auto& s : p.species) {
    s.npcelx = s.npcely = 2; s.npcelz = 1;
    s.q_over_m = -25;
    s.rho_init = 4 * pi;
  }
  if (species_count == 3) {
    p.species[0].u0 = 0.25;
    p.species[1].u0 = -0.25;
    p.species[2].q_over_m = 1;
    p.species[2].rho_init = 8 * pi;
    p.two_stream_seed_mode_x = 1;
    p.two_stream_seed_density_amplitude = 0.125;
    p.two_stream_seed_phase = 0.2;
  }
  return p;
}

template<class Precision> void quiet_two_stream(Executor executor, tpic::config::PrecisionSelection profile) {
  auto p = fixture(3);
  p.quiet_start = true;
  const auto input = tpic::simulation::prepare_input(std::move(p), {.rank = 1, .precision = profile});
  for (std::size_t species = 0; species < 3; ++species) {
    const auto plan = tpic::test::prepare_particle_load(input, species);
    auto owner = tpic::initialization::make_initial_particles<Precision>(
        plan, input.domain, input.initial_case, species, executor);
    require(owner.valid() && owner.size() == 48 && owner.capacity() >= 48,
            "TwoStream population was not published");
    const auto data = download<Precision>(owner, executor);
    for (int index = 0; index < 48; ++index) {
      const int cell = index / 4;
      const int i = cell % 2, j = (cell / 2) % 3, k = cell / 6;
      const double rx = (index % 2 + 0.5) / 2;
      const double ry = ((index / 2) % 2 + 0.5) / 2;
      require(data.rx[index] == rx && data.ry[index] == ry && data.rz[index] == 0.5,
              "TwoStream quiet subcell placement changed");
      // Halo-2 particle keys: extended shape (6,7,6), x contiguous.
      require(data.cell[index] == (i + 2) + 6 * (j + 2) + 42 * (k + 2),
              "TwoStream particle key differs from the expected grid");
      require(data.id[index] == species * 120 + 72 + index,
              "particle IDs overlap or lost the unequal-rank offset");
      require(data.u[index] == input.parameters.species[species].u0 &&
                  data.v[index] == 0 && data.w[index] == 0,
              "cold TwoStream drift changed");
      const double x = 3 + i + rx;
      const double expected_q = species == 2 ? 2.0 :
          -(1 + 0.125 * std::cos(2 * pi * x / 5 + 0.2));
      require(std::abs(static_cast<double>(data.q[index]) - expected_q) < 1e-13,
              "TwoStream charge sign, normalization or electron seed changed");
    }
  }
}

template<class Precision> void random_uniform(Executor executor, tpic::config::PrecisionSelection profile) {
  auto p = fixture(1);
  p.species[0].uth = 0.1; p.species[0].vth = 0.2; p.species[0].wth = 0.3;
  p.species[0].u0 = 0.25;
  const auto input = tpic::simulation::prepare_input(p, {.rank = 1, .precision = profile});
  auto plan = tpic::test::prepare_particle_load(input, 0);
  auto first = tpic::initialization::make_initial_particles<Precision>(plan, input.domain, input.initial_case, 0, executor);
  auto second = tpic::initialization::make_initial_particles<Precision>(plan, input.domain, input.initial_case, 0, executor);
  const auto values = download<Precision>(first, executor);
  require(first.size() == 48 && values == download<Precision>(second, executor),
          "Uniform startup is not reproducible for a fixed seed and decomposition");
  bool randomized_position = false, thermal_velocity = false;
  for (int index = 0; index < 48; ++index) {
    for (const auto* coordinate : {&values.rx, &values.ry, &values.rz})
      require((*coordinate)[index] >= 0 && (*coordinate)[index] < 1,
              "Uniform stored fraction is not canonical");
    require(std::isfinite(values.u[index]) && std::isfinite(values.v[index]) &&
                std::isfinite(values.w[index]) && values.q[index] == -1,
            "Uniform thermal loading or charge normalization failed");
    require(values.cell[index] >= 0 && values.cell[index] < plan.cells.count &&
                values.id[index] == 72 + index, "Uniform key or identity is invalid");
    randomized_position |= values.rx[index] != (index % 2 + 0.5) / 2;
    thermal_velocity |= values.u[index] != 0.25 && values.v[index] != 0;
  }
  require(randomized_position && thermal_velocity, "Uniform random draws were skipped");
}

// QuietStart on a case loaded at random by default: both species sit on the
// same subcell centres, and the velocities are still drawn.
template<class Precision> void quiet_uniform(Executor executor, tpic::config::PrecisionSelection profile) {
  auto p = fixture(2);
  p.quiet_start = true;
  p.species[1].q_over_m = 1;
  for (auto& s : p.species) s.uth = s.vth = s.wth = 0.1;
  const auto input = tpic::simulation::prepare_input(std::move(p), {.rank = 1, .precision = profile});
  std::array<Snapshot<Precision>, 2> data;
  for (std::size_t species = 0; species < 2; ++species) {
    const auto plan = tpic::test::prepare_particle_load(input, species);
    data[species] = download<Precision>(tpic::initialization::make_initial_particles<Precision>(
        plan, input.domain, input.initial_case, species, executor), executor);
  }
  const auto& [electrons, ions] = data;
  require(electrons.rx == ions.rx && electrons.ry == ions.ry && electrons.rz == ions.rz &&
              electrons.cell == ions.cell, "QuietStart species are not on the same points");
  bool thermal_velocity = false;
  for (int index = 0; index < 48; ++index) {
    require(electrons.rx[index] == (index % 2 + 0.5) / 2 && electrons.ry[index] == ((index / 2) % 2 + 0.5) / 2 &&
                electrons.rz[index] == 0.5, "QuietStart particle off its subcell centre");
    thermal_velocity |= electrons.u[index] != 0 && ions.u[index] != electrons.u[index];
  }
  require(thermal_velocity, "QuietStart skipped the velocity draws");
}

// TwoStream, a quiet case of the shipped deck, without QuietStart: random
// positions, and the electron seed follows each stored position.
template<class Precision> void random_two_stream(Executor executor, tpic::config::PrecisionSelection profile) {
  const auto input = tpic::simulation::prepare_input(fixture(3), {.rank = 1, .precision = profile});
  const auto plan = tpic::test::prepare_particle_load(input, 0);
  const auto data = download<Precision>(tpic::initialization::make_initial_particles<Precision>(
      plan, input.domain, input.initial_case, 0, executor), executor);
  const double tolerance = std::is_same_v<typename Precision::Storage::Weight, float> ? 1e-6 : 1e-13;
  bool randomized_position = false;
  for (int index = 0; index < 48; ++index) {
    randomized_position |= data.rx[index] != (index % 2 + 0.5) / 2;
    const auto at = plan.cells.decode(data.cell[index]);
    const double x = plan.origin[0] + (at.i + static_cast<double>(data.rx[index])) * plan.spacing[0];
    require(std::abs(static_cast<double>(data.q[index]) + 1 + 0.125 * std::cos(2 * pi * x / 5 + 0.2)) < tolerance,
            "random TwoStream electron seed does not follow the stored position");
  }
  require(randomized_position, "TwoStream without QuietStart kept the subcell centres");
}

void endpoints() {
  int cell = 1;
  require(tpic::data::canonicalize_particle_fraction<double>(cell, 1.0) == 0 && cell == 2,
          "exact RNG upper endpoint did not carry into the halo");
  cell = 1;
  require(tpic::data::canonicalize_particle_fraction<float>(
              cell, std::nextafter(1.0, 0.0)) == 0 && cell == 2,
          "float position rounding to one did not carry");
  cell = -1;
  require(tpic::data::canonicalize_particle_fraction<double>(cell, 0.0) == 0 && cell == -1,
          "canonical zero changed its particle cell");
}
// currentFromAmpere of the seeded sheet, loaded cold: every particle carries
// its species' share of c*curl/(4*pi) at its own position, whatever the cell
// shape that scales its charge, and the two shares add up to the whole.
template<class Precision> void ampere_sheet(Executor executor, tpic::config::PrecisionSelection profile, bool quiet) {
  tpic::config::SimulationParameters p;
  p.case_name = "GEMHarris";
  p.quiet_start = quiet;
  p.dt = 0.1; p.ncycles = 1;
  p.grid.nxc = 8; p.grid.nyc = 16; p.grid.nzc = 1;
  p.grid.lx = 4; p.grid.ly = 8; p.grid.lz = 1;
  p.output.restart_cycle = 0;
  p.harris_b0x = 0.02;
  p.pert_hump = 0.04 * pi;
  p.deltax_hump = p.deltay_hump = 1.0;
  p.kx_hump = p.ky_hump = pi / (10.0 * p.delta);
  p.current_from_ampere = true;
  p.ns = 2;
  p.species.resize(2);
  for (auto& s : p.species) {
    s.npcelx = s.npcely = 2; s.npcelz = 1;
    s.rho_init = 1.0;
  }
  p.species[0].q_over_m = -25; p.species[0].w0 = 0.006;
  p.species[1].q_over_m = 1; p.species[1].w0 = -0.03;
  const auto input = tpic::simulation::prepare_input(std::move(p), {.precision = profile});
  const auto& sheet = std::get<tpic::initialization::HarrisSheetCase>(input.initial_case);
  constexpr bool narrow = std::is_same_v<typename Precision::Storage::Velocity, float> ||
      std::is_same_v<typename Precision::Storage::Weight, float>;
  const double tolerance = narrow ? 2e-6 : 2e-13;
  // The current of one particle slot: cell volume over particles per cell times c*curl/(4*pi).
  std::vector<double> whole, carried;
  for (std::size_t species = 0; species < 2; ++species) {
    const auto plan = tpic::test::prepare_particle_load(input, species);
    auto owner = tpic::initialization::make_initial_particles<Precision>(
        plan, input.domain, input.initial_case, species, executor);
    const auto data = download<Precision>(owner, executor);
    require(data.w.size() == 8 * 16 * 4, "Ampere sheet population was not published");
    whole.resize(data.w.size());
    carried.resize(data.w.size());
    const auto loading = sheet.profile(species);
    // The species' share: sign(qom)*w0 over the sum of both.
    const double share = (species == 0 ? -0.006 : -0.03) / (-0.006 - 0.03);
    for (std::size_t index = 0; index < data.w.size(); ++index) {
      const auto at = plan.cells.decode(data.cell[index]);
      const double x = plan.origin[0] + (at.i + static_cast<double>(data.rx[index])) * plan.spacing[0];
      const double y = plan.origin[1] + (at.j + static_cast<double>(data.ry[index])) * plan.spacing[1];
      whole[index] = (0.5 * 0.5 * 1.0 / 4) * input.parameters.c * loading.curl.at(x, y) / (4 * pi);
      const double current = static_cast<double>(data.q[index]) * static_cast<double>(data.w[index]);
      require(data.u[index] == 0 && data.v[index] == 0 &&
                  std::abs(current - share * whole[index]) <= tolerance * (std::abs(whole[index]) + 1e-9),
              "Ampere sheet particle does not carry its share of the curl");
      carried[index] += current;
    }
  }
  // Under QuietStart the slot of each species is the same point, where the two shares add up.
  if (quiet)
    for (std::size_t index = 0; index < whole.size(); ++index)
      require(std::abs(carried[index] - whole[index]) <= 2 * tolerance * (std::abs(whole[index]) + 1e-9),
              "the two Ampere shares do not add up to the curl");
}

// A Gaussian cylinder along z cut at two widths, on both ranks of the x
// split: electrons sit on the ions with the opposite charge, every point lies
// inside the support, and the ranks load each subcell inside it once. The
// unit grid and two subcells per axis keep the support test exact.
template<class Precision> void gaussian_expansion(Executor executor, tpic::config::PrecisionSelection profile) {
  auto p = fixture(2);
  p.case_name = "GaussianExpansion";
  p.quiet_start = true;
  p.grid.nxc = 12; p.grid.nyc = 10; p.grid.nzc = 2;
  p.grid.lx = 12; p.grid.ly = 10; p.grid.lz = 2;
  p.species[1].q_over_m = 1;
  p.species[0].uth = p.species[0].vth = p.species[0].wth = 0.1;
  p.species[0].npcelz = p.species[1].npcelz = 2;
  p.gaussian_expansion_center_x = 6; p.gaussian_expansion_center_y = 5;
  p.gaussian_expansion_sigma_x = p.gaussian_expansion_sigma_y = 2;
  p.gaussian_expansion_cutoff = 2;
  const double tolerance = std::is_same_v<typename Precision::Storage::Weight, float> ? 1e-7 : 1e-15;
  // The subcell centres of the 24 x 20 x-y lattice inside radius 4, on 4 z subcells.
  std::uint64_t inside = 0, loaded = 0;
  for (int j = 0; j < 20; ++j)
    for (int i = 0; i < 24; ++i) {
      const double x = (i + 0.5) / 2 - 6, y = (j + 0.5) / 2 - 5;
      inside += 4 * (x * x + y * y <= 16);
    }
  for (int rank = 0; rank < 2; ++rank) {
    const auto input = tpic::simulation::prepare_input(p, {.rank = rank, .precision = profile});
    std::array<Snapshot<Precision>, 2> data;
    for (std::size_t species = 0; species < 2; ++species) {
      const auto plan = tpic::test::prepare_particle_load(input, species);
      const auto owner = tpic::initialization::make_initial_particles<Precision>(
          plan, input.domain, input.initial_case, species, executor);
      data[species] = download<Precision>(owner, executor);
      require(owner.size() == plan.count && input.particles.species[species].global_count == inside,
              "GaussianExpansion count differs from the plan");
      for (std::size_t index = 0; index < data[species].id.size(); ++index)
        require(data[species].id[index] == plan.first_id + index, "GaussianExpansion IDs are not the plan's");
    }
    const auto& [electrons, ions] = data;
    require(electrons.rx == ions.rx && electrons.ry == ions.ry && electrons.rz == ions.rz &&
                electrons.cell == ions.cell && ions.u == decltype(ions.u)(ions.u.size()),
            "electrons are not on the cold ions");
    const auto cells = tpic::data::prepare_particle_cell_map(input.domain);
    for (std::size_t index = 0; index < ions.q.size(); ++index) {
      const auto at = cells.decode(electrons.cell[index]);
      const double x = input.domain.origin[0] + at.i + static_cast<double>(ions.rx[index]) - 6;
      const double y = at.j + static_cast<double>(ions.ry[index]) - 5;
      const double expected = 0.125 * std::exp(-(x * x + y * y) / 8);
      require(x * x + y * y <= 16 && electrons.q[index] == -ions.q[index] &&
                  std::abs(static_cast<double>(ions.q[index]) - expected) <= tolerance * expected,
              "GaussianExpansion point outside the support or with the wrong charge");
    }
    loaded += ions.q.size();
  }
  require(loaded == inside, "the ranks did not load every subcell of the support once");
}

} // namespace

int main() {
  lpfx::device::DeviceProperties properties{};
  auto error = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(error)) return runtime_unavailable(error) ? 77 : 1;
  auto stream = Traits::null_stream;
  error = Traits::create_stream(stream);
  if (!Traits::succeeded(error)) return runtime_unavailable(error) ? 77 : 1;
  int status = 0;
  try {
    const Executor executor{stream};
    endpoints();
    for (auto profile : {tpic::config::PrecisionSelection{tpic::config::StoragePrecision::double_precision, tpic::config::ComputePrecision::double_precision},
                         tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_position, tpic::config::ComputePrecision::double_precision},
                         tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_particles_moments_fields, tpic::config::ComputePrecision::double_precision}})
      tpic::config::dispatch_precision(profile, [&]<class Precision> {
        quiet_two_stream<Precision>(executor, profile);
        random_uniform<Precision>(executor, profile);
        quiet_uniform<Precision>(executor, profile);
        random_two_stream<Precision>(executor, profile);
        for (const bool quiet : {true, false}) ampere_sheet<Precision>(executor, profile, quiet);
        gaussian_expansion<Precision>(executor, profile);
      });
    std::cout << "TPIC native particle loading smoke passed\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(stream))) status = 1;
  return status;
}
