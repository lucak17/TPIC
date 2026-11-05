#include <tpic/diagnostics/async_particle_snapshot_device.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/initialization/particles/particle_loading_device.hpp>
#include <tpic/io/detail/output_layout.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include "../support/pinned_transfer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>
#include "../support/test_builders.hpp"

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Snapshot = tpic::diagnostics::AsyncParticleSnapshot<Backend>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Function> void rejects(Function&& function) {
  try { function(); }
  catch (const std::runtime_error&) { return; }
  throw std::runtime_error("particle snapshot accepted an invalid lifecycle or source");
}
__global__ void delay_transfer(unsigned long long ticks) {
  const auto start = clock64();
  while (clock64() - start < ticks) {}
}
template <class Precision, class View>
__global__ void set_values(View particles, bool corrupt) {
  using S = tpic::data::ParticleSchema<Precision>;
  for (unsigned index = blockIdx.x * blockDim.x + threadIdx.x; index < particles.size();
       index += blockDim.x * gridDim.x) {
    particles.template get<typename S::U>()[index] = corrupt ? -9000 : 0.25;
    particles.template get<typename S::V>()[index] = corrupt ? -9000 : -0.5;
    particles.template get<typename S::W>()[index] = corrupt ? -9000 : 1.5;
    particles.template get<typename S::Charge>()[index] = corrupt ? -9000 : -2;
    particles.template get<typename S::Id>()[index] = (std::uint64_t(1) << 60) + index;
    if (corrupt) {
      particles.template get<typename S::Rx>()[index] = -9000;
      particles.template get<typename S::Ry>()[index] = -9000;
      particles.template get<typename S::Rz>()[index] = -9000;
      particles.template get<typename S::Cell>()[index] = -1;
    }
  }
}

// The device values of one attribute, downloaded for comparison.
template <class Attribute, class Owner> auto download(const Owner& owner, Executor stream) {
  std::vector<typename Attribute::value_type> values(owner.size());
  tpic::test::download_bytes(values.data(), owner.template get<Attribute>(),
                             values.size() * sizeof(values[0]), stream);
  return values;
}
// The lent span of one attribute, which must hold exactly the stored type.
template <class Value> std::span<const Value> stored(const tpic::data::ParticleValues& values) {
  require(std::holds_alternative<std::span<const Value>>(values), "particle record lost its storage type");
  return std::get<std::span<const Value>>(values);
}

template <class Precision>
void run(Executor producer, Executor transfer, tpic::config::PrecisionSelection precision) {
  using S = tpic::data::ParticleSchema<Precision>;
  using Position = typename Precision::Storage::Position;
  using Velocity = typename Precision::Storage::Velocity;
  using Weight = typename Precision::Storage::Weight;
  tpic::config::SimulationParameters parameters;
  parameters.case_name = "Uniform";
  parameters.dt = 0.1; parameters.ncycles = 1;
  parameters.grid = {.lx = 10, .ly = 9, .lz = 4, .nxc = 5, .nyc = 3, .nzc = 2};
  parameters.topology.xlen = 2;
  parameters.output.restart_cycle = 0;
  parameters.ns = 1;
  parameters.species.resize(1);
  auto& species = parameters.species.front();
  species.q_over_m = -1;
  species.rho_init = 1;
  species.npcelx = species.npcely = species.npcelz = 1;
  const auto input = tpic::simulation::prepare_input(parameters, {.rank = 1, .precision = precision});
  const auto plan = tpic::test::prepare_particle_load(input, 0);
  auto owner = tpic::initialization::make_initial_particles<Precision>(plan, input.domain, input.initial_case, 0, producer);
  auto empty = tpic::initialization::make_initial_particles<Precision>(plan, input.domain, input.initial_case, 0, producer);
  empty.clear();
  require(owner.size() > 1, "particle fixture holds too few records");
  Snapshot capture(2, producer, transfer);
  require(capture.pinned_bytes() == 0, "particle capture allocated before its first capture");
  tpic::data::OwnedOutputSnapshot output;
  output.cycle = 7; output.time = 1.25; output.final_state = true;
  output.fields_present = false; output.particles_present = true;
  rejects([&] { capture.host_wait(); });
  rejects([&] { capture.device_wait_sources(producer); });
  set_values<Precision><<<1, 128, 0, producer.stream()>>>(owner.view(), false);
  lpfx::device::check_device_launch<Backend>("particle test setup failed");
  const auto keys = download<typename S::Cell>(owner, producer);
  const std::array fractions{download<typename S::Rx>(owner, producer), download<typename S::Ry>(owner, producer),
                             download<typename S::Rz>(owner, producer)};

  // The copies run on the delayed transfer stream; a source overwrite ordered
  // after the release event must not reach the captured images.
  capture.begin();
  rejects([&] { capture.begin(); });
  delay_transfer<<<1, 1, 0, transfer.stream()>>>(20000000ULL);
  lpfx::device::check_device_launch<Backend>("particle test delay failed");
  capture.template copy<Precision>(owner, 0);
  rejects([&] { capture.finish_capture(); });
  capture.template copy<Precision>(empty, 1);
  capture.finish_capture();
  capture.device_wait_sources(producer);
  set_values<Precision><<<1, 128, 0, producer.stream()>>>(owner.view(), true);
  lpfx::device::check_device_launch<Backend>("particle test overwrite failed");
  capture.host_wait();
  capture.lend_into(output);
  require(!capture.pending() && output.cycle == 7 && output.time == 1.25 && output.final_state,
          "particle capture changed epoch metadata");
  std::size_t image_bytes = 0;
  for (const auto width : S::type::widths) image_bytes += (owner.size() * width + 63) / 64 * 64;
  require(capture.pinned_bytes() == image_bytes,
          "particle image included empty species or inactive owner capacity");
  require(output.particles.size() == 2 && output.particles[1].species == 1 && output.particles[1].id.empty(),
          "empty species not retained");
  const auto& records = output.particles[0];
  require(records.id.size() == owner.size() && records.cell.size() == owner.size(),
          "particle records included inactive capacity or lost records");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const auto lent = stored<Position>(records.fraction[axis]);
    require(std::vector<Position>(lent.begin(), lent.end()) == fractions[axis], "stored fraction changed");
  }
  for (std::size_t index = 0; index < owner.size(); ++index)
    require(records.cell[index] == keys[index] && stored<Velocity>(records.velocity[0])[index] == Velocity(.25) &&
                stored<Velocity>(records.velocity[1])[index] == Velocity(-.5) &&
                stored<Velocity>(records.velocity[2])[index] == Velocity(1.5) &&
                stored<Weight>(records.charge)[index] == Weight(-2) &&
                records.id[index] == (std::uint64_t(1) << 60) + index,
            "particle records changed after the released source was overwritten");

  // Engines write positions (offset + cell + fraction) * spacing in double.
  const auto geometry = tpic::io::detail::prepare_particle_geometry(input.domain);
  std::vector<double> positions(3 * owner.size());
  tpic::io::detail::convert_particles(tpic::io::detail::particle_position, records, geometry, 0, positions);
  for (std::size_t index = 0; index < owner.size(); ++index) {
    const auto cell = geometry.cells.decode(keys[index]);
    const int coordinate[3]{cell.i, cell.j, cell.k};
    for (std::size_t axis = 0; axis < 3; ++axis)
      require(positions[3 * index + axis] ==
                  (input.domain.global_offset[axis] + coordinate[axis] + double(fractions[axis][index])) *
                      input.domain.spacing[axis], "global particle position changed");
  }

  // A capture of the same population reuses the lent images.
  const auto* first_id = records.id.data();
  capture.begin();
  capture.template copy<Precision>(owner, 0);
  capture.template copy<Precision>(empty, 1);
  capture.finish_capture();
  capture.host_wait();
  capture.lend_into(output);
  require(output.particles[0].id.data() == first_id && capture.pinned_bytes() == image_bytes,
          "particle capture reallocated its reusable images");

  // An initially empty species allocates nothing and grows on its first records.
  {
    Snapshot initially_empty(1, producer, transfer);
    initially_empty.begin();
    initially_empty.template copy<Precision>(empty, 0);
    initially_empty.finish_capture();
    initially_empty.host_wait();
    tpic::data::OwnedOutputSnapshot saved;
    initially_empty.lend_into(saved);
    require(saved.particles[0].id.empty() && initially_empty.pinned_bytes() == 0,
            "empty particle capture allocated storage");
    initially_empty.begin();
    initially_empty.template copy<Precision>(owner, 0);
    initially_empty.finish_capture();
    initially_empty.host_wait();
    initially_empty.lend_into(saved);
    require(initially_empty.pinned_bytes() == image_bytes &&
                std::vector<std::uint64_t>(saved.particles[0].id.begin(), saved.particles[0].id.end()) ==
                    std::vector<std::uint64_t>(output.particles[0].id.begin(), output.particles[0].id.end()),
            "initially empty capture did not capture newly arrived particles");
  }

  // The capture copies records as stored; the engines reject invalid ones
  // before anything reaches a file. The source keeps the overwritten cell keys;
  // its fractions are valid again, so only the keys are invalid.
  tpic::test::upload_bytes(owner.template get<typename S::Rx>(), fractions[0].data(),
                           fractions[0].size() * sizeof(fractions[0][0]), producer);
  tpic::test::upload_bytes(owner.template get<typename S::Ry>(), fractions[1].data(),
                           fractions[1].size() * sizeof(fractions[1][0]), producer);
  tpic::test::upload_bytes(owner.template get<typename S::Rz>(), fractions[2].data(),
                           fractions[2].size() * sizeof(fractions[2][0]), producer);
  capture.begin();
  capture.template copy<Precision>(owner, 0);
  capture.template copy<Precision>(empty, 1);
  capture.finish_capture();
  capture.host_wait();
  capture.lend_into(output);
  bool rejected = false;
  try {
    tpic::io::detail::convert_particles(tpic::io::detail::particle_position, output.particles[0], geometry, 0,
                                        positions);
  } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "invalid particle records were converted");
}

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
      error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}
} // namespace

int main() {
  lpfx::device::DeviceProperties properties{};
  const auto status = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(status)) return runtime_unavailable(status) ? 77 : 1;
  auto producer = Traits::null_stream, transfer = Traits::null_stream;
  if (!Traits::succeeded(Traits::create_stream(producer))) return 1;
  if (!Traits::succeeded(Traits::create_stream(transfer))) {
    static_cast<void>(Traits::destroy_stream(producer)); return 1;
  }
  int result = 0;
  try {
    run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(Executor{producer}, Executor{transfer}, tpic::config::PrecisionSelection{tpic::config::StoragePrecision::double_precision, tpic::config::ComputePrecision::double_precision});
    run<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(Executor{producer}, Executor{transfer}, tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_position, tpic::config::ComputePrecision::double_precision});
    run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(Executor{producer}, Executor{transfer}, tpic::config::PrecisionSelection{tpic::config::StoragePrecision::single_particles_moments_fields, tpic::config::ComputePrecision::double_precision});
    std::cout << "TPIC asynchronous particle snapshot checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; result = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(transfer))) result = 1;
  if (!Traits::succeeded(Traits::destroy_stream(producer))) result = 1;
  return result;
}
