#ifndef TPIC_SIMULATION_STATE_SPECIES_DEVICE_HPP
#define TPIC_SIMULATION_STATE_SPECIES_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/core/error.hpp>
#include <tpic/communication/particles/particle_migration_policy.hpp>

#include <tpic/ecsim/particles/particle_sort_device.hpp>
#include <tpic/initialization/particles/particle_centering_device.hpp>
#include <tpic/initialization/particles/particle_loading_device.hpp>

#include <tpic/communication/particles/particle_geometry.hpp>
#include <tpic/io/checkpoint.hpp>
#include <tpic/simulation/execution/active_backend_device.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tpic::simulation {

// A restart's particles of one species: the checkpoint's records in the cells
// of this rank, read in chunks into one pinned staging block, which is freed
// on return, and copied into their final slots. The chunk does not shrink with
// the local count: the unsorted records of a piece are filtered chunk by chunk.
template <class Precision>
[[nodiscard]] auto make_restored_particles(const data::PreparedRestart& restart, const config::PreparedDomain& domain,
    std::size_t species, const initialization::ParticleLoadPlan& load, Executor executor) {
  using Schema = typename data::ParticleSchema<Precision>::type;
  using Names = data::ParticleSchema<Precision>;
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Pinned = lpfx::device::PinnedHostAllocation<Backend>;
  constexpr std::size_t chunk = io::restart_staging_records;
  const auto block = [](std::size_t width) { return (chunk * width + 63) / 64 * 64; };
  std::size_t bytes = 0;
  for (const auto width : Schema::widths) bytes += block(width);
  Pinned staging;
  lpfx::device::check_device_error<Backend>(Pinned::create(bytes, staging),
                                            "TPIC restart: particle staging allocation failed");
  // One block per attribute, in schema order: the braced list evaluates left
  // to right.
  auto* cursor = static_cast<std::byte*>(staging.raw_base());
  const auto take = [&]<std::size_t A>() {
    using Value = typename Schema::template value_type<A>;
    const std::span<Value> values(reinterpret_cast<Value*>(cursor), chunk);
    cursor += block(sizeof(Value));
    return values;
  };
  const auto blocks = [&]<std::size_t... A>(std::index_sequence<A...>) {
    return std::tuple{take.template operator()<A>()...};
  }(std::make_index_sequence<Schema::attribute_count>{});
  const auto& [rx, ry, rz, u, v, w, charge, cell, id] = blocks;
  static_assert(Schema::template index_of<typename Names::Rx> == 0 &&
                Schema::template index_of<typename Names::Id> == 8);
  const io::ParticleStaging records{.fraction = {rx, ry, rz}, .velocity = {u, v, w}, .charge = charge,
                                    .cell = cell, .id = id};
  return initialization::make_initial_particles<Precision>(load.capacity, load.count, executor, [&](auto range) {
    std::size_t offset = 0;
    io::read_restart_particles(restart, domain, species, records, [&](std::size_t count) {
      lpfx::require(offset + count <= range.size(), "TPIC restart: more particle records than counted");
      [&]<std::size_t... A>(std::index_sequence<A...>) {
        (lpfx::device::check_device_error<Backend>(
             Traits::copy_host_to_device_async(range.template get<A>() + offset, std::get<A>(blocks).data(),
                                               count * sizeof(typename Schema::template value_type<A>),
                                               executor.stream()),
             "TPIC restart: particle upload failed"), ...);
      }(std::make_index_sequence<Schema::attribute_count>{});
      // The next chunk refills the staging block.
      lpfx::device::check_device_error<Backend>(Traits::synchronize_stream(executor.stream()),
                                                "TPIC restart: particle upload failed");
      offset += count;
    });
    lpfx::require(offset == range.size(), "TPIC restart: fewer particle records than counted");
  });
}

// One species' device owners. The species index is not part of the type: the
// pack is a std::array of the compiled capacity, and only the migration
// request, a distinct LPFx operation per species, takes the index as a template
// argument. A species beyond the ns of the run is never loaded: it holds
// an empty owner (initialize_unused) and no slots or sort workspace.
template <class Precision, bool Partial> struct Species {
  using Schema = typename data::ParticleSchema<Precision>::type;
  using Owner = lpfx::Particles<Schema, lpfx::DeviceSpace<Backend>>;

  Owner particles;
  lpfx::DeviceParticleClassificationSlots<Backend> slots;
  std::conditional_t<Partial, ecsim::PartialParticleSort<Precision, Backend>,
                     ecsim::FullParticleSort<Precision, Backend>>
      ordering;

  void initialize(const RunPlan &plan, std::size_t index, const lpfx::device::DeviceProperties &properties,
                  Executor executor, Executor sort_executor) {
    const auto &input = plan.input;
    const auto &load = plan.particle_loads[index];
    particles = plan.restart
        ? make_restored_particles<Precision>(*plan.restart, input.domain, index, load, executor)
        : initialization::make_initial_particles<Precision>(load, input.domain, input.initial_case, index, executor);
    if (Partial || !input.particles.bypass_local_migration)
      lpfx::check_error(slots.reserve(particles.capacity()),
                        "TPIC species: particle classification reservation failed");
    if (Partial || input.particles.order == config::ParticleOrder::full)
      ordering.initialize(particles, input.domain, properties, sort_executor);
  }

  // The registration requires a valid owner for every compiled species; an
  // unused one gets capacity zero, which allocates nothing.
  void initialize_unused() {
    lpfx::PreparedDeviceParticleStorage<Schema, Backend> storage;
    lpfx::check_error(lpfx::try_prepare_device_particle_storage(0, storage),
                      "TPIC species: unused species storage preparation failed");
    auto made = lpfx::make_device_particles(storage);
    lpfx::check_error(made, "TPIC species: unused species owner creation failed");
    particles = std::move(made).value();
  }

  // An unused species registers no traffic: seed zero and no arrival reserve.
  template <std::size_t Index, class Route>
  [[nodiscard]] auto migration_request(const simulation::PreparedInput &input,
                                       communication::ParticleMigrationGeometry geometry,
                                       Executor routing) noexcept {
    namespace d = lpfx::distributed;
    const bool used = Index < input.particles.species.size();
    const auto seed = used ? input.particles.species[Index].initial_migration_capacity : 0;
    return d::migrate<"particles", Index>(particles, input.domain.particle_topology,
        communication::particle_migration_policy<Precision>(geometry), d::on<Route>(routing), d::using_flags(slots),
        d::traffic_seed(seed, d::arrivals_up_to_owner,
                        used && (Partial || !input.particles.bypass_local_migration)),
        d::resident_access<Partial>);
  }
};

// TPIC owns the sort receipt. Match it to LPFx's retained partition before
// publishing numerical work after append or storage growth.
template <class Species, class Migration>
[[nodiscard]] auto checked_completed_partial_receipt(
    const Species& species, const Migration& migration) {
  const auto partition = migration.completed_partition();
  const auto* receipt = &species.ordering.receipt();
  lpfx::check_error(partition, "TPIC species: partial migration has no completed reorder");
  lpfx::require(partition.value().resident_prefix == receipt->prefix_size() &&
                    partition.value().total_size == species.particles.size() &&
                    partition.value().generation == receipt->migration_generation() &&
                    partition.value().resident_prefix <= species.particles.size(),
                "TPIC species: partial partition disagrees with its receipt or owner");
  return receipt;
}

// Workers carry a run-time species index; each species' migration operation is
// a distinct type, so a caller that needs it also needs a compile-time index.
// This compare chain over at most config::max_real_species entries calls
// function.template operator()<S>(std::get<S>(pack)) for a tuple or an array.
template <std::size_t S = 0, class Pack, class Function>
decltype(auto) visit_species(Pack& pack, unsigned index, Function&& function) {
  if (index == S) return function.template operator()<S>(std::get<S>(pack));
  if constexpr (S + 1 < std::tuple_size_v<std::remove_const_t<Pack>>)
    return visit_species<S + 1>(pack, index, std::forward<Function>(function));
  else throw std::logic_error("TPIC species: species index out of range");
}

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_STATE_SPECIES_DEVICE_HPP
