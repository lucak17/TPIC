#ifndef TPIC_SIMULATION_ACTIONS_LOCAL_SPECIES_ACTIONS_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_LOCAL_SPECIES_ACTIONS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/distributed/particle/migration/stage.hpp>
#include <lpfx/distributed/runtime/host_control.hpp>
#include <lpfx/distributed/runtime/error_reporting.hpp>
#include <tpic/communication/particles/initial_particle_classification_device.hpp>
#include <tpic/simulation/execution/execution_layout.hpp>
#include <tpic/simulation/state/species_device.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <variant>

namespace tpic::simulation {

// The mutation of one species' owner: startup boundaries and classification,
// migration stages including the route, sorter retirement and ordering, and
// the checked owner publication.
template <class Precision, bool Partial, class SpeciesPack, class Ports>
class LocalSpeciesActions {
  static constexpr std::size_t Ns = std::tuple_size_v<SpeciesPack>;
  using MigrationStage = lpfx::distributed::ParticleMigrationStage;
public:
  LocalSpeciesActions(const RunPlan& plan, const lpfx::device::DeviceProperties& properties,
      const lpfx::MpiCallDomain& domain, SpeciesPack& species,
      initialization::VelocityCentering<Precision, Backend>& centering, Ports& ports,
      const communication::ParticleMigrationGeometry& geometry)
      : plan_(plan), input_(plan.input), properties_(properties), domain_(domain), species_(species),
        centering_(centering), ports_(ports), geometry_(geometry) {}
  [[nodiscard]] bool bypassed() const noexcept {
    return !Partial && input_.particles.bypass_local_migration;
  }
  [[nodiscard]] bool full_order() const noexcept {
    return !Partial && input_.particles.order == config::ParticleOrder::full;
  }

  void host_blocking_center(unsigned s) {
    std::visit([&](const auto& selected) {
      centering_.host_blocking_center_if_requested(plan_.particle_loads[s], selected.profile(s),
                                     species_[s].particles, ports_.template get<"centering">());
    }, input_.initial_case);
  }
  // The whole-domain owner only wraps or reflects existing particles: the
  // startup boundaries need no migration slot per particle.
  void apply_initial_boundaries(unsigned s, unsigned* invalid, Executor control) {
    lpfx::check_error(communication::enqueue_local_initial_particle_boundaries<Precision>(
                          species_[s].particles, geometry_, invalid, control),
                      "TPIC local species: startup particle boundary submission failed");
  }
  void classify_initial(unsigned s, Executor control) {
    visit(s, [&]<std::size_t S>(auto& species, auto& operation) {
      lpfx::check_error(communication::enqueue_initial_particle_classification<Precision>(
                            operation, species.particles, geometry_, control),
                        "TPIC local species: particle boundary classification publication failed");
    });
  }
  // The bin epoch survives through the mover, whose position changes make it
  // physically stale. Join and retire all readers before compaction or growth.
  void host_wait_retired(unsigned s, Executor local) {
    visit(s, [&]<std::size_t S>(auto& species, auto& operation) {
      species.ordering.host_wait_retired(local);
      if constexpr (Partial)
        lpfx::require(operation.reusable(),
                      "TPIC local species: partial migration retained an active communication lease");
    });
  }
  void host_wait_resident(unsigned s) {
    visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      lpfx::check_error(operation.template wait<MigrationStage::resident>(),
                        "TPIC local species: particle resident compaction failed");
    });
  }
  void host_wait_route(unsigned s) {
    visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      lpfx::check_error(operation.template wait<MigrationStage::route>(),
                        "TPIC local species: particle routing failed");
    });
  }
  void host_wait_sorted(unsigned s) { species_[s].ordering.host_wait_complete(); }
  decltype(auto) close_empty_readers(unsigned s, Executor local) {
    return species_[s].ordering.close_empty_readers(local);
  }
  [[nodiscard]] bool host_wait_finalize(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      return lpfx::check_progress(operation.template wait<MigrationStage::finalize>(),
                                  "TPIC local species: particle migration finalization failed");
    });
  }
  // TPIC retains the numerical receipt through deposition and the mover.
  void finish_initial(unsigned s) {
    visit(s, [&]<std::size_t S>(auto& species, auto& operation) {
      if constexpr (Partial) static_cast<void>(checked_completed_partial_receipt(species, operation));
      else lpfx::require(operation.reusable(), "TPIC local species: migration did not conclude");
    });
  }
  // Owner sizes are host metadata; LPFx keeps these small counters on its
  // host transport without uploading them into the device reduction path.
  void host_blocking_check_population() {
    const auto count = input_.particles.species.size();
    std::array<std::uint64_t, Ns> local{}, global{};
    for (std::size_t s = 0; s != count; ++s) local[s] = species_[s].particles.size();
    lpfx::check_mpi_error(lpfx::distributed::host_all_reduce<std::uint64_t,
                              lpfx::distributed::reduction_operation::sum>(
                              domain_, std::span<const std::uint64_t>{local.data(), count},
                              std::span<std::uint64_t>{global.data(), count}),
                          "TPIC local species: global particle count reduction failed");
    for (std::size_t s = 0; s != count; ++s)
      lpfx::require(global[s] == input_.particles.species[s].global_count,
                    "TPIC local species: migration changed the global species population");
  }

  void retire_after(unsigned s, const Event& moved_event, Executor local) {
    if (Partial || full_order()) species_[s].ordering.retire_after(moved_event);
    if constexpr (Partial) species_[s].ordering.order_reuse(local);
  }
  [[nodiscard]] bool progress_retirement(unsigned s) {
    if constexpr (Partial) return true; // The local stream explicitly inherits retirement.
    else return !full_order() || lpfx::check_progress(species_[s].ordering.progress_retirement(),
                                                       "TPIC local species: full sort retirement failed");
  }
  void begin(unsigned s, Executor local) {
    visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      lpfx::check_acceptance(operation.begin(local), "TPIC local species: migration begin failed");
    });
  }
  [[nodiscard]] bool progress_prepare(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      return lpfx::check_progress(operation.template progress<MigrationStage::prepare>(),
                                  "TPIC local species: particle preparation failed");
    });
  }
  [[nodiscard]] bool resident_access_ready(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) { return operation.resident_access_ready(); });
  }
  [[nodiscard]] auto acquire_resident_access(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      auto acquired = operation.acquire_resident_access();
      lpfx::check_error(acquired, "TPIC local species: resident writer acquisition failed");
      return std::move(acquired).value();
    });
  }
  template <class Range> void sort_resident(unsigned s, Range&& range, Executor sort) {
    species_[s].ordering.enqueue(std::forward<Range>(range), sort);
  }
  [[nodiscard]] bool progress_compact(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      return lpfx::check_progress(operation.template progress<MigrationStage::resident>(),
                                  "TPIC local species: resident compaction failed");
    });
  }
  [[nodiscard]] bool progress_sort(unsigned s) {
    return lpfx::check_progress(species_[s].ordering.progress_complete(), "TPIC local species: particle sort failed");
  }
  [[nodiscard]] auto resident_read(unsigned s) {
    auto acquired = species_[s].ordering.resident_read();
    lpfx::require(acquired.has_value() && acquired.value().valid(),
                  "TPIC local species: checked resident capability acquisition failed");
    return acquired.value();
  }
  [[nodiscard]] bool progress_route(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      return lpfx::check_progress(operation.template progress<MigrationStage::route>(),
                                  "TPIC routing: particle migration route failed");
    });
  }
  [[nodiscard]] bool progress_release_access(unsigned s, const Event& readers_done) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      return lpfx::check_progress(operation.release_resident_access(readers_done),
                                  "TPIC local species: resident access release failed");
    });
  }
  void release_growth_scratch(unsigned s) {
    visit(s, [&]<std::size_t S>(auto& species, auto& operation) {
      const auto requirements = operation.finalization_requirements();
      lpfx::check_error(requirements, "TPIC local species: finalization before routing completion");
      if (requirements.value().requires_growth) species.ordering.release_growth_scratch();
    });
  }
  [[nodiscard]] bool progress_finalize(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto&, auto& operation) {
      return lpfx::check_progress(operation.template progress<MigrationStage::finalize>(),
                                  "TPIC local species: particle finalization failed");
    });
  }
  [[nodiscard]] bool progress_conclude(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto& species, auto& operation) {
      const bool ready =
          lpfx::check_progress(operation.conclude(), "TPIC local species: migration conclusion failed");
      if constexpr (Partial)
        if (ready && operation.storage_changed())
          species.ordering.rebind_after_growth(species.particles, properties_);
      return ready;
    });
  }
  void sort(unsigned s, Executor sort) {
    if constexpr (!Partial) {
      auto& species = species_[s];
      species.ordering.rebind_to_owner(species.particles, properties_, sort);
      species.ordering.enqueue(species.particles, input_.domain, sort);
    }
  }
  [[nodiscard]] ParticleRange completed(unsigned s) {
    return visit(s, [&]<std::size_t S>(auto& species, auto& operation) -> ParticleRange {
      std::uint32_t first = 0;
      if constexpr (Partial)
        first = checked_completed_partial_receipt(species, operation)->prefix_size();
      else if (full_order())
        lpfx::require(species.ordering.receipt().prefix_size() == species.particles.size(),
                      "TPIC local species: full sort does not cover the completed owner");
      const auto total = species.particles.size();
      lpfx::require(first <= total, "TPIC local species: completed prefix exceeds owner");
      return {first, total - first, total};
    });
  }
  // Classification storage can be refreshed only after LPFx relinquishes its lease.
  void refresh(unsigned s) {
    auto& species = species_[s];
    lpfx::require(species.slots.reusable(), "TPIC local species: migration retained classification resources");
    lpfx::check_error(species.slots.reserve(species.particles.capacity()),
                      "TPIC local species: classification resource refresh failed");
  }

private:
  // function.template operator()<S>(species, migration) for the species index s.
  template <class Function> decltype(auto) visit(unsigned s, Function&& function) {
    return visit_species(species_, s, [&]<std::size_t S>(auto& species) -> decltype(auto) {
      return function.template operator()<S>(species, ports_.template get<"particles", S>());
    });
  }

  const RunPlan& plan_;
  const simulation::PreparedInput& input_;
  const lpfx::device::DeviceProperties& properties_;
  const lpfx::MpiCallDomain& domain_;
  SpeciesPack& species_;
  initialization::VelocityCentering<Precision, Backend>& centering_;
  Ports& ports_;
  const communication::ParticleMigrationGeometry& geometry_;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_LOCAL_SPECIES_ACTIONS_DEVICE_HPP
