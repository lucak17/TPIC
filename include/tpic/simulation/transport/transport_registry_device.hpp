#ifndef TPIC_SIMULATION_TRANSPORT_TRANSPORT_REGISTRY_DEVICE_HPP
#define TPIC_SIMULATION_TRANSPORT_TRANSPORT_REGISTRY_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/distributed/runtime/mpi_file_writer.hpp>
#include <tpic/communication/transport_routes.hpp>
#include <tpic/communication/mass_halo/mass_halo_device.hpp>
#include <tpic/diagnostics/gauss_residual_device.hpp>
#include <tpic/diagnostics/scalar_diagnostics_device.hpp>
#include <tpic/ecsim/gmres/gmres_device.hpp>
#include <tpic/ecsim/maxwell/maxwell_device.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/simulation/execution/active_backend_device.hpp>
#include <tpic/simulation/execution/execution_streams.hpp>
#include <tpic/simulation/state/species_device.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tpic::simulation {

// One declaration recipe supplies both the exact communication type and its
// preparation values. TPIC supplies every executor and retains all numerics.
template <class Precision, std::size_t Ns, bool Partial> class TransportRegistry {
public:
  using MassStorage = communication::MassHaloStorage<Precision, Backend>;
  // The species pack, declared here once. std::array constructs its elements
  // in index order and destroys them in reverse; std::get<S> and
  // std::tuple_size apply to it.
  using SpeciesPack = std::array<Species<Precision, Partial>, Ns>;

  // Cold inputs only. The simulation supplies these after owners and streams
  // reach their final addresses; no copy is retained after prepare().
  struct Resources {
    const simulation::PreparedInput& input;
    data::DeviceFieldImage<Precision, Backend>& fields;
    data::MagneticNodeOwner<Precision, Backend>& magnetic_node;
    data::CurrentOwner<Precision, Backend>& current;
    SpeciesPack& species;
    initialization::VelocityCentering<Precision, Backend>& centering;
    ecsim::MaxwellOperator<Precision, Backend>& maxwell;
    ecsim::Gmres<Precision, Backend>& gmres;
    diagnostics::ScalarDiagnostics<Precision, Backend>& diagnostics;
    diagnostics::GaussResidual<Precision, Backend>& gauss;
    data::OutputMomentOwner<Precision, Backend>& output_moments;
    const communication::ParticleMigrationGeometry& geometry;
    const StreamExecutors<Backend, Ns>& executors;
  };

private:
  static auto describe(const Resources& r, MassStorage& mass_storage) noexcept {
    namespace d = lpfx::distributed;
    using Route = communication::DeviceRoute;
    const auto fields = d::fields(r.input.domain.field_neighborhood, d::uniform_halos);
    const auto startup = fields.template on<Route>(r.executors.control);
    const auto moments = fields.template on<Route>(r.executors.current);
    const auto solver = fields.template on<Route>(r.executors.solver);
    const auto next_b = fields.template on<Route>(r.executors.field_update);
    const auto gauss = fields.template on<Route>(r.executors.diagnostics);
    return [&]<std::size_t... Index>(std::index_sequence<Index...>) {
      return d::operations(
          d::copy_ghosts<"B.cell">(r.fields.magnetic_cell, startup),
          d::copy_ghosts<"B.node">(r.magnetic_node, startup),
          d::sum_shared<"J.add">(r.current, moments),
          d::copy_ghosts<"J.ghosts">(r.current, moments),
          d::all_reduce<"centering">(r.centering.collective_operand(), d::sum,
                                    d::on<Route>(r.executors.control)),
          d::copy_ghosts<"solver.theta">(r.maxwell.theta, solver),
          d::copy_ghosts<"solver.input">(r.maxwell.input, solver),
          d::copy_ghosts<"solver.curl">(r.maxwell.curl_cell, solver),
          d::copy_ghosts<"solver.smooth_a">(r.maxwell.smooth_a, solver),
          d::copy_ghosts<"solver.smooth_b">(r.maxwell.smooth_b, solver),
          d::copy_ghosts<"B.next.cell">(r.maxwell.next_cell, next_b),
          d::all_reduce<"gmres">(r.gmres.collective_operand(), d::sum,
                                 d::on<communication::GmresRoute>(r.executors.solver)),
          d::all_reduce<"diagnostics">(r.diagnostics.collective_operand(), d::sum,
                                       d::on<Route>(r.executors.diagnostics)),
          d::sum_shared<"gauss.charge">(r.gauss.charge, gauss),
          d::copy_ghosts<"gauss.charge.ghosts">(r.gauss.charge, gauss),
          d::copy_ghosts<"gauss.smooth">(r.gauss.smooth, gauss),
          d::all_reduce<"gauss">(r.gauss.collective_operand(), d::sum, d::on<Route>(r.executors.diagnostics)),
          d::sum_shared<"output.moments">(r.output_moments,
                                          fields.template on<Route>(r.executors.output_capture)),
          d::prepared<"mass.add">(mass_storage.plan().template phase<true>().message_plan(),
                                  mass_storage.template buffers<true>(), d::on<Route>(r.executors.mass)),
          d::prepared<"mass.ghosts">(mass_storage.plan().template phase<false>().message_plan(),
                                     mass_storage.template buffers<false>(), d::on<Route>(r.executors.mass)),
          std::tuple{std::get<Index>(r.species).template migration_request<Index, Route>(
              r.input, r.geometry, r.executors.route)...});
    }(std::make_index_sequence<Ns>{});
  }

public:
  using Communication = lpfx::distributed::communication_for<decltype(describe(
      std::declval<const Resources&>(), std::declval<MassStorage&>()))>;
  // The type of a registered operation, named by its key.
  template <lpfx::field::fixed_string Key, std::uint64_t... Indices>
  using Operation = std::remove_reference_t<
      decltype(std::declval<Communication&>().template get<Key, Indices...>())>;

  TransportRegistry(const config::PreparedDomain& domain, data::CompactMassMatrix<Precision, Backend>& mass)
      : mass_storage_(domain, mass) {}
  TransportRegistry(const TransportRegistry &) = delete;
  TransportRegistry &operator=(const TransportRegistry &) = delete;

  void prepare(const lpfx::distributed::CommunicationSetup& setup, const Resources& resources) {
    lpfx::require(!prepared_ && !closed_, "TPIC communication: registry prepared twice");
    mass_storage_.prepare(setup, resources.executors.mass);
    lpfx::check_error(lpfx::distributed::prepare(communication_, setup,
                                                 describe(resources, mass_storage_)),
                      "TPIC communication: complete registration failed");
    lpfx::check_error(field_files_.try_open(setup), "TPIC communication: field file communicator failed");
    prepared_ = true;
  }
  template <lpfx::field::fixed_string Key, std::uint64_t... Indices>
  decltype(auto) get() { return communication_.template get<Key, Indices...>(); }
  MassStorage &mass_storage() noexcept { return mass_storage_; }
  // The output writer's collective file writes.
  lpfx::distributed::MpiFileWriter &field_files() noexcept { return field_files_; }
  void close() {
    if (closed_) return;
    lpfx::require(prepared_, "TPIC communication: registry closed before preparation");
    lpfx::require(mass_storage_.quiescent() && communication_.template get<"mass.add">().reusable() &&
                      communication_.template get<"mass.ghosts">().reusable(),
                  "TPIC communication: close requires a quiescent mass halo");
    lpfx::check_error(communication_.close(), "TPIC communication: close failed");
    mass_storage_.close();
    field_files_.close();
    closed_ = true;
  }

private:
  MassStorage mass_storage_;
  Communication communication_;
  lpfx::distributed::MpiFileWriter field_files_;
  bool prepared_{}, closed_{};
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_TRANSPORT_TRANSPORT_REGISTRY_DEVICE_HPP
