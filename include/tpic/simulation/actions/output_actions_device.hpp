#ifndef TPIC_SIMULATION_ACTIONS_OUTPUT_ACTIONS_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_OUTPUT_ACTIONS_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/core/error_reporting.hpp>

#include <lpfx/profiling.hpp>

#include <tpic/communication/fields/field_exchange.hpp>
#include <tpic/data/fields/field_storage.hpp>

#include <tpic/config/output_selection.hpp>

#include <tpic/diagnostics/async_field_snapshot_device.hpp>
#include <tpic/diagnostics/output_moment_deposition_device.hpp>
#include <tpic/diagnostics/async_particle_snapshot_device.hpp>
#include <tpic/ecsim/deposition/moment_field_device.hpp>
#include <tpic/io/async_output_writer.hpp>
#include <tpic/io/checkpoint.hpp>

#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/data/output/output_moment_schema.hpp>
#include <tpic/simulation/execution/active_backend_device.hpp>
#include <tpic/simulation/execution/execution_layout.hpp>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace tpic::simulation {

// The snapshot capture of the committed state: field and particle snapshots,
// the output moments and the checkpoints, which share the particle capture.
// The capture stream is bound to the snapshots and the moment owner; the
// writer threads and the host captures belong to the caller.
template <class Precision, class SpeciesPack, class Ports> class OutputActions {
  // The compiled species capacity; it also marks "no species deposits moments".
  static constexpr std::size_t Ns = std::tuple_size_v<SpeciesPack>;

public:
  OutputActions(const simulation::PreparedInput &input,
              RunMode mode, data::DeviceFieldImage<Precision, Backend> &fields,
              data::MagneticNodeOwner<Precision, Backend> &magnetic_node,
              const data::GaussCellOwner<Precision, Backend> &gauss_residual, SpeciesPack &species,
              Ports &ports, Executor executor, Executor transfer_executor)
      : input_(input), fields_(fields), magnetic_node_(magnetic_node), gauss_residual_(gauss_residual),
        species_(species), ports_(ports), executor_(executor), selection_(input.output),
        output_cells_(data::prepare_particle_cell_map(input.domain)),
        output_moment_plan_(ecsim::prepare_moment_field_plan(input.domain)),
        output_moments_(data::make_zero_field<typename data::OutputMomentSchema<Precision>::type>(
            input.domain, executor)),
        checkpoint_(prepare_checkpoint_header(input)) {
    if (!snapshots_enabled(input, mode)) return;
    if (input.output.fields_enabled())
      snapshot_ = std::make_unique<FieldSnapshot>(input.domain, selection_.arrays(), executor_,
                                             transfer_executor, selection_.scratch_component_count());
    if (input.output.particles_enabled() || input.output.checkpoints_enabled())
      particle_snapshot_ = std::make_unique<ParticleSnapshot>(input.particles.species.size(), executor_,
                                                              transfer_executor);
  }
  data::OutputMomentOwner<Precision, Backend> &moments() noexcept { return output_moments_; }
  bool fields_enabled() const noexcept { return snapshot_ != nullptr; }
  bool particles_enabled() const noexcept {
    return particle_snapshot_ != nullptr && input_.output.particles_enabled();
  }
  bool checkpoints_enabled() const noexcept {
    return particle_snapshot_ != nullptr && input_.output.checkpoints_enabled();
  }
  // The capture streams exist: an output channel or checkpoints are enabled.
  bool enabled() const noexcept { return snapshot_ != nullptr || particle_snapshot_ != nullptr; }
  bool fields_due(int cycle) const noexcept {
    return fields_enabled() && input_.output.should_write_fields(cycle);
  }
  bool particles_due(int cycle) const noexcept {
    return particles_enabled() && input_.output.should_write_particles(cycle);
  }
  // At the cadence, and in the final state with CallFinalize.
  bool checkpoint_due(int cycle, bool final_state = false) const noexcept {
    return checkpoints_enabled() &&
           (input_.output.should_write_checkpoint(cycle) || (final_state && input_.output.checkpoint_final));
  }
  // A snapshot of VTK fields alone publishes nothing: each shared file is
  // complete when its collective write returns.
  bool publishes(const Epoch &epoch) const noexcept {
    return epoch.particles || (epoch.fields && input_.output.field_engine != config::OutputEngineKind::Vtk);
  }
  void configure(io::AsyncOutputWriter &writer) const {
    if (fields_enabled() || particles_enabled())
      writer.configure(selection_, input_.domain,
                       io::make_output_engine(selection_, input_.domain, config::precision_metadata(input_.precision),
                                              ports_.field_files()));
  }
  void configure_checkpoints(io::AsyncOutputWriter &writer) const {
    if (checkpoints_enabled())
      writer.configure(selection_, input_.domain, io::make_checkpoint_engine(input_.output, input_.domain));
  }

  // The particle capture serves the particle output and the checkpoint alike.
  void begin_capture(const Epoch &epoch) {
    epoch_ = epoch;
    if (captures_particles()) {
      particle_snapshot_->begin();
      for (std::size_t s = 0; s != input_.particles.species.size(); ++s)
        particle_snapshot_->template copy<Precision>(species_[s].particles, s);
      particle_snapshot_->finish_capture();
    }
    if (!epoch.fields) return;
    snapshot_->begin();
    for (const auto total : selection_.total_slots())
      if (total != config::no_slot)
        snapshot_->clear(total);
    if (input_.output.output_e)
      snapshot_->template pack<0, 1, 2>(fields_.electric, selection_.electric_slot());
    if (input_.output.output_b)
      snapshot_->template pack<0, 1, 2>(magnetic_node_, selection_.magnetic_slot());
    if (!input_.output.output_gauss) return;
    if (epoch.gauss_finish) snapshot_->template pack<0>(gauss_residual_, selection_.gauss_slot());
    else snapshot_->omit(selection_.gauss_slot());
  }
  [[nodiscard]] unsigned first_moment_species() const {
    for (unsigned s = 0; s != input_.particles.species.size(); ++s)
      if (moments_selected(s)) return s;
    return Ns;
  }
  [[nodiscard]] bool moments_selected(unsigned s) const {
    return selection_.species(s).deposition_mask != 0;
  }
  void deposit(unsigned s, unsigned *invalid) {
    lpfx::check_error(ecsim::enqueue_clear_moment_field(output_moments_, executor_),
                      "TPIC output: output moments clear failed");
    lpfx::check_error(diagnostics::enqueue_deposit_output_moments<Precision>(
                          species_[s].particles, output_moments_, output_cells_,
                          input_.domain.inverse_cell_volume, input_.parameters.species[s].q_over_m,
                          selection_.species(s).deposition_mask, invalid, executor_),
                      "TPIC output: output moment deposition failed");
  }
  void complete_assembly() {
    communication::complete_field_exchange(ports_.template get<"output.moments">(),
        "TPIC output: exchange rejected", "TPIC output: exchange failed");
    lpfx::check_error(
        ecsim::enqueue_scale_moment_dual_volume(output_moments_, output_moment_plan_, executor_),
        "TPIC output: output moment physical-volume correction failed");
  }
  // Shared endpoints agree after add; no consumer reads output ghosts.
  void pack(unsigned s) {
    [&]<std::size_t... Quantity>(std::index_sequence<Quantity...>) {
      (pack_moment<Quantity>(s), ...);
    }(std::make_index_sequence<config::output_quantity_count>{});
  }
  void detach() {
    snapshot_->finish_capture();
    LPFX_PROFILE_MARK("TPIC::OutputSourcesRead");
  }
  void device_wait_sources(Executor executor) const {
    if (epoch_.fields) snapshot_->device_wait_sources(executor);
    if (captures_particles()) particle_snapshot_->device_wait_sources(executor);
  }
  [[nodiscard]] bool transfer_ready() {
    return (!epoch_.fields || snapshot_->ready()) && (!captures_particles() || particle_snapshot_->ready());
  }
  void host_wait_transfer() {
    if (epoch_.fields) snapshot_->host_wait();
    if (captures_particles()) particle_snapshot_->host_wait();
  }
  // The checkpoint of the captured state without its particle records: the
  // header and E at the local nodes and B at the local cells, copied on the
  // capture stream, which waited for the committed state, and complete on return.
  void host_blocking_capture_checkpoint(data::OwnedOutputSnapshot &checkpoint, double initial_energy) {
    checkpoint.cycle = epoch_.cycle;
    checkpoint.time = double(epoch_.cycle) * input_.parameters.dt;
    checkpoint.final_state = epoch_.final_state;
    checkpoint.fields_present = checkpoint.particles_present = false;
    checkpoint.checkpoint_present = true;
    checkpoint.checkpoint = checkpoint_;
    checkpoint.checkpoint.cycle = epoch_.cycle;
    checkpoint.checkpoint.initial_energy = initial_energy;
    copy_interiors(fields_.electric, 1, checkpoint, 0);
    copy_interiors(fields_.magnetic_cell, 0, checkpoint, data::checkpoint_node_fields);
    lpfx::device::check_device_error<Backend>(
        lpfx::device::BackendTraits<Backend>::synchronize_stream(executor_.stream()),
        "TPIC checkpoint: field capture failed");
  }
  void lend_checkpoint_particles(data::OwnedOutputSnapshot &checkpoint) {
    particle_snapshot_->lend_into(checkpoint);
  }
  // Copies the field arrays and lends the particle records into the capture.
  void copy_into(data::OwnedOutputSnapshot &capture) {
    capture.cycle = epoch_.cycle;
    capture.time = double(epoch_.cycle) * input_.parameters.dt;
    capture.final_state = epoch_.final_state;
    capture.fields_present = epoch_.fields;
    capture.particles_present = epoch_.particles;
    if (epoch_.fields) snapshot_->copy_into(capture);
    if (epoch_.particles) particle_snapshot_->lend_into(capture);
  }

private:
  [[nodiscard]] bool captures_particles() const noexcept { return epoch_.particles || epoch_.checkpoint; }
  // The three components of a field owner, interior only (both ends of every
  // axis at nodes), into checkpoint fields [first, first + 3): one 2D copy per
  // z plane. The ghosts are not read: rhs_fields may be writing B ghosts.
  template <class Owner>
  void copy_interiors(const Owner &field, int nodes, data::OwnedOutputSnapshot &checkpoint, std::size_t first) {
    const auto x = std::size_t(input_.domain.local_cells[0] + nodes);
    const auto y = std::size_t(input_.domain.local_cells[1] + nodes);
    const auto z = std::size_t(input_.domain.local_cells[2] + nodes);
    [&]<std::size_t... C>(std::index_sequence<C...>) {
      ([&] {
        const auto view = field.view().template component<C>();
        using Value = std::remove_cv_t<std::remove_pointer_t<decltype(view.core_data())>>;
        auto &values = checkpoint.checkpoint_fields[first + C];
        auto &host = std::holds_alternative<std::vector<Value>>(values)
            ? std::get<std::vector<Value>>(values) : values.template emplace<std::vector<Value>>();
        host.resize(x * y * z);
        for (std::size_t k = 0; k < z; ++k)
          lpfx::device::check_device_error<Backend>(
              lpfx::device::BackendTraits<Backend>::copy_device_to_host_2d_async(
                  host.data() + k * x * y, x * sizeof(Value), view.core_data() + k * view.stride_z(),
                  view.stride_y() * sizeof(Value), x * sizeof(Value), y, executor_.stream()),
              "TPIC checkpoint: field capture submission failed");
      }(), ...);
    }(std::make_index_sequence<3>{});
  }

  template <std::size_t Quantity> void pack_moment(std::size_t species) {
    constexpr auto quantity = data::output_moment_quantities[Quantity];
    const auto slot = selection_.species(species).slots[Quantity];
    const auto total = selection_.total_slots()[Quantity];
    if (slot == config::no_slot && total == config::no_slot)
      return;
    constexpr auto first = quantity.first_component;
    [&]<std::size_t... Component>(std::index_sequence<Component...>) {
      if (slot != config::no_slot && total != config::no_slot)
        snapshot_->template pack_and_add<(first + Component)...>(output_moments_, slot, total,
                                                                 quantity.scale);
      else if (slot != config::no_slot)
        snapshot_->template pack<(first + Component)...>(output_moments_, slot, quantity.scale);
      else
        snapshot_->template add<(first + Component)...>(output_moments_, total, quantity.scale);
    }(std::make_index_sequence<quantity.components>{});
  }

  const simulation::PreparedInput &input_;
  data::DeviceFieldImage<Precision, Backend> &fields_;
  data::MagneticNodeOwner<Precision, Backend> &magnetic_node_;
  const data::GaussCellOwner<Precision, Backend> &gauss_residual_;
  SpeciesPack &species_;
  Ports &ports_;
  Executor executor_;
  const config::PreparedOutputSelection selection_;
  const data::ParticleCellMap output_cells_;
  const ecsim::MomentFieldPlan output_moment_plan_; // Prepared once at construction.
  data::OutputMomentOwner<Precision, Backend> output_moments_;
  using FieldSnapshot = diagnostics::AsyncFieldSnapshot<Backend>;
  std::unique_ptr<FieldSnapshot> snapshot_;
  using ParticleSnapshot = diagnostics::AsyncParticleSnapshot<Backend>;
  std::unique_ptr<ParticleSnapshot> particle_snapshot_;
  // A capture adds its cycle and energy baseline.
  const data::CheckpointHeader checkpoint_;
  Epoch epoch_{}; // the epoch being captured
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_OUTPUT_ACTIONS_DEVICE_HPP
