#include <lpfx/profiling.hpp>

#include <tpic/io/async_output_writer.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tpic::io {

void AsyncOutputWriter::configure(const config::PreparedOutputSelection& selection,
                                 const config::PreparedDomain& domain,
                                 std::unique_ptr<OutputEngine> engine) {
  if (configured_)
    throw std::logic_error("TPIC output: asynchronous output can only be configured once");
  const auto& output = selection.output();
  fields_enabled_ = output.fields_enabled();
  particles_enabled_ = output.particles_enabled();
  if (!fields_enabled_ && !particles_enabled_ && !output.checkpoints_enabled())
    throw std::invalid_argument("TPIC output: asynchronous output requires enabled fields, particles or checkpoints");
  if (output.save_dir.empty())
    throw std::invalid_argument("TPIC output: output directory is empty");
  if (!domain.field_neighborhood.is_valid() || domain.interior_node_count <= 0)
    throw std::invalid_argument("TPIC output: output requires a prepared domain");
  arrays_.assign(selection.arrays().begin(), selection.arrays().end());
  node_count_ = static_cast<std::size_t>(domain.interior_node_count);
  cell_count_ = static_cast<std::size_t>(domain.interior_cell_count);
  species_count_ = selection.species_count();
  engine_ = std::move(engine);
  root_ = domain.field_neighborhood.topology().input().rank() == 0;
  configured_ = true;
}

void AsyncOutputWriter::submit(data::OwnedOutputSnapshot&& snapshot) {
  if (!configured_)
    throw std::logic_error("TPIC output: asynchronous writer is not configured");
  if (snapshot_)
    throw std::logic_error("TPIC output: asynchronous output previous snapshot is pending");
  if (generation_ == lpfx::HostFence::max_generation)
    throw std::overflow_error("TPIC output: asynchronous output generation exhausted");
  validate(snapshot);
  snapshot_.emplace(std::move(snapshot));
  failure_ = nullptr;
  // The private worker remains open and at most one snapshot is admitted.
  // LPFx publishes its owned data before the program can access it.
  if (!worker_.submit(std::uint64_t{++generation_})) std::terminate();
}

void AsyncOutputWriter::host_wait_piece() {
  if (!snapshot_)
    throw std::logic_error("TPIC output: asynchronous output has no pending snapshot");
  if (!completed_fence_.wait(generation_)) std::terminate();
  if (failure_) std::rethrow_exception(failure_);
}

data::OwnedOutputSnapshot AsyncOutputWriter::complete_snapshot() {
  LPFX_PROFILE_SCOPE("TPIC::Output::PublishSnapshot");
  if (!snapshot_ || !completed_fence_.ready(generation_))
    throw std::logic_error("TPIC output: asynchronous output completion requires a finished piece");
  if (failure_) std::rethrow_exception(failure_);
  // The completion fence transfers writer visibility to the controller. The
  // worker cannot access it again until a new snapshot is admitted.
  if (root_) engine_->publish();
  auto completed = std::move(*snapshot_);
  snapshot_.reset();
  return completed;
}

const std::string& AsyncOutputWriter::fields_path() const {
  if (snapshot_)
    throw std::logic_error("TPIC output: asynchronous output path requires completed publication");
  if (!configured_)
    throw std::logic_error("TPIC output: asynchronous output is not configured");
  return engine_->manifest_path();
}

const std::string& AsyncOutputWriter::particles_path() const {
  if (snapshot_ || !configured_)
    throw std::logic_error("TPIC output: asynchronous output particle path requires completed publication");
  return engine_->particles_path();
}

// Shape checks only: the field capture rejects nonfinite values before it
// copies into the owned snapshot.
void AsyncOutputWriter::validate(const data::OwnedOutputSnapshot& snapshot) const {
  if (!snapshot.fields_present && !snapshot.particles_present && !snapshot.checkpoint_present)
    throw std::invalid_argument("TPIC output: output snapshot contains no selected data");
  if (snapshot.checkpoint_present) {
    if (snapshot.fields_present || snapshot.particles_present)
      throw std::invalid_argument("TPIC output: a checkpoint snapshot carries only the checkpoint");
    for (std::size_t field = 0; field < snapshot.checkpoint_fields.size(); ++field)
      if (std::visit([](const auto& values) { return values.size(); }, snapshot.checkpoint_fields[field]) !=
          (field < data::checkpoint_node_fields ? node_count_ : cell_count_))
        throw std::invalid_argument("TPIC output: checkpoint field capture shape mismatch");
  }
  if (snapshot.fields_present) {
    if (!fields_enabled_ || snapshot.arrays.size() != arrays_.size())
      throw std::invalid_argument("TPIC output: output capture is missing configured fields");
    for (std::size_t i = 0; i < arrays_.size(); ++i) {
      const auto& array = snapshot.arrays[i];
      const auto& descriptor = arrays_[i];
      const auto samples = descriptor.cell ? cell_count_ : node_count_;
      if (descriptor.components <= 0 || array.name != descriptor.name ||
          array.components != descriptor.components || array.cell != descriptor.cell ||
          samples > std::numeric_limits<std::size_t>::max() /
                        static_cast<std::size_t>(descriptor.components) ||
          (array.values.empty() ? !descriptor.cell
                                : array.values.size() != samples *
                                                             static_cast<std::size_t>(descriptor.components)))
        throw std::invalid_argument("TPIC output: output field capture order/shape mismatch");
    }
  }
  if (!snapshot.particles_present && !snapshot.checkpoint_present) return;
  if ((snapshot.particles_present && !particles_enabled_) || snapshot.particles.size() != species_count_)
    throw std::invalid_argument("TPIC output: output particle capture is missing configured species");
  // Selection bounds the species count at configuration. Keep this tiny
  // per-epoch validation workspace inline instead of allocating a bitmap.
  std::array<bool, config::max_real_species> seen{};
  for (const auto& particles : snapshot.particles) {
    if (particles.species >= species_count_ || seen[particles.species])
      throw std::invalid_argument("TPIC output: output particle capture has invalid or repeated species");
    seen[particles.species] = true;
    const auto count = particles.id.size();
    if (particles.cell.size() != count)
      throw std::invalid_argument("TPIC output: output particle capture shape mismatch");
    for (const auto& values : {particles.fraction[0], particles.fraction[1], particles.fraction[2],
                               particles.velocity[0], particles.velocity[1], particles.velocity[2],
                               particles.charge})
      if (std::visit([](auto stored) { return stored.size(); }, values) != count)
        throw std::invalid_argument("TPIC output: output particle capture shape mismatch");
  }
}

void AsyncOutputWriter::run_piece(std::uint64_t generation) noexcept {
  LPFX_PROFILE_SCOPE("TPIC::Output::WritePiece");
  const auto& snapshot = *snapshot_;
  try {
    if (final_written_ || snapshot.cycle < 0 || snapshot.cycle <= last_cycle_)
      throw std::invalid_argument("TPIC output: output cycles must increase before one final snapshot");
    if (!std::isfinite(snapshot.time) || snapshot.time < 0.0 ||
        (last_cycle_ >= 0 && snapshot.time <= last_time_))
      throw std::invalid_argument("TPIC output: output times must be finite and increase");
    engine_->write_piece(data::OutputSnapshotView(snapshot));
    last_cycle_ = snapshot.cycle;
    last_time_ = snapshot.time;
    final_written_ = snapshot.final_state;
  } catch (...) {
    failure_ = std::current_exception();
  }
  if (!completed_fence_.signal(generation)) std::terminate();
}

} // namespace tpic::io
