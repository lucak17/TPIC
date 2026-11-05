#ifndef TPIC_DIAGNOSTICS_ASYNC_PARTICLE_SNAPSHOT_DEVICE_HPP
#define TPIC_DIAGNOSTICS_ASYNC_PARTICLE_SNAPSHOT_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <tpic/data/output/output_snapshot.hpp>
#include <tpic/data/particles/particle_schema.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace tpic::diagnostics {

// Captures the committed active prefix of each species as its raw stored
// records: one device-to-host copy per attribute on the transfer stream into one
// pinned image per species, with no kernel and no device staging. lend_into
// hands the images to a writer through its snapshot, the output's and the
// checkpoint's alike; the capture rewrites them only in its next capture, after
// every writer has returned the snapshots that borrow them. The next mover can
// overwrite the sources once transfer_done_event_ completes.
template <class Backend> class AsyncParticleSnapshot {
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Pinned = lpfx::device::PinnedHostAllocation<Backend>;
  enum class State { idle, capturing, transferring, ready };
  // Every attribute block of an image starts on this boundary.
  static constexpr std::size_t block_alignment = 64;
  struct SpeciesImage {
    Pinned host;
    data::ParticleRecords records;
    bool captured{};
  };

public:
  using Executor = lpfx::device::Executor<Backend>;

  AsyncParticleSnapshot(std::size_t species, Executor producer, Executor transfer)
      : producer_executor_(producer), transfer_executor_(transfer), species_(species) {
    lpfx::require(producer.bound() && transfer.bound() && species != 0,
                  "TPIC particle snapshot: invalid executors or species");
    lpfx::device::create_event(capture_ready_event_, "TPIC particle snapshot: event allocation failed");
    lpfx::device::create_event(transfer_done_event_, "TPIC particle snapshot: event allocation failed");
  }
  AsyncParticleSnapshot(const AsyncParticleSnapshot&) = delete;
  AsyncParticleSnapshot& operator=(const AsyncParticleSnapshot&) = delete;
  ~AsyncParticleSnapshot() noexcept {
    if (pending() && !Traits::succeeded(Traits::synchronize_stream(transfer_executor_.stream())))
      std::abort();
  }
  // A capture is being copied.
  [[nodiscard]] bool pending() const noexcept {
    return state_ == State::capturing || state_ == State::transferring;
  }
  [[nodiscard]] std::size_t pinned_bytes() const noexcept {
    std::size_t bytes{};
    for (const auto& species : species_) bytes += species.host.capacity_bytes();
    return bytes;
  }
  // The producer stream has already waited for the committed state: the
  // transfer stream starts copying only after this point of the producer.
  void begin() {
    lpfx::require(!pending(), "TPIC particle snapshot: previous capture still copying");
    for (auto& species : species_) species.captured = false;
    sources_recorded_ = false;
    lpfx::device::record_event(capture_ready_event_, producer_executor_,
        "TPIC particle snapshot: capture readiness event failed");
    lpfx::device::wait_event(capture_ready_event_, transfer_executor_,
        "TPIC particle snapshot: transfer dependency failed");
    state_ = State::capturing;
  }
  template <class Precision, class Owner> void copy(const Owner& particles, std::size_t index) {
    static_assert(std::is_same_v<Backend, typename Owner::backend_type>);
    using Schema = data::ParticleSchema<Precision>;
    lpfx::require(state_ == State::capturing && index < species_.size(),
                  "TPIC particle snapshot: no active capture or invalid species");
    auto& species = species_[index];
    lpfx::require(!species.captured && particles.valid() && particles.size() <= particles.capacity(),
                  "TPIC particle snapshot: duplicate species or invalid owner");
    const std::size_t count = particles.size();
    const auto block = [count](std::size_t width) {
      return (count * width + block_alignment - 1) / block_alignment * block_alignment;
    };
    lpfx::require(count <= std::numeric_limits<std::size_t>::max() / 2 / Schema::type::record_width,
                  "TPIC particle snapshot: allocation overflow");
    std::size_t bytes = 0;
    for (const auto width : Schema::type::widths) bytes += block(width);
    reserve(species, bytes);
    auto* cursor = static_cast<std::byte*>(species.host.raw_base());
    const auto copy_attribute = [&]<class Attribute>() {
      using Value = typename Attribute::value_type;
      auto* destination = reinterpret_cast<Value*>(cursor);
      cursor += block(sizeof(Value));
      if (count != 0)
        lpfx::device::check_device_error<Backend>(Traits::copy_device_to_host_async(
            destination, particles.template get<Attribute>(), count * sizeof(Value),
            transfer_executor_.stream()), "TPIC particle snapshot: transfer submission failed");
      return std::span<const Value>(destination, count);
    };
    species.records = data::ParticleRecords{
        .species = index,
        .fraction = {copy_attribute.template operator()<typename Schema::Rx>(),
                     copy_attribute.template operator()<typename Schema::Ry>(),
                     copy_attribute.template operator()<typename Schema::Rz>()},
        .velocity = {copy_attribute.template operator()<typename Schema::U>(),
                     copy_attribute.template operator()<typename Schema::V>(),
                     copy_attribute.template operator()<typename Schema::W>()},
        .charge = copy_attribute.template operator()<typename Schema::Charge>(),
        .cell = copy_attribute.template operator()<typename Schema::Cell>(),
        .id = copy_attribute.template operator()<typename Schema::Id>()};
    species.captured = true;
  }
  void finish_capture() {
    lpfx::require(state_ == State::capturing, "TPIC particle snapshot: no active capture");
    for (const auto& species : species_)
      lpfx::require(species.captured, "TPIC particle snapshot: species not captured");
    // The copies read the sources: their completion releases them and completes the images.
    lpfx::device::record_event(transfer_done_event_, transfer_executor_,
        "TPIC particle snapshot: transfer event failed");
    sources_recorded_ = true;
    state_ = State::transferring;
  }
  [[nodiscard]] bool ready() {
    lpfx::require(state_ == State::transferring || state_ == State::ready,
                  "TPIC particle snapshot: transfer was not submitted");
    if (state_ == State::ready) return true;
    const auto status = transfer_done_event_.query();
    if (Traits::not_ready(status)) return false;
    lpfx::device::check_device_error<Backend>(status, "TPIC particle snapshot: transfer query failed");
    state_ = State::ready;
    return true;
  }
  void host_wait() {
    lpfx::require(state_ == State::transferring || state_ == State::ready,
                  "TPIC particle snapshot: transfer was not submitted");
    lpfx::device::check_device_error<Backend>(transfer_done_event_.synchronize(),
                                              "TPIC particle snapshot: transfer completion failed");
    state_ = State::ready;
  }
  void device_wait_sources(Executor executor) const {
    lpfx::require(sources_recorded_ && executor.bound(),
                  "TPIC particle snapshot: source-read completion was not recorded");
    lpfx::device::wait_event(transfer_done_event_, executor,
        "TPIC particle snapshot: source consumer dependency failed");
  }
  // Lends the completed images: output.particles borrows them until the writer
  // returns the snapshot. The engines validate and convert the records.
  void lend_into(data::OwnedOutputSnapshot& output) {
    lpfx::require(ready(), "TPIC particle snapshot: transfer is not complete");
    output.particles.clear();
    for (const auto& species : species_) output.particles.push_back(species.records);
  }

private:
  // Grows an image to the largest population seen; the old image is released
  // first, so a growth never holds two images.
  void reserve(SpeciesImage& species, std::size_t bytes) {
    if (bytes <= species.host.capacity_bytes()) return;
    species.host = Pinned{};
    lpfx::device::check_device_error<Backend>(Pinned::create(bytes, species.host),
                                              "TPIC particle snapshot: pinned allocation failed");
  }
  Executor producer_executor_, transfer_executor_;
  std::vector<SpeciesImage> species_;
  lpfx::device::DeviceEvent<Backend> capture_ready_event_, transfer_done_event_;
  State state_{State::idle};
  bool sources_recorded_{};
};

} // namespace tpic::diagnostics

#endif // TPIC_DIAGNOSTICS_ASYNC_PARTICLE_SNAPSHOT_DEVICE_HPP
