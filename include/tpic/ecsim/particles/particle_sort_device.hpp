#ifndef TPIC_ECSIM_PARTICLES_PARTICLE_SORT_DEVICE_HPP
#define TPIC_ECSIM_PARTICLES_PARTICLE_SORT_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/algorithm/particle/device/sort.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/data/particles/particle_schema.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tpic::ecsim {

template <class Precision, class Backend>
using ParticleSorter = lpfx::DeviceCellSort<
    typename data::ParticleSchema<Precision>::type,
    lpfx::ParticleAttributeKey<typename data::ParticleSchema<Precision>::Cell::name_type>,
    Backend>;

// Resources and construction recipe shared by both sort owners; each owner
// keeps its own submission, completion and retirement contract. Reverse
// destruction keeps the borrowed receipt, sorter, event and host lease in
// lifetime order: the sorter's inspection lease is released before its
// budget. Every receipt is retired before normal teardown; active runtime
// failures are handled by the simulation's communicator-abort boundary.
template <class Precision, class Backend> class ParticleSortOwner {
public:
  using Owner = lpfx::Particles<typename data::ParticleSchema<Precision>::type,
                                lpfx::DeviceSpace<Backend>>;
  using Executor = lpfx::device::Executor<Backend>;
  using Sorter = ParticleSorter<Precision, Backend>;
  using Receipt = typename Sorter::submission_type;

  ParticleSortOwner(const ParticleSortOwner&) = delete;
  ParticleSortOwner& operator=(const ParticleSortOwner&) = delete;

  void initialize(Owner& owner, const config::PreparedDomain& domain,
                  const lpfx::device::DeviceProperties& properties,
                  Executor executor) {
    lpfx::require(!sorter_ && last_use_event_.empty() && !status_budget_.sealed(),
                  "TPIC particle sort: already initialized");
    lpfx::require(owner.valid() && executor.bound(),
                  "TPIC particle sort: invalid owner or executor");
    lpfx::require(domain.particle_cell_count_with_halo > 0,
                  "TPIC particle sort: invalid cell-bin count");
    bin_count_ = static_cast<std::uint32_t>(domain.particle_cell_count_with_halo);
    lpfx::require(status_budget_.try_set_limit(sizeof(std::uint32_t)) && status_budget_.seal(),
                  "TPIC particle sort: status budget preparation failed");
    lpfx::device::check_device_error<Backend>(
        Event::create(last_use_event_), "TPIC particle sort: last-consumer event creation failed");
    create_sorter(owner, properties, executor);
  }

protected:
  using Event = lpfx::device::DeviceEvent<Backend>;

  ParticleSortOwner() = default;
  ~ParticleSortOwner() = default;

  void create_sorter(Owner& owner,
                     const lpfx::device::DeviceProperties& properties,
                     Executor executor) {
    using Key = lpfx::ParticleAttributeKey<
        typename data::ParticleSchema<Precision>::Cell::name_type>;
    lpfx::PreparedDeviceParticleSortWorkspace<> workspace;
    lpfx::DirectCellBinLayout<std::int32_t> bins;
    if (lpfx::try_prepare_device_particle_sort_workspace(owner.capacity(), bin_count_,
            properties, workspace) != lpfx::ErrorCode::success ||
        !lpfx::try_make_direct_cell_bin_layout(bin_count_, bins))
      throw std::runtime_error("TPIC particle sort: workspace preparation failed");
    auto made = lpfx::make_device_cell_sort<Key>(workspace, bins, owner, executor);
    lpfx::check_error(made, "TPIC particle sort: sorter creation failed");
    sorter_ = std::make_unique<Sorter>(std::move(made).value());
    lpfx::check_error(sorter_->prepare_status_inspection(status_budget_),
                      "TPIC particle sort: status inspection preparation failed");
  }

  lpfx::HostMemoryBudget status_budget_;
  Event last_use_event_;
  std::unique_ptr<Sorter> sorter_;
  // Borrowed per-sort token: inline storage avoids an allocation each step.
  std::optional<Receipt> receipt_;
  std::uint32_t bin_count_{};
};

// Full-owner sorting follows completed migration. The checked receipt owns the
// bin epoch through deposition and the final sorted mover, then retires before
// ordinary migration can compact or replace the particle allocation. LPFx's
// rotating sorter reuses the auxiliary slot already present in TPIC's owner.
// An uninitialized wrapper allocates nothing, including in unsorted mode.
template <class Precision, class Backend>
class FullParticleSort : public ParticleSortOwner<Precision, Backend> {
  using Base = ParticleSortOwner<Precision, Backend>;
  using typename Base::Event;
  using typename Base::Sorter;
  using Base::bin_count_;
  using Base::create_sorter;
  using Base::last_use_event_;
  using Base::receipt_;
  using Base::sorter_;

public:
  using typename Base::Executor;
  using typename Base::Owner;
  using typename Base::Receipt;

  // Cold phase after migration, which can replace the allocation even at
  // unchanged capacity. Rebuilds the sorter only after the previous receipt
  // has retired; reusable() alone is not a completion fence.
  void rebind_to_owner(Owner& owner,
                         const lpfx::device::DeviceProperties& properties,
                         Executor executor) {
    lpfx::require(sorter_ && !receipt_ && !retiring_,
                  "TPIC particle sort: preparation before receipt retirement");
    lpfx::require(owner.valid() && executor.bound() && sorter_->bound_owner() == &owner,
                  "TPIC particle sort: preparation owner or executor changed");
    if (!sorter_->reusable()) {
      sorter_.reset();
      create_sorter(owner, properties, executor);
    }
  }

  // Stream: executor, after rebind_to_owner and completed migration. Sorts
  // the owner and enqueues the status inspection; allocates and waits for
  // nothing. The caller polls progress_complete() before the first bin reader.
  void enqueue(Owner& owner, const config::PreparedDomain& domain, Executor executor) {
    lpfx::require(sorter_ && !receipt_ && !retiring_ && sorter_->reusable(),
                  "TPIC particle sort: unprepared sorter or unretired receipt");
    lpfx::require(executor.bound() && sorter_->bound_owner() == &owner &&
                      domain.particle_cell_count_with_halo > 0 &&
                      static_cast<std::uint32_t>(domain.particle_cell_count_with_halo) ==
                          bin_count_,
                  "TPIC particle sort: owner or cell-bin layout changed");
    auto queued = sorter_->enqueue(executor);
    lpfx::check_error(queued, "TPIC particle sort: enqueue failed");
    receipt_.emplace(std::move(queued).value());
    lpfx::check_error(sorter_->inspect(*receipt_, executor),
                      "TPIC particle sort: status inspection failed");
  }

  [[nodiscard]] lpfx::Status progress_complete() {
    if (!sorter_ || !receipt_ || completed_)
      return lpfx::Status::failure(lpfx::ErrorCode::invalid_state);
    const auto status = sorter_->conclude(*receipt_);
    if (!status.is_pending()) {
      completed_ = true;
      published_ = status.is_success();
    }
    return status;
  }

  // Blocking completion for startup; the time step uses progress_complete().
  void host_wait_complete() {
    lpfx::require(sorter_ && receipt_.has_value() && !completed_,
                  "TPIC particle sort: no pending sort completion");
    const auto status = sorter_->wait(*receipt_);
    completed_ = true;
    published_ = status.is_success();
    lpfx::check_error(status, "TPIC particle sort: invalid cell key or producer failure");
  }

  [[nodiscard]] const Receipt& receipt() const {
    lpfx::require(published_ && receipt_.has_value() && receipt_->valid(),
                  "TPIC particle sort: no successfully completed sorted receipt");
    return *receipt_;
  }

  // last_reader must follow every ordinary reader, including the mover.
  // The sort producer stream joins that event; migration must poll actual
  // retirement before mutating this standalone sorter's owner.
  void retire_after(const Event& last_reader) {
    if (!receipt_) return;
    lpfx::require(completed_ && !last_reader.empty(),
                  "TPIC particle sort: retirement before checked completion");
    lpfx::check_error(sorter_->retire_after(std::move(*receipt_), last_reader),
                      "TPIC particle sort: receipt retirement failed");
    receipt_.reset();
    completed_ = false;
    published_ = false;
    retiring_ = true;
  }

  [[nodiscard]] lpfx::Status progress_retirement() {
    if (!sorter_ || receipt_) return lpfx::Status::failure(lpfx::ErrorCode::invalid_state);
    if (!sorter_->quiescent_for_close()) return lpfx::Status::pending();
    retiring_ = false;
    return lpfx::Status::success();
  }

  // Blocking retirement for startup and teardown; the time step uses
  // retire_after() and progress_retirement().
  void host_wait_retired(Executor executor) {
    if (receipt_) {
      lpfx::require(executor.bound(), "TPIC particle sort: retirement needs a bound executor");
      lpfx::device::check_device_error<Backend>(
          last_use_event_.record(executor), "TPIC particle sort: last-consumer event record failed");
      retire_after(last_use_event_);
    }
    if (retiring_) {
      lpfx::check_error(sorter_->wait_for_retirement(),
                        "TPIC particle sort: retirement completion failed");
      retiring_ = false;
    }
  }

private:
  bool published_{};
  bool completed_{};
  bool retiring_{};
};

// TPIC owns resident sorting, inspection and the numerical prefix receipt.
// Migration exposes a protected prefix and appends the unsorted arrival tail.
// The local action explicitly joins readers, releases growth scratch and
// rebinds the sorter after migration grows the particle allocation.
// The sorting executor and particle owner must outlive this numerical wrapper.
template <class Precision, class Backend>
class PartialParticleSort : public ParticleSortOwner<Precision, Backend> {
  using Base = ParticleSortOwner<Precision, Backend>;
  using Schema = data::ParticleSchema<Precision>;
  using typename Base::Event;
  using Base::last_use_event_;
  using Base::receipt_;
  using Base::sorter_;

public:
  using typename Base::Executor;
  using typename Base::Owner;
  using typename Base::Receipt;
  using typename Base::Sorter;

  // Stream: executor, which the caller has made wait for *range.ready. Sorts
  // the resident prefix and enqueues the status inspection; the caller polls
  // progress_complete() before the first bin reader.
  void enqueue(lpfx::ResidentReorderRange<typename Schema::type, Backend> range,
               Executor executor) {
    lpfx::require(!receipt_, "TPIC particle sort: previous receipt is still live");
    auto made = sorter().enqueue_ready(std::move(range), executor);
    lpfx::check_error(made, "TPIC particle sort: submission failed");
    receipt_.emplace(std::move(made).value());
    completed_ = false;
    lpfx::check_error(sorter().inspect(*receipt_, executor),
                      "TPIC particle sort: inspection submission failed");
  }
  [[nodiscard]] lpfx::Status progress_complete() {
    if (completed_) return lpfx::Status::success();
    if (!receipt_) return lpfx::Status::failure(lpfx::ErrorCode::invalid_state);
    auto status = sorter().conclude(*receipt_);
    if (status.is_success()) completed_ = true;
    return status;
  }
  void host_wait_complete() {
    lpfx::require(receipt_.has_value(), "TPIC particle sort: no pending submission");
    if (!completed_)
      lpfx::check_error(sorter().wait(*receipt_), "TPIC particle sort: invalid sort");
    completed_ = true;
  }
  [[nodiscard]] auto resident_read() const {
    lpfx::require(completed_ && receipt_.has_value(),
                  "TPIC particle sort: incomplete resident writer");
    return sorter().acquire_resident_read(*receipt_);
  }
  [[nodiscard]] const Receipt& receipt() const {
    lpfx::require(completed_ && receipt_ && receipt_->valid(),
                  "TPIC particle sort: invalid receipt");
    return *receipt_;
  }
  void release_growth_scratch() {
    lpfx::require(completed_ && sorter().release_growth_scratch() == lpfx::ErrorCode::success,
                  "TPIC particle sort: growth scratch release failed");
  }
  void rebind_after_growth(Owner& owner, const lpfx::device::DeviceProperties& properties) {
    lpfx::require(receipt_.has_value() &&
                      sorter().rebind_after_growth(owner, properties, &*receipt_) ==
                          lpfx::ErrorCode::success,
                  "TPIC particle sort: growth rebind failed");
  }
  void retire_after(const Event& last_reader) {
    if (!receipt_) return;
    lpfx::require(completed_ && sorter().retire_after(std::move(*receipt_), last_reader) ==
                                    lpfx::ErrorCode::success,
                  "TPIC particle sort: receipt retirement failed");
    receipt_.reset();
    completed_ = false;
  }
  void order_reuse(Executor executor) { sorter().order_reuse(executor); }
  void host_wait_retired(Executor executor) {
    if (receipt_) {
      lpfx::device::check_device_error<Backend>(last_use_event_.record(executor),
                                                "TPIC particle sort: last use record failed");
      retire_after(last_use_event_);
    }
    lpfx::check_error(sorter().wait_for_retirement(), "TPIC particle sort: retirement failed");
  }
  // Blocking startup has an explicitly empty early reader set.
  const Event& close_empty_readers(Executor executor) {
    lpfx::require(completed_, "TPIC particle sort: reader closure before writer completion");
    lpfx::device::check_device_error<Backend>(last_use_event_.record(executor),
                                              "TPIC particle sort: reader closure record failed");
    return last_use_event_;
  }

private:
  [[nodiscard]] Sorter& sorter() const {
    lpfx::require(sorter_ != nullptr, "TPIC particle sort: uninitialized sorter");
    return *sorter_;
  }
  bool completed_{};
};

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_PARTICLES_PARTICLE_SORT_DEVICE_HPP
