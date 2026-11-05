#ifndef TPIC_SIMULATION_TIMING_STAGE_TIMING_HPP
#define TPIC_SIMULATION_TIMING_STAGE_TIMING_HPP

#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/handles.hpp>
#include <tpic/simulation/timing/timing_regions.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <new>
#include <stdexcept>

namespace tpic::simulation {

// Host scopes are wall timers that read the host clock without synchronizing
// the device. Device scopes record stream events without waiting at their
// boundaries. Nested scopes include their
// children, and concurrent streams overlap: neither table is additive. All event
// pairs are prepared once, and reused only after their closed cycle is gathered.
// Each worker owns one instance. Signal the done fence after closing the records and before the main
// thread queries or merges them; this class intentionally does not share mutable scope
// state between workers. Profiling owns its own timing-enabled events (LPFx's
// dependency events disable timing), leaving LPFx's synchronization intact.
template <class Backend, class Clock = std::chrono::steady_clock>
class StageTiming {
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Executor = lpfx::device::Executor<Backend>;
  using Event = typename Traits::event_type;
  using Error = typename Traits::error_type;

  struct EventSlot {
    Event start_event{Traits::null_event};
    Event stop_event{Traits::null_event};
    bool used{};
  };
  // idle -> active (begin_cycle) -> pending (close_cycle) -> collected (progress_collect_cycle).
  enum class CycleState { idle, active, pending, collected };

public:
  explicit StageTiming(bool enabled = false,
                       std::size_t species_count = timing_instance_count, int first_cycle = 0)
      : enabled_(enabled), species_count_(species_count), first_cycle_(first_cycle) {
    if (!enabled_) return;
    if (species_count == 0 || species_count > timing_instance_count)
      throw std::logic_error("TPIC timing: unsupported species count");
    try {
      for (std::size_t region = 0; region < timing_region_count; ++region) {
        const auto r = static_cast<TimingRegion>(region);
        for (std::size_t instance = 0; instance < simulation::device_instance_capacity(r, species_count_);
             ++instance) {
          for (std::size_t phase = 0; phase < simulation::device_phase_capacity(r); ++phase) {
            auto& slot = slots_[slot_index(region, instance, phase)];
            lpfx::device::check_device_error<Backend>(Traits::create_timing_event(slot.start_event),
                                                      "TPIC timing: creating start event");
            lpfx::device::check_device_error<Backend>(Traits::create_timing_event(slot.stop_event),
                                                      "TPIC timing: creating stop event");
          }
        }
      }
    } catch (...) {
      destroy_events();
      throw;
    }
  }

  ~StageTiming() { destroy_events(); }
  StageTiming(const StageTiming&) = delete;
  StageTiming& operator=(const StageTiming&) = delete;

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  class Scope {
  public:
    ~Scope() noexcept {
      if (!owner_) return;
      const auto stop = Clock::now();
      owner_->table_.add_host_time(region_, std::chrono::duration<double, std::milli>(stop - start_).count());
      if (slot_) {
        const Error error = Traits::record_event(slot_->stop_event, stream_);
        if (!Traits::succeeded(error)) owner_->deferred_error_ = error;
      }
      --owner_->open_scopes_;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

  private:
    friend class StageTiming;
    Scope(StageTiming* owner, TimingRegion region)
        : owner_(owner), region_(region) {
      if (!owner_) return;
      ++owner_->open_scopes_;
      start_ = Clock::now();
    }
    Scope(StageTiming* owner, TimingRegion region, Executor executor, std::size_t instance, TimingPhase phase)
        : owner_(owner), region_(region), stream_(executor.stream()) {
      if (!owner_) return;
      if (instance >= simulation::device_instance_capacity(region, owner_->species_count_))
        throw std::logic_error("TPIC timing: region instance exceeds persistent event slots");
      if (!executor.bound())
        throw std::logic_error("TPIC timing: device scope requires an admitted executor");
      if (static_cast<std::size_t>(phase) >= simulation::device_phase_capacity(region))
        throw std::logic_error("TPIC timing: region phase exceeds persistent event slots");
      slot_ = &owner_->slots_[slot_index(timing_index(region), instance, static_cast<std::size_t>(phase))];
      if (slot_->used)
        throw std::logic_error("TPIC timing: device region instance reused before its cycle was collected");
      lpfx::device::check_device_error<Backend>(Traits::record_event(slot_->start_event, stream_),
                                                "TPIC timing: recording start event");
      slot_->used = true;
      ++owner_->open_scopes_;
      start_ = Clock::now();
    }
    StageTiming* owner_{};
    TimingRegion region_{};
    typename Clock::time_point start_{};
    typename Traits::stream_type stream_{Traits::null_stream};
    EventSlot* slot_{};
  };

  [[nodiscard]] Scope scope(TimingRegion region) {
    return Scope(recording() ? this : nullptr, region);
  }
  [[nodiscard]] Scope scope(TimingRegion region, Executor executor, std::size_t instance = 0,
                           TimingPhase phase = TimingPhase::full) {
    return Scope(recording() ? this : nullptr, region, executor, instance, phase);
  }

  void begin_cycle(int cycle, bool snapshot) {
    if (!enabled_) return;
    if (state_ == CycleState::active || state_ == CycleState::pending || open_scopes_)
      throw std::logic_error("TPIC timing: previous cycle or scope is still active");
    table_.begin_cycle(cycle, snapshot, cycle == first_cycle_);
    deferred_error_ = Traits::success_error;
    for (auto& slot : slots_) slot.used = false;
    state_ = CycleState::active;
    merged_ = false;
  }

  // The worker closes before signalling its done fence. No device operation
  // occurs here, so main can collect later without extending the worker's task.
  void close_cycle() {
    if (!enabled_) return;
    if (state_ != CycleState::active || open_scopes_)
      throw std::logic_error("TPIC timing: close_cycle requires a cycle with all scopes closed");
    lpfx::device::check_device_error<Backend>(deferred_error_, "TPIC timing: recording stop event");
    state_ = CycleState::pending;
  }

  // All-or-nothing collection: an unfinished stop event leaves every event
  // slot and sample unchanged. Traits::event_elapsed_time is reached only after
  // all stop events queried complete, never through profile-only waiting.
  [[nodiscard]] bool progress_collect_cycle() {
    if (!enabled_) return true;
    if (state_ != CycleState::pending || open_scopes_)
      throw std::logic_error("TPIC timing: collection requires a closed pending cycle");
    for (const auto& slot : slots_) {
      if (!slot.used) continue;
      const auto error = Traits::query_event(slot.stop_event);
      if (Traits::not_ready(error)) return false;
      lpfx::device::check_device_error<Backend>(error, "TPIC timing: querying stop event");
    }
    collect_cycle();
    return true;
  }

  // Call on main only after the worker's done-fence signal. Closed worker
  // events must already be collected; main may still have its own host scope
  // open. Merge exactly one epoch, without counting it as another main cycle.
  void merge_cycle(StageTiming& worker) {
    if (!enabled_ || !worker.enabled_) return;
    if (&worker == this || worker.state_ != CycleState::collected || worker.open_scopes_ ||
        worker.merged_ || (state_ != CycleState::active && state_ != CycleState::collected) ||
        !table_.same_cycle(worker.table_))
      throw std::logic_error("TPIC timing: invalid worker cycle merge");
    table_.merge_cycle(worker.table_);
    worker.merged_ = true;
  }

  [[nodiscard]] const TimingTable& table() const noexcept { return table_; }

  void print_cycle(std::FILE* file = stdout) const {
    if (enabled_) table_.print_cycle(file);
  }
  void print_summary(std::FILE* file = stdout) const {
    if (enabled_) table_.print_summary(file);
  }

private:
  static constexpr std::size_t slot_index(std::size_t region, std::size_t instance,
                                          std::size_t phase) noexcept {
    return (region * timing_instance_count + instance) * timing_phase_count + phase;
  }
  [[nodiscard]] bool recording() const noexcept { return enabled_ && state_ == CycleState::active; }

  void collect_cycle() {
    for (std::size_t region = 0; region < timing_region_count; ++region) {
      for (std::size_t instance = 0; instance < timing_instance_count; ++instance) {
        for (std::size_t phase = 0; phase < timing_phase_count; ++phase) {
          auto& slot = slots_[slot_index(region, instance, phase)];
          if (!slot.used) continue;
          float milliseconds{};
          lpfx::device::check_device_error<Backend>(
              Traits::event_elapsed_time(milliseconds, slot.start_event, slot.stop_event),
              "TPIC timing: reading elapsed event time");
          table_.add_device_time(region, double{milliseconds});
          slot.used = false;
        }
      }
    }
    table_.count_cycle();
    state_ = CycleState::collected;
  }
  void destroy_events() noexcept {
    if (!enabled_) return;
    for (auto& slot : slots_) {
      if (slot.start_event != Traits::null_event) (void)Traits::destroy_event(slot.start_event);
      if (slot.stop_event != Traits::null_event) (void)Traits::destroy_event(slot.stop_event);
    }
  }

  bool enabled_{};
  CycleState state_{CycleState::idle};
  bool merged_{};
  std::size_t species_count_{timing_instance_count};
  int first_cycle_{}; // the warm-up cycle
  std::size_t open_scopes_{};
  Error deferred_error_{Traits::success_error};
  std::array<EventSlot, timing_region_count * timing_instance_count * timing_phase_count> slots_{};
  TimingTable table_;
};

// A nonmovable timing scope held in prepared inline storage. Its event pair
// spans repeated progress calls without allocating at epoch admission.
template <class Backend>
class ProgressScope {
  using Timing = StageTiming<Backend>;
  using Scope = typename Timing::Scope;

public:
  ProgressScope() = default;
  ProgressScope(const ProgressScope&) = delete;
  ProgressScope& operator=(const ProgressScope&) = delete;
  ~ProgressScope() { close(); }
  void begin(Timing& timer, TimingRegion region,
             lpfx::device::Executor<Backend> executor, unsigned instance = 0) {
    lpfx::require(!live_, "TPIC timing: duplicate progress timing scope");
    ::new (static_cast<void*>(storage_)) Scope(timer.scope(region, executor, instance));
    live_ = true;
  }
  void close() noexcept {
    if (!live_) return;
    std::launder(reinterpret_cast<Scope*>(storage_))->~Scope();
    live_ = false;
  }

private:
  alignas(Scope) std::byte storage_[sizeof(Scope)];
  bool live_{};
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_TIMING_STAGE_TIMING_HPP
