#ifndef TPIC_SIMULATION_TIMING_TIMING_REGIONS_HPP
#define TPIC_SIMULATION_TIMING_TIMING_REGIONS_HPP

#include <tpic/config/limits.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace tpic::simulation {

// Welford's recurrence keeps the variance stable for long production runs.
class RunningTimingStats {
public:
  void add(double milliseconds) noexcept {
    ++count_;
    total_ += milliseconds;
    minimum_ = std::min(minimum_, milliseconds);
    maximum_ = std::max(maximum_, milliseconds);
    const double delta = milliseconds - mean_;
    mean_ += delta / static_cast<double>(count_);
    squared_deviation_ += delta * (milliseconds - mean_);
  }
  // Merge published worker statistics without reconstructing individual
  // samples. The parallel Welford formula preserves the sample variance.
  void merge(const RunningTimingStats& other) noexcept {
    if (!other.count_) return;
    if (!count_) {
      *this = other;
      return;
    }
    const auto combined = count_ + other.count_;
    const double delta = other.mean_ - mean_;
    squared_deviation_ += other.squared_deviation_ + delta * delta *
        (static_cast<double>(count_) * static_cast<double>(other.count_) /
         static_cast<double>(combined));
    mean_ += delta * (static_cast<double>(other.count_) / static_cast<double>(combined));
    total_ += other.total_;
    minimum_ = std::min(minimum_, other.minimum_);
    maximum_ = std::max(maximum_, other.maximum_);
    count_ = combined;
  }
  [[nodiscard]] std::uint64_t count() const noexcept { return count_; }
  [[nodiscard]] double total_ms() const noexcept { return total_; }
  [[nodiscard]] double min_ms() const noexcept { return count_ ? minimum_ : 0.; }
  [[nodiscard]] double max_ms() const noexcept { return count_ ? maximum_ : 0.; }
  [[nodiscard]] double mean_ms() const noexcept { return mean_; }
  [[nodiscard]] double stddev_ms() const noexcept {
    return count_ > 1 ? std::sqrt(std::max(0., squared_deviation_) /
                                static_cast<double>(count_ - 1)) : 0.;
  }
private:
  std::uint64_t count_{};
  double total_{};
  double minimum_{std::numeric_limits<double>::infinity()};
  double maximum_{-std::numeric_limits<double>::infinity()};
  double mean_{};
  double squared_deviation_{};
};

// The declaration order is the print order of every timing row.
enum class TimingRegion : std::size_t {
  CycleTotal, ClearMoments, CurrentDeposition, MassDeposition, CurrentAssembly, MassAssembly,
  ParticleDiagnostics, DiagnosticsCollect, FieldDiagnostics, GaussDiagnostics, GaussCollect,
  FieldSolve, RhsFields, FinishRhs, SolveMaxwell, FieldAdvance, ParticleField,
  Mover, ParticleMigration, ParticleSort, FieldCommit,
  OutputMoments, OutputFields, OutputDiagnostics, OutputWriterWait, OutputCapturePublish, Checkpoint, count
};
// A deposition interval is recorded per particle range: resident, tail or the whole owner.
enum class TimingPhase : std::size_t { full, resident, tail, count };
// The first cycle of a run is the warm-up: it belongs to "all" only.
enum class TimingGroup : std::size_t { all, ordinary, output, count };

inline constexpr std::size_t timing_region_count = static_cast<std::size_t>(TimingRegion::count);
inline constexpr std::size_t timing_phase_count = static_cast<std::size_t>(TimingPhase::count);
inline constexpr std::size_t timing_group_count = static_cast<std::size_t>(TimingGroup::count);
inline constexpr std::size_t timing_instance_count = config::max_real_species;

// Device event pairs of a region: one per species, or a fixed count (0 for a host-only region).
struct TimingRegionDescriptor {
  const char* name;
  bool per_species;
  std::size_t device_instances;
  bool phased;
};

inline constexpr std::array<TimingRegionDescriptor, timing_region_count> timing_regions{{
    {"CycleTotal", false, 1, false},
    {"ClearMoments", false, 2, false},
    {"CurrentDeposition", true, 0, true},
    {"MassDeposition", true, 0, true},
    {"CurrentAssembly", false, 1, false},
    {"MassAssembly", false, 1, false},
    {"ParticleDiagnostics", true, 0, false},
    {"DiagnosticsCollect", false, 1, false},
    {"FieldDiagnostics", false, 1, false},
    {"GaussDiagnostics", false, 1, false},
    {"GaussCollect", false, 0, false},
    {"FieldSolve", false, 1, false},
    {"RhsFields", false, 1, false},
    {"FinishRhs", false, 1, false},
    {"SolveMaxwell", false, 1, false},
    {"FieldAdvance", false, 1, false},
    {"ParticleField", false, 1, false},
    {"Mover", true, 0, false},
    {"ParticleMigration", true, 0, false},
    {"ParticleSort", true, 0, false},
    {"FieldCommit", false, 1, false},
    {"OutputMoments", true, 0, false},
    {"OutputFields", false, 0, false},
    {"OutputDiagnostics", false, 0, false},
    {"OutputWriterWait", false, 0, false},
    {"OutputCapturePublish", false, 0, false},
    {"Checkpoint", false, 0, false},
}};

[[nodiscard]] constexpr std::size_t timing_index(TimingRegion region) noexcept {
  return static_cast<std::size_t>(region);
}
[[nodiscard]] constexpr std::size_t timing_index(TimingGroup group) noexcept {
  return static_cast<std::size_t>(group);
}

[[nodiscard]] constexpr std::size_t device_instance_capacity(
    TimingRegion region, std::size_t species_count = timing_instance_count) noexcept {
  const auto& descriptor = timing_regions[timing_index(region)];
  return descriptor.per_species ? species_count : descriptor.device_instances;
}

[[nodiscard]] constexpr std::size_t device_phase_capacity(TimingRegion region) noexcept {
  return timing_regions[timing_index(region)].phased ? timing_phase_count : 1;
}

// Device event pairs that one timer prepares for species_count species.
[[nodiscard]] constexpr std::size_t timing_event_pairs(std::size_t species_count) noexcept {
  std::size_t pairs = 0;
  for (std::size_t region = 0; region != timing_region_count; ++region)
    pairs += device_instance_capacity(static_cast<TimingRegion>(region), species_count) *
             device_phase_capacity(static_cast<TimingRegion>(region));
  return pairs;
}

// Host and device statistics of one timer: the current cycle and the three groups
// across cycles. Recording is inline; only the table printing is out of line.
class TimingTable {
public:
  void begin_cycle(int cycle, bool snapshot, bool warmup) noexcept {
    cycle_ = cycle;
    snapshot_ = snapshot;
    warmup_ = warmup;
    cycle_host_ = {};
    cycle_device_ = {};
  }
  void add_host_time(TimingRegion region, double milliseconds) noexcept {
    const auto r = timing_index(region);
    cycle_host_[r].add(milliseconds);
    update_groups(host_, r, [=](RunningTimingStats& stats) { stats.add(milliseconds); });
  }
  void add_device_time(std::size_t region, double milliseconds) noexcept {
    cycle_device_[region].add(milliseconds);
    update_groups(device_, region, [=](RunningTimingStats& stats) { stats.add(milliseconds); });
  }
  void count_cycle() noexcept {
    ++cycle_counts_[timing_index(TimingGroup::all)];
    if (!warmup_) ++cycle_counts_[timing_index(cycle_group())];
  }
  // Merges the closed cycle of a worker that timed the same cycle.
  void merge_cycle(const TimingTable& worker) noexcept {
    for (std::size_t region = 0; region < timing_region_count; ++region) {
      cycle_host_[region].merge(worker.cycle_host_[region]);
      cycle_device_[region].merge(worker.cycle_device_[region]);
      update_groups(host_, region, [&](RunningTimingStats& stats) { stats.merge(worker.cycle_host_[region]); });
      update_groups(device_, region, [&](RunningTimingStats& stats) { stats.merge(worker.cycle_device_[region]); });
    }
  }
  [[nodiscard]] bool same_cycle(const TimingTable& other) const noexcept {
    return cycle_ == other.cycle_ && snapshot_ == other.snapshot_;
  }

  [[nodiscard]] const RunningTimingStats& stats(TimingRegion region,
      TimingGroup group = TimingGroup::all, bool device = false) const noexcept {
    return (device ? device_ : host_)[timing_index(group)][timing_index(region)];
  }
  [[nodiscard]] const RunningTimingStats& cycle_stats(TimingRegion region, bool device = false) const noexcept {
    return (device ? cycle_device_ : cycle_host_)[timing_index(region)];
  }

  void print_cycle(std::FILE* file) const;
  void print_summary(std::FILE* file) const;

private:
  using RegionStats = std::array<RunningTimingStats, timing_region_count>;
  using GroupStats = std::array<RegionStats, timing_group_count>;

  [[nodiscard]] TimingGroup cycle_group() const noexcept {
    return snapshot_ ? TimingGroup::output : TimingGroup::ordinary;
  }
  template <class Update>
  void update_groups(GroupStats& groups, std::size_t region, Update update) noexcept {
    update(groups[timing_index(TimingGroup::all)][region]);
    if (!warmup_) update(groups[timing_index(cycle_group())][region]);
  }

  int cycle_{};
  bool snapshot_{}, warmup_{};
  GroupStats host_{};
  GroupStats device_{};
  RegionStats cycle_host_{};
  RegionStats cycle_device_{};
  std::array<std::uint64_t, timing_group_count> cycle_counts_{};
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_TIMING_TIMING_REGIONS_HPP
