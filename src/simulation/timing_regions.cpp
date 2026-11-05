#include <tpic/simulation/timing/timing_regions.hpp>

namespace tpic::simulation {
namespace {

using RegionStats = std::array<RunningTimingStats, timing_region_count>;

void print_cycle_values(std::FILE* file, const RegionStats& regions) {
  for (std::size_t region = 0; region < timing_region_count; ++region) {
    const auto& stats = regions[region];
    if (stats.count())
      std::fprintf(file, " %s=%.3f", timing_regions[region].name, stats.total_ms());
  }
  std::fprintf(file, "\n");
}

void print_table(std::FILE* file, const char* name, const RegionStats& regions) {
  bool present = false;
  for (const auto& stats : regions) present |= stats.count() != 0;
  if (!present) return;
  std::fprintf(file, "%s [ms]\n", name);
  std::fprintf(file, "  %-29s %9s %13s %12s %12s %12s %12s\n", "Region", "Calls",
               "Total", "Avg", "Min", "Max", "StdDev");
  for (std::size_t region = 0; region < timing_region_count; ++region) {
    const auto& stats = regions[region];
    if (!stats.count()) continue;
    std::fprintf(file, "  %-29s %9llu %13.3f %12.3f %12.3f %12.3f %12.3f\n",
                 timing_regions[region].name, static_cast<unsigned long long>(stats.count()),
                 stats.total_ms(), stats.mean_ms(), stats.min_ms(), stats.max_ms(), stats.stddev_ms());
  }
}

} // namespace

void TimingTable::print_cycle(std::FILE* file) const {
  std::fprintf(file, "  Detailed timing cycle %d (%s) host wall ms:", cycle_,
               warmup_ ? "warmup" : snapshot_ ? "output" : "ordinary");
  print_cycle_values(file, cycle_host_);
  std::fprintf(file, "  Device stream elapsed ms (inclusive; overlapping):");
  print_cycle_values(file, cycle_device_);
}

void TimingTable::print_summary(std::FILE* file) const {
  std::fprintf(file, "\nTPIC detailed timing summary (rank-local)\n");
  std::fprintf(file, "Host wall scopes read the host clock without device synchronization and include nested scopes.\n");
  std::fprintf(file, "Device stream elapsed includes waits and concurrent activity; nested/overlapping rows are not additive.\n");
  std::fprintf(file, "Ordinary/output groups exclude the first cycle; StdDev is the sample standard deviation.\n");
  constexpr std::array<const char*, timing_group_count> groups{"All cycles", "Ordinary cycles (after the first)",
                                                               "Output cycles (after the first)"};
  for (std::size_t group = 0; group < timing_group_count; ++group) {
    if (!cycle_counts_[group]) continue;
    std::fprintf(file, "\n%s: %llu cycles\n", groups[group],
                 static_cast<unsigned long long>(cycle_counts_[group]));
    print_table(file, "Host wall", host_[group]);
    print_table(file, "Device stream elapsed", device_[group]);
  }
}

} // namespace tpic::simulation
