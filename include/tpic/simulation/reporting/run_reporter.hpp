#ifndef TPIC_SIMULATION_REPORTING_RUN_REPORTER_HPP
#define TPIC_SIMULATION_REPORTING_RUN_REPORTER_HPP

#include <tpic/data/output/gauss_sample.hpp>
#include <tpic/data/output/scalar_sample.hpp>
#include <tpic/simulation/timing/timing_regions.hpp>

#include <array>

namespace tpic::simulation {

// Console lines of a run, printed by rank 0 only. Presentation and the
// rank-0 loop statistics: the driver runs every collective before the call.
class RunReporter {
public:
  explicit RunReporter(bool root, int first_cycle = 0) noexcept : root_(root), first_cycle_(first_cycle) {}
  void begin();
  void cycle(int cycle, double time);
  // Records the cycle time (the first cycle is the warm-up) and prints the solve and cycle-time lines.
  void finish_cycle(int cycle, bool captures, int iterations, double residual, double elapsed_ms);
  void bootstrap(double seconds);
  // initial_energy is the total energy of the run's first state.
  void final_state(const data::ScalarDiagnosticSample& sample, double initial_energy);
  void energy(const data::ScalarDiagnosticSample &sample, double initial_energy);
  void gauss(const data::GaussSample &sample);
  void loop(std::array<double, 2> maximum_rank_seconds, int cycles);

private:
  bool root_;
  int first_cycle_;
  RunningTimingStats cycle_times_, ordinary_times_, capture_times_;
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_REPORTING_RUN_REPORTER_HPP
