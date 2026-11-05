#include <lpfx/core/error_reporting.hpp>
#include <tpic/simulation/reporting/run_reporter.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace tpic::simulation {

void RunReporter::begin() {
  if (!root_) return;
  std::printf("\n============================================================\n"
              "  ECSIM Initialization complete\n"
              "  Starting ECSIM Main Simulation Loop\n"
              "============================================================\n");
}
void RunReporter::cycle(int cycle, double time) {
  if (!root_) return;
  std::printf("\n ======= ECSIM Cycle %d  |  t = %.6e =======\n", cycle, time);
}
void RunReporter::finish_cycle(int cycle, bool captures, int iterations, double residual,
                               double elapsed_ms) {
  if (!root_) return;
  cycle_times_.add(elapsed_ms);
  if (cycle > first_cycle_)
    (captures ? capture_times_ : ordinary_times_).add(elapsed_ms);
  std::printf("  Maxwell GMRES : converged in %d iterations; residual = %.6e\n"
              "  Cycle time    : %.3f ms (rank 0 wall time)\n",
              iterations, residual, elapsed_ms);
}
void RunReporter::bootstrap(double seconds) {
  if (!root_) return;
  std::printf(
      "  Bootstrap host admission time: %.6f s (device completion included in total run)\n",
      seconds);
}
void RunReporter::final_state(const data::ScalarDiagnosticSample& sample, double initial_energy) {
  if (!root_) return;
  std::printf("\n ======= ECSIM Final State  |  cycle = %d, t = %.6e =======\n",
              sample.cycle, sample.time);
  energy(sample, initial_energy);
}

void RunReporter::energy(const data::ScalarDiagnosticSample &sample, double initial_energy) {
  if (!root_) return;
  const double electric = sample.values[0], magnetic = sample.values[1];
  double kinetic = 0;
  for (std::size_t species = 0; species < sample.species_count; ++species)
    kinetic += sample.values[data::field_scalar_count +
                             data::species_scalar_count * species];
  const double total = electric + magnetic + kinetic;
  lpfx::require(std::isfinite(kinetic) && std::isfinite(total),
                "TPIC reporter: nonfinite system energy in console diagnostics");
  const double change = total - initial_energy;
  std::printf("  Field energy  : electric = %.12e  magnetic = %.12e\n"
              "  Total energy  : system   = %.12e  kinetic  = %.12e\n",
              electric, magnetic, total, kinetic);
  if (initial_energy != 0)
    std::printf("  Energy change : dE/E0 = %+.6e  dE = %+.6e\n",
                change / initial_energy, change);
  else
    std::printf(
        "  Energy change : dE/E0 = n/a (zero initial energy)  dE = %+.6e\n",
        change);
  std::fflush(stdout);
}

void RunReporter::gauss(const data::GaussSample &sample) {
  if (!root_) return;
  using Index = data::GaussScalarIndex;
  std::printf("  Gauss residual: cycle %d t = %.6e  rms = %.6e  drift = %.6e  wall = %.6e  charge = %.6e\n",
              sample.cycle, sample.time, sample.rms[Index::residual], sample.rms[Index::drift],
              sample.rms[Index::wall], sample.rms[Index::charge]);
  std::fflush(stdout);
}

void RunReporter::loop(std::array<double, 2> seconds, int cycles) {
  if (!root_) return;
  std::printf("\n=========================================\n"
              "ECSIM Main Loop Complete\n"
              "  Total cycles: %d\n"
              "  Total time: %.9f s (maximum rank)\n"
              "  Steady-state cycles: %d\n"
              "  Steady-state time: %.9f s (maximum rank; first cycle excluded)\n",
              cycles, seconds[0], std::max(cycles - 1, 0), seconds[1]);
  if (cycles > 0)
    std::printf("  Average per cycle: %.3f ms (loop time / cycles)\n"
                "  Std. dev. per cycle: %.3f ms (rank 0 samples)\n",
                seconds[0] * 1000.0 / cycles, cycle_times_.stddev_ms());
  const auto print = [](const char *label, const RunningTimingStats &stats) {
    std::printf("  %s: %llu cycles; avg=%.3f std=%.3f min=%.3f max=%.3f ms\n",
                label, static_cast<unsigned long long>(stats.count()),
                stats.mean_ms(), stats.stddev_ms(), stats.min_ms(),
                stats.max_ms());
  };
  print("Ordinary after warm-up", ordinary_times_);
  print("Output after warm-up", capture_times_);
  std::printf("  Loop totals include cycle reporting; final-state "
              "diagnostics/output are excluded.\n"
              "=========================================\n");
}

} // namespace tpic::simulation
