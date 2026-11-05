#include <lpfx/backend/traits.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace lpfx::device {
struct TimingTestBackend {};
template <> struct BackendTraits<TimingTestBackend> {
  using stream_type = int;
  using event_type = int;
  using error_type = int;
  static constexpr int null_stream = 0;
  static constexpr int null_event = 0;
  static constexpr int success_error = 0;
  static constexpr bool succeeded(int error) noexcept { return error == success_error; }
  // No diagnostic strings: errors report their numeric code only.
  static constexpr const char* native_error_name(int) noexcept { return nullptr; }
  static constexpr const char* native_error_string(int) noexcept { return nullptr; }

  // Fake timing events: recorded stream times stand in for device timestamps.
  inline static int creations{}, destructions{}, records{}, elapsed_reads{}, queries{};
  inline static int fail_creation{}, fail_record{};
  inline static bool ready{true};
  inline static std::array<float, 4096> timestamps{};
  inline static std::array<float, 3> stream_time{};
  static int create_timing_event(int& event) noexcept {
    if (fail_creation && creations + 1 == fail_creation) return 1;
    event = ++creations;
    return 0;
  }
  static int destroy_event(int) noexcept { ++destructions; return 0; }
  static int record_event(int event, int stream) noexcept {
    ++records;
    if (fail_record && records == fail_record) return 1;
    timestamps[static_cast<std::size_t>(event)] = stream_time[static_cast<std::size_t>(stream)];
    return 0;
  }
  static int query_event(int) noexcept { ++queries; return ready ? 0 : 2; }
  static bool not_ready(int error) noexcept { return error == 2; }
  static int event_elapsed_time(float& milliseconds, int start, int stop) noexcept {
    ++elapsed_reads;
    milliseconds = timestamps[static_cast<std::size_t>(stop)] -
                   timestamps[static_cast<std::size_t>(start)];
    return 0;
  }
};
} // namespace lpfx::device

#include <tpic/simulation/timing/stage_timing.hpp>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, const char* message) {
  require(std::abs(actual - expected) < 1.e-9, message);
}
template <class Timer> void finish_cycle(Timer& timer) {
  timer.close_cycle();
  require(timer.progress_collect_cycle(), "closed cycle with completed events was not collected");
}
template <class Function> void require_throws(Function function, const char* message) {
  bool threw = false;
  try { function(); } catch (const std::exception&) { threw = true; }
  require(threw, message);
}

struct TestClock {
  using duration = std::chrono::milliseconds;
  using time_point = std::chrono::time_point<TestClock, duration>;
  inline static int reads{};
  inline static long long milliseconds{};
  static time_point now() noexcept {
    ++reads;
    return time_point(duration(milliseconds));
  }
};

// The BackendTraits specialization above supplies the fake event functions.
struct TestEvents : lpfx::device::BackendTraits<lpfx::device::TimingTestBackend> {
  static void reset() {
    creations = destructions = records = elapsed_reads = queries = 0;
    fail_creation = fail_record = 0;
    ready = true;
    timestamps.fill(0);
    stream_time.fill(0);
    TestClock::reads = 0;
    TestClock::milliseconds = 0;
  }
};

using Timing = tpic::simulation::StageTiming<lpfx::device::TimingTestBackend, TestClock>;
using tpic::simulation::TimingGroup;
using tpic::simulation::TimingPhase;
using tpic::simulation::TimingRegion;
using Executor = lpfx::device::Executor<lpfx::device::TimingTestBackend>;

void statistics() {
  tpic::simulation::RunningTimingStats stats;
  require(stats.count() == 0, "empty statistics count");
  near(stats.min_ms(), 0., "empty minimum");
  near(stats.max_ms(), 0., "empty maximum");
  near(stats.stddev_ms(), 0., "empty deviation");
  stats.add(2.);
  near(stats.stddev_ms(), 0., "one-sample deviation");
  stats.add(4.);
  stats.add(6.);
  require(stats.count() == 3, "sample count");
  near(stats.total_ms(), 12., "sample total");
  near(stats.mean_ms(), 4., "sample mean");
  near(stats.min_ms(), 2., "sample minimum");
  near(stats.max_ms(), 6., "sample maximum");
  near(stats.stddev_ms(), 2., "sample standard deviation must use N-1");

  tpic::simulation::RunningTimingStats large;
  large.add(1.e12 + 2.);
  large.add(1.e12 + 4.);
  large.add(1.e12 + 6.);
  near(large.stddev_ms(), 2., "large values lose sample variance");

  tpic::simulation::RunningTimingStats left, right;
  left.add(2.);
  right.add(4.);
  right.add(6.);
  left.merge(right);
  require(left.count() == stats.count(), "merged sample count");
  near(left.total_ms(), stats.total_ms(), "merged total");
  near(left.mean_ms(), stats.mean_ms(), "merged mean");
  near(left.stddev_ms(), stats.stddev_ms(), "merged variance");
  near(left.min_ms(), stats.min_ms(), "merged minimum");
  near(left.max_ms(), stats.max_ms(), "merged maximum");
}

void disabled_and_inactive() {
  TestEvents::reset();
  {
    Timing timer(false);
    timer.begin_cycle(0, true);
    { auto scope = timer.scope(TimingRegion::FieldSolve); }
    { auto scope = timer.scope(TimingRegion::CurrentDeposition, Executor(1), 100); }
    finish_cycle(timer);
    timer.print_cycle(nullptr);
    timer.print_summary(nullptr);
    require(TestClock::reads == 0, "disabled timer queried the clock");
  }
  require(TestEvents::creations == 0 && TestEvents::records == 0 &&
          TestEvents::queries == 0 && TestEvents::destructions == 0,
          "disabled timer issued a backend operation");

  TestEvents::reset();
  {
    Timing timer(true);
    const auto creations = TestEvents::creations;
    const std::size_t pair_count = tpic::simulation::timing_event_pairs(tpic::simulation::timing_instance_count);
    require(creations == 2 * static_cast<int>(pair_count),
            "event pairs were not prepared persistently");
    { auto scope = timer.scope(TimingRegion::OutputWriterWait, Executor(1)); }
    require(TestClock::reads == 0 && TestEvents::records == 0,
            "startup/final output was included in cycle timers");
    timer.begin_cycle(0, false);
    finish_cycle(timer);
    timer.begin_cycle(1, false);
    finish_cycle(timer);
    require(TestEvents::creations == creations, "cycle boundary allocated events");
  }
  require(TestEvents::destructions == TestEvents::creations, "persistent events leaked");

  TestEvents::reset();
  {
    Timing timer(true, 2);
    const std::size_t pair_count = tpic::simulation::timing_event_pairs(2);
    require(TestEvents::creations == 2 * static_cast<int>(pair_count),
            "event allocation ignored the actual species count");
    timer.begin_cycle(1, false);
    require_throws([&] {
      auto scope = timer.scope(TimingRegion::Mover, Executor(1), 2);
    }, "GPU scope accepted an unprepared species slot");
    require_throws([&] {
      auto scope = timer.scope(TimingRegion::OutputWriterWait, Executor(1));
    }, "host-only scope accepted an unprepared device event slot");
    finish_cycle(timer);
  }
}

void scopes_and_groups() {
  TestEvents::reset();
  Timing timer(true);
  timer.begin_cycle(0, true);
  {
    auto outer = timer.scope(TimingRegion::FieldSolve, Executor(1));
    {
      auto first = timer.scope(TimingRegion::CurrentDeposition, Executor(2), 0);
      TestClock::milliseconds += 3;
      TestEvents::stream_time[2] += 7;
    }
    {
      auto second = timer.scope(TimingRegion::CurrentDeposition, Executor(2), 9);
      TestClock::milliseconds += 4;
      TestEvents::stream_time[2] += 11;
    }
    TestClock::milliseconds += 3;
    TestEvents::stream_time[1] += 20;
  }
  require(TestEvents::queries == 0 && TestEvents::elapsed_reads == 0,
          "a scope boundary waited for GPU completion");
  finish_cycle(timer);
  require(TestEvents::queries == 3 && TestEvents::elapsed_reads == 3,
          "collection did not gather each used event pair once");
  near(timer.table().cycle_stats(TimingRegion::FieldSolve).total_ms(), 10., "nested host parent");
  near(timer.table().cycle_stats(TimingRegion::FieldSolve, true).total_ms(), 20., "GPU parent elapsed");
  near(timer.table().cycle_stats(TimingRegion::CurrentDeposition).total_ms(), 7., "species host sum");
  near(timer.table().cycle_stats(TimingRegion::CurrentDeposition, true).total_ms(), 18., "species GPU sum");
  require(timer.table().stats(TimingRegion::CurrentDeposition).count() == 2 &&
          timer.table().stats(TimingRegion::CurrentDeposition, TimingGroup::all, true).count() == 2,
          "species calls were replaced with a cycle count");
  require(timer.table().stats(TimingRegion::FieldSolve, TimingGroup::output).count() == 0 &&
          timer.table().stats(TimingRegion::FieldSolve, TimingGroup::ordinary).count() == 0,
          "cycle zero was included in steady groups");

  timer.begin_cycle(1, false);
  {
    auto scope = timer.scope(TimingRegion::FieldSolve, Executor(1));
    TestClock::milliseconds += 5;
    TestEvents::stream_time[1] += 9;
  }
  finish_cycle(timer);
  near(timer.table().cycle_stats(TimingRegion::FieldSolve).total_ms(), 5., "cycle accumulator was not reset");
  near(timer.table().stats(TimingRegion::FieldSolve, TimingGroup::ordinary, true).total_ms(), 9., "ordinary GPU sum");
  require(timer.table().stats(TimingRegion::FieldSolve, TimingGroup::ordinary).count() == 1,
          "ordinary cycle missing");
  timer.begin_cycle(2, true);
  {
    auto scope = timer.scope(TimingRegion::FieldSolve);
    TestClock::milliseconds += 8;
  }
  finish_cycle(timer);
  require(timer.table().stats(TimingRegion::FieldSolve, TimingGroup::all).count() == 3 &&
          timer.table().stats(TimingRegion::FieldSolve, TimingGroup::output).count() == 1,
          "cycle classification is wrong");
  near(timer.table().stats(TimingRegion::FieldSolve).total_ms(), 23., "all cycles sum");

  std::FILE* file = std::tmpfile();
  require(file != nullptr, "cannot create timing output capture");
  timer.print_cycle(file);
  timer.print_summary(file);
  std::rewind(file);
  std::string output;
  std::array<char, 1024> buffer{};
  while (const auto count = std::fread(buffer.data(), 1, buffer.size(), file))
    output.append(buffer.data(), count);
  std::fclose(file);
  require(output.find("CurrentDeposition") != std::string::npos &&
          output.find("Ordinary cycles (after the first)") != std::string::npos &&
          output.find("Output cycles (after the first)") != std::string::npos &&
          output.find("StdDev") != std::string::npos &&
          output.find("not additive") != std::string::npos,
          "summary omitted comparison labels or overlap warning");
}

void lifetime_errors() {
  TestEvents::reset();
  {
    Timing timer(true);
    timer.begin_cycle(1, false);
    {
      auto scope = timer.scope(TimingRegion::SolveMaxwell, Executor(1));
      require_throws([&] { timer.close_cycle(); }, "close_cycle accepted an open scope");
    }
    require_throws([&] {
      auto scope = timer.scope(TimingRegion::SolveMaxwell, Executor(1));
    }, "device event pair was reused within a cycle");
    require_throws([&] {
      auto scope = timer.scope(TimingRegion::Mover, Executor(1), tpic::simulation::timing_instance_count);
    }, "out-of-range device event instance was accepted");
    finish_cycle(timer);
  }
  require(TestEvents::creations == TestEvents::destructions, "error handling leaked events");

  TestEvents::reset();
  TestEvents::fail_creation = 3;
  require_throws([] { Timing timer(true); }, "event construction failure was ignored");
  require(TestEvents::creations == 2 && TestEvents::destructions == 2,
          "partially created profiler leaked events");

  TestEvents::reset();
  TestEvents::fail_record = 2;
  {
    Timing timer(true);
    timer.begin_cycle(1, false);
    { auto scope = timer.scope(TimingRegion::SolveMaxwell, Executor(1)); }
    require_throws([&] { timer.close_cycle(); }, "stop-event error was lost in RAII destructor");
  }
  require(TestEvents::creations == TestEvents::destructions, "stop-event failure leaked events");
}

void deferred_phase_collection_and_worker_merge() {
  TestEvents::reset();
  Timing main(true, 2), deposition(true, 2), mover(true, 2);
  main.begin_cycle(1, false);
  deposition.begin_cycle(1, false);
  mover.begin_cycle(1, false);
  {
    auto resident = deposition.scope(TimingRegion::CurrentDeposition, Executor(1), 0,
                                      TimingPhase::resident);
    TestClock::milliseconds += 2;
    TestEvents::stream_time[1] += 3;
  }
  {
    auto tail = deposition.scope(TimingRegion::CurrentDeposition, Executor(1), 0,
                                  TimingPhase::tail);
    TestClock::milliseconds += 4;
    TestEvents::stream_time[1] += 5;
  }
  {
    auto moving = mover.scope(TimingRegion::Mover, Executor(2), 1);
    TestClock::milliseconds += 6;
    TestEvents::stream_time[2] += 7;
  }
  require_throws([&] {
    auto duplicate = deposition.scope(TimingRegion::CurrentDeposition, Executor(1), 0,
                                      TimingPhase::resident);
  }, "resident event reused before collection");
  require_throws([&] { main.merge_cycle(deposition); }, "unpublished worker merged");
  deposition.close_cycle();
  mover.close_cycle();
  require(TestEvents::queries == 0 && TestEvents::elapsed_reads == 0,
          "closing workers waited for their events");
  require_throws([&] { deposition.begin_cycle(2, false); }, "pending worker slots reused");
  require_throws([&] { main.merge_cycle(deposition); }, "uncollected worker merged");
  TestEvents::ready = false;
  require(!deposition.progress_collect_cycle() && !deposition.progress_collect_cycle(),
          "unfinished worker stop event accepted");
  require(TestEvents::elapsed_reads == 0 &&
              deposition.table().cycle_stats(TimingRegion::CurrentDeposition, true).count() == 0,
          "pending collection inspected elapsed time");
  TestEvents::ready = true;
  require(deposition.progress_collect_cycle() && mover.progress_collect_cycle(),
          "completed workers not collected");
  main.merge_cycle(deposition);
  main.merge_cycle(mover);
  require_throws([&] { main.merge_cycle(deposition); }, "worker epoch merged twice");
  main.close_cycle();
  require(main.progress_collect_cycle(), "empty main event collection failed");
  require(TestEvents::elapsed_reads == 3,
          "query collection inserted profiling synchronization");
  require(main.table().stats(TimingRegion::CurrentDeposition).count() == 2 &&
              main.table().stats(TimingRegion::CurrentDeposition, TimingGroup::ordinary, true).count() == 2,
          "resident/tail samples collapsed during worker merge");
  near(main.table().cycle_stats(TimingRegion::CurrentDeposition).total_ms(), 6., "merged worker host sum");
  near(main.table().cycle_stats(TimingRegion::CurrentDeposition, true).total_ms(), 8., "merged worker GPU sum");
  near(main.table().cycle_stats(TimingRegion::Mover, true).total_ms(), 7., "merged mover GPU sum");
  require_throws([&] { (void)main.progress_collect_cycle(); }, "collected cycle collected twice");
  deposition.begin_cycle(2, true);
  {
    auto tail = deposition.scope(TimingRegion::CurrentDeposition, Executor(1), 0,
                                  TimingPhase::tail);
    TestEvents::stream_time[1] += 11;
  }
  deposition.close_cycle();
  require(deposition.progress_collect_cycle(), "next-generation tail slot was not reusable");
  require_throws([&] { main.merge_cycle(deposition); }, "different worker epoch merged");
}
} // namespace

int main() {
  try {
    statistics();
    disabled_and_inactive();
    scopes_and_groups();
    lifetime_errors();
    deferred_phase_collection_and_worker_merge();
    std::cout << "timing statistics, stream attribution and event lifecycle checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
