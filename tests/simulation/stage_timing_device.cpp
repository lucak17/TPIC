#include <lpfx/backend/backend.hpp>
#include <tpic/simulation/timing/stage_timing.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Timing = tpic::simulation::StageTiming<Backend>;
using tpic::simulation::TimingGroup;
using tpic::simulation::TimingPhase;
using tpic::simulation::TimingRegion;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(Traits::error_type error) {
  require(Traits::succeeded(error), "native timing test backend operation failed");
}

struct Stream {
  Traits::stream_type value{Traits::null_stream};
  Stream() { check(Traits::create_stream(value)); }
  ~Stream() { (void)Traits::destroy_stream(value); }
};
struct Buffer {
  std::byte* value{};
  static constexpr std::size_t bytes = 1 << 20;
  Buffer() { check(Traits::allocate_device(value, bytes)); }
  ~Buffer() { (void)Traits::deallocate_device(value); }
};

void run() {
  Stream first, second;
  Buffer first_buffer, second_buffer;
  Timing timer(true);
  for (int cycle = 0; cycle < 3; ++cycle) {
    timer.begin_cycle(cycle, cycle == 2);
    {
      auto main = timer.scope(TimingRegion::FieldSolve, Executor(first.value));
      {
        auto child = timer.scope(TimingRegion::MassDeposition, Executor(first.value), 0);
        check(Traits::memset_device_async(first_buffer.value, 0, Buffer::bytes, first.value));
      }
      {
        auto child = timer.scope(TimingRegion::CurrentDeposition, Executor(second.value), 0);
        check(Traits::memset_device_async(second_buffer.value, 0, Buffer::bytes, second.value));
      }
    }
    timer.close_cycle();
    while (!timer.progress_collect_cycle()) std::this_thread::yield();
    for (const auto region : {TimingRegion::FieldSolve, TimingRegion::MassDeposition,
                              TimingRegion::CurrentDeposition}) {
      const auto& stats = timer.table().cycle_stats(region, true);
      require(stats.count() == 1, "native event range was not collected");
      require(std::isfinite(stats.total_ms()) && stats.total_ms() >= 0.,
              "native event range returned an invalid elapsed time");
    }
  }
  require(timer.table().stats(TimingRegion::FieldSolve).count() == 3 &&
          timer.table().stats(TimingRegion::CurrentDeposition, TimingGroup::ordinary, true).count() == 1 &&
          timer.table().stats(TimingRegion::MassDeposition, TimingGroup::output, true).count() == 1,
          "native event reuse changed the sample grouping");

  Timing deposition(true);
  timer.begin_cycle(3, false);
  deposition.begin_cycle(3, false);
  for (const auto phase : {TimingPhase::resident, TimingPhase::tail}) {
    auto range = deposition.scope(TimingRegion::CurrentDeposition, Executor(first.value), 0, phase);
    check(Traits::memset_device_async(first_buffer.value, 0, Buffer::bytes, first.value));
  }
  deposition.close_cycle();
  while (!deposition.progress_collect_cycle()) std::this_thread::yield();
  timer.merge_cycle(deposition);
  timer.close_cycle();
  require(timer.progress_collect_cycle(), "empty parent event collection failed");
  require(timer.table().cycle_stats(TimingRegion::CurrentDeposition, true).count() == 2,
          "native resident/tail events were lost during deferred worker merge");
}
} // namespace

int main() {
  lpfx::device::DeviceProperties properties;
  const auto queried = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(queried)) {
    if (Traits::runtime_unavailable(queried)) {
      std::cout << "SKIP: no usable GPU device\n";
      return 77;
    }
    std::cerr << "native timing test: device query failed ("
              << static_cast<int>(queried) << ")\n";
    return 1;
  }
  try {
    run();
    std::cout << "native GPU timing event and stream reuse checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
