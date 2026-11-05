#include <lpfx/backend/traits.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace lpfx::device {
struct StreamTestBackend {};
struct PriorityStreamTestBackend {};

struct StreamTestState {
  using stream_type = int;
  using error_type = int;
  static constexpr int null_stream = 0, success_error = 0;
  static bool succeeded(int error) noexcept { return error == 0; }
  // No diagnostic strings: errors report their numeric code only.
  static const char* native_error_name(int) noexcept { return nullptr; }
  static const char* native_error_string(int) noexcept { return nullptr; }
  inline static int creations{}, destructions{}, priority_queries{}, priority_creations{}, inspections{}, submissions{};
  inline static int current_device = 2;
  inline static bool fail_create{}, fail_inspect{}, fail_priority{};
  inline static std::array<bool, 64> nonblocking{};
  inline static std::array<std::int32_t, 64> device{};
  static void reset() {
    creations = destructions = priority_queries = priority_creations = inspections = submissions = 0;
    current_device = 2;
    fail_create = fail_inspect = fail_priority = false;
    nonblocking.fill(true);
    device.fill(2);
  }
  static int create_stream(int& stream) noexcept {
    if (fail_create) return 1;
    stream = 10 + ++creations;
    nonblocking[stream] = true;
    device[stream] = current_device;
    return 0;
  }
  static int destroy_stream(int) noexcept { ++destructions; return 0; }
  static int synchronize_stream(int) noexcept { return 0; }
  static bool is_default_stream(int stream) noexcept {
    return stream == 0 || stream == 1 || stream == 2;
  }
  static int query_stream_nonblocking(int stream, bool& value) noexcept {
    ++inspections;
    if (fail_inspect || stream < 0 || stream >= static_cast<int>(device.size())) return 1;
    value = nonblocking[stream];
    return 0;
  }
  static int query_stream_device_ordinal(int stream, std::int32_t& ordinal) noexcept {
    if (stream < 0 || stream >= static_cast<int>(device.size())) return 1;
    ordinal = device[stream];
    return 0;
  }
  // Remaining DeviceDiscoveryBackend role, required by explicit-executor
  // validation; only query_stream_device_ordinal is ever reached.
  static bool runtime_unavailable(int) noexcept { return false; }
  static int query_device_count(int&) noexcept { return 1; }
  static int select_device(int) noexcept { return 1; }
  static int query_current_device_memory(DeviceMemoryInfo&) noexcept { return 1; }
  static int query_current_device_properties(DeviceProperties&) noexcept { return 1; }
  static int query_current_device_identity(DeviceIdentity&, std::int32_t&) noexcept { return 1; }
  static int query_device_pointer_ordinal(const void*, std::int32_t&) noexcept { return 1; }
};
template <> struct BackendTraits<StreamTestBackend> : StreamTestState {};
template <> struct BackendTraits<PriorityStreamTestBackend> : StreamTestState {
  using StreamTestState::create_stream;
  static int query_stream_priority_range(int& least, int& greatest) noexcept {
    ++priority_queries;
    if (fail_priority) return 1;
    least = 0;
    greatest = -2;
    return 0;
  }
  static int create_stream(int& stream, int priority) noexcept {
    ++priority_creations;
    return priority == -2 ? StreamTestState::create_stream(stream) : 1;
  }
};
} // namespace lpfx::device

#include <tpic/simulation/execution/execution_streams.hpp>

#include <string_view>
#include <vector>

namespace {
using State = lpfx::device::StreamTestState;
using Backend = lpfx::device::StreamTestBackend;
using Executor = lpfx::device::Executor<Backend>;
using Streams = tpic::simulation::ExecutionStreams<Backend>;
using PriorityStreams = tpic::simulation::ExecutionStreams<lpfx::device::PriorityStreamTestBackend>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Function> void rejected(Function function, const char* message) {
  try { function(); } catch (const std::runtime_error&) { return; }
  throw std::runtime_error(message);
}

void borrowed_admission() {
  State::reset();
  Streams registry(2);
  for (int stream : {0, 1, 2})
    rejected([&] { (void)registry.admit(Executor{stream}, "main"); }, "default stream admitted");
  require(State::inspections == 0, "special default handle reached native inspection");
  State::nonblocking[3] = false;
  rejected([&] { (void)registry.admit(Executor{3}, "blocking"); }, "blocking borrowed stream admitted");
  State::device[4] = 1;
  rejected([&] { (void)registry.admit(Executor{4}, "wrong device"); }, "wrong-device stream admitted");
  State::fail_inspect = true;
  rejected([&] { (void)registry.admit(Executor{5}, "broken"); }, "native query failure admitted");
  State::fail_inspect = false;
  require(registry.size() == 0, "rejected stream changed registry");
  const auto main = registry.admit(Executor{5}, "main");
  require(main.stream() == 5 && registry.size() == 1 &&
              registry.registration(0).role == "main" && !registry.registration(0).owned,
          "borrowed stream registration lost identity/provenance");
  (void)registry.admit(main, "startup alias");
  require(registry.size() == 1, "intentional borrowed alias consumed another slot");
  registry.seal();
  rejected([&] { (void)registry.admit(Executor{6}, "late"); }, "sealed registry mutated");
  rejected([&] { (void)registry.create("late"); }, "sealed registry created a stream");
  require(State::creations == 0 && State::destructions == 0, "borrowed stream ownership changed");
}

void ownership_and_priority() {
  State::reset();
  static_assert(!Streams::high_priority_supported);
  static_assert(PriorityStreams::high_priority_supported);
  {
    Streams registry(2);
    const auto normal = registry.create("move");
    const auto route = registry.create("route", true);
    require(normal.stream() != route.stream() && registry.size() == 2 &&
                registry.registration(0).owned && !registry.registration(1).high_priority,
            "normal-capability fallback collapsed task streams");
    State::fail_create = true;
    rejected([&] { (void)registry.create("failed"); }, "failed creation fell back to a stream");
    State::fail_create = false;
    State::fail_inspect = true;
    rejected([&] { (void)registry.create("failed inspection"); }, "uninspected owned stream admitted");
    State::fail_inspect = false;
    require(State::creations == 3 && State::destructions == 1 && registry.size() == 2,
            "failed inspection leaked an owned stream or registration");
  }
  require(State::creations == State::destructions, "owned streams leaked");
  State::reset();
  {
    PriorityStreams registry(2);
    (void)registry.create("move");
    (void)registry.create("route", true);
    require(State::priority_queries == 1 && State::priority_creations == 1 &&
                registry.registration(1).high_priority, "route did not request highest priority");
    State::fail_priority = true;
    rejected([&] { (void)registry.create("failed priority", true); },
             "runtime priority failure used an error fallback");
    require(State::creations == 2 && registry.size() == 2, "priority failure mutated ownership");
  }
  require(State::creations == State::destructions, "priority streams leaked");
  State::reset();
  {
    tpic::simulation::ExecutionStreams<Backend, 1> registry(2);
    (void)registry.create("one");
    rejected([&] { (void)registry.create("overflow"); }, "fixed registry grew");
    require(State::creations == 1, "capacity rejected after creating a stream");
  }
}

// The created executors follow the role table in creation order; snapshot-only
// roles alias diagnostics and control without snapshot output, sort aliases
// local when unsorted. Two species run in a pack compiled for three: the third
// owns no stream.
void role_executors() {
  using Layout = tpic::simulation::ExecutionLayout;
  for (const bool snapshot : {false, true}) {
    for (const bool sorted : {false, true}) {
      State::reset();
      Streams registry(2);
      const auto control = registry.admit(Executor{5}, "control");
      const tpic::simulation::StreamExecutors<Backend, 3> executors(registry, control, snapshot, sorted, 2);
      require(registry.size() == Layout::streams(2, sorted, snapshot),
              "created executors disagree with the stream layout");
      std::vector<std::string_view> expected{"control", "current", "mass", "diagnostics",
          "diagnostics_helper", "solver", "solver_interior", "solver_mass", "route", "move",
          "field_update"};
      if (snapshot) expected.insert(expected.end(), {"output_capture", "output_transfer"});
      expected.insert(expected.end(), {"local", "local"});
      if (sorted) expected.insert(expected.end(), {"sort", "sort"});
      for (std::size_t i = 0; i < registry.size(); ++i)
        require(registry.registration(i).role == expected[i], "stream creation order changed");
      require((executors.output_capture.stream() == executors.diagnostics.stream()) != snapshot &&
                  (executors.output_transfer.stream() == control.stream()) != snapshot,
              "snapshot-only roles alias the wrong streams");
      for (std::size_t s = 0; s < 2; ++s)
        require((executors.sort[s].stream() == executors.local[s].stream()) != sorted,
                "sort stream aliasing disagrees with the particle order");
      require(!executors.local[2].bound() && !executors.sort[2].bound(),
              "an unused species of the capacity owns a stream");
    }
  }
}
} // namespace

int main() {
  try {
    borrowed_admission();
    ownership_and_priority();
    role_executors();
    std::cout << "pipeline stream ownership, admission and priority checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
