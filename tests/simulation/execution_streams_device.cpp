#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <tpic/simulation/execution/execution_streams.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Streams = tpic::simulation::ExecutionStreams<Backend>;
using Event = lpfx::device::DeviceEvent<Backend>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(Traits::error_type error) {
  require(Traits::succeeded(error), "native pipeline stream operation failed");
}
template <class Function> void rejected(Function function, const char* message) {
  try { function(); } catch (const std::runtime_error&) { return; }
  throw std::runtime_error(message);
}

__global__ void write_stream_marker_kernel(unsigned* target, unsigned value) { *target = value; }

void run(std::int32_t ordinal) {
  lpfx::device::DeviceStream<Backend> main_owner;
  check(lpfx::device::DeviceStream<Backend>::create(main_owner));
  Streams streams(ordinal);
  const auto main = streams.admit(main_owner.executor(), "main/startup");
  for (const auto role : {"move", "field_update", "current", "mass", "diagnostics",
                          "interior", "local[0]", "sort[0]", "local[1]", "sort[1]", "transfer"})
    (void)streams.create(role);
  (void)streams.create("route", true);
  rejected([&] { (void)Streams::validate_borrowed(Executor{Traits::null_stream}, ordinal); },
           "native null stream admitted");
  rejected([&] { (void)Streams::validate_borrowed(main, ordinal + 1); },
           "native wrong-device executor admitted");
#if defined(__HIPCC__)
  for (const auto stream : {hipStreamLegacy, hipStreamPerThread})
    rejected([&] { (void)Streams::validate_borrowed(Executor{stream}, ordinal); },
             "native HIP special default admitted");
  hipStream_t blocking{};
  check(hipStreamCreateWithFlags(&blocking, hipStreamDefault));
#else
  for (const auto stream : {cudaStreamLegacy, cudaStreamPerThread})
    rejected([&] { (void)Streams::validate_borrowed(Executor{stream}, ordinal); },
             "native CUDA special default admitted");
  cudaStream_t blocking{};
  check(cudaStreamCreateWithFlags(&blocking, cudaStreamDefault));
#endif
  rejected([&] { (void)Streams::validate_borrowed(Executor{blocking}, ordinal); },
           "native blocking borrowed stream admitted");
  check(Traits::destroy_stream(blocking));
  streams.seal();

  lpfx::device::DeviceAllocation<Backend> device;
  lpfx::device::PinnedHostAllocation<Backend> host;
  const auto bytes = streams.size() * sizeof(unsigned);
  check(lpfx::device::DeviceAllocation<Backend>::create(bytes, device));
  check(lpfx::device::PinnedHostAllocation<Backend>::create(bytes, host));
  std::array<Event, 32> completed;
  for (std::size_t index = 0; index < streams.size(); ++index) {
    const auto executor = streams.registration(index).executor;
    bool nonblocking{};
    std::int32_t actual{-1};
    check(Traits::query_stream_nonblocking(executor.stream(), nonblocking));
    check(Traits::query_stream_device_ordinal(executor.stream(), actual));
    require(nonblocking && actual == ordinal, "owned native stream flags/affinity changed");
    check(Event::create(completed[index]));
    auto* target = reinterpret_cast<unsigned*>(device.raw_base()) + index;
    check(Traits::memset_device_async(target, 0, sizeof(unsigned), executor.stream()));
    lpfx::device::launch_or_throw<write_stream_marker_kernel>("stream marker launch", executor,
        lpfx::device::LaunchGrid{1, 1}, target, static_cast<unsigned>(index + 1));
    check(completed[index].record(executor));
    check(completed[index].wait_on(main));
  }
  check(Traits::copy_device_to_host_async(host.raw_base(), device.raw_base(), bytes, main.stream()));
  check(Traits::synchronize_stream(main.stream()));
  for (std::size_t index = 0; index < streams.size(); ++index)
    require(reinterpret_cast<const unsigned*>(host.raw_base())[index] == index + 1,
            "named lane kernel/copy/event dependency failed");
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
    std::cerr << "native stream test: device query failed ("
              << static_cast<int>(queried) << ")\n";
    return 1;
  }
  try {
    run(properties.device_ordinal);
    std::cout << "native named stream flags, affinity and submissions passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
