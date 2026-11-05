#ifndef LPFX_EXECUTION_HOST_WORKER_HPP
#define LPFX_EXECUTION_HOST_WORKER_HPP

#include <concepts>
#include <cstddef>
#include <thread>
#include <type_traits>
#include <utility>

#include <lpfx/execution/host/channel.hpp>

namespace lpfx {

// One persistent thread consumes an inline, bounded mailbox. The worker is
// constructed idle; start(), called once, launches the thread, which holds the
// program and calls it directly for every job: no futures and no per-job
// erasure or allocation. Jobs are small default-constructible handoff records.
// The noexcept program publishes algorithm status/completion through
// caller-owned records, channels, or fences. Queue only runnable work: a
// program waiting for work queued behind itself creates a dependency cycle.
//
// Setup runs once on the new thread, before the first job (e.g. device
// binding). If setup can fail, either terminate the process on failure or
// publish its result through a caller-owned status and fence and admit no job
// until that result is checked. No setup-success inference is made from start
// returning.
//
// submit/try_submit/close support multiple callers. start, join and
// destruction require one controlling thread, distinct from this worker. Close
// all dependency waits needed by programs before joining a cancelled pipeline.
// Accepted jobs drain once started; their referenced storage must remain alive
// through join().
template <class Job, std::size_t Capacity = 1>
  requires(Capacity > 0 && std::is_nothrow_default_constructible_v<Job> &&
           std::is_nothrow_move_constructible_v<Job> &&
           std::is_nothrow_move_assignable_v<Job> &&
           std::is_nothrow_destructible_v<Job>)
class HostWorker final {
public:
  ~HostWorker() { join(); }

  template <class Program, class Setup = void (*)() noexcept>
    requires requires(Program &program, Setup &setup, Job &&job) {
      { program(std::move(job)) } noexcept -> std::same_as<void>;
      { setup() } noexcept -> std::same_as<void>;
    }
  void start(Program program, Setup setup = []() noexcept {}) {
    thread_ = std::thread([this, program = std::move(program),
                           setup = std::move(setup)]() mutable noexcept {
      setup();
      Job job{};
      while (inbox_.receive(job))
        program(std::move(job));
    });
  }

  [[nodiscard]] bool submit(Job &&job) noexcept {
    return inbox_.send(std::move(job));
  }

  [[nodiscard]] HostChannelStatus try_submit(Job &&job) noexcept {
    return inbox_.try_send(std::move(job));
  }

  void close() noexcept { inbox_.close(); }

  void join() noexcept {
    close();
    if (thread_.joinable())
      thread_.join();
  }

private:
  HostChannel<Job, Capacity> inbox_{};
  std::thread thread_;
};

} // namespace lpfx

#endif
