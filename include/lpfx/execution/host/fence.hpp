#ifndef LPFX_EXECUTION_HOST_FENCE_HPP
#define LPFX_EXECUTION_HOST_FENCE_HPP

#include <atomic>
#include <cstdint>
#include <limits>

namespace lpfx {

// One producer publishes completed, strictly increasing generations to any
// number of observers. Generation zero is initially complete. No reset/ABA
// protocol is needed; use a new fence after max_generation is exhausted.
// signal() publishes CPU writes; it does not own their storage or wait for GPU
// completion. Publish a recorded DeviceEvent separately when GPU work is
// pending. close() may run concurrently and wakes waits that cannot now be
// satisfied. Close is irreversible. Destroy only after the producer, any
// closer and all observers have returned: signal() and close() still wake
// waiters after their state becomes visible.
class HostFence final {
public:
  static constexpr std::uint64_t max_generation =
      std::numeric_limits<std::uint64_t>::max() >> 1;

  // False means closed, out of range, or not newer than the published value.
  [[nodiscard]] bool signal(const std::uint64_t generation) noexcept {
    if (generation > max_generation)
      return false;
    auto observed = state_.load(std::memory_order_relaxed);
    for (;;) {
      if ((observed & closed_bit) != 0 || generation <= observed)
        return false;
      if (state_.compare_exchange_weak(observed, generation,
                                       std::memory_order_release,
                                       std::memory_order_relaxed)) {
        state_.notify_all();
        return true;
      }
    }
  }

  [[nodiscard]] bool ready(const std::uint64_t generation) const noexcept {
    return generation <=
           (state_.load(std::memory_order_acquire) & max_generation);
  }

  // False means closure before this generation, or an out-of-range request.
  [[nodiscard]] bool wait(const std::uint64_t generation) const noexcept {
    if (generation > max_generation)
      return false;
    auto observed = state_.load(std::memory_order_acquire);
    for (;;) {
      if ((observed & max_generation) >= generation)
        return true;
      if ((observed & closed_bit) != 0)
        return false;
      state_.wait(observed, std::memory_order_acquire);
      observed = state_.load(std::memory_order_acquire);
    }
  }

  void close() noexcept {
    state_.fetch_or(closed_bit, std::memory_order_release);
    state_.notify_all();
  }

private:
  static constexpr std::uint64_t closed_bit = max_generation + 1;
  std::atomic<std::uint64_t> state_{0};
};

} // namespace lpfx

#endif
