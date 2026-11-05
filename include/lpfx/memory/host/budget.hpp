#ifndef LPFX_MEMORY_HOST_BUDGET_HPP
#define LPFX_MEMORY_HOST_BUDGET_HPP

#include <algorithm>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <utility>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/result.hpp>

namespace lpfx {

// A prepared host owner has one retained allocation. Keep only the aggregate
// information needed to admit and allocate it; identity and domain metadata do
// not affect either operation.
class HostMemoryRequirement final {
public:
  [[nodiscard]] constexpr bool try_add(const std::size_t bytes,
                                       const std::size_t alignment) noexcept {
    if (bytes == 0) {
      return true;
    }
    std::size_t next_bytes = 0;
    if (!std::has_single_bit(alignment) ||
        !detail::checked_add(bytes_, bytes, next_bytes)) {
      return false;
    }
    bytes_ = next_bytes;
    alignment_ = std::max(alignment_, alignment);
    return true;
  }

  [[nodiscard]] constexpr bool empty() const noexcept { return bytes_ == 0; }

  [[nodiscard]] constexpr std::size_t bytes() const noexcept { return bytes_; }

  [[nodiscard]] constexpr std::size_t alignment() const noexcept {
    return alignment_;
  }

private:
  std::size_t bytes_ = 0;
  std::size_t alignment_ = 1;
};

namespace detail {

template <class Layout>
concept SingleHostAllocationLayout = requires(const Layout &layout) {
  { layout.valid() } -> std::same_as<bool>;
  { layout.total_bytes() } -> std::same_as<std::size_t>;
};

} // namespace detail

template <std::size_t Alignment, detail::SingleHostAllocationLayout Layout>
  requires(std::has_single_bit(Alignment))
[[nodiscard]] constexpr bool try_make_single_host_allocation_requirement(
    const Layout &layout, HostMemoryRequirement &output) noexcept {
  HostMemoryRequirement candidate{};
  if (!layout.valid() || !candidate.try_add(layout.total_bytes(), Alignment)) {
    return false;
  }
  output = candidate;
  return true;
}

class HostMemoryBudget;

// The lease is the reservation. Moving it into an allocation keeps the bytes
// charged; destruction or explicit release returns them to the budget.
class HostMemoryLease final {
public:
  constexpr HostMemoryLease() noexcept = default;

  HostMemoryLease(HostMemoryLease &&other) noexcept
      : state_(std::exchange(other.state_, {})) {}

  HostMemoryLease &operator=(HostMemoryLease &&other) noexcept {
    if (this != &other) {
      release();
      state_ = std::exchange(other.state_, {});
    }
    return *this;
  }

  ~HostMemoryLease() noexcept { release(); }

  [[nodiscard]] bool active() const noexcept { return state_.budget != nullptr; }
  // Cold allocation replacement preserves the original admission lifetime.
  [[nodiscard]] HostMemoryBudget *budget() const noexcept {
    return state_.budget;
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return state_.bytes; }
  [[nodiscard]] std::size_t alignment() const noexcept {
    return state_.alignment;
  }

  // True when this lease is the admission for an allocation of exactly bytes
  // from an allocator aligned to allocator_alignment. Zero bytes need no lease.
  [[nodiscard]] bool
  matches(const std::size_t bytes,
          const std::size_t allocator_alignment) const noexcept {
    if (bytes == 0) {
      return !active() && state_.bytes == 0;
    }
    return active() && state_.bytes == bytes &&
           state_.alignment <= allocator_alignment;
  }

  void release() noexcept;

private:
  struct State {
    HostMemoryBudget *budget = nullptr;
    std::size_t bytes = 0;
    std::size_t alignment = 1;
  };

  constexpr HostMemoryLease(HostMemoryBudget *const budget,
                            const HostMemoryRequirement requirement) noexcept
      : state_{budget, requirement.bytes(), requirement.alignment()} {}

  State state_{};

  friend class HostMemoryBudget;
};

class HostMemoryBudget final {
public:
  // Setup is deliberately cold and single-threaded. Once sealed, limit_bytes_
  // is immutable and admissions use only the atomic used-byte counter.
  [[nodiscard]] bool try_set_limit(const std::size_t limit_bytes) noexcept {
    if (sealed_.load(std::memory_order_acquire) || configured_) {
      return false;
    }
    limit_bytes_ = limit_bytes;
    configured_ = true;
    return true;
  }

  [[nodiscard]] bool seal() noexcept {
    if (!configured_ || sealed_.load(std::memory_order_relaxed)) {
      return false;
    }
    sealed_.store(true, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool sealed() const noexcept {
    return sealed_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::size_t limit() const noexcept { return limit_bytes_; }

  [[nodiscard]] std::size_t used() const noexcept {
    return used_bytes_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] Result<HostMemoryLease>
  try_admit(const HostMemoryRequirement requirement) noexcept {
    if (requirement.empty()) {
      return Result<HostMemoryLease>::from_value(HostMemoryLease{});
    }
    if (!sealed()) {
      return Result<HostMemoryLease>::from_error(ErrorCode::invalid_state);
    }

    std::size_t current = used_bytes_.load(std::memory_order_relaxed);
    for (;;) {
      std::size_t next = 0;
      if (!detail::checked_add(current, requirement.bytes(), next)) {
        return Result<HostMemoryLease>::from_error(ErrorCode::overflow);
      }
      if (next > limit_bytes_) {
        return Result<HostMemoryLease>::from_error(
            ErrorCode::capacity_exceeded);
      }
      if (used_bytes_.compare_exchange_weak(current, next,
                                            std::memory_order_relaxed,
                                            std::memory_order_relaxed)) {
        return Result<HostMemoryLease>::from_value(
            HostMemoryLease{this, requirement});
      }
    }
  }

private:
  std::size_t limit_bytes_ = 0;
  std::atomic<std::size_t> used_bytes_ = 0;
  std::atomic<bool> sealed_ = false;
  bool configured_ = false;

  friend class HostMemoryLease;
};

inline void HostMemoryLease::release() noexcept {
  if (state_.budget != nullptr) {
    state_.budget->used_bytes_.fetch_sub(state_.bytes,
                                         std::memory_order_relaxed);
    state_ = {};
  }
}

} // namespace lpfx

#endif
