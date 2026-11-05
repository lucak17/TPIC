#ifndef LPFX_EXECUTION_HOST_CHANNEL_HPP
#define LPFX_EXECUTION_HOST_CHANNEL_HPP

#include <array>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace lpfx {

enum class HostChannelStatus { success, would_block, closed };

// A bounded FIFO for typed ownership handoffs. Multiple producers and consumers
// are supported. Storage is inline; sending/receiving never allocates
// internally. Successful send publishes the record and prior writes to its
// receiver. Any pointed-to storage must outlive its consumer and have an
// explicit owner. close() rejects new records, wakes waiters, and preserves
// accepted records for draining. Destroy only after all users have stopped
// accessing the channel.
template <class T, std::size_t Capacity>
  requires(Capacity > 0 && std::is_nothrow_move_constructible_v<T> &&
           std::is_nothrow_move_assignable_v<T> &&
           std::is_nothrow_destructible_v<T>)
class HostChannel final {
public:
  // Rejection leaves the caller's record unchanged.
  [[nodiscard]] HostChannelStatus try_send(T &&value) noexcept {
    std::unique_lock lock{mutex_};
    if (closed_)
      return HostChannelStatus::closed;
    if (count_ == Capacity)
      return HostChannelStatus::would_block;
    insert(std::move(value));
    lock.unlock();
    readable_.notify_one();
    return HostChannelStatus::success;
  }

  [[nodiscard]] bool send(T &&value) noexcept {
    std::unique_lock lock{mutex_};
    writable_.wait(lock, [this] { return closed_ || count_ < Capacity; });
    if (closed_)
      return false;
    insert(std::move(value));
    lock.unlock();
    readable_.notify_one();
    return true;
  }

  // An unsuccessful receive leaves the caller's record unchanged.
  [[nodiscard]] HostChannelStatus try_receive(T &value) noexcept {
    std::unique_lock lock{mutex_};
    if (count_ == 0)
      return closed_ ? HostChannelStatus::closed
                     : HostChannelStatus::would_block;
    extract(value);
    lock.unlock();
    writable_.notify_one();
    return HostChannelStatus::success;
  }

  [[nodiscard]] bool receive(T &value) noexcept {
    std::unique_lock lock{mutex_};
    readable_.wait(lock, [this] { return closed_ || count_ != 0; });
    if (count_ == 0)
      return false;
    extract(value);
    lock.unlock();
    writable_.notify_one();
    return true;
  }

  void close() noexcept {
    {
      std::lock_guard lock{mutex_};
      closed_ = true;
    }
    readable_.notify_all();
    writable_.notify_all();
  }

private:
  void insert(T &&value) noexcept {
    slots_[(head_ + count_) % Capacity].emplace(std::move(value));
    ++count_;
  }

  void extract(T &value) noexcept {
    value = std::move(*slots_[head_]);
    slots_[head_].reset();
    head_ = (head_ + 1) % Capacity;
    --count_;
  }

  std::array<std::optional<T>, Capacity> slots_{};
  std::mutex mutex_{};
  std::condition_variable readable_{};
  std::condition_variable writable_{};
  std::size_t head_ = 0;
  std::size_t count_ = 0;
  bool closed_ = false;
};

} // namespace lpfx

#endif
