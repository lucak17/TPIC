#ifndef LPFX_DISTRIBUTED_DETAIL_STATIC_VECTOR_HPP
#define LPFX_DISTRIBUTED_DETAIL_STATIC_VECTOR_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace lpfx::detail {

template <class T, std::size_t Capacity>
  requires std::is_trivially_copyable_v<T> &&
           std::is_nothrow_default_constructible_v<T> &&
           (Capacity == 0 ||
            Capacity - 1 < std::numeric_limits<std::uint32_t>::max())
class static_vector final {
public:
  using value_type = T;

  constexpr static_vector() noexcept = default;

  [[nodiscard]] static constexpr std::size_t capacity() noexcept {
    return Capacity;
  }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] constexpr std::span<T> view() noexcept {
    return {entries_.data(), size_};
  }
  [[nodiscard]] constexpr std::span<const T> view() const noexcept {
    return {entries_.data(), size_};
  }

  [[nodiscard]] constexpr bool
  try_assign(const std::span<const T> input) noexcept {
    if (input.size() > Capacity) {
      return false;
    }
    for (std::size_t index = 0; index != input.size(); ++index) {
      entries_[index] = input[index];
    }
    size_ = static_cast<std::uint32_t>(input.size());
    return true;
  }

  [[nodiscard]] constexpr bool try_append(const T &value) noexcept {
    return try_insert(size_, value);
  }

  // Requires position <= size().
  [[nodiscard]] constexpr bool try_insert(const std::size_t position,
                                          const T &value) noexcept {
    if (size_ == Capacity) {
      return false;
    }
    for (std::size_t index = size_; index != position; --index) {
      entries_[index] = entries_[index - 1];
    }
    entries_[position] = value;
    ++size_;
    return true;
  }

  constexpr void clear() noexcept { size_ = 0; }

private:
  std::array<T, Capacity> entries_{};
  std::uint32_t size_{};
};

} // namespace lpfx::detail

#endif
