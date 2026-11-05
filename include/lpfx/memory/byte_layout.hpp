#ifndef LPFX_MEMORY_BYTE_LAYOUT_HPP
#define LPFX_MEMORY_BYTE_LAYOUT_HPP

#include <bit>
#include <cstddef>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx::detail {

// Places consecutive arrays in one allocation. Each array starts at the next
// multiple of its alignment. A failed append leaves the builder unchanged.
class ByteLayoutBuilder final {
public:
  // invalid_argument unless alignment is a power of two >= alignof(T);
  // overflow when the end offset is not representable.
  template <class T>
  [[nodiscard]] constexpr ErrorCode append(const std::size_t count,
                                           const std::size_t alignment,
                                           std::size_t &offset) noexcept {
    if (!std::has_single_bit(alignment) || alignment < alignof(T)) {
      return ErrorCode::invalid_argument;
    }
    std::size_t start = 0;
    std::size_t bytes = 0;
    std::size_t end = 0;
    if (!checked_align_up(bytes_, alignment, start) ||
        !checked_byte_count<T>(count, bytes) ||
        !checked_add(start, bytes, end)) {
      return ErrorCode::overflow;
    }
    bytes_ = end;
    if (alignment_ < alignment) {
      alignment_ = alignment;
    }
    offset = start;
    return ErrorCode::success;
  }

  template <class T>
  [[nodiscard]] constexpr ErrorCode append(const std::size_t count,
                                           std::size_t &offset) noexcept {
    return append<T>(count, alignof(T), offset);
  }

  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return bytes_;
  }

  // Largest alignment appended so far (1 when empty).
  [[nodiscard]] constexpr std::size_t alignment() const noexcept {
    return alignment_;
  }

private:
  std::size_t bytes_ = 0;
  std::size_t alignment_ = 1;
};

} // namespace lpfx::detail

#endif
