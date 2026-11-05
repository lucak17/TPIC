#ifndef LPFX_MEMORY_HOST_ALLOCATOR_HPP
#define LPFX_MEMORY_HOST_ALLOCATOR_HPP

#include <bit>
#include <concepts>
#include <cstddef>
#include <new>

#include <lpfx/memory/space.hpp>

namespace lpfx {

template <std::size_t Alignment>
  requires(Alignment >= alignof(std::max_align_t) &&
           std::has_single_bit(Alignment))
struct HostAlignedAllocator {
  using space_type = HostSpace;

  static constexpr std::size_t minimum_data_alignment = Alignment;

  [[nodiscard]] static void *allocate(const std::size_t bytes) noexcept {
    if (bytes == 0) {
      return nullptr;
    }
    return ::operator new(bytes, std::align_val_t{Alignment}, std::nothrow);
  }

  static void deallocate(void *const pointer) noexcept {
    if (pointer != nullptr) {
      ::operator delete(pointer, std::align_val_t{Alignment});
    }
  }
};

template <>
struct DefaultAllocator<HostSpace> final
    : HostAlignedAllocator<alignof(std::max_align_t)> {};

template <> struct AllocatorTraits<HostSpace> final {
  static constexpr std::size_t minimum_data_alignment =
      DefaultAllocator<HostSpace>::minimum_data_alignment;
};

namespace detail {

template <class Allocator>
concept HostRawAllocator =
    std::same_as<typename Allocator::space_type, HostSpace> &&
    requires(const std::size_t bytes, void *const pointer) {
      { Allocator::minimum_data_alignment } -> std::convertible_to<std::size_t>;
      { Allocator::allocate(bytes) } noexcept -> std::same_as<void *>;
      { Allocator::deallocate(pointer) } noexcept -> std::same_as<void>;
    } && (Allocator::minimum_data_alignment >= alignof(std::max_align_t)) &&
    std::has_single_bit(
        static_cast<std::size_t>(Allocator::minimum_data_alignment));

} // namespace detail

} // namespace lpfx

#endif
