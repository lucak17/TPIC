#ifndef LPFX_CORE_MEMORY_INTRINSICS_HPP
#define LPFX_CORE_MEMORY_INTRINSICS_HPP

#include <bit>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/attributes.hpp>

namespace lpfx::detail {

template <std::size_t Alignment, class T>
  requires(std::has_single_bit(Alignment))
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE T *
assume_aligned_pointer(T *const pointer) noexcept {
  return static_cast<T *>(__builtin_assume_aligned(pointer, Alignment));
}

// Copy a compile-time-sized object representation without imposing scalar
// alignment on dense wire storage. The builtin lets host and device compilers
// select naturally wide unaligned transactions.
template <std::size_t Bytes>
  requires(Bytes != 0)
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void
copy_constant_size_representation(std::byte *const destination,
                                  const std::byte *const source) noexcept {
  __builtin_memcpy(destination, source, Bytes);
}

#if LPFX_DEVICE_PASS
// Chunk types may alias any object representation, so the typed copy below
// stays exempt from type-based alias analysis on every device compiler.
template <std::size_t Bytes> struct AlignedCopyChunk;
template <> struct AlignedCopyChunk<2> {
  typedef std::uint16_t __attribute__((__may_alias__)) type;
};
template <> struct AlignedCopyChunk<4> {
  typedef std::uint32_t __attribute__((__may_alias__)) type;
};
template <> struct AlignedCopyChunk<8> {
  typedef std::uint64_t __attribute__((__may_alias__)) type;
};
#endif

// Copy a compile-time-sized object representation between two addresses that
// are both Alignment-aligned. GPU compilers split a copy of unknown alignment
// into byte accesses, so device code issues one typed access per aligned
// chunk of at most 8 bytes. Host code and Alignment 1 keep the builtin copy.
// Either way the bytes are copied in order, unchanged.
template <std::size_t Bytes, std::size_t Alignment>
  requires(Bytes != 0) && (std::has_single_bit(Alignment)) &&
          (Bytes % Alignment == 0)
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void
copy_aligned_representation(std::byte *const destination,
                            const std::byte *const source) noexcept {
#if LPFX_DEVICE_PASS
  if constexpr (Alignment != 1) {
    constexpr std::size_t chunk = Alignment < 8 ? Alignment : 8;
    using Chunk = typename AlignedCopyChunk<chunk>::type;
    auto *const to = static_cast<Chunk *>(
        __builtin_assume_aligned(static_cast<void *>(destination), chunk));
    const auto *const from = static_cast<const Chunk *>(
        __builtin_assume_aligned(static_cast<const void *>(source), chunk));
#pragma unroll
    for (std::size_t index = 0; index != Bytes / chunk; ++index) {
      to[index] = from[index];
    }
  } else {
    copy_constant_size_representation<Bytes>(destination, source);
  }
#else
  copy_constant_size_representation<Bytes>(destination, source);
#endif
}

} // namespace lpfx::detail

#endif
