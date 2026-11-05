#ifndef TPIC_DEVICE_THREAD_INDEXING_HPP
#define TPIC_DEVICE_THREAD_INDEXING_HPP

#include <lpfx/core/attributes.hpp>

// The x-fastest index decode the grid kernels use after the LPFx grid-stride loop.
// It is always inlined, generates the index arithmetic a hand-written decode
// would, and is host-callable for region and layout code.
namespace tpic::device {

struct GridCoordinates {
  int i, j, k;
};

struct FaceCoordinates {
  int first, second;
};

// Decode a flat x-fastest index over nx * ny * (any) samples. The row quotient
// index / nx is formed once and reused, so the decode costs two divisions.
// Callers shift the result by their halo or region offset.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr GridCoordinates decode_x_fastest(
    const unsigned index, const unsigned nx, const unsigned ny) noexcept {
  const unsigned row = index / nx;
  return {int(index % nx), int(row % ny), int(row / ny)};
}

// The two-dimensional form, first axis fastest.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr FaceCoordinates decode_x_fastest(
    const unsigned index, const unsigned first_extent) noexcept {
  return {int(index % first_extent), int(index / first_extent)};
}

} // namespace tpic::device

#endif // TPIC_DEVICE_THREAD_INDEXING_HPP
