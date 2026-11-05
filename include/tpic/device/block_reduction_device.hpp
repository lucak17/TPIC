#ifndef TPIC_DEVICE_BLOCK_REDUCTION_DEVICE_HPP
#define TPIC_DEVICE_BLOCK_REDUCTION_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/backend/launch.hpp>

#include <cstddef>

// The cross-block reducer shared by the scalar and Gauss diagnostics and
// GMRES. Their block kernels sum with lpfx::device::subgroup::block_reduce_sum and this
// kernel keeps the same tree, so results are bitwise those of one LPFx block
// reduction per component; the tree differs between 32- and 64-wide warps and
// is fixed within each backend.
namespace tpic::device::kernels {

// One block per output row: the block sums partial_count block partials of its
// row in thread-stride order and thread 0 stores the sum. Rows at or above
// filled_rows store zero (the padding of a fixed-size collective batch).
// Launch: one block of lpfx::device::default_block_threads per row; no
// validation flag.
template <class T>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void reduce_block_partials_kernel(
    const T* partials, T* values, unsigned row_stride, unsigned partial_count,
    unsigned filled_rows) {
  constexpr unsigned threads = lpfx::device::default_block_threads;
  __shared__ T scratch[lpfx::device::subgroup::block_warp_count<threads>];
  const unsigned row = blockIdx.x;
  if (row >= filled_rows) {
    if (threadIdx.x == 0) values[row] = T(0);
    return;
  }
  T sum{};
  for (unsigned index = threadIdx.x; index < partial_count; index += blockDim.x)
    sum += partials[static_cast<std::size_t>(row) * row_stride + index];
  sum = lpfx::device::subgroup::block_reduce_sum<threads>(sum, scratch);
  if (threadIdx.x == 0) values[row] = sum;
}

} // namespace tpic::device::kernels

#endif // TPIC_DEVICE_BLOCK_REDUCTION_DEVICE_HPP
