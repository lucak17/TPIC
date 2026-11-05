#ifndef TPIC_DEVICE_BLOCK_THREADS_HPP
#define TPIC_DEVICE_BLOCK_THREADS_HPP

// Block sizes of the kernel families that do not use
// lpfx::device::default_block_threads, the block of the field, solver,
// diagnostics, startup and other unsorted particle kernels. A kernel's
// __launch_bounds__ and its lpfx::device::KernelLaunch name the same constant,
// so the bound and the launch agree. Host-includable.
namespace tpic::device {

// Sorted-cell particle kernels (one native warp per cell) and unsorted mass
// deposition.
inline constexpr unsigned particle_block_threads = 128;
// Mass-matrix apply and mass-halo kernels.
inline constexpr unsigned mass_matrix_block_threads = 128;

} // namespace tpic::device

#endif // TPIC_DEVICE_BLOCK_THREADS_HPP
