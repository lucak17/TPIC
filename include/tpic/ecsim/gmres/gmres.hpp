#ifndef TPIC_ECSIM_GMRES_GMRES_HPP
#define TPIC_ECSIM_GMRES_GMRES_HPP

#include <lpfx/backend/properties.hpp>
#include <lpfx/backend/launch.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace tpic::ecsim {

// The block reductions require blockDim.x == gmres_reduction_threads; every
// GMRES launch but the single-thread back substitution uses this constant.
inline constexpr unsigned gmres_reduction_threads = lpfx::device::default_block_threads;
using GmresLaunch = lpfx::device::KernelLaunch<gmres_reduction_threads>;
inline constexpr unsigned gmres_dot_tile = 4;
// Four resident blocks per SM expose enough parallelism without making the
// final scalar reduction large.
inline constexpr int gmres_blocks_per_multiprocessor = 4;

// Workspace of restart length m, shared by Gmres and the memory estimate.
// Device vectors: m + 1 Krylov basis vectors, then x, b and r unless refined
// keeps them in separate reduction-type vectors, then the operator image.
[[nodiscard]] constexpr std::size_t gmres_vector_count(std::size_t restart, bool refined) noexcept {
  return restart + (refined ? 2 : 5);
}
// Entries of the fixed collective batch (at most m projection dots, or the
// two residual norms) and blocks of its partial reduction.
template <std::unsigned_integral Count>
[[nodiscard]] constexpr Count gmres_batch_capacity(Count restart) noexcept {
  return restart > Count{2} ? restart : Count{2};
}
// Dense state: the (m + 1) x m Hessenberg matrix, m cosines, m sines, m + 1
// least-squares right-hand sides, m coefficients and the two status values.
[[nodiscard]] constexpr std::size_t gmres_state_count(std::size_t restart) noexcept {
  return restart * (restart + 1) + 4 * restart + 3;
}
// The solver's work fields (solution, right-hand side, operator input and
// image) are stored in the wider of the field compute type and the solver
// reduction type, so the residual b - A*x is evaluated in the reduction type
// from the same fields.
template <class Precision>
using SolverStorage = std::conditional_t<
    (sizeof(typename Precision::Compute::SolverReduction) > sizeof(typename Precision::Compute::Electric)),
    typename Precision::Compute::SolverReduction, typename Precision::Compute::Electric>;
// With a Krylov type narrower than the reduction type the solution, the
// right-hand side, the residual and the image of the solution are separate
// reduction-type vectors.
inline constexpr std::size_t gmres_residual_vector_count = 4;
// Pinned host controls read back at each decision point.
inline constexpr std::size_t gmres_control_count = 2;
static_assert(gmres_vector_count(30, false) == 30 + 5 && gmres_vector_count(30, true) == 30 + 2 &&
              gmres_batch_capacity(30U) == 30U && gmres_batch_capacity(1U) == 2U &&
              gmres_state_count(30) == 30 * 31 + 4 * 30 + 3);

// Shared by device allocation and the memory estimate.
[[nodiscard]] inline unsigned gmres_reduction_blocks(
    unsigned count, const lpfx::device::DeviceProperties& properties) {
  if (count == 0 || properties.maximum_threads_per_block < gmres_reduction_threads ||
      properties.maximum_grid_size_x == 0 || properties.multiprocessor_count == 0 ||
      (properties.subgroup_width != 32 && properties.subgroup_width != 64))
    throw std::invalid_argument("TPIC GMRES: invalid reduction launch admission");
  return static_cast<unsigned>(std::min<std::uint64_t>({
      GmresLaunch::blocks(count),
      std::uint64_t(properties.multiprocessor_count) * gmres_blocks_per_multiprocessor,
      properties.maximum_grid_size_x}));
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_GMRES_GMRES_HPP
