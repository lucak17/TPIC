#ifndef TPIC_ECSIM_GMRES_GMRES_DEVICE_HPP
#define TPIC_ECSIM_GMRES_GMRES_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/profiling.hpp>

#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/distributed/collective_operand.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/block_reduction_device.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/ecsim/fields/cell_to_node.hpp>
#include <tpic/ecsim/gmres/gmres.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tpic::ecsim {

template <class Reduction> struct GmresResult {
  bool converged = false;
  int iterations = 0;
  Reduction initial_residual{}, residual{}, rhs_norm{}, relative_residual{};
};

// Direction of a field-vector transfer: pack copies a nodal field owner into
// a compact vector, unpack copies the vector back.
enum class TransferDirection { pack, unpack };

// Device control record the host reads after each Arnoldi step. breakdown is
// one when the Arnoldi norm vanished, zero otherwise.
template <class Reduction> struct GmresIterationStatus {
  Reduction residual_estimate;
  Reduction breakdown;
};

namespace detail {

// Compile-time tails of gmres_normalize_kernel, run by block 0, thread 0 with
// the same norm2[0] the normalization reads.

// Restart: the least-squares right-hand side starts at the residual norm.
template <class Reduction> struct RestartRhsTail {
  Reduction* rhs;

  LPFX_DEVICE LPFX_ALWAYS_INLINE void operator()(const Reduction* norm2) const noexcept {
    rhs[0] = config::convert<Reduction>(::sqrt(static_cast<double>(norm2[0])));
  }
};

// Arnoldi step: append the subdiagonal norm to the Hessenberg column, apply
// the previous Givens rotations, form the new rotation, rotate the
// right-hand side and publish the residual estimate and the breakdown flag.
template <class Reduction> struct ArnoldiLeastSquaresUpdate {
  Reduction* column;
  Reduction* cosine;
  Reduction* sine;
  Reduction* rhs;
  GmresIterationStatus<Reduction>* status;
  int step;

  LPFX_DEVICE LPFX_ALWAYS_INLINE void operator()(const Reduction* norm2) const noexcept {
    const Reduction subdiagonal = config::convert<Reduction>(::sqrt(static_cast<double>(norm2[0])));
    column[step + 1] = subdiagonal;
    for (int row = 0; row < step; ++row) {
      const Reduction upper = column[row], lower = column[row + 1];
      column[row] = cosine[row] * upper + sine[row] * lower;
      column[row + 1] = -sine[row] * upper + cosine[row] * lower;
    }
    const Reduction diagonal = column[step];
    const Reduction denominator = config::convert<Reduction>(::hypot(
        static_cast<double>(diagonal), static_cast<double>(subdiagonal)));
    const Reduction c = denominator != Reduction(0) ? diagonal / denominator : Reduction(1);
    const Reduction s = denominator != Reduction(0) ? subdiagonal / denominator : Reduction(0);
    cosine[step] = c;
    sine[step] = s;
    column[step] = c * diagonal + s * subdiagonal;
    column[step + 1] = Reduction(0);
    const Reduction upper = rhs[step], lower = rhs[step + 1];
    rhs[step] = c * upper + s * lower;
    rhs[step + 1] = -s * upper + c * lower;
    status->residual_estimate = config::convert<Reduction>(::fabs(static_cast<double>(rhs[step + 1])));
    // A zero Arnoldi norm closes the invariant subspace. Accept its update,
    // then require the true residual; an optimistic zero estimate is not success.
    status->breakdown = subdiagonal == Reduction(0) ? Reduction(1) : Reduction(0);
  }
};

// The normalized Krylov vector is also stored into the operator input owner,
// which the next operator application reads, converted to the owner's value
// type. Packed index n = component * nodes + node.
template <class View> struct OperatorInputStore {
  View field;
  data::GridShape shape;

  template <class Scalar>
  LPFX_DEVICE LPFX_ALWAYS_INLINE void operator()(const unsigned n, const Scalar value) {
    const unsigned nodes = static_cast<unsigned>(shape.count);
    const unsigned component = n / nodes;
    const auto at = device::decode_x_fastest(n - component * nodes, shape.nx, shape.ny);
    if (component == 0)
      store<0>(at, value);
    else if (component == 1)
      store<1>(at, value);
    else
      store<2>(at, value);
  }

private:
  template <int Component, class Scalar>
  LPFX_DEVICE LPFX_ALWAYS_INLINE void store(const device::GridCoordinates at,
                                            const Scalar value) {
    const auto target = field.template component<Component>();
    target(at.i, at.j, at.k) = config::convert<typename decltype(target)::value_type>(value);
  }
};

} // namespace detail

namespace kernels {

// Compact x/y/z vectors hold every interior node (cells+1 per axis), so a node
// shared by neighbouring ranks enters a global dot product once per rank that
// stores it. Field padding and ghosts never enter a Krylov dot product.
// Grid-stride with GmresLaunch blocks; no validation flag.
template <TransferDirection Direction, class Scalar, class View>
__global__ __launch_bounds__(gmres_reduction_threads) void gmres_field_vector_transfer_kernel(
    View field, Scalar* vector, data::GridShape shape) {
  const auto x = field.template component<0>();
  const auto y = field.template component<1>();
  const auto z = field.template component<2>();
  lpfx::device::for_each_grid_index(static_cast<unsigned>(shape.count), [&](const unsigned n) {
    const auto at = device::decode_x_fastest(n, shape.nx, shape.ny);
    const int i = at.i, j = at.j, k = at.k;
    const auto size = static_cast<std::size_t>(shape.count);
    if constexpr (Direction == TransferDirection::pack) {
      vector[n] = config::convert<Scalar>(x(i, j, k));
      vector[size + n] = config::convert<Scalar>(y(i, j, k));
      vector[2 * size + n] = config::convert<Scalar>(z(i, j, k));
    } else {
      x(i, j, k) = config::convert<typename decltype(x)::value_type>(vector[n]);
      y(i, j, k) = config::convert<typename decltype(y)::value_type>(vector[size + n]);
      z(i, j, k) = config::convert<typename decltype(z)::value_type>(vector[2 * size + n]);
    }
  });
}

// Each block owns a vector segment and four adjacent basis vectors. The image
// load and loop/index work are shared by the four dot products. Partial sums
// use warp shuffles and a small shared exchange, never global atomics. Grid:
// reduction blocks along x, column tiles along y; no validation flag.
template <class Scalar, class Reduction>
__global__ __launch_bounds__(gmres_reduction_threads) void gmres_dot_batch_kernel(
    const Scalar* basis, const Scalar* image, Reduction* partials, unsigned count, int columns) {
  constexpr unsigned tile = gmres_dot_tile;
  __shared__ Reduction scratch[tile][lpfx::device::subgroup::block_warp_count<gmres_reduction_threads>];
  for (unsigned first = blockIdx.y * tile; first < static_cast<unsigned>(columns);
       first += gridDim.y * tile) {
    Reduction sum[tile]{};
    lpfx::device::for_each_grid_index(count, [&](const unsigned n) {
      const Reduction value = config::convert<Reduction>(image[n]);
#pragma unroll
      for (unsigned row = 0; row < tile; ++row)
        if (first + row < static_cast<unsigned>(columns))
          sum[row] += config::convert<Reduction>(basis[
              static_cast<std::size_t>(first + row) * count + n]) * value;
    });
    // R1: one LPFx block reduction per filled row, each stored before the
    // next starts. Reducing the rows together around one barrier measured
    // 1.4% to 4% slower for this kernel.
#pragma unroll
    for (unsigned row = 0; row < tile; ++row)
      if (first + row < static_cast<unsigned>(columns)) {
        const Reduction total = lpfx::device::subgroup::block_reduce_sum<
            gmres_reduction_threads>(sum[row], scratch[row]);
        if (threadIdx.x == 0)
          partials[static_cast<std::size_t>(first + row) * gridDim.x + blockIdx.x] = total;
      }
    __syncthreads(); // The next row tile reuses all four warp exchanges.
  }
}

// Form the residual while its operands are in registers. The squared norm is
// taken after storage rounding, matching a separate residual and dot kernel.
// Grid: the reduction blocks, grid-stride; one partial per block and sum.
template <bool RhsNorm, class Scalar, class Reduction>
__global__ __launch_bounds__(gmres_reduction_threads) void gmres_residual_squared_norm_kernel(
    const Scalar* rhs, const Scalar* image, Scalar* residual, Reduction* partials, unsigned count) {
  constexpr unsigned sums = RhsNorm ? 2 : 1;
  __shared__ Reduction scratch[sums][lpfx::device::subgroup::block_warp_count<gmres_reduction_threads>];
  Reduction sum[sums]{}; // Residual, then (RhsNorm) right-hand side.
  lpfx::device::for_each_grid_index(count, [&](const unsigned n) {
    const Scalar rhs_value = rhs[n];
    const Scalar value = rhs_value - image[n];
    residual[n] = value;
    sum[0] += config::convert<Reduction>(value) * config::convert<Reduction>(value);
    if constexpr (RhsNorm)
      sum[sums - 1] += config::convert<Reduction>(rhs_value) * config::convert<Reduction>(rhs_value);
  });
  lpfx::device::subgroup::block_reduce_sum<gmres_reduction_threads>(sum, scratch);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = sum[0];
    if constexpr (RhsNorm) partials[gridDim.x + blockIdx.x] = sum[sums - 1];
  }
}

// target = source / sqrt(norm2[0]) (zero for a zero norm); Store also writes
// each value into the operator input. Tail runs before the loop on block 0,
// thread 0. Grid: the reduction blocks, grid-stride. R1: every thread forms
// scale in double (square root and reciprocal, about fifteen FP64
// instructions), so a grid of one thread per element is FP64-bound; the
// reduction grid forms it a few times per SM and keeps the per-element work.
template <class Source, class Scalar, class Reduction, class Tail, class Store>
__global__ __launch_bounds__(gmres_reduction_threads) void gmres_normalize_kernel(
    const Source* source, const Reduction* norm2, Scalar* target, unsigned count, Tail tail,
    Store store) {
  if (blockIdx.x == 0 && threadIdx.x == 0) tail(norm2);
  const Reduction scale = norm2[0] > Reduction(0)
      ? Reduction(1) / config::convert<Reduction>(::sqrt(static_cast<double>(norm2[0])))
      : Reduction(0);
  lpfx::device::for_each_grid_index(count, [&](const unsigned n) {
    const Scalar value = config::convert<Scalar>(config::convert<Reduction>(source[n]) * scale);
    target[n] = value;
    store(n, value);
  });
}

// Classical Gram-Schmidt against the first columns basis vectors with the
// reduced dots, storing the rounded image and its squared-norm block partial;
// block 0 copies the dots into the Hessenberg column. Grid: the reduction
// blocks, grid-stride.
template <class Scalar, class Reduction>
__global__ __launch_bounds__(gmres_reduction_threads) void gmres_orthogonalize_squared_norm_kernel(
    const Scalar* basis, const Reduction* dots, Scalar* image,
    Reduction* hessenberg_column, Reduction* partials, unsigned count, int columns) {
  __shared__ Reduction scratch[lpfx::device::subgroup::block_warp_count<gmres_reduction_threads>];
  Reduction norm2{};
  lpfx::device::for_each_grid_index(count, [&](const unsigned n) {
    Reduction value = config::convert<Reduction>(image[n]);
    for (int column = 0; column < columns; ++column)
      value -= dots[column] * config::convert<Reduction>(
          basis[static_cast<std::size_t>(column) * count + n]);
    const Scalar stored = config::convert<Scalar>(value);
    image[n] = stored;
    norm2 += config::convert<Reduction>(stored) * config::convert<Reduction>(stored);
  });
  if (blockIdx.x == 0 && threadIdx.x == 0)
    for (int row = 0; row < columns; ++row) hessenberg_column[row] = dots[row];
  const Reduction total = lpfx::device::subgroup::block_reduce_sum<
      gmres_reduction_threads>(norm2, scratch);
  if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

// Back substitution of the upper-triangular least-squares system by a single
// thread (one block of one thread); a zero diagonal gives a zero coefficient.
template <class Reduction>
__global__ __launch_bounds__(1) void gmres_back_substitute_kernel(const Reduction* hessenberg,
    const Reduction* rhs, Reduction* coefficients, int leading, int columns) {
  for (int row = columns - 1; row >= 0; --row) {
    Reduction sum = Reduction(0);
    for (int column = row + 1; column < columns; ++column)
      sum += hessenberg[static_cast<std::size_t>(column) * leading + row] * coefficients[column];
    const Reduction diagonal = hessenberg[static_cast<std::size_t>(row) * leading + row];
    coefficients[row] = diagonal != Reduction(0) ? (rhs[row] - sum) / diagonal : Reduction(0);
  }
}

// x += sum of coefficient * basis column in Reduction arithmetic, one
// narrowing per node. Grid-stride with GmresLaunch blocks.
template <class Scalar, class Reduction, class Solution>
__global__ __launch_bounds__(gmres_reduction_threads) void gmres_update_solution_kernel(
    const Scalar* basis, const Reduction* coefficients, Solution* solution, unsigned count,
    int columns) {
  lpfx::device::for_each_grid_index(count, [&](const unsigned n) {
    Reduction value = config::convert<Reduction>(solution[n]);
    for (int column = 0; column < columns; ++column)
      value += coefficients[column] * config::convert<Reduction>(
          basis[static_cast<std::size_t>(column) * count + n]);
    solution[n] = config::convert<Solution>(value);
  });
}

} // namespace kernels

// Allocate before the application's one complete communication registration.
// The transport registry registers collective_operand() once, routed on the
// executor and rank count given here; keep this owner alive until communication closes. Every
// global reduction sends the fixed maximum batch; unused entries are zero.
// All large arrays, reductions and small dense algebra remain on device. Host
// reads are limited to two convergence/control scalars at each decision point.
// Scalar types the Krylov basis and the Arnoldi image. Reduction types the dot
// products, the Hessenberg state and the solution, right-hand side and
// residual, so the convergence test b - A*x is evaluated in Reduction. When
// Scalar is narrower, each restart cycle is one refinement step.
template <class Precision, class Backend> class Gmres {
  using Scalar = typename Precision::Compute::Electric;
  using Reduction = typename Precision::Compute::SolverReduction;
  static constexpr bool refined = !std::is_same_v<Scalar, Reduction>;
  using Executor = lpfx::device::Executor<Backend>;
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Allocation = lpfx::device::DeviceAllocation<Backend>;
  using PinnedAllocation = lpfx::device::PinnedHostAllocation<Backend>;

public:
  template <class Cells>
  Gmres(Cells cells, int restart,
        const lpfx::device::DeviceProperties& properties, Executor executor,
        std::uint32_t ranks)
      : executor_(executor), shape_(make_shape(cells)), restart_(checked_restart(restart)),
        count_(3U * static_cast<unsigned>(shape_.count)),
        reduction_blocks_(prepare(count_, properties, executor, ranks)), ranks_(ranks),
        vectors_(allocate(checked_bytes(gmres_vector_count(static_cast<std::size_t>(restart_), refined), count_,
                                        sizeof(Scalar)))),
        residual_vectors_(refined ? allocate(checked_bytes(gmres_residual_vector_count, count_, sizeof(Reduction)))
                                  : Allocation{}),
        scalars_(allocate(checked_bytes(gmres_batch_capacity(static_cast<std::size_t>(restart_)), 1, sizeof(Reduction)))),
        state_(allocate(checked_bytes(gmres_state_count(static_cast<std::size_t>(restart_)), 1, sizeof(Reduction)))),
        partials_(allocate(checked_bytes(gmres_batch_capacity(static_cast<std::size_t>(restart_)),
                                        reduction_blocks_, sizeof(Reduction)))) {
    lpfx::device::check_device_error<Backend>(
        PinnedAllocation::create(gmres_control_count * sizeof(Reduction), controls_),
        "TPIC GMRES: scalar readback allocation failed");
  }

  Gmres(const Gmres&) = delete;
  Gmres& operator=(const Gmres&) = delete;

  [[nodiscard]] auto collective_operand() {
    return lpfx::distributed::device_collective_in_place<Backend>(
        values(), gmres_batch_capacity(static_cast<std::uint64_t>(restart_)));
  }

  // apply_operator(input, output) consumes/writes the supplied persistent nodal
  // owners on this executor in Scalar arithmetic, apply_residual in Reduction
  // arithmetic; with one type they are the same apply. Each refreshes its own
  // halos and completes borrowed
  // communication operations before returning. Only interiors are packed;
  // operator input/output must be distinct and must not alias solution or rhs.
  // The external solution is committed only after true-residual convergence.
  // Collective admission, matching controls on all ranks and fatal error policy
  // are application-owned. There are no per-solve or per-iteration allocations.
  template <class SolutionOwner, class RhsOwner, class InputOwner, class OutputOwner,
            class Operator, class ResidualOperator, class Collective>
  [[nodiscard]] GmresResult<Reduction> host_blocking_solve(
      SolutionOwner& solution, const RhsOwner& rhs,
      InputOwner& input, OutputOwner& output, Operator&& apply_operator,
      ResidualOperator&& apply_residual, Collective& collective, int max_iterations,
      Reduction tolerance) {
    LPFX_PROFILE_SCOPE("TPIC::GMRES");
    validate_owner(solution);
    validate_owner(rhs);
    validate_owner(input);
    validate_owner(output);
    if (max_iterations <= 0 || !std::isfinite(tolerance) || tolerance < Reduction(0))
      throw std::invalid_argument("TPIC GMRES: invalid iteration count or tolerance");
    GmresResult<Reduction> result;
    try {
      pack(solution, x());
      pack(rhs, b());
      // The operator reads its prepared input: x() is unpacked into it, and each
      // normalization stores the new Krylov vector into it directly.
      const auto apply_prepared = [&] {
        apply_operator(input, output);
        pack(output, image());
      };
      const auto apply_solution = [&] {
        unpack(x(), input);
        apply_residual(input, output);
        pack(output, solution_image());
      };
      apply_solution();
      enqueue_residual_squared_norm<true>();
      const auto initial = host_blocking_reduce_sums(collective);
      result.residual = result.initial_residual = norm(initial[0]);
      result.rhs_norm = norm(initial[1]);
      const Reduction scale = result.rhs_norm == Reduction(0) ? Reduction(1) : result.rhs_norm;
      const Reduction absolute_tolerance = scale * tolerance;
      if (!std::isfinite(absolute_tolerance))
        throw std::overflow_error("TPIC GMRES: nonfinite absolute tolerance");
      const auto converged = [&] {
        result.relative_residual = result.residual / scale;
        result.converged = result.residual <= absolute_tolerance;
      };
      converged();
      while (!result.converged && result.iterations < max_iterations) {
        LPFX_PROFILE_SCOPE("TPIC::GMRES::Restart");
        zero(state_, gmres_state_count(static_cast<std::size_t>(restart_)) * sizeof(Reduction));
        normalize(r(), basis(0), detail::RestartRhsTail<Reduction>{least_squares_rhs()}, input);
        const int columns = std::min(restart_, max_iterations - result.iterations);
        // A Krylov type narrower than Reduction cannot reduce the residual of one
        // cycle below its own resolution: end the cycle there and refine.
        const Reduction cycle_tolerance = refined
            ? std::max(absolute_tolerance, result.residual * Reduction(16) *
                                               config::convert<Reduction>(std::numeric_limits<Scalar>::epsilon()))
            : absolute_tolerance;
        int completed = 0;
        bool breakdown = false;
        for (int column = 0; column < columns; ++column) {
          LPFX_PROFILE_SCOPE("TPIC::GMRES::Iteration");
          apply_prepared(); // Operator input holds basis(column).
          enqueue_projection_dots(column + 1);
          complete_allreduce(collective);
          auto* h_column = hessenberg() + static_cast<std::size_t>(column) * (restart_ + 1);
          lpfx::device::launch_or_throw<kernels::gmres_orthogonalize_squared_norm_kernel<Scalar, Reduction>>(
              "TPIC GMRES: kernel launch failed", executor_,
              lpfx::device::LaunchGrid{reduction_blocks_, threads}, basis(0), values(), image(), h_column,
              partials(), count_, column + 1);
          enqueue_batch_partial_reduction(1); // Stream order keeps the projection's dot values alive.
          complete_allreduce(collective);
          // Also stores basis(column + 1) into the operator input; a loop exit
          // overwrites it with x() before the next application.
          normalize(image(), basis(column + 1), detail::ArnoldiLeastSquaresUpdate<Reduction>{
              h_column, cosine(), sine(), least_squares_rhs(), status(), column}, input);
          const auto controls = host_blocking_read_controls(status());
          const GmresIterationStatus<Reduction> estimate{controls[0], controls[1]};
          if (!std::isfinite(estimate.residual_estimate) || !std::isfinite(estimate.breakdown))
            throw std::overflow_error("TPIC GMRES: nonfinite Arnoldi recurrence");
          completed = column + 1;
          breakdown = estimate.breakdown != Reduction(0);
          if (estimate.residual_estimate <= cycle_tolerance || breakdown) break;
        }
        lpfx::device::launch_or_throw<kernels::gmres_back_substitute_kernel<Reduction>>(
            "TPIC GMRES: kernel launch failed", executor_, lpfx::device::LaunchGrid{1, 1},
            hessenberg(), least_squares_rhs(), coefficients(), restart_ + 1, completed);
        lpfx::device::launch_or_throw<GmresLaunch, kernels::gmres_update_solution_kernel<Scalar, Reduction, Reduction>>(
            "TPIC GMRES: kernel launch failed", executor_, count_, basis(0), coefficients(), x(), count_, completed);
        result.iterations += completed;
        // Always verify b - A*x after an accepted update, including an early
        // estimated convergence and the final allowed iteration.
        apply_solution();
        enqueue_residual_squared_norm<false>();
        result.residual = norm(host_blocking_reduce_sums(collective)[0]);
        converged();
        if (breakdown && !result.converged) break;
      }
      if (result.converged) unpack(x(), solution);
      return result;
    } catch (...) {
      // Every persistent allocation remains alive until accepted work drains.
      if (!Traits::succeeded(Traits::synchronize_stream(executor_.stream()))) std::abort();
      throw;
    }
  }

private:
  static constexpr unsigned threads = gmres_reduction_threads;
  template <class Cells> static data::GridShape make_shape(Cells cells) {
    data::GridShape shape;
    if (data::try_make_grid_shape(cells, 1, {}, shape) != lpfx::ErrorCode::success ||
        shape.count > std::numeric_limits<int>::max() / 3)
      throw std::invalid_argument("TPIC GMRES: compact nodal vector exceeds signed index capacity");
    return shape;
  }
  static int checked_restart(int restart) {
    // The partial reducer launches one block per collective entry (max(restart, 2)).
    if (restart <= 0 || restart > int(lpfx::device::kernel_grid_block_limit))
      throw std::invalid_argument("TPIC GMRES: invalid restart length");
    return restart;
  }
  static std::size_t checked_bytes(std::size_t count, std::size_t width, std::size_t element) {
    constexpr auto limit = std::numeric_limits<std::size_t>::max();
    if (width > limit / element || count > limit / (width * element))
      throw std::overflow_error("TPIC GMRES: allocation size overflow");
    return count * width * element;
  }
  static_assert(sizeof(std::size_t) >= 8); // checked_restart keeps the state count within size_t.
  static unsigned prepare(unsigned count,
      const lpfx::device::DeviceProperties& properties, Executor executor,
      std::uint32_t ranks) {
    if (!executor.bound() || ranks == 0)
      throw std::invalid_argument("TPIC GMRES: invalid device reduction admission");
    return gmres_reduction_blocks(count, properties);
  }
  static Allocation allocate(std::size_t bytes) {
    Allocation result;
    lpfx::device::check_device_error<Backend>(Allocation::create(bytes, result),
                                              "TPIC GMRES: device allocation failed");
    return result;
  }
  Scalar* basis(int column) {
    return reinterpret_cast<Scalar*>(vectors_.raw_base()) + static_cast<std::size_t>(column) * count_;
  }
  Scalar* image() { return basis(restart_ + (refined ? 1 : 4)); }
  // The solution, right-hand side, residual and image of the solution. With
  // one type they are the four vectors after the basis, the last one image().
  Reduction* residual_vector(int index) {
    if constexpr (refined)
      return reinterpret_cast<Reduction*>(residual_vectors_.raw_base()) + static_cast<std::size_t>(index) * count_;
    else
      return basis(restart_ + 1 + index);
  }
  Reduction* x() { return residual_vector(0); }
  Reduction* b() { return residual_vector(1); }
  Reduction* r() { return residual_vector(2); }
  Reduction* solution_image() { return residual_vector(3); }
  Reduction* values() { return reinterpret_cast<Reduction*>(scalars_.raw_base()); }
  Reduction* partials() { return reinterpret_cast<Reduction*>(partials_.raw_base()); }
  Reduction* hessenberg() { return reinterpret_cast<Reduction*>(state_.raw_base()); }
  Reduction* cosine() { return hessenberg() + static_cast<std::size_t>(restart_) * (restart_ + 1); }
  Reduction* sine() { return cosine() + restart_; }
  Reduction* least_squares_rhs() { return sine() + restart_; }
  Reduction* coefficients() { return least_squares_rhs() + restart_ + 1; }
  GmresIterationStatus<Reduction>* status() {
    return reinterpret_cast<GmresIterationStatus<Reduction>*>(coefficients() + restart_);
  }

  template <class Owner> void validate_owner(const Owner& owner) const {
    static_assert(Owner::component_count == 3);
    static_assert(std::is_same_v<typename Owner::space_type::backend_type, Backend>);
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      static_assert((std::is_same_v<typename decltype(owner.view().template component<I>())::location_type,
                                   lpfx::Node> && ...));
      static_assert((std::is_same_v<typename decltype(owner.view().template component<I>())::value_type,
                                   SolverStorage<Precision>> && ...));
    }(std::make_index_sequence<3>{});
    if (!owner.initialized()) throw std::invalid_argument("TPIC GMRES: uninitialized nodal owner");
    const auto shape = make_shape(owner.cells());
    if (shape.nx != shape_.nx || shape.ny != shape_.ny || shape.nz != shape_.nz)
      throw std::invalid_argument("TPIC GMRES: incompatible nodal owner shape");
  }
  template <class Owner, class Vector> void pack(const Owner& owner, Vector* vector) {
    lpfx::device::launch_or_throw<GmresLaunch,
        kernels::gmres_field_vector_transfer_kernel<TransferDirection::pack, Vector,
            decltype(owner.view())>>(
        "TPIC GMRES: kernel launch failed", executor_, shape_.count, owner.view(), vector, shape_);
  }
  template <class Vector, class Owner> void unpack(Vector* vector, Owner& owner) {
    lpfx::device::launch_or_throw<GmresLaunch,
        kernels::gmres_field_vector_transfer_kernel<TransferDirection::unpack, Vector,
            decltype(owner.view())>>(
        "TPIC GMRES: kernel launch failed", executor_, shape_.count, owner.view(), vector, shape_);
  }
  template <class Source, class Tail, class Owner> void normalize(Source* source, Scalar* target, Tail tail,
                                                                   Owner& input) {
    using Store = detail::OperatorInputStore<decltype(input.view())>;
    lpfx::device::launch_or_throw<kernels::gmres_normalize_kernel<Source, Scalar, Reduction, Tail, Store>>(
        "TPIC GMRES: kernel launch failed", executor_, lpfx::device::LaunchGrid{reduction_blocks_, threads},
        source, values(), target, count_, tail, Store{input.view(), shape_});
  }
  static Reduction norm(Reduction norm2) {
    if (!std::isfinite(norm2) || norm2 < Reduction(0))
      throw std::overflow_error("TPIC GMRES: nonfinite or negative global squared norm");
    return config::convert<Reduction>(std::sqrt(static_cast<double>(norm2)));
  }
  void zero(Allocation& allocation, std::size_t bytes) {
    lpfx::device::check_device_error<Backend>(
        Traits::memset_device_async(allocation.raw_base(), 0, bytes, executor_.stream()),
        "TPIC GMRES: device zero submission failed");
  }
  // Stream: executor_, after the block partials. One block per entry of the
  // fixed collective batch, entries from columns on zero; the caller then
  // allreduces the batch.
  void enqueue_batch_partial_reduction(int columns) {
    lpfx::device::launch_or_throw<device::kernels::reduce_block_partials_kernel<Reduction>>(
        "TPIC GMRES: kernel launch failed", executor_,
        lpfx::device::LaunchGrid{gmres_batch_capacity(static_cast<unsigned>(restart_)), threads},
        partials(), values(), reduction_blocks_, reduction_blocks_, static_cast<unsigned>(columns));
  }
  // Stream: executor_, after the operator image. Reduces the dots of the image
  // with the first columns basis vectors into the batch.
  void enqueue_projection_dots(int columns) {
    using Launch = lpfx::device::KernelLaunch<threads, gmres_dot_tile>;
    const dim3 grid(reduction_blocks_, Launch::blocks(columns));
    lpfx::device::launch_or_throw<kernels::gmres_dot_batch_kernel<Scalar, Reduction>>(
        "TPIC GMRES: kernel launch failed", executor_, lpfx::device::LaunchGrid{grid, threads},
        basis(0), image(), partials(), count_, columns);
    enqueue_batch_partial_reduction(columns);
  }
  // Stream: executor_, after the operator image of x. Writes r = b - image
  // and reduces its squared norm (and with RhsNorm that of b) into the batch.
  template <bool RhsNorm> void enqueue_residual_squared_norm() {
    lpfx::device::launch_or_throw<kernels::gmres_residual_squared_norm_kernel<RhsNorm, Reduction, Reduction>>(
        "TPIC GMRES: kernel launch failed", executor_, lpfx::device::LaunchGrid{reduction_blocks_, threads},
        b(), solution_image(), r(), partials(), count_);
    enqueue_batch_partial_reduction(RhsNorm ? 2 : 1);
  }
  // MPI blocks for the reduction and enqueues the copy of the sums to the
  // device; NCCL/RCCL return at enqueue.
  template <class Collective> void complete_allreduce(Collective& collective) {
    // An in-place sum across one rank is already complete on this executor.
    // Avoid the device-aware MPI pre/post synchronization and epoch lifecycle.
    if (ranks_ == 1) return;
    const auto accepted = collective.begin();
    lpfx::check_acceptance(accepted, "TPIC GMRES: global scalar batch rejected");
    lpfx::check_error(collective.wait(), "TPIC GMRES: LPFx operation failed");
    lpfx::check_error(collective.conclude(), "TPIC GMRES: LPFx operation failed");
  }
  // Completes the batch reduction and returns its first two sums: the staged
  // route holds them in its pinned host image, so only one rank or a
  // stream-ordered route reads them back.
  template <class Collective> std::array<Reduction, 2> host_blocking_reduce_sums(Collective& collective) {
    complete_allreduce(collective);
    if constexpr (requires { collective.staged_output(); }) {
      if (ranks_ > 1) {
        const auto sums = collective.staged_output();
        lpfx::check_error(sums, "TPIC GMRES: staged reduction unavailable");
        return {sums.value()[0], sums.value()[1]};
      }
    }
    return host_blocking_read_controls(values());
  }
  std::array<Reduction, 2> host_blocking_read_controls(const void* source) {
    auto* result = reinterpret_cast<Reduction*>(controls_.raw_base());
    const auto copied = Traits::copy_device_to_host_async(result, source, 2 * sizeof(Reduction),
                                                          executor_.stream());
    if (!Traits::succeeded(copied)) {
      if (!Traits::succeeded(Traits::synchronize_stream(executor_.stream()))) std::abort();
      lpfx::device::check_device_error<Backend>(copied,
                                                "TPIC GMRES: scalar readback submission failed");
    }
    if (!Traits::succeeded(Traits::synchronize_stream(executor_.stream()))) std::abort();
    return {result[0], result[1]};
  }

  Executor executor_;
  data::GridShape shape_;
  int restart_;
  unsigned count_;
  unsigned reduction_blocks_;
  std::uint32_t ranks_;
  Allocation vectors_, residual_vectors_, scalars_, state_, partials_;
  PinnedAllocation controls_;
};

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_GMRES_GMRES_DEVICE_HPP
