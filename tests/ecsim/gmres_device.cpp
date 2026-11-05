// Batched Arnoldi reductions and fused vector/norm operations are compared
// against independent host arithmetic. Small grids force tails/grid strides;
// row counts cover every four-vector tile boundary and the normal restart.
#include <tpic/ecsim/gmres/gmres_device.hpp>
#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <tpic/config/precision.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
namespace kernels = tpic::ecsim::kernels;
constexpr unsigned threads = tpic::ecsim::gmres_reduction_threads;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(Traits::error_type error) { require(Traits::succeeded(error), "GMRES test backend failure"); }

template <class T> struct Buffer {
  lpfx::device::DeviceAllocation<Backend> allocation;
  std::size_t count;
  explicit Buffer(std::size_t size) : count(size) {
    check(decltype(allocation)::create(size * sizeof(T), allocation));
  }
  T* data() { return reinterpret_cast<T*>(allocation.raw_base()); }
  void upload(const std::vector<T>& host, Executor executor) {
    require(host.size() == count, "GMRES upload size mismatch");
    check(Traits::copy_host_to_device_async(data(), host.data(), count * sizeof(T), executor.stream()));
  }
  std::vector<T> download(Executor executor) {
    std::vector<T> host(count);
    check(Traits::copy_device_to_host_async(host.data(), data(), count * sizeof(T), executor.stream()));
    check(Traits::synchronize_stream(executor.stream()));
    return host;
  }
};

void near(double actual, long double expected, const char* message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <=
      2e-12L * std::max(1.0L, std::abs(expected)), message);
}

template <class Scalar> void run_case(Executor executor, unsigned count, int columns, unsigned blocks) {
  constexpr int capacity = 32;
  std::vector<Scalar> basis(static_cast<std::size_t>(columns) * count), image(count), rhs(count);
  for (unsigned n = 0; n < count; ++n) {
    image[n] = Scalar(0.2 + double(n % 17) / 19 - double(n % 13) / 11);
    rhs[n] = Scalar(-0.3 + double(n % 23) / 31);
    for (int row = 0; row < columns; ++row)
      basis[static_cast<std::size_t>(row) * count + n] =
          Scalar((0.1 + double((n + row) % 7) / 9 - double((n * 3 + row) % 11) / 13) /
                 std::sqrt(double(count)));
  }
  Buffer<Scalar> dbasis(basis.size()), dimage(count), drhs(count), dresidual(count);
  Buffer<double> partials(std::size_t(capacity) * blocks), values(capacity), h(capacity);
  dbasis.upload(basis, executor); dimage.upload(image, executor); drhs.upload(rhs, executor);
  // Two y blocks force a second row tile at restart=30, while x=1 forces
  // multiple vector iterations. CUDA/HIP native warp widths both participate.
  lpfx::device::launch_or_throw<kernels::gmres_dot_batch_kernel<Scalar, double>>("GMRES test launch", executor,
      lpfx::device::LaunchGrid{dim3(blocks, std::min(2, (columns + 3) / 4)), threads},
      dbasis.data(), dimage.data(), partials.data(), count, columns);
  lpfx::device::launch_or_throw<tpic::device::kernels::reduce_block_partials_kernel<double>>(
      "GMRES test launch", executor, lpfx::device::LaunchGrid{unsigned(capacity), threads}, partials.data(), values.data(),
      blocks, blocks, unsigned(columns));
  check(Traits::take_last_launch_error());
  const auto dots = values.download(executor);
  for (int row = 0; row < capacity; ++row) {
    long double expected{};
    if (row < columns)
      for (unsigned n = 0; n < count; ++n)
        expected += static_cast<long double>(basis[static_cast<std::size_t>(row) * count + n]) * image[n];
    near(dots[row], expected, "GMRES batched projection dot mismatch");
    if (row >= columns) require(dots[row] == 0, "GMRES unused collective entry is not zero");
  }
  lpfx::device::launch_or_throw<kernels::gmres_orthogonalize_squared_norm_kernel<Scalar, double>>(
      "GMRES test launch", executor, lpfx::device::LaunchGrid{blocks, threads},
      dbasis.data(), values.data(), dimage.data(), h.data(), partials.data(), count, columns);
  lpfx::device::launch_or_throw<tpic::device::kernels::reduce_block_partials_kernel<double>>(
      "GMRES test launch", executor, lpfx::device::LaunchGrid{unsigned(capacity), threads}, partials.data(), values.data(),
      blocks, blocks, unsigned(1));
  check(Traits::take_last_launch_error());
  const auto projected = dimage.download(executor);
  const auto hessenberg = h.download(executor);
  const auto norm = values.download(executor);
  long double projected_norm{};
  for (unsigned n = 0; n < count; ++n) {
    double expected = double(image[n]);
    for (int row = 0; row < columns; ++row)
      expected -= dots[row] * double(basis[static_cast<std::size_t>(row) * count + n]);
    const double tolerance = std::is_same_v<Scalar, float> ? 2e-7 : 2e-12;
    require(std::abs(double(projected[n]) - double(Scalar(expected))) <= tolerance,
            "GMRES projected vector mismatch");
    projected_norm += static_cast<long double>(projected[n]) * projected[n];
  }
  near(norm[0], projected_norm, "GMRES fused projection norm mismatch");
  for (int row = 0; row < columns; ++row)
    require(hessenberg[row] == dots[row], "GMRES Hessenberg dots overwritten too early");
  for (const bool include_rhs : {false, true}) {
    if (include_rhs)
      lpfx::device::launch_or_throw<kernels::gmres_residual_squared_norm_kernel<true, Scalar, double>>(
          "GMRES test launch", executor, lpfx::device::LaunchGrid{blocks, threads}, drhs.data(), dimage.data(), dresidual.data(), partials.data(), count);
    else
      lpfx::device::launch_or_throw<kernels::gmres_residual_squared_norm_kernel<false, Scalar, double>>(
          "GMRES test launch", executor, lpfx::device::LaunchGrid{blocks, threads}, drhs.data(), dimage.data(), dresidual.data(), partials.data(), count);
    lpfx::device::launch_or_throw<tpic::device::kernels::reduce_block_partials_kernel<double>>(
      "GMRES test launch", executor, lpfx::device::LaunchGrid{unsigned(capacity), threads}, partials.data(), values.data(),
      blocks, blocks, unsigned(include_rhs ? 2 : 1));
    check(Traits::take_last_launch_error());
    const auto residual = dresidual.download(executor);
    const auto norms = values.download(executor);
    long double residual_norm{}, rhs_norm{};
    for (unsigned n = 0; n < count; ++n) {
      require(residual[n] == Scalar(rhs[n] - projected[n]), "GMRES residual mismatch");
      residual_norm += static_cast<long double>(residual[n]) * residual[n];
      rhs_norm += static_cast<long double>(rhs[n]) * rhs[n];
    }
    near(norms[0], residual_norm, "GMRES fused residual norm mismatch");
    near(norms[1], include_rhs ? rhs_norm : 0, "GMRES fused rhs norm mismatch");
  }
  // Invalid arithmetic must still reach the solver's nonfinite check.
  image[0] = std::numeric_limits<Scalar>::quiet_NaN();
  dimage.upload(image, executor);
  lpfx::device::launch_or_throw<kernels::gmres_residual_squared_norm_kernel<false, Scalar, double>>(
      "GMRES test launch", executor, lpfx::device::LaunchGrid{blocks, threads}, drhs.data(), dimage.data(), dresidual.data(), partials.data(), count);
  lpfx::device::launch_or_throw<tpic::device::kernels::reduce_block_partials_kernel<double>>(
      "GMRES test launch", executor, lpfx::device::LaunchGrid{unsigned(capacity), threads}, partials.data(), values.data(),
      blocks, blocks, unsigned(1));
  check(Traits::take_last_launch_error());
  require(std::isnan(values.download(executor)[0]), "GMRES nonfinite residual was hidden");
}

bool unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
         error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}

struct TestCells {
  constexpr int x() const { return 2; }
  constexpr int y() const { return 1; }
  constexpr int z() const { return 1; }
};
struct TestComponent {
  using location_type = lpfx::Node;
  using value_type = double;
  double* values;
  __host__ __device__ double& operator()(int i, int j, int k) const {
    return values[i + 3 * j + 6 * k];
  }
};
struct TestView {
  double* values;
  template <unsigned Component> __host__ __device__ TestComponent component() const {
    return {values + 12 * Component};
  }
};
struct TestOwner {
  static constexpr unsigned component_count = 3;
  using space_type = lpfx::DeviceSpace<Backend>;
  Buffer<double> memory{36};
  bool initialized() const { return true; }
  TestCells cells() const { return {}; }
  TestView view() const { return {const_cast<TestOwner*>(this)->memory.data()}; }
};
struct UnusedCollective {
  lpfx::Status begin() { throw std::runtime_error("GMRES rank-one collective was submitted"); }
  lpfx::Status wait() { throw std::runtime_error("GMRES rank-one collective was waited"); }
  lpfx::Status conclude() { throw std::runtime_error("GMRES rank-one collective was concluded"); }
};
__global__ void apply_diagonal_operator_kernel(const double* input, double* output, bool identity) {
  const unsigned n = threadIdx.x;
  if (n < 36) output[n] = input[n] * (identity ? 1.0 : 1.0 + 0.15 * (n % 5));
}

// The same operator in single arithmetic: the Arnoldi apply of a solver whose
// Krylov type is narrower than its reduction type.
__global__ void apply_diagonal_operator_single_kernel(const double* input, double* output, bool identity) {
  const unsigned n = threadIdx.x;
  if (n < 36)
    output[n] = double(float(input[n]) * (identity ? 1.0f : 1.0f + 0.15f * float(n % 5)));
}

// Compute selects the Krylov type (Compute::Electric) and the reduction type.
// With a narrower Krylov type every restart is one refinement step, and the
// residual apply stays in double.
template <class Compute> void run_solver(Executor executor, const lpfx::device::DeviceProperties& properties) {
  constexpr bool refined =
      !std::is_same_v<typename Compute::Electric, typename Compute::SolverReduction>;
  TestOwner solution, rhs, input, output;
  std::vector<double> exact(36), right(36), zero(36, 0);
  for (unsigned n = 0; n < 36; ++n) {
    exact[n] = 0.2 + double(n % 7) / 9;
    right[n] = exact[n] * (1 + 0.15 * (n % 5));
  }
  solution.memory.upload(zero, executor);
  rhs.memory.upload(right, executor);
  tpic::ecsim::Gmres<tpic::config::Precision<tpic::config::StorageDouble, Compute>, Backend> solver(
      TestCells{}, 2, properties, executor, 1);
  UnusedCollective collective;
  bool identity = false;
  const auto apply_residual = [&](TestOwner& from, TestOwner& to) {
    apply_diagonal_operator_kernel<<<1, 64, 0, executor.stream()>>>(
        from.memory.data(), to.memory.data(), identity);
    check(Traits::take_last_launch_error());
  };
  const auto apply = [&](TestOwner& from, TestOwner& to) {
    if (!refined) return apply_residual(from, to);
    apply_diagonal_operator_single_kernel<<<1, 64, 0, executor.stream()>>>(
        from.memory.data(), to.memory.data(), identity);
    check(Traits::take_last_launch_error());
  };
  const auto limited = solver.host_blocking_solve(solution, rhs, input, output, apply, apply_residual,
      collective, 1, 1e-14);
  require(!limited.converged && limited.iterations == 1, "GMRES iteration limit was not honored");
  require(solution.memory.download(executor) == zero, "GMRES committed an unconverged solution");
  const auto solved = solver.host_blocking_solve(solution, rhs, input, output, apply, apply_residual,
      collective, 80, 1e-11);
  require(solved.converged && solved.iterations > 2, "GMRES restarted solve did not converge");
  const auto actual = solution.memory.download(executor);
  long double norm2{}, rhs_norm2{};
  for (unsigned n = 0; n < 36; ++n) {
    const long double residual = right[n] - (1 + 0.15 * (n % 5)) * actual[n];
    norm2 += residual * residual;
    rhs_norm2 += static_cast<long double>(right[n]) * right[n];
    require(std::abs(actual[n] - exact[n]) <= 2e-10, "GMRES diagonal solution mismatch");
  }
  require(std::sqrt(norm2 / rhs_norm2) <= 1e-11L, "GMRES accepted without true-residual convergence");
  // Reusing all persistent buffers must retain initial convergence and zero rhs.
  const auto already = solver.host_blocking_solve(solution, rhs, input, output, apply, apply_residual,
      collective, 80, 1e-11);
  require(already.converged && already.iterations == 0, "GMRES repeated solved state was lost");
  solution.memory.upload(zero, executor);
  rhs.memory.upload(zero, executor);
  const auto empty = solver.host_blocking_solve(solution, rhs, input, output, apply, apply_residual,
      collective, 80, 1e-11);
  require(empty.converged && empty.iterations == 0 && empty.rhs_norm == 0,
          "GMRES zero right-hand side changed");
  // An identity operator closes the Krylov space in one step; its update still
  // passes the explicitly recomputed residual before being committed.
  identity = true;
  rhs.memory.upload(exact, executor);
  const auto invariant = solver.host_blocking_solve(solution, rhs, input, output, apply, apply_residual,
      collective, 80, 1e-11);
  // A narrower Krylov vector carries the direction to its own resolution only:
  // refinement needs further one-step cycles.
  require(invariant.converged &&
              (refined ? invariant.iterations >= 1 && invariant.iterations <= 3 : invariant.iterations == 1),
          "GMRES invariant subspace solve changed");
}
} // namespace

int main() {
  lpfx::device::DeviceProperties properties;
  auto error = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(error)) {
    if (unavailable(error)) {
      std::cout << "TPIC GMRES reduction tests skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto stream = Traits::null_stream;
  error = Traits::create_stream(stream);
  if (!Traits::succeeded(error)) return unavailable(error) ? 77 : 1;
  int status = 0;
  try {
    const Executor executor{stream};
    for (unsigned count : {1U, 31U, 257U, 4099U})
      for (int columns : {1, 4, 5, 13, 30})
        for (unsigned blocks : {1U, 5U}) {
          run_case<double>(executor, count, columns, blocks);
          run_case<float>(executor, count, columns, blocks);
        }
    run_solver<tpic::config::ComputeDouble>(executor, properties);
    run_solver<tpic::config::ComputeSingleParticlesMomentsFields>(executor, properties);
    std::cout << "TPIC GMRES reduction tests passed\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    check(Traits::synchronize_stream(stream));
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(stream))) status = 1;
  return status;
}
