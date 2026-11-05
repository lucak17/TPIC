// Differential coverage for cell-owned mass accumulation. The independent
// unsorted kernel is the scalar oracle. Matrix deposition additionally uses independent
// host FP64 cell sums, including its deliberately later storage conversion.
// Artificial bins isolate validation, tails and strides from sorting.
#include <tpic/device/block_threads.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp>
#include "../support/test_builders.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
         error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}

template<class T> struct Buffer {
  lpfx::device::DeviceAllocation<Backend> allocation;
  std::size_t count;
  explicit Buffer(std::size_t n) : count(n) {
    require(Traits::succeeded(decltype(allocation)::create(n * sizeof(T), allocation)),
            "sorted mass: allocation failed");
  }
  T* data() { return reinterpret_cast<T*>(allocation.raw_base()); }
  void upload(const std::vector<T>& values, Executor executor) {
    require(values.size() == count && Traits::succeeded(Traits::copy_host_to_device_async(
        data(), values.data(), count * sizeof(T), executor.stream())),
        "sorted mass: upload failed");
  }
  std::vector<T> download(Executor executor) {
    std::vector<T> result(count);
    require(Traits::succeeded(Traits::copy_device_to_host_async(
        result.data(), data(), count * sizeof(T), executor.stream())) &&
        Traits::succeeded(Traits::synchronize_stream(executor.stream())),
        "sorted mass: download failed");
    return result;
  }
  void clear(Executor executor) {
    require(Traits::succeeded(Traits::memset_device_async(
        data(), 0, count * sizeof(T), executor.stream())), "sorted mass: clear failed");
  }
};

template<class Precision> struct Particles {
  const typename Precision::Storage::Position *rx, *ry, *rz;
  const typename Precision::Storage::Weight* charge;
  const std::int32_t* cell;
  template<class Attribute> __host__ __device__ auto get() const {
    using S = tpic::data::ParticleSchema<Precision>;
    if constexpr (std::is_same_v<Attribute, typename S::Rx>) return rx;
    else if constexpr (std::is_same_v<Attribute, typename S::Ry>) return ry;
    else if constexpr (std::is_same_v<Attribute, typename S::Rz>) return rz;
    else if constexpr (std::is_same_v<Attribute, typename S::Charge>) return charge;
    else { static_assert(std::is_same_v<Attribute, typename S::Cell>); return cell; }
  }
};

template<class T> struct FieldComponent {
  using location_type = lpfx::Node;
  const T* values;
  int stride_y, stride_z;
  __host__ __device__ T operator()(int i, int j, int k) const {
    return values[i + stride_y * j + stride_z * k];
  }
};

template<class Precision> struct Magnetic {
  FieldComponent<typename Precision::Storage::Magnetic> component[3];
  template<class Attribute> __host__ __device__ auto get() const {
    using F = tpic::data::FieldSchema<Precision>;
    if constexpr (std::is_same_v<Attribute, typename F::Bnx>) return component[0];
    else if constexpr (std::is_same_v<Attribute, typename F::Bny>) return component[1];
    else { static_assert(std::is_same_v<Attribute, typename F::Bnz>); return component[2]; }
  }
};

enum class Case { ordinary, empty, zero_charge, suffix, key, fraction, endpoint, charge, field,
                  response_overflow, large_finite, accumulated_overflow, storage_overflow,
                  narrowing_cancellation, narrowing_rounding, metadata, metadata_count,
                  metadata_first, metadata_last, producer, halo };
enum class Implementation { scalar, matrix_core };

template<class Precision>
bool matrix_core_available(int device) {
  return tpic::ecsim::matrix_core_mass_kernel_available<
      Precision, Particles<Precision>, Magnetic<Precision>>(device);
}

template<class Precision>
std::vector<typename Precision::Storage::Mass> host_matrix_core_mass(
    const std::vector<typename Precision::Storage::Position>& rx,
    const std::vector<typename Precision::Storage::Position>& ry,
    const std::vector<typename Precision::Storage::Position>& rz,
    const std::vector<typename Precision::Storage::Weight>& charge,
    const std::vector<std::int32_t>& keys,
    const std::vector<typename Precision::Storage::Magnetic>& bx,
    const std::vector<typename Precision::Storage::Magnetic>& by,
    const std::vector<typename Precision::Storage::Magnetic>& bz,
    const std::vector<unsigned>& counts, const std::vector<unsigned>& offsets,
    tpic::data::ParticleCellMap cells, tpic::data::MassMatrixLayout layout,
    double beta, unsigned tail_count) {
  using M = typename Precision::Storage::Mass;
  std::vector<M> result(layout.storage_size());
  // Deliberately avoid the device gather/response helpers. Every cell keeps a
  // complete FP64 8x8x9 tensor until its canonical pairs are cast into Mass.
  auto accumulate = [&](unsigned index, auto& tensor) {
    const auto cell = cells.decode(keys[index]);
    double weights[8], field[3]{};
    for (int corner = 0; corner < 8; ++corner) {
      weights[corner] = ((corner & 1) ? double(rx[index]) : 1.0 - rx[index]) *
          ((corner & 2) ? double(ry[index]) : 1.0 - ry[index]) *
          ((corner & 4) ? double(rz[index]) : 1.0 - rz[index]);
      const int i = cell.i + (corner & 1), j = cell.j + ((corner >> 1) & 1),
                k = cell.k + ((corner >> 2) & 1);
      field[0] += weights[corner] * bx[i + 11*j + 55*k];
      field[1] += weights[corner] * by[i + 13*j + 78*k];
      field[2] += weights[corner] * bz[i + 17*j + 119*k];
    }
    const double x = beta * field[0], y = beta * field[1], z = beta * field[2];
    const double inverse = 1.0 / (1.0 + (x*x + y*y + z*z));
    const double alpha[9] = {
        (1.0 + x*x)*inverse, x*y*inverse + z*inverse, x*z*inverse - y*inverse,
        x*y*inverse - z*inverse, (1.0 + y*y)*inverse, y*z*inverse + x*inverse,
        x*z*inverse + y*inverse, y*z*inverse - x*inverse, (1.0 + z*z)*inverse};
    const double scale = beta * charge[index];
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b)
        for (int component = 0; component < 9; ++component)
          tensor[(a*8 + b)*9 + component] +=
              (alpha[component] * scale) * (weights[a] * weights[b]);
  };
  auto flush = [&](tpic::data::GridIndex cell, const auto& tensor) {
    for (int a = 0; a < 8; ++a) {
      const auto vertex = layout.vertex_index(cell.i + (a & 1),
          cell.j + ((a >> 1) & 1), cell.k + ((a >> 2) & 1));
      for (int b = 0; b < 8; ++b) {
        const int stencil = tpic::ecsim::mass_pair_stencil(a, b);
        if (stencil < 0) continue;
        for (int component = 0; component < 9; ++component)
          result[layout.entry_index(vertex, stencil) + component] +=
              static_cast<M>(tensor[(a*8 + b)*9 + component]);
      }
    }
  };
  for (unsigned bin = 0; bin < counts.size(); ++bin) {
    if (counts[bin] == 0) continue;
    std::array<double, 8*8*9> tensor{};
    for (unsigned offset = 0; offset < counts[bin]; ++offset)
      accumulate(offsets[bin] + offset, tensor);
    flush(cells.decode(static_cast<int>(bin)), tensor);
  }
  // The partial-sort tail remains a scalar per-particle deposition, including
  // the conversion before its atomic add in mixed storage.
  for (unsigned index = offsets.back(); index < offsets.back() + tail_count; ++index) {
    std::array<double, 8*8*9> tensor{};
    accumulate(index, tensor);
    flush(cells.decode(keys[index]), tensor);
  }
  return result;
}

template<class Precision, Implementation Impl = Implementation::scalar>
void run_case(Executor executor, Case test) {
  using P = typename Precision::Storage::Position;
  using Q = typename Precision::Storage::Weight;
  using B = typename Precision::Storage::Magnetic;
  using M = typename Precision::Storage::Mass;
  constexpr double beta = -0.07;
  tpic::config::PreparedDomain domain{};
  domain.local_cells = {4, 2, 2}; // Thin periodic-run geometry, adjacent cells.
  tpic::test::fill_domain_counts(domain);
  const auto cells = tpic::data::prepare_particle_cell_map(domain);
  const auto layout = tpic::data::prepare_mass_matrix_layout(domain);
  const std::array<unsigned, 16> occupancy{
      0, 1, 3, 4, 5, 31, 32, 33, 65, 129, 257, 64, 127, 128, 130, 19};
  std::vector<P> rx, ry, rz;
  std::vector<Q> charge;
  std::vector<std::int32_t> keys;
  std::vector<unsigned> counts(cells.count), offsets(cells.count + 1);
  for (unsigned compact = 0; compact < occupancy.size(); ++compact) {
    const int i = compact % 4, j = (compact / 4) % 2, k = compact / 8;
    const int key = cells.encode(i, j, k);
    counts[key] = occupancy[compact];
    for (unsigned n = 0; n < occupancy[compact]; ++n) {
      rx.push_back(P((n % 17) / 17.0));
      ry.push_back(P((n % 13) / 13.0));
      rz.push_back(P((n % 11) / 11.0));
      charge.push_back(Q((n % 7 == 0 ? -0.3 : 0.8) * (1 + 0.05 * compact)));
      keys.push_back(key);
    }
  }
  if (test == Case::halo) {
    --counts[keys[0]];
    keys[0] = 0;
    ++counts[0];
  }
  if (test == Case::suffix) counts[keys.back()] -= 19;
  if (test == Case::empty) std::fill(counts.begin(), counts.end(), 0U);
  for (int bin = 0; bin < cells.count; ++bin) offsets[bin + 1] = offsets[bin] + counts[bin];
  const unsigned prefix = offsets.back(), total = static_cast<unsigned>(keys.size());
  if (test == Case::key) keys[1] = -1;
  if (test == Case::fraction) rx[1] = std::numeric_limits<P>::quiet_NaN();
  if (test == Case::endpoint) rx[1] = P(1);
  if (test == Case::zero_charge) std::fill(charge.begin(), charge.end(), Q(0));
  if (test == Case::response_overflow) std::fill(charge.begin(), charge.end(), Q(0));
  if (test == Case::charge) charge[1] = std::numeric_limits<Q>::infinity();
  if (test == Case::storage_overflow) charge[1] = Q(1e300);
  double deposition_beta = beta;
  if (test == Case::response_overflow) deposition_beta = std::numeric_limits<double>::max();
  if (test == Case::large_finite) deposition_beta = 1.0;
  const bool corner_fixture = test == Case::large_finite || test == Case::accumulated_overflow ||
      test == Case::narrowing_cancellation || test == Case::narrowing_rounding;
  if (corner_fixture) {
    // At zero B and corner zero, diagonal contributions are beta * charge.
    // large_finite exercises the largest finite scale with an identity response.
    // Dense accumulated_overflow cells exceed Mass's range only in the cell sum.
    std::fill(rx.begin(), rx.end(), P(0));
    std::fill(ry.begin(), ry.end(), P(0));
    std::fill(rz.begin(), rz.end(), P(0));
    std::fill(charge.begin(), charge.end(), test == Case::accumulated_overflow ?
        Q(std::numeric_limits<M>::max()) : Q(0));
    charge[1] = Q(std::numeric_limits<M>::max());
  }
  if (test == Case::narrowing_cancellation) {
    // Both records belong to the same cell. Their finite FP64 contributions
    // cancel in the matrix-core kernel; scalar FP32 conversion overflows to opposite infinities
    // and its final NaN must reject the deposit.
    charge[1] = Q(1e300);
    charge[2] = Q(-1e300);
  }
  if (test == Case::narrowing_rounding) {
    charge[1] = Q(1.0 / beta);
    charge[2] = Q((-1.0 + 0x1p-26) / beta);
    if constexpr (std::is_same_v<M, float>) {
      const double a = beta * charge[1], b = beta * charge[2];
      require(float(a) + float(b) == 0 && float(a + b) != 0,
          "mixed mass: fixture must distinguish narrowing before addition");
    }
  }
  if (test == Case::metadata) ++offsets[cells.encode(1, 0, 0) + 1];
  if (test == Case::metadata_count) counts[cells.encode(1, 0, 0)] =
      std::numeric_limits<unsigned>::max();
  if (test == Case::metadata_first) offsets.front() = 1;
  if (test == Case::metadata_last) --offsets.back();

  Buffer<P> dx(total), dy(total), dz(total);
  Buffer<Q> dq(total);
  Buffer<std::int32_t> dk(total);
  Buffer<unsigned> dc(cells.count), df(cells.count + 1), failure(1), invalid(1);
  dx.upload(rx, executor); dy.upload(ry, executor); dz.upload(rz, executor);
  dq.upload(charge, executor); dk.upload(keys, executor);
  dc.upload(counts, executor); df.upload(offsets, executor);
  const std::vector<unsigned> producer_failure{test == Case::producer ? 1U : 0U};
  failure.upload(producer_failure, executor);
  Particles<Precision> particles{dx.data(), dy.data(), dz.data(), dq.data(), dk.data()};

  // Distinct component pitches catch accidental use of another field's layout.
  Buffer<B> dbx(11 * 5 * 3), dby(13 * 6 * 3), dbz(17 * 7 * 3);
  std::vector<B> bx(dbx.count), by(dby.count), bz(dbz.count);
  for (int k = 0; k <= 2; ++k)
    for (int j = 0; j <= 2; ++j)
      for (int i = 0; i <= 4; ++i) {
        bx[i + 11*j + 55*k] = B(0.2 + 0.07*i - 0.03*j + 0.01*k);
        by[i + 13*j + 78*k] = B(-0.4 + 0.02*i + 0.05*j - 0.04*k);
        bz[i + 17*j + 119*k] = B(0.7 - 0.01*i + 0.03*j + 0.02*k);
      }
  if (test == Case::field) bx[1] = std::numeric_limits<B>::quiet_NaN();
  if (corner_fixture) {
    std::fill(bx.begin(), bx.end(), B(0));
    std::fill(by.begin(), by.end(), B(0));
    std::fill(bz.begin(), bz.end(), B(0));
  }
  dbx.upload(bx, executor); dby.upload(by, executor); dbz.upload(bz, executor);
  Magnetic<Precision> magnetic{{FieldComponent<B>{dbx.data(), 11, 55},
      FieldComponent<B>{dby.data(), 13, 78}, FieldComponent<B>{dbz.data(), 17, 119}}};
  Buffer<M> sorted(layout.storage_size()), unsorted(layout.storage_size());
  sorted.clear(executor); unsorted.clear(executor); invalid.clear(executor);
  tpic::ecsim::ParticleBins bins{{dc.data(), df.data(), unsigned(cells.count)},
      failure.data(), prefix};
  // A failed producer must be checked before dereferencing any borrowed views.
  if (test == Case::producer) {
    bins.view.counts = bins.view.offsets = nullptr;
    particles = {};
    magnetic = {};
  }
  // One block forces every cell warp through repeated grid-stride iterations.
  if constexpr (Impl == Implementation::matrix_core) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
    tpic::ecsim::kernels::sorted_mass_matrix_mma_kernel<Precision>
        <<<1, tpic::ecsim::sorted_cell_threads, 0, executor.stream()>>>(
#else
    tpic::ecsim::kernels::sorted_mass_matrix_mfma4_kernel<Precision>
        <<<1, tpic::ecsim::mass_mfma4_threads, 0, executor.stream()>>>(
#endif
        particles, magnetic, sorted.data(), layout, cells, deposition_beta, bins, invalid.data());
  } else
    tpic::ecsim::kernels::sorted_mass_matrix_scalar_kernel<Precision>
        <<<1, tpic::ecsim::sorted_cell_threads, 0, executor.stream()>>>(
        particles, magnetic, sorted.data(), layout, cells, deposition_beta, bins, invalid.data());
  if (test == Case::suffix)
    tpic::ecsim::kernels::unsorted_mass_matrix_kernel<Precision>
        <<<1, tpic::device::particle_block_threads, 0, executor.stream()>>>(
        particles, magnetic, sorted.data(), layout, cells, deposition_beta,
        prefix, total - prefix, invalid.data());
  require(Traits::succeeded(Traits::take_last_launch_error()), "sorted mass: launch failed");
  const bool should_fail = test != Case::ordinary && test != Case::empty &&
      test != Case::zero_charge && test != Case::suffix && test != Case::large_finite &&
      test != Case::narrowing_rounding &&
      !(test == Case::narrowing_cancellation &&
          (Impl != Implementation::scalar || std::is_same_v<M, double>)) &&
      !(test == Case::storage_overflow && std::is_same_v<M, double>);
  require((invalid.download(executor)[0] != 0) == should_fail,
          "sorted mass: invalid input flag mismatch");
  if (test == Case::producer) {
    const auto actual = sorted.download(executor);
    require(std::all_of(actual.begin(), actual.end(), [](M value) { return value == M(0); }),
        "sorted mass: failed producer must not deposit");
  }
  if (should_fail || test == Case::storage_overflow) return;

  std::vector<M> expected;
  if constexpr (Impl == Implementation::matrix_core)
    expected = host_matrix_core_mass<Precision>(rx, ry, rz, charge, keys, bx, by, bz,
        counts, offsets, cells, layout, deposition_beta,
        test == Case::suffix ? total - prefix : 0);
  else
  {
    tpic::ecsim::kernels::unsorted_mass_matrix_kernel<Precision>
        <<<7, tpic::device::particle_block_threads, 0, executor.stream()>>>(
        particles, magnetic, unsorted.data(), layout, cells, deposition_beta, 0,
        test == Case::empty ? 0 : total, invalid.data());
    require(Traits::succeeded(Traits::take_last_launch_error()), "scalar mass: launch failed");
    expected = unsorted.download(executor);
  }
  const auto actual = sorted.download(executor);
  require(invalid.download(executor)[0] == 0, "scalar mass: invalid unsorted result");
  const double tolerance = std::is_same_v<M, float> ? 3e-5 : 3e-12;
  double scale = 0;
  for (const auto value : expected) scale = std::max(scale, std::abs(double(value)));
  for (std::size_t index = 0; index < expected.size(); ++index)
    require(std::isfinite(double(actual[index])) &&
        std::abs(double(actual[index]) - double(expected[index])) <= tolerance * scale + 1e-15,
        Impl == Implementation::scalar ?
            "sorted mass: coefficient mismatch against scalar deposition" :
            "matrix mass: coefficient mismatch against host FP64 cell sums");
}

template<class Precision, Implementation Impl = Implementation::scalar> void run(Executor executor) {
  for (const auto test : {Case::ordinary, Case::empty, Case::zero_charge,
       Case::suffix, Case::key, Case::fraction, Case::endpoint,
       Case::charge, Case::field, Case::response_overflow, Case::large_finite,
       Case::accumulated_overflow,
       Case::storage_overflow, Case::narrowing_cancellation, Case::narrowing_rounding,
       Case::metadata, Case::metadata_count, Case::metadata_first, Case::metadata_last,
       Case::producer, Case::halo}) run_case<Precision, Impl>(executor, test);
}
} // namespace

int main() {
  lpfx::device::DeviceProperties properties;
  auto error = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(error)) {
    if (runtime_unavailable(error)) {
      std::cout << "TPIC sorted mass tests skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto stream = Traits::null_stream;
  error = Traits::create_stream(stream);
  if (!Traits::succeeded(error)) return runtime_unavailable(error) ? 77 : 1;
  int status = 0;
  try {
    const Executor executor{stream};
    run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor);
    run<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor);
    run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor);
    if (matrix_core_available<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(properties.device_ordinal) &&
        matrix_core_available<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(properties.device_ordinal) &&
        matrix_core_available<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(properties.device_ordinal)) {
      run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>, Implementation::matrix_core>(executor);
      run<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>, Implementation::matrix_core>(executor);
      run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>, Implementation::matrix_core>(executor);
      std::cout << "TPIC matrix-core mass tests passed\n";
    } else {
      std::cout << "TPIC matrix-core mass tests skipped: supported FP64 matrix device and kernel image required\n";
    }
    std::cout << "TPIC sorted mass tests passed\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(stream))) status = 1;
  return status;
}
