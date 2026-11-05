// Full-sort publication boundaries, independent executors, retirement and
// allocation replacement. No MPI runtime is needed for this lifecycle check.
#include <lpfx/api/particle/device_particles.hpp>
#include <lpfx/data/particle/device/initialization.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/mass_matrix/compact_mass_matrix_device.hpp>
#include <tpic/ecsim/particles/particle_sort_device.hpp>
#include <tpic/ecsim/deposition/current_deposition_device.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp>
#include <tpic/ecsim/deposition/moment_field_device.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include "../support/test_builders.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
constexpr unsigned bin_count = 19;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function> void require_rejected(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error(message);
}

template <class Function> void poll_success(Function&& function, const char* message) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (;;) {
    const auto status = function();
    if (status.is_success()) return;
    require(status.is_pending() && std::chrono::steady_clock::now() < deadline, message);
    std::this_thread::yield();
  }
}

template <class Precision, class Range>
__global__ void initialize_sort_particles_kernel(Range range) {
  using S = tpic::data::ParticleSchema<Precision>;
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range.size()) return;
  range.template store<typename S::Rx>(i, i * 0.125);
  range.template store<typename S::Ry>(i, i * -0.25);
  range.template store<typename S::Rz>(i, i * 0.5);
  range.template store<typename S::U>(i, i * -0.5);
  range.template store<typename S::V>(i, i * 0.25);
  range.template store<typename S::W>(i, i * -0.125);
  range.template store<typename S::Charge>(i, i * 2.0);
  range.template store<typename S::Cell>(i, (i * 7 + 3) % bin_count);
  range.template store<typename S::Id>(i, i + 100U);
}

template <class Precision> auto make_owner(unsigned count, unsigned capacity, Executor executor) {
  using Schema = typename tpic::data::ParticleSchema<Precision>::type;
  lpfx::PreparedDeviceParticleStorage<Schema, Backend> storage;
  require(lpfx::try_prepare_device_particle_storage(capacity, storage) == lpfx::ErrorCode::success,
          "particle sort: storage preparation failed");
  auto made = lpfx::make_device_particles(storage);
  require(made.has_value(), "particle sort: owner allocation failed");
  auto owner = std::move(made).value();
  if (count != 0) {
    auto prepared = lpfx::make_device_particle_initializer(owner);
    require(prepared.has_value(), "particle sort: initializer creation failed");
    auto initializer = std::move(prepared).value();
    auto admitted = initializer.begin_append(count, executor);
    require(admitted.has_value(), "particle sort: fixture admission failed");
    initialize_sort_particles_kernel<Precision><<<(count + 127) / 128, 128, 0, executor.stream()>>>(admitted.value());
    require(Traits::succeeded(Traits::take_last_launch_error()) &&
                initializer.publish(executor).is_success() && initializer.wait().is_success(),
            "particle sort: fixture publication failed");
  }
  return owner;
}

template <class T> auto download(const T* source, unsigned count, Executor executor) {
  std::vector<T> values(count);
  if (count != 0)
    require(Traits::succeeded(Traits::copy_device_to_host_async(values.data(), source,
                count * sizeof(T), executor.stream())) &&
                Traits::succeeded(Traits::synchronize_stream(executor.stream())),
            "particle sort: verification download failed");
  return values;
}

template <class Attribute, class Owner>
void verify_attribute(const Owner& owner, const std::vector<std::uint64_t>& ids,
                      double factor, Executor executor) {
  const auto values = download(owner.cview().template get<Attribute>(), owner.size(), executor);
  for (unsigned i = 0; i != owner.size(); ++i)
    require(values[i] == (ids[i] - 100U) * factor, "particle sort: attribute mapping corrupted");
}

template <class Precision, class Owner, class Sort>
void verify(const Owner& owner, const Sort& sort, Executor executor) {
  using S = tpic::data::ParticleSchema<Precision>;
  const auto& receipt = sort.receipt();
  require(receipt.prefix_size() == owner.size(), "particle sort: incorrect sorted prefix");
  const auto ids = download(owner.cview().template get<typename S::Id>(), owner.size(), executor);
  const auto keys = download(owner.cview().template get<typename S::Cell>(), owner.size(), executor);
  require(std::is_sorted(keys.begin(), keys.end()), "particle sort: keys not sorted");
  auto identities = ids;
  std::sort(identities.begin(), identities.end());
  for (unsigned i = 0; i != owner.size(); ++i) {
    require(identities[i] == i + 100U, "particle sort: duplicated or missing particle");
    require(keys[i] == ((ids[i] - 100U) * 7U + 3U) % bin_count,
            "particle sort: key mapping corrupted");
  }
  verify_attribute<typename S::Rx>(owner, ids, 0.125, executor);
  verify_attribute<typename S::Ry>(owner, ids, -0.25, executor);
  verify_attribute<typename S::Rz>(owner, ids, 0.5, executor);
  verify_attribute<typename S::U>(owner, ids, -0.5, executor);
  verify_attribute<typename S::V>(owner, ids, 0.25, executor);
  verify_attribute<typename S::W>(owner, ids, -0.125, executor);
  verify_attribute<typename S::Charge>(owner, ids, 2.0, executor);
  const auto bins = receipt.bins();
  const auto counts = download(bins.counts, bin_count, executor);
  const auto offsets = download(bins.offsets, bin_count + 1, executor);
  require(offsets.front() == 0 && offsets.back() == owner.size(),
          "particle sort: incorrect bin boundaries");
  for (unsigned bin = 0; bin != bin_count; ++bin)
    require(offsets[bin + 1] == offsets[bin] + counts[bin] &&
                counts[bin] == std::count(keys.begin(), keys.end(), static_cast<int>(bin)),
            "particle sort: bin counts disagree with committed keys");
}

template <class Precision>
void run(const lpfx::device::DeviceProperties& properties, Executor first, Executor second) {
  using S = tpic::data::ParticleSchema<Precision>;
  using Sort = tpic::ecsim::FullParticleSort<Precision, Backend>;
  tpic::config::PreparedDomain domain;
  domain.particle_cell_count_with_halo = bin_count;
  auto a = make_owner<Precision>(257, 257, first);
  auto b = make_owner<Precision>(129, 257, second);
  Sort sort_a, sort_b;
  sort_a.initialize(a, domain, properties, first);
  sort_b.initialize(b, domain, properties, second);
  require_rejected([&] { sort_a.host_wait_complete(); }, "particle sort: completion without enqueue accepted");
  const auto* old_cell = a.cview().template get<typename S::Cell>();
  sort_a.rebind_to_owner(a, properties, first);
  sort_a.enqueue(a, domain, first);
  sort_b.rebind_to_owner(b, properties, second);
  sort_b.enqueue(b, domain, second);
  require(a.cview().template get<typename S::Cell>() == old_cell,
          "particle sort: owner mapping published before checked completion");
  require_rejected([&] { static_cast<void>(sort_a.receipt()); },
                   "particle sort: pending receipt exposed to ordinary consumers");
  require_rejected([&] {
    sort_a.rebind_to_owner(a, properties, first);
    sort_a.enqueue(a, domain, first);
  },
                   "particle sort: accepted second enqueue with live receipt");
  // Completing in reverse order exercises independent receipt/status owners.
  poll_success([&] { return sort_b.progress_complete(); }, "particle sort: B polling failed");
  poll_success([&] { return sort_a.progress_complete(); }, "particle sort: A polling failed");
  require(sort_a.progress_complete().is_failure(), "particle sort: duplicate polling accepted");
  verify<Precision>(a, sort_a, first);
  verify<Precision>(b, sort_b, second);
  require_rejected([&] { sort_a.host_wait_complete(); }, "particle sort: duplicate completion accepted");
  lpfx::device::DeviceEvent<Backend> last_a, last_b;
  require(Traits::succeeded(decltype(last_a)::create(last_a)) &&
          Traits::succeeded(decltype(last_b)::create(last_b)) &&
          Traits::succeeded(last_a.record(first)) && Traits::succeeded(last_b.record(second)),
          "particle sort: explicit retirement event failed");
  sort_a.retire_after(last_a);
  sort_b.retire_after(last_b);
  require_rejected([&] { sort_a.rebind_to_owner(a, properties, first); },
                   "particle sort: preparation before retirement poll accepted");
  require_rejected([&] { sort_a.enqueue(a, domain, first); },
                   "particle sort: enqueue before retirement poll accepted");
  poll_success([&] { return sort_b.progress_retirement(); }, "particle sort: B retirement poll failed");
  poll_success([&] { return sort_a.progress_retirement(); }, "particle sort: A retirement poll failed");
  require_rejected([&] { static_cast<void>(sort_a.receipt()); },
                   "particle sort: retired receipt exposed");

  // Ordinary migration may replace capacity/storage after completed retirement.
  // Preserve every live attribute and force standalone-sort binding recreation.
  require(lpfx::particle::resize_capacity(a, a.capacity() + 1U, first) == lpfx::ErrorCode::success,
          "particle sort: preserving owner growth failed");
  require_rejected([&] { sort_a.enqueue(a, domain, first); },
                   "particle sort: invalidated sorter submitted before preparation");
  sort_a.rebind_to_owner(a, properties, first);
  sort_a.enqueue(a, domain, first);
  poll_success([&] { return sort_a.progress_complete(); }, "particle sort: grown completion failed");
  verify<Precision>(a, sort_a, first);
  sort_a.host_wait_retired(first);

  // Replacement at unchanged capacity must also invalidate the old binding.
  a = make_owner<Precision>(33, a.capacity(), first);
  sort_a.rebind_to_owner(a, properties, first);
  sort_a.enqueue(a, domain, first);
  sort_a.host_wait_complete();
  verify<Precision>(a, sort_a, first);
  sort_a.host_wait_retired(first);
  a.clear();
  sort_a.rebind_to_owner(a, properties, first);
  sort_a.enqueue(a, domain, first);
  sort_a.host_wait_complete();
  verify<Precision>(a, sort_a, first);
  sort_a.host_wait_retired(first);

  // Failed checked attempts retain LPFx's retirement obligation but must never
  // become ordinary numerical consumers' successfully published receipt.
  const std::int32_t invalid_cell = -1;
  require(Traits::succeeded(Traits::copy_host_to_device_async(
              b.view().template get<typename S::Cell>(), &invalid_cell, sizeof(invalid_cell), second.stream())),
          "particle sort: invalid fixture upload failed");
  sort_b.rebind_to_owner(b, properties, second);
  sort_b.enqueue(b, domain, second);
  require_rejected([&] { sort_b.host_wait_complete(); }, "particle sort: invalid cell accepted");
  require_rejected([&] { static_cast<void>(sort_b.receipt()); },
                   "particle sort: failed receipt exposed");
  sort_b.host_wait_retired(second);
}

template <class Precision, class View>
__global__ void initialize_deposition_particles_kernel(View particles, tpic::data::ParticleCellMap cells) {
  using S = tpic::data::ParticleSchema<Precision>;
  const unsigned p = threadIdx.x;
  if (p >= particles.size()) return;
  particles.template get<typename S::Cell>()[p] = cells.encode(p % 2, 0, 0);
  particles.template get<typename S::Rx>()[p] = 0.125 * (p % 7);
  particles.template get<typename S::Ry>()[p] = 0.25;
  particles.template get<typename S::Rz>()[p] = 0.5;
}

template <class Left, class Right>
__global__ void compare_sorted_unsorted_currents_kernel(Left a, Right b, unsigned* invalid, double tolerance) {
  const unsigned n = threadIdx.x;
  if (n >= 12) return;
  const int i = n % 3, j = (n / 3) % 2, k = n / 6;
  [&]<std::size_t... C>(std::index_sequence<C...>) {
    const auto compare = [&](auto component) {
      const double left = a.template component<decltype(component)::value>()(i,j,k);
      const double right = b.template component<decltype(component)::value>()(i,j,k);
      if (!::isfinite(left) || !::isfinite(right) ||
          ::fabs(left - right) > tolerance * (1 + ::fabs(left))) atomicExch(invalid, 1U);
    };
    (compare(std::integral_constant<std::size_t, C>{}), ...);
  }(std::make_index_sequence<3>{});
}

// Exercise actual owner-based adapter instantiations and resident rejection;
// valid early capabilities are created only by LPFx migration integration.
template <class Precision>
void deposition_adapters(const lpfx::device::DeviceProperties& properties, Executor executor) {
  namespace ecsim = tpic::ecsim;
  using F = tpic::data::FieldSchema<Precision>;
  using S = tpic::data::ParticleSchema<Precision>;
  using Read = lpfx::ResidentParticleRead<typename S::type, Backend>;
  using Receipt = lpfx::DeviceSortSubmission<typename S::type, Backend>;
  static_assert(std::is_same_v<ecsim::ConstParticleReadView<Precision, Backend>,
                decltype(std::declval<const Receipt&>().resident_particle_view())>);
  static_assert(std::is_const_v<std::remove_pointer_t<decltype(
      std::declval<ecsim::ConstParticleReadView<Precision, Backend>>().template get<0>())>>);
  tpic::config::PreparedDomain domain;
  domain.local_cells = {2, 1, 1};
  domain.inverse_cell_volume = 2.;
  tpic::test::fill_domain_counts(domain);
  const auto cells = tpic::data::prepare_particle_cell_map(domain);
  require(cells.count > 0, "deposition adapter: geometry preparation failed");
  auto invalid_domain = domain;
  invalid_domain.local_cells[0] = 0;
  invalid_domain.particle_cell_count_with_halo = 0;
  bool invalid_rejected = false;
  try {
    (void)tpic::data::prepare_particle_cell_map(invalid_domain);
  } catch (const std::invalid_argument&) {
    invalid_rejected = true;
  }
  require(invalid_rejected, "deposition adapter: invalid geometry accepted");
  const tpic::data::ParticleCellMap invalid_cells{};
  const auto make_field = [&]<class Schema>() {
    auto made = lpfx::field::make<Schema>(lpfx::field::options{.cells = {2, 1, 1}, .halo = 1});
    require(made.has_value(), "deposition adapter: field creation failed");
    auto host = std::move(made).value();
    return tpic::data::upload_field(host, executor);
  };
  auto magnetic = make_field.template operator()<typename F::BNode>();
  auto unsorted_current = make_field.template operator()<typename F::Current>();
  auto sorted_current = make_field.template operator()<typename F::Current>();
  tpic::data::CompactMassMatrix<Precision, Backend> unsorted_mass(domain), sorted_mass(domain);
  lpfx::device::DeviceAllocation<Backend> failure;
  require(Traits::succeeded(decltype(failure)::create(sizeof(unsigned), failure)),
          "deposition adapter: invalid word allocation failed");
  auto* invalid = reinterpret_cast<unsigned*>(failure.raw_base());
  const double tolerance = std::is_same_v<typename Precision::Storage::Mass, float> ? 3.e-5 : 2.e-12;
  const Read empty_read;
  require(ecsim::enqueue_resident_current<Precision>(empty_read, magnetic, sorted_current,
              cells, domain.inverse_cell_volume, .125, invalid, executor) == lpfx::ErrorCode::invalid_state &&
          ecsim::enqueue_resident_mass_matrix<Precision>(empty_read, magnetic, sorted_mass,
              cells, .125, invalid, executor) == lpfx::ErrorCode::invalid_state,
          "deposition adapter: unissued resident capability accepted");
  require(ecsim::enqueue_resident_current<Precision>(empty_read, magnetic, sorted_current,
              invalid_cells, domain.inverse_cell_volume, .125, invalid, executor) == lpfx::ErrorCode::invalid_argument &&
          ecsim::enqueue_resident_mass_matrix<Precision>(empty_read, magnetic, sorted_mass,
              cells, .125, invalid, Executor{Traits::null_stream}) == lpfx::ErrorCode::invalid_argument &&
          ecsim::enqueue_resident_current<Precision>(empty_read, magnetic, sorted_current,
              cells, domain.inverse_cell_volume, .125, nullptr, executor) == lpfx::ErrorCode::invalid_argument,
          "deposition adapter: invalid resident launch accepted");
  require(ecsim::enqueue_resident_mass_matrix<
              Precision, tpic::config::MassDepositionKernel::matrix_core>(empty_read, magnetic, sorted_mass,
              cells, .125, invalid, executor) == lpfx::ErrorCode::invalid_state,
          "deposition adapter: matrix-core accepted unissued resident capability");
  auto particles = make_owner<Precision>(17, 17, executor);
  initialize_deposition_particles_kernel<Precision><<<1, 32, 0, executor.stream()>>>(particles.view(), cells);
  require(Traits::succeeded(Traits::take_last_launch_error()),
          "deposition adapter: fixture launch failed");
  ecsim::FullParticleSort<Precision, Backend> sorter;
  sorter.initialize(particles, domain, properties, executor);
  sorter.enqueue(particles, domain, executor);
  poll_success([&] { return sorter.progress_complete(); }, "deposition adapter: sort polling failed");
  const auto compare = [&](bool matrix_core) {
    require(Traits::succeeded(Traits::memset_device_async(invalid, 0, sizeof(unsigned), executor.stream())) &&
            ecsim::enqueue_clear_moment_field(unsorted_current, executor) == lpfx::ErrorCode::success &&
            ecsim::enqueue_clear_moment_field(sorted_current, executor) == lpfx::ErrorCode::success &&
            ecsim::enqueue_clear_mass_matrix(unsorted_mass, executor) == lpfx::ErrorCode::success &&
            ecsim::enqueue_clear_mass_matrix(sorted_mass, executor) == lpfx::ErrorCode::success,
            "deposition adapter: clear failed");
    require(ecsim::enqueue_unsorted_current<Precision>(particles, magnetic, unsorted_current,
                cells, domain.inverse_cell_volume, .125, 0, particles.size(), invalid, executor) == lpfx::ErrorCode::success &&
            ecsim::enqueue_sorted_current<Precision>(particles, sorter.receipt(), magnetic, sorted_current,
                cells, domain.inverse_cell_volume, .125, invalid, executor) == lpfx::ErrorCode::success &&
            ecsim::enqueue_unsorted_mass_matrix<Precision>(particles, magnetic, unsorted_mass,
                cells, .125, 0, particles.size(), invalid, executor) == lpfx::ErrorCode::success,
            "deposition adapter: numerical submission failed");
    if (matrix_core)
      require(ecsim::enqueue_sorted_mass_matrix<
                  Precision, tpic::config::MassDepositionKernel::matrix_core>(particles, sorter.receipt(), magnetic,
                  sorted_mass, cells, .125, invalid, executor) == lpfx::ErrorCode::success,
              "deposition adapter: matrix-core submission failed");
    else
      require(ecsim::enqueue_sorted_mass_matrix<Precision>(particles, sorter.receipt(), magnetic,
                  sorted_mass, cells, .125, invalid, executor) == lpfx::ErrorCode::success,
              "deposition adapter: scalar mass submission failed");
    compare_sorted_unsorted_currents_kernel<<<1, 32, 0, executor.stream()>>>(
        unsorted_current.view(), sorted_current.view(), invalid, tolerance);
    require(Traits::succeeded(Traits::take_last_launch_error()) && download(invalid, 1, executor)[0] == 0,
            "deposition adapter: invalid numerical result or current mismatch");
    const auto unsorted = download(unsorted_mass.data(), unsorted_mass.layout().storage_size(), executor);
    const auto actual = download(sorted_mass.data(), sorted_mass.layout().storage_size(), executor);
    for (std::size_t n = 0; n < actual.size(); ++n)
      require(std::isfinite(actual[n]) &&
                  std::abs(double(actual[n]) - unsorted[n]) <= tolerance * (1 + std::abs(double(unsorted[n]))),
              "deposition adapter: mass mismatch");
  };
  compare(false);
  if (ecsim::matrix_core_mass_deposition_available<Precision>(properties.device_ordinal)) compare(true);
  sorter.host_wait_retired(executor);
  particles.clear();
  sorter.enqueue(particles, domain, executor);
  poll_success([&] { return sorter.progress_complete(); }, "deposition adapter: empty sort failed");
  compare(false);
  sorter.host_wait_retired(executor);
}

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}
} // namespace

int main() {
  lpfx::device::DeviceProperties properties{};
  const auto queried = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(queried)) {
    if (runtime_unavailable(queried)) {
      std::cout << "TPIC particle sort skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto first = Traits::null_stream, second = Traits::null_stream;
  if (!Traits::succeeded(Traits::create_stream(first))) return 1;
  if (!Traits::succeeded(Traits::create_stream(second))) {
    static_cast<void>(Traits::destroy_stream(first));
    return 1;
  }
  int status = 0;
  try {
    run<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(properties, Executor{first}, Executor{second});
    run<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(properties, Executor{first}, Executor{second});
    run<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(properties, Executor{first}, Executor{second});
    deposition_adapters<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(properties, Executor{first});
    deposition_adapters<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(properties, Executor{first});
    deposition_adapters<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(properties, Executor{first});
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  static_cast<void>(Traits::destroy_stream(second));
  static_cast<void>(Traits::destroy_stream(first));
  return status;
}
