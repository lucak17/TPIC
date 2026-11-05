#include <tpic/config/domain.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/data/output/output_snapshot.hpp>
// Selected strided snapshots retain old source values through a delayed D2H,
// reuse their bounded slots, preserve FP64 species-total rounding and reject
// incomplete epochs. No MPI or file writer is used by this device fixture.
#include <tpic/diagnostics/async_field_snapshot_device.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/memory/space.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using Backend = lpfx::device::ActiveBackend;
using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;
using Allocation = lpfx::device::DeviceAllocation<Backend>;
using Snapshot = tpic::diagnostics::AsyncFieldSnapshot<Backend>;
using Cells = decltype(std::declval<tpic::config::PreparedDomain>().field_neighborhood.local_partition().local_cells());
constexpr int nx = 17, ny = 5, nz = 3, pitch_x = 23, pitch_y = 8;
constexpr int allocated_nodes = pitch_x * pitch_y * (nz + 2);
constexpr int field_components = 6;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Function> void rejects(Function&& function, std::string_view expected) {
  try {
    function();
  } catch (const std::runtime_error& error) {
    require(std::string_view(error.what()).find(expected) != std::string_view::npos,
            "snapshot rejected operation with unexpected error");
    return;
  }
  throw std::runtime_error("snapshot accepted invalid lifecycle or input");
}

template <class Scalar> struct Component {
  using location_type = lpfx::Node;
  Scalar* data;
  __host__ __device__ Scalar& operator()(int i, int j, int k) const {
    return data[i + pitch_x * (j + pitch_y * k)];
  }
};
template <class Scalar> struct FieldView {
  Scalar* data;
  template <std::size_t Index> __host__ __device__ Component<Scalar> component() const {
    return {data + Index * allocated_nodes};
  }
};
template <class Scalar> struct FieldOwner {
  using space_type = lpfx::DeviceSpace<Backend>;
  static constexpr unsigned component_count = field_components;
  FieldView<Scalar> data;
  Cells extent;
  bool live = true;
  auto cells() const { return extent; }
  bool initialized() const { return live; }
  auto view() const { return data; }
};

__global__ void delay_snapshot_transfer_kernel(unsigned long long ticks) {
  const auto start = clock64();
  while (clock64() - start < ticks) {}
}
template <class Scalar>
__global__ void replace_snapshot_source_kernel(Scalar* values, unsigned count) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x)
    values[i] = Scalar(-8192.0);
}

std::vector<double> expected_values(const auto& source, int component, double scale,
                                    const tpic::config::PreparedDomain& domain) {
  std::vector<double> result;
  for (int k = 0; k <= domain.local_cells[2]; ++k)
    for (int j = 0; j <= domain.local_cells[1]; ++j)
      for (int i = 0; i <= domain.local_cells[0]; ++i) {
        // Materialize each product before addition, as the device rounds
        // scale * value before adding it, even with host FMA contraction.
        volatile double scaled = scale * static_cast<double>(source[
            component * allocated_nodes + i + pitch_x * (j + pitch_y * k)]);
        result.push_back(static_cast<double>(scaled));
      }
  return result;
}

template <class Scalar>
void run(Executor producer, Executor transfer, int local_nx, int local_ny, int local_nz) {
  tpic::config::GridConfig grid{.lx = double(local_nx), .ly = double(local_ny), .lz = double(local_nz),
      .nxc = local_nx, .nyc = local_ny, .nzc = local_nz};
  const auto domain = tpic::config::prepare_domain(grid, {}, 0, 1);
  const std::array<tpic::data::OutputArrayDescriptor, 5> descriptors{{
      {"B", 3}, {"rho_0", 1}, {"rho", 1}, {"J", 3}, {"PXX", 1}}};
  Snapshot capture(domain, descriptors, producer, transfer, 3);
  require(capture.node_count() == static_cast<std::size_t>(domain.interior_node_count),
          "snapshot has wrong logical node count");
  const auto node_bytes = capture.node_count() * sizeof(double);
  // Only the two-launch (HIP) accumulation path keeps a scratch slab.
  constexpr auto vector_scratch = tpic::data::field_snapshot_scratch_components<Backend>(3);
  require(capture.device_bytes() == (9 + vector_scratch) * node_bytes &&
              capture.pinned_bytes() == 9 * node_bytes,
          "snapshot storage differs from selected arrays plus scratch budget");
  require(capture.find("absent") == tpic::config::no_slot && capture.find("B") == 0,
          "snapshot descriptor lookup failed");
  const auto magnetic = capture.find("B"), species = capture.find("rho_0"),
      total = capture.find("rho"), current = capture.find("J"), pressure = capture.find("PXX");
  std::array<Scalar, field_components * allocated_nodes> input;
  Allocation source;
  require(Traits::succeeded(Allocation::create(sizeof(input), source)),
          "snapshot source allocation failed");
  FieldOwner<Scalar> field{{reinterpret_cast<Scalar*>(source.raw_base())}, domain.field_neighborhood.local_partition().local_cells()};
  tpic::data::OwnedOutputSnapshot output{7, 1.75, true, {}};
  std::array<const double*, descriptors.size()> owned_addresses{};
  std::array<std::size_t, descriptors.size()> owned_capacities{};

  rejects([&] { (void)capture.ready(); }, "not submitted");
  rejects([&] { capture.host_wait(); }, "not submitted");
  rejects([&] { capture.device_wait_sources(producer); }, "not recorded");
  rejects([&] { capture.template pack<0>(field, species); }, "no active capture");
  for (int epoch = 0; epoch < 3; ++epoch) {
    input.fill(std::numeric_limits<Scalar>::quiet_NaN()); // Padding must never be packed.
    for (int component = 0; component < field_components; ++component)
      for (int k = 0; k <= local_nz; ++k)
        for (int j = 0; j <= local_ny; ++j)
          for (int i = 0; i <= local_nx; ++i)
            input[component * allocated_nodes + i + pitch_x * (j + pitch_y * k)] =
                Scalar((component + 1) * (1 + i + 2*j - 3*k) / 7. + epoch / 3.);
    require(Traits::succeeded(Traits::copy_host_to_device_async(
                source.raw_base(), input.data(), sizeof(input), producer.stream())),
            "snapshot source upload failed");
    capture.begin();
    rejects([&] { capture.begin(); }, "not collected");
    rejects([&] { capture.finish_capture(); }, "not initialized");
    rejects([&] { capture.clear(tpic::config::no_slot); }, "invalid array slot");
    rejects([&] { capture.template add<0>(field, total); }, "not cleared");
    rejects([&] { capture.template pack<0>(field, magnetic); }, "component count");
    rejects([&] { capture.template pack<0>(field, species,
        std::numeric_limits<double>::infinity()); }, "invalid field owner or scale");
    field.live = false;
    rejects([&] { capture.template pack<0>(field, species); }, "invalid field owner or scale");
    field.live = true;

    capture.clear(total);
    capture.clear(current);
    capture.clear(pressure);
    rejects([&] { capture.clear(total); }, "already initialized");
    capture.template pack<2, 0, 1>(field, magnetic, .3);
    capture.template pack_and_add<3>(field, species, total, .1);
    rejects([&] { capture.template pack<3>(field, species); }, "already initialized");
    rejects([&] { capture.template pack_and_add<3>(field, species, total); },
            "invalid packed total");
    rejects([&] { capture.template pack_and_add<3>(field, total, total); },
            "invalid packed total");
    capture.template add<3>(field, total, .2);
    capture.template add<3>(field, total, -.4);
    capture.template add<0, 1, 2>(field, current, .1);
    capture.template add<0, 1, 2>(field, current, .2);
    capture.template add<0, 1, 2>(field, current, -.4);
    capture.template add<5>(field, pressure, .7);

    // Delay the one bulk copy while already-packed original fields are
    // overwritten. Only the immutable selected slab may feed that transfer.
    delay_snapshot_transfer_kernel<<<1, 1, 0, transfer.stream()>>>(20000000ULL);
    require(Traits::succeeded(Traits::take_last_launch_error()), "snapshot transfer delay failed");
    capture.finish_capture();
    require(capture.pending(), "snapshot lost pending transfer ownership");
    rejects([&] { capture.begin(); }, "not collected");
    rejects([&] { capture.template pack<0>(field, species); }, "no active capture");
    rejects([&] { capture.finish_capture(); }, "no active capture");
    capture.device_wait_sources(producer);
    replace_snapshot_source_kernel<<<32, 256, 0, producer.stream()>>>(
        field.data.data, field_components * allocated_nodes);
    require(Traits::succeeded(Traits::take_last_launch_error()) &&
                Traits::succeeded(Traits::synchronize_stream(producer.stream())),
            "snapshot source replacement failed");
    if (!capture.ready()) capture.host_wait();
    capture.copy_into(output);
    require(!capture.pending() && output.cycle == 7 && output.time == 1.75 && output.final_state,
            "snapshot collection changed epoch or retained staging");
    capture.device_wait_sources(producer); // Event remains valid after host collection.
    rejects([&] { capture.copy_into(output); }, "not submitted");
    for (std::size_t slot = 0; slot < descriptors.size(); ++slot) {
      const auto& array = output.arrays[slot];
      require(array.name == descriptors[slot].name && array.components == descriptors[slot].components &&
                  array.values.size() == capture.node_count() * array.components,
              "snapshot descriptor order or shape changed");
      if (epoch != 0)
        require(array.values.data() == owned_addresses[slot] &&
                    array.values.capacity() == owned_capacities[slot],
                "snapshot reused host arrays by allocating replacement storage");
      owned_addresses[slot] = array.values.data();
      owned_capacities[slot] = array.values.capacity();
    }
    const auto rho0 = expected_values(input, 3, .1, domain);
    auto rho = rho0;
    for (const double scale : {.2, -.4}) {
      const auto contribution = expected_values(input, 3, scale, domain);
      for (std::size_t i = 0; i < rho.size(); ++i) rho[i] += contribution[i];
    }
    require(output.arrays[species].values == rho0 && output.arrays[total].values == rho &&
                output.arrays[pressure].values == expected_values(input, 5, .7, domain),
            "snapshot scalar packing or FP64 total rounding differs from host");
    for (int component = 0; component < 3; ++component) {
      constexpr int reordered[]{2, 0, 1};
      const auto b = expected_values(input, reordered[component], .3, domain);
      auto j = expected_values(input, component, .1, domain);
      for (const double scale : {.2, -.4}) {
        const auto contribution = expected_values(input, component, scale, domain);
        for (std::size_t i = 0; i < j.size(); ++i) j[i] += contribution[i];
      }
      for (std::size_t node = 0; node < capture.node_count(); ++node)
        require(output.arrays[magnetic].values[3*node + component] == b[node] &&
                    output.arrays[current].values[3*node + component] == j[node],
                "snapshot vector ordering or FP64 total rounding differs from host");
    }
  }

  const std::array<tpic::data::OutputArrayDescriptor, 1> scalar{{{"rho", 1}}};
  {
    const std::array<tpic::data::OutputArrayDescriptor, 2> selected{{{"rho_0", 1}, {"rho", 1}}};
    Snapshot no_scratch(domain, selected, producer, transfer, 0);
    require(no_scratch.device_bytes() == 2 * node_bytes,
            "reused scalar contribution retained unused scratch");
    no_scratch.begin();
    no_scratch.clear(1);
    if constexpr (!tpic::data::fused_field_snapshot_accumulation<Backend>)
      rejects([&] { no_scratch.template add<0>(field, 1); }, "planned scratch");
    no_scratch.template pack_and_add<0>(field, 0, 1);
    no_scratch.finish_capture();
    no_scratch.host_wait();
    tpic::data::OwnedOutputSnapshot reused;
    no_scratch.copy_into(reused);
    require(reused.arrays[0].values == reused.arrays[1].values,
            "scratch-free total changed its packed contribution");
  }
  {
    Snapshot scalar_scratch(domain, scalar, producer, transfer, 1);
    require(scalar_scratch.device_bytes() ==
                (1 + tpic::data::field_snapshot_scratch_components<Backend>(1)) * node_bytes,
            "scalar contribution retained vector-sized scratch");
    scalar_scratch.begin();
    scalar_scratch.clear(0);
    scalar_scratch.template add<0>(field, 0);
    scalar_scratch.finish_capture();
    scalar_scratch.host_wait();
    tpic::data::OwnedOutputSnapshot accumulated;
    scalar_scratch.copy_into(accumulated);
    require(accumulated.arrays[0].values == std::vector<double>(capture.node_count(), -8192.0),
            "scalar scratch changed its contribution");
  }
  // Reject a nonfinite selected value only after D2H makes it host-visible.
  // The rejected array keeps its existing output values, and destruction
  // safely releases the already-completed transfer during failure unwinding.
  {
    Snapshot invalid(domain, scalar, producer, transfer, 0);
    input[0] = std::numeric_limits<Scalar>::infinity();
    require(Traits::succeeded(Traits::copy_host_to_device_async(
                source.raw_base(), input.data(), sizeof(input), producer.stream())),
            "snapshot invalid-source upload failed");
    invalid.begin();
    invalid.template pack<0>(field, 0);
    invalid.finish_capture();
    invalid.host_wait();
    const auto retained = output.arrays.front().values;
    rejects([&] { invalid.copy_into(output); }, "nonfinite field value");
    require(output.arrays.front().values == retained, "failed snapshot altered existing owned output");
  }
  // An unfinished capture can unwind before its final transfer event exists.
  {
    Snapshot unfinished(domain, scalar, producer, transfer, 0);
    unfinished.begin();
    unfinished.template pack<1>(field, 0);
  }
  const std::array<tpic::data::OutputArrayDescriptor, 2> duplicate{{{"rho", 1}, {"rho", 1}}};
  const std::array<tpic::data::OutputArrayDescriptor, 1> malformed{{{"rho", 2}}};
  rejects([&] { Snapshot bad(domain, duplicate, producer, transfer, 0); }, "duplicate array descriptor");
  rejects([&] { Snapshot bad(domain, malformed, producer, transfer, 0); }, "invalid or duplicate");
  rejects([&] { Snapshot bad(domain, scalar, producer, transfer, 2); }, "invalid scratch component");
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
      std::cout << "TPIC async snapshot skipped: no usable device/driver\n";
      return 77;
    }
    return 1;
  }
  auto producer = Traits::null_stream, transfer = Traits::null_stream;
  if (!Traits::succeeded(Traits::create_stream(producer))) return 1;
  if (!Traits::succeeded(Traits::create_stream(transfer))) {
    static_cast<void>(Traits::destroy_stream(producer));
    return 1;
  }
  int status = 0;
  try {
    run<double>(Executor{producer}, Executor{transfer}, nx, ny, nz);
    run<float>(Executor{producer}, Executor{transfer}, nx, ny, nz);
    run<double>(Executor{producer}, Executor{transfer}, 1, 1, 1);
    std::cout << "TPIC asynchronous selected snapshot checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    static_cast<void>(Traits::synchronize_stream(producer));
    static_cast<void>(Traits::synchronize_stream(transfer));
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(transfer))) status = 1;
  if (!Traits::succeeded(Traits::destroy_stream(producer))) status = 1;
  return status;
}
