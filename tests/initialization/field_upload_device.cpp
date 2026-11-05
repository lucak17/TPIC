#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/config/domain.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/initialization/fields/field_image.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "../support/test_builders.hpp"

namespace {

// The startup upload of the Simulation constructor: E, then B, in one braced list.
template <class Precision, class Backend>
tpic::data::DeviceFieldImage<Precision, Backend> upload_image(
    tpic::initialization::HostFieldImage<Precision>& host, lpfx::device::Executor<Backend> executor) {
  return {tpic::data::upload_field(host.electric, executor),
          tpic::data::upload_field(host.magnetic_cell, executor)};
}

using Traits = lpfx::device::ActiveBackendTraits;
using Executor = lpfx::device::ActiveExecutor;

bool runtime_unavailable(Traits::error_type error) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
  return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
         error == cudaErrorSystemDriverMismatch;
#else
  return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#endif
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<class Field, class Function>
void for_each_sample(Field& field, Function function) {
  const auto visit = [&]<std::size_t I>() {
    const auto component = field.view().template component<I>();
    const auto halo = component.layout().halo();
    const auto interior = component.layout().interior_extents();
    for (int k = -static_cast<int>(halo.z_minus);
         k < static_cast<int>(interior.z() + halo.z_plus); ++k)
      for (int j = -static_cast<int>(halo.y_minus);
           j < static_cast<int>(interior.y() + halo.y_plus); ++j)
        for (int i = -static_cast<int>(halo.x_minus);
             i < static_cast<int>(interior.x() + halo.x_plus); ++i)
          function(component(i, j, k), I, i, j, k);
  };
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (visit.template operator()<I>(), ...);
  }(std::make_index_sequence<Field::component_count>{});
}

template<class Field>
std::vector<double> stamp(Field& field, double offset) {
  std::vector<double> expected;
  for_each_sample(field, [&](auto& value, std::size_t component,
                             int i, int j, int k) {
    using Value = std::remove_reference_t<decltype(value)>;
    value = static_cast<Value>(offset + 1000.0 * component +
                               100.0 * k + 10.0 * j + i + 0.125);
    expected.push_back(static_cast<double>(value));
  });
  return expected;
}

template<class Host, class Device>
void download_and_compare(Host& host, Device& device,
                          const std::vector<double>& expected, Executor executor) {
  for_each_sample(host, [](auto& value, auto...) { value = -99999; });
  auto result = lpfx::make_device_field_transfer(host.view(), device);
  require(result.has_value(), "native download preparation failed");
  auto transfer = std::move(result).value();
  const auto submitted = transfer.download(executor);
  require(submitted.is_success() && submitted.requires_conclusion() && transfer.pending(),
          "native field download was not submitted");
  const auto completed = transfer.wait();
  require(completed.is_success() && !transfer.pending(),
          "native field download did not complete");
  std::size_t index = 0;
  for_each_sample(host, [&](const auto& value, auto...) {
    require(index < expected.size() && static_cast<double>(value) == expected[index],
            "native field roundtrip changed an interior or halo sample");
    ++index;
  });
  require(index == expected.size(), "native field sample count changed");
}

template<class Precision>
void roundtrip(Executor executor) {
  tpic::config::SimulationParameters p;
  p.case_name = "Uniform";
  p.grid.nxc = 4; p.grid.nyc = 3; p.grid.nzc = 2;
  p.ns = 1;
  p.species.resize(1);
  p.species[0].q_over_m = -1.0;
  p.species[0].rho_init = 1.0;
  const auto domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  auto host = tpic::test::make_host_field_image<Precision>(p, domain);
  const auto electric = stamp(host.electric, 10.0);
  const auto magnetic = stamp(host.magnetic_cell, -3000.0);
  auto device = upload_image(host, executor);
  require(device.electric.valid() && device.electric.initialized() &&
              device.magnetic_cell.valid() && device.magnetic_cell.initialized(),
          "native upload returned unpublished device fields");
  download_and_compare(host.electric, device.electric, electric, executor);
  download_and_compare(host.magnetic_cell, device.magnetic_cell, magnetic, executor);
}

} // namespace

int main() {
  lpfx::device::DeviceProperties properties{};
  auto error = Traits::query_current_device_properties(properties);
  if (!Traits::succeeded(error)) {
    if (runtime_unavailable(error)) {
      std::cout << "TPIC native field upload skipped: no usable device/driver\n";
      return 77;
    }
    std::cerr << "TPIC native field upload: device query failed\n";
    return 1;
  }
  if (properties.device_ordinal < 0) return 1;
  auto stream = Traits::null_stream;
  error = Traits::create_stream(stream);
  if (!Traits::succeeded(error)) return runtime_unavailable(error) ? 77 : 1;
  if (stream == Traits::null_stream) return 1;
  int status = 0;
  try {
    const Executor executor{stream};
    roundtrip<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(executor);
    roundtrip<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(executor);
    roundtrip<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(executor);
    std::cout << "TPIC native field upload roundtrips passed\n";
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    status = 1;
  }
  if (!Traits::succeeded(Traits::destroy_stream(stream))) status = 1;
  return status;
}
