#include <tpic/data/fields/field_storage.hpp>
#include <tpic/config/domain.hpp>
#include <device/mock_backend.hpp>
#include <tpic/initialization/fields/field_image.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include "../support/test_builders.hpp"

namespace {

// The startup upload of the Simulation constructor: E, then B, in one braced list.
template <class Precision, class Backend>
tpic::data::DeviceFieldImage<Precision, Backend> upload_image(
    tpic::initialization::HostFieldImage<Precision>& host, lpfx::device::Executor<Backend> executor) {
  return {tpic::data::upload_field(host.electric, executor),
          tpic::data::upload_field(host.magnetic_cell, executor)};
}

using Backend = lpfx::device::MockBackend;
using Mock = lpfx::device::BackendTraits<Backend>;
using Executor = lpfx::device::Executor<Backend>;
constexpr Executor executor{lpfx::device::MockStream{1}};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template<class Exception, class Function>
void rejects(Function&& function, const char* message) {
  try {
    function();
  } catch (const Exception& error) {
    require(std::string(error.what()).starts_with("TPIC"),
            "upload failure has no TPIC context");
    return;
  }
  throw std::runtime_error(message);
}

tpic::config::SimulationParameters parameters() {
  tpic::config::SimulationParameters p;
  p.case_name = "Uniform";
  p.grid.nxc = 4; p.grid.nyc = 3; p.grid.nzc = 2;
  p.ns = 1;
  p.species.resize(1);
  p.species[0].q_over_m = -1.0;
  p.species[0].rho_init = 1.0;
  p.b0x = 0.125; p.b0y = -0.25; p.b0z = 0.5;
  return p;
}

template<class View, class Function>
void for_each_sample(View view, Function function) {
  const auto visit = [&]<std::size_t I>() {
    const auto component = view.template component<I>();
    const auto halo = component.layout().halo();
    const auto interior = component.layout().interior_extents();
    using Coordinate = typename decltype(component)::coordinate_type;
    for (Coordinate k = -static_cast<Coordinate>(halo.z_minus);
         k < static_cast<Coordinate>(interior.z() + halo.z_plus); ++k)
      for (Coordinate j = -static_cast<Coordinate>(halo.y_minus);
           j < static_cast<Coordinate>(interior.y() + halo.y_plus); ++j)
        for (Coordinate i = -static_cast<Coordinate>(halo.x_minus);
             i < static_cast<Coordinate>(interior.x() + halo.x_plus); ++i)
          function(component(i, j, k), I, i, j, k);
  };
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (visit.template operator()<I>(), ...);
  }(std::make_index_sequence<View::component_count>{});
}

template<class Field>
void stamp(Field& field, double offset) {
  // Coordinate-dependent, exactly representable values exercise every logical
  // sample and component, including all halo corners and padded rows.
  for_each_sample(field.view(), [=](auto& value, std::size_t component,
                                    auto i, auto j, auto k) {
    using Value = std::remove_reference_t<decltype(value)>;
    value = static_cast<Value>(offset + 1000.0 * component +
                               100.0 * k + 10.0 * j + i + 0.125);
  });
}

template<class Host, class Device>
void same_samples(const Host& host, const Device& device) {
  require(device.cells() == host.cells(), "device cells differ from host");
  const auto compare = [&]<std::size_t I>() {
    const auto expected = host.view().template component<I>();
    const auto actual = device.view().template component<I>();
    require(device.template component_halo<I>() == expected.layout().halo(),
            "device halo differs from host");
    const auto halo = expected.layout().halo();
    const auto extents = expected.layout().interior_extents();
    for (int k = -static_cast<int>(halo.z_minus);
         k < static_cast<int>(extents.z() + halo.z_plus); ++k)
      for (int j = -static_cast<int>(halo.y_minus);
           j < static_cast<int>(extents.y() + halo.y_plus); ++j)
        for (int i = -static_cast<int>(halo.x_minus);
             i < static_cast<int>(extents.x() + halo.x_plus); ++i)
          // This read is permitted only because the test backend owns host
          // memory. Native device verification uses an explicit download.
          require(actual(i, j, k) == expected(i, j, k),
                  "uploaded field sample differs from host image");
  };
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (compare.template operator()<I>(), ...);
  }(std::make_index_sequence<Host::component_count>{});
}

template<class Precision>
void image_upload() {
  Mock::reset();
  const auto p = parameters();
  const auto domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  auto host = tpic::test::make_host_field_image<Precision>(p, domain);
  stamp(host.electric, 10.0);
  stamp(host.magnetic_cell, -3000.0);
  {
    auto device = upload_image(host, executor);
    require(device.electric.valid() && device.electric.initialized() &&
                device.magnetic_cell.valid() && device.magnetic_cell.initialized(),
            "upload returned unpublished device owners");
    require(Mock::host_to_device_copies == 2 && Mock::device_to_host_copies == 0 &&
                Mock::device_to_device_copies == 0 && Mock::fill_calls == 0,
            "startup must issue exactly one upload per field");
    require(Mock::host_to_device_bytes == device.electric.capacity_bytes() +
                                              device.magnetic_cell.capacity_bytes(),
            "upload byte count differs from final device allocations");
    require(Mock::event_records == 2 && Mock::event_synchronizations == 2 &&
                Mock::event_ready,
            "upload returned before both pending events were waited");
    require(Mock::event_creations == 2 && Mock::event_destructions == 2 &&
                Mock::pinned_allocations == 2 && Mock::pinned_deallocations == 2,
            "startup retained temporary transfer resources");
    require(Mock::device_allocations == 2 && Mock::device_deallocations == 0,
            "final device owners were not retained");
    same_samples(host.electric, device.electric);
    same_samples(host.magnetic_cell, device.magnetic_cell);
  }
  require(Mock::device_deallocations == 2, "device image leaked allocations");
}

void asymmetric_halos_and_pending_transfer() {
  Mock::reset();
  using X = lpfx::field::component<"x", double, lpfx::field::node,
                                  lpfx::AlignedPitch<256>>;
  using Y = lpfx::field::component<"y", double, lpfx::field::node,
                                  lpfx::CoreAlignedPitch<256>>;
  using Z = lpfx::field::component<"z", double, lpfx::field::node>;
  using Schema = lpfx::field::schema<X, Y, Z>;
  auto result = lpfx::field::make<Schema>(lpfx::field::shape{
      lpfx::cells{3, 2, 2}, lpfx::field::halos::per_component(
          lpfx::field::halos::for_component<X>(
              lpfx::field::halos::asymmetric(2, 1, 0, 2, 1, 0)),
          lpfx::field::halos::for_component<Y>(
              lpfx::field::halos::asymmetric(0, 2, 1, 0, 2, 1)),
          lpfx::field::halos::for_component<Z>(
              lpfx::field::halos::uniform(1)))}, lpfx::field::values<Schema>());
  require(result.has_value(), "failed to create asymmetric host fixture");
  auto host = std::move(result).value();
  stamp(host, 20.0);
  auto device = tpic::data::upload_field(host, executor);
  same_samples(host, device);
  const auto plan = tpic::data::prepare_device_field_plan(host, executor);
  auto uninitialized = lpfx::make_device_field(plan);
  require(uninitialized.has_value(), "failed to prepare pending transfer owner");
  auto pending_device = std::move(uninitialized).value();
  auto prepared = lpfx::make_device_field_transfer(host.view(), pending_device);
  require(prepared.has_value(), "failed to prepare pending transfer");
  auto transfer = std::move(prepared).value();
  const auto submitted = transfer.upload(executor);
  require(submitted.is_success() && submitted.requires_conclusion() &&
              transfer.pending() && !pending_device.initialized() && !Mock::event_ready,
          "LPFx upload published the field before completion");
  const auto completed = transfer.wait();
  require(completed.is_success() && !completed.requires_conclusion() &&
              !transfer.pending() && pending_device.initialized() && Mock::event_ready,
          "LPFx wait did not publish the completed field");
  same_samples(host, pending_device);
}

void preparation_failures() {
  const auto p = parameters();
  const auto domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  auto host = tpic::test::make_host_field_image(p, domain);
  Mock::reset();
  rejects<std::invalid_argument>([&] {
    (void)upload_image(host, Executor{Mock::null_stream});
  }, "unbound stream was accepted");
  require(Mock::device_allocation_attempts == 0 && Mock::pinned_allocations == 0 &&
              Mock::event_creations == 0 && Mock::copy_calls == 0,
          "invalid stream reached backend allocation or submission");
  lpfx::field::host<tpic::data::FieldSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>::E> invalid;
  rejects<std::invalid_argument>([&] {
    (void)tpic::data::upload_field(invalid, executor);
  }, "invalid host owner was accepted");
  require(Mock::device_allocation_attempts == 0,
          "invalid host owner reached device allocation");

  for (int failure = 0; failure < 3; ++failure) {
    Mock::reset();
    if (failure == 0) Mock::fail_next_allocation = true;
    if (failure == 1) Mock::pinned_offset = Mock::pinned_storage.size();
    if (failure == 2) Mock::fail_next_event = true;
    rejects<std::runtime_error>([&] {
      (void)upload_image(host, executor);
    }, "backend preparation failure was accepted");
    require(Mock::copy_calls == 0 && Mock::event_records == 0,
            "failed preparation submitted device work");
    require(Mock::device_allocations == Mock::device_deallocations &&
                Mock::pinned_allocations == Mock::pinned_deallocations &&
                Mock::event_creations == Mock::event_destructions,
            "failed preparation leaked resources");
  }

  Mock::reset();
  const auto plan = tpic::data::prepare_device_field_plan(host.electric, executor);
  constexpr auto alignment = Mock::minimum_device_allocation_alignment;
  // Leave room for exactly the first field. The second allocation fails after
  // E's upload completed, exercising cleanup of the partly constructed image.
  Mock::device_offset = (Mock::device_storage.size() - plan.total_bytes()) /
                        alignment * alignment;
  rejects<std::runtime_error>([&] {
    (void)upload_image(host, executor);
  }, "second field allocation failure was accepted");
  require(Mock::device_allocation_attempts == 2 && Mock::device_allocations == 1 &&
              Mock::host_to_device_copies == 1 && Mock::event_synchronizations == 1,
          "partial construction fixture did not fail after the first completed upload");
  require(Mock::device_allocations == Mock::device_deallocations &&
              Mock::pinned_allocations == Mock::pinned_deallocations &&
              Mock::event_creations == Mock::event_destructions,
          "partly constructed device image leaked resources");
}

} // namespace

int main() {
  try {
    image_upload<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>();
    image_upload<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>();
    image_upload<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>();
    asymmetric_halos_and_pending_transfer();
    preparation_failures();
    std::cout << "TPIC field upload mock checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
