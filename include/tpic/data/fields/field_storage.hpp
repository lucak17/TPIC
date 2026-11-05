#ifndef TPIC_DATA_FIELDS_FIELD_STORAGE_HPP
#define TPIC_DATA_FIELDS_FIELD_STORAGE_HPP

#include <lpfx/core/error_reporting.hpp>
#include <lpfx/api/field/field.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/output/gauss_sample.hpp>
#include <tpic/data/output/output_moment_schema.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tpic::data {

// The public low-level LPFx device owner of a field schema, and the owners of
// the field schemas, named once for every holder.
template <class Schema, class Backend>
using DeviceFieldOwner = typename lpfx::field::device<Schema, Backend>::storage_owner_type;
template <class Precision, class Backend>
using ElectricOwner = DeviceFieldOwner<typename FieldSchema<Precision>::E, Backend>;
template <class Precision, class Backend>
using MagneticCellOwner = DeviceFieldOwner<typename FieldSchema<Precision>::BCell, Backend>;
template <class Precision, class Backend>
using MagneticNodeOwner = DeviceFieldOwner<typename FieldSchema<Precision>::BNode, Backend>;
template <class Precision, class Backend>
using CurrentOwner = DeviceFieldOwner<typename FieldSchema<Precision>::Current, Backend>;
template <class Precision, class Backend>
using OutputMomentOwner = DeviceFieldOwner<typename OutputMomentSchema<Precision>::type, Backend>;
template <class Precision, class Backend>
using GaussCellOwner = DeviceFieldOwner<typename GaussFieldSchema<Precision>::Cell, Backend>;

// Holds the low-level LPFx owner because the LPFx convenience factory uploads
// zeros first and cannot adopt storage.
template <class Precision, class Backend> struct DeviceFieldImage {
  ElectricOwner<Precision, Backend> electric;
  MagneticCellOwner<Precision, Backend> magnetic_cell;
};

// Device storage plan of an initialized host image: component halos come from
// the host image, device alignment and pitches from LPFx.
template <class Schema, class Backend>
[[nodiscard]] auto prepare_device_field_plan(
    lpfx::field::host<Schema>& host, lpfx::device::Executor<Backend> executor) {
  if (!host.initialized() || !executor.bound())
    throw std::invalid_argument("TPIC field upload: initialized host and bound executor required");
  typename lpfx::field::device<Schema, Backend>::storage_plan_type plan;
  const auto halos = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return lpfx::make_field_halo_pack<lpfx::LogicalIndexPolicy32,
        typename Schema::template component<I>...>(
            host.template component_halo<I>()...);
  }(std::make_index_sequence<Schema::component_count>{});
  lpfx::check_error(lpfx::try_prepare_field_storage(host.cells(), halos, plan),
                    "TPIC field upload: LPFx device layout preparation failed");
  return plan;
}

// Cold startup operation: one final upload, checked completion, then release
// temporary staging. No device view escapes before initialized() is published.
// LPFx's submitted native transfer failures retain its fail-fast policy.
template <class Schema, class Backend>
[[nodiscard]] auto upload_field(lpfx::field::host<Schema>& host,
                                lpfx::device::Executor<Backend> executor) {
  const auto plan = prepare_device_field_plan(host, executor);
  auto made = lpfx::make_device_field(plan);
  lpfx::check_error(made, "TPIC field upload: LPFx device allocation failed");
  auto device = std::move(made).value();
  {
    auto prepared = lpfx::make_device_field_transfer(host.view(), device);
    lpfx::check_error(prepared, "TPIC field upload: LPFx transfer preparation failed");
    auto transfer = std::move(prepared).value();
    auto status = transfer.upload(executor);
    if (status.requires_conclusion()) status = transfer.wait();
    lpfx::require(
        lpfx::check_progress(status, "TPIC field upload: LPFx initialization did not complete") &&
            device.initialized(),
        "TPIC field upload: LPFx initialization did not complete");
  } // The borrower must be destroyed BEFORE the returned owner can move.
  return device;
}

// Startup scratch/derived fields use the same checked upload and publication
// as analytic fields. The temporary zero image is released before registration.
template <class Schema, class Backend>
[[nodiscard]] auto make_zero_field(
    const config::PreparedDomain& domain, lpfx::device::Executor<Backend> executor) {
  // prepare_device_field_plan checks the executor.
  if (!domain.field_neighborhood.local_partition().is_valid())
    throw std::invalid_argument("TPIC zero field: invalid domain");
  auto made = lpfx::field::make<Schema>(lpfx::field::options{
      .cells = {domain.local_cells[0], domain.local_cells[1], domain.local_cells[2]},
      .halo = config::PreparedDomain::field_halo});
  lpfx::check_error(made, "TPIC zero field: LPFx image allocation failed");
  auto host = std::move(made).value();
  return upload_field(host, executor);
}

} // namespace tpic::data

#endif // TPIC_DATA_FIELDS_FIELD_STORAGE_HPP
