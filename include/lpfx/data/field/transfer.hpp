#ifndef LPFX_DATA_FIELD_TRANSFER_HPP
#define LPFX_DATA_FIELD_TRANSFER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/field/owner.hpp>
#include <lpfx/execution/device/completion.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {

namespace detail {

template <class Backend>
concept DeviceFieldTransferBackend =
    device::DeviceAllocationBackend<Backend> &&
    device::PinnedHostAllocationBackend<Backend> &&
    device::DeviceCopyBackend<Backend> && device::DeviceEventBackend<Backend>;

// One component's allocated samples as rows of row_bytes: host rows at
// host_pitch and device-layout rows (device memory, or the staging image) at
// device_pitch. The layout guarantees stride_z == stride_y * allocated y, so
// (y, z) rows flatten. Contiguous spans collapse to one row; an empty
// component has rows == 0.
struct FieldTransferRows final {
  std::byte *host = nullptr;
  std::byte *device = nullptr;
  std::size_t row_bytes = 0;
  std::size_t rows = 0;
  std::size_t host_pitch = 0;
  std::size_t device_pitch = 0;
  bool pitched = false;
};
static_assert(sizeof(FieldTransferRows) == 7 * sizeof(std::size_t) &&
              std::is_trivially_copyable_v<FieldTransferRows>);

template <class Backend> struct FieldTransferStaging final {
  HostMemoryLease admission{};
  device::PinnedHostAllocation<Backend> bytes{};
  std::byte *device_base = nullptr;
};

struct NoFieldTransferStaging final {};

enum class FieldTransferPhase : std::uint8_t {
  unprepared = 0,
  idle = 1,
  pending_upload = 2,
  pending_download = 3
};

// Host and device layouts must describe the same samples; pitches may differ.
template <class Host, class DevicePlan>
[[nodiscard]] constexpr bool
field_transfer_layouts_match(const Host &host,
                             const DevicePlan &device) noexcept {
  bool match = true;
  meta::for_each_index<DevicePlan::component_count>([&](const auto index) {
    constexpr std::size_t I = decltype(index)::value;
    const auto left = host.template component<I>().layout();
    const auto right = device.template component<I>().layout();
    match = match && left.interior_extents() == right.interior_extents() &&
            left.halo() == right.halo() &&
            left.allocation_extents() == right.allocation_extents();
  });
  return match;
}

template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ErrorCode try_bind_field_transfer_rows(
    std::byte *const host, const std::size_t host_pitch,
    std::byte *const device_base,
    const ComponentStorageReport<Location, IndexPolicy> &device,
    FieldTransferRows &output) noexcept {
  const auto extents = device.layout().allocation_extents();
  FieldTransferRows rows{
      host,
      device_base + device.byte_offset() + device.leading_prefix_bytes(),
      static_cast<std::size_t>(extents.x()) * device.element_bytes(),
      static_cast<std::size_t>(extents.y()) *
          static_cast<std::size_t>(extents.z()),
      host_pitch,
      device.row_pitch_bytes(),
      false};
  if (rows.row_bytes == 0 || rows.rows == 0) {
    output = {};
    return ErrorCode::success;
  }
  if (rows.rows == 1 ||
      (rows.host_pitch == rows.row_bytes && rows.device_pitch == rows.row_bytes)) {
    if (!checked_multiply(rows.row_bytes, rows.rows, rows.row_bytes)) {
      return ErrorCode::overflow;
    }
    rows.rows = 1;
  }
  output = rows;
  return ErrorCode::success;
}

template <bool Upload>
inline void copy_staged_field_rows(const FieldTransferRows &rows) noexcept {
  for (std::size_t row = 0; row != rows.rows; ++row) {
    std::byte *const host = rows.host + row * rows.host_pitch;
    std::byte *const staging = rows.device + row * rows.device_pitch;
    if constexpr (Upload) {
      std::memcpy(staging, host, rows.row_bytes);
    } else {
      std::memcpy(host, staging, rows.row_bytes);
    }
  }
}

template <bool Upload, class Backend>
inline void
submit_direct_field_rows(const FieldTransferRows &rows,
                         const device::Executor<Backend> executor) noexcept {
  using Traits = device::BackendTraits<Backend>;
  if (rows.rows == 0) {
    return;
  }
  std::byte *const destination = Upload ? rows.device : rows.host;
  const std::byte *const source = Upload ? rows.host : rows.device;
  const auto destination_pitch = Upload ? rows.device_pitch : rows.host_pitch;
  const auto source_pitch = Upload ? rows.host_pitch : rows.device_pitch;
  if (rows.pitched) {
    const auto error = Upload ? Traits::copy_host_to_device_2d_async(
                                    destination, destination_pitch, source,
                                    source_pitch, rows.row_bytes, rows.rows,
                                    executor.stream())
                              : Traits::copy_device_to_host_2d_async(
                                    destination, destination_pitch, source,
                                    source_pitch, rows.row_bytes, rows.rows,
                                    executor.stream());
    if (!Traits::succeeded(error)) {
      std::abort();
    }
    return;
  }
  for (std::size_t row = 0; row != rows.rows; ++row) {
    const auto error = Upload ? Traits::copy_host_to_device_async(
                                    destination + row * destination_pitch,
                                    source + row * source_pitch,
                                    rows.row_bytes, executor.stream())
                              : Traits::copy_device_to_host_async(
                                    destination + row * destination_pitch,
                                    source + row * source_pitch,
                                    rows.row_bytes, executor.stream());
    if (!Traits::succeeded(error)) {
      std::abort();
    }
  }
}

} // namespace detail

// Binds host samples to one device owner once; upload(executor),
// download(executor), conclude() and wait() then reuse the binding. Staged
// transfers pack host rows into one pinned device-layout image and move it
// with one contiguous native copy. Direct transfers copy between a pinned
// owner and the device with one native copy per row span (or one 2D copy).
// Both owners must remain alive and unmoved until destruction. A pending
// download writes the host samples in place; no host access is allowed
// until successful conclusion.
template <detail::CartesianIndexPolicy IndexPolicy,
          detail::DeviceFieldTransferBackend Backend, bool Staged,
          detail::FieldComponentType... Components>
  requires detail::FieldPack<DeviceSpace<Backend>, Components...>
class DeviceFieldTransfer final {
public:
  using device_owner_type =
      FieldOwner3D<IndexPolicy, field::storage::device<Backend>, Components...>;
  static constexpr std::size_t component_count = sizeof...(Components);
  static constexpr bool uses_staging = Staged;

  constexpr DeviceFieldTransfer() noexcept = default;

  DeviceFieldTransfer(DeviceFieldTransfer &&other) noexcept
      : staging_(std::move(other.staging_)), event_(std::move(other.event_)),
        state_(std::exchange(other.state_, {})) {}

  DeviceFieldTransfer &operator=(DeviceFieldTransfer &&) = delete;

  ~DeviceFieldTransfer() noexcept {
    if (pending()) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return state_.phase != detail::FieldTransferPhase::unprepared;
  }

  [[nodiscard]] constexpr bool pending() const noexcept {
    return state_.phase == detail::FieldTransferPhase::pending_upload ||
           state_.phase == detail::FieldTransferPhase::pending_download;
  }

  [[nodiscard]] constexpr std::size_t staging_capacity_bytes() const noexcept {
    if constexpr (Staged) {
      return staging_.bytes.capacity_bytes();
    } else {
      return 0;
    }
  }

  [[nodiscard]] Status
  upload(const device::Executor<Backend> executor) noexcept {
    return submit<true>(executor);
  }

  [[nodiscard]] Status
  download(const device::Executor<Backend> executor) noexcept {
    return submit<false>(executor);
  }

  [[nodiscard]] Status conclude() noexcept {
    if (!pending()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    const Status completed = event_.template advance<false>();
    if (completed.is_pending()) {
      return completed;
    }
    if (completed.is_failure()) {
      std::abort();
    }
    complete();
    return Status::success();
  }

  [[nodiscard]] Status wait() noexcept {
    if (!pending()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if (event_.template advance<true>().is_failure()) {
      std::abort();
    }
    complete();
    return Status::success();
  }

private:
  using rows_type = std::array<detail::FieldTransferRows, component_count>;
  using staging_type =
      std::conditional_t<Staged, detail::FieldTransferStaging<Backend>,
                         detail::NoFieldTransferStaging>;

  struct State {
    rows_type rows{};
    device_owner_type *device = nullptr;
    detail::FieldTransferPhase phase = detail::FieldTransferPhase::unprepared;
  };

  DeviceFieldTransfer(staging_type &&staging,
                      device::DeviceEvent<Backend> &&event,
                      const rows_type &rows, device_owner_type &device) noexcept
      : staging_(std::move(staging)), event_(std::move(event)),
        state_{rows, std::addressof(device), detail::FieldTransferPhase::idle} {}

  template <bool Upload>
  [[nodiscard]] Status
  submit(const device::Executor<Backend> executor) noexcept {
    if (state_.phase != detail::FieldTransferPhase::idle) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if constexpr (!Upload) {
      if (!state_.device->initialized()) {
        return Status::failure(ErrorCode::invalid_state);
      }
    }
    if (event_.empty()) {
      if constexpr (Upload) {
        state_.device->publish_initialized();
      }
      return Status::success();
    }
    if (!executor.bound()) {
      return Status::failure(ErrorCode::invalid_state);
    }
    if constexpr (Staged) {
      std::byte *const staging = staging_.bytes.raw_base();
      const std::size_t bytes = staging_.bytes.capacity_bytes();
      if constexpr (Upload) {
        for (const auto &rows : state_.rows) {
          detail::copy_staged_field_rows<true>(rows);
        }
        detail::copy_and_record<detail::DeviceCopyDirection::to_device>(
            staging_.device_base, staging, bytes, executor, event_);
      } else {
        detail::copy_and_record<detail::DeviceCopyDirection::to_host>(
            staging, staging_.device_base, bytes, executor, event_);
      }
    } else {
      for (const auto &rows : state_.rows) {
        detail::submit_direct_field_rows<Upload, Backend>(rows, executor);
      }
      if (!device::BackendTraits<Backend>::succeeded(event_.record(executor))) {
        std::abort();
      }
    }
    state_.phase = Upload ? detail::FieldTransferPhase::pending_upload
                          : detail::FieldTransferPhase::pending_download;
    return Status::accepted();
  }

  void complete() noexcept {
    if (state_.phase == detail::FieldTransferPhase::pending_upload) {
      state_.device->publish_initialized();
    } else if constexpr (Staged) {
      for (const auto &rows : state_.rows) {
        detail::copy_staged_field_rows<false>(rows);
      }
    }
    state_.phase = detail::FieldTransferPhase::idle;
  }

  [[no_unique_address]] staging_type staging_{};
  device::DeviceEvent<Backend> event_{};
  State state_{};

  friend struct detail::PrivateConstructionAccess;
};

// Staged transfer from any mutable host view with the device owner's
// samples. A non-null budget admits the staging image.
template <detail::CartesianIndexPolicy IndexPolicy,
          detail::DeviceFieldTransferBackend Backend,
          detail::FieldComponentType... Components>
[[nodiscard]] Result<DeviceFieldTransfer<IndexPolicy, Backend, true, Components...>>
make_device_field_transfer(
    const FieldPackView3D<IndexPolicy, HostSpace, false, Components...> host,
    FieldOwner3D<IndexPolicy, field::storage::device<Backend>, Components...>
        &device,
    HostMemoryBudget *const budget = nullptr) noexcept {
  using Transfer = DeviceFieldTransfer<IndexPolicy, Backend, true, Components...>;
  using Output = Result<Transfer>;
  using Traits = device::BackendTraits<Backend>;
  if (!device.valid() ||
      !detail::field_transfer_layouts_match(host, device.plan())) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  const std::size_t bytes = device.plan().total_bytes();
  detail::FieldTransferStaging<Backend> staging{};
  if (budget != nullptr) {
    HostMemoryRequirement requirement{};
    if (!requirement.try_add(bytes, alignof(std::max_align_t))) {
      return Output::from_error(ErrorCode::overflow);
    }
    auto admitted = budget->try_admit(requirement);
    LPFX_TRY_RESULT(admitted, Output);
    staging.admission = std::move(admitted).value();
  }
  if (!Traits::succeeded(
          device::PinnedHostAllocation<Backend>::create(bytes, staging.bytes))) {
    return Output::from_error(ErrorCode::allocation_failed);
  }
  device::DeviceEvent<Backend> event{};
  if (bytes != 0) {
    if (!Traits::succeeded(device::DeviceEvent<Backend>::create(event))) {
      return Output::from_error(ErrorCode::backend_failure);
    }
    std::memset(staging.bytes.raw_base(), 0, bytes);
  }
  staging.device_base = device.raw_base();

  std::array<detail::FieldTransferRows, sizeof...(Components)> rows{};
  const auto bind = [&](const auto index) -> ErrorCode {
    constexpr std::size_t I = decltype(index)::value;
    const auto component = host.template component<I>();
    const auto layout = component.layout();
    const auto &report = device.plan().template component<I>();
    return detail::try_bind_field_transfer_rows(
        reinterpret_cast<std::byte *>(component.core_data() -
                                      layout.core_offset()),
        static_cast<std::size_t>(layout.stride_y()) * report.element_bytes(),
        staging.bytes.raw_base(), report, rows[I]);
  };
  LPFX_TRY_RESULT(meta::for_each_until_error<sizeof...(Components)>(bind),
                  Output);
  return Output::from_value(detail::PrivateConstructionAccess::create<Transfer>(
      std::move(staging), std::move(event), rows, device));
}

// Direct transfer between a pinned host owner and a device owner. The current
// device must own the device allocation, and later executors must use it, so
// the native pitch limit queried here stays valid.
template <detail::CartesianIndexPolicy IndexPolicy,
          detail::DeviceFieldTransferBackend Backend,
          detail::FieldComponentType... Components>
[[nodiscard]] Result<DeviceFieldTransfer<IndexPolicy, Backend, false, Components...>>
make_device_field_transfer(
    FieldOwner3D<IndexPolicy, field::storage::pinned<Backend>, Components...>
        &host,
    FieldOwner3D<IndexPolicy, field::storage::device<Backend>, Components...>
        &device) noexcept {
  using Transfer =
      DeviceFieldTransfer<IndexPolicy, Backend, false, Components...>;
  using Output = Result<Transfer>;
  using Traits = device::BackendTraits<Backend>;
  if (!host.valid() || !device.valid() ||
      !detail::field_transfer_layouts_match(host.plan(), device.plan())) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  std::array<detail::FieldTransferRows, sizeof...(Components)> rows{};
  std::byte *const host_base = host.raw_base();
  std::byte *const device_base = device.raw_base();
  const auto bind = [&](const auto index) -> ErrorCode {
    constexpr std::size_t I = decltype(index)::value;
    const auto &report = host.plan().template component<I>();
    return detail::try_bind_field_transfer_rows(
        host_base + report.byte_offset() + report.leading_prefix_bytes(),
        report.row_pitch_bytes(), device_base,
        device.plan().template component<I>(), rows[I]);
  };
  LPFX_TRY_RESULT(meta::for_each_until_error<sizeof...(Components)>(bind),
                  Output);

  bool has_work = false;
  bool queried = false;
  std::size_t max_pitch = 0;
  for (auto &component : rows) {
    has_work = has_work || component.rows != 0;
    if (component.rows <= 1) {
      continue;
    }
    if (!queried) {
      if (!Traits::succeeded(
              Traits::query_host_device_max_copy_pitch(max_pitch))) {
        return Output::from_error(ErrorCode::backend_failure);
      }
      queried = true;
    }
    component.pitched = max_pitch != 0 &&
                        component.host_pitch <= max_pitch &&
                        component.device_pitch <= max_pitch;
  }
  device::DeviceEvent<Backend> event{};
  if (has_work &&
      !Traits::succeeded(device::DeviceEvent<Backend>::create(event))) {
    return Output::from_error(ErrorCode::backend_failure);
  }
  return Output::from_value(detail::PrivateConstructionAccess::create<Transfer>(
      detail::NoFieldTransferStaging{}, std::move(event), rows, device));
}

namespace detail {

// Initializes an uninitialized device owner from one pinned device-layout
// image with one native copy, waits for it and publishes the owner.
template <CartesianIndexPolicy IndexPolicy, DeviceFieldTransferBackend Backend,
          FieldComponentType... Components>
[[nodiscard]] ErrorCode try_upload_field_values(
    FieldOwner3D<IndexPolicy, field::storage::device<Backend>, Components...>
        &device,
    const FieldInitialValues<Components...> &values,
    const device::Executor<Backend> executor) noexcept {
  using Plan = PreparedFieldStorage<IndexPolicy, DeviceSpace<Backend>,
                                    Components...>;
  using Image =
      device::AlignedPinnedHostAllocation<Backend, Plan::allocation_alignment>;
  using Traits = device::BackendTraits<Backend>;
  const Plan &plan = device.plan();
  Image image{};
  LPFX_TRY(Image::create(plan.total_bytes(), image));
  device::DeviceEvent<Backend> event{};
  if (!Traits::succeeded(device::DeviceEvent<Backend>::create(event))) {
    return ErrorCode::backend_failure;
  }
  std::memset(image.raw_base(), 0, plan.total_bytes());
  std::tuple<typename Components::value_type *...> starts{};
  LPFX_TRY(try_initialize_field_components(image.raw_base(), plan, values,
                                           starts));
  copy_and_record<DeviceCopyDirection::to_device>(
      device.raw_base(), image.raw_base(), plan.total_bytes(), executor, event);
  if (!Traits::succeeded(event.synchronize())) {
    std::abort();
  }
  device.publish_initialized();
  return ErrorCode::success;
}

} // namespace detail

} // namespace lpfx

#endif
