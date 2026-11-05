#ifndef TPIC_DIAGNOSTICS_ASYNC_FIELD_SNAPSHOT_DEVICE_HPP
#define TPIC_DIAGNOSTICS_ASYNC_FIELD_SNAPSHOT_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <lpfx/core/attributes.hpp>
#include <tpic/device/thread_indexing.hpp>
#include <tpic/data/fields/grid_shape.hpp>
#include <tpic/config/output_selection.hpp>
#include <tpic/data/output/output_snapshot.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tpic::diagnostics {
namespace detail {

// A template so that the assertion fires only where it is instantiated: the
// CUDA host pass parses it, a HIP build never instantiates it.
template <class Backend = lpfx::device::ActiveBackend>
LPFX_ALWAYS_INLINE __device__ void accumulate_snapshot_value(const double value,
    const double scale, double* const packed, double* const total, const std::size_t position) {
  static_assert(data::fused_field_snapshot_accumulation<Backend>,
                "only CUDA has a verified contraction-free product");
#if defined(__CUDA_ARCH__)
  const double product = __dmul_rn(scale, value);
  if (packed != nullptr) packed[position] = product;
  total[position] += product;
#endif
}

} // namespace detail

namespace kernels {

template <class View, std::size_t... Component>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void pack_field_snapshot_kernel(
    View field, double* packed, double scale, data::GridShape shape) {
  constexpr std::size_t components = sizeof...(Component);
  lpfx::device::for_each_grid_index(static_cast<unsigned>(shape.count), [&](const unsigned index) {
    const auto [i, j, k] = device::decode_x_fastest(index, unsigned(shape.nx), unsigned(shape.ny));
    std::size_t position = components * static_cast<std::size_t>(index);
    ((packed[position++] = scale *
        static_cast<double>(field.template component<Component>()(i, j, k))), ...);
  });
}

// Two-launch accumulation (HIP): the pack kernel rounds scale * value to
// double in its store, and this kernel adds the stored values to the total.
// Internal linkage gives each translation unit its own copy of this
// non-template kernel; a unit that captures no field snapshot leaves it unused.
[[maybe_unused]] static __global__ __launch_bounds__(lpfx::device::default_block_threads) void
add_packed_field_snapshot_kernel(const double* source, double* total, std::size_t count) {
  const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += stride)
    total[i] += source[i];
}

// Fused accumulation (CUDA): __dmul_rn rounds scale * value before the
// addition, so the total receives the same double as on the two-launch path
// and the product cannot contract into a fused multiply-add. A non-null packed
// slab also receives the rounded product.
template <class View, std::size_t... Component>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void accumulate_field_snapshot_kernel(
    View field, double* packed, double* total, double scale, data::GridShape shape) {
  static_assert(sizeof(View) != 0 && data::fused_field_snapshot_accumulation<
                    lpfx::device::ActiveBackend>,
                "only CUDA has a verified contraction-free product");
  constexpr std::size_t components = sizeof...(Component);
  lpfx::device::for_each_grid_index(static_cast<unsigned>(shape.count), [&](const unsigned index) {
    const auto [i, j, k] = device::decode_x_fastest(index, unsigned(shape.nx), unsigned(shape.ny));
    std::size_t position = components * static_cast<std::size_t>(index);
    (detail::accumulate_snapshot_value(
         static_cast<double>(field.template component<Component>()(i, j, k)), scale,
         packed, total, position++), ...);
  });
}

} // namespace kernels

// One application snapshot: selected logical-node (or cell) arrays share an
// immutable device slab, one pinned download and one producer/transfer event
// pair. This uses TPIC's backend-neutral Float64, interleaved output layout.
// The capture and collection methods belong to one controller thread;
// device_wait_sources may run on another thread ordered after finish_capture
// by a host fence. The CPU writer receives only copied OwnedOutputSnapshot
// values after transfer completion.
template <class Backend> class AsyncFieldSnapshot {
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Allocation = lpfx::device::DeviceAllocation<Backend>;
  using Pinned = lpfx::device::PinnedHostAllocation<Backend>;
  enum class State { idle, capturing, transferring, ready };
  enum class Content { empty, packed, total, absent };
  struct Array {
    data::OutputArrayDescriptor descriptor;
    std::size_t offset{};
    std::size_t count{};
    Content content{Content::empty};
  };

public:
  using Executor = lpfx::device::Executor<Backend>;

  AsyncFieldSnapshot(const config::PreparedDomain& domain,
                     std::span<const data::OutputArrayDescriptor> descriptors,
                     Executor producer, Executor transfer, std::size_t scratch_components)
      : producer_executor_(producer), transfer_executor_(transfer) {
    lpfx::require(scratch_components == 0 || scratch_components == 1 || scratch_components == 3,
                  "TPIC field snapshot: invalid scratch component count");
    const auto cells = domain.field_neighborhood.local_partition().local_cells();
    lpfx::require(producer.bound() && transfer.bound() && !descriptors.empty() &&
                      data::try_make_grid_shape(cells, 1, {}, shape_) == lpfx::ErrorCode::success &&
                      data::try_make_grid_shape(cells, 0, {}, cell_shape_) == lpfx::ErrorCode::success,
                  "TPIC field snapshot: invalid domain, descriptors or executors");
    const auto nodes = node_count();
    lpfx::require(nodes <= std::numeric_limits<std::size_t>::max() / (3 * sizeof(double)),
                  "TPIC field snapshot: scratch allocation overflow");
    arrays_.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
      lpfx::require(!descriptor.name.empty() &&
                        (descriptor.components == 1 || descriptor.components == 3) &&
                        find(descriptor.name) == config::no_slot,
                    "TPIC field snapshot: invalid or duplicate array descriptor");
      const auto count = static_cast<std::size_t>(descriptor.components) *
                         static_cast<std::size_t>(descriptor.cell ? cell_shape_.count : shape_.count);
      lpfx::require(count <= std::numeric_limits<std::size_t>::max() / sizeof(double) - values_,
                    "TPIC field snapshot: selected allocation overflow");
      arrays_.push_back({descriptor, values_, count, Content::empty});
      values_ += count;
    }
    scratch_values_ = data::field_snapshot_scratch_components<Backend>(scratch_components) * nodes;
    lpfx::require(scratch_values_ <=
                      std::numeric_limits<std::size_t>::max() / sizeof(double) - values_,
                  "TPIC field snapshot: packing allocation overflow");
    lpfx::device::check_device_error<Backend>(
        Allocation::create((values_ + scratch_values_) * sizeof(double), device_),
        "TPIC field snapshot: storage or event allocation failed");
    lpfx::device::check_device_error<Backend>(
        Pinned::create(values_ * sizeof(double), host_),
        "TPIC field snapshot: storage or event allocation failed");
    lpfx::device::create_event(source_read_done_event_,
        "TPIC field snapshot: storage or event allocation failed");
    lpfx::device::create_event(transfer_done_event_,
        "TPIC field snapshot: storage or event allocation failed");
  }

  AsyncFieldSnapshot(const AsyncFieldSnapshot&) = delete;
  AsyncFieldSnapshot& operator=(const AsyncFieldSnapshot&) = delete;

  ~AsyncFieldSnapshot() noexcept {
    // Normal collection has completed both lanes. Exception unwinding may
    // interrupt capture or transfer submission before the final event exists.
    if (state_ != State::idle &&
        (!Traits::succeeded(Traits::synchronize_stream(producer_executor_.stream())) ||
         !Traits::succeeded(Traits::synchronize_stream(transfer_executor_.stream()))))
      std::abort();
  }

  [[nodiscard]] config::Slot find(std::string_view name) const noexcept {
    for (config::Slot slot = 0; slot < arrays_.size(); ++slot)
      if (arrays_[slot].descriptor.name == name) return slot;
    return config::no_slot;
  }
  [[nodiscard]] std::size_t node_count() const noexcept {
    return static_cast<std::size_t>(shape_.count);
  }
  [[nodiscard]] std::size_t device_bytes() const noexcept { return device_.capacity_bytes(); }
  [[nodiscard]] std::size_t pinned_bytes() const noexcept { return host_.capacity_bytes(); }
  [[nodiscard]] bool pending() const noexcept { return state_ != State::idle; }

  void begin() {
    lpfx::require(state_ == State::idle, "TPIC field snapshot: previous capture not collected");
    for (auto& array : arrays_) array.content = Content::empty;
    sources_recorded_ = false;
    state_ = State::capturing;
  }

  void clear(config::Slot slot) {
    auto& array = capture_array(slot);
    lpfx::require(array.content == Content::empty,
                  "TPIC field snapshot: array already initialized");
    lpfx::device::check_device_error<Backend>(
        Traits::memset_device_async(packed(array), 0, array.count * sizeof(double),
                                    producer_executor_.stream()),
        "TPIC field snapshot: total clear failed");
    array.content = Content::total;
  }

  template <std::size_t... Component, class Owner>
  void pack(const Owner& field, config::Slot slot, double scale = 1) {
    auto& array = capture_array(slot);
    lpfx::require(array.content == Content::empty,
                  "TPIC field snapshot: array already initialized");
    lpfx::require(array.descriptor.components == sizeof...(Component) &&
                      array.descriptor.cell == std::is_same_v<ecsim::field_location_t<Owner>, lpfx::Cell>,
                  "TPIC field snapshot: component count or location differs from descriptor");
    pack_to<Component...>(field, packed(array), scale);
    array.content = Content::packed;
  }

  // Leaves the array out of this capture: no values reach the writers.
  void omit(config::Slot slot) {
    auto& array = capture_array(slot);
    lpfx::require(array.content == Content::empty, "TPIC field snapshot: array already initialized");
    array.content = Content::absent;
  }

  // Adds scale * field to a cleared total whose species array is not selected.
  template <std::size_t... Component, class Owner>
  void add(const Owner& field, config::Slot total, double scale = 1) {
    auto& array = capture_array(total);
    lpfx::require(array.content == Content::total, "TPIC field snapshot: total was not cleared");
    lpfx::require(array.descriptor.components == sizeof...(Component),
                  "TPIC field snapshot: component count differs from descriptor");
    if constexpr (fused) {
      accumulate_to<Component...>(field, nullptr, packed(array), scale);
    } else {
      lpfx::require(array.count <= scratch_values_,
                    "TPIC field snapshot: contribution exceeds planned scratch");
      pack_to<Component...>(field, device_values() + values_, scale);
      add_values(device_values() + values_, array);
    }
  }

  // Packs a selected species array and adds the same rounded values to its
  // cleared total: one launch on CUDA, pack then add on HIP. This is the only
  // way a packed species array contributes to its total.
  template <std::size_t... Component, class Owner>
  void pack_and_add(const Owner& field, config::Slot slot, config::Slot total, double scale = 1) {
    auto& species = capture_array(slot);
    auto& sum = capture_array(total);
    lpfx::require(slot != total && species.content == Content::empty &&
                      sum.content == Content::total && species.count == sum.count &&
                      species.descriptor.components == sizeof...(Component),
                  "TPIC field snapshot: invalid packed total contribution");
    if constexpr (fused) {
      accumulate_to<Component...>(field, packed(species), packed(sum), scale);
    } else {
      pack_to<Component...>(field, packed(species), scale);
      add_values(packed(species), sum);
    }
    species.content = Content::packed;
  }

  void finish_capture() {
    lpfx::require(state_ == State::capturing, "TPIC field snapshot: no active capture");
    for (const auto& array : arrays_)
      lpfx::require(array.content != Content::empty,
                    "TPIC field snapshot: selected array not initialized");
    lpfx::device::record_event(source_read_done_event_, producer_executor_,
        "TPIC field snapshot: source-read event failed");
    sources_recorded_ = true;
    // Establish failure cleanup ownership before either cross-stream wait or
    // async copy can be accepted by the backend.
    state_ = State::transferring;
    lpfx::device::wait_event(source_read_done_event_, transfer_executor_,
        "TPIC field snapshot: transfer submission failed");
    lpfx::device::check_device_error<Backend>(
        Traits::copy_device_to_host_async(host_.raw_base(), device_.raw_base(),
                                          values_ * sizeof(double), transfer_executor_.stream()),
        "TPIC field snapshot: transfer submission failed");
    lpfx::device::record_event(transfer_done_event_, transfer_executor_,
        "TPIC field snapshot: transfer submission failed");
  }

  [[nodiscard]] bool ready() {
    lpfx::require(state_ == State::transferring || state_ == State::ready,
                  "TPIC field snapshot: transfer was not submitted");
    if (state_ == State::ready) return true;
    const auto status = transfer_done_event_.query();
    if (Traits::not_ready(status)) return false;
    lpfx::device::check_device_error<Backend>(status, "TPIC field snapshot: transfer query failed");
    state_ = State::ready;
    return true;
  }

  void host_wait() {
    lpfx::require(state_ == State::transferring || state_ == State::ready,
                  "TPIC field snapshot: transfer was not submitted");
    lpfx::device::check_device_error<Backend>(transfer_done_event_.synchronize(),
                                              "TPIC field snapshot: transfer completion failed");
    state_ = State::ready;
  }

  void device_wait_sources(Executor executor) const {
    lpfx::require(sources_recorded_ && executor.bound(),
                  "TPIC field snapshot: source-read completion was not recorded");
    lpfx::device::wait_event(source_read_done_event_, executor,
        "TPIC field snapshot: source consumer dependency failed");
  }

  // Does not block. The controller polls ready() or waits explicitly first.
  // Existing matching host vectors retain their capacity across snapshots;
  // epoch metadata belongs to the controller and is deliberately untouched.
  void copy_into(data::OwnedOutputSnapshot& snapshot) {
    lpfx::require(ready(), "TPIC field snapshot: transfer is not complete");
    const auto* values = reinterpret_cast<const double*>(host_.raw_base());
    snapshot.arrays.resize(arrays_.size());
    for (config::Slot slot = 0; slot < arrays_.size(); ++slot) {
      const auto& array = arrays_[slot];
      auto& output = snapshot.arrays[slot];
      output.name = array.descriptor.name;
      output.components = array.descriptor.components;
      output.cell = array.descriptor.cell;
      if (array.content == Content::absent) {
        output.values.clear();
        continue;
      }
      const auto* first = values + array.offset;
      lpfx::require(std::all_of(first, first + array.count, [](double value) { return std::isfinite(value); }),
                    "TPIC field snapshot: nonfinite field value");
      output.values.assign(first, first + array.count);
    }
    state_ = State::idle;
  }

private:
  Array& capture_array(config::Slot slot) {
    lpfx::require(state_ == State::capturing, "TPIC field snapshot: no active capture");
    lpfx::require(slot < arrays_.size(), "TPIC field snapshot: invalid array slot");
    return arrays_[slot];
  }
  double* device_values() noexcept { return reinterpret_cast<double*>(device_.raw_base()); }
  double* packed(const Array& array) noexcept { return device_values() + array.offset; }

  static constexpr bool fused = data::fused_field_snapshot_accumulation<Backend>;

  template <std::size_t... Component, class Owner>
  void require_source(const Owner& field, double scale) const {
    static_assert(sizeof...(Component) == 1 || sizeof...(Component) == 3);
    static_assert(((Component < Owner::component_count) && ...));
    static_assert(std::is_same_v<Backend, typename Owner::space_type::backend_type>);
    static_assert(ecsim::FieldAt<Owner, ecsim::field_location_t<Owner>>);
    lpfx::require(field.initialized() && field.cells().x() == cell_shape_.nx &&
                      field.cells().y() == cell_shape_.ny && field.cells().z() == cell_shape_.nz &&
                      std::isfinite(scale),
                  "TPIC field snapshot: invalid field owner or scale");
  }
  template <std::size_t... Component, class Owner>
  void pack_to(const Owner& field, double* destination, double scale) {
    require_source<Component...>(field, scale);
    const auto& shape = std::is_same_v<ecsim::field_location_t<Owner>, lpfx::Cell> ? cell_shape_ : shape_;
    lpfx::device::launch_or_throw<lpfx::device::KernelLaunch<>,
        kernels::pack_field_snapshot_kernel<decltype(field.view()), Component...>>(
        "TPIC field snapshot: kernel launch failed", producer_executor_, shape.count,
        field.view(), destination, scale, shape);
  }
  template <std::size_t... Component, class Owner>
  void accumulate_to(const Owner& field, double* destination, double* total, double scale) {
    static_assert(std::is_same_v<ecsim::field_location_t<Owner>, lpfx::Node>);
    require_source<Component...>(field, scale);
    lpfx::device::launch_or_throw<lpfx::device::KernelLaunch<>,
        kernels::accumulate_field_snapshot_kernel<decltype(field.view()), Component...>>(
        "TPIC field snapshot: kernel launch failed", producer_executor_, shape_.count,
        field.view(), destination, total, scale, shape_);
  }
  void add_values(const double* source, Array& total) {
    lpfx::device::launch_or_throw<lpfx::device::KernelLaunch<>, kernels::add_packed_field_snapshot_kernel>(
        "TPIC field snapshot: kernel launch failed", producer_executor_, total.count,
        source, packed(total), total.count);
  }

  Executor producer_executor_, transfer_executor_;
  data::GridShape shape_{}, cell_shape_{};
  std::vector<Array> arrays_;
  std::size_t values_{}, scratch_values_{};
  Allocation device_;
  Pinned host_;
  lpfx::device::DeviceEvent<Backend> source_read_done_event_, transfer_done_event_;
  State state_{State::idle};
  bool sources_recorded_{};
};

} // namespace tpic::diagnostics

#endif // TPIC_DIAGNOSTICS_ASYNC_FIELD_SNAPSHOT_DEVICE_HPP
