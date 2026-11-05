#ifndef LPFX_DATA_FIELD_OWNER_HPP
#define LPFX_DATA_FIELD_OWNER_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/field/view.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx {

namespace field::storage {

struct pageable final {};

template <class Backend> struct pinned final {
  using backend_type = Backend;
};

template <class Backend> struct device final {
  using backend_type = Backend;
};

} // namespace field::storage

// One element value per component, in pack order.
template <detail::FieldComponentType... Components>
using FieldInitialValues = std::tuple<typename Components::value_type...>;

namespace detail {

template <class Storage> struct FieldStorageTraits;

template <> struct FieldStorageTraits<field::storage::pageable> final {
  using space_type = HostSpace;
  template <std::size_t Alignment>
  using allocation_type = RawHostAllocation<HostAlignedAllocator<Alignment>>;
};

template <device::PinnedHostAllocationBackend Backend>
struct FieldStorageTraits<field::storage::pinned<Backend>> final {
  using space_type = HostSpace;
  template <std::size_t Alignment>
  using allocation_type = device::AlignedPinnedHostAllocation<Backend, Alignment>;
};

template <device::DeviceAllocationBackend Backend>
struct FieldStorageTraits<field::storage::device<Backend>> final {
  using space_type = DeviceSpace<Backend>;
  template <std::size_t>
  using allocation_type = device::DeviceAllocation<Backend>;
};

template <class Storage>
concept FieldStorage =
    requires { typename FieldStorageTraits<Storage>::space_type; };

template <FieldStorage Storage>
using field_storage_space_t = typename FieldStorageTraits<Storage>::space_type;

} // namespace detail

// One allocation holding every component block of a prepared plan, and the
// permanently bound view. Host owners are initialized on creation; a device
// owner becomes initialized once a completed write publishes it.
template <detail::CartesianIndexPolicy IndexPolicy, class Storage,
          detail::FieldComponentType... Components>
  requires detail::FieldStorage<Storage> &&
           detail::FieldPack<detail::field_storage_space_t<Storage>,
                             Components...>
class FieldOwner3D final {
public:
  using index_policy = IndexPolicy;
  using storage_type = Storage;
  using space_type = detail::field_storage_space_t<Storage>;
  using plan_type = PreparedFieldStorage<IndexPolicy, space_type, Components...>;
  using view_type = FieldPackView3D<IndexPolicy, space_type, false, Components...>;
  using const_view_type =
      FieldPackView3D<IndexPolicy, space_type, true, Components...>;
  using allocation_type = typename detail::FieldStorageTraits<
      Storage>::template allocation_type<plan_type::allocation_alignment>;
  using count_type = typename IndexPolicy::count_type;
  static constexpr std::size_t component_count = sizeof...(Components);

  constexpr FieldOwner3D() noexcept = default;

  FieldOwner3D(FieldOwner3D &&other) noexcept
      : allocation_(std::move(other.allocation_)),
        state_(std::exchange(other.state_, {})) {}

  FieldOwner3D &operator=(FieldOwner3D &&other) noexcept {
    if (this != &other) {
      allocation_ = std::move(other.allocation_);
      state_ = std::exchange(other.state_, {});
    }
    return *this;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return state_.plan.valid();
  }

  [[nodiscard]] constexpr bool initialized() const noexcept {
    return state_.published;
  }

  [[nodiscard]] constexpr CartesianExtent3<count_type> cells() const noexcept {
    CartesianExtent3<count_type> result{};
    if (valid()) {
      using First = std::tuple_element_t<0, std::tuple<Components...>>;
      static_cast<void>(
          try_make_location_cell_extents<typename First::location_type,
                                         IndexPolicy>(
              state_.plan.template component<0>().layout().interior_extents(),
              result));
    }
    return result;
  }

  template <std::size_t I>
    requires(I < component_count)
  [[nodiscard]] constexpr HaloDepth3<count_type>
  component_halo() const noexcept {
    return state_.plan.template component<I>().layout().halo();
  }

  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }

  [[nodiscard]] constexpr const plan_type &plan() const noexcept {
    return state_.plan;
  }

  // Hard precondition for device owners: initialized(), or the caller is the
  // write that initializes the storage.
  [[nodiscard]] constexpr view_type view() noexcept { return state_.view; }

  [[nodiscard]] constexpr const_view_type view() const noexcept {
    return state_.view;
  }

  [[nodiscard]] std::byte *raw_base() noexcept { return allocation_.raw_base(); }

  // Hard precondition: valid(), and a completed write has initialized every
  // sample of the view. Re-publication is idempotent.
  void publish_initialized() noexcept { state_.published = true; }

  // Hard precondition: no submitted operation still references the storage.
  // A device owner returns the native release status.
  auto reset_capacity() noexcept {
    state_ = {};
    if constexpr (detail::is_device_space<space_type>) {
      return allocation_.reset();
    } else {
      allocation_.reset_capacity();
    }
  }

private:
  struct State {
    view_type view{};
    plan_type plan{};
    bool published = false;
  };

  FieldOwner3D(allocation_type &&allocation, const view_type view,
               const plan_type &plan, const bool published) noexcept
      : allocation_(std::move(allocation)), state_{view, plan, published} {}

  allocation_type allocation_{};
  State state_{};

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

template <class View, class Plan, class... T, std::size_t... I>
[[nodiscard]] constexpr View
make_field_view(const std::tuple<T *...> &starts, const Plan &plan,
                std::index_sequence<I...>) noexcept {
  return PrivateConstructionAccess::create<View>(
      make_component_view<typename Plan::space_type>(
          std::get<I>(starts), plan.template component<I>().layout())...);
}

// Starts every component lifetime in host storage and writes its initial
// value to every allocated sample; prefixes and row padding keep their bytes.
template <CartesianIndexPolicy IndexPolicy, class Space,
          FieldComponentType... Components>
[[nodiscard]] ErrorCode try_initialize_field_components(
    std::byte *const base,
    const PreparedFieldStorage<IndexPolicy, Space, Components...> &plan,
    const FieldInitialValues<Components...> &values,
    std::tuple<typename Components::value_type *...> &starts) noexcept {
  const auto initialize = [&](const auto index) -> ErrorCode {
    constexpr std::size_t I = decltype(index)::value;
    using T = std::tuple_element_t<I, FieldInitialValues<Components...>>;
    const auto &report = plan.template component<I>();
    T *&start = std::get<I>(starts);
    if (!try_start_host_array_lifetime<T>(base + report.byte_offset(),
                                          report.block_bytes() / sizeof(T),
                                          start)) {
      return ErrorCode::invalid_state;
    }
    const auto layout = report.layout();
    const auto extents = layout.allocation_extents();
    const std::size_t rows = static_cast<std::size_t>(extents.y()) *
                             static_cast<std::size_t>(extents.z());
    const std::size_t pitch = static_cast<std::size_t>(layout.stride_y());
    T *const first = start + static_cast<std::size_t>(layout.leading_prefix());
    for (std::size_t row = 0; row != rows; ++row) {
      std::fill_n(first + row * pitch, static_cast<std::size_t>(extents.x()),
                  std::get<I>(values));
    }
    return ErrorCode::success;
  };
  return meta::for_each_until_error<sizeof...(Components)>(initialize);
}

template <class Storage, class Plan>
[[nodiscard]] Result<typename FieldStorageTraits<Storage>::template allocation_type<
    Plan::allocation_alignment>>
allocate_host_field(const Plan &plan, HostMemoryBudget *const budget) noexcept {
  constexpr std::size_t alignment = Plan::allocation_alignment;
  using Allocation = typename FieldStorageTraits<
      Storage>::template allocation_type<alignment>;
  using Output = Result<Allocation>;
  if constexpr (std::same_as<Storage, field::storage::pageable>) {
    using Allocator = HostAlignedAllocator<alignment>;
    if (budget == nullptr) {
      return allocate_host_raw_unadmitted<Allocator>(plan.total_bytes());
    }
    return allocate_single_host_layout<Allocator>(*budget, plan);
  } else {
    if (budget == nullptr) {
      Allocation allocation{};
      LPFX_TRY_RESULT(Allocation::create(plan.total_bytes(), allocation),
                      Output);
      return Output::from_value(std::move(allocation));
    }
    std::size_t requested = 0;
    HostMemoryRequirement requirement{};
    if (!Allocation::try_allocation_bytes(plan.total_bytes(), requested) ||
        !requirement.try_add(requested, alignment)) {
      return Output::from_error(ErrorCode::overflow);
    }
    auto admission = budget->try_admit(requirement);
    LPFX_TRY_RESULT(admission, Output);
    return Allocation::allocate(std::move(admission).value(),
                                plan.total_bytes());
  }
}

} // namespace detail

// Allocates and initializes a host owner. A non-null budget admits the
// allocation for the owner's lifetime; otherwise it is unadmitted.
template <class Storage = field::storage::pageable,
          detail::CartesianIndexPolicy IndexPolicy,
          detail::FieldComponentType... Components>
  requires std::same_as<detail::field_storage_space_t<Storage>, HostSpace> &&
           detail::FieldPack<HostSpace, Components...>
[[nodiscard]] Result<FieldOwner3D<IndexPolicy, Storage, Components...>>
make_host_field(
    const PreparedFieldStorage<IndexPolicy, HostSpace, Components...> &plan,
    const FieldInitialValues<Components...> &values,
    HostMemoryBudget *const budget = nullptr) noexcept {
  using Owner = FieldOwner3D<IndexPolicy, Storage, Components...>;
  using Output = Result<Owner>;
  if (!plan.valid()) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  auto allocated = detail::allocate_host_field<Storage>(plan, budget);
  LPFX_TRY_RESULT(allocated, Output);
  auto allocation = std::move(allocated).value();
  std::tuple<typename Components::value_type *...> starts{};
  LPFX_TRY_RESULT(detail::try_initialize_field_components(
                      allocation.raw_base(), plan, values, starts),
                  Output);
  const auto view = detail::make_field_view<typename Owner::view_type>(
      starts, plan, std::index_sequence_for<Components...>{});
  return Output::from_value(detail::PrivateConstructionAccess::create<Owner>(
      std::move(allocation), view, plan, true));
}

// Allocates an uninitialized device owner.
template <detail::CartesianIndexPolicy IndexPolicy,
          device::DeviceAllocationBackend Backend,
          detail::FieldComponentType... Components>
  requires detail::FieldPack<DeviceSpace<Backend>, Components...>
[[nodiscard]] Result<
    FieldOwner3D<IndexPolicy, field::storage::device<Backend>, Components...>>
make_device_field(const PreparedFieldStorage<IndexPolicy, DeviceSpace<Backend>,
                                             Components...> &plan) noexcept {
  using Owner =
      FieldOwner3D<IndexPolicy, field::storage::device<Backend>, Components...>;
  using Output = Result<Owner>;
  using Allocation = typename Owner::allocation_type;
  using Traits = device::BackendTraits<Backend>;
  if (!plan.valid() ||
      Owner::plan_type::allocation_alignment >
          Traits::minimum_device_allocation_alignment) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  Allocation allocation{};
  if (!Traits::succeeded(Allocation::create(plan.total_bytes(), allocation))) {
    return Output::from_error(ErrorCode::allocation_failed);
  }
  std::byte *const base = allocation.raw_base();
  const std::tuple<typename Components::value_type *...> starts{
      reinterpret_cast<typename Components::value_type *>(
          base + plan.template component<
                     meta::index_of<Components, Components...>>()
                     .byte_offset())...};
  const auto view = detail::make_field_view<typename Owner::view_type>(
      starts, plan, std::index_sequence_for<Components...>{});
  return Output::from_value(detail::PrivateConstructionAccess::create<Owner>(
      std::move(allocation), view, plan, false));
}

} // namespace lpfx

#endif
