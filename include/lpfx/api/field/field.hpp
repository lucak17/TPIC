#ifndef LPFX_API_FIELD_FIELD_HPP
#define LPFX_API_FIELD_FIELD_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/api/field/descriptors.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/field/owner.hpp>
#include <lpfx/data/field/transfer.hpp>
#include <lpfx/domain/cartesian/index_policy.hpp>
#include <lpfx/execution/host/serial.hpp>

namespace lpfx::field {

template <class Backend>
inline constexpr storage::pinned<Backend> pinned{};

namespace detail {

template <class... Left, class... Right>
constexpr meta::type_list<Left..., Right...>
operator+(meta::type_list<Left...>, meta::type_list<Right...>) noexcept {
  return {};
}

// Requested components in the order of Available.
template <class... Requested, class... Available>
auto canonical_selection(meta::type_list<Available...>)
    -> decltype((meta::type_list<>{} + ... +
                 std::conditional_t<meta::contains<Available, Requested...>,
                                    meta::type_list<Available>,
                                    meta::type_list<>>{}));

template <class IndexPolicy, class Space, bool Const, class... Available,
          class... Selected>
[[nodiscard]] LPFX_HOST_DEVICE constexpr auto
project_view(const FieldPackView3D<IndexPolicy, Space, Const, Available...> source,
             meta::type_list<Selected...>) noexcept {
  return lpfx::detail::PrivateConstructionAccess::create<
      FieldPackView3D<IndexPolicy, Space, Const, Selected...>>(
      source.template component<Selected>()...);
}

template <class Owner, class List> struct Selection;

} // namespace detail

// Zero-copy view of the requested components, in the source's order.
template <detail::Component... Requested,
          lpfx::detail::CartesianIndexPolicy IndexPolicy, class Space,
          bool Const, lpfx::detail::FieldComponentType... Available>
  requires(sizeof...(Requested) > 0) && meta::all_distinct<Requested...> &&
          (meta::contains<Requested, Available...> && ...)
[[nodiscard]] LPFX_HOST_DEVICE constexpr auto
project(const FieldPackView3D<IndexPolicy, Space, Const, Available...>
            source) noexcept {
  return detail::project_view(
      source, decltype(detail::canonical_selection<Requested...>(
                  meta::type_list<Available...>{})){});
}

// Borrowed subset of an owner's components. The owner must outlive it.
template <class Owner, detail::Component... Components>
  requires(sizeof...(Components) > 0) && meta::all_distinct<Components...>
class basic_selection final {
public:
  using owner_type = std::remove_const_t<Owner>;
  using schema_type = schema<Components...>;
  using owner_schema_type = typename owner_type::schema_type;
  using index_policy = typename owner_type::index_policy;
  using space_type = typename owner_type::space_type;
  using executor_type = typename owner_type::executor_type;
  static constexpr std::size_t component_count = sizeof...(Components);

  constexpr explicit basic_selection(Owner &owner) noexcept : owner_(&owner) {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    return owner_->valid();
  }

  [[nodiscard]] constexpr bool initialized() const noexcept {
    return owner_->initialized();
  }

  [[nodiscard]] constexpr auto view() const noexcept {
    return project<Components...>(owner_->view());
  }

  [[nodiscard]] constexpr auto cview() const noexcept {
    return project<Components...>(owner_->cview());
  }

  [[nodiscard]] constexpr auto cells() const noexcept {
    return owner_->cells();
  }

  template <detail::Component Component>
    requires schema_type::template contains<Component>
  [[nodiscard]] constexpr auto extents() const noexcept {
    return owner_->template extents<Component>();
  }

  template <std::size_t I>
    requires(I < component_count)
  [[nodiscard]] constexpr auto component_halo() const noexcept {
    return owner_->template component_halo<owner_schema_type::template index<
        typename schema_type::template component<I>>>();
  }

  template <detail::Component Component>
    requires schema_type::template contains<Component>
  [[nodiscard]] constexpr auto halo() const noexcept {
    return owner_->template halo<Component>();
  }

  [[nodiscard]] constexpr decltype(auto) executor() const noexcept {
    return owner_->executor();
  }

  template <detail::Component Component>
    requires schema_type::template contains<Component>
  [[nodiscard]] constexpr auto get() const noexcept {
    return owner_->template get<Component>();
  }

  template <detail::Component... Requested>
    requires(sizeof...(Requested) > 0) && meta::all_distinct<Requested...> &&
            (schema_type::template contains<Requested> && ...)
  [[nodiscard]] constexpr auto select() const noexcept {
    using selected = decltype(detail::canonical_selection<Requested...>(
        typename schema_type::component_list{}));
    return typename detail::Selection<Owner, selected>::type{*owner_};
  }

  [[nodiscard]] constexpr Owner &owner() const noexcept { return *owner_; }

private:
  Owner *owner_;
};

namespace detail {

template <class Owner, class... Components>
struct Selection<Owner, meta::type_list<Components...>> final {
  using type = basic_selection<Owner, Components...>;
};

template <class Schema, class IndexPolicy, class Storage> struct FieldTypes;

template <class... Components, class IndexPolicy, class Storage>
  requires lpfx::detail::CartesianIndexPolicy<IndexPolicy> &&
           lpfx::detail::FieldStorage<Storage> &&
           lpfx::detail::FieldPack<
               lpfx::detail::field_storage_space_t<Storage>, Components...>
struct FieldTypes<schema<Components...>, IndexPolicy, Storage> final {
  using owner_type = FieldOwner3D<IndexPolicy, Storage, Components...>;
  using executor_type = SerialExecutor;
};

template <class... Components, class IndexPolicy, class Backend>
  requires lpfx::detail::CartesianIndexPolicy<IndexPolicy> &&
           lpfx::detail::FieldStorage<storage::device<Backend>> &&
           lpfx::detail::FieldPack<DeviceSpace<Backend>, Components...>
struct FieldTypes<schema<Components...>, IndexPolicy, storage::device<Backend>>
    final {
  using owner_type =
      FieldOwner3D<IndexPolicy, storage::device<Backend>, Components...>;
  using executor_type = lpfx::device::Executor<Backend>;
};

template <class Schema, class IndexPolicy, class Storage>
concept Storable =
    requires { typename FieldTypes<Schema, IndexPolicy, Storage>::owner_type; };

} // namespace detail

// Movable owner of one schema's components in pageable host, pinned host or
// device storage. Device fields carry the executor they were created with.
template <detail::Schema Schema,
          lpfx::detail::CartesianIndexPolicy IndexPolicy = LogicalIndexPolicy32,
          class Storage = storage::pageable>
  requires detail::Storable<Schema, IndexPolicy, Storage>
class basic_field final {
  using types = detail::FieldTypes<Schema, IndexPolicy, Storage>;

public:
  using schema_type = Schema;
  using index_policy = IndexPolicy;
  using storage_type = Storage;
  using storage_owner_type = typename types::owner_type;
  using storage_plan_type = typename storage_owner_type::plan_type;
  using space_type = typename storage_owner_type::space_type;
  using executor_type = typename types::executor_type;
  using count_type = typename IndexPolicy::count_type;
  static constexpr std::size_t component_count = Schema::component_count;

  constexpr basic_field() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return storage_.valid();
  }

  [[nodiscard]] constexpr bool initialized() const noexcept {
    return storage_.initialized();
  }

  [[nodiscard]] constexpr auto view() noexcept { return storage_.view(); }
  [[nodiscard]] constexpr auto view() const noexcept { return storage_.view(); }
  [[nodiscard]] constexpr auto cview() const noexcept { return view(); }

  template <detail::Component Component>
    requires Schema::template contains<Component>
  [[nodiscard]] constexpr auto get() noexcept {
    return view().template component<Component>();
  }

  template <detail::Component Component>
    requires Schema::template contains<Component>
  [[nodiscard]] constexpr auto get() const noexcept {
    return view().template component<Component>();
  }

  template <detail::Component... Components>
    requires(sizeof...(Components) > 0) && meta::all_distinct<Components...> &&
            (Schema::template contains<Components> && ...)
  [[nodiscard]] constexpr auto select() & noexcept {
    return typename detail::Selection<
        basic_field, decltype(detail::canonical_selection<Components...>(
                         typename Schema::component_list{}))>::type{*this};
  }

  template <detail::Component... Components>
    requires(sizeof...(Components) > 0) && meta::all_distinct<Components...> &&
            (Schema::template contains<Components> && ...)
  [[nodiscard]] constexpr auto select() const & noexcept {
    return typename detail::Selection<
        const basic_field, decltype(detail::canonical_selection<Components...>(
                               typename Schema::component_list{}))>::type{
        *this};
  }

  template <detail::Component... Components> auto select() && = delete;

  [[nodiscard]] constexpr CartesianExtent3<count_type> cells() const noexcept {
    return storage_.cells();
  }

  template <detail::Component Component>
    requires Schema::template contains<Component>
  [[nodiscard]] constexpr CartesianExtent3<count_type> extents() const noexcept {
    return storage_.plan()
        .template component<Schema::template index<Component>>()
        .layout()
        .interior_extents();
  }

  template <std::size_t I>
    requires(I < component_count)
  [[nodiscard]] constexpr HaloDepth3<count_type>
  component_halo() const noexcept {
    return storage_.template component_halo<I>();
  }

  template <detail::Component Component>
    requires Schema::template contains<Component>
  [[nodiscard]] constexpr HaloDepth3<count_type> halo() const noexcept {
    return component_halo<Schema::template index<Component>>();
  }

  [[nodiscard]] constexpr const executor_type &executor() const noexcept {
    return executor_;
  }

  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    return storage_.capacity_bytes();
  }

  [[nodiscard]] constexpr storage_owner_type &storage() noexcept {
    return storage_;
  }
  [[nodiscard]] constexpr const storage_owner_type &storage() const noexcept {
    return storage_;
  }

private:
  constexpr basic_field(storage_owner_type &&storage,
                        const executor_type executor) noexcept
      : storage_(std::move(storage)), executor_(executor) {}

  storage_owner_type storage_{};
  [[no_unique_address]] executor_type executor_;

  friend struct lpfx::detail::PrivateConstructionAccess;
};

template <detail::Schema Schema,
          lpfx::detail::CartesianIndexPolicy IndexPolicy = LogicalIndexPolicy32>
using host = basic_field<Schema, IndexPolicy>;

template <detail::Schema Schema, class Backend,
          lpfx::detail::CartesianIndexPolicy IndexPolicy = LogicalIndexPolicy32>
using pinned_host = basic_field<Schema, IndexPolicy, storage::pinned<Backend>>;

template <detail::Schema Schema, class Backend,
          lpfx::detail::CartesianIndexPolicy IndexPolicy = LogicalIndexPolicy32>
using device = basic_field<Schema, IndexPolicy, storage::device<Backend>>;

namespace detail {

// Where make places a field: a host storage tag, or a device executor.
template <class Target> struct TargetTraits;

template <> struct TargetTraits<storage::pageable> final {
  using storage_type = storage::pageable;
};

template <class Backend> struct TargetTraits<storage::pinned<Backend>> final {
  using storage_type = storage::pinned<Backend>;
};

template <lpfx::detail::DeviceFieldTransferBackend Backend>
struct TargetTraits<lpfx::device::Executor<Backend>> final {
  using storage_type = storage::device<Backend>;
};

template <class Target>
concept FieldTarget = requires { typename TargetTraits<Target>::storage_type; };

template <class Schema, class IndexPolicy, class Target>
using field_for_target_t =
    basic_field<Schema, IndexPolicy,
                typename TargetTraits<Target>::storage_type>;

template <class Fill, class Schema> inline constexpr bool fill_constructible = false;

template <class Fill, class... Components>
inline constexpr bool fill_constructible<Fill, schema<Components...>> =
    (std::is_nothrow_constructible_v<
         typename Components::value_type,
         decltype(get<Components>(std::declval<const Fill &>()))> &&
     ...);

template <class Schema, class Fill> struct InitialValues;

template <class... Components, class Fill>
struct InitialValues<schema<Components...>, Fill> final {
  [[nodiscard]] static constexpr FieldInitialValues<Components...>
  make(const Fill &initial) noexcept {
    return FieldInitialValues<Components...>{
        typename Components::value_type(get<Components>(initial))...};
  }
};

template <class Schema, class IndexPolicy, class HaloSpec, class Fill,
          class Target>
  requires FieldTarget<Target> &&
           Storable<Schema, IndexPolicy,
                    typename TargetTraits<Target>::storage_type> &&
           ExactFor<HaloSpec, Schema> && ExactFor<Fill, Schema> &&
           fill_constructible<Fill, Schema>
[[nodiscard]] Result<field_for_target_t<Schema, IndexPolicy, Target>>
make_field(const shape<HaloSpec> &descriptor, const Fill &initial,
           [[maybe_unused]] const Target target) noexcept {
  using Field = field_for_target_t<Schema, IndexPolicy, Target>;
  using Output = Result<Field>;
  using Owner = typename Field::storage_owner_type;
  using Plan = typename Owner::plan_type;
  constexpr bool on_device =
      std::same_as<typename Field::executor_type, Target>;
  if constexpr (on_device) {
    if (!target.bound()) {
      return Output::from_error(ErrorCode::invalid_argument);
    }
  }
  if (!descriptor.cells().valid()) {
    return Output::from_error(descriptor.cells().error());
  }
  using Count = typename IndexPolicy::count_type;
  CartesianExtent3<Count> cells{};
  LPFX_TRY_RESULT(try_make_cartesian_extent3(descriptor.cells().x(),
                                             descriptor.cells().y(),
                                             descriptor.cells().z(), cells),
                  Output);
  std::array<HaloDepth3<Count>, Schema::component_count> halos{};
  const auto make_halo = [&](const auto index) -> ErrorCode {
    constexpr std::size_t I = decltype(index)::value;
    using Component = typename Schema::template component<I>;
    return get<Component>(descriptor.halos()).try_make(halos[I]);
  };
  LPFX_TRY_RESULT(meta::for_each_until_error<Schema::component_count>(make_halo),
                  Output);
  Plan plan{};
  LPFX_TRY_RESULT(try_prepare_field_storage(cells, halos, plan), Output);
  const auto values = InitialValues<Schema, Fill>::make(initial);
  if constexpr (on_device) {
    auto made = make_device_field(plan);
    LPFX_TRY_RESULT(made, Output);
    Owner owner = std::move(made).value();
    LPFX_TRY_RESULT(lpfx::detail::try_upload_field_values(owner, values, target),
                    Output);
    return Output::from_value(
        lpfx::detail::PrivateConstructionAccess::create<Field>(
            std::move(owner), target));
  } else {
    auto made = make_host_field<typename Field::storage_type>(plan, values);
    LPFX_TRY_RESULT(made, Output);
    return Output::from_value(
        lpfx::detail::PrivateConstructionAccess::create<Field>(
            std::move(made).value(), SerialExecutor{}));
  }
}

} // namespace detail

// make<Schema>(options or shape, [initial], [target]) where initial is a fill,
// a scalar for every component, or omitted (value-initialized), and target is
// storage::pageable (default), pinned<Backend> or a bound device executor.
// A braced descriptor ({.cells = ..., .halo = ...}) is an options.
template <detail::Schema Schema,
          lpfx::detail::CartesianIndexPolicy IndexPolicy = LogicalIndexPolicy32,
          class Descriptor = options, class Initial,
          class Target = storage::pageable>
  requires(!detail::FieldTarget<Initial>) &&
          requires(const Descriptor &descriptor, const Initial &initial,
                   const Target target) {
            detail::make_field<Schema, IndexPolicy>(
                detail::as_shape(descriptor), detail::as_fill(initial), target);
          }
[[nodiscard]] auto make(const Descriptor &descriptor, const Initial &initial,
                        const Target target = {}) noexcept {
  return detail::make_field<Schema, IndexPolicy>(
      detail::as_shape(descriptor), detail::as_fill(initial), target);
}

template <detail::Schema Schema,
          lpfx::detail::CartesianIndexPolicy IndexPolicy = LogicalIndexPolicy32,
          class Descriptor = options, class Target = storage::pageable>
  requires detail::FieldTarget<Target> &&
           requires(const Descriptor &descriptor, const Target target) {
             detail::make_field<Schema, IndexPolicy>(
                 detail::as_shape(descriptor), values<Schema>(), target);
           }
[[nodiscard]] auto make(const Descriptor &descriptor,
                        const Target target = {}) noexcept {
  return detail::make_field<Schema, IndexPolicy>(detail::as_shape(descriptor),
                                                 values<Schema>(), target);
}

// Binds a host field to a device field of the same schema: staged for
// pageable storage, direct for pinned storage. Both owners must stay at the
// same addresses and outlive the returned transfer.
template <detail::Schema Schema, lpfx::detail::CartesianIndexPolicy IndexPolicy,
          class HostStorage, class Backend>
  requires std::same_as<HostStorage, storage::pageable> ||
           std::same_as<HostStorage, storage::pinned<Backend>>
[[nodiscard]] auto
make_transfer(basic_field<Schema, IndexPolicy, HostStorage> &host,
              basic_field<Schema, IndexPolicy, storage::device<Backend>>
                  &device) noexcept {
  if constexpr (std::same_as<HostStorage, storage::pageable>) {
    return make_device_field_transfer(host.view(), device.storage());
  } else {
    return make_device_field_transfer(host.storage(), device.storage());
  }
}

} // namespace lpfx::field

#endif
