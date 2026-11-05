#ifndef LPFX_DATA_FIELD_VIEW_HPP
#define LPFX_DATA_FIELD_VIEW_HPP

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/memory_intrinsics.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/field/layout.hpp>
#include <lpfx/memory/lifetime.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx {

namespace detail {

// Per-space state of ComponentView3D. Host views keep the complete layout;
// kernel-facing device views keep only the two non-unit strides.
template <class T, class Space, CartesianLocation Location,
          CartesianIndexPolicy IndexPolicy>
class ComponentViewBase {
public:
  using layout_type = ComponentLayout3D<Location, IndexPolicy>;
  using stride_type = typename IndexPolicy::stride_type;

  [[nodiscard]] LPFX_HOST_DEVICE constexpr layout_type layout() const noexcept {
    return layout_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto extents() const noexcept {
    return layout_.interior_extents();
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr stride_type
  stride_y() const noexcept {
    return layout_.stride_y();
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr stride_type
  stride_z() const noexcept {
    return layout_.stride_z();
  }

protected:
  LPFX_HOST_DEVICE constexpr ComponentViewBase() noexcept {}

  LPFX_HOST_DEVICE constexpr ComponentViewBase(T *const core,
                                               const layout_type layout) noexcept
      : core_(core), layout_(layout) {}

  T *core_ = nullptr;
  layout_type layout_{};
};

template <class T, device::DeviceBackend Backend, CartesianLocation Location,
          CartesianIndexPolicy IndexPolicy>
class ComponentViewBase<T, DeviceSpace<Backend>, Location, IndexPolicy> {
public:
  using backend_type = Backend;
  using stride_type = typename IndexPolicy::stride_type;

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr stride_type
  stride_y() const noexcept {
    return stride_y_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr stride_type
  stride_z() const noexcept {
    return stride_z_;
  }

protected:
  LPFX_HOST_DEVICE constexpr ComponentViewBase() noexcept {}

  LPFX_HOST_DEVICE constexpr ComponentViewBase(
      T *const core, const ComponentLayout3D<Location, IndexPolicy> layout) noexcept
      : core_(core), stride_y_(layout.stride_y()), stride_z_(layout.stride_z()) {}

  LPFX_HOST_DEVICE constexpr ComponentViewBase(T *const core,
                                               const stride_type stride_y,
                                               const stride_type stride_z) noexcept
      : core_(core), stride_y_(stride_y), stride_z_(stride_z) {}

  T *core_ = nullptr;
  stride_type stride_y_{};
  stride_type stride_z_{};
};

template <class Space> inline constexpr bool is_device_space = false;

template <class Backend>
inline constexpr bool is_device_space<DeviceSpace<Backend>> = true;

// std::tuple access is not device-callable with every supported standard
// library; this recursive aggregate supplies the one operation views need.
template <class... Views> struct FieldViewTuple;

template <class View> struct FieldViewTuple<View> final {
  View value{};

  template <std::size_t I>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr View get() const noexcept {
    static_assert(I == 0);
    return value;
  }
};

template <class First, class Second, class... Rest>
struct FieldViewTuple<First, Second, Rest...> final {
  First value{};
  FieldViewTuple<Second, Rest...> tail{};

  template <std::size_t I>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto get() const noexcept {
    if constexpr (I == 0) {
      return value;
    } else {
      return tail.template get<I - 1>();
    }
  }
};

template <class First, class... Rest>
[[nodiscard]] LPFX_HOST_DEVICE constexpr FieldViewTuple<First, Rest...>
make_field_view_tuple(const First first, const Rest... rest) noexcept {
  if constexpr (sizeof...(Rest) == 0) {
    return {first};
  } else {
    return {first, make_field_view_tuple<Rest...>(rest...)};
  }
}

} // namespace detail

// One field component in one address space. The core pointer addresses
// logical (0,0,0); x has unit stride.
template <class T, class Space, detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy>
  requires detail::LifetimeViewElement<T>
class ComponentView3D final
    : public detail::ComponentViewBase<T, Space, Location, IndexPolicy> {
  using base_type = detail::ComponentViewBase<T, Space, Location, IndexPolicy>;

public:
  using element_type = T;
  using value_type = std::remove_const_t<T>;
  using space_type = Space;
  using location_type = Location;
  using index_policy = IndexPolicy;
  using coordinate_type = typename IndexPolicy::coordinate_type;
  using count_type = typename IndexPolicy::count_type;
  using stride_type = typename IndexPolicy::stride_type;

  static constexpr stride_type stride_x = 1;

  LPFX_HOST_DEVICE constexpr ComponentView3D() noexcept {}

  template <class U>
    requires std::same_as<T, const U> && (!detail::is_device_space<Space>)
  LPFX_HOST_DEVICE constexpr ComponentView3D(
      const ComponentView3D<U, Space, Location, IndexPolicy> other) noexcept
      : base_type(other.core_data(), other.layout()) {}

  template <class U>
    requires std::same_as<T, const U> && detail::is_device_space<Space>
  LPFX_HOST_DEVICE constexpr ComponentView3D(
      const ComponentView3D<U, Space, Location, IndexPolicy> other) noexcept
      : base_type(other.core_data(), other.stride_y(), other.stride_z()) {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr T *core_data() const noexcept {
    return this->core_;
  }

  // Hard precondition: every logical x in the row at (j,k) is in the
  // allocated interval. The returned pointer addresses logical x == 0.
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr T *
  row_data(const coordinate_type j, const coordinate_type k) const noexcept {
    const std::ptrdiff_t linear =
        static_cast<std::ptrdiff_t>(j) *
            static_cast<std::ptrdiff_t>(this->stride_y()) +
        static_cast<std::ptrdiff_t>(k) *
            static_cast<std::ptrdiff_t>(this->stride_z());
    return this->core_ + linear;
  }

  // Hard precondition: (i,j,k) is in the allocated interval.
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr T &
  operator()(const coordinate_type i, const coordinate_type j,
             const coordinate_type k) const noexcept {
    return row_data(j, k)[i];
  }

  // Non-negative x keeps its unsigned type: narrowing it prevents GCC from
  // proving monotonic unit-stride access in count-based loops.
  template <class Count>
    requires std::same_as<std::remove_cv_t<Count>, count_type>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr T &
  operator()(const Count i, const coordinate_type j,
             const coordinate_type k) const noexcept {
    return row_data(j, k)[i];
  }

private:
  // (core, layout) in every space; (core, stride_y, stride_z) on device.
  template <class... Geometry>
  LPFX_HOST_DEVICE constexpr explicit ComponentView3D(
      T *const core, const Geometry... geometry) noexcept
      : base_type(core, geometry...) {
    static_assert(std::is_standard_layout_v<ComponentView3D> &&
                  std::is_trivially_copyable_v<ComponentView3D>);
    static_assert(!detail::is_device_space<Space> ||
                  sizeof(ComponentView3D) ==
                      sizeof(T *) + 2 * sizeof(stride_type));
  }

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianIndexPolicy IndexPolicy, class Space, bool Const,
          detail::FieldComponentType... Components>
  requires detail::FieldTags<Components...>
class FieldPackView3D final {
  template <class Component>
  using component_view_t =
      ComponentView3D<std::conditional_t<Const,
                                         const typename Component::value_type,
                                         typename Component::value_type>,
                      Space, typename Component::location_type, IndexPolicy>;

public:
  using index_policy = IndexPolicy;
  using space_type = Space;
  using coordinate_type = typename IndexPolicy::coordinate_type;
  static constexpr std::size_t component_count = sizeof...(Components);

  LPFX_HOST_DEVICE constexpr FieldPackView3D() noexcept {}

  template <bool Mutable>
    requires(Const && !Mutable)
  LPFX_HOST_DEVICE constexpr FieldPackView3D(
      const FieldPackView3D<IndexPolicy, Space, Mutable, Components...>
          other) noexcept
      : FieldPackView3D(other, std::index_sequence_for<Components...>{}) {}

  template <std::size_t I>
    requires(I < component_count)
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto component() const noexcept {
    return components_.template get<I>();
  }

  // Key is a component type of the pack or its tag.
  template <class Key>
    requires detail::FieldPackKey<Key, Components...>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto component() const noexcept {
    return components_
        .template get<detail::field_pack_index<Key, Components...>>();
  }

  template <class Key>
    requires detail::FieldPackKey<Key, Components...>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto get() const noexcept {
    return component<Key>();
  }

  // Available when the CoreAlignedPitch policy guarantees alignment of
  // logical x == 0 in every allocated row. Hard precondition: (j,k) is in the
  // allocated interval.
  template <std::size_t I>
    requires(I < component_count) &&
            (std::tuple_element_t<I, std::tuple<Components...>>::pitch_policy::
                 kind == FieldPitchKind::core_aligned)
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE auto *
  aligned_row_data(const coordinate_type j,
                   const coordinate_type k) const noexcept {
    constexpr std::size_t alignment =
        std::tuple_element_t<I, std::tuple<Components...>>::pitch_policy::
            alignment_bytes;
    return detail::assume_aligned_pointer<alignment>(
        component<I>().row_data(j, k));
  }

  template <class Key>
    requires detail::FieldPackKey<Key, Components...> &&
             (std::tuple_element_t<detail::field_pack_index<Key, Components...>,
                                   std::tuple<Components...>>::pitch_policy::
                  kind == FieldPitchKind::core_aligned)
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE auto *
  aligned_row_data(const coordinate_type j,
                   const coordinate_type k) const noexcept {
    return aligned_row_data<detail::field_pack_index<Key, Components...>>(j,
                                                                            k);
  }

private:
  template <class Source, std::size_t... I>
  LPFX_HOST_DEVICE constexpr FieldPackView3D(
      const Source source, std::index_sequence<I...>) noexcept
      : FieldPackView3D(component_view_t<Components>(
            source.template component<I>())...) {}

  LPFX_HOST_DEVICE constexpr explicit FieldPackView3D(
      const component_view_t<Components>... components) noexcept
      : components_(detail::make_field_view_tuple<component_view_t<Components>...>(
            components...)) {
    static_assert(std::is_standard_layout_v<FieldPackView3D> &&
                  std::is_trivially_copyable_v<FieldPackView3D>);
  }

  detail::FieldViewTuple<component_view_t<Components>...> components_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianIndexPolicy IndexPolicy, class Backend,
          detail::FieldComponentType... Components>
using DeviceFieldPackView3D =
    FieldPackView3D<IndexPolicy, DeviceSpace<Backend>, false, Components...>;

namespace detail {

// start addresses the first element of the component block.
template <class Space, class T, CartesianLocation Location,
          CartesianIndexPolicy IndexPolicy>
[[nodiscard]] constexpr ComponentView3D<T, Space, Location, IndexPolicy>
make_component_view(T *const start,
                    const ComponentLayout3D<Location, IndexPolicy> layout) noexcept {
  return PrivateConstructionAccess::create<
      ComponentView3D<T, Space, Location, IndexPolicy>>(
      start + (static_cast<std::size_t>(layout.leading_prefix()) +
               layout.core_offset()),
      layout);
}

} // namespace detail

} // namespace lpfx

#endif
