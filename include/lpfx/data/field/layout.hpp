#ifndef LPFX_DATA_FIELD_LAYOUT_HPP
#define LPFX_DATA_FIELD_LAYOUT_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/field/component.hpp>
#include <lpfx/domain/cartesian/halo.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/memory/byte_layout.hpp>

namespace lpfx {

template <detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy>
class ComponentLayout3D final {
public:
  using location_type = Location;
  using index_policy = IndexPolicy;
  using coordinate_type = typename IndexPolicy::coordinate_type;
  using count_type = typename IndexPolicy::count_type;
  using offset_type = typename IndexPolicy::offset_type;
  using stride_type = typename IndexPolicy::stride_type;

  static constexpr stride_type stride_x = 1;

  LPFX_HOST_DEVICE constexpr ComponentLayout3D() noexcept {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianExtent3<count_type>
  interior_extents() const noexcept {
    return interior_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr HaloDepth3<count_type>
  halo() const noexcept {
    return halo_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianExtent3<count_type>
  allocation_extents() const noexcept {
    return allocation_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr offset_type
  leading_prefix() const noexcept {
    return leading_prefix_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr stride_type
  stride_y() const noexcept {
    return stride_y_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr stride_type
  stride_z() const noexcept {
    return stride_z_;
  }

  // Elements from the first allocated sample (after the leading prefix) to
  // logical (0,0,0). Preparation proves it smaller than the block.
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::size_t
  core_offset() const noexcept {
    return static_cast<std::size_t>(halo_.x_minus) +
           static_cast<std::size_t>(halo_.y_minus) *
               static_cast<std::size_t>(stride_y_) +
           static_cast<std::size_t>(halo_.z_minus) *
               static_cast<std::size_t>(stride_z_);
  }

  [[nodiscard]] friend constexpr bool
  operator==(const ComponentLayout3D &,
             const ComponentLayout3D &) noexcept = default;

private:
  constexpr ComponentLayout3D(const CartesianExtent3<count_type> interior,
                              const HaloDepth3<count_type> halo,
                              const CartesianExtent3<count_type> allocation,
                              const offset_type leading_prefix,
                              const stride_type stride_y,
                              const stride_type stride_z) noexcept
      : interior_(interior), halo_(halo), allocation_(allocation),
        leading_prefix_(leading_prefix), stride_y_(stride_y),
        stride_z_(stride_z) {}

  CartesianExtent3<count_type> interior_{};
  HaloDepth3<count_type> halo_{};
  CartesianExtent3<count_type> allocation_{};
  offset_type leading_prefix_{};
  stride_type stride_y_{};
  stride_type stride_z_{};

  friend struct detail::PrivateConstructionAccess;
};

// One halo per component, in pack order.
template <detail::CartesianIndexPolicy IndexPolicy, class... Components>
using FieldHaloPack = std::array<HaloDepth3<typename IndexPolicy::count_type>,
                                 sizeof...(Components)>;

template <detail::CartesianIndexPolicy IndexPolicy,
          detail::FieldComponentType... Components, class... Halos>
  requires detail::FieldTags<Components...> &&
           (sizeof...(Components) == sizeof...(Halos)) &&
           (std::same_as<std::remove_cvref_t<Halos>,
                         HaloDepth3<typename IndexPolicy::count_type>> &&
            ...)
[[nodiscard]] constexpr FieldHaloPack<IndexPolicy, Components...>
make_field_halo_pack(Halos &&...values) noexcept {
  return {std::forward<Halos>(values)...};
}

// Placement of one component block: rows of stride_y elements, planes of
// stride_z elements, and the leading prefix before the first sample.
template <detail::CartesianLocation Location,
          detail::CartesianIndexPolicy IndexPolicy>
class ComponentStorageReport final {
public:
  using location_type = Location;
  using layout_type = ComponentLayout3D<Location, IndexPolicy>;

  constexpr ComponentStorageReport() noexcept = default;

  [[nodiscard]] constexpr layout_type layout() const noexcept {
    return layout_;
  }
  [[nodiscard]] constexpr std::size_t byte_offset() const noexcept {
    return byte_offset_;
  }
  [[nodiscard]] constexpr std::size_t block_bytes() const noexcept {
    return block_bytes_;
  }
  [[nodiscard]] constexpr std::size_t element_bytes() const noexcept {
    return element_bytes_;
  }
  [[nodiscard]] constexpr std::size_t end_byte_offset() const noexcept {
    return byte_offset_ + block_bytes_;
  }
  [[nodiscard]] constexpr std::size_t row_pitch_bytes() const noexcept {
    return static_cast<std::size_t>(layout_.stride_y()) * element_bytes_;
  }
  [[nodiscard]] constexpr std::size_t plane_pitch_bytes() const noexcept {
    return static_cast<std::size_t>(layout_.stride_z()) * element_bytes_;
  }
  [[nodiscard]] constexpr std::size_t leading_prefix_bytes() const noexcept {
    return static_cast<std::size_t>(layout_.leading_prefix()) * element_bytes_;
  }

private:
  constexpr ComponentStorageReport(const layout_type layout,
                                   const std::size_t byte_offset,
                                   const std::size_t block_bytes,
                                   const std::size_t element_bytes) noexcept
      : layout_(layout), byte_offset_(byte_offset), block_bytes_(block_bytes),
        element_bytes_(element_bytes) {}

  layout_type layout_{};
  std::size_t byte_offset_{};
  std::size_t block_bytes_{};
  std::size_t element_bytes_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::CartesianIndexPolicy IndexPolicy, class Space,
          detail::FieldComponentType... Components>
  requires detail::FieldPack<Space, Components...>
class PreparedFieldStorage final {
public:
  using index_policy = IndexPolicy;
  using space_type = Space;
  using reports_type =
      std::tuple<ComponentStorageReport<typename Components::location_type,
                                        IndexPolicy>...>;
  static constexpr std::size_t component_count = sizeof...(Components);
  static constexpr std::size_t allocation_alignment =
      std::max({detail::component_start_alignment<Space, Components>...});

  constexpr PreparedFieldStorage() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }

  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return total_bytes_;
  }

  template <std::size_t I>
    requires(I < component_count)
  [[nodiscard]] constexpr const auto &component() const noexcept {
    return std::get<I>(components_);
  }

private:
  constexpr PreparedFieldStorage(const reports_type &reports,
                                 const std::size_t total_bytes) noexcept
      : components_(reports), total_bytes_(total_bytes), valid_(true) {}

  reports_type components_{};
  std::size_t total_bytes_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

// Only physical properties participate in arithmetic specialization. Keep
// this boundary outlined at runtime: otherwise GCC clones it into
// tag-dependent callers. Constant evaluation still uses the same body.
template <CartesianLocation Location, CartesianIndexPolicy IndexPolicy,
          std::size_t ElementBytes, FieldPitchKind PitchKind,
          std::size_t PitchBytes>
[[nodiscard]] LPFX_NOINLINE constexpr ErrorCode try_prepare_component_layout(
    const CartesianExtent3<typename IndexPolicy::count_type> &cells,
    const HaloDepth3<typename IndexPolicy::count_type> &halo,
    ComponentLayout3D<Location, IndexPolicy> &output,
    std::size_t &block_elements) noexcept {
  using Count = typename IndexPolicy::count_type;
  using Offset = typename IndexPolicy::offset_type;
  using Stride = typename IndexPolicy::stride_type;
  using Layout = ComponentLayout3D<Location, IndexPolicy>;

  CartesianExtent3<Count> interior{};
  LPFX_TRY((try_make_location_sample_extents<Location, IndexPolicy>(
      cells, interior)));

  std::array<Count, 3> allocated{};
  const auto allocate_axis = [&](const auto axis) -> ErrorCode {
    constexpr std::size_t Axis = decltype(axis)::value;
    AllocatedAxisInterval<IndexPolicy> interval{};
    LPFX_TRY(try_make_allocated_axis_interval<IndexPolicy>(
        interior.template get<Axis>(), halo.template get<Axis>(-1),
        halo.template get<Axis>(1), interval));
    allocated[Axis] = interval.extent();
    return ErrorCode::success;
  };
  LPFX_TRY(for_each_axis(allocate_axis));
  CartesianExtent3<Count> allocation{};
  LPFX_TRY(try_make_cartesian_extent3(allocated[0], allocated[1], allocated[2],
                                      allocation));

  std::size_t allocation_x{};
  std::size_t row_bytes{};
  if (!checked_integral_cast(allocation.x(), allocation_x) ||
      !checked_multiply(allocation_x, ElementBytes, row_bytes)) {
    return ErrorCode::overflow;
  }
  if constexpr (PitchKind != FieldPitchKind::packed) {
    if (!checked_align_up(row_bytes, PitchBytes, row_bytes)) {
      return ErrorCode::overflow;
    }
  }
  Stride stride_y{};
  Stride extent_y{};
  Stride stride_z{};
  Offset plane_elements{};
  Offset extent_z{};
  Offset planes{};
  if (!checked_integral_cast(row_bytes / ElementBytes, stride_y) ||
      !checked_integral_cast(allocation.y(), extent_y) ||
      !checked_multiply(stride_y, extent_y, stride_z) ||
      !checked_integral_cast(stride_z, plane_elements) ||
      !checked_integral_cast(allocation.z(), extent_z) ||
      !checked_multiply(plane_elements, extent_z, planes)) {
    return ErrorCode::overflow;
  }

  // The core offset is below planes, so it is representable from here on.
  Offset leading_prefix{};
  if constexpr (PitchKind == FieldPitchKind::core_aligned) {
    constexpr std::size_t alignment_elements = PitchBytes / ElementBytes;
    const std::size_t misalignment =
        PrivateConstructionAccess::create<Layout>(interior, halo, allocation,
                                                  Offset{}, stride_y, stride_z)
            .core_offset() %
        alignment_elements;
    if (!checked_integral_cast(
            misalignment == 0 ? 0 : alignment_elements - misalignment,
            leading_prefix)) {
      return ErrorCode::overflow;
    }
  }

  Offset block{};
  std::ptrdiff_t addressable{};
  std::size_t elements{};
  if (!checked_add(leading_prefix, planes, block) ||
      !checked_integral_cast(block, addressable) ||
      !checked_integral_cast(block, elements)) {
    return ErrorCode::overflow;
  }
  output = PrivateConstructionAccess::create<Layout>(
      interior, halo, allocation, leading_prefix, stride_y, stride_z);
  block_elements = elements;
  return ErrorCode::success;
}

} // namespace detail

// Components are placed in pack order, each at the next multiple of its start
// alignment. output is written only on success.
template <detail::CartesianIndexPolicy IndexPolicy, class Space,
          detail::FieldComponentType... Components>
  requires detail::FieldPack<Space, Components...>
[[nodiscard]] constexpr ErrorCode try_prepare_field_storage(
    const CartesianExtent3<typename IndexPolicy::count_type> &cells,
    const FieldHaloPack<IndexPolicy, Components...> &halos,
    PreparedFieldStorage<IndexPolicy, Space, Components...> &output) noexcept {
  using Plan = PreparedFieldStorage<IndexPolicy, Space, Components...>;
  typename Plan::reports_type reports{};
  detail::ByteLayoutBuilder builder{};
  const auto place = [&](const auto index) -> ErrorCode {
    constexpr std::size_t I = decltype(index)::value;
    using Component = std::tuple_element_t<I, std::tuple<Components...>>;
    using Location = typename Component::location_type;
    using T = typename Component::value_type;
    ComponentLayout3D<Location, IndexPolicy> layout{};
    std::size_t elements{};
    LPFX_TRY((detail::try_prepare_component_layout<
              Location, IndexPolicy, sizeof(T), Component::pitch_policy::kind,
              Component::pitch_policy::alignment_bytes>(cells, halos[I], layout,
                                                        elements)));
    std::size_t offset{};
    LPFX_TRY(builder.append<T>(
        elements, detail::component_start_alignment<Space, Component>,
        offset));
    std::get<I>(reports) = detail::PrivateConstructionAccess::create<
        ComponentStorageReport<Location, IndexPolicy>>(
        layout, offset, elements * sizeof(T), sizeof(T));
    return ErrorCode::success;
  };
  LPFX_TRY(meta::for_each_until_error<sizeof...(Components)>(place));
  output = detail::PrivateConstructionAccess::create<Plan>(
      reports, builder.total_bytes());
  return ErrorCode::success;
}

} // namespace lpfx

#endif
