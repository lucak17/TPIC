#ifndef LPFX_ALGORITHM_REDUCE_HOST_HPP
#define LPFX_ALGORITHM_REDUCE_HOST_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/domain/cartesian/location.hpp>
#include <lpfx/algorithm/reduce/operation.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx {

namespace detail {

template <class Component, class Accumulator>
concept HostFieldReductionComponent =
    std::floating_point<Accumulator> &&
    requires(
        const Component component,
        const typename Component::index_policy::coordinate_type coordinate) {
      typename Component::element_type;
      typename Component::value_type;
      typename Component::space_type;
      typename Component::location_type;
      typename Component::index_policy;
      requires std::floating_point<typename Component::value_type>;
      requires std::same_as<
          std::remove_const_t<typename Component::element_type>,
          typename Component::value_type>;
      requires std::same_as<typename Component::space_type, HostSpace>;
      requires CartesianLocation<typename Component::location_type>;
      requires CartesianIndexPolicy<typename Component::index_policy>;
      requires Component::stride_x == 1;
      {
        component.layout().interior_extents()
      } -> std::same_as<
          CartesianExtent3<typename Component::index_policy::count_type>>;
      {
        component.row_data(coordinate, coordinate)
      } -> std::same_as<typename Component::element_type *>;
    };

template <class Accumulator, class First, class... Rest>
concept HostFieldReductionComponents =
    HostFieldReductionComponent<First, Accumulator> &&
    (HostFieldReductionComponent<Rest, Accumulator> && ...) &&
    (std::same_as<typename First::location_type,
                  typename Rest::location_type> &&
     ...) &&
    (std::same_as<typename First::index_policy, typename Rest::index_policy> &&
     ...);

// Order of the final combine of the four lanes; each caller's order is part of
// its bitwise result.
enum class LaneCombine : std::uint8_t {
  linear, // ((0 op 1) op 2) op 3
  tree,   // (0 op 1) op (2 op 3)
};

template <ReductionAccumulator Accumulator> struct HostReductionLanes final {
  Accumulator lane_0;
  Accumulator lane_1;
  Accumulator lane_2;
  Accumulator lane_3;
};

template <class Semantics>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr HostReductionLanes<
    typename Semantics::accumulator_type>
make_host_reduction_lanes() noexcept {
  const auto identity = Semantics::identity();
  return {identity, identity, identity, identity};
}

// Element i of [0, end) feeds lane i % 4; the tail feeds lanes 0, 1, 2.
template <class Semantics, class Count, class Elements>
LPFX_ALWAYS_INLINE constexpr void
fold_host_reduction_lanes(HostReductionLanes<typename Semantics::accumulator_type> &lanes,
                          const Count end, const Elements elements) noexcept {
  Count i{};
  for (; end - i >= Count{4}; i += Count{4}) {
    lanes.lane_0 = Semantics::combine(lanes.lane_0, elements(i));
    lanes.lane_1 = Semantics::combine(lanes.lane_1, elements(i + Count{1}));
    lanes.lane_2 = Semantics::combine(lanes.lane_2, elements(i + Count{2}));
    lanes.lane_3 = Semantics::combine(lanes.lane_3, elements(i + Count{3}));
  }
  if (i != end) {
    lanes.lane_0 = Semantics::combine(lanes.lane_0, elements(i));
    ++i;
  }
  if (i != end) {
    lanes.lane_1 = Semantics::combine(lanes.lane_1, elements(i));
    ++i;
  }
  if (i != end) {
    lanes.lane_2 = Semantics::combine(lanes.lane_2, elements(i));
  }
}

template <LaneCombine Order, class Semantics>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr typename Semantics::accumulator_type
combine_host_reduction_lanes(
    const HostReductionLanes<typename Semantics::accumulator_type>
        lanes) noexcept {
  if constexpr (Order == LaneCombine::tree) {
    return Semantics::combine(Semantics::combine(lanes.lane_0, lanes.lane_1),
                              Semantics::combine(lanes.lane_2, lanes.lane_3));
  } else {
    return Semantics::combine(
        Semantics::combine(Semantics::combine(lanes.lane_0, lanes.lane_1),
                           lanes.lane_2),
        lanes.lane_3);
  }
}

template <ReductionOperation Operation, ReductionAccumulator Accumulator,
          ReductionInput... T>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_contiguous_reduction(const std::size_t count,
                                const T *const... values) noexcept {
  using Semantics = ReductionSemantics<Operation, Accumulator>;
  if (count == 0) {
    return {};
  }
  auto lanes = make_host_reduction_lanes<Semantics>();
  fold_host_reduction_lanes<Semantics>(
      lanes, count, ReductionElements<Semantics, true, const T *...>{values...});
  return ReductionResult<Accumulator>::from_value(
      combine_host_reduction_lanes<LaneCombine::tree, Semantics>(lanes));
}

// z/y/x traversal of the component interiors; one or two (dot) components.
template <ReductionOperation Operation, ReductionAccumulator Accumulator,
          class First, class... Rest>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_host_field_reduction(const First first,
                                const Rest... rest) noexcept {
  using Semantics = ReductionSemantics<Operation, Accumulator>;
  using IndexPolicy = typename First::index_policy;
  using Coordinate = typename IndexPolicy::coordinate_type;
  using Count = typename IndexPolicy::count_type;
  using Elements =
      ReductionElements<Semantics, true, typename First::element_type *,
                        typename Rest::element_type *...>;

  const auto extents = first.layout().interior_extents();
  if (extents.x() == 0 || extents.y() == 0 || extents.z() == 0) {
    return {};
  }

  auto lanes = make_host_reduction_lanes<Semantics>();
  for (Count k{}; k != extents.z(); ++k) {
    const Coordinate coordinate_k = static_cast<Coordinate>(k);
    for (Count j{}; j != extents.y(); ++j) {
      const Coordinate coordinate_j = static_cast<Coordinate>(j);
      fold_host_reduction_lanes<Semantics>(
          lanes, extents.x(),
          Elements{first.row_data(coordinate_j, coordinate_k),
                   rest.row_data(coordinate_j, coordinate_k)...});
    }
  }
  return ReductionResult<Accumulator>::from_value(
      combine_host_reduction_lanes<LaneCombine::linear, Semantics>(lanes));
}

} // namespace detail

// Stateless host folds with four fixed accumulator lanes. Every element is
// converted to Accumulator before arithmetic. For nonempty input any converted
// NaN propagates to the result of every operation, including min, max and
// abs-max. Empty input returns an empty result. Nothing is allocated, retained
// or synchronized.
//
// Contiguous ranges: nonempty pointers must be valid for count elements; dot
// permits exact aliasing of its two inputs.
template <ReductionOperation Operation, std::floating_point Accumulator,
          detail::ReductionInput T>
  requires(Operation != ReductionOperation::dot)
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_range_reduction(const T *const values,
                           const std::size_t count) noexcept {
  return detail::run_serial_contiguous_reduction<Operation, Accumulator>(
      count, values);
}

template <ReductionOperation Operation, std::floating_point Accumulator,
          detail::ReductionInput Left, detail::ReductionInput Right>
  requires(Operation == ReductionOperation::dot)
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_range_reduction(const Left *const left, const Right *const right,
                           const std::size_t count) noexcept {
  return detail::run_serial_contiguous_reduction<Operation, Accumulator>(
      count, left, right);
}

template <ReductionOperation Operation, std::floating_point Accumulator,
          class T, std::size_t Extent>
  requires(Operation != ReductionOperation::dot) &&
          detail::ReductionInput<std::remove_const_t<T>>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_range_reduction(const std::span<T, Extent> values) noexcept {
  return detail::run_serial_contiguous_reduction<Operation, Accumulator>(
      values.size(), static_cast<const std::remove_const_t<T> *>(values.data()));
}

// Hard precondition: left.size() == right.size().
template <ReductionOperation Operation, std::floating_point Accumulator,
          class Left, std::size_t LeftExtent, class Right,
          std::size_t RightExtent>
  requires(Operation == ReductionOperation::dot) &&
          detail::ReductionInput<std::remove_const_t<Left>> &&
          detail::ReductionInput<std::remove_const_t<Right>>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_range_reduction(const std::span<Left, LeftExtent> left,
                           const std::span<Right, RightExtent> right) noexcept {
  return detail::run_serial_contiguous_reduction<Operation, Accumulator>(
      left.size(), static_cast<const std::remove_const_t<Left> *>(left.data()),
      static_cast<const std::remove_const_t<Right> *>(right.data()));
}

// Field interiors: rows are read at unit x stride and no halo sample is read.
// Dot operands share location and index policy statically; equal interior
// extents are a hard precondition. Aliasing the two operands is valid.
template <ReductionOperation Operation, std::floating_point Accumulator,
          class Component>
  requires(Operation != ReductionOperation::dot) &&
          detail::HostFieldReductionComponents<Accumulator, Component>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_field_reduction(const Component component) noexcept {
  return detail::run_serial_host_field_reduction<Operation, Accumulator>(
      component);
}

template <ReductionOperation Operation, std::floating_point Accumulator,
          class Left, class Right>
  requires(Operation == ReductionOperation::dot) &&
          detail::HostFieldReductionComponents<Accumulator, Left, Right>
[[nodiscard]] LPFX_ALWAYS_INLINE constexpr ReductionResult<Accumulator>
run_serial_field_reduction(const Left left, const Right right) noexcept {
  return detail::run_serial_host_field_reduction<Operation, Accumulator>(left,
                                                                         right);
}

} // namespace lpfx

#endif
