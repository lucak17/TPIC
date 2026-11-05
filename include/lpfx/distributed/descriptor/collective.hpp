#ifndef LPFX_DISTRIBUTED_DESCRIPTOR_COLLECTIVE_HPP
#define LPFX_DISTRIBUTED_DESCRIPTOR_COLLECTIVE_HPP

#include <cstdint>
#include <limits>
#include <type_traits>

#include <lpfx/distributed/descriptor/family.hpp>

namespace lpfx::distributed {

enum class collective_kind : std::uint8_t {
  all_reduce = 1,
  reduce = 2,
  broadcast = 3,
  all_gather = 4,
  reduce_scatter = 5,
};

enum class reduction_operation : std::uint8_t {
  sum = 1,
  product = 2,
  minimum = 3,
  maximum = 4,
};

inline constexpr std::uint32_t no_collective_root =
    std::numeric_limits<std::uint32_t>::max();

namespace detail {

template <reduction_operation Reduction>
inline constexpr bool valid_reduction_operation =
    Reduction >= reduction_operation::sum &&
    Reduction <= reduction_operation::maximum;

} // namespace detail

// Rooted kinds name a root; broadcast and all-gather carry no reduction.
template <collective_kind Kind,
          reduction_operation Reduction = reduction_operation::sum,
          std::uint32_t Root = no_collective_root>
  requires(Kind >= collective_kind::all_reduce &&
           Kind <= collective_kind::reduce_scatter &&
           detail::valid_reduction_operation<Reduction> &&
           ((Kind == collective_kind::reduce ||
             Kind == collective_kind::broadcast) ==
            (Root != no_collective_root)) &&
           (Reduction == reduction_operation::sum ||
            (Kind != collective_kind::broadcast &&
             Kind != collective_kind::all_gather)))
struct collective_descriptor final {
  static constexpr operation_family family = operation_family::collective;
  static constexpr collective_kind kind = Kind;
  static constexpr reduction_operation reduction = Reduction;
  static constexpr std::uint32_t root = Root;
};

namespace detail {

template <class T> inline constexpr bool is_collective_descriptor = false;
template <collective_kind Kind, reduction_operation Reduction,
          std::uint32_t Root>
inline constexpr bool
    is_collective_descriptor<collective_descriptor<Kind, Reduction, Root>> =
        true;

template <class Descriptor>
concept CollectiveDescriptor =
    is_collective_descriptor<std::remove_cvref_t<Descriptor>>;

} // namespace detail

template <reduction_operation Reduction>
using all_reduce_descriptor =
    collective_descriptor<collective_kind::all_reduce, Reduction>;

template <reduction_operation Reduction, std::uint32_t Root>
using reduce_descriptor =
    collective_descriptor<collective_kind::reduce, Reduction, Root>;

template <std::uint32_t Root>
using broadcast_descriptor =
    collective_descriptor<collective_kind::broadcast, reduction_operation::sum,
                          Root>;

using all_gather_descriptor =
    collective_descriptor<collective_kind::all_gather>;

template <reduction_operation Reduction>
using reduce_scatter_descriptor =
    collective_descriptor<collective_kind::reduce_scatter, Reduction>;

} // namespace lpfx::distributed

#endif
