#ifndef LPFX_DISTRIBUTED_TRANSPORT_WORKSPACE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_WORKSPACE_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/memory/byte_layout.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

template <class T>
concept HostWorkspaceElement =
    std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

// One budget-admitted host allocation holding one array per element type in
// order, each aligned for its type. Array lifetimes start at preparation and
// an empty array has a null base.
template <HostWorkspaceElement... T>
  requires(sizeof...(T) != 0)
class HostWorkspace final {
  static constexpr std::size_t alignment =
      std::max({alignof(std::max_align_t), alignof(T)...});
  using allocator_type = HostAlignedAllocator<alignment>;

public:
  static constexpr std::size_t array_count = sizeof...(T);
  using counts_type = std::array<std::size_t, array_count>;
  template <std::size_t I>
  using element_type = std::tuple_element_t<I, std::tuple<T...>>;

  using allocation_type = lpfx::detail::RawHostAllocation<allocator_type>;
  using pointers_type = std::tuple<T *...>;

  // Admits and allocates the arrays and starts their lifetimes. For owners
  // that keep their own typed views next to the bare allocation. Failure
  // leaves both outputs unchanged.
  [[nodiscard]] static ErrorCode try_allocate(HostMemoryBudget &budget,
                                              const counts_type &counts,
                                              allocation_type &allocation,
                                              pointers_type &data) noexcept {
    lpfx::detail::ByteLayoutBuilder layout{};
    counts_type offsets{};
    LPFX_TRY(meta::for_each_until_error<array_count>(
        [&](const auto index) noexcept {
          constexpr std::size_t I = decltype(index)::value;
          return layout.append<element_type<I>>(counts[I], offsets[I]);
        }));
    HostMemoryRequirement requirement{};
    if (!requirement.try_add(layout.total_bytes(), alignment)) {
      return ErrorCode::overflow;
    }
    auto admission = budget.try_admit(requirement);
    LPFX_TRY(admission);
    auto allocated = lpfx::detail::allocate_host_raw<allocator_type>(
        std::move(admission).value(), layout.total_bytes());
    LPFX_TRY(allocated);
    allocation = std::move(allocated).value();
    std::byte *const base = allocation.raw_base();
    meta::for_each_index<array_count>([&](const auto index) noexcept {
      constexpr std::size_t I = decltype(index)::value;
      using U = element_type<I>;
      std::get<I>(data) =
          counts[I] == 0
              ? nullptr
              : std::launder(static_cast<U *>(std::memmove(
                    base + offsets[I], base + offsets[I], counts[I] * sizeof(U))));
    });
    return ErrorCode::success;
  }

  // Requires an empty workspace; failure leaves it empty.
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &budget,
                                      const counts_type &counts) noexcept {
    if (!empty()) {
      return ErrorCode::invalid_state;
    }
    LPFX_TRY(try_allocate(budget, counts, allocation_, data_));
    counts_ = counts;
    return ErrorCode::success;
  }

  template <std::size_t I>
  [[nodiscard]] element_type<I> *data() const noexcept {
    return std::get<I>(data_);
  }
  template <std::size_t I>
  [[nodiscard]] std::size_t size() const noexcept {
    return counts_[I];
  }
  template <std::size_t I>
  [[nodiscard]] std::span<element_type<I>> span() const noexcept {
    return {std::get<I>(data_), counts_[I]};
  }
  [[nodiscard]] std::byte *raw_base() const noexcept {
    return const_cast<std::byte *>(allocation_.raw_base());
  }
  [[nodiscard]] std::size_t capacity_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }
  [[nodiscard]] bool empty() const noexcept { return allocation_.empty(); }
  void reset() noexcept {
    allocation_.reset_capacity();
    data_ = {};
    counts_ = {};
  }

private:
  allocation_type allocation_{};
  pointers_type data_{};
  counts_type counts_{};
};

} // namespace lpfx::distributed::detail

#endif
