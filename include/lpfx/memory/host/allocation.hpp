#ifndef LPFX_MEMORY_HOST_ALLOCATION_HPP
#define LPFX_MEMORY_HOST_ALLOCATION_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <tuple>
#include <utility>

#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/memory/byte_layout.hpp>
#include <lpfx/memory/host/allocator.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx::detail {

template <HostRawAllocator Allocator> class RawHostAllocation {
public:
  constexpr RawHostAllocation() noexcept = default;

  RawHostAllocation(RawHostAllocation &&other) noexcept
      : base_(std::exchange(other.base_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)),
        admission_(std::move(other.admission_)) {}

  RawHostAllocation &operator=(RawHostAllocation &&other) noexcept {
    if (this != &other) {
      reset_capacity();
      base_ = std::exchange(other.base_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
      admission_ = std::move(other.admission_);
    }
    return *this;
  }

  ~RawHostAllocation() noexcept { reset_capacity(); }

  [[nodiscard]] std::byte *raw_base() noexcept { return base_; }
  [[nodiscard]] const std::byte *raw_base() const noexcept { return base_; }
  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    return bytes_;
  }
  [[nodiscard]] static constexpr std::size_t alignment() noexcept {
    return Allocator::minimum_data_alignment;
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return base_ == nullptr;
  }
  [[nodiscard]] HostMemoryBudget *admission_budget() const noexcept {
    return admission_.budget();
  }

  void reset_capacity() noexcept {
    if (base_ != nullptr) {
      Allocator::deallocate(base_);
      base_ = nullptr;
      bytes_ = 0;
    }
    admission_.release();
  }

private:
  RawHostAllocation(std::byte *const base, const std::size_t bytes,
                    HostMemoryLease &&admission) noexcept
      : base_(base), bytes_(bytes), admission_(std::move(admission)) {}

  std::byte *base_ = nullptr;
  std::size_t bytes_ = 0;
  HostMemoryLease admission_{};

  friend struct PrivateConstructionAccess;
};

using DefaultRawHostAllocation = RawHostAllocation<DefaultAllocator<HostSpace>>;

// The caller has matched admission to bytes, or passes an inactive lease.
template <HostRawAllocator Allocator>
[[nodiscard]] Result<RawHostAllocation<Allocator>>
allocate_host_raw_unchecked(HostMemoryLease &&admission,
                            const std::size_t bytes) noexcept {
  using Allocation = RawHostAllocation<Allocator>;
  std::byte *base = nullptr;
  if (bytes != 0) {
    base = static_cast<std::byte *>(Allocator::allocate(bytes));
    if (base == nullptr) {
      return Result<Allocation>::from_error(ErrorCode::allocation_failed);
    }
  }
  return Result<Allocation>::from_value(
      PrivateConstructionAccess::create<Allocation>(base, bytes,
                                                    std::move(admission)));
}

template <HostRawAllocator Allocator>
[[nodiscard]] Result<RawHostAllocation<Allocator>>
allocate_host_raw(HostMemoryLease admission, const std::size_t bytes) noexcept {
  if (!admission.matches(bytes, Allocator::minimum_data_alignment)) {
    return Result<RawHostAllocation<Allocator>>::from_error(
        ErrorCode::invalid_argument);
  }
  return allocate_host_raw_unchecked<Allocator>(std::move(admission), bytes);
}

// Self-contained owners carry an inactive lease and therefore have no lifetime
// dependency on a caller-owned budget.
template <HostRawAllocator Allocator>
[[nodiscard]] Result<RawHostAllocation<Allocator>>
allocate_host_raw_unadmitted(const std::size_t bytes) noexcept {
  return allocate_host_raw_unchecked<Allocator>(HostMemoryLease{}, bytes);
}

template <HostRawAllocator Allocator, SingleHostAllocationLayout Layout>
[[nodiscard]] Result<RawHostAllocation<Allocator>>
allocate_single_host_layout(HostMemoryBudget &budget,
                            const Layout &layout) noexcept {
  using Output = Result<RawHostAllocation<Allocator>>;
  HostMemoryRequirement requirement{};
  if (!try_make_single_host_allocation_requirement<
          Allocator::minimum_data_alignment>(layout, requirement)) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  auto admission = budget.try_admit(requirement);
  LPFX_TRY_RESULT(admission, Output);
  return allocate_host_raw<Allocator>(std::move(admission).value(),
                                      layout.total_bytes());
}

// One host allocation holding one array per element type, in order, each
// starting at a multiple of Alignment. Every element lifetime is started, and
// an empty array has a null base.
template <std::size_t Alignment, LifetimeStartEligible... T>
  requires(sizeof...(T) > 0 && std::has_single_bit(Alignment) &&
           Alignment >= alignof(std::max_align_t) &&
           ((Alignment >= alignof(T)) && ...))
class HostArrayPack final {
public:
  static constexpr std::size_t array_count = sizeof...(T);
  using allocator_type = HostAlignedAllocator<Alignment>;
  using counts_type = std::array<std::size_t, array_count>;
  template <std::size_t I>
  using element_type = std::tuple_element_t<I, std::tuple<T...>>;

  constexpr HostArrayPack() noexcept = default;

  HostArrayPack(HostArrayPack &&other) noexcept
      : allocation_(std::move(other.allocation_)),
        data_(std::exchange(other.data_, {})),
        counts_(std::exchange(other.counts_, {})) {}

  HostArrayPack &operator=(HostArrayPack &&other) noexcept {
    allocation_ = std::move(other.allocation_);
    data_ = std::exchange(other.data_, {});
    counts_ = std::exchange(other.counts_, {});
    return *this;
  }

  // Byte offset of every array and the allocation size for counts; overflow
  // when either is not representable.
  [[nodiscard]] static constexpr ErrorCode
  try_layout(const counts_type &counts, counts_type &offsets,
             std::size_t &total_bytes) noexcept {
    ByteLayoutBuilder builder{};
    counts_type candidate{};
    LPFX_TRY(meta::for_each_until_error<array_count>(
        [&](const auto index) noexcept {
          constexpr std::size_t I = decltype(index)::value;
          return builder.append<element_type<I>>(
              counts[I], Alignment, candidate[I]);
        }));
    offsets = candidate;
    total_bytes = builder.total_bytes();
    return ErrorCode::success;
  }

  // Admits the allocation against budget, or allocates unadmitted when budget
  // is null (self-contained owners). Returns Result<HostArrayPack>; the return
  // type is deduced because Result requires a complete value type.
  [[nodiscard]] static auto
  create(HostMemoryBudget *const budget, const counts_type &counts) noexcept {
    using Output = Result<HostArrayPack>;
    counts_type offsets{};
    std::size_t bytes = 0;
    LPFX_TRY_RESULT(try_layout(counts, offsets, bytes), Output);
    HostMemoryLease admission{};
    if (budget != nullptr) {
      HostMemoryRequirement requirement{};
      static_cast<void>(requirement.try_add(bytes, Alignment));
      auto admitted = budget->try_admit(requirement);
      LPFX_TRY_RESULT(admitted, Output);
      admission = std::move(admitted).value();
    }
    auto allocated =
        allocate_host_raw_unchecked<allocator_type>(std::move(admission), bytes);
    LPFX_TRY_RESULT(allocated, Output);

    HostArrayPack pack{};
    pack.allocation_ = std::move(allocated).value();
    pack.counts_ = counts;
    std::byte *const base = pack.allocation_.raw_base();
    meta::for_each_index<array_count>([&](const auto index) noexcept {
      constexpr std::size_t I = decltype(index)::value;
      std::get<I>(pack.data_) =
          start_host_array_lifetime<element_type<I>>(base + offsets[I],
                                                     counts[I]);
    });
    return Output::from_value(std::move(pack));
  }

  template <std::size_t I>
  [[nodiscard]] constexpr element_type<I> *data() noexcept {
    return std::get<I>(data_);
  }
  template <std::size_t I>
  [[nodiscard]] constexpr const element_type<I> *data() const noexcept {
    return std::get<I>(data_);
  }
  template <std::size_t I>
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return counts_[I];
  }
  template <std::size_t I>
  [[nodiscard]] constexpr BufferView<element_type<I>, HostSpace>
  view() noexcept {
    return make_buffer_view(data<I>(), size<I>());
  }
  template <std::size_t I>
  [[nodiscard]] constexpr BufferView<const element_type<I>, HostSpace>
  view() const noexcept {
    return make_buffer_view(data<I>(), size<I>());
  }

  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return allocation_.empty();
  }

  void reset() noexcept {
    allocation_.reset_capacity();
    data_ = {};
    counts_ = {};
  }

private:
  RawHostAllocation<allocator_type> allocation_{};
  std::tuple<T *...> data_{};
  counts_type counts_{};
};

} // namespace lpfx::detail

#endif
