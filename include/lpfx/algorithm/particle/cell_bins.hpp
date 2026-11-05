#ifndef LPFX_ALGORITHM_PARTICLE_CELL_BINS_HPP
#define LPFX_ALGORITHM_PARTICLE_CELL_BINS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include <lpfx/algorithm/particle/key.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/allocator.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

// Device bins published by a device cell sort: bin_count counts and
// bin_count + 1 exclusive offsets.
struct DeviceCellBinsView final {
  const std::uint32_t *counts{};
  const std::uint32_t *offsets{};
  std::uint32_t bin_count{};
};

namespace detail {
struct CellBinsAccess;
} // namespace detail

template <detail::ParticleSortUnsignedIndex Count,
          detail::ParticleSortUnsignedIndex Offset>
class PreparedCellBins final {
public:
  using count_type = Count;
  using offset_type = Offset;

  static constexpr std::size_t allocation_alignment =
      std::max({AllocatorTraits<HostSpace>::minimum_data_alignment,
                alignof(Count), alignof(Offset)});
  using storage_type =
      detail::HostArrayPack<allocation_alignment, Count, Offset>;

  constexpr PreparedCellBins() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::size_t bin_count() const noexcept {
    return bin_count_;
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return total_bytes_;
  }
  [[nodiscard]] constexpr typename storage_type::counts_type
  array_counts() const noexcept {
    return {bin_count_, bin_count_ + 1};
  }

private:
  constexpr PreparedCellBins(const std::size_t bin_count,
                             const std::size_t total_bytes) noexcept
      : bin_count_(bin_count), total_bytes_(total_bytes), valid_(true) {}

  std::size_t bin_count_{};
  std::size_t total_bytes_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSortUnsignedIndex Count,
          detail::ParticleSortUnsignedIndex Offset,
          detail::checked_integer BinCount>
[[nodiscard]] constexpr ErrorCode
try_prepare_cell_bins(const BinCount requested_bin_count,
                      PreparedCellBins<Count, Offset> &output) noexcept {
  using Prepared = PreparedCellBins<Count, Offset>;
  std::size_t bin_count{};
  if (!detail::checked_integral_cast(requested_bin_count, bin_count) ||
      bin_count == 0) {
    return ErrorCode::invalid_argument;
  }
  if (bin_count == std::numeric_limits<std::size_t>::max()) {
    return ErrorCode::overflow;
  }
  typename Prepared::storage_type::counts_type offsets{};
  std::size_t total_bytes{};
  LPFX_TRY(Prepared::storage_type::try_layout({bin_count, bin_count + 1},
                                              offsets, total_bytes));
  output = detail::PrivateConstructionAccess::create<Prepared>(bin_count,
                                                               total_bytes);
  return ErrorCode::success;
}

template <detail::ParticleSortUnsignedIndex Count,
          detail::ParticleSortUnsignedIndex Offset>
class CellBinsView final {
public:
  using count_type = Count;
  using offset_type = Offset;

  constexpr CellBinsView() noexcept = default;

  [[nodiscard]] constexpr std::size_t bin_count() const noexcept {
    return bin_count_;
  }
  [[nodiscard]] constexpr std::uint64_t generation() const noexcept {
    return generation_;
  }
  [[nodiscard]] constexpr BufferView<const Count, HostSpace>
  counts() const noexcept {
    return detail::make_buffer_view<const Count>(counts_, bin_count_);
  }
  [[nodiscard]] constexpr BufferView<const Offset, HostSpace>
  offsets() const noexcept {
    return detail::make_buffer_view<const Offset>(offsets_, bin_count_ + 1);
  }

private:
  constexpr CellBinsView(const Count *const counts, const Offset *const offsets,
                         const std::size_t bin_count,
                         const std::uint64_t generation) noexcept
      : counts_(counts), offsets_(offsets), bin_count_(bin_count),
        generation_(generation) {}

  const Count *counts_ = nullptr;
  const Offset *offsets_ = nullptr;
  std::size_t bin_count_{};
  std::uint64_t generation_{};

  friend struct detail::PrivateConstructionAccess;
};

// Last completed bins image. Each successful sort publication advances the
// completed generation; a rejected sort leaves the previous image untouched.
template <detail::ParticleSortUnsignedIndex Count,
          detail::ParticleSortUnsignedIndex Offset>
class CellBins final {
public:
  using count_type = Count;
  using offset_type = Offset;
  using prepared_type = PreparedCellBins<Count, Offset>;

  constexpr CellBins() noexcept = default;

  CellBins(CellBins &&other) noexcept
      : storage_(std::move(other.storage_)),
        prepared_(std::exchange(other.prepared_, {})),
        generation_(std::exchange(other.generation_, 0)) {}

  CellBins &operator=(CellBins &&other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      prepared_ = std::exchange(other.prepared_, {});
      generation_ = std::exchange(other.generation_, 0);
    }
    return *this;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return prepared_.valid();
  }
  [[nodiscard]] constexpr std::size_t bin_count() const noexcept {
    return prepared_.bin_count();
  }
  [[nodiscard]] constexpr std::uint64_t completed_generation() const noexcept {
    return generation_;
  }
  // False once the completed generation can no longer advance.
  [[nodiscard]] constexpr bool can_publish() const noexcept {
    return generation_ != std::numeric_limits<std::uint64_t>::max();
  }

  [[nodiscard]] bool
  try_completed_view(CellBinsView<Count, Offset> &output) const noexcept {
    if (!valid() || generation_ == 0) {
      return false;
    }
    output = detail::PrivateConstructionAccess::create<
        CellBinsView<Count, Offset>>(storage_.template data<0>(),
                                     storage_.template data<1>(), bin_count(),
                                     generation_);
    return true;
  }

  void reset_capacity() noexcept {
    storage_.reset();
    prepared_ = {};
    generation_ = 0;
  }

private:
  using storage_type = typename prepared_type::storage_type;

  CellBins(storage_type &&storage, const prepared_type prepared) noexcept
      : storage_(std::move(storage)), prepared_(prepared) {}

  storage_type storage_{};
  prepared_type prepared_{};
  std::uint64_t generation_{};

  friend struct detail::CellBinsAccess;
  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

// Writer side for the sort algorithms. Callers check can_publish() before
// the first write, then publish exactly one completed image.
struct CellBinsAccess final {
  template <class Count, class Offset>
  [[nodiscard]] static Count *counts(CellBins<Count, Offset> &bins) noexcept {
    return bins.storage_.template data<0>();
  }
  template <class Count, class Offset>
  [[nodiscard]] static Offset *offsets(CellBins<Count, Offset> &bins) noexcept {
    return bins.storage_.template data<1>();
  }
  template <class Count, class Offset>
  static void publish(CellBins<Count, Offset> &bins) noexcept {
    ++bins.generation_;
  }
};

} // namespace detail

template <detail::ParticleSortUnsignedIndex Count,
          detail::ParticleSortUnsignedIndex Offset>
[[nodiscard]] Result<CellBins<Count, Offset>>
make_host_cell_bins(HostMemoryBudget &budget,
                    const PreparedCellBins<Count, Offset> &prepared) noexcept {
  using Owner = CellBins<Count, Offset>;
  using Storage = typename PreparedCellBins<Count, Offset>::storage_type;
  if (!prepared.valid()) {
    return Result<Owner>::from_error(ErrorCode::invalid_argument);
  }
  auto storage = Storage::create(&budget, prepared.array_counts());
  LPFX_TRY_RESULT(storage, Result<Owner>);
  return Result<Owner>::from_value(detail::PrivateConstructionAccess::create<
                                   Owner>(std::move(storage).value(),
                                          prepared));
}

} // namespace lpfx

#endif
