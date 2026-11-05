#ifndef LPFX_ALGORITHM_PARTICLE_HOST_SORT_HPP
#define LPFX_ALGORITHM_PARTICLE_HOST_SORT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <lpfx/algorithm/particle/cell_bins.hpp>
#include <lpfx/algorithm/particle/key.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/host/storage.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/allocator.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

template <detail::ParticleSortIndexPolicy IndexPolicy>
class PreparedSerialParticleSortWorkspace final {
public:
  using index_policy = IndexPolicy;
  using cursor_type = typename IndexPolicy::cursor_type;
  using map_index_type = typename IndexPolicy::map_index_type;

  static constexpr std::size_t allocation_alignment =
      std::max({AllocatorTraits<HostSpace>::minimum_data_alignment,
                alignof(cursor_type), alignof(map_index_type)});
  using storage_type =
      detail::HostArrayPack<allocation_alignment, cursor_type, map_index_type>;

  constexpr PreparedSerialParticleSortWorkspace() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::size_t bin_count() const noexcept {
    return bin_count_;
  }
  [[nodiscard]] constexpr std::size_t particle_capacity() const noexcept {
    return particle_capacity_;
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return total_bytes_;
  }

private:
  constexpr PreparedSerialParticleSortWorkspace(
      const std::size_t bin_count, const std::size_t particle_capacity,
      const std::size_t total_bytes) noexcept
      : bin_count_(bin_count), particle_capacity_(particle_capacity),
        total_bytes_(total_bytes), valid_(true) {}

  std::size_t bin_count_{};
  std::size_t particle_capacity_{};
  std::size_t total_bytes_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSortIndexPolicy IndexPolicy,
          detail::checked_integer BinCount,
          detail::checked_integer ParticleCapacity>
[[nodiscard]] constexpr ErrorCode try_prepare_serial_particle_sort_workspace(
    const BinCount requested_bin_count,
    const ParticleCapacity requested_particle_capacity,
    PreparedSerialParticleSortWorkspace<IndexPolicy> &output) noexcept {
  using Prepared = PreparedSerialParticleSortWorkspace<IndexPolicy>;
  if constexpr (std::is_signed_v<BinCount>) {
    if (requested_bin_count < BinCount{0}) {
      return ErrorCode::invalid_argument;
    }
  }
  if constexpr (std::is_signed_v<ParticleCapacity>) {
    if (requested_particle_capacity < ParticleCapacity{0}) {
      return ErrorCode::invalid_argument;
    }
  }
  std::size_t bin_count{};
  std::size_t particle_capacity{};
  if (!detail::checked_integral_cast(requested_bin_count, bin_count) ||
      !detail::checked_integral_cast(requested_particle_capacity,
                                     particle_capacity)) {
    return ErrorCode::overflow;
  }
  if (bin_count == 0) {
    return ErrorCode::invalid_argument;
  }
  typename Prepared::storage_type::counts_type offsets{};
  std::size_t total_bytes{};
  LPFX_TRY(Prepared::storage_type::try_layout({bin_count, particle_capacity},
                                              offsets, total_bytes));
  output = detail::PrivateConstructionAccess::create<Prepared>(
      bin_count, particle_capacity, total_bytes);
  return ErrorCode::success;
}

// Cursor and map scratch of one serial sort; any content is overwritten by
// the next run.
template <detail::ParticleSortIndexPolicy IndexPolicy>
class SerialParticleSortWorkspace final {
public:
  using index_policy = IndexPolicy;
  using cursor_type = typename IndexPolicy::cursor_type;
  using map_index_type = typename IndexPolicy::map_index_type;
  using prepared_type = PreparedSerialParticleSortWorkspace<IndexPolicy>;

  constexpr SerialParticleSortWorkspace() noexcept = default;

  SerialParticleSortWorkspace(SerialParticleSortWorkspace &&other) noexcept
      : storage_(std::move(other.storage_)),
        prepared_(std::exchange(other.prepared_, {})) {}

  SerialParticleSortWorkspace &
  operator=(SerialParticleSortWorkspace &&other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      prepared_ = std::exchange(other.prepared_, {});
    }
    return *this;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return prepared_.valid();
  }
  [[nodiscard]] constexpr std::size_t bin_count() const noexcept {
    return prepared_.bin_count();
  }
  [[nodiscard]] constexpr std::size_t particle_capacity() const noexcept {
    return prepared_.particle_capacity();
  }
  [[nodiscard]] cursor_type *cursors() noexcept {
    return storage_.template data<0>();
  }
  [[nodiscard]] map_index_type *map() noexcept {
    return storage_.template data<1>();
  }

  void reset_capacity() noexcept {
    storage_.reset();
    prepared_ = {};
  }

private:
  using storage_type = typename prepared_type::storage_type;

  SerialParticleSortWorkspace(storage_type &&storage,
                              const prepared_type prepared) noexcept
      : storage_(std::move(storage)), prepared_(prepared) {}

  storage_type storage_{};
  prepared_type prepared_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSortIndexPolicy IndexPolicy>
[[nodiscard]] Result<SerialParticleSortWorkspace<IndexPolicy>>
make_host_serial_particle_sort_workspace(
    HostMemoryBudget &budget,
    const PreparedSerialParticleSortWorkspace<IndexPolicy> &prepared) noexcept {
  using Owner = SerialParticleSortWorkspace<IndexPolicy>;
  using Storage =
      typename PreparedSerialParticleSortWorkspace<IndexPolicy>::storage_type;
  if (!prepared.valid()) {
    return Result<Owner>::from_error(ErrorCode::invalid_argument);
  }
  auto storage = Storage::create(
      &budget, {prepared.bin_count(), prepared.particle_capacity()});
  LPFX_TRY_RESULT(storage, Result<Owner>);
  return Result<Owner>::from_value(
      detail::PrivateConstructionAccess::create<Owner>(
          std::move(storage).value(), prepared));
}

// Stable counting sort of one host owner by a direct-cell key. The bound
// particles, bins and workspace must stay alive, unmoved and structurally
// unchanged (no reset or capacity change) while the plan is used.
template <detail::ParticleSchemaType Schema,
          detail::ParticleKeyAccessor KeyAccessor,
          detail::ParticleSortIndexPolicy IndexPolicy>
  requires detail::ParticleDirectCellKeyFor<Schema, KeyAccessor>
class PreparedSerialParticleSort final {
public:
  using schema_type = Schema;
  using key_accessor = KeyAccessor;
  using key_name_type = typename KeyAccessor::name_type;
  using key_type = detail::particle_key_t<Schema, KeyAccessor>;
  using index_policy = IndexPolicy;
  using count_type = typename IndexPolicy::count_type;
  using offset_type = typename IndexPolicy::offset_type;
  using cursor_type = typename IndexPolicy::cursor_type;
  using map_index_type = typename IndexPolicy::map_index_type;
  using particle_owner = Particles<Schema, HostSpace>;
  using bins_owner = CellBins<count_type, offset_type>;
  using workspace_owner = SerialParticleSortWorkspace<IndexPolicy>;

  constexpr PreparedSerialParticleSort() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return particles_ != nullptr;
  }

  // Hard precondition: valid(). Every recoverable rejection happens before
  // the owner or the completed bins image changes.
  [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode run() noexcept {
    using Count = count_type;
    using Offset = offset_type;
    using Cursor = cursor_type;
    using MapIndex = map_index_type;
    constexpr std::size_t key_index =
        detail::particle_key_index<Schema, KeyAccessor>;

    // A local copy: the cursor, map and bins stores below may alias this
    // object for the compiler (e.g. std::uint64_t cursors and
    // layout_.bin_count_), which would reload the layout per particle.
    const DirectCellBinLayout<key_type> layout = layout_;
    const auto particle_view = std::as_const(*particles_).view();
    const std::size_t particle_count = particle_view.size();
    if (particle_count > map_capacity_) [[unlikely]] {
      return ErrorCode::invalid_argument;
    }
    if (!bins_->can_publish()) [[unlikely]] {
      return ErrorCode::invalid_state;
    }
    const auto keys = particle_view.template get<key_index>();
    if (!fits<Offset>(particle_count) || !fits<Cursor>(particle_count) ||
        (particle_count != 0 && !fits<MapIndex>(particle_count - 1)))
        [[unlikely]] {
      return ErrorCode::capacity_exceeded;
    }

    const std::size_t bin_count = layout.bin_count();
    Count *const counts = detail::CellBinsAccess::counts(*bins_);
    Offset *const offsets = detail::CellBinsAccess::offsets(*bins_);
    if (particle_count == 0) {
      for (std::size_t bin = 0; bin != bin_count; ++bin) {
        counts[bin] = Count{0};
        offsets[bin] = Offset{0};
      }
      offsets[bin_count] = Offset{0};
      detail::CellBinsAccess::publish(*bins_);
      return ErrorCode::success;
    }

    Cursor *const cursors = cursors_;
    for (std::size_t bin = 0; bin != bin_count; ++bin) {
      cursors[bin] = Cursor{0};
    }

    for (std::size_t source = 0; source != particle_count; ++source) {
      std::size_t bin{};
      if (!layout.try_bin(keys[source], bin)) [[unlikely]] {
        return ErrorCode::invalid_argument;
      }
      ++cursors[bin];
    }

    // Offsets stay in scratch so every recoverable rejection leaves the last
    // completed CellBins image untouched.
    Cursor prefix{0};
    for (std::size_t bin = 0; bin != bin_count; ++bin) {
      const Cursor count = cursors[bin];
      if constexpr (std::numeric_limits<Count>::digits <
                    std::numeric_limits<Cursor>::digits) {
        constexpr Cursor maximum_count =
            static_cast<Cursor>(std::numeric_limits<Count>::max());
        if (cursors[bin] > maximum_count) [[unlikely]] {
          return ErrorCode::capacity_exceeded;
        }
      }
      cursors[bin] = prefix;
      prefix = static_cast<Cursor>(prefix + count);
    }

    MapIndex *const map = map_;
    for (std::size_t source = 0; source != particle_count; ++source) {
      const std::size_t bin = layout.unchecked_bin(keys[source]);
      const std::size_t destination = static_cast<std::size_t>(cursors[bin]++);
      map[destination] = static_cast<MapIndex>(source);
    }

    // The checked key histogram partitions [0, particle_count), so advancing
    // each bin's prefix cursor initializes every map entry exactly once with
    // a source below the unchanged size. The map is a separate allocation,
    // disjoint from all particle storage.
    detail::HostParticlesAccess::reorder_admitted(
        *particles_,
        detail::make_buffer_view<const MapIndex>(map, particle_count));

    Cursor previous_end{0};
    offsets[0] = Offset{0};
    for (std::size_t bin = 0; bin != bin_count; ++bin) {
      const Cursor end = cursors[bin];
      counts[bin] = static_cast<Count>(end - previous_end);
      offsets[bin + 1] = static_cast<Offset>(end);
      previous_end = end;
    }
    detail::CellBinsAccess::publish(*bins_);
    return ErrorCode::success;
  }

private:
  constexpr PreparedSerialParticleSort(
      particle_owner &particles, bins_owner &bins, cursor_type *const cursors,
      map_index_type *const map, const std::size_t map_capacity,
      const DirectCellBinLayout<key_type> layout) noexcept
      : particles_(&particles), bins_(&bins), cursors_(cursors), map_(map),
        map_capacity_(map_capacity), layout_(layout) {}

  template <class To>
  [[nodiscard]] static constexpr bool fits(const std::size_t value) noexcept {
    To converted{};
    return detail::checked_integral_cast(value, converted);
  }

  particle_owner *particles_ = nullptr;
  bins_owner *bins_ = nullptr;
  cursor_type *cursors_ = nullptr;
  map_index_type *map_ = nullptr;
  std::size_t map_capacity_{};
  DirectCellBinLayout<key_type> layout_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSchemaType Schema,
          detail::ParticleKeyAccessor KeyAccessor,
          detail::ParticleSortIndexPolicy IndexPolicy>
  requires detail::ParticleDirectCellKeyFor<Schema, KeyAccessor>
[[nodiscard]] constexpr ErrorCode try_prepare_serial_particle_sort(
    Particles<Schema, HostSpace> &particles,
    CellBins<typename IndexPolicy::count_type,
             typename IndexPolicy::offset_type> &bins,
    SerialParticleSortWorkspace<IndexPolicy> &workspace,
    const DirectCellBinLayout<detail::particle_key_t<Schema, KeyAccessor>>
        layout,
    PreparedSerialParticleSort<Schema, KeyAccessor, IndexPolicy>
        &output) noexcept {
  if (!particles.valid() || !bins.valid() || !workspace.valid() ||
      !layout.valid() || bins.bin_count() != layout.bin_count() ||
      workspace.bin_count() != layout.bin_count() ||
      workspace.particle_capacity() != particles.capacity()) {
    return ErrorCode::invalid_argument;
  }
  output = detail::PrivateConstructionAccess::create<
      PreparedSerialParticleSort<Schema, KeyAccessor, IndexPolicy>>(
      particles, bins, workspace.cursors(), workspace.map(),
      workspace.particle_capacity(), layout);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
