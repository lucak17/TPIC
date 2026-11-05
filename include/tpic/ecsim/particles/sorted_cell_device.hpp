#ifndef TPIC_ECSIM_PARTICLES_SORTED_CELL_DEVICE_HPP
#define TPIC_ECSIM_PARTICLES_SORTED_CELL_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/algorithm/particle/device/sort.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/backend/launch.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/block_threads.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/particles/particle_cell.hpp>
#include <tpic/ecsim/particles/cic.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace tpic::ecsim {

// CUDA warps and HIP wave32/wave64 use the same fixed block size.
inline constexpr unsigned sorted_cell_threads = device::particle_block_threads;
// Cells owned by a block of Threads threads when each native subgroup of
// Width lanes owns one cell.
template <unsigned Width, unsigned Threads = sorted_cell_threads>
inline constexpr unsigned cells_per_block = Threads / Width;
// Size shared storage for the smallest supported native subgroup, including
// host passes.
inline constexpr unsigned smallest_subgroup_width = 32;
inline constexpr unsigned sorted_cell_max_warps = cells_per_block<smallest_subgroup_width>;
using SortedCellLaunch = lpfx::device::KernelLaunch<sorted_cell_threads, sorted_cell_max_warps>;
// One cached cell field: three components, each at the eight CIC corners.
inline constexpr int cached_cell_values = 3 * cic_corner_count;

template <class Compute> struct CachedCicComponent {
  using location_type = lpfx::Node;
  const Compute* values;
  data::GridIndex cell;

  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE Compute operator()(int i, int j, int k) const noexcept {
    return values[(i - cell.i) + 2 * (j - cell.j) + 4 * (k - cell.k)];
  }
};

// One cached cell field: three components, each at the eight CIC corners,
// read through the normal component-gather interface. The interpolation
// summation order and the conversion to Compute remain unchanged.
template <class Compute> struct CachedCellNodes {
  const Compute* values;
  data::GridIndex cell;

  template <std::size_t Axis>
  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE CachedCicComponent<Compute> component() const noexcept {
    static_assert(Axis < 3);
    return {values + cic_corner_count * Axis, cell};
  }
};

// Tag-to-axis adapter for gathers that address B by schema tag.
template <class Precision, class Compute> struct CachedMagneticCell {
  CachedCellNodes<Compute> nodes;

  template <class Tag>
  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE CachedCicComponent<Compute> get() const noexcept {
    using Fields = data::FieldSchema<Precision>;
    static_assert(std::is_same_v<Tag, typename Fields::Bnx> ||
                  std::is_same_v<Tag, typename Fields::Bny> ||
                  std::is_same_v<Tag, typename Fields::Bnz>);
    constexpr std::size_t axis = std::is_same_v<Tag, typename Fields::Bnx> ? 0 :
        std::is_same_v<Tag, typename Fields::Bny> ? 1 : 2;
    return nodes.template component<axis>();
  }
};

// Lanes 0-7 load one corner each of three node components. The caller
// synchronizes the warp before any lane reads the cache.
template <class Compute, class Component0, class Component1, class Component2>
__device__ LPFX_ALWAYS_INLINE void fill_cell_nodes(Component0 c0, Component1 c1, Component2 c2,
    data::GridIndex cell, Compute* values) noexcept {
  const unsigned lane = lpfx::device::subgroup::warp_lane();
  if (lane < cic_corner_count) {
    const auto node = cic_corner(cell, lane);
    values[lane] = config::convert<Compute>(c0(node.i, node.j, node.k));
    values[cic_corner_count + lane] = config::convert<Compute>(c1(node.i, node.j, node.k));
    values[2 * cic_corner_count + lane] = config::convert<Compute>(c2(node.i, node.j, node.k));
  }
}

template <class Precision, class Compute, class MagneticView>
__device__ LPFX_ALWAYS_INLINE void fill_cell_magnetic(MagneticView magnetic,
    data::GridIndex cell, Compute* values) noexcept {
  using Fields = data::FieldSchema<Precision>;
  fill_cell_nodes(magnetic.template get<typename Fields::Bnx>(),
      magnetic.template get<typename Fields::Bny>(),
      magnetic.template get<typename Fields::Bnz>(), cell, values);
}

// Fill this warp's cache with the cell's magnetic field and make it visible
// to every lane. The caller synchronizes again before reusing the storage.
template <class Precision, class Compute, class MagneticView>
[[nodiscard]] __device__ LPFX_ALWAYS_INLINE CachedMagneticCell<Precision, Compute>
cache_cell_magnetic(MagneticView magnetic, data::GridIndex cell, Compute* values) noexcept {
  fill_cell_magnetic<Precision>(magnetic, cell, values);
  lpfx::device::subgroup::warp_synchronize();
  return {{values, cell}};
}

// Interior cells in the x-fastest order of the particle cell map.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr unsigned interior_cell_count(
    data::ParticleCellMap cells) noexcept {
  return static_cast<unsigned>(cells.nx) * static_cast<unsigned>(cells.ny) *
      static_cast<unsigned>(cells.nz);
}

// Result of opening an interior bin for the sorted mover.
enum class InteriorBinStatus {
  open,
  // The bin's own range is inconsistent; first and count are not usable.
  unreadable,
  // The bin's range is usable, but a neighbouring halo bin holds particles.
  halo_occupied
};

// Borrowed device metadata for one checked sort epoch. The receipt and its
// sorter survive every consumer, including the mover that makes bins stale.
struct ParticleBins {
  lpfx::DeviceCellBinsView view{};
  const std::uint32_t* failure{};
  std::uint32_t prefix{};

  // The producer's failure word; no bin metadata may be read when set.
  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool failed() const noexcept { return *failure != 0; }

  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool ready(unsigned* invalid) const noexcept {
    if (!failed()) return true;
    atomicExch(invalid, 1U);
    return false;
  }

  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool range(
      unsigned bin, unsigned& first, unsigned& count,
      unsigned* invalid) const noexcept {
    first = view.offsets[bin];
    count = view.counts[bin];
    const unsigned end = view.offsets[bin + 1];
    if (first > prefix || count > prefix - first || end != first + count ||
        (bin == 0 && first != 0) ||
        (bin + 1 == view.bin_count && end != prefix)) {
      atomicExch(invalid, 1U);
      return false;
    }
    return true;
  }

  // Open the interior cell with x-fastest index `interior`. Every halo bin
  // holds no particle exactly when the interior ranges tile [0, prefix): the
  // first interior bin starts at 0, each interior row ends where the next
  // interior row starts, and the last one ends at prefix. Metadata of an empty
  // halo bin is not read. Reports nothing; the caller owns the reporting form.
  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE InteriorBinStatus interior_range(
      data::ParticleCellMap cells, unsigned interior, data::GridIndex& cell,
      unsigned& bin, unsigned& first, unsigned& count) const noexcept {
    const unsigned nx = static_cast<unsigned>(cells.nx);
    const unsigned ny = static_cast<unsigned>(cells.ny);
    const unsigned yz = interior / nx;
    cell = {static_cast<int>(interior - yz * nx), static_cast<int>(yz % ny),
            static_cast<int>(yz / ny)};
    bin = static_cast<unsigned>(cells.encode(cell.i, cell.j, cell.k));
    first = view.offsets[bin];
    count = view.counts[bin];
    const unsigned end = view.offsets[bin + 1];
    if (first > prefix || count > prefix - first || end != first + count)
      return InteriorBinStatus::unreadable;
    if (interior == 0 && first != 0) return InteriorBinStatus::halo_occupied;
    if (cell.i + 1 == cells.nx) {
      unsigned next = prefix;
      if (cell.j + 1 < cells.ny)
        next = view.offsets[cells.encode(0, cell.j + 1, cell.k)];
      else if (cell.k + 1 < cells.nz)
        next = view.offsets[cells.encode(0, 0, cell.k + 1)];
      if (end != next) return InteriorBinStatus::halo_occupied;
    }
    return InteriorBinStatus::open;
  }
};

// Open a sorted bin in the uniform per-lane form: every lane reads the same
// range and decodes the same cell. A bin outside the interior holding
// particles is reported once, from lane 0. Returns false for an empty or
// rejected bin; the warp then continues with its next bin.
[[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool open_sorted_cell(
    const ParticleBins& bins, data::ParticleCellMap cells, unsigned bin,
    unsigned* invalid, unsigned& first, unsigned& count, data::GridIndex& cell) noexcept {
  if (!bins.range(bin, first, count, invalid) || count == 0) return false;
  cell = cells.decode(static_cast<int>(bin));
  if (cells.interior_bin(cell.i, cell.j, cell.k) < 0) {
    if (lpfx::device::subgroup::warp_lane() == 0) atomicExch(invalid, 1U);
    return false;
  }
  return true;
}

// Bins of the resident prefix, through LPFx's checked resident read only: the
// owner's published size may still change, so no owner view is taken.
template <class Schema, class Backend>
[[nodiscard]] lpfx::ErrorCode device_wait_resident_particle_bins(
    const lpfx::ResidentParticleRead<Schema, Backend>& residents,
    data::ParticleCellMap cells, lpfx::device::Executor<Backend> executor,
    ParticleBins& bins) noexcept {
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  if (!residents.valid() || residents.generation() == 0)
    return lpfx::ErrorCode::invalid_state;
  const auto view = residents.resident_particle_view();
  const auto prefix = residents.prefix_size();
  if (prefix > std::uint32_t(std::numeric_limits<int>::max()))
    return lpfx::ErrorCode::overflow;
  if (view.size() != prefix) return lpfx::ErrorCode::invalid_argument;
  const auto metadata = residents.bins();
  if (cells.count <= 0 || metadata.bin_count != static_cast<unsigned>(cells.count) ||
      metadata.counts == nullptr || metadata.offsets == nullptr ||
      residents.failure_word() == nullptr)
    return lpfx::ErrorCode::invalid_argument;
  if (prefix != 0) {
    const bool pointers_present = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return ((view.template get<I>() != nullptr) && ...);
    }(std::make_index_sequence<decltype(view)::attribute_count>{});
    if (!pointers_present) return lpfx::ErrorCode::invalid_argument;
  }
  using Traits = lpfx::device::BackendTraits<Backend>;
  if (!Traits::succeeded(residents.wait_on(executor)))
    return lpfx::ErrorCode::backend_failure;
  bins = {metadata, residents.failure_word(), prefix};
  return lpfx::ErrorCode::success;
}

// Bins of a completed full sort, or of a partial sort whose migration has
// concluded. Comparing every attribute address requires the receipt's
// committed mapping and the same owner.
template <class ParticleOwner, class Schema, class Backend>
[[nodiscard]] lpfx::ErrorCode device_wait_particle_bins(
    const ParticleOwner& particles,
    const lpfx::DeviceSortSubmission<Schema, Backend>& receipt,
    data::ParticleCellMap cells, lpfx::device::Executor<Backend> executor,
    ParticleBins& bins) noexcept {
  static_assert(std::is_same_v<typename ParticleOwner::schema_type, Schema> &&
                std::is_same_v<typename ParticleOwner::backend_type, Backend>);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  if (!particles.valid() || !receipt.valid()) return lpfx::ErrorCode::invalid_state;
  if (receipt.prefix_size() > particles.size()) return lpfx::ErrorCode::invalid_argument;
  const auto view = receipt.bins();
  // LPFx uses the stored halo-2 Cell key directly. Halo bins must be empty
  // after migration; the particle algebra rejects noninterior records.
  if (view.bin_count != static_cast<unsigned>(cells.count) ||
      view.counts == nullptr || view.offsets == nullptr || receipt.failure_word() == nullptr)
    return lpfx::ErrorCode::invalid_argument;
  const auto published = particles.cview();
  const auto pending = receipt.pending_particle_view();
  const bool same_owner = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return ((published.template get<I>() == pending.template get<I>()) && ...);
  }(std::make_index_sequence<ParticleOwner::attribute_count>{});
  if (!same_owner) return lpfx::ErrorCode::invalid_state;
  using Traits = lpfx::device::BackendTraits<Backend>;
  if (!Traits::succeeded(receipt.wait_on(executor))) return lpfx::ErrorCode::backend_failure;
  bins = {view, receipt.failure_word(), receipt.prefix_size()};
  return lpfx::ErrorCode::success;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_PARTICLES_SORTED_CELL_DEVICE_HPP
