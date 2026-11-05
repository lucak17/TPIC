#ifndef LPFX_ALGORITHM_PARTICLE_HOST_COMPACTION_HPP
#define LPFX_ALGORITHM_PARTICLE_HOST_COMPACTION_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/host/storage.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/allocator.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

namespace detail {

template <class T>
concept ParticleCompactionIndex =
    checked_integer<T> && std::is_unsigned_v<T> && LifetimeStartEligible<T>;

template <class Predicate, class View>
concept NothrowParticleKeepPredicate = requires(
    const Predicate &predicate, const View &view, const std::size_t index) {
  { predicate(view, index) } noexcept -> std::same_as<bool>;
};

template <class View, std::size_t Attribute>
using particle_view_value_t =
    typename decltype(std::declval<const View &>()
                          .template get<Attribute>())::value_type;

// One pass over the move lists copies every attribute of a record: holes lie
// in [0, retained) and fillers in [retained, count), so no copy reads a slot
// that another copy writes, and the attributes are distinct arrays.
template <class View, ParticleCompactionIndex Index, std::size_t... Attribute>
LPFX_ALWAYS_INLINE void
move_particle_records(const View &particles, const Index *const holes,
                      const Index *const fillers, const std::size_t move_count,
                      std::index_sequence<Attribute...>) noexcept {
  std::byte *const bases[] = {
      particles.template get<Attribute>().raw_base()...};
  for (std::size_t move = 0; move != move_count; ++move) {
    const std::size_t hole = static_cast<std::size_t>(holes[move]);
    const std::size_t filler = static_cast<std::size_t>(fillers[move]);
    (std::memcpy(bases[Attribute] +
                     hole * sizeof(particle_view_value_t<View, Attribute>),
                 bases[Attribute] +
                     filler * sizeof(particle_view_value_t<View, Attribute>),
                 sizeof(particle_view_value_t<View, Attribute>)),
     ...);
  }
}

// Exact in-place hole fill of a host particle prefix. keep(i) classifies
// record i of [0, particle_count), of which exactly retained_count are kept.
// Holes are the removed records of [0, retained_count) in ascending order;
// fillers are the kept records of [retained_count, particle_count) in
// descending order. holes and fillers each hold move_capacity indices.
// Returns invalid_state, before moving any record, when the two counts differ
// or exceed min(retained, removed) or move_capacity; keep may then have been
// called for every record. List entries past move_count are unspecified. The
// owner size is not changed.
template <class View, class Keep, ParticleCompactionIndex Index>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode compact_exact_in_place(
    const View &particles, const Keep &keep, const std::size_t retained_count,
    const std::size_t particle_count, Index *const holes, Index *const fillers,
    const std::size_t move_capacity, std::size_t &move_count) noexcept {
  move_count = 0;
  if (retained_count == 0 || retained_count == particle_count) {
    return ErrorCode::success;
  }
  // Branch-free lists: every candidate is stored at the list end (a local
  // sink once the list is full) and the count advances only for a selected
  // record. Overflow is still rejected before any record moves.
  Index sink{};
  std::size_t hole_count{};
  for (std::size_t index = 0; index != retained_count; ++index) {
    Index *const slot = hole_count < move_capacity ? holes + hole_count : &sink;
    *slot = static_cast<Index>(index);
    hole_count += static_cast<std::size_t>(!keep(index));
  }
  std::size_t filler_count{};
  for (std::size_t end = particle_count; end != retained_count; --end) {
    const std::size_t index = end - 1;
    Index *const slot =
        filler_count < move_capacity ? fillers + filler_count : &sink;
    *slot = static_cast<Index>(index);
    filler_count += static_cast<std::size_t>(keep(index));
  }
  const std::size_t removed_count = particle_count - retained_count;
  const std::size_t maximum_moves =
      retained_count < removed_count ? retained_count : removed_count;
  if (hole_count > move_capacity || filler_count != hole_count ||
      hole_count > maximum_moves) [[unlikely]] {
    return ErrorCode::invalid_state;
  }
  move_particle_records(particles, holes, fillers, hole_count,
                        std::make_index_sequence<View::attribute_count>{});
  move_count = hole_count;
  return ErrorCode::success;
}

struct ParticleClassificationKeep final {
  const std::uint8_t *classification;

  [[nodiscard]] LPFX_ALWAYS_INLINE bool
  operator()(const std::size_t index) const noexcept {
    return classification[index] != 0;
  }
};

} // namespace detail

template <detail::ParticleCompactionIndex Index>
class PreparedSerialParticleCompactionWorkspace final {
public:
  using index_type = Index;

  static constexpr std::size_t allocation_alignment = std::max(
      AllocatorTraits<HostSpace>::minimum_data_alignment, alignof(Index));
  using storage_type = detail::HostArrayPack<allocation_alignment,
                                             std::uint8_t, Index, Index>;

  constexpr PreparedSerialParticleCompactionWorkspace() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::size_t particle_capacity() const noexcept {
    return particle_capacity_;
  }
  // Capacity of each of the hole and filler lists.
  [[nodiscard]] constexpr std::size_t move_capacity() const noexcept {
    return particle_capacity_ / 2;
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return total_bytes_;
  }
  [[nodiscard]] constexpr typename storage_type::counts_type
  array_counts() const noexcept {
    return {particle_capacity_, move_capacity(), move_capacity()};
  }

private:
  constexpr PreparedSerialParticleCompactionWorkspace(
      const std::size_t particle_capacity,
      const std::size_t total_bytes) noexcept
      : particle_capacity_(particle_capacity), total_bytes_(total_bytes),
        valid_(true) {}

  std::size_t particle_capacity_{};
  std::size_t total_bytes_{};
  bool valid_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleCompactionIndex Index,
          detail::checked_integer Capacity>
[[nodiscard]] constexpr ErrorCode
try_prepare_serial_particle_compaction_workspace(
    const Capacity requested_capacity,
    PreparedSerialParticleCompactionWorkspace<Index> &output) noexcept {
  using Prepared = PreparedSerialParticleCompactionWorkspace<Index>;
  std::size_t particle_capacity{};
  if (!detail::checked_integral_cast(requested_capacity, particle_capacity)) {
    return ErrorCode::invalid_argument;
  }
  if (particle_capacity != 0) {
    Index last_index{};
    if (!detail::checked_integral_cast(particle_capacity - 1, last_index)) {
      return ErrorCode::capacity_exceeded;
    }
  }
  const std::size_t move_capacity = particle_capacity / 2;
  typename Prepared::storage_type::counts_type offsets{};
  std::size_t total_bytes{};
  LPFX_TRY(Prepared::storage_type::try_layout(
      {particle_capacity, move_capacity, move_capacity}, offsets,
      total_bytes));
  output = detail::PrivateConstructionAccess::create<Prepared>(
      particle_capacity, total_bytes);
  return ErrorCode::success;
}

// Classification byte per record plus the hole and filler lists; any content
// is overwritten by the next run.
template <detail::ParticleCompactionIndex Index>
class SerialParticleCompactionWorkspace final {
public:
  using index_type = Index;
  using prepared_type = PreparedSerialParticleCompactionWorkspace<Index>;

  constexpr SerialParticleCompactionWorkspace() noexcept = default;

  SerialParticleCompactionWorkspace(
      SerialParticleCompactionWorkspace &&other) noexcept
      : storage_(std::move(other.storage_)),
        prepared_(std::exchange(other.prepared_, {})) {}

  SerialParticleCompactionWorkspace &
  operator=(SerialParticleCompactionWorkspace &&other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      prepared_ = std::exchange(other.prepared_, {});
    }
    return *this;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return prepared_.valid();
  }
  [[nodiscard]] constexpr std::size_t particle_capacity() const noexcept {
    return prepared_.particle_capacity();
  }
  [[nodiscard]] constexpr std::size_t move_capacity() const noexcept {
    return prepared_.move_capacity();
  }
  [[nodiscard]] std::uint8_t *classification() noexcept {
    return storage_.template data<0>();
  }
  [[nodiscard]] Index *holes() noexcept { return storage_.template data<1>(); }
  [[nodiscard]] Index *fillers() noexcept {
    return storage_.template data<2>();
  }

  void reset_capacity() noexcept {
    storage_.reset();
    prepared_ = {};
  }

private:
  using storage_type = typename prepared_type::storage_type;

  SerialParticleCompactionWorkspace(storage_type &&storage,
                                    const prepared_type prepared) noexcept
      : storage_(std::move(storage)), prepared_(prepared) {}

  storage_type storage_{};
  prepared_type prepared_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleCompactionIndex Index>
[[nodiscard]] Result<SerialParticleCompactionWorkspace<Index>>
make_host_serial_particle_compaction_workspace(
    HostMemoryBudget &budget,
    const PreparedSerialParticleCompactionWorkspace<Index> &prepared) noexcept {
  using Owner = SerialParticleCompactionWorkspace<Index>;
  using Storage =
      typename PreparedSerialParticleCompactionWorkspace<Index>::storage_type;
  if (!prepared.valid()) {
    return Result<Owner>::from_error(ErrorCode::invalid_argument);
  }
  auto storage = Storage::create(&budget, prepared.array_counts());
  LPFX_TRY_RESULT(storage, Result<Owner>);
  return Result<Owner>::from_value(
      detail::PrivateConstructionAccess::create<Owner>(
          std::move(storage).value(), prepared));
}

struct SerialParticleCompactionResult final {
  std::size_t original_count{};
  std::size_t retained_count{};
  std::size_t move_count{};
};

// Exact-count unstable compaction of one host owner. The bound particles and
// workspace must stay alive, unmoved and structurally unchanged (no reset or
// capacity change) while the plan is used.
template <detail::ParticleSchemaType Schema,
          detail::ParticleCompactionIndex Index>
class PreparedSerialParticleCompaction final {
public:
  using schema_type = Schema;
  using index_type = Index;
  using particle_owner = Particles<Schema, HostSpace>;
  using workspace_owner = SerialParticleCompactionWorkspace<Index>;
  using const_view_type = typename particle_owner::const_view_type;

  constexpr PreparedSerialParticleCompaction() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return particles_ != nullptr;
  }

  // Keeps every record for which predicate(view, index) is true and shrinks
  // the owner to expected_retained_count. Hard precondition: valid().
  // invalid_argument when the kept count differs from
  // expected_retained_count; the owner is then unchanged.
  template <class Predicate>
    requires detail::NothrowParticleKeepPredicate<Predicate, const_view_type>
  [[nodiscard]] LPFX_ALWAYS_INLINE Result<SerialParticleCompactionResult>
  run(const Predicate &predicate,
      const std::size_t expected_retained_count) noexcept {
    using Output = Result<SerialParticleCompactionResult>;
    const std::size_t particle_count = particles_->size();
    if (particle_count > particle_capacity_ ||
        expected_retained_count > particle_count) [[unlikely]] {
      return Output::from_error(ErrorCode::invalid_argument);
    }

    std::uint8_t *const classification = classification_;
    const const_view_type source_view = std::as_const(*particles_).view();
    std::size_t retained_count{};
    for (std::size_t source = 0; source != particle_count; ++source) {
      const bool keep = predicate(source_view, source);
      classification[source] = static_cast<std::uint8_t>(keep);
      retained_count += static_cast<std::size_t>(keep);
    }
    if (retained_count != expected_retained_count) [[unlikely]] {
      return Output::from_error(ErrorCode::invalid_argument);
    }

    std::size_t move_count{};
    LPFX_TRY_RESULT(detail::compact_exact_in_place(
                        particles_->view(),
                        detail::ParticleClassificationKeep{classification},
                        retained_count, particle_count, holes_, fillers_,
                        particle_capacity_ / 2, move_count),
                    Output);
    static_cast<void>(particles_->try_shrink(retained_count));
    return Output::from_value(SerialParticleCompactionResult{
        particle_count, retained_count, move_count});
  }

private:
  constexpr PreparedSerialParticleCompaction(
      particle_owner &particles, std::uint8_t *const classification,
      Index *const holes, Index *const fillers,
      const std::size_t particle_capacity) noexcept
      : particles_(&particles), classification_(classification), holes_(holes),
        fillers_(fillers), particle_capacity_(particle_capacity) {}

  particle_owner *particles_ = nullptr;
  std::uint8_t *classification_ = nullptr;
  Index *holes_ = nullptr;
  Index *fillers_ = nullptr;
  std::size_t particle_capacity_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleSchemaType Schema,
          detail::ParticleCompactionIndex Index>
[[nodiscard]] constexpr ErrorCode try_prepare_serial_particle_compaction(
    Particles<Schema, HostSpace> &particles,
    SerialParticleCompactionWorkspace<Index> &workspace,
    PreparedSerialParticleCompaction<Schema, Index> &output) noexcept {
  if (!particles.valid() || !workspace.valid() ||
      particles.capacity() != workspace.particle_capacity()) {
    return ErrorCode::invalid_argument;
  }
  output = detail::PrivateConstructionAccess::create<
      PreparedSerialParticleCompaction<Schema, Index>>(
      particles, workspace.classification(), workspace.holes(),
      workspace.fillers(), workspace.particle_capacity());
  return ErrorCode::success;
}

} // namespace lpfx

#endif
