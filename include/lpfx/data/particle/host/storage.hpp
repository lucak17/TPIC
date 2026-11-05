#ifndef LPFX_DATA_PARTICLE_HOST_STORAGE_HPP
#define LPFX_DATA_PARTICLE_HOST_STORAGE_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/host/view.hpp>
#include <lpfx/data/particle/layout.hpp>
#include <lpfx/data/particle/slots.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

namespace detail {

struct HostParticlesAccess;

struct ParticleRawSlot final {
  std::byte *raw_base{};
};

[[nodiscard]] inline bool particle_intervals_disjoint(
    const void *const first_base, const std::size_t first_bytes,
    const void *const second_base, const std::size_t second_bytes) noexcept {
  if (first_bytes == 0 || second_bytes == 0) {
    return true;
  }
  if (first_base == nullptr || second_base == nullptr) {
    return false;
  }

  const auto first = reinterpret_cast<std::uintptr_t>(first_base);
  const auto second = reinterpret_cast<std::uintptr_t>(second_base);
  std::uintptr_t first_extent{};
  std::uintptr_t second_extent{};
  std::uintptr_t first_end{};
  std::uintptr_t second_end{};
  if (!checked_integral_cast(first_bytes, first_extent) ||
      !checked_integral_cast(second_bytes, second_extent) ||
      !checked_add(first, first_extent, first_end) ||
      !checked_add(second, second_extent, second_end)) {
    return false;
  }
  return first_end <= second || second_end <= first;
}

template <LifetimeStartEligible T>
LPFX_ALWAYS_INLINE void
host_particle_store(T *const base, const std::size_t index,
                    const T &value) noexcept {
  if constexpr (std::is_assignable_v<T &, const T &>) {
    base[index] = value;
  } else {
    std::byte *const destination =
        reinterpret_cast<std::byte *>(base + index);
    static_cast<void>(
        std::memcpy(destination, std::addressof(value), sizeof(T)));
  }
}

} // namespace detail

// One budget-admitted (or self-contained) allocation laid out by
// PreparedParticleStorage: the attribute blocks plus a widest-attribute
// scratch block that rotates through the reorder carousel.
template <detail::ParticleSchemaType Schema>
class Particles<Schema, HostSpace> final {
public:
  using schema_type = Schema;
  using space_type = HostSpace;
  using view_type = BasicParticleView<Schema, HostSpace, false>;
  using const_view_type = BasicParticleView<Schema, HostSpace, true>;
  static constexpr std::size_t attribute_count = Schema::attribute_count;

  constexpr Particles() noexcept = default;

  Particles(Particles &&other) noexcept
      : allocation_(std::move(other.allocation_)),
        state_(std::exchange(other.state_, {})) {}

  Particles &operator=(Particles &&other) noexcept {
    if (this != &other) {
      allocation_ = std::move(other.allocation_);
      state_ = std::exchange(other.state_, {});
    }
    return *this;
  }

  [[nodiscard]] constexpr bool valid() const noexcept { return state_.valid; }
  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return state_.carousel.size;
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] constexpr std::size_t capacity() const noexcept {
    return state_.carousel.capacity;
  }

  void clear() noexcept { state_.carousel.size = 0; }

  [[nodiscard]] bool try_shrink(const std::size_t new_size) noexcept {
    if (new_size > size()) {
      return false;
    }
    state_.carousel.size = new_size;
    return true;
  }

  [[nodiscard]] view_type view() noexcept {
    return detail::PrivateConstructionAccess::create<view_type>(
        state_.carousel.bases(), state_.carousel.size);
  }

  [[nodiscard]] const_view_type view() const noexcept { return prefix(size()); }

  [[nodiscard]] const_view_type cview() const noexcept { return view(); }

  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] auto get() noexcept {
    return view().template get<Name>();
  }

  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] auto get() const noexcept {
    return view().template get<Name>();
  }

  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }

  [[nodiscard]] bool append(const ParticleValues<Schema> &values) noexcept {
    if (size() == capacity()) {
      return false;
    }
    store(state_.carousel.size, values);
    ++state_.carousel.size;
    return true;
  }

  template <class... Values>
    requires requires(Values &&...values) {
      detail::ParticleValuesFactory<Schema>::make(
          std::forward<Values>(values)...);
    }
  [[nodiscard]] bool append(Values &&...values) noexcept {
    return append(detail::ParticleValuesFactory<Schema>::make(
        std::forward<Values>(values)...));
  }

  // Applies source_for_destination: record d becomes the old record
  // source_for_destination[d] and size() becomes the map size. Returns false,
  // without changes, unless every map element is below size() and the map
  // storage is disjoint from this owner's allocation.
  template <detail::ParticleMapIndex MapIndex>
  [[nodiscard]] bool try_reorder(
      const BufferView<MapIndex, HostSpace> source_for_destination) noexcept {
    if (!valid() || source_for_destination.size() > size()) {
      return false;
    }

    using Index = std::remove_const_t<MapIndex>;
    std::size_t map_bytes{};
    if (!detail::checked_byte_count<Index>(source_for_destination.size(),
                                           map_bytes) ||
        !detail::particle_intervals_disjoint(source_for_destination.data(),
                                             map_bytes, allocation_.raw_base(),
                                             allocation_.capacity_bytes())) {
      return false;
    }
    for (std::size_t destination = 0;
         destination != source_for_destination.size(); ++destination) {
      std::size_t source{};
      if (!detail::checked_integral_cast(source_for_destination[destination],
                                         source) ||
          source >= size()) {
        return false;
      }
    }

    reorder_admitted(source_for_destination);
    return true;
  }

  void reset_capacity() noexcept {
    allocation_.reset_capacity();
    state_ = {};
  }

private:
  using allocator_type =
      HostAlignedAllocator<PreparedParticleStorage<Schema>::allocation_alignment>;
  using allocation_type = detail::RawHostAllocation<allocator_type>;
  using carousel_type =
      detail::ParticleSlotCarousel<Schema, detail::ParticleRawSlot,
                                   std::size_t>;

  struct State {
    carousel_type carousel{};
    bool valid{};
  };

  Particles(allocation_type &&allocation, const carousel_type &carousel) noexcept
      : allocation_(std::move(allocation)), state_{carousel, true} {}

  [[nodiscard]] const_view_type prefix(const std::size_t count) const noexcept {
    return detail::PrivateConstructionAccess::create<const_view_type>(
        state_.carousel.bases(), count);
  }

  void store(const std::size_t index,
             const ParticleValues<Schema> &values) noexcept {
    for_each_attribute<Schema>([&](const auto attribute) noexcept {
      constexpr std::size_t I = decltype(attribute)::value;
      using T = typename Schema::template value_type<I>;
      detail::host_particle_store(
          std::launder(
              reinterpret_cast<T *>(state_.carousel.slots[I].raw_base)),
          index, values.template get<I>());
    });
  }

  template <std::size_t I, detail::ParticleMapIndex MapIndex>
  LPFX_ALWAYS_INLINE void scatter_attribute(
      const BufferView<MapIndex, HostSpace> source_for_destination) noexcept {
    using T = typename Schema::template value_type<I>;
    auto &slots = state_.carousel.slots;
    const T *const source =
        std::launder(reinterpret_cast<const T *>(slots[I].raw_base));
    T *const destination = detail::start_host_array_lifetime<T>(
        slots[attribute_count].raw_base, state_.carousel.capacity);
    LPFX_INDEPENDENT_LOOP
    for (std::size_t destination_index = 0;
         destination_index != source_for_destination.size();
         ++destination_index) {
      const std::size_t source_index =
          static_cast<std::size_t>(source_for_destination[destination_index]);
      detail::host_particle_store(destination, destination_index,
                                  source[source_index]);
    }
  }

  template <bool Forward, detail::ParticleMapIndex MapIndex>
  LPFX_ALWAYS_INLINE void scatter_attributes(
      const BufferView<MapIndex, HostSpace> source_for_destination) noexcept {
    static_cast<void>(state_.carousel.template rotate<Forward>(
        [&](const auto attribute) noexcept {
          this->template scatter_attribute<decltype(attribute)::value>(
              source_for_destination);
          return ErrorCode::success;
        }));
  }

  template <detail::ParticleMapIndex MapIndex>
  void reorder_admitted(
      const BufferView<MapIndex, HostSpace> source_for_destination) noexcept {
    auto &carousel = state_.carousel;
    if (source_for_destination.size() == 0) {
      carousel.size = 0;
      return;
    }
    if (carousel.forward) {
      scatter_attributes<true>(source_for_destination);
    } else {
      scatter_attributes<false>(source_for_destination);
    }
    carousel.size = source_for_destination.size();
    carousel.forward = !carousel.forward;
  }

  allocation_type allocation_{};
  State state_{};

  friend struct detail::PrivateConstructionAccess;
  friend struct detail::HostParticlesAccess;
};

namespace detail {

struct HostParticlesAccess final {
  template <ParticleSchemaType Schema>
  [[nodiscard]] static HostMemoryBudget *
  admission_budget(const Particles<Schema, HostSpace> &particles) noexcept {
    return particles.allocation_.admission_budget();
  }

  // Hard preconditions: particles is valid; the map's length is at most the
  // current size; every live map element is nonnegative, representable as
  // size_t, and less than that size; the map storage is disjoint from the
  // complete particle allocation (including inactive capacity and scratch).
  // Neither owner nor map may change concurrently between that proof and
  // completion. Callers deriving these facts while constructing a map may
  // reuse them without an additional validation walk.
  template <ParticleSchemaType Schema, ParticleMapIndex MapIndex>
  static void reorder_admitted(
      Particles<Schema, HostSpace> &particles,
      const BufferView<MapIndex, HostSpace> source_for_destination) noexcept {
    particles.reorder_admitted(source_for_destination);
  }

  // Hard preconditions: index is in the owner's inactive capacity suffix and
  // the enclosing prepared protocol publishes no view covering that slot until
  // every schema attribute has been constructed.
  template <ParticleSchemaType Schema>
  static void
  construct_unpublished(Particles<Schema, HostSpace> &particles,
                        const std::size_t index,
                        const ParticleValues<Schema> &values) noexcept {
    particles.store(index, values);
  }

  // Hard preconditions: [0,new_size) is completely initialized and new_size
  // does not exceed capacity.
  template <ParticleSchemaType Schema>
  static void publish_initialized_size(Particles<Schema, HostSpace> &particles,
                                       const std::size_t new_size) noexcept {
    particles.state_.carousel.size = new_size;
  }

  // The caller must copy a complete prefix representation for this attribute
  // before publishing that prefix.
  template <std::size_t I, ParticleSchemaType Schema>
  [[nodiscard]] static std::byte *
  raw_attribute_base(Particles<Schema, HostSpace> &particles) noexcept {
    return particles.state_.carousel.slots[I].raw_base;
  }

  template <ParticleSchemaType Schema>
  [[nodiscard]] static BasicParticleView<Schema, HostSpace, true>
  const_prefix_view(const Particles<Schema, HostSpace> &particles,
                    const std::size_t count) noexcept {
    return particles.prefix(count);
  }
};

// The single host construction pipeline: allocate plan (admitted against
// budget when non-null), start every attribute lifetime, call
// initialize(meta::index_constant<I>{}, T *base) per attribute, and publish
// size records.
template <ParticleSchemaType Schema, class Initialize>
[[nodiscard]] Result<Particles<Schema, HostSpace>>
make_host_particles_with(HostMemoryBudget *const budget,
                         const PreparedParticleStorage<Schema> &plan,
                         const std::size_t size,
                         Initialize &&initialize) noexcept {
  using Owner = Particles<Schema, HostSpace>;
  using Allocator = HostAlignedAllocator<
      PreparedParticleStorage<Schema>::allocation_alignment>;
  if (!plan.valid() || size > plan.capacity()) {
    return Result<Owner>::from_error(ErrorCode::invalid_argument);
  }
  auto allocation =
      budget != nullptr
          ? allocate_single_host_layout<Allocator>(*budget, plan)
          : allocate_host_raw_unadmitted<Allocator>(plan.total_bytes());
  LPFX_TRY_RESULT(allocation, Result<Owner>);

  const auto bases = plan.bases(allocation.value().raw_base());
  for_each_attribute<Schema>([&](const auto attribute) noexcept {
    constexpr std::size_t I = decltype(attribute)::value;
    initialize(attribute,
               start_host_array_lifetime<typename Schema::template value_type<I>>(
                   bases[I], plan.capacity()));
  });
  ParticleSlotCarousel<Schema, ParticleRawSlot, std::size_t> carousel{};
  for (std::size_t slot = 0; slot != bases.size(); ++slot) {
    carousel.slots[slot].raw_base = bases[slot];
  }
  carousel.size = size;
  carousel.capacity = plan.capacity();
  return Result<Owner>::from_value(PrivateConstructionAccess::create<Owner>(
      std::move(allocation).value(), carousel));
}

} // namespace detail

// Allocates plan against budget, or self-contained when budget is null; a
// non-null budget must outlive the owner.
template <detail::ParticleSchemaType Schema>
[[nodiscard]] Result<Particles<Schema, HostSpace>>
make_host_particles(HostMemoryBudget *const budget,
                    const PreparedParticleStorage<Schema> &plan,
                    const std::size_t initial_size,
                    const ParticleValues<Schema> &initial_values) noexcept {
  return detail::make_host_particles_with(
      budget, plan, initial_size,
      [&](const auto attribute, auto *const base) noexcept {
        constexpr std::size_t I = decltype(attribute)::value;
        using T = typename Schema::template value_type<I>;
        if (initial_size == 0) {
          return;
        }
        if constexpr (std::is_assignable_v<T &, const T &>) {
          std::fill_n(base, initial_size, initial_values.template get<I>());
        } else {
          for (std::size_t index = 0; index != initial_size; ++index) {
            detail::host_particle_store(base, index,
                                        initial_values.template get<I>());
          }
        }
      });
}

// Grows source into a new owner that keeps the source's admission budget;
// self-contained and zero-capacity owners stay self-contained. The caller has
// retired every reader and writer of source. Preserve copies the first
// preserved_count records, which must be initialized even beyond size(), and
// publishes size(); discard publishes size zero. Failure leaves source
// unchanged.
template <ParticleResizeMode Mode = ParticleResizeMode::preserve,
          detail::ParticleSchemaType Schema>
[[nodiscard]] Result<Particles<Schema, HostSpace>>
make_grown_host_particles(const Particles<Schema, HostSpace> &source,
                          const std::size_t requested_capacity,
                          const std::size_t preserved_count) noexcept {
  using Owner = Particles<Schema, HostSpace>;
  constexpr bool preserve = Mode == ParticleResizeMode::preserve;
  if (!source.valid() || requested_capacity < source.capacity() ||
      (preserve && (preserved_count < source.size() ||
                    preserved_count > source.capacity()))) {
    return Result<Owner>::from_error(ErrorCode::invalid_argument);
  }
  PreparedParticleStorage<Schema> plan{};
  LPFX_TRY_RESULT(try_prepare_particle_layout(requested_capacity, plan),
                  Result<Owner>);
  const auto prefix =
      detail::HostParticlesAccess::const_prefix_view(source, preserved_count);
  return detail::make_host_particles_with(
      detail::HostParticlesAccess::admission_budget(source), plan,
      preserve ? source.size() : 0,
      [&](const auto attribute, auto *const base) noexcept {
        constexpr std::size_t I = decltype(attribute)::value;
        if (preserve && preserved_count != 0) {
          static_cast<void>(std::memcpy(
              base, prefix.template get<I>().data(),
              preserved_count * Schema::widths[I]));
        }
      });
}

} // namespace lpfx

#endif
