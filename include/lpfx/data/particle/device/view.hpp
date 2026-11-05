#ifndef LPFX_DATA_PARTICLE_DEVICE_VIEW_HPP
#define LPFX_DATA_PARTICLE_DEVICE_VIEW_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/fixed_array.hpp>
#include <lpfx/core/memory_intrinsics.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/particle/layout.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx {

// A device particle view is exactly one pointer per schema attribute plus the
// published 32-bit prefix count used by all current kernels.  Attribute choice
// is compile-time; get/store lower to one pointer load and one indexed access.
// A mutable view converts implicitly to the const view of the same bases.
template <detail::ParticleSchemaType Schema, device::DeviceBackend Backend,
          bool Const>
class BasicParticleView<Schema, DeviceSpace<Backend>, Const> final {
public:
  using schema_type = Schema;
  using space_type = DeviceSpace<Backend>;
  using count_type = std::uint32_t;
  static constexpr std::size_t attribute_count = Schema::attribute_count;
  // A view-only extension backend intentionally has no allocation guarantee.
  // Keep the pointer view usable for such backends while publishing an
  // alignment contract only when the backend can actually allocate storage.
  static constexpr std::size_t data_alignment = [] {
    if constexpr (device::DeviceAllocationBackend<Backend>) {
      return detail::particle_allocation_alignment<space_type, Schema>;
    } else {
      return std::size_t{0};
    }
  }();

  LPFX_HOST_DEVICE constexpr BasicParticleView() noexcept {}

  template <bool OtherConst>
    requires(Const && !OtherConst)
  LPFX_HOST_DEVICE constexpr BasicParticleView(
      const BasicParticleView<Schema, space_type, OtherConst> other) noexcept
      : bases_(other.bases_), count_(other.count_) {}

  [[nodiscard]] LPFX_HOST_DEVICE constexpr count_type size() const noexcept {
    return count_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool empty() const noexcept {
    return count_ == 0;
  }

  template <std::size_t I>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto get() const noexcept {
    using Value = typename Schema::template value_type<I>;
    using Element = std::conditional_t<Const, const Value, Value>;
    return reinterpret_cast<Element *>(bases_[I]);
  }

  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] LPFX_HOST_DEVICE constexpr auto get() const noexcept {
    return get<Schema::template index_of<Name>>();
  }

  // Owner-backed SoA bases all obey data_alignment. This explicit accessor
  // preserves that proof through generic device kernels without changing the
  // pointer-only view representation or adding a runtime check.
  template <std::size_t I>
    requires device::DeviceAllocationBackend<Backend>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE auto *
  aligned_data() const noexcept {
    return detail::assume_aligned_pointer<data_alignment>(get<I>());
  }

  template <class Name>
    requires Schema::template contains<Name> &&
             device::DeviceAllocationBackend<Backend>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE auto *
  aligned_data() const noexcept {
    return aligned_data<Schema::template index_of<Name>>();
  }

  template <std::size_t I>
    requires(!Const)
  LPFX_HOST_DEVICE constexpr void
  store(const count_type index,
        const typename Schema::template value_type<I> value) const noexcept {
    get<I>()[index] = value;
  }

private:
  using bases_type = detail::DeviceFixedArray<std::byte *, attribute_count>;

  LPFX_HOST_DEVICE constexpr BasicParticleView(const bases_type bases,
                                               const count_type count) noexcept
      : bases_(bases), count_(count) {
    static_assert(std::is_trivially_copyable_v<BasicParticleView>);
  }

  bases_type bases_{};
  count_type count_{};

  template <detail::ParticleSchemaType, class, bool>
  friend class BasicParticleView;
  friend struct detail::PrivateConstructionAccess;
};

namespace detail {

// One typed load/store pair per schema attribute; no record proxy, loop bound,
// callback or runtime dispatch enters a device lane.
template <ParticleSchemaType Schema, device::DeviceBackend Backend>
struct DeviceParticleRecordCopy final {
  BasicParticleView<Schema, DeviceSpace<Backend>, false> destination;
  std::uint32_t destination_index;
  BasicParticleView<Schema, DeviceSpace<Backend>, true> source;
  std::uint32_t source_index;

  template <std::size_t I>
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void
  operator()(meta::index_constant<I>) const noexcept {
    destination.template get<I>()[destination_index] =
        source.template get<I>()[source_index];
  }
};

template <ParticleSchemaType Schema, device::DeviceBackend Backend>
LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void copy_device_particle_record(
    const BasicParticleView<Schema, DeviceSpace<Backend>, false> destination,
    const std::uint32_t destination_index,
    const BasicParticleView<Schema, DeviceSpace<Backend>, true> source,
    const std::uint32_t source_index) noexcept {
  for_each_attribute<Schema>(DeviceParticleRecordCopy<Schema, Backend>{
      destination, destination_index, source, source_index});
}

// Direct SoA record move consumed by the exact-count compaction kernel.
template <ParticleSchemaType Schema, device::DeviceBackend Backend>
class DeviceParticleRecordMover final {
public:
  using view_type = BasicParticleView<Schema, DeviceSpace<Backend>, false>;

  constexpr explicit DeviceParticleRecordMover(
      const view_type particles) noexcept
      : particles_(particles) {}

  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void
  operator()(const std::uint32_t destination,
             const std::uint32_t source) const noexcept {
    copy_device_particle_record<Schema, Backend>(particles_, destination,
                                                 particles_, source);
  }

private:
  view_type particles_{};
};

template <ParticleSchemaType Schema, device::DeviceBackend Backend>
[[nodiscard]] constexpr DeviceParticleRecordMover<Schema, Backend>
make_device_particle_record_mover(
    const BasicParticleView<Schema, DeviceSpace<Backend>, false>
        particles) noexcept {
  return DeviceParticleRecordMover<Schema, Backend>{particles};
}

} // namespace detail

} // namespace lpfx

#endif
