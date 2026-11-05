#ifndef LPFX_DATA_PARTICLE_HOST_VIEW_HPP
#define LPFX_DATA_PARTICLE_HOST_VIEW_HPP

#include <array>
#include <cstddef>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/memory_intrinsics.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/particle/layout.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx {

namespace detail {

template <class T>
concept ParticleMapIndex =
    LifetimeViewElement<T> && checked_integer<std::remove_const_t<T>>;

} // namespace detail

// One value per schema attribute, in schema order.
template <detail::ParticleSchemaType Schema> class ParticleValues final {
public:
  constexpr ParticleValues() noexcept = default;

  template <std::size_t I>
  [[nodiscard]] constexpr const auto &get() const noexcept {
    return std::get<I>(values_);
  }

private:
  using values_type = typename Schema::values_type;

  constexpr explicit ParticleValues(const values_type &values) noexcept
      : values_(values) {}

  values_type values_{};

  friend struct detail::PrivateConstructionAccess;
};

template <detail::ParticleAttribute... Attributes, class... Values>
  requires(sizeof...(Attributes) == sizeof...(Values)) &&
          (std::same_as<std::remove_cvref_t<Values>,
                        typename Attributes::value_type> &&
           ...)
[[nodiscard]] constexpr ParticleValues<ParticleSchema<Attributes...>>
make_particle_values(Values &&...values) noexcept {
  return detail::PrivateConstructionAccess::create<
      ParticleValues<ParticleSchema<Attributes...>>>(
      std::tuple<typename Attributes::value_type...>{
          std::forward<Values>(values)...});
}

namespace detail {

template <ParticleSchemaType Schema> struct ParticleValuesFactory;

template <ParticleAttribute... Attributes>
struct ParticleValuesFactory<ParticleSchema<Attributes...>> final {
  static constexpr bool default_supported =
      (std::is_nothrow_default_constructible_v<
           typename Attributes::value_type> &&
       ...);

  template <class... Values>
    requires(sizeof...(Values) == sizeof...(Attributes)) &&
            (std::is_nothrow_constructible_v<typename Attributes::value_type,
                                             Values &&> &&
             ...)
  [[nodiscard]] static constexpr auto make(Values &&...values) noexcept {
    return make_particle_values<Attributes...>(
        typename Attributes::value_type(std::forward<Values>(values))...);
  }
};

} // namespace detail

template <detail::ParticleSchemaType Schema, bool Const>
class BasicParticleView<Schema, HostSpace, Const> final {
public:
  using schema_type = Schema;
  using space_type = HostSpace;
  using count_type = std::size_t;
  static constexpr std::size_t attribute_count = Schema::attribute_count;
  static constexpr std::size_t data_alignment =
      detail::particle_allocation_alignment<HostSpace, Schema>;

  constexpr BasicParticleView() noexcept = default;

  [[nodiscard]] constexpr count_type size() const noexcept { return count_; }

  [[nodiscard]] constexpr bool empty() const noexcept { return count_ == 0; }

  template <std::size_t I> [[nodiscard]] constexpr auto get() const noexcept {
    using Value = typename Schema::template value_type<I>;
    using Element = std::conditional_t<Const, const Value, Value>;
    std::byte *const raw_base = bases_[I];
    Element *const base =
        raw_base == nullptr
            ? nullptr
            : std::launder(reinterpret_cast<Element *>(raw_base));
    return detail::make_buffer_view<Element>(base, count_);
  }

  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] constexpr auto get() const noexcept {
    return get<Schema::template index_of<Name>>();
  }

  // Returns the same SoA base as get(), carrying the schema-wide allocation
  // alignment into compiler code generation. Hard precondition: this view was
  // produced by an LPFx particle owner, or every supplied attribute base obeys
  // data_alignment. No runtime check or branch is generated.
  template <std::size_t I>
  [[nodiscard]] LPFX_ALWAYS_INLINE auto *aligned_data() const noexcept {
    return detail::assume_aligned_pointer<data_alignment>(get<I>().data());
  }

  template <class Name>
    requires Schema::template contains<Name>
  [[nodiscard]] LPFX_ALWAYS_INLINE auto *aligned_data() const noexcept {
    return aligned_data<Schema::template index_of<Name>>();
  }

private:
  // Const controls only the typed API; the stored bases are identical.
  constexpr BasicParticleView(
      const std::array<std::byte *, attribute_count> &bases,
      const count_type count) noexcept
      : bases_(bases), count_(count) {}

  std::array<std::byte *, attribute_count> bases_{};
  count_type count_{};

  friend struct detail::PrivateConstructionAccess;
};

} // namespace lpfx

#endif
