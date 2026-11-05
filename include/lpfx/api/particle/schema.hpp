#ifndef LPFX_API_PARTICLE_SCHEMA_HPP
#define LPFX_API_PARTICLE_SCHEMA_HPP

#include <cstddef>
#include <utility>

#include <lpfx/core/fixed_string.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/host/view.hpp>

namespace lpfx::particle {

using lpfx::field::fixed_string;

// Names exist only in types. Named attributes use the native particle schema,
// storage, wire contracts, views, and algorithms without any adaptation.
template <fixed_string Name, LifetimeStartEligible T,
          lpfx::detail::ParticleWirePolicy WirePolicy = Communicate,
          lpfx::detail::ParticleAlignmentPolicy AlignmentPolicy =
              DefaultAttributeAlignment>
  requires(lpfx::field::detail::valid_name<Name>())
using attribute = Attribute<lpfx::field::detail::name_tag_t<Name>, T, WirePolicy,
                            AlignmentPolicy>;

template <lpfx::detail::ParticleAttribute... Attributes>
using schema = ParticleSchema<Attributes...>;

struct options final {
  // Required capacity before configured headroom and slot rounding are applied.
  std::size_t capacity{};
  std::size_t size{};
};

// Whether resize_capacity keeps the published records (preserve) or publishes
// size zero (discard).
template <ParticleResizeMode Mode> struct resize_policy final {};
using preserve_t = resize_policy<ParticleResizeMode::preserve>;
using discard_t = resize_policy<ParticleResizeMode::discard>;
inline constexpr preserve_t preserve{};
inline constexpr discard_t discard{};

template <lpfx::detail::ParticleSchemaType Schema>
  requires lpfx::detail::ParticleValuesFactory<Schema>::default_supported
[[nodiscard]] constexpr ParticleValues<Schema> values() noexcept {
  return ParticleValues<Schema>{};
}

template <lpfx::detail::ParticleSchemaType Schema, class... Values>
  requires requires(Values &&...initial) {
    lpfx::detail::ParticleValuesFactory<Schema>::make(
        std::forward<Values>(initial)...);
  }
[[nodiscard]] constexpr auto values(Values &&...initial) noexcept {
  return lpfx::detail::ParticleValuesFactory<Schema>::make(
      std::forward<Values>(initial)...);
}

} // namespace lpfx::particle

#endif
