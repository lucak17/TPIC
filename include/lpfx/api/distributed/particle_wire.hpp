#ifndef LPFX_API_DISTRIBUTED_PARTICLE_WIRE_HPP
#define LPFX_API_DISTRIBUTED_PARTICLE_WIRE_HPP

#include <lpfx/api/distributed/key.hpp>
#include <lpfx/data/field/wire_traits.hpp>
#include <lpfx/distributed/particle/wire/contract.hpp>

namespace lpfx::distributed {
// Native-width scalar transport; a policy may override wire_scalar/codec per
// attribute without duplicating schema order or Communicate/RebuildLocally.
template <class Codec> struct particle_scalars final {
  template <class Attribute> using wire_scalar = ParticleWireScalar<
      typename Attribute::value_type,
      field::wire_traits<typename Attribute::value_type>::identity>;
  template <class> using codec = Codec;
};
namespace detail {
template <class Attribute, class Policy, class Treatment = typename Attribute::wire_policy>
struct InferredParticleAttribute {
  using type = RebuiltParticleWireAttribute<Attribute,
      operation_key<Attribute::name_type::name>::identity,
      field::wire_traits<typename Attribute::value_type>::identity>;
};
template <class Attribute, class Policy>
struct InferredParticleAttribute<Attribute, Policy, Communicate> {
  using type = CommunicatedParticleWireAttribute<Attribute,
      operation_key<Attribute::name_type::name>::identity,
      field::wire_traits<typename Attribute::value_type>::identity,
      typename Policy::template wire_scalar<Attribute>, typename Policy::template codec<Attribute>>;
};
template <ParticleWireLayout Layout, class Schema, ParticleWireEndianness Endian, class Hop,
          class Routing, class Policy, class Attributes = typename Schema::attributes_type>
struct InferredParticleContract;
template <ParticleWireLayout Layout, class Schema, ParticleWireEndianness Endian, class Hop,
          class Routing, class Policy, class... Attributes>
struct InferredParticleContract<Layout, Schema, Endian, Hop, Routing, Policy,
                                std::tuple<Attributes...>> {
  using type = BasicParticleWireContract<Layout, Schema, Endian, Hop, Routing,
      typename InferredParticleAttribute<Attributes, Policy>::type...>;
};
} // namespace detail
// Layout: packed for host owners (MpiHost); aligned when device codecs produce
// and consume the images (MpiStaged, MpiDirect, NCCL/RCCL).
template <class Schema, ParticleWireEndianness Endian, class Hop, class Routing, class Scalars,
          ParticleWireLayout Layout = ParticleWireLayout::packed>
using particle_wire = typename detail::InferredParticleContract<Layout, Schema, Endian, Hop,
    Routing, Scalars>::type;
} // namespace lpfx::distributed
#endif
