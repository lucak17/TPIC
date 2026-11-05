#ifndef LPFX_DISTRIBUTED_BINDING_PARTICLE_HPP
#define LPFX_DISTRIBUTED_BINDING_PARTICLE_HPP

#include <concepts>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include <lpfx/data/particle/device/classification.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/data/particle/host/storage.hpp>
#include <lpfx/distributed/binding/bound_operation.hpp>
#include <lpfx/distributed/common/executor.hpp>
#include <lpfx/distributed/particle/wire/codec.hpp>
#include <lpfx/distributed/plan/particle.hpp>
#include <lpfx/distributed/plan/particle_destination.hpp>
#include <lpfx/distributed/plan/particle_semantics.hpp>

namespace lpfx::distributed {

namespace detail {

template <class ReceivedOwner, class Position, class Rank>
concept ParticleReceivedOwnerCallback =
    lpfx::detail::CartesianRank<Rank> &&
    requires(const ReceivedOwner &received_owner, const std::uint64_t round,
             const std::uint32_t lane, const Position &position,
             ParticleDestination3D<Rank> &destination) {
      {
        received_owner(round, lane, position, destination)
      } noexcept -> std::same_as<bool>;
    };

template <class Space, class Semantics, class View, class Position, class Rank>
inline constexpr bool particle_semantics_match_space =
    ParticleInitialCallbackFor<typename Semantics::initial_owner_type, View,
                               Rank> &&
    ParticleReceivedOwnerCallback<typename Semantics::received_owner_type,
                                  Position, Rank>;

// Device kernels see the trivially copyable projection device_view().
template <device::DeviceBackend Backend, class Semantics, class View,
          class Position, class Rank>
inline constexpr bool particle_semantics_match_space<
    DeviceSpace<Backend>, Semantics, View, Position, Rank> =
    requires(const Semantics &semantics) {
      { semantics.device_view() } noexcept;
      requires std::is_trivially_copyable_v<decltype(semantics.device_view())>;
      requires ParticleReceivedOwnerCallback<
          typename decltype(semantics.device_view())::received_owner_type,
          Position, Rank>;
      requires(
          particle_classification_pipeline_v<Semantics> ==
              ParticleClassificationPipeline::user_flags ||
          ParticleInitialDestinationCallback<
              typename decltype(semantics.device_view())::initial_owner_type,
              View, Rank>);
    };

template <class Plan, class Schema, class Space, class Executor,
          class Semantics>
concept ParticleBindingParts =
    ParticleSemanticsType<Semantics> &&
    std::same_as<Schema, typename Plan::schema_type> &&
    CommunicationExecutorForSpace<Space, Executor> &&
    lpfx::detail::ParticleWireCodecContext<
        typename Semantics::codec_context_type, typename Plan::contract_type> &&
    particle_semantics_match_space<
        Space, Semantics, typename Particles<Schema, Space>::const_view_type,
        typename Plan::canonical_position_type, typename Plan::rank_type>;

// The generic bind() admits the position pipeline, and user flags on the
// host; device user flags bind through their classification slots.
template <class Definition, lpfx::detail::CartesianRank Rank,
          lpfx::detail::ParticleSchemaType Schema, class Space, class Executor,
          class Semantics>
struct BindingFamily<PreparedParticlePlan<Definition, Rank>,
                     Particles<Schema, Space>, Executor, Semantics> {
  static constexpr bool compatible =
      ParticleBindingParts<PreparedParticlePlan<Definition, Rank>, Schema,
                           Space, Executor, Semantics>;
  static constexpr bool bindable =
      particle_classification_pipeline_v<Semantics> ==
          ParticleClassificationPipeline::positions ||
      std::same_as<Space, HostSpace>;

  [[nodiscard]] static constexpr bool
  valid(const Particles<Schema, Space> &owner) noexcept {
    if constexpr (std::same_as<Space, HostSpace>)
      return owner.valid() && owner.capacity() <= UINT32_MAX;
    else
      return owner.valid();
  }
};

// Semantics of a user-flag binding: the user's callbacks and codec context,
// the classification slots its producer registers into, and the registration
// context the migration checks. Decoder is the flag decoder of
// prepare_classification (void when the caller produces the flags itself).
template <ParticleSemanticsType Semantics, class Decoder,
          device::DeviceBackend Backend, bool ExternalResident>
class BoundParticleSemantics final {
public:
  using initial_owner_type = typename Semantics::initial_owner_type;
  using received_owner_type = typename Semantics::received_owner_type;
  using codec_context_type = typename Semantics::codec_context_type;
  using flag_decoder_type = Decoder;
  static constexpr auto classification_pipeline =
      ParticleClassificationPipeline::user_flags;
  static constexpr bool external_resident_access = ExternalResident;

  BoundParticleSemantics(Semantics semantics,
                         DeviceParticleClassificationSlots<Backend> &slots,
                         const void *const context) noexcept
      : semantics_(std::move(semantics)), slots_(&slots), context_(context) {}

  [[nodiscard]] const initial_owner_type &initial_owner() const noexcept {
    return semantics_.initial_owner();
  }
  [[nodiscard]] const received_owner_type &received_owner() const noexcept {
    return semantics_.received_owner();
  }
  [[nodiscard]] const codec_context_type &codec_context() const noexcept {
    return semantics_.codec_context();
  }
  [[nodiscard]] auto device_view() const noexcept {
    return semantics_.device_view();
  }
  [[nodiscard]] DeviceParticleClassificationSlots<Backend> &
  classification_slots() const noexcept {
    return *slots_;
  }
  [[nodiscard]] const void *classification_context() const noexcept {
    return context_;
  }
  void bind_classification_context(const void *const context) noexcept {
    context_ = context;
  }

private:
  [[no_unique_address]] Semantics semantics_;
  DeviceParticleClassificationSlots<Backend> *slots_;
  const void *context_;
};

} // namespace detail

template <class Transport, class Plan, class Owner, class Executor,
          class Semantics>
using ParticleBinding =
    BoundOperation<Transport, Plan, Owner, Executor, Semantics>;

// Device user flags: the producer registers decisions into slots, and the
// migration never instantiates a source-position classifier.
template <class Transport, class Plan, class Owner, class Executor,
          class Semantics>
  requires(detail::particle_classification_pipeline_v<Semantics> ==
           ParticleClassificationPipeline::user_flags) &&
          (!std::same_as<typename Owner::space_type, HostSpace>) &&
          std::same_as<typename Semantics::initial_owner_type,
                       detail::NoInitialParticleClassifier> &&
          detail::CompatibleBinding<
              Plan, Owner, Executor,
              detail::BoundParticleSemantics<
                  Semantics, void, typename Executor::backend_type, false>>
[[nodiscard]] auto
bind(const Plan &plan, Owner &owner, Executor executor, Semantics semantics,
     DeviceParticleClassificationSlots<typename Executor::backend_type>
         &slots) noexcept {
  using Bound = detail::BoundParticleSemantics<
      Semantics, void, typename Executor::backend_type, false>;
  return ParticleBinding<Transport, Plan, Owner, Executor, Bound>{
      plan, owner, std::move(executor),
      Bound{std::move(semantics), slots, std::addressof(plan)}};
}

template <class Transport, class Plan, class Owner, class Executor,
          class Semantics>
void bind(const Plan &&, Owner &, Executor, Semantics,
          DeviceParticleClassificationSlots<typename Executor::backend_type>
              &) = delete;

} // namespace lpfx::distributed

#endif
