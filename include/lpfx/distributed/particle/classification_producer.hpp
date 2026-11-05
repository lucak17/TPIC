#ifndef LPFX_DISTRIBUTED_PARTICLE_CLASSIFICATION_PRODUCER_HPP
#define LPFX_DISTRIBUTED_PARTICLE_CLASSIFICATION_PRODUCER_HPP

#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/data/particle/device/classification.hpp>
#include <lpfx/distributed/particle/migration/routing.hpp>
#include <lpfx/distributed/plan/particle_destination.hpp>

namespace lpfx::distributed {
// Static action decoders need only the registration view. Owner-coordinate
// decoders additionally retain prepared routing; neither form stores a Decoder.
namespace detail {
struct NoParticleFlagRouting final {};
} // namespace detail

template <class Routing, device::DeviceBackend Backend, class Decoder>
  requires(
      detail::ParticleActionFlagDecoder<Decoder> ||
      detail::ParticleOwnerFlagDecoder<Decoder, typename Routing::rank_type>)
class ParticleClassificationProducer final {
public:
  using rank_type = typename Routing::rank_type;
  using slots_view_type = DeviceParticleRegisteredClassificationView<Backend>;

  explicit constexpr ParticleClassificationProducer(
      slots_view_type slots) noexcept
    requires detail::ParticleActionFlagDecoder<Decoder>
      : slots_(slots) {}
  constexpr ParticleClassificationProducer(Routing routing,
                                           slots_view_type slots) noexcept
    requires(!detail::ParticleActionFlagDecoder<Decoder>)
      : routing_(routing), slots_(slots) {}
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint32_t size() const noexcept {
    return slots_.size();
  }
  [[nodiscard]] LPFX_DEVICE bool
  register_flag(const std::uint32_t index,
                const std::uint64_t flag) const noexcept {
    if constexpr (detail::ParticleActionFlagDecoder<Decoder>) {
      const auto action = Decoder::decode(flag);
      return action.valid() ? slots_.register_particle(index, action.bucket())
                            : slots_.invalidate();
    } else {
      ParticleDestination3D<rank_type> destination;
      std::uint8_t bucket{};
      if (!Decoder::decode(flag, destination) ||
          route_bucket(routing_, destination, bucket) != ErrorCode::success)
        return slots_.invalidate();
      return slots_.register_particle(index, bucket);
    }
  }

private:
  [[no_unique_address]] std::conditional_t<
      detail::ParticleActionFlagDecoder<Decoder>, detail::NoParticleFlagRouting,
      Routing> routing_;
  slots_view_type slots_;
};


// An admitted producer batch borrows persistent slots and owns no GPU work.
// Destruction never waits or publishes. The caller must publish or cancel after
// explicitly handling any application submissions on its chosen executor.
template <class Producer, class Backend> class ParticleClassificationBatch final {
public:
  using Slots = DeviceParticleClassificationSlots<Backend>;
  using Executor = device::Executor<Backend>;
  ParticleClassificationBatch(Producer producer, Slots &slots, Executor executor) noexcept
      : producer_(producer), slots_(&slots), executor_(executor) {}
  ParticleClassificationBatch(ParticleClassificationBatch &&other) noexcept
      : producer_(other.producer_), slots_(std::exchange(other.slots_, nullptr)),
        executor_(other.executor_) {}
  ParticleClassificationBatch &operator=(ParticleClassificationBatch &&) = delete;
  [[nodiscard]] Producer device_view() const noexcept { return producer_; }
  [[nodiscard]] ErrorCode publish() noexcept {
    if (!slots_) return ErrorCode::invalid_state;
    const auto error = slots_->publish(executor_);
    if (error == ErrorCode::success) slots_ = nullptr;
    return error;
  }
  [[nodiscard]] ErrorCode cancel() noexcept {
    if (!slots_) return ErrorCode::invalid_state;
    const auto error = slots_->cancel(executor_);
    if (error == ErrorCode::success) slots_ = nullptr;
    return error;
  }
private:
  Producer producer_;
  Slots *slots_;
  Executor executor_;
};
} // namespace lpfx::distributed
#endif
