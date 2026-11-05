#ifndef LPFX_DISTRIBUTED_PLAN_PARTICLE_DESTINATION_HPP
#define LPFX_DISTRIBUTED_PLAN_PARTICLE_DESTINATION_HPP

#include <concepts>
#include <cstdint>

#include <lpfx/core/attributes.hpp>
#include <lpfx/distributed/particle/limits.hpp>
#include <lpfx/domain/cartesian/geometry.hpp>
#include <lpfx/domain/cartesian/index_policy.hpp>

namespace lpfx {

// The one route decision of a particle record: a prepared peer group in
// [0, peer_count), or one of these two codes.
inline constexpr std::uint8_t particle_bucket_resident = UINT8_MAX;
inline constexpr std::uint8_t particle_bucket_discard = UINT8_MAX - 1;

} // namespace lpfx

namespace lpfx::distributed {

// A decoded application flag can name its prepared peer group directly (not
// an MPI rank or an application direction).
class ParticleClassificationAction final {
public:
  LPFX_HOST_DEVICE constexpr ParticleClassificationAction() noexcept = default;
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr ParticleClassificationAction
  local() noexcept {
    return ParticleClassificationAction{particle_bucket_resident};
  }
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr ParticleClassificationAction
  discard() noexcept {
    return ParticleClassificationAction{particle_bucket_discard};
  }
  [[nodiscard]] LPFX_HOST_DEVICE static constexpr ParticleClassificationAction
  transfer(const std::uint32_t peer) noexcept {
    return peer < particle_migration_maximum_peer_count
               ? ParticleClassificationAction{static_cast<std::uint8_t>(peer)}
               : ParticleClassificationAction{};
  }
  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool valid() const noexcept {
    return bucket_ != invalid_bucket;
  }
  // Hard precondition: valid().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr std::uint8_t bucket() const noexcept {
    return bucket_;
  }

private:
  static constexpr std::uint8_t invalid_bucket = UINT8_MAX - 2;

  LPFX_HOST_DEVICE explicit constexpr ParticleClassificationAction(
      const std::uint8_t bucket) noexcept
      : bucket_(bucket) {}

  std::uint8_t bucket_ = invalid_bucket;
};

// A classified particle has a final Cartesian owner or is explicitly
// discarded. Peer groups and transit are derived from the prepared routing.
template <lpfx::detail::CartesianRank Rank> class ParticleDestination3D final {
public:
  LPFX_HOST_DEVICE constexpr ParticleDestination3D() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE static constexpr ParticleDestination3D
  owner(const CartesianIndex3<Rank> coordinate) noexcept {
    return ParticleDestination3D{Kind::owner, coordinate};
  }

  [[nodiscard]] LPFX_HOST_DEVICE static constexpr ParticleDestination3D
  discard() noexcept {
    return ParticleDestination3D{Kind::discard, {}};
  }

  // A default value is invalid, so a callback that returns true without
  // selecting owner(...) or discard() is detected.
  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool valid() const noexcept {
    return kind_ != Kind::unset;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool discarded() const noexcept {
    return kind_ == Kind::discard;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool has_owner() const noexcept {
    return kind_ == Kind::owner;
  }

  // Hard precondition: has_owner().
  [[nodiscard]] LPFX_HOST_DEVICE constexpr CartesianIndex3<Rank>
  coordinate() const noexcept {
    return coordinate_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE friend constexpr bool
  operator==(const ParticleDestination3D &,
             const ParticleDestination3D &) noexcept = default;

private:
  enum class Kind : std::uint8_t { unset = 0, owner = 1, discard = 2 };

  LPFX_HOST_DEVICE constexpr ParticleDestination3D(
      const Kind kind, const CartesianIndex3<Rank> coordinate) noexcept
      : coordinate_(coordinate), kind_(kind) {}

  CartesianIndex3<Rank> coordinate_{};
  Kind kind_ = Kind::unset;
};

namespace detail {

template <class Callback, class View, class Rank, class Lane = std::uint32_t>
concept ParticleInitialDestinationCallback =
    lpfx::detail::CartesianRank<Rank> &&
    requires(const Callback &callback, const View source, const Lane lane,
             ParticleDestination3D<Rank> &destination) {
      { callback(source, lane, destination) } noexcept -> std::same_as<bool>;
    };

template <class Callback, class View, class Lane = std::uint32_t>
concept ParticleInitialActionCallback =
    requires(const Callback &callback, const View source, const Lane lane,
             ParticleClassificationAction &action) {
      { callback(source, lane, action) } noexcept -> std::same_as<bool>;
    };

template <class Callback, class View, class Rank, class Lane = std::uint32_t>
concept ParticleInitialCallbackFor =
    ParticleInitialDestinationCallback<Callback, View, Rank, Lane> ||
    ParticleInitialActionCallback<Callback, View, Lane>;

template <class Decoder>
concept ParticleActionFlagDecoder = requires(const std::uint64_t flag) {
  { Decoder::decode(flag) } noexcept -> std::same_as<ParticleClassificationAction>;
};

template <class Decoder, class Rank>
concept ParticleOwnerFlagDecoder =
    requires(const std::uint64_t flag, ParticleDestination3D<Rank> &destination) {
      { Decoder::decode(flag, destination) } noexcept -> std::same_as<bool>;
    };

} // namespace detail

} // namespace lpfx::distributed

#endif
