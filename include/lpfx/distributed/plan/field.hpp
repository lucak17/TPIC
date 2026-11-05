#ifndef LPFX_DISTRIBUTED_PLAN_FIELD_HPP
#define LPFX_DISTRIBUTED_PLAN_FIELD_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/field/pattern.hpp>
#include <lpfx/distributed/field/registration.hpp>
#include <lpfx/distributed/registration/record.hpp>
#include <lpfx/domain/cartesian/partition_neighborhood.hpp>

namespace lpfx::distributed {

template <std::uint64_t ChannelIdentity,
          detail::FieldOperationSpecType Operation>
  requires(ChannelIdentity != 0)
struct FieldPlan final {
  using descriptor_type = Operation;
  using spec_type = Operation;

  static constexpr std::uint64_t channel_identity = ChannelIdentity;
};

template <class Definition, class IndexPolicy, class Rank>
class PreparedFieldPlan;

template <std::uint64_t ChannelIdentity,
          detail::FieldOperationSpecType Operation,
          lpfx::detail::CartesianIndexPolicy IndexPolicy,
          lpfx::detail::CartesianRank Rank>
class PreparedFieldPlan<FieldPlan<ChannelIdentity, Operation>, IndexPolicy,
                        Rank>
    final {
public:
  using definition_type = FieldPlan<ChannelIdentity, Operation>;
  using descriptor_type = Operation;
  using spec_type = Operation;
  using index_policy = IndexPolicy;
  using rank_type = Rank;
  using pattern_type = PreparedFieldPattern3D<Operation, IndexPolicy, Rank>;
  using geometry_type = typename pattern_type::geometry_type;
  using channel_type =
      typename lpfx::detail::FieldRegistrationTraits<Operation>::channel_type;

  static constexpr std::uint64_t channel_identity = ChannelIdentity;

  constexpr PreparedFieldPlan() noexcept = default;

  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return pattern_.is_valid();
  }

  [[nodiscard]] constexpr const pattern_type &pattern() const noexcept {
    return pattern_;
  }

  [[nodiscard]] constexpr const channel_type &
  registration_channel() const noexcept {
    return channel_;
  }

private:
  constexpr PreparedFieldPlan(const pattern_type &pattern,
                              const channel_type &channel) noexcept
      : pattern_(pattern), channel_(channel) {}

  pattern_type pattern_{};
  channel_type channel_{};

  friend struct lpfx::detail::PrivateConstructionAccess;
};

template <class Definition, lpfx::detail::CartesianIndexPolicy IndexPolicy,
          lpfx::detail::CartesianRank Rank>
[[nodiscard]] constexpr ErrorCode try_prepare_field_plan(
    const Definition,
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const typename PreparedFieldPlan<Definition, IndexPolicy,
                                     Rank>::geometry_type &local_geometry,
    const std::array<typename PreparedFieldPlan<Definition, IndexPolicy,
                                                Rank>::geometry_type,
                     26> &peer_geometries,
    PreparedFieldPlan<Definition, IndexPolicy, Rank> &output) noexcept {
  using Prepared = PreparedFieldPlan<Definition, IndexPolicy, Rank>;
  typename Prepared::pattern_type pattern{};
  LPFX_TRY(lpfx::try_prepare_field_pattern(topology, local_geometry,
                                           peer_geometries, pattern));
  typename Prepared::channel_type channel{};
  LPFX_TRY(try_make_channel_id(Prepared::channel_identity, channel.channel_id));
  LPFX_TRY(lpfx::detail::try_prepare_field_registration_channel(pattern, channel));
  output = lpfx::detail::PrivateConstructionAccess::create<Prepared>(pattern,
                                                                     channel);
  return ErrorCode::success;
}

// Canonical-partition path: builder(partition, geometry) -> ErrorCode yields
// the geometry of the local rank and of every transported neighbor.
template <class Definition, lpfx::detail::CartesianIndexPolicy IndexPolicy,
          lpfx::detail::CartesianRank Rank, class GeometryBuilder>
  requires requires(
      GeometryBuilder &builder,
      const CartesianPartition3D<IndexPolicy, Rank> &partition,
      typename PreparedFieldPlan<Definition, IndexPolicy, Rank>::geometry_type
          &geometry) {
    { builder(partition, geometry) } noexcept -> std::same_as<ErrorCode>;
  }
[[nodiscard]] constexpr ErrorCode try_prepare_field_plan(
    const Definition definition,
    const PreparedCartesianPartitionNeighborhood3D<IndexPolicy, Rank>
        &neighborhood,
    GeometryBuilder &&geometry_builder,
    PreparedFieldPlan<Definition, IndexPolicy, Rank> &output) noexcept {
  using Prepared = PreparedFieldPlan<Definition, IndexPolicy, Rank>;
  using Geometry = typename Prepared::geometry_type;
  if (!neighborhood.is_valid()) {
    return ErrorCode::invalid_argument;
  }
  Geometry local_geometry{};
  LPFX_TRY(geometry_builder(neighborhood.local_partition(), local_geometry));
  std::array<Geometry, 26> peer_geometries{};
  for (std::size_t index = 0; index != peer_geometries.size(); ++index) {
    if (!cartesian_neighbor_mask_contains(Prepared::spec_type::transport_mask,
                                          index) ||
        !neighborhood.has_neighbor_partition_by_index(index)) {
      continue;
    }
    if (neighborhood.topology().neighbor_by_index(index).is_self_peer()) {
      peer_geometries[index] = local_geometry;
      continue;
    }
    LPFX_TRY(geometry_builder(neighborhood.neighbor_partition_by_index(index),
                              peer_geometries[index]));
  }
  return try_prepare_field_plan(definition, neighborhood.topology(),
                                local_geometry, peer_geometries, output);
}

} // namespace lpfx::distributed

#endif
