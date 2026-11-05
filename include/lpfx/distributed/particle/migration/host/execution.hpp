#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_HOST_EXECUTION_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_HOST_EXECUTION_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include <lpfx/algorithm/particle/host/compaction.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/data/particle/host/storage.hpp>
#include <lpfx/distributed/particle/migration/capacity.hpp>
#include <lpfx/distributed/particle/migration/routing.hpp>
#include <lpfx/distributed/particle/migration/stage.hpp>
#include <lpfx/distributed/particle/wire/codec.hpp>
#include <lpfx/distributed/particle/wire/storage.hpp>
#include <lpfx/memory/host/allocation.hpp>
#include <lpfx/memory/lifetime.hpp>

namespace lpfx::distributed::detail {

// Stores the source index of every sent record at
// prefixes.particle_offset(bucket) onward, in ascending source order.
// Resident and discarded records are skipped.
template <class Prefixes>
LPFX_ALWAYS_INLINE void
bucket_scatter(const std::uint8_t *const buckets, const std::size_t count,
               const Prefixes &prefixes, std::uint32_t *const map) noexcept {
  std::array<std::size_t, particle_migration_maximum_peer_count> cursors;
  for (std::uint8_t peer = 0; peer != prefixes.peer_count(); ++peer)
    cursors[peer] = static_cast<std::size_t>(prefixes.particle_offset(peer));
  for (std::size_t index = 0; index != count; ++index) {
    const std::uint8_t bucket = buckets[index];
    if (bucket != particle_bucket_resident &&
        bucket != particle_bucket_discard)
      map[cursors[bucket]++] = static_cast<std::uint32_t>(index);
  }
}

struct ResidentParticleKeep final {
  const std::uint8_t *buckets;

  [[nodiscard]] LPFX_ALWAYS_INLINE bool
  operator()(const std::size_t index) const noexcept {
    return buckets[index] == particle_bucket_resident;
  }
};

// Decoded attribute values go straight into an unpublished owner slot.
template <lpfx::detail::ParticleSchemaType Schema> struct HostParticleSink final {
  std::array<std::byte *, Schema::attribute_count> bases;

  LPFX_EXEC_CHECK_DISABLE
  template <std::size_t Attribute, class Value>
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void
  store(const std::size_t index, const Value &value) const noexcept {
    using Stored = typename Schema::template value_type<Attribute>;
    lpfx::detail::host_particle_store<Stored>(
        reinterpret_cast<Stored *>(bases[Attribute]), index, value);
  }
};

// Host epoch execution. Classification, packing and decoding run
// synchronously in the calling thread; growth replaces the storage in place.
template <lpfx::detail::ParticleWireContractType Contract,
          lpfx::detail::CartesianRank Rank, class Owner, class Executor,
          class Semantics>
  requires std::same_as<typename Owner::space_type, HostSpace>
class HostParticleExecution final {
public:
  using contract_type = Contract;
  using schema_type = typename Contract::schema_type;
  using owner_type = Owner;
  using executor_type = Executor;
  using semantics_type = Semantics;
  using routing_type =
      PreparedParticleRoutingPattern<Rank, typename Contract::hop_policy>;
  using capacity_type = PreparedParticleMigrationCapacity<Contract>;
  using protocol_type = ParticleMigrationProtocol<
      routing_type, const typename Semantics::initial_owner_type &,
      const typename Semantics::received_owner_type &>;
  using wire_storage_type = HostParticleMigrationWireStorage<Contract>;
  // Buckets and the source map of the largest classification: the owner or
  // one incoming batch.
  using scratch_type =
      lpfx::detail::HostArrayPack<64, std::uint8_t, std::uint32_t>;
  struct storage_type final {
    wire_storage_type wire;
    scratch_type scratch;
  };
  using outgoing_prefixes_type = PreparedParticleOutgoingPrefixes<Contract>;
  using incoming_prefixes_type = PreparedParticleIncomingPrefixes<Contract>;

  HostParticleExecution() noexcept = default;
  LPFX_NONMOVABLE(HostParticleExecution);

  [[nodiscard]] static ErrorCode
  try_prepare_storage(HostMemoryBudget &budget, const capacity_type &capacity,
                      const std::size_t owner_capacity,
                      const device::DeviceProperties &,
                      storage_type &output) noexcept {
    std::uint32_t owner_extent{};
    std::uint32_t incoming_extent{};
    if (!lpfx::detail::checked_integral_cast(owner_capacity, owner_extent) ||
        !lpfx::detail::checked_integral_cast(capacity.maximum_incoming(),
                                             incoming_extent)) {
      return ErrorCode::capacity_exceeded;
    }
    const std::size_t extent =
        owner_extent > incoming_extent ? owner_extent : incoming_extent;
    PreparedParticleMigrationWireStorage<Contract> prepared{};
    LPFX_TRY(try_prepare_particle_migration_wire_storage(capacity, prepared));
    auto scratch = scratch_type::create(&budget, {extent, extent});
    LPFX_TRY(scratch);
    auto wire = make_host_particle_migration_wire_storage(budget, prepared);
    LPFX_TRY(wire);
    output.scratch = std::move(scratch).value();
    output.wire = std::move(wire).value();
    return ErrorCode::success;
  }

  template <class Pattern>
  void prepare(const Pattern &pattern, owner_type &owner,
               storage_type &&storage) noexcept {
    routing_ = &pattern.routing();
    capacity_ = &pattern.capacity();
    owner_ = &owner;
    storage_ = std::move(storage);
  }

  // Replaces quiescent storage between epochs.
  void adopt(storage_type &&storage) noexcept { storage_ = std::move(storage); }

  // Count-admission growth. The first round carries the classification into
  // the new scratch; later rounds carry the transit image in slot A.
  [[nodiscard]] ErrorCode grow(storage_type &&storage,
                               const bool first_round) noexcept {
    if (first_round) {
      if (source_count_ > storage.scratch.template size<0>())
        return ErrorCode::capacity_exceeded;
      if (source_count_ != 0)
        std::memcpy(storage.scratch.template data<0>(),
                    storage_.scratch.template data<0>(), source_count_);
      const auto outgoing =
          static_cast<std::size_t>(outgoing_.total_particles());
      if (outgoing != 0)
        std::memcpy(storage.scratch.template data<1>(),
                    storage_.scratch.template data<1>(),
                    outgoing * sizeof(std::uint32_t));
    } else {
      const auto bytes = std::as_const(storage_.wire).bytes(outgoing_);
      if (bytes.size() > storage.wire.slot_a_capacity().size())
        return ErrorCode::capacity_exceeded;
      if (bytes.size() != 0)
        std::memcpy(storage.wire.slot_a_capacity().data(), bytes.data(),
                    bytes.size());
    }
    storage_ = std::move(storage);
    return ErrorCode::success;
  }

  // Initial growth into an empty larger owner. The current owner is read
  // once: outgoing records are packed into the new wire image and residents
  // are copied into next. The caller commits next after this returns, so the
  // current owner stays untouched until every read completes.
  [[nodiscard]] ErrorCode grow_initial_owner(storage_type &&storage,
                                             owner_type &next,
                                             const semantics_type &semantics) noexcept {
    const std::size_t retained = resident_count_;
    if (next.size() != 0 || retained > next.capacity())
      return ErrorCode::invalid_state;
    const Status packed = pack(std::as_const(*owner_), storage.wire, semantics);
    if (!packed.is_success())
      return packed.code();
    const std::uint8_t *const buckets = storage_.scratch.template data<0>();
    std::uint32_t *const map = storage.scratch.template data<1>();
    const bool contiguous = retained == source_count_;
    if (retained != 0 && !contiguous) {
      std::size_t destination = 0;
      for (std::size_t lane = 0; lane != source_count_; ++lane)
        if (buckets[lane] == particle_bucket_resident)
          map[destination++] = static_cast<std::uint32_t>(lane);
      if (destination != retained)
        return ErrorCode::invalid_state;
    }
    const auto original = std::as_const(*owner_).view();
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ([&] {
        const auto *const values =
            reinterpret_cast<const std::byte *>(original.template get<I>().data());
        constexpr std::size_t width = schema_type::widths[I];
        auto *const target =
            lpfx::detail::HostParticlesAccess::template raw_attribute_base<I>(
                next);
        if (retained != 0) {
          if (contiguous)
            std::memcpy(target, values, retained * width);
          else
            for (std::size_t lane = 0; lane != retained; ++lane)
              std::memcpy(target + lane * width, values + map[lane] * width,
                          width);
        }
      }(), ...);
    }(std::make_index_sequence<schema_type::attribute_count>{});
    lpfx::detail::HostParticlesAccess::publish_initialized_size(next, retained);
    // Packing and compaction are complete: the all-resident extent lets the
    // compaction submission finish immediately.
    source_count_ = retained;
    storage_ = std::move(storage);
    return ErrorCode::success;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_initial_classification(
      const protocol_type &protocol, const executor_type &) noexcept {
    const auto source = std::as_const(*owner_).view();
    const std::size_t count = source.size();
    if (count > storage_.scratch.template size<0>())
      return Status::failure(ErrorCode::invalid_argument);
    classified_ = false;
    std::array<std::uint64_t, particle_migration_maximum_peer_count> counts;
    // Only prepared peers are observed; unused slots need no writes.
    const auto peers = routing_->peer_count();
    for (std::uint8_t peer = 0; peer != peers; ++peer)
      counts[peer] = 0;
    std::uint8_t *const buckets = storage_.scratch.template data<0>();
    // A private copy keeps the routing fields in registers: the byte stores
    // into buckets may otherwise alias the caller's protocol.
    const protocol_type local = protocol;
    std::size_t resident = 0;
    for (std::size_t index = 0; index != count; ++index) {
      std::uint8_t bucket{};
      const ErrorCode error = local.classify_initial(source, index, bucket);
      if (error != ErrorCode::success)
        return Status::failure(error);
      buckets[index] = bucket;
      if (bucket == particle_bucket_resident)
        ++resident;
      else if (bucket != particle_bucket_discard)
        ++counts[bucket];
    }
    outgoing_prefixes_type prefixes{};
    const ErrorCode admission = try_prepare_initial_particle_outgoing_counts(
        *capacity_, std::span<const std::uint64_t>{counts.data(), peers},
        prefixes);
    if (admission != ErrorCode::success)
      return Status::failure(admission);
    if (routing_->round_count() == 0 && prefixes.total_particles() != 0)
      return Status::failure(ErrorCode::invalid_argument);
    // Only sent records enter the source map.
    if (prefixes.total_particles() != 0)
      bucket_scatter(buckets, count, prefixes,
                     storage_.scratch.template data<1>());
    outgoing_ = prefixes;
    source_count_ = count;
    resident_count_ = resident;
    final_prefix_ = 0;
    classified_ = true;
    return Status::success();
  }

  void discard_initial_classification() noexcept {
    classified_ = false;
    outgoing_ = {};
    resident_count_ = 0;
    final_prefix_ = 0;
    next_final_prefix_ = 0;
  }

  // Each peer's SoA image is written directly into its final payload.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_initial_pack(
      const semantics_type &semantics, const executor_type &) noexcept {
    return pack(std::as_const(*owner_), storage_.wire, semantics);
  }

  // Packing has consumed the outgoing source map; it holds the exact
  // hole/filler lists, and the buckets are read directly.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status compact_residents() noexcept {
    if (!classified_ || owner_->size() != source_count_)
      return Status::failure(ErrorCode::invalid_state);
    const std::size_t removed = source_count_ - resident_count_;
    const std::size_t maximum_moves =
        resident_count_ < removed ? resident_count_ : removed;
    std::uint32_t *const holes = storage_.scratch.template data<1>();
    std::size_t moves{};
    if (lpfx::detail::compact_exact_in_place(
            owner_->view(),
            ResidentParticleKeep{storage_.scratch.template data<0>()},
            resident_count_, source_count_, holes, holes + maximum_moves,
            maximum_moves, moves) != ErrorCode::success)
      return Status::failure(ErrorCode::invalid_state);
    static_cast<void>(owner_->try_shrink(resident_count_));
    return Status::success();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_received_classification(
      const std::uint64_t round, const incoming_prefixes_type &incoming,
      const protocol_type &protocol, const executor_type &) noexcept {
    const std::uint64_t total = incoming.total_particles();
    if (total > storage_.scratch.template size<0>())
      return Status::failure(ErrorCode::capacity_exceeded);
    const auto incoming_bytes = std::as_const(storage_.wire).bytes(incoming);
    const ConstHostParticlePeerWireImageView<Contract> packets{
        incoming_bytes.data(), incoming};
    std::uint8_t *const buckets = storage_.scratch.template data<0>();
    std::array<std::uint64_t, particle_migration_maximum_peer_count>
        transit_counts;
    const auto peers = routing_->peer_count();
    for (std::uint8_t peer = 0; peer != peers; ++peer)
      transit_counts[peer] = 0;
    std::uint64_t final_count = 0;
    std::uint64_t transit_count = 0;
    // Private protocol copy, as in the initial classification.
    const protocol_type local = protocol;
    for (std::uint8_t peer = 0; peer != packets.peer_count(); ++peer) {
      const auto image = packets.packet(peer);
      const auto packet_offset = packets.particle_offset(peer);
      for (std::size_t local_lane = 0; local_lane != image.particle_count();
           ++local_lane) {
        const auto lane = packet_offset + local_lane;
        lpfx::detail::ParticleWireCanonicalPosition<Contract> position{};
        std::uint8_t bucket{};
        if (!lpfx::detail::load_particle_canonical_position(image, local_lane,
                                                            position) ||
            local.classify_received(round, static_cast<std::uint32_t>(lane),
                                    position,
                                    bucket) != ErrorCode::success)
          return Status::failure(ErrorCode::invalid_argument);
        buckets[lane] = bucket;
        if (bucket == particle_bucket_resident) {
          ++final_count;
        } else if (bucket != particle_bucket_discard) {
          ++transit_counts[bucket];
          ++transit_count;
        }
      }
    }

    // The program sealed every incoming peer extent before payload
    // publication; admit only the classified split and next-round transit.
    std::uint64_t next_final_prefix{};
    outgoing_prefixes_type next{};
    ErrorCode admission = lpfx::detail::try_admit_particle_received_split(
        *capacity_, total, final_count, transit_count,
        total - final_count - transit_count, final_prefix_, resident_count_,
        owner_->capacity(), next_final_prefix);
    if (admission == ErrorCode::success)
      admission = try_admit_transit_particle_outgoing(
          *capacity_,
          std::span<const std::uint64_t>{transit_counts.data(), peers},
          transit_count, next);
    if (admission != ErrorCode::success)
      return Status::failure(admission);
    // Finals are decoded packet-major from the buckets; only transit records
    // enter the source map.
    if (transit_count != 0)
      bucket_scatter(buckets, static_cast<std::size_t>(total), next,
                     storage_.scratch.template data<1>());
    next_final_prefix_ = next_final_prefix;
    outgoing_ = next;
    return Status::success();
  }

  // Every final record is decoded into its unpublished owner slot, then
  // transit records are copied from slot B into slot A.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_received_processing(
      const incoming_prefixes_type &incoming, const semantics_type &semantics,
      const executor_type &) noexcept {
    const std::uint32_t *const map = storage_.scratch.template data<1>();
    const auto incoming_bytes = std::as_const(storage_.wire).bytes(incoming);
    const ConstHostParticlePeerWireImageView<Contract> incoming_packets{
        incoming_bytes.data(), incoming};
    const HostParticleSink<schema_type> sink{sink_bases(
        std::make_index_sequence<schema_type::attribute_count>{})};
    const std::size_t first =
        resident_count_ + static_cast<std::size_t>(final_prefix_);
    // Finals fill owner slots in ascending received order (peer-major,
    // lane-minor), read straight from the classification buckets.
    const std::uint8_t *const buckets = storage_.scratch.template data<0>();
    std::size_t ordinal = 0;
    for (std::uint8_t peer = 0; peer != incoming_packets.peer_count(); ++peer) {
      const auto image = incoming_packets.packet(peer);
      const std::uint8_t *const packet_buckets =
          buckets + incoming_packets.particle_offset(peer);
      for (std::size_t lane = 0; lane != image.particle_count(); ++lane) {
        if (packet_buckets[lane] != particle_bucket_resident)
          continue;
        if (!lpfx::detail::decode_particle_record(image, lane,
                                                  semantics.codec_context(),
                                                  sink, first + ordinal++))
          return Status::failure(ErrorCode::invalid_argument);
      }
    }
    const HostParticlePeerWireImageView<Contract> outgoing_packets{
        storage_.wire.bytes(outgoing_).data(), outgoing_};
    for (std::uint8_t peer = 0; peer != outgoing_packets.peer_count(); ++peer) {
      const auto image = outgoing_packets.packet(peer);
      const auto packet_offset = outgoing_packets.particle_offset(peer);
      for (std::size_t lane = 0; lane != image.particle_count(); ++lane) {
        const auto source = incoming_packets.resolve(map[packet_offset + lane]);
        lpfx::detail::copy_particle_wire_record(source.image, source.lane,
                                                image, lane);
      }
    }
    final_prefix_ = next_final_prefix_;
    return Status::success();
  }

  template <ParticleExecutionStage Stage, bool Wait>
  [[nodiscard]] constexpr Status advance() const noexcept {
    return Status::success();
  }

  void publish_completed_migration() noexcept {
    lpfx::detail::HostParticlesAccess::publish_initialized_size(
        *owner_, resident_count_ + static_cast<std::size_t>(final_prefix_));
  }

  [[nodiscard]] std::size_t resident_count() const noexcept {
    return resident_count_;
  }
  [[nodiscard]] std::uint64_t cumulative_final_count() const noexcept {
    return final_prefix_;
  }
  // Records of the owner prefix that a grown owner must keep.
  [[nodiscard]] std::size_t preserved_count() const noexcept {
    return resident_count_ + static_cast<std::size_t>(final_prefix_);
  }
  [[nodiscard]] const outgoing_prefixes_type &
  outgoing_prefixes() const noexcept {
    return outgoing_;
  }
  [[nodiscard]] wire_storage_type &wire_storage() noexcept {
    return storage_.wire;
  }

private:
  // Packs the outgoing records of source (grouped by the source map) into
  // wire, attribute-major per peer.
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  pack(const owner_type &source, wire_storage_type &wire,
       const semantics_type &semantics) const noexcept {
    const std::uint32_t *const sources = storage_.scratch.template data<1>();
    const HostParticlePeerWireImageView<Contract> packets{
        wire.bytes(outgoing_).data(), outgoing_};
    for (std::uint8_t peer = 0; peer != packets.peer_count(); ++peer) {
      const auto image = packets.packet(peer);
      if (image.empty())
        continue;
      const ErrorCode error = pack_host_particle_wire_image<Contract>(
          source.view(),
          lpfx::detail::make_buffer_view(sources + packets.particle_offset(peer),
                                         image.particle_count()),
          image, semantics.codec_context());
      if (error != ErrorCode::success)
        return Status::failure(error);
    }
    return Status::success();
  }

  template <std::size_t... I>
  [[nodiscard]] std::array<std::byte *, schema_type::attribute_count>
  sink_bases(std::index_sequence<I...>) const noexcept {
    return {lpfx::detail::HostParticlesAccess::template raw_attribute_base<I>(
        *owner_)...};
  }

  const routing_type *routing_ = nullptr;
  const capacity_type *capacity_ = nullptr;
  owner_type *owner_ = nullptr;
  storage_type storage_{};
  outgoing_prefixes_type outgoing_{};
  std::size_t source_count_ = 0;
  std::size_t resident_count_ = 0;
  std::uint64_t final_prefix_ = 0;
  std::uint64_t next_final_prefix_ = 0;
  bool classified_ = false;
};

} // namespace lpfx::distributed::detail

#endif
