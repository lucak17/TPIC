#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_PARTICLE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_PARTICLE_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <utility>

#include <mpi.h>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/particle/migration/payload_slices.hpp>
#include <lpfx/distributed/particle/migration/routing.hpp>
#include <lpfx/distributed/transport/mpi/agreement.hpp>
#include <lpfx/distributed/transport/mpi/communicator.hpp>
#include <lpfx/distributed/transport/mpi/p2p.hpp>
#include <lpfx/distributed/transport/workspace.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

// The migration program owns exactly one of these. Count and payload batches
// borrow the same slots because their live intervals never overlap.
class MpiParticleRequestStorage final {
public:
  [[nodiscard]] ErrorCode try_prepare_resources(HostMemoryBudget &budget,
                                                const std::size_t capacity) noexcept {
    if (!slots_.empty() ||
        capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return ErrorCode::invalid_argument;
    }
    return slots_.try_prepare(budget, {capacity, capacity});
  }
  [[nodiscard]] ErrorCode bind(const MpiCallDomain &domain,
                               lpfx::detail::PreparedMpiRequestBatch &batch) noexcept {
    return batch.bind(domain, slots_.data<0>(), slots_.data<1>(), capacity());
  }
  [[nodiscard]] std::size_t capacity() const noexcept { return slots_.size<0>(); }
  void reset_resources() noexcept { reset(); }
  void reset() noexcept { slots_.reset(); }

private:
  HostWorkspace<MPI_Request, MPI_Status> slots_{};
};

template <class Pattern> class MpiParticleCountPort final {
public:
  using pattern_type = Pattern;
  using request_storage_type = MpiParticleRequestStorage;

  [[nodiscard]] bool empty() const noexcept { return batch_.empty(); }
  [[nodiscard]] bool is_valid() const noexcept { return batch_.is_valid(); }
  [[nodiscard]] bool reusable() const noexcept { return is_valid() && !batch_.active(); }
  [[nodiscard]] static std::size_t request_capacity(const pattern_type &pattern) noexcept {
    return pattern.routing().round_count() == 0 ? 0 : 2U * pattern.routing().peer_count();
  }
  template <class Plan, class Session, class Setup>
    requires std::same_as<typename Plan::pattern_type, pattern_type>
  [[nodiscard]] ErrorCode try_prepare_resources(
      HostMemoryBudget &, Session &session, const Setup &setup,
      const Plan &plan, const int tag_base, request_storage_type &backing) noexcept {
    if (!empty() || !setup.is_valid() || !plan.is_valid()) {
      return ErrorCode::invalid_state;
    }
    auto &context = session.mpi().control();
    if (!context.active() || backing.capacity() < request_capacity(plan.pattern())) {
      return ErrorCode::invalid_argument;
    }
    const auto &pattern = plan.pattern();
    for (std::uint8_t peer = 0; peer != pattern.routing().peer_count(); ++peer) {
      const auto rank = pattern.peer_rank(peer);
      if (!std::in_range<int>(rank) || static_cast<int>(rank) < 0 ||
          static_cast<int>(rank) >= context.size() ||
          static_cast<int>(rank) == context.rank()) {
        return ErrorCode::invalid_argument;
      }
    }
    return bind(pattern, context, tag_base, backing);
  }
  // Growth changes only storage bounds. Reuse the validated communicator and
  // tags, but bind the peer plan to the replacement's permanent address.
  [[nodiscard]] ErrorCode prepare_growth(
      const MpiParticleCountPort &previous, const pattern_type &pattern,
      request_storage_type &backing) noexcept {
    if (!empty() || !previous.reusable() ||
        backing.capacity() < request_capacity(pattern)) {
      return ErrorCode::invalid_state;
    }
    return bind(pattern, *previous.context_, previous.tag_base_, backing);
  }
  [[nodiscard]] const MpiCallDomain &native_domain() const noexcept {
    return context_->domain();
  }
  // Rank-symmetric error agreement on the control context.
  [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
  agree_error(const ErrorCode local) const noexcept {
    assert(reusable());
    return agree_or_abort(*context_, local);
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start(
      const std::span<const std::uint64_t> outgoing,
      const std::span<std::uint64_t> incoming) noexcept {
    return start_with_peer_index<std::size_t>(outgoing, incoming);
  }
  template <class PeerIndex>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start_with_peer_index(
      const std::span<const std::uint64_t> outgoing,
      const std::span<std::uint64_t> incoming) noexcept {
    static_assert(std::same_as<PeerIndex, std::size_t> ||
                  std::same_as<PeerIndex, std::uint8_t>);
    const std::size_t peers = pattern_->routing().peer_count();
    if (outgoing.size() != peers || incoming.size() != peers) {
      return Status::failure(ErrorCode::invalid_argument);
    }
    // Naming the callback preserves its inline contract through CUDA host
    // lowering. Reference captures retain the standalone host call layout.
    struct SubmitCounts {
      const std::size_t &peers;
      const std::span<std::uint64_t> &incoming;
      MpiParticleCountPort &port;
      const std::span<const std::uint64_t> &outgoing;
#if defined(__clang__) || defined(__CUDACC__) || defined(__HIPCC__)
      LPFX_ALWAYS_INLINE
#endif
      ErrorCode operator()(const std::span<MPI_Request> requests) const noexcept {
        for (PeerIndex peer = 0; peer != peers; ++peer) {
          if (MPI_Irecv(incoming.data() + peer, 1, MPI_UINT64_T,
                        port.native_peer(static_cast<std::uint8_t>(peer)),
                        port.tag_base_, port.context_->native_handle(),
                        &requests[peer]) != MPI_SUCCESS)
            return ErrorCode::backend_failure;
        }
        for (PeerIndex peer = 0; peer != peers; ++peer) {
          if (MPI_Isend(outgoing.data() + peer, 1, MPI_UINT64_T,
                        port.native_peer(static_cast<std::uint8_t>(peer)),
                        port.tag_base_, port.context_->native_handle(),
                        &requests[peers + peer]) != MPI_SUCCESS)
            return ErrorCode::backend_failure;
        }
        return ErrorCode::success;
      }
    };
    const Status started = batch_.start(
        2U * peers, SubmitCounts{peers, incoming, *this, outgoing});
    return started.is_success() ? complete_exchange() : started;
  }
  template <bool Wait> [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() noexcept {
    if (!batch_.active()) { return Status::success(); }
    const Status completed = batch_.template advance<Wait>();
    return completed.is_success() ? complete_exchange() : completed;
  }
  // The bound pattern. Hard precondition: is_valid().
  [[nodiscard]] const pattern_type &pattern() const noexcept { return *pattern_; }
  [[nodiscard]] int native_peer(const std::uint8_t peer) const noexcept {
    return static_cast<int>(pattern_->peer_rank(peer));
  }
  [[nodiscard]] int tag_base() const noexcept { return tag_base_; }
  void reset_resources() noexcept {
    batch_.reset(); context_ = nullptr; pattern_ = nullptr; tag_base_ = -1;
  }
  void reset() noexcept { reset_resources(); }
private:
  [[nodiscard]] ErrorCode bind(const pattern_type &pattern,
                               MpiControlContext &context, const int tag_base,
                               request_storage_type &backing) noexcept {
    if (!pattern.valid()) { return ErrorCode::invalid_argument; }
    LPFX_TRY(backing.bind(context.domain(), batch_));
    pattern_ = &pattern;
    context_ = &context;
    tag_base_ = tag_base;
    return ErrorCode::success;
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE Status complete_exchange() noexcept {
    const std::size_t peers = pattern_->routing().peer_count();
    if (peers != 0) {
      MpiCallLease lease{};
      if (try_acquire_disjoint_mpi_call(context_->domain(), lease)
          != ErrorCode::success) { std::abort(); }
      const auto statuses = batch_.completed_statuses();
      for (std::size_t peer = 0; peer != peers; ++peer) {
        int received = 0;
        if (statuses[peer].MPI_SOURCE != native_peer(static_cast<std::uint8_t>(peer)) ||
            statuses[peer].MPI_TAG != tag_base_ ||
            MPI_Get_count(&statuses[peer], MPI_UINT64_T, &received) != MPI_SUCCESS ||
            received != 1) { std::abort(); }
      }
    }
    return batch_.release();
  }
  lpfx::detail::PreparedMpiRequestBatch batch_{};
  const pattern_type *pattern_ = nullptr;
  MpiControlContext *context_ = nullptr;
  int tag_base_ = -1;
};

// Dynamic particle prefixes change each round. One checked slice traversal
// materializes native arguments into reusable capacity-backed records; issue
// and receive-count auditing never re-enumerate or redo checked arithmetic.
struct MpiParticleByteRecipe final {
  const std::byte *data = nullptr;
  int bytes = 0;
  int peer = -1;
};

// Particle payload MPI P2P for ParticlePayloadPort. Each nonempty peer packet
// is one MPI_BYTE message on the count port's tag base + 1.
template <class Contract> class MpiParticleP2p final {
public:
  static constexpr bool stream_ordered = false;
  static constexpr std::uint64_t maximum_peer_particles =
      std::uint64_t{std::numeric_limits<int>::max()} / Contract::record_wire_width;

  [[nodiscard]] bool empty() const noexcept { return batch_.empty(); }
  [[nodiscard]] bool is_valid() const noexcept { return batch_.is_valid(); }
  [[nodiscard]] bool reusable() const noexcept { return batch_.reusable(); }

  template <class Pattern>
  [[nodiscard]] static ErrorCode request_capacity(const Pattern &pattern,
                                                  std::size_t &output) noexcept {
    if (pattern.routing().round_count() == 0) {
      output = 0;
      return ErrorCode::success;
    }
    constexpr std::uint64_t native_max = std::numeric_limits<int>::max();
    std::size_t count = 0;
    for (std::uint8_t peer = 0; peer != pattern.routing().peer_count(); ++peer) {
      for (const std::uint64_t particles :
           {pattern.capacity().maximum_peer_incoming(peer),
            pattern.capacity().maximum_peer_outgoing(peer)}) {
        std::uint64_t bytes = 0;
        if (!lpfx::detail::checked_multiply(
                particles, Contract::record_wire_width, bytes)) {
          return ErrorCode::overflow;
        }
        // Each peer is one MPI_BYTE message. Reject unrepresentable packet
        // capacities at preparation rather than fragmenting the payload or
        // narrowing its count after the count exchange has already started.
        if (bytes > native_max) {
          return ErrorCode::capacity_exceeded;
        }
        count += bytes != 0;
      }
    }
    output = count;
    return ErrorCode::success;
  }

  // Retains the payload context even when the initial bounds are zero: a
  // later count exchange may introduce the first nonempty packet.
  template <class Session, class Setup, class Plan, class CountPort>
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &budget, Session &session,
                                      const Setup &, const Plan &plan,
                                      const CountPort &count,
                                      MpiParticleRequestStorage &backing) noexcept {
    MpiPayloadContext *const context = session.mpi().has_payload_context()
        ? &session.mpi().payload() : nullptr;
    return prepare(budget, plan, count, backing, context);
  }
  template <class Plan, class CountPort>
  [[nodiscard]] ErrorCode prepare_growth(const MpiParticleP2p &previous,
                                         HostMemoryBudget &budget, const Plan &plan,
                                         const CountPort &count,
                                         MpiParticleRequestStorage &backing) noexcept {
    return prepare(budget, plan, count, backing, previous.context_);
  }

  // The caller has checked reusable().
  template <class Executor, class Outgoing, class OutgoingBytes,
            class Incoming, class IncomingBytes, class CountPort>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start(const Executor &, const Outgoing &outgoing_prefixes,
        const OutgoingBytes outgoing, const Incoming &incoming_prefixes,
        const IncomingBytes incoming, const CountPort &count) noexcept {
    std::size_t recipe_count = 0;
    auto materialize = [&](const auto slice) noexcept {
      if (recipe_count == recipes_.size()) { return ErrorCode::capacity_exceeded; }
      recipes_[recipe_count++] = {slice.data, static_cast<int>(slice.bytes),
                                  count.native_peer(slice.peer)};
      return ErrorCode::success;
    };
    ErrorCode error = for_each_particle_payload_slice<CommunicationFlow::receive>(
        incoming_prefixes, incoming, std::numeric_limits<int>::max(),
        materialize);
    receive_count_ = recipe_count;
    if (error == ErrorCode::success) {
      error = for_each_particle_payload_slice<CommunicationFlow::send>(
          outgoing_prefixes, outgoing, std::numeric_limits<int>::max(),
          materialize);
    }
    if (error != ErrorCode::success) { return Status::failure(error); }
    const Status started = batch_.start(recipe_count,
        [this](const std::span<MPI_Request> requests)
#if defined(__clang__)
        __attribute__((always_inline))
#endif
        noexcept {
          const auto communicator = context_->native_handle();
          for (std::size_t index = 0; index != receive_count_; ++index) {
            const auto &recipe = recipes_[index];
            if (MPI_Irecv(const_cast<std::byte *>(recipe.data), recipe.bytes, MPI_BYTE,
                          recipe.peer, payload_tag_, communicator,
                          &requests[index]) != MPI_SUCCESS) { return ErrorCode::backend_failure; }
          }
          const auto recipe_count = requests.size();
          for (std::size_t index = receive_count_; index != recipe_count;
               ++index) {
            const auto &recipe = recipes_[index];
            if (MPI_Isend(recipe.data, recipe.bytes, MPI_BYTE, recipe.peer, payload_tag_,
                          communicator, &requests[index]) != MPI_SUCCESS) {
              return ErrorCode::backend_failure;
            }
          }
          return ErrorCode::success;
        });
    // Start completes synchronously only for an empty native batch. Successful
    // materialization then also proves receive_count_ is zero, so there are no
    // receive statuses to audit before releasing that completed batch.
    return started.is_success() ? batch_.release() : started;
  }
  template <bool Wait> [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() noexcept {
    if (!batch_.active()) { return Status::success(); }
    const Status completed = batch_.template advance<Wait>();
    return completed.is_success() ? complete() : completed;
  }
  [[nodiscard]] std::size_t maximum_work_count() const noexcept { return recipes_.size(); }
  void reset() noexcept {
    batch_.reset(); storage_.reset(); recipes_ = {};
    receive_count_ = 0; context_ = nullptr; payload_tag_ = -1;
  }

private:
  template <class Plan, class CountPort>
  [[nodiscard]] ErrorCode prepare(HostMemoryBudget &budget, const Plan &plan,
                                  const CountPort &count,
                                  MpiParticleRequestStorage &backing,
                                  MpiPayloadContext *const context) noexcept {
    std::size_t capacity = 0;
    LPFX_TRY(request_capacity(plan.pattern(), capacity));
    if (capacity > backing.capacity()) { return ErrorCode::capacity_exceeded; }
    if (capacity != 0 && context == nullptr) { return ErrorCode::invalid_state; }
    LPFX_TRY(storage_.try_prepare(budget, {capacity}));
    const auto &domain = context != nullptr ? context->domain() : count.native_domain();
    const ErrorCode bound = backing.bind(domain, batch_);
    if (bound != ErrorCode::success) { storage_.reset(); return bound; }
    recipes_ = storage_.span<0>();
    context_ = context;
    payload_tag_ = count.tag_base() + 1;
    return ErrorCode::success;
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE Status complete() noexcept {
    const auto receives = receive_count_;
    if (receives != 0) {
      MpiCallLease lease{};
      if (try_acquire_disjoint_mpi_call(context_->domain(), lease)
          != ErrorCode::success) { std::abort(); }
      const auto statuses = batch_.completed_statuses();
      for (std::size_t index = 0; index != receives; ++index) {
        const auto &recipe = recipes_[index];
        int received = 0;
        if (statuses[index].MPI_SOURCE != recipe.peer || statuses[index].MPI_TAG != payload_tag_ ||
            MPI_Get_count(&statuses[index], MPI_BYTE, &received) != MPI_SUCCESS ||
            received != recipe.bytes) { std::abort(); }
      }
    }
    receive_count_ = 0;
    return batch_.release();
  }

  lpfx::detail::PreparedMpiRequestBatch batch_{};
  HostWorkspace<MpiParticleByteRecipe> storage_{};
  std::span<MpiParticleByteRecipe> recipes_{};
  std::size_t receive_count_ = 0;
  MpiPayloadContext *context_ = nullptr;
  int payload_tag_ = -1;
};

} // namespace lpfx::distributed::detail

namespace lpfx::distributed {

// This one-shot collective precedes routing construction and therefore uses
// the bootstrap communicator. Valid process-local setups and rank-consistent
// call order are preconditions; every ordinary local input error is reduced.
template <lpfx::detail::CartesianRank Rank,
          lpfx::detail::AdaptiveParticleHopPolicyType HopPolicy>
[[nodiscard]] ErrorCode try_agree_mpi_particle_hop_bound(
    const CommunicationSetup &setup,
    const PreparedCanonicalCartesianTopology3D<Rank> &topology,
    const std::uint64_t local_maximum,
    AgreedParticleHopBound<Rank, HopPolicy> &output) noexcept {
  if (!setup.is_valid()) { return ErrorCode::invalid_state; }
  ErrorCode local = output.active() ? ErrorCode::invalid_state : ErrorCode::success;
  if (!topology.is_valid() ||
      static_cast<std::uint64_t>(topology.input().communicator_size()) !=
          static_cast<std::uint64_t>(setup.size()) ||
      static_cast<std::uint64_t>(topology.input().rank()) != static_cast<std::uint64_t>(setup.rank())) {
    local = ErrorCode::invalid_argument;
  }
  const auto input = topology.input();
  const auto shape = input.process_shape();
  const auto periodicity = input.periodicity();
  const std::uint64_t bound = topology.is_valid()
      ? lpfx::detail::particle_topology_hop_bound(input) : 0;
  std::array<std::uint64_t, 1> maximum{local_maximum};
  LPFX_TRY(detail::agree_contract(
      setup, local,
      std::array<std::uint64_t, 10>{
          static_cast<std::uint64_t>(shape.x()), static_cast<std::uint64_t>(shape.y()),
          static_cast<std::uint64_t>(shape.z()), static_cast<std::uint64_t>(periodicity.x),
          static_cast<std::uint64_t>(periodicity.y), static_cast<std::uint64_t>(periodicity.z),
          bound, static_cast<std::uint64_t>(HopPolicy::kind), HopPolicy::particle_wire_id,
          HopPolicy::consensus_identity},
      maximum));
  if (maximum[0] > bound) { return ErrorCode::capacity_exceeded; }
  output = lpfx::detail::PrivateConstructionAccess::create<
      AgreedParticleHopBound<Rank, HopPolicy>>(input, maximum[0]);
  return ErrorCode::success;
}

} // namespace lpfx::distributed

#endif
