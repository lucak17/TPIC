#ifndef LPFX_DISTRIBUTED_TRANSPORT_XCCL_P2P_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_XCCL_P2P_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/detail/fixed_byte.hpp>
#include <lpfx/distributed/particle/migration/payload_slices.hpp>
#include <lpfx/distributed/particle/wire/storage.hpp>
#include <lpfx/distributed/transport/workspace.hpp>
#include <lpfx/distributed/transport/xccl/session.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::distributed::detail {

template <class Byte> struct XcclFixedByteRecipe final {
  Byte *data = nullptr;
  std::size_t size_bytes = 0;
  int peer = -1;
};

// Fixed-message XCCL P2P for FixedBytePort. NCCL/RCCL byte counts are size_t:
// one prepared recipe per fixed logical message is sufficient. Families supply
// canonical (peer, route) order; cold projection validates that order and
// discards the route key. Repeated epochs visit only native recipes.
template <XcclGroupedNativeAdapter Native> class XcclP2p final {
  using Send = XcclFixedByteRecipe<const std::byte>;
  using Receive = XcclFixedByteRecipe<std::byte>;

public:
  using session_type = XcclSession<Native>;

  constexpr XcclP2p() noexcept = default;
  XcclP2p(const XcclP2p &) = delete;
  XcclP2p &operator=(const XcclP2p &) = delete;

  [[nodiscard]] bool empty() const noexcept { return session_ == nullptr; }
  [[nodiscard]] bool is_valid() const noexcept {
    return session_ != nullptr && session_->ready();
  }

  // Native access is unstaged: address is the identity.
  template <class Session, class Address>
    requires XcclSessionProvider<Session, Native>
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &budget,
                                      Session &shared_session,
                                      const FixedByteExchange &shape,
                                      const std::span<const int>,
                                      Address &&) noexcept {
    session_type &session = shared_session.xccl();
    if (!session.ready()) {
      return ErrorCode::invalid_state;
    }
    LPFX_TRY(storage_.try_prepare(budget,
                                  {shape.sends.size(), shape.receives.size()}));
    if (!copy_canonical(shape.sends, shape.rank, shape.rank_count,
                        storage_.template span<0>()) ||
        !copy_canonical(shape.receives, shape.rank, shape.rank_count,
                        storage_.template span<1>())) {
      storage_.reset();
      return ErrorCode::invalid_argument;
    }
    sends_ = storage_.template span<0>();
    receives_ = storage_.template span<1>();
    session_ = &session;
    return ErrorCode::success;
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  submit(const device::Executor<typename Native::device_backend_type>
             &executor) noexcept {
    const auto communicator = session_->native_handle();
    return session_->template submit<XcclSubmissionMode::grouped>(
        executor, [&]()
#if defined(__clang__)
        __attribute__((always_inline))
#endif
        noexcept {
          for (const Receive &recipe : receives_) {
            const auto result = Native::template transfer<true>(
                recipe.data, recipe.size_bytes, recipe.peer, communicator,
                executor.stream());
            if (!Native::succeeded(result)) {
              return result;
            }
          }
          for (const Send &recipe : sends_) {
            const auto result = Native::template transfer<false>(
                recipe.data, recipe.size_bytes, recipe.peer, communicator,
                executor.stream());
            if (!Native::succeeded(result)) {
              return result;
            }
          }
          return Native::success_result;
        });
  }

  [[nodiscard]] bool has_incoming_work() const noexcept {
    return !receives_.empty();
  }
  [[nodiscard]] std::size_t work_count() const noexcept {
    return sends_.size() + receives_.size();
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return storage_.capacity_bytes();
  }
  void reset() noexcept {
    session_ = nullptr;
    receives_ = {};
    sends_ = {};
    storage_.reset();
  }

private:
  // Family plans already canonicalize peer envelopes. Preserve their order
  // and reject invalid or ambiguous matching keys in one linear projection.
  template <class Message, class Recipe>
  [[nodiscard]] static bool copy_canonical(
      const std::span<const Message> input, const std::uint32_t rank,
      const std::uint32_t rank_count, const std::span<Recipe> output) noexcept {
    std::uint64_t previous = 0;
    for (std::size_t index = 0; index != input.size(); ++index) {
      const auto &message = input[index];
      if (message.data == nullptr || message.size_bytes == 0 ||
          message.peer >= rank_count || message.peer == rank ||
          message.peer > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        return false;
      const std::uint64_t key = (static_cast<std::uint64_t>(message.peer) << 32U) |
                                static_cast<std::uint64_t>(message.route);
      if (index != 0 && key <= previous)
        return false;
      output[index] = {message.data, message.size_bytes, static_cast<int>(message.peer)};
      previous = key;
    }
    return true;
  }

  HostWorkspace<Send, Receive> storage_{};
  std::span<const Send> sends_{};
  std::span<const Receive> receives_{};
  session_type *session_ = nullptr;
};

// Particle payload XCCL P2P for ParticlePayloadPort. Payload borrows the one
// Communication-owned session; counts are a distinct MPI role. Each transfer
// submits one entire contiguous peer packet from device storage. start()
// returns once the grouped transfers are enqueued on the executor's stream:
// every later reader of the payload is ordered after them on that stream, so
// the port keeps no completion state.
template <XcclGroupedNativeAdapter Native, class Contract>
class XcclParticleP2p final {
public:
  using session_type = XcclSession<Native>;
  static constexpr bool stream_ordered = true;
  static constexpr std::uint64_t maximum_peer_particles = UINT32_MAX;

  [[nodiscard]] bool empty() const noexcept { return session_ == nullptr; }
  [[nodiscard]] bool is_valid() const noexcept { return session_ != nullptr; }
  [[nodiscard]] static constexpr bool reusable() noexcept { return true; }
  template <class Pattern>
  [[nodiscard]] static constexpr ErrorCode request_capacity(
      const Pattern &, std::size_t &output) noexcept {
    output = 0;
    return ErrorCode::success;
  }

  template <class Session, class Setup, class Plan, class CountPort,
            class RequestStorage>
  [[nodiscard]] ErrorCode try_prepare(HostMemoryBudget &, Session &shared_session,
                                      const Setup &, const Plan &plan,
                                      const CountPort &, RequestStorage &) noexcept {
    bool work = false;
    for (std::uint8_t peer = 0; plan.pattern().routing().round_count() != 0 &&
         peer != plan.pattern().routing().peer_count(); ++peer) {
      work = work || plan.pattern().capacity().maximum_peer_outgoing(peer) != 0 ||
                      plan.pattern().capacity().maximum_peer_incoming(peer) != 0;
    }
    auto &session = shared_session.xccl();
    if (work && !session.ready()) {
      return ErrorCode::invalid_state;
    }
    session_ = &session;
    return ErrorCode::success;
  }
  // Composition has already provisioned the session for this routing
  // topology, including initially zero capacity. Growth only transfers the
  // session borrow.
  template <class Plan, class CountPort, class RequestStorage>
  [[nodiscard]] ErrorCode prepare_growth(const XcclParticleP2p &previous,
                                         HostMemoryBudget &, const Plan &,
                                         const CountPort &,
                                         RequestStorage &) noexcept {
    session_ = previous.session_;
    return ErrorCode::success;
  }

  template <class Executor, class Outgoing, class OutgoingBytes,
            class Incoming, class IncomingBytes, class CountPort>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start(const Executor &executor, const Outgoing &outgoing_prefixes,
        const OutgoingBytes outgoing, const Incoming &incoming_prefixes,
        const IncomingBytes incoming, const CountPort &count) noexcept {
    if (outgoing_prefixes.total_particles() == 0 &&
        incoming_prefixes.total_particles() == 0) {
      return Status::success();
    }
    const auto communicator = session_->native_handle();
    return session_->template submit<XcclSubmissionMode::grouped>(
        executor, [&]()
#if defined(__clang__)
        __attribute__((always_inline))
#endif
        noexcept {
          auto native = Native::success_result;
          const auto issue = [&](const auto slice) noexcept {
            if constexpr (std::remove_cvref_t<decltype(slice)>::flow == CommunicationFlow::receive) {
              native = Native::template transfer<true>(slice.data, slice.bytes,
                  count.native_peer(slice.peer), communicator, executor.stream());
            } else {
              native = Native::template transfer<false>(slice.data, slice.bytes,
                  count.native_peer(slice.peer), communicator, executor.stream());
            }
            return Native::succeeded(native) ? ErrorCode::success : ErrorCode::backend_failure;
          };
          ErrorCode error = for_each_particle_payload_slice<CommunicationFlow::receive>(
              incoming_prefixes, incoming,
              std::numeric_limits<std::uint64_t>::max(), issue);
          if (error == ErrorCode::success) {
            error = for_each_particle_payload_slice<CommunicationFlow::send>(
                outgoing_prefixes, outgoing,
                std::numeric_limits<std::uint64_t>::max(), issue);
          }
          return error == ErrorCode::success ? Native::success_result
              : (Native::succeeded(native) ? Native::invalid_argument_result : native);
        });
  }
  [[nodiscard]] static constexpr std::size_t maximum_work_count() noexcept {
    return 2U * particle_migration_maximum_peer_count;
  }
  void reset() noexcept { session_ = nullptr; }

private:
  session_type *session_ = nullptr;
};

} // namespace lpfx::distributed::detail

#endif
