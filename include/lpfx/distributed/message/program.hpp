#ifndef LPFX_DISTRIBUTED_MESSAGE_PROGRAM_HPP
#define LPFX_DISTRIBUTED_MESSAGE_PROGRAM_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/binding/message.hpp>
#include <lpfx/distributed/detail/byte_interval.hpp>
#include <lpfx/distributed/detail/fixed_byte.hpp>
#include <lpfx/distributed/detail/fixed_exchange_program.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed::detail {

// The fixed-exchange execution of messages: no source capture and no apply.
// The plan has coalesced every fragment into one envelope per peer, so
// preparation only resolves those ranges against the bound buffers.
template <class Buffers> struct MessageExchange final {
  using space_type = typename Buffers::space_type;
  static constexpr bool has_source = false;
  static constexpr bool has_apply = false;
  static constexpr bool asynchronous = false;

  template <class Setup, class Binding>
  [[nodiscard]] static ErrorCode try_validate(const Setup &setup,
                                              const Binding &binding) noexcept {
    if (!binding.valid() || !setup.is_valid()) {
      return ErrorCode::invalid_argument;
    }
    const auto &plan = binding.plan();
    const Buffers &buffers = binding.resource();
    if (!std::cmp_equal(plan.rank(), setup.rank()) ||
        !std::cmp_equal(plan.rank_count(), setup.size()) ||
        buffers.outgoing_bytes() != plan.outgoing_bytes() ||
        buffers.incoming_bytes() != plan.incoming_bytes()) {
      return ErrorCode::invalid_argument;
    }
    lpfx::detail::ByteInterval outgoing{};
    lpfx::detail::ByteInterval incoming{};
    LPFX_TRY(lpfx::detail::try_make_byte_interval(
        buffers.outgoing_data(), buffers.outgoing_bytes(), outgoing));
    LPFX_TRY(lpfx::detail::try_make_byte_interval(
        buffers.incoming_data(), buffers.incoming_bytes(), incoming));
    if (!lpfx::detail::disjoint(outgoing, incoming)) {
      return ErrorCode::invalid_argument;
    }
    if constexpr (!std::same_as<space_type, HostSpace>) {
      using Backend = typename space_type::backend_type;
      LPFX_TRY(setup.template try_verify_device_affinity<Backend>(
          binding.executor()));
      if (buffers.outgoing_bytes() != 0) {
        LPFX_TRY(setup.template try_verify_device_pointer_affinity<Backend>(
            buffers.outgoing_data()));
      }
      if (buffers.incoming_bytes() != 0) {
        LPFX_TRY(setup.template try_verify_device_pointer_affinity<Backend>(
            buffers.incoming_data()));
      }
    }
    return ErrorCode::success;
  }

  template <class Binding>
  [[nodiscard]] static bool needs_native_work(const Binding &binding) noexcept {
    return !binding.plan().peer_messages().empty();
  }

  template <class Setup, class Binding, class Executor, class Emit>
  [[nodiscard]] static ErrorCode prepare(HostMemoryBudget &, const Setup &,
                                         const Binding &binding,
                                         const Executor &,
                                         Emit &&emit) noexcept {
    const auto &plan = binding.plan();
    const Buffers &buffers = binding.resource();
    constexpr std::size_t capacity =
        std::remove_cvref_t<Binding>::plan_type::maximum_peer_count;
    std::array<FixedByteSendMessage, capacity> sends{};
    std::array<FixedByteReceiveMessage, capacity> receives{};
    std::size_t send_count = 0;
    std::size_t receive_count = 0;
    for (const auto &peer_message : plan.peer_messages()) {
      std::uint32_t peer = 0;
      if (!lpfx::detail::checked_integral_cast(peer_message.peer, peer)) {
        return ErrorCode::overflow;
      }
      if (!peer_message.outgoing.empty()) {
        sends[send_count++] = {
            buffers.outgoing_data() + peer_message.outgoing.offset_bytes,
            peer_message.outgoing.size_bytes, peer, 0};
      }
      if (!peer_message.incoming.empty()) {
        receives[receive_count++] = {
            buffers.incoming_data() + peer_message.incoming.offset_bytes,
            peer_message.incoming.size_bytes, peer, 0};
      }
    }
    std::uint32_t rank = 0;
    std::uint32_t rank_count = 0;
    if (!lpfx::detail::checked_integral_cast(plan.rank(), rank) ||
        !lpfx::detail::checked_integral_cast(plan.rank_count(), rank_count)) {
      return ErrorCode::overflow;
    }
    return emit(FixedByteExchange{
        {sends.data(), send_count},
        {receives.data(), receive_count},
        {buffers.outgoing_data(), buffers.outgoing_bytes()},
        {buffers.incoming_data(), buffers.incoming_bytes()},
        rank,
        rank_count});
  }

  [[nodiscard]] static constexpr ErrorCode reset() noexcept {
    return ErrorCode::success;
  }
  [[nodiscard]] static constexpr std::size_t workspace_bytes() noexcept {
    return 0;
  }
};

} // namespace lpfx::distributed::detail

#endif
