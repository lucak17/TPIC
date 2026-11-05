#ifndef LPFX_DISTRIBUTED_FIELD_PROGRAM_HPP
#define LPFX_DISTRIBUTED_FIELD_PROGRAM_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/distributed/detail/fixed_byte.hpp>
#include <lpfx/distributed/detail/fixed_exchange_program.hpp>
#include <lpfx/distributed/field/lowering.hpp>
#include <lpfx/memory/host/budget.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed::detail {

// The fixed-exchange execution of a field operation: capture every source
// once, exchange the fixed remote images once, then apply the dependent passes
// in schedule order.
template <class Execution> class FieldExchange final {
public:
  using execution_type = Execution;
  using pattern_type = typename Execution::pattern_type;
  using spec_type = typename pattern_type::spec_type;
  using space_type = typename Execution::owner_type::space_type;
  static constexpr bool has_source = true;
  static constexpr bool has_apply = true;
  static constexpr bool asynchronous = Execution::asynchronous;

  template <class Setup, class Binding>
  [[nodiscard]] static ErrorCode try_validate(const Setup &setup,
                                              const Binding &binding) noexcept {
    if (!binding.valid() || !setup.is_valid()) {
      return ErrorCode::invalid_argument;
    }
    const auto &topology = binding.plan().pattern().topology().input();
    if (!std::cmp_equal(topology.rank(), setup.rank()) ||
        !std::cmp_equal(topology.communicator_size(), setup.size())) {
      return ErrorCode::invalid_argument;
    }
    LPFX_TRY(validate_field_execution_binding(binding.plan(), binding.resource()));
    if constexpr (!std::same_as<space_type, HostSpace>) {
      using Backend = typename space_type::backend_type;
      LPFX_TRY(setup.template try_verify_device_affinity<Backend>(
          binding.executor()));
      const auto field = binding.resource().view();
      return meta::for_each_until_error<spec_type::component_count>(
          [&](const auto index) noexcept {
            const auto *const core =
                field.template component<decltype(index)::value>().core_data();
            return core == nullptr
                       ? ErrorCode::success
                       : setup.template try_verify_device_pointer_affinity<
                             Backend>(core);
          });
    }
    return ErrorCode::success;
  }

  template <class Binding>
  [[nodiscard]] static bool needs_native_work(const Binding &binding) noexcept {
    return lpfx::detail::field_needs_native_work(binding.plan().pattern());
  }

  // One send and one receive per remote direction with bytes, coalesced into
  // one envelope per peer on route 0.
  template <class Setup, class Binding, class Executor, class Emit>
  [[nodiscard]] ErrorCode prepare(HostMemoryBudget &budget, const Setup &setup,
                                  const Binding &binding,
                                  const Executor &executor,
                                  Emit &&emit) noexcept {
    FieldExecutionImages images{};
    LPFX_TRY(execution_.prepare(budget, binding.plan(), binding.resource(),
                                executor, images));
    constexpr std::size_t capacity =
        cartesian_neighbor_mask_count(spec_type::transport_mask);
    std::array<FixedByteSendMessage, capacity> sends{};
    std::array<FixedByteReceiveMessage, capacity> receives{};
    std::size_t send_count = 0;
    std::size_t receive_count = 0;
    const auto &pattern = binding.plan().pattern();
    for (std::size_t d = 0; d != pattern_type::direction_count; ++d) {
      const auto direction = pattern.direction(d);
      if (direction.peer_kind() != FieldPeerKind::remote_peer) {
        continue;
      }
      std::uint32_t peer = 0;
      if (!lpfx::detail::checked_integral_cast(direction.peer_rank(), peer)) {
        return ErrorCode::overflow;
      }
      constexpr FieldFlow out = FieldFlow::outbound;
      constexpr FieldFlow in = FieldFlow::inbound;
      if (const std::size_t bytes = direction.template message_bytes<out>();
          bytes != 0) {
        sends[send_count++] = {
            images.outgoing.data() + direction.template image_offset_bytes<out>(),
            bytes, peer, 0};
      }
      if (const std::size_t bytes = direction.template message_bytes<in>();
          bytes != 0) {
        receives[receive_count++] = {
            images.incoming.data() + direction.template image_offset_bytes<in>(),
            bytes, peer, 0};
      }
    }
    LPFX_TRY(try_coalesce_fixed_byte_messages(
        std::span{sends.data(), send_count}, send_count));
    LPFX_TRY(try_coalesce_fixed_byte_messages(
        std::span{receives.data(), receive_count}, receive_count));
    return emit(FixedByteExchange{
        {sends.data(), send_count},
        {receives.data(), receive_count},
        {images.outgoing.data(), images.outgoing.size()},
        {images.incoming.data(), images.incoming.size()},
        static_cast<std::uint32_t>(setup.rank()),
        static_cast<std::uint32_t>(setup.size())});
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode capture() noexcept {
    return execution_.template execute<FieldSemanticPass::source>();
  }

  LPFX_ALWAYS_INLINE void apply() noexcept {
    apply(typename lpfx::detail::FieldSchedulePolicy<
          spec_type::kind>::pass_sequence{});
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode record_completion() noexcept {
    return execution_.record_completion();
  }

  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance_apply() noexcept {
    return execution_.template advance_apply<Wait>();
  }

  [[nodiscard]] ErrorCode reset() noexcept { return execution_.reset(); }

  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return execution_.workspace_bytes();
  }

private:
  template <FieldSemanticPass Pass>
  LPFX_ALWAYS_INLINE void apply_pass() noexcept {
    if constexpr (Pass != FieldSemanticPass::source) {
      fixed_exchange_require_success(execution_.template execute<Pass>());
    }
  }

  template <FieldSemanticPass... Passes>
  LPFX_ALWAYS_INLINE void
  apply(lpfx::detail::FieldPassSequence<Passes...>) noexcept {
    (apply_pass<Passes>(), ...);
  }

  Execution execution_{};
};

} // namespace lpfx::distributed::detail

#endif
