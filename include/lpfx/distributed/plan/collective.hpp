#ifndef LPFX_DISTRIBUTED_PLAN_COLLECTIVE_HPP
#define LPFX_DISTRIBUTED_PLAN_COLLECTIVE_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/field/wire_traits.hpp>
#include <lpfx/distributed/descriptor/collective.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>
#include <lpfx/distributed/registration/record.hpp>

namespace lpfx::distributed {

template <std::uint64_t ChannelIdentity, field::wire_scalar T,
          detail::CollectiveDescriptor Descriptor>
  requires(ChannelIdentity != 0)
struct CollectivePlan final {
  using value_type = T;
  using descriptor_type = Descriptor;

  static constexpr std::uint64_t channel_identity = ChannelIdentity;
};

template <class Definition> class PreparedCollectivePlan final {
public:
  using definition_type = Definition;
  using value_type = typename definition_type::value_type;
  using descriptor_type = typename definition_type::descriptor_type;
  using channel_type = RegistrationChannel<64, 0, 0>;

  static constexpr std::uint64_t channel_identity =
      definition_type::channel_identity;

  constexpr PreparedCollectivePlan() noexcept = default;

  // Only try_prepare_collective_plan produces a nonzero rank count.
  [[nodiscard]] constexpr bool is_valid() const noexcept {
    return rank_count_ != 0;
  }

  [[nodiscard]] constexpr const channel_type &
  registration_channel() const noexcept {
    return channel_;
  }

  [[nodiscard]] constexpr std::size_t input_count() const noexcept {
    return input_count_;
  }
  [[nodiscard]] constexpr std::size_t output_count() const noexcept {
    return output_count_;
  }
  [[nodiscard]] constexpr std::uint32_t rank_count() const noexcept {
    return rank_count_;
  }

  [[nodiscard]] static constexpr bool permits_exact_in_place() noexcept {
    return descriptor_type::kind == collective_kind::all_reduce ||
           descriptor_type::kind == collective_kind::reduce ||
           descriptor_type::kind == collective_kind::broadcast;
  }

  [[nodiscard]] static constexpr bool requires_exact_in_place() noexcept {
    return descriptor_type::kind == collective_kind::broadcast;
  }

  [[nodiscard]] static constexpr bool
  counts_are_valid(const std::size_t input_count,
                   const std::size_t output_count,
                   const std::uint32_t rank_count) noexcept {
    if (rank_count == 0) {
      return false;
    }
    if constexpr (permits_exact_in_place()) {
      return input_count == output_count;
    } else if constexpr (descriptor_type::kind == collective_kind::all_gather) {
      return input_count <= std::numeric_limits<std::size_t>::max() /
                                static_cast<std::size_t>(rank_count) &&
             output_count == input_count * static_cast<std::size_t>(rank_count);
    } else {
      return output_count <= std::numeric_limits<std::size_t>::max() /
                                 static_cast<std::size_t>(rank_count) &&
             input_count == output_count * static_cast<std::size_t>(rank_count);
    }
  }

private:
  constexpr PreparedCollectivePlan(const std::size_t input_count,
                                   const std::size_t output_count,
                                   const std::uint32_t rank_count,
                                   const channel_type &channel) noexcept
      : channel_(channel), input_count_(input_count),
        output_count_(output_count), rank_count_(rank_count) {}

  channel_type channel_{};
  std::size_t input_count_ = 0;
  std::size_t output_count_ = 0;
  std::uint32_t rank_count_ = 0;

  friend struct lpfx::detail::PrivateConstructionAccess;
};

template <class Definition>
[[nodiscard]] constexpr ErrorCode try_prepare_collective_plan(
    const Definition, const std::size_t input_count,
    const std::size_t output_count, const std::uint32_t rank_count,
    PreparedCollectivePlan<Definition> &output) noexcept {
  using Plan = PreparedCollectivePlan<Definition>;
  using Descriptor = typename Plan::descriptor_type;
  if (!Plan::counts_are_valid(input_count, output_count, rank_count) ||
      (Descriptor::root != no_collective_root &&
       Descriptor::root >= rank_count)) {
    return ErrorCode::invalid_argument;
  }
  std::uint64_t input_count_word{};
  std::uint64_t output_count_word{};
  if (!lpfx::detail::checked_integral_cast(input_count, input_count_word) ||
      !lpfx::detail::checked_integral_cast(output_count, output_count_word)) {
    return ErrorCode::overflow;
  }

  typename Plan::channel_type channel{};
  LPFX_TRY(try_make_channel_id(Definition::channel_identity, channel.channel_id));
  std::array<std::byte, Plan::channel_type::invariant_descriptor_capacity>
      descriptor{};
  lpfx::detail::integer_codec<std::endian::little>::write_sequence(
      descriptor.data(),
      std::array<std::uint64_t, 8>{
          static_cast<std::uint64_t>(Descriptor::kind),
          static_cast<std::uint64_t>(Descriptor::reduction),
          field::wire_traits<typename Definition::value_type>::identity,
          sizeof(typename Definition::value_type), input_count_word,
          output_count_word, Descriptor::root, rank_count});
  static_cast<void>(channel.invariant_descriptor.try_assign(descriptor));

  output = lpfx::detail::PrivateConstructionAccess::create<Plan>(
      input_count, output_count, rank_count, channel);
  return ErrorCode::success;
}

} // namespace lpfx::distributed

#endif
