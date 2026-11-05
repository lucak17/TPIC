#ifndef LPFX_DISTRIBUTED_COLLECTIVE_OPERAND_HPP
#define LPFX_DISTRIBUTED_COLLECTIVE_OPERAND_HPP

#include <cstddef>
#include <ranges>
#include <span>
#include <type_traits>

#include <lpfx/distributed/detail/byte_interval.hpp>
#include <lpfx/memory/lifetime.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx::distributed {

// A typed, borrowed collective operand. Exact aliasing is type state so
// native source selection has no repeated-path branch. Separate-buffer
// overlap is rejected once by the collective semantic owner during cold
// validation.
template <LifetimeStartEligible T, class Space, bool ExactInPlace = false>
class collective_operand final {
public:
  using value_type = T;
  using space_type = Space;
  static constexpr bool exact_in_place = ExactInPlace;

  collective_operand() = delete;

  constexpr collective_operand(const T *const input,
                               const std::size_t input_count, T *const output,
                               const std::size_t output_count) noexcept
      : input_(input), output_(output), input_count_(input_count),
        output_count_(output_count) {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    if ((input_count_ != 0 && input_ == nullptr) ||
        (output_count_ != 0 && output_ == nullptr)) {
      return false;
    }
    const bool exact_alias = input_ == output_ && input_count_ == output_count_;
    if constexpr (ExactInPlace) {
      return exact_alias;
    } else {
      // Empty separate buffers may share the implementation's empty-span
      // sentinel because there is no byte range to alias.
      return !exact_alias || (input_count_ == 0 && output_count_ == 0);
    }
  }

  [[nodiscard]] constexpr const T *input_data() const noexcept {
    return input_;
  }
  [[nodiscard]] constexpr T *output_data() const noexcept { return output_; }
  [[nodiscard]] constexpr std::size_t input_count() const noexcept {
    return input_count_;
  }
  [[nodiscard]] constexpr std::size_t output_count() const noexcept {
    return output_count_;
  }
  [[nodiscard]] static constexpr bool in_place() noexcept {
    return ExactInPlace;
  }
  [[nodiscard]] constexpr bool separate_buffers_disjoint() const noexcept {
    if (input_count_ == 0 || output_count_ == 0) {
      return true;
    }
    lpfx::detail::ByteInterval input{};
    lpfx::detail::ByteInterval output{};
    return lpfx::detail::try_make_byte_interval(input_, input_count_, input) ==
               ErrorCode::success &&
           lpfx::detail::try_make_byte_interval(output_, output_count_,
                                                output) == ErrorCode::success &&
           lpfx::detail::disjoint(input, output);
  }

private:
  const T *input_ = nullptr;
  T *output_ = nullptr;
  std::size_t input_count_ = 0;
  std::size_t output_count_ = 0;
};

namespace detail {

template <class T> inline constexpr bool is_collective_operand = false;
template <LifetimeStartEligible T, class Space, bool ExactInPlace>
inline constexpr bool
    is_collective_operand<collective_operand<T, Space, ExactInPlace>> = true;

template <class T>
concept CollectiveOperand = is_collective_operand<std::remove_cvref_t<T>>;

} // namespace detail

template <LifetimeStartEligible T>
[[nodiscard]] constexpr collective_operand<T, HostSpace>
host_collective_buffers(const std::span<const T> input,
                        const std::span<T> output) noexcept {
  return {input.data(), input.size(), output.data(), output.size()};
}

// A const span can otherwise bind an owning temporary range. Reject that
// conversion at the factory boundary; explicit spans/raw pointers still carry
// the caller's ordinary backing-storage lifetime precondition.
template <LifetimeStartEligible T, class Input, class Output>
  requires((std::ranges::range<Input> && !std::ranges::borrowed_range<Input>) ||
           (std::ranges::range<Output> && !std::ranges::borrowed_range<Output>))
void host_collective_buffers(Input &&, Output &&) = delete;

template <LifetimeStartEligible T>
[[nodiscard]] constexpr collective_operand<T, HostSpace, true>
host_collective_in_place(const std::span<T> values) noexcept {
  return {values.data(), values.size(), values.data(), values.size()};
}

template <device::DeviceBackend Backend, LifetimeStartEligible T>
[[nodiscard]] constexpr auto
device_collective_buffers(const T *const input, const std::size_t input_count,
                          T *const output,
                          const std::size_t output_count) noexcept
    -> collective_operand<T, DeviceSpace<Backend>> {
  return {input, input_count, output, output_count};
}

template <device::DeviceBackend Backend, LifetimeStartEligible T>
[[nodiscard]] constexpr auto
device_collective_in_place(T *const values, const std::size_t count) noexcept
    -> collective_operand<T, DeviceSpace<Backend>, true> {
  return {values, count, values, count};
}

} // namespace lpfx::distributed

#endif
