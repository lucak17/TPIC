#ifndef LPFX_API_DISTRIBUTED_KEY_HPP
#define LPFX_API_DISTRIBUTED_KEY_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/fixed_string.hpp>
#include <lpfx/distributed/detail/integer_codec.hpp>

namespace lpfx::distributed::detail {

// Names are cold registration metadata. Normalize structural strings before
// they enter concrete types (including CUDA host-stub signatures).
// cudafe++ drops consteval from lambdas when emitting host code, and EDG (nvcc)
// lacks P2564; a named consteval function keeps the loop index inside an
// immediate-function context for both front ends.
template <class Name, std::uint64_t... Indices>
[[nodiscard]] consteval std::uint64_t operation_identity() noexcept {
  std::array<std::uint64_t, 10 + sizeof...(Indices)> words{};
  words[0] = Name::name.size();
  for (std::size_t i = 0; i != 8; ++i)
    words[i + 1] = field::detail::pack_name_chunk(Name::name, i);
  words[9] = sizeof...(Indices);
  [[maybe_unused]] std::size_t index = 10;
  ((words[index++] = Indices), ...);
  return lpfx::detail::hash_words(words);
}

template <class Name, std::uint64_t... Indices> struct OperationKey final {
  static constexpr auto name = Name::name;
  static_assert(field::detail::valid_name<name>());
  static constexpr std::uint64_t identity =
      operation_identity<Name, Indices...>();
};

template <field::fixed_string Name, std::uint64_t... Indices>
using operation_key = OperationKey<field::detail::name_tag_t<Name>, Indices...>;

// Supplied by the native opt-in. Request declarations and the public facade
// remain independent of MPI and vendor headers.
template <class Setup> struct ObjectPreparation;

} // namespace lpfx::distributed::detail
#endif
