#ifndef LPFX_DATA_PARTICLE_SLOTS_HPP
#define LPFX_DATA_PARTICLE_SLOTS_HPP

#include <array>
#include <cstddef>
#include <utility>

#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/data/particle/schema.hpp>

namespace lpfx::detail {

// Logical SoA slots of a particle owner: slot I < N holds attribute I and slot
// N is the reorder scratch. A reorder scatters each attribute into scratch and
// swaps the two slots. Forward sweeps visit attributes by descending width,
// so each vacated slot fits the next attribute; the next sweep retraces those
// swaps in reverse, which keeps every slot sufficient on later cycles.
// Slot has at least a std::byte *raw_base member.
template <ParticleSchemaType Schema, class Slot, class Count>
struct ParticleSlotCarousel final {
  static constexpr std::size_t attribute_count = Schema::attribute_count;

  std::array<Slot, attribute_count + 1> slots{};
  Count size{};
  Count capacity{};
  bool forward{true};

  [[nodiscard]] constexpr std::array<std::byte *, attribute_count>
  bases() const noexcept {
    std::array<std::byte *, attribute_count> result{};
    for (std::size_t attribute = 0; attribute != attribute_count;
         ++attribute) {
      result[attribute] = slots[attribute].raw_base;
    }
    return result;
  }

  // For every attribute I in the Forward or reverse sweep order, calls
  // function(meta::index_constant<I>{}) -> ErrorCode and then swaps slot I
  // with scratch. Stops at the first failure, leaving the failing slot and
  // every later one unswapped.
  template <bool Forward, class Function>
  [[nodiscard]] constexpr ErrorCode rotate(Function &&function) noexcept {
    return meta::for_each_until_error<attribute_count>(
        [&](const auto stage) noexcept {
          constexpr std::size_t I =
              (Forward ? Schema::forward_order
                       : Schema::reverse_order)[decltype(stage)::value];
          LPFX_TRY(function(meta::index_constant<I>{}));
          std::swap(slots[I], slots[attribute_count]);
          return ErrorCode::success;
        });
  }
};

} // namespace lpfx::detail

#endif
