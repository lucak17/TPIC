#ifndef LPFX_DATA_PARTICLE_LAYOUT_HPP
#define LPFX_DATA_PARTICLE_LAYOUT_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/data/particle/schema.hpp>
#include <lpfx/memory/host/allocator.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx {

template <detail::ParticleSchemaType Schema, class Space, bool Const>
class BasicParticleView;

template <detail::ParticleSchemaType Schema, class Space> class Particles;

namespace detail {

// Alignment shared by every SoA block of Schema in Space.
template <class Space, ParticleSchemaType Schema>
inline constexpr std::size_t particle_allocation_alignment =
    AllocatorTraits<Space>::minimum_data_alignment < Schema::alignment
        ? Schema::alignment
        : AllocatorTraits<Space>::minimum_data_alignment;

// One contiguous allocation of Blocks SoA blocks for capacity records, each
// starting at a multiple of allocation_alignment. Block I < attribute_count
// holds attribute I; block attribute_count, when present, is the reorder
// scratch sized for the widest attribute.
template <ParticleSchemaType Schema, class Space, std::size_t Blocks>
  requires(Blocks == Schema::attribute_count ||
           Blocks == Schema::attribute_count + 1)
class ParticleBlockLayout final {
public:
  using schema_type = Schema;
  using space_type = Space;

  static constexpr std::size_t attribute_count = Schema::attribute_count;
  static constexpr std::size_t block_count = Blocks;
  static constexpr std::size_t allocation_alignment =
      particle_allocation_alignment<Space, Schema>;

  [[nodiscard]] static constexpr std::size_t
  width(const std::size_t block) noexcept {
    return block < attribute_count ? Schema::widths[block]
                                   : Schema::widest_width;
  }

  constexpr ParticleBlockLayout() noexcept = default;

  [[nodiscard]] constexpr bool valid() const noexcept { return valid_; }
  [[nodiscard]] constexpr std::size_t capacity() const noexcept {
    return capacity_;
  }
  [[nodiscard]] constexpr std::size_t
  offset(const std::size_t block) const noexcept {
    return offsets_[block];
  }
  [[nodiscard]] constexpr std::size_t
  bytes(const std::size_t block) const noexcept {
    return capacity_ * width(block);
  }
  // Sum of all block bytes, excluding alignment gaps.
  [[nodiscard]] constexpr std::size_t logical_bytes() const noexcept {
    return capacity_ * (Schema::record_width +
                        (Blocks == attribute_count ? 0 : Schema::widest_width));
  }
  [[nodiscard]] constexpr std::size_t total_bytes() const noexcept {
    return offsets_[Blocks - 1] + bytes(Blocks - 1);
  }

  // Block bases inside an allocation of total_bytes(); null for a null base.
  [[nodiscard]] constexpr std::array<std::byte *, Blocks>
  bases(std::byte *const base) const noexcept {
    std::array<std::byte *, Blocks> result{};
    if (base != nullptr) {
      for (std::size_t block = 0; block != Blocks; ++block) {
        result[block] = base + offsets_[block];
      }
    }
    return result;
  }

private:
  constexpr ParticleBlockLayout(
      const std::size_t capacity,
      const std::array<std::size_t, Blocks> &offsets) noexcept
      : capacity_(capacity), offsets_(offsets), valid_(true) {}

  std::size_t capacity_{};
  std::array<std::size_t, Blocks> offsets_{};
  bool valid_{};

  friend struct PrivateConstructionAccess;
};

} // namespace detail

// Contiguous attribute blocks, used for transfer staging images.
template <detail::ParticleSchemaType Schema, class Space>
using PreparedParticleLayout =
    detail::ParticleBlockLayout<Schema, Space, Schema::attribute_count>;

// Host owner storage: the attribute blocks followed by the reorder scratch.
template <detail::ParticleSchemaType Schema>
using PreparedParticleStorage =
    detail::ParticleBlockLayout<Schema, HostSpace,
                                Schema::attribute_count + 1>;

// Every reported offset, size and total is bounded by the checked final end.
template <detail::ParticleSchemaType Schema, class Space, std::size_t Blocks,
          detail::checked_integer Capacity>
[[nodiscard]] constexpr ErrorCode try_prepare_particle_layout(
    const Capacity requested_capacity,
    detail::ParticleBlockLayout<Schema, Space, Blocks> &output) noexcept {
  using Layout = detail::ParticleBlockLayout<Schema, Space, Blocks>;
  if constexpr (std::signed_integral<Capacity>) {
    if (requested_capacity < 0) {
      return ErrorCode::invalid_argument;
    }
  }
  std::size_t capacity{};
  if (!detail::checked_integral_cast(requested_capacity, capacity)) {
    return ErrorCode::overflow;
  }
  std::array<std::size_t, Blocks> offsets{};
  std::size_t end = 0;
  for (std::size_t block = 0; block != Blocks; ++block) {
    std::size_t bytes = 0;
    if (!detail::checked_multiply(capacity, Layout::width(block), bytes) ||
        !detail::checked_align_up(end, Layout::allocation_alignment,
                                  offsets[block]) ||
        !detail::checked_add(offsets[block], bytes, end)) {
      return ErrorCode::overflow;
    }
  }
  output = detail::PrivateConstructionAccess::create<Layout>(capacity, offsets);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
