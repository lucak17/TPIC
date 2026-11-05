#ifndef TPIC_CONFIG_OUTPUT_SELECTION_HPP
#define TPIC_CONFIG_OUTPUT_SELECTION_HPP

#include <tpic/config/output.hpp>
#include <tpic/data/output/output_moment_quantities.hpp>
#include <tpic/data/output/output_snapshot.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace tpic::config {

// Index of an array in the prepared field snapshot; no_slot marks an absent array.
using Slot = std::size_t;
inline constexpr Slot no_slot = std::numeric_limits<Slot>::max();

// One immutable selection drives deposition masks, conversion/packing slots,
// writer descriptors and snapshot sizing. It owns the mutable input's values.
class PreparedOutputSelection {
public:
  struct SpeciesSelection {
    std::uint16_t deposition_mask{};
    std::array<Slot, output_quantity_count> slots{};
  };

  explicit PreparedOutputSelection(const PreparedOutput &output);
  [[nodiscard]] const PreparedOutput &output() const noexcept {
    return output_;
  }
  [[nodiscard]] std::span<const data::OutputArrayDescriptor> arrays() const noexcept {
    return arrays_;
  }
  // Temporary conversion storage is needed only for a total contribution
  // whose species array is not already selected and available for reuse.
  [[nodiscard]] std::size_t scratch_component_count() const noexcept {
    return scratch_component_count_;
  }
  [[nodiscard]] std::size_t species_count() const noexcept {
    return species_.size();
  }
  [[nodiscard]] const SpeciesSelection &species(std::size_t index) const {
    return species_.at(index);
  }
  [[nodiscard]] const std::array<Slot, output_quantity_count> &
  total_slots() const noexcept {
    return totals_;
  }
  [[nodiscard]] Slot electric_slot() const noexcept { return electric_; }
  [[nodiscard]] Slot magnetic_slot() const noexcept { return magnetic_; }
  [[nodiscard]] Slot gauss_slot() const noexcept { return gauss_; }

private:
  const PreparedOutput output_;
  std::vector<data::OutputArrayDescriptor> arrays_;
  std::vector<SpeciesSelection> species_;
  std::array<Slot, output_quantity_count> totals_{};
  std::size_t scratch_component_count_{};
  Slot electric_{no_slot}, magnetic_{no_slot}, gauss_{no_slot};
};

} // namespace tpic::config

#endif // TPIC_CONFIG_OUTPUT_SELECTION_HPP
