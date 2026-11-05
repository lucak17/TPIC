#include <tpic/config/output_selection.hpp>

#include <tpic/config/detail/errors.hpp>

#include <algorithm>
#include <string>
#include <utility>

namespace tpic::config {

namespace {

constexpr detail::Errors errors{"output"};

// Field arrays have one or three components: the selection checks this once,
// at compile time, for the moment table and uses the literal 3 for E and B.
static_assert([] {
  for (const auto& quantity : data::output_moment_quantities)
    if (quantity.components != 1 && quantity.components != 3) return false;
  return true;
}(), "output moment quantities have one or three components");

} // namespace

PreparedOutputSelection::PreparedOutputSelection(const PreparedOutput &output)
    : output_(output) {
  // The one check of a selection's species count; prepare_output sizes every
  // mask from the admitted count, so it guards hand-built records.
  const auto count = output.species_selected.front().size();
  errors.require(count != 0 && count <= static_cast<std::size_t>(max_real_species),
                 "field output requires species masks for 1 to " +
                     std::to_string(max_real_species) + " species");
  for (const auto &mask : output.species_selected)
    errors.require(mask.size() == count,
                   "field output species masks have different species counts");
  species_.resize(count);
  totals_.fill(no_slot);
  for (auto &species : species_)
    species.slots.fill(no_slot);
  if (!output.fields_enabled())
    return;
  const auto add = [&](std::string name, std::size_t components, bool cell = false) {
    const auto slot = arrays_.size();
    arrays_.push_back({std::move(name), static_cast<int>(components), cell});
    return slot;
  };
  if (output.output_e)
    electric_ = add("E", 3);
  if (output.output_b)
    magnetic_ = add("B", 3);
  if (output.output_gauss)
    gauss_ = add("Gauss", 1, true);
  for (std::size_t s = 0; s < count; ++s) {
    auto &species = species_[s];
    const auto suffix = "_" + std::to_string(s);
    for (std::size_t q = 0; q < output_quantity_count; ++q) {
      const auto quantity = data::output_moment_quantities[q];
      const bool individual = output.species_selected[q][s];
      const bool total = output.total_selected[q];
      if (individual)
        species.slots[q] =
            add(std::string(quantity.name) + suffix, quantity.components);
      else if (total)
        scratch_component_count_ = std::max(scratch_component_count_, quantity.components);
      if (individual || total)
        species.deposition_mask |= quantity.mask();
    }
  }
  for (std::size_t q = 0; q < output_quantity_count; ++q) {
    const auto quantity = data::output_moment_quantities[q];
    if (output.total_selected[q])
      totals_[q] = add(std::string(quantity.name), quantity.components);
  }
}

} // namespace tpic::config
