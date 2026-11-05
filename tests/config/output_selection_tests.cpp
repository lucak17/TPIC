#include <tpic/config/output.hpp>
#include <tpic/config/output_selection.hpp>

#include "../support/golden_data.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using tpic::config::PreparedOutput;
using tpic::config::PreparedOutputSelection;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
PreparedOutput fixture() {
  PreparedOutput output;
  output.field_cycle = 1;
  output.species_selected.fill(std::vector<bool>(3, false));
  return output;
}
void selective_layout() {
  auto output = fixture();
  output.output_e = true;
  output.species_selected[tpic::config::output_rho_quantity][0] = output.species_selected[tpic::config::output_rho_quantity][2] = output.total_selected[tpic::config::output_rho_quantity] = true;
  output.species_selected[tpic::config::output_current_quantity][1] = output.total_selected[tpic::config::output_current_quantity] = true;
  output.species_selected[tpic::data::first_pressure_quantity + 0][0] = output.total_selected[tpic::data::first_pressure_quantity + 0] = true;
  output.species_selected[tpic::data::first_pressure_quantity + 1][2] = output.species_selected[tpic::data::first_pressure_quantity + 5][1] = true;
  output.total_selected[tpic::data::first_pressure_quantity + 5] = true;
  const PreparedOutputSelection selection(output);
  const char *expected[] = {"E",     "rho_0", "PXX_0", "J_1", "PZZ_1", "rho_2",
                            "PXY_2", "rho",   "J",     "PXX", "PZZ"};
  require(selection.arrays().size() == std::size(expected),
          "selected array count changed");
  for (std::size_t i = 0; i < std::size(expected); ++i)
    require(selection.arrays()[i].name == expected[i],
            "species/quantity output order changed");
  require(selection.scratch_component_count() == 3,
          "unselected species current contributions lost vector scratch");
  require(selection.electric_slot() == 0 &&
              selection.magnetic_slot() == tpic::config::no_slot,
          "field slot selection changed");
  require(selection.species(0).slots[0] == 1 &&
              selection.species(0).slots[2] == 2 &&
              selection.species(1).slots[1] == 3 &&
              selection.species(2).slots[3] == 6 &&
              selection.total_slots()[0] == 7 &&
              selection.total_slots()[7] == 10,
          "species and total output mapping changed");
  // Every total requires all species, while PXY_2 adds exactly one source bit.
  require(
      selection.species(0).deposition_mask == 0x21f &&
          selection.species(1).deposition_mask == 0x21f &&
          selection.species(2).deposition_mask == 0x23f,
      "selected deposition mask does not cover individual arrays and totals");
  output.species_selected[tpic::config::output_rho_quantity][0] = false;
  output.total_selected[tpic::data::first_pressure_quantity + 5] = false;
  require(selection.output().species_selected[tpic::config::output_rho_quantity][0] &&
              selection.output().total_selected[tpic::data::first_pressure_quantity + 5],
          "prepared selection borrowed mutable configuration");
}
void totals_and_disabled() {
  auto output = fixture();
  output.total_selected[tpic::config::output_rho_quantity] = output.total_selected[tpic::config::output_current_quantity] = true;
  for (std::size_t p = 0; p < tpic::config::output_quantity_count - tpic::data::first_pressure_quantity; ++p)
    output.total_selected[tpic::data::first_pressure_quantity + p] = true;
  const PreparedOutputSelection totals(output);
  require(totals.arrays().size() == 8,
          "totals-only output allocates species arrays");
  for (std::size_t s = 0; s < totals.species_count(); ++s) {
    require(totals.species(s).deposition_mask == 0x3ff,
            "total omitted species deposition");
    for (const auto slot : totals.species(s).slots)
      require(slot == tpic::config::no_slot,
              "totals-only output retained individual slot");
  }
  output.field_cycle = 0;
  const PreparedOutputSelection disabled(output);
  require(disabled.arrays().empty() &&
              disabled.species(0).deposition_mask == 0 &&
              disabled.scratch_component_count() == 0,
          "disabled output retained active packing/deposition selection");
  output.species_selected[tpic::data::first_pressure_quantity + 0].pop_back();
  try {
    static_cast<void>(PreparedOutputSelection(output));
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("malformed species mask was accepted");
}
void minimal_scratch() {
  auto output = fixture();
  output.output_e = output.output_b = true;
  require(PreparedOutputSelection(output).scratch_component_count() == 0,
          "E/B output allocated unused accumulation scratch");
  output.total_selected[tpic::config::output_rho_quantity] = true;
  require(PreparedOutputSelection(output).scratch_component_count() == 1,
          "scalar total did not select exactly one temporary component");
  output.species_selected[tpic::config::output_rho_quantity] = {true, true, true};
  output.species_selected[tpic::config::output_current_quantity][0] = true;
  require(PreparedOutputSelection(output).scratch_component_count() == 0,
          "packed density contributions were not reused");
  output.total_selected[tpic::config::output_current_quantity] = true;
  require(PreparedOutputSelection(output).scratch_component_count() == 3,
          "partially selected species current lost temporary vector storage");
  output.species_selected[tpic::config::output_current_quantity] = {true, true, true};
  require(PreparedOutputSelection(output).scratch_component_count() == 0,
          "packed current contributions were not reused");
  output.total_selected[tpic::data::first_pressure_quantity + 5] = true;
  require(PreparedOutputSelection(output).scratch_component_count() == 1,
          "pressure total reserved more than its scalar temporary");
  output.species_selected[tpic::data::first_pressure_quantity + 5] = {true, true, true};
  require(PreparedOutputSelection(output).scratch_component_count() == 0,
          "packed pressure contributions were not reused");
}

// Tag selection code: B, E, rho_tot and J_tot flags, then the species masks of
// rho, J and the six pressure components, then the six pressure totals.
std::string bits(const std::vector<bool> &mask) {
  std::string text;
  for (const bool selected : mask)
    text += selected ? '1' : '0';
  return text;
}
std::string encode(const PreparedOutput &output) {
  std::string text{output.output_b ? '1' : '0', output.output_e ? '1' : '0',
                   output.total_selected[tpic::config::output_rho_quantity] ? '1' : '0',
                   output.total_selected[tpic::config::output_current_quantity] ? '1' : '0'};
  text += " rho=" + bits(output.species_selected[tpic::config::output_rho_quantity]) + " J=" + bits(output.species_selected[tpic::config::output_current_quantity]) + " P=";
  for (std::size_t component = 0; component < tpic::config::output_quantity_count - tpic::data::first_pressure_quantity; ++component)
    text += (component ? "," : "") + bits(output.species_selected[tpic::data::first_pressure_quantity + component]);
  text += " Ptot=";
  for (std::size_t component = 0; component < tpic::config::output_quantity_count - tpic::data::first_pressure_quantity; ++component)
    text += output.total_selected[tpic::data::first_pressure_quantity + component] ? '1' : '0';
  return text;
}
std::string selection_code(int ns, const std::string &field_tag, const std::string &moments_tag) {
  tpic::config::SimulationParameters parameters;
  parameters.ns = ns;
  parameters.output.field_tag = field_tag;
  parameters.output.moments_tag = moments_tag;
  try {
    return encode(tpic::config::prepare_output(parameters));
  } catch (const std::overflow_error &) {
    return "overflow";
  } catch (const std::invalid_argument &) {
    return "rejected";
  }
}
// Derived by hand from the tag grammar: '+'-separated tokens, each matched
// whole; token+digits selects an in-range species; a bare token selects every
// species unless an in-range index of it was selected; any other token is
// rejected; out-of-int indices overflow; empty field and moment tags select
// the defaults.
void hand_checked_tags() {
  const std::string zeros = " P=000,000,000,000,000,000 Ptot=000000";
  struct Case {
    int ns;
    const char *field, *moments;
    std::string expected;
  };
  const Case cases[] = {
      {3, "", "", "1111 rho=111 J=111" + zeros},
      {3, "none", "", "rejected"},
      {3, "B+E", "rho_tot", "1110 rho=000 J=000" + zeros},
      {3, "surroundBtextE", "", "rejected"},
      {3, "B+", "", "rejected"},
      {3, "", "rho", "0000 rho=111 J=000" + zeros},
      {3, "", "rho1", "0000 rho=010 J=000" + zeros},
      {3, "", "rho1rho", "rejected"},
      {3, "", "rho1+rho", "0000 rho=010 J=000" + zeros},
      {3, "", "rho5", "0000 rho=000 J=000" + zeros},
      {3, "", "rho5rho", "rejected"},
      {3, "", "rho5+rho", "0000 rho=111 J=000" + zeros},
      {3, "", "rho-1", "rejected"},
      {3, "", "prerho2suffix", "rejected"},
      {3, "", "rho00000000000000000000000000002", "0000 rho=001 J=000" + zeros},
      {3, "", "rho_tot+rho", "0010 rho=111 J=000" + zeros},
      {3, "", "RHO", "rejected"},
      {3, "", "j", "rejected"},
      {3, "", "J_tot", "0001 rho=000 J=000" + zeros},
      {3, "", "J_totJ2", "rejected"},
      {3, "", "J_tot+J2", "0001 rho=000 J=001" + zeros},
      {3, "", "rhoJ", "rejected"},
      {3, "", "rho+J", "0000 rho=111 J=111" + zeros},
      {3, "", "+rho", "rejected"},
      {3, "", "PXY1", "0000 rho=000 J=000 P=000,010,000,000,000,000 Ptot=000000"},
      {3, "", "PXX_tot", "0000 rho=000 J=000 P=000,000,000,000,000,000 Ptot=100000"},
      {3, "", "PXX_totally", "rejected"},
      {1, "E", "rho0", "0100 rho=1 J=0 P=0,0,0,0,0,0 Ptot=000000"},
      {10, "", "J2147483647", "0000 rho=0000000000 J=0000000000 P=0000000000,0000000000,"
                              "0000000000,0000000000,0000000000,0000000000 Ptot=000000"},
      {3, "", "rho2147483648", "overflow"},
      {3, "", "PZZ99999999999999999999", "overflow"},
      {3, "", "rho2147483648+rhoJ", "overflow"},
      {3, "B", "rhoJ+rho2147483648", "rejected"}};
  for (const auto &item : cases)
    if (selection_code(item.ns, item.field, item.moments) != item.expected)
      throw std::runtime_error(std::string("output tag selection changed: FieldOutputTag=") +
                               item.field + ", MomentsOutputTag=" + item.moments);
}
// 2000 seeded tag fixtures (ns 1-10). mt19937 with modulo selection gives the
// same fixtures with every standard library.
tpic::test::golden::Lines seeded_tags() {
  std::mt19937 generator(8128);
  const std::array<std::string, 30> fragments{
      "", "+", "rho", "rho0", "rho2", "rho99", "rho_tot", "J", "J1", "J_tot",
      "PXX", "PXX0", "PXX99", "PXX_tot", "PXY1", "PXZ", "PYY4", "PYZ", "PZZ_tot",
      "RHO", "j", "something", "rho00000000000000000000000000002", "J2147483647",
      "rho2147483648", "rho-1", "PXX_totally", "prerho2suffix",
      "PZZ99999999999999999999", "rhoJ"};
  const std::array<std::string, 5> fields{"", "none", "B", "B+E", "surroundBtextE"};
  tpic::test::golden::Lines lines;
  for (int ns = 1; ns <= 10; ++ns) {
    for (int fixture = 0; fixture != 200; ++fixture) {
      const auto field = fields[generator() % fields.size()];
      std::string moments;
      for (int fragment = 0; fragment != fixture % 6; ++fragment)
        moments += fragments[generator() % fragments.size()];
      lines.push_back(std::to_string(ns) + ' ' + std::to_string(fixture) + ' ' +
                      selection_code(ns, field, moments));
    }
  }
  return lines;
}
} // namespace
int main(int argc, char **argv) {
  try {
    const auto options = tpic::test::golden::parse_options(argc, argv);
    hand_checked_tags();
    tpic::test::golden::check(options, "output_tag_masks.txt", seeded_tags());
    selective_layout();
    totals_and_disabled();
    minimal_scratch();
    static_assert(tpic::data::output_moment_quantities[1].first_component == 1 &&
                  tpic::data::output_moment_quantities[1].components == 3);
    static_assert(tpic::data::output_moment_quantities[7].first_component == 9 &&
                  tpic::data::output_moment_quantities[7].components == 1);
    require(std::abs(tpic::data::output_moment_quantities[0].scale -
                     4 * std::acos(-1.)) < 1.e-14,
            "rho lost 4*pi output normalization");
    for (std::size_t q = 1; q < tpic::data::output_moment_quantities.size();
         ++q)
      require(tpic::data::output_moment_quantities[q].scale == 1,
              "current/raw pressure were rescaled");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
