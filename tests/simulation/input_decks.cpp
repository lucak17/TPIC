// Host preparation of every shipped deck in input/ and performance/scaling/: admission and
// warning count per deck, compared with tests/data/golden/input_decks.txt. The
// table describes the decks, so it leaves out the one warning that depends on
// the build: checkpoints requested in a build without a checkpoint backend.
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/simulation/planning/run_plan.hpp>

#include "../support/golden_data.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace golden = tpic::test::golden;

std::string prepare(const std::filesystem::path& path) {
  try {
    const auto plan = tpic::simulation::prepare_input(
        tpic::config::load_simulation_parameters(path.string()));
    const auto deck_warnings = std::count_if(plan.warnings.begin(), plan.warnings.end(),
        [](const std::string& warning) {
          return !warning.starts_with("RestartOutputCycle: this build has no checkpoint backend");
        });
    return std::string(plan.host_admitted() ? "admitted" : "refused") +
           " warnings=" + std::to_string(deck_warnings);
  } catch (const std::exception&) {
    return "error";
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = golden::parse_options(argc, argv);
    golden::Lines lines;
    for (const auto& deck : golden::deck_files(options.source_root))
      lines.push_back(deck.generic_string() + ' ' + prepare(options.source_root / deck));
    golden::check(options, "input_decks.txt", lines);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
