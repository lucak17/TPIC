#include <tpic/simulation/execution/deposition_ledger.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tpic::simulation;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class Function> void rejected(Function&& function, const char* message) {
  try { function(); } catch (const std::logic_error&) { return; }
  throw std::runtime_error(message);
}
template <class Ledger> void complete(Ledger& ledger) {
  for (const auto quantity : {MomentQuantity::current, MomentQuantity::mass})
    while (auto job = ledger.next(quantity)) ledger.submit(*job);
}

void readiness_and_coverage() {
  DepositionLedger<3, true> ledger{3};
  rejected([&] { ledger.offer({1, 0, DepositionPhase::resident, {0, 2, 2}}, 1); },
           "descriptor admitted without an epoch");
  ledger.begin(1, true);
  // Ready pass precedes species tie breaking. Empty residents/tails are real
  // phases and still need both quantity acknowledgements.
  ledger.offer({1, 2, DepositionPhase::resident, {0, 0, 0}}, 1);
  ledger.offer({1, 1, DepositionPhase::tail, {2, 1, 3}}, 2);
  ledger.offer({1, 1, DepositionPhase::resident, {0, 2, 2}}, 2);
  ledger.offer({1, 0, DepositionPhase::resident, {0, 3, 3}}, 2);
  ledger.offer({1, 0, DepositionPhase::tail, {3, 0, 3}}, 2);
  ledger.offer({1, 2, DepositionPhase::tail, {0, 4, 4}}, 3);
  const unsigned species[] = {2, 0, 0, 1, 1, 2};
  const DepositionPhase phase[] = {DepositionPhase::resident, DepositionPhase::resident,
      DepositionPhase::tail, DepositionPhase::resident, DepositionPhase::tail, DepositionPhase::tail};
  for (unsigned index = 0; index < 6; ++index) {
    const auto current = ledger.next(MomentQuantity::current);
    require(current && current->input.species == species[index] && current->input.phase == phase[index],
            "ready ordering or deterministic phase tie break changed");
    require(!ledger.submit(*current), "first quantity released a live host descriptor");
    rejected([&] { ledger.submit(*current); }, "duplicate quantity accepted");
  }
  require(!ledger.next(MomentQuantity::current) && !ledger.all_submitted(MomentQuantity::mass),
          "current submissions completed or consumed the mass queue");
  while (const auto mass = ledger.next(MomentQuantity::mass))
    require(ledger.submit(*mass), "second quantity retained a fully consumed descriptor");
  rejected([&] { ledger.begin(1, true); }, "generation reused");
  ledger.begin(2, false);
  require(!ledger.next(MomentQuantity::current), "terminal output-only epoch created solver work");
  rejected([&] { ledger.offer({2, 0, DepositionPhase::resident, {0, 0, 0}}, 1); },
           "terminal epoch accepted a descriptor");
}

void malformed_descriptors() {
  DepositionLedger<1, true> ledger{1};
  rejected([&] { ledger.begin(0, true); }, "zero generation accepted");
  ledger.begin(1, true);
  for (const auto input : {
           DepositionInput{2, 0, DepositionPhase::resident, {0, 2, 2}},
           DepositionInput{1, 1, DepositionPhase::resident, {0, 2, 2}},
           DepositionInput{1, 0, DepositionPhase::whole, {0, 2, 2}},
           DepositionInput{1, 0, DepositionPhase::resident, {1, 1, 2}},
           DepositionInput{1, 0, DepositionPhase::resident, {0, 1, 2}},
           DepositionInput{1, 0, DepositionPhase::tail, {3, 0, 2}}})
    rejected([&] { ledger.offer(input, 1); }, "malformed descriptor accepted");
  const DepositionInput resident{1, 0, DepositionPhase::resident, {0, 2, 2}};
  ledger.offer(resident, 1);
  rejected([&] { ledger.offer(resident, 2); }, "duplicate resident phase accepted");
  rejected([&] { ledger.offer({1, 0, DepositionPhase::tail, {1, 3, 4}}, 2); },
           "overlapping resident/tail ranges accepted");
  rejected([&] { ledger.offer({1, 0, DepositionPhase::tail, {3, 1, 4}}, 2); },
           "gap between resident/tail ranges accepted");
  ledger.offer({1, 0, DepositionPhase::tail, {2, 2, 4}}, 2);
  auto job = *ledger.next(MomentQuantity::current);
  ++job.input.generation;
  rejected([&] { ledger.submit(job); }, "stale job accepted");
  job = *ledger.next(MomentQuantity::current);
  ++job.input.range.count;
  rejected([&] { ledger.submit(job); }, "modified offered range accepted");
  complete(ledger);
}

void whole_modes_and_generations() {
  DepositionLedger<2, false> ledger{2};
  for (std::uint64_t generation = 1; generation <= 16; ++generation) {
    ledger.begin(generation, true);
    rejected([&] { ledger.offer({generation, 0, DepositionPhase::resident, {0, 1, 1}}, 1); },
             "whole-owner ordering accepted a resident phase");
    ledger.offer({generation, 1, DepositionPhase::whole, {0, 0, 0}}, 1);
    ledger.offer({generation, 0, DepositionPhase::whole, {0, 7, 7}}, 2);
    complete(ledger);
  }
}

// The species of the run, not the compiled capacity, fix the expected inputs.
void fewer_species_than_capacity() {
  DepositionLedger<3, false> ledger{2};
  ledger.begin(1, true);
  rejected([&] { ledger.offer({1, 2, DepositionPhase::whole, {0, 1, 1}}, 1); },
           "an unused species of the capacity was accepted");
  ledger.offer({1, 0, DepositionPhase::whole, {0, 4, 4}}, 1);
  require(!ledger.all_submitted(MomentQuantity::current), "one of two species completed the epoch");
  ledger.offer({1, 1, DepositionPhase::whole, {0, 0, 0}}, 1);
  complete(ledger);
  rejected([] { DepositionLedger<3, false> invalid{4}; }, "a species count above the capacity was accepted");
}
} // namespace

int main() {
  try {
    readiness_and_coverage();
    malformed_descriptors();
    whole_modes_and_generations();
    fewer_species_than_capacity();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "TPIC deposition ledger tests passed\n";
}
