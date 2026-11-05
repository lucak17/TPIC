#ifndef TPIC_SIMULATION_EXECUTION_DEPOSITION_LEDGER_HPP
#define TPIC_SIMULATION_EXECUTION_DEPOSITION_LEDGER_HPP

#include <tpic/simulation/execution/execution_layout.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace tpic::simulation {

// CPU-only exact-once accounting. Readiness order selects numerical work only;
// it never determines transport order. Each input supplies both quantities.
// Ns is the compiled species capacity, species the species of the run.
template <std::size_t Ns, bool Partial> class DepositionLedger {
public:
  explicit DepositionLedger(std::size_t species) : species_(species) {
    if (species == 0 || species > Ns)
      throw std::logic_error("TPIC deposition: species count outside the compiled capacity");
  }

  struct Job {
    DepositionInput input{};
    MomentQuantity quantity{};
  };

  void begin(std::uint64_t generation, bool moments) {
    if (generation <= generation_)
      throw std::logic_error("TPIC deposition: stale generation");
    generation_ = generation;
    moments_ = moments;
    entries_ = {};
    submitted_ = {};
    expected_ = moments ? species_ * (Partial ? 2 : 1) : 0;
  }

  void offer(DepositionInput input, std::uint64_t ready_pass) {
    if (!moments_ || input.generation != generation_ || input.species >= species_ ||
        input.phase > DepositionPhase::whole ||
        (Partial ? input.phase == DepositionPhase::whole
                 : input.phase != DepositionPhase::whole))
      throw std::logic_error("TPIC deposition: invalid input epoch/species/phase");
    if (input.range.first > input.range.total ||
        input.range.count != input.range.total - input.range.first ||
        (input.phase != DepositionPhase::tail && input.range.first != 0))
      throw std::logic_error("TPIC deposition: invalid particle range");
    auto& entry = entries_[index(input.species, input.phase)];
    if (entry.offered)
      throw std::logic_error("TPIC deposition: duplicate phase");
    if constexpr (Partial) {
      const auto& other = entries_[index(input.species,
          input.phase == DepositionPhase::resident ? DepositionPhase::tail
                                                   : DepositionPhase::resident)];
      if (other.offered) {
        const auto residents = input.phase == DepositionPhase::resident
            ? input.range.count : other.input.range.count;
        const auto tail_first = input.phase == DepositionPhase::tail
            ? input.range.first : other.input.range.first;
        if (residents != tail_first)
          throw std::logic_error("TPIC deposition: resident/tail coverage mismatch");
      }
    }
    entry.input = input;
    entry.ready_pass = ready_pass;
    entry.offered = true;
  }

  // A missing ready job is a value result; polling must not allocate on the heap.
  [[nodiscard]] std::optional<Job> next(MomentQuantity quantity) const {
    const auto q = static_cast<unsigned>(quantity);
    if (q >= 2) throw std::logic_error("TPIC deposition: invalid quantity");
    const Entry* best = nullptr;
    for (const auto& entry : entries_) {
      if (!entry.offered || (entry.submitted & (1U << q))) continue;
      if (!best || entry.ready_pass < best->ready_pass ||
          (entry.ready_pass == best->ready_pass &&
           (entry.input.species < best->input.species ||
            (entry.input.species == best->input.species &&
             entry.input.phase < best->input.phase))))
        best = &entry;
    }
    return best ? std::optional<Job>{{best->input, quantity}} : std::nullopt;
  }

  // True means both quantity submissions have relinquished the host descriptor.
  bool submit(const Job& job) {
    if (job.input.generation != generation_ || job.input.species >= species_ ||
        job.input.phase > DepositionPhase::whole)
      throw std::logic_error("TPIC deposition: stale job");
    auto& entry = entries_[index(job.input.species, job.input.phase)];
    const auto q = static_cast<unsigned>(job.quantity);
    if (!entry.offered || q >= 2 || (entry.submitted & (1U << q)) ||
        entry.input.phase != job.input.phase ||
        entry.input.range.first != job.input.range.first ||
        entry.input.range.count != job.input.range.count ||
        entry.input.range.total != job.input.range.total)
      throw std::logic_error("TPIC deposition: duplicate/late submission");
    entry.submitted |= 1U << q;
    ++submitted_[q];
    return entry.submitted == 3;
  }

  [[nodiscard]] bool all_submitted(MomentQuantity q) const {
    return submitted_[static_cast<unsigned>(q)] == expected_;
  }

private:
  struct Entry {
    DepositionInput input{};
    std::uint64_t ready_pass{};
    unsigned submitted{};
    bool offered{};
  };
  static constexpr std::size_t index(unsigned species, DepositionPhase phase) {
    return species * 3 + static_cast<unsigned>(phase);
  }
  const std::size_t species_;
  std::uint64_t generation_{};
  std::array<Entry, Ns * 3> entries_{};
  std::array<std::size_t, 2> submitted_{};
  std::size_t expected_{};
  bool moments_{};
};

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_EXECUTION_DEPOSITION_LEDGER_HPP
