#ifndef TPIC_SIMULATION_REPORTING_STARTUP_REPORT_HPP
#define TPIC_SIMULATION_REPORTING_STARTUP_REPORT_HPP

#include <tpic/simulation/planning/run_plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tpic::simulation {

// What a finished simulation reports: completed steps and the sample counts
// and files of its diagnostics, snapshots and checkpoints.
struct RunResult {
  int steps_completed{};
  std::uint64_t diagnostic_samples{};
  std::string diagnostics_path;
  std::uint64_t gauss_samples{};
  std::string gauss_path;
  std::uint64_t field_samples{};
  std::string fields_path;
  std::uint64_t particle_samples{};
  std::string particles_path;
  std::uint64_t checkpoints{};
  std::string checkpoint_path;
};

// One rank's startup memory estimate (memory_estimator.hpp computes it). Every
// member is a std::size_t, so the record has no padding bytes; the startup
// report gathers it as bytes.
struct MemoryEstimate {
  // Native pitched/aligned owner plans and reduction plans; no GPU allocation.
  std::size_t particle_device_bytes{}, field_device_bytes{}, mass_device_bytes{};
  std::size_t solver_device_bytes{}, centering_device_bytes{}, sort_device_bytes{};
  std::size_t diagnostics_device_bytes{}, diagnostics_pinned_host_bytes{};
  // TPIC-owned transfer endpoints, excluding LPFx transport staging and native
  // allocation/page overhead. Cold staging is reported separately from reuse.
  std::size_t persistent_pinned_host_bytes{};
  // Field staging is eager. Particle pinned/owned images appear on the first
  // capture and retain their observed high-water marks. Output totals below
  // describe the initial active population, not eager owner-capacity storage.
  std::size_t output_initial_pinned_host_bytes{}, output_pinned_host_bytes{};
  std::size_t output_owned_host_bytes{};
  std::size_t startup_pinned_host_estimate_bytes{};
  std::size_t validation_device_bytes{}, validation_host_bytes{};
  std::size_t output_device_bytes{};
  // A subset of output_device_bytes, excluding the reusable moment field owner.
  std::size_t output_staging_device_bytes{};
  std::size_t persistent_device_bytes{};
  // Initial communication images plus a conservative metadata allowance.
  // Growth is a separate hypothetical simultaneous replacement scenario, not
  // reserved storage or a startup requirement.
  std::size_t transport_device_bytes{}, growth_reserve_bytes{};
  // Informational subset of transport: cold explicit final-arrival/received
  // classification reservation bounded by the initial migration seed, alongside
  // the main owners. The logical final bound does not imply eager allocation.
  std::size_t migration_reserved_device_bytes{};
  std::size_t runtime_reserve_bytes = 64 * 1024 * 1024;
  std::size_t estimated_device_bytes{};
  // Concrete host object storage: the stream registry, the deposition ledger,
  // the host channels and fences, and the worker objects with their mailboxes,
  // all at the compiled species capacity.
  // Worker-local timing is separate. Adapter-specific
  // wrappers (including the output writer), allocator bookkeeping and native
  // thread stacks are excluded. These numbers must not be presented as total
  // process RSS. The worker count includes the output and checkpoint writers when
  // enabled; only the pipeline workers and the main thread own a StageTiming.
  std::size_t runtime_host_bytes{}, timing_host_bytes{};
  std::size_t worker_count{}, stream_count{};
  std::size_t validation_count{}, dependency_event_count{};
  std::size_t timing_event_count{};
};

// One rank's memory sample, gathered as bytes on rank 0 by both memory
// reports. Every member is eight bytes wide, so the record has no padding and
// its bytes are fully defined. The estimate is zero where it is not valid.
struct RankMemoryReport {
  std::uint64_t local_rank{};
  std::uint64_t device{};
  std::array<std::uint64_t, 2> device_uuid{};
  std::uint64_t memory_valid{};
  std::uint64_t free_bytes{};
  std::uint64_t total_bytes{};
  std::uint64_t estimate_valid{};
  MemoryEstimate estimate{};
};
static_assert(std::is_trivially_copyable_v<RankMemoryReport>,
              "the memory report is gathered as bytes");
static_assert(std::has_unique_object_representations_v<RankMemoryReport>,
              "the memory report has no padding bytes");

// Console reports of startup and completion, host-compiled. Every collective
// has run before the call; these functions only format.
void print_input_check(const PreparedInput& input, int rank);
void print_configuration(const PreparedInput& input, std::string_view path, RunMode mode,
                         std::string_view device_backend, std::string_view payload_backend,
                         const std::optional<data::PreparedRestart>& restart);
void print_memory_plan(const std::vector<RankMemoryReport>& reports);
void print_initialized_memory(const std::vector<RankMemoryReport>& reports);
void print_completion(const RunResult& result, const PreparedInput& input, RunMode mode);
// One error line on stderr: a message that starts with "TPIC " as it is, any
// other after "TPIC: ", followed by the rank when one is given.
void print_error(std::string_view message, int rank = -1) noexcept;

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_REPORTING_STARTUP_REPORT_HPP
