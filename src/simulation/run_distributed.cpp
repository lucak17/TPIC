#include <lpfx/distributed/runtime/error_reporting.hpp>
#include <lpfx/distributed/runtime/host_control.hpp>
#include <lpfx/distributed/runtime/device_runtime.hpp>
#include <tpic/communication/transport_routes.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp>
#include <tpic/io/checkpoint.hpp>
#include <tpic/simulation/driver/run_distributed.hpp>
#include <tpic/simulation/planning/run_plan.hpp>
#include <tpic/simulation/reporting/memory_estimator.hpp>
#include <tpic/simulation/reporting/startup_report.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tpic::simulation {

// Defined in driver/simulation_device.hpp and instantiated once per compiled
// precision pair by simulation_dispatch.cpp.
template <class Precision>
[[nodiscard]] RunResult run_simulation(const RunPlan& plan,
    const lpfx::distributed::CommunicationSetup& setup, lpfx::device::ActiveExecutor executor);

namespace {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
constexpr std::string_view device_backend_name = "CUDA";
#else
constexpr std::string_view device_backend_name = "HIP";
#endif

int close_runtime(auto& runtime) {
  const auto closed = runtime.close();
  if (closed.succeeded()) return 0;
  lpfx::distributed::print_runtime_error(closed, "TPIC startup: runtime close");
  if (runtime.active()) runtime.abort(1);
  return 1;
}

// The gather protocol of both memory reports: every rank samples the free
// device memory between two barriers, before any rank allocates further, and
// rank 0 receives one record per rank. Other ranks receive an empty vector.
std::vector<RankMemoryReport> gather_memory_reports(const lpfx::MpiCallDomain& domain,
                                                    RankMemoryReport local, const std::string& report) {
  using Traits = lpfx::device::ActiveBackendTraits;
  lpfx::check_mpi_error(lpfx::distributed::host_barrier(domain),
                        ("TPIC startup: " + report + " agreement failed").c_str());
  lpfx::device::DeviceMemoryInfo memory{};
  local.memory_valid = Traits::succeeded(Traits::query_current_device_memory(memory));
  local.free_bytes = memory.free_bytes;
  local.total_bytes = memory.total_bytes;
  std::vector<RankMemoryReport> reports(domain.rank() == 0 ? static_cast<std::size_t>(domain.size()) : 0);
  lpfx::check_mpi_error(lpfx::distributed::host_gather<std::byte>(
                            domain, std::as_bytes(std::span{&local, 1}),
                            std::as_writable_bytes(std::span{reports})),
                        ("TPIC startup: " + report + " gather failed").c_str());
  lpfx::check_mpi_error(lpfx::distributed::host_barrier(domain),
                        ("TPIC startup: " + report + " completion failed").c_str());
  return reports;
}

void report_memory_plan(const lpfx::MpiCallDomain& domain, int local_rank, int device,
                        const MemoryEstimate* estimate) {
  RankMemoryReport local{};
  local.local_rank = static_cast<std::uint64_t>(local_rank);
  local.device = static_cast<std::uint64_t>(device);
  lpfx::device::DeviceIdentity identity{};
  std::int32_t ordinal{};
  using Traits = lpfx::device::ActiveBackendTraits;
  static_assert(sizeof(identity.bytes) == sizeof(local.device_uuid));
  if (Traits::succeeded(Traits::query_current_device_identity(identity, ordinal)))
    std::memcpy(local.device_uuid.data(), identity.bytes.data(), identity.bytes.size());
  if (estimate) {
    local.estimate_valid = 1;
    local.estimate = *estimate;
  }
  const auto reports = gather_memory_reports(domain, local, "memory report");
  if (domain.rank() == 0) print_memory_plan(reports);
}

// Host admission remains on LPFx's MPI control path for every device route.
void agree(const lpfx::MpiCallDomain& domain, bool local, const char* message) {
  const int accepted = local ? 1 : 0;
  int all = 0;
  lpfx::check_mpi_error(
      lpfx::distributed::host_all_reduce<int, lpfx::distributed::reduction_operation::minimum>(
          domain, {&accepted, 1}, {&all, 1}),
      "TPIC startup: admission agreement failed");
  lpfx::require(all != 0, message);
}

// Replaces text by the root's.
void broadcast_text(const lpfx::MpiCallDomain& domain, std::string& text, const char* what) {
  int size = static_cast<int>(text.size());
  lpfx::check_mpi_error(lpfx::distributed::host_broadcast<int>(domain, {&size, 1}), what);
  text.resize(static_cast<std::size_t>(size));
  lpfx::check_mpi_error(lpfx::distributed::host_broadcast<char>(domain, std::span<char>{text}), what);
}

std::string agreed_input(const lpfx::MpiCallDomain& domain, std::string_view path, int rank) {
  std::string text, error;
  try {
    std::ifstream file{std::string(path), std::ios::binary};
    if (!file) throw std::runtime_error("TPIC input: unable to open input file " + std::string(path));
    text.assign(std::istreambuf_iterator<char>(file), {});
    if (file.bad()) throw std::runtime_error("TPIC input: unable to read input file " + std::string(path));
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("TPIC input: input file exceeds the MPI count range");
  } catch (const std::exception& exception) {
    error = exception.what();
    print_error(error, rank);
  }
  agree(domain, error.empty(), "TPIC startup: an input file could not be read");
  std::string rank0_text = text;
  broadcast_text(domain, rank0_text, "TPIC startup: input snapshot broadcast failed");
  agree(domain, text == rank0_text, "TPIC startup: input contents differ across ranks");
  // Parse these exact agreed bytes, without reopening the file. This establishes
  // identical species ordering, precision inputs and communication participation.
  return text;
}

// Runs on the root and agrees on its failure.
template <class Work>
void root_agreed(const lpfx::MpiCallDomain& domain, int rank, const char* message, Work&& work) {
  std::string error;
  if (rank == 0) {
    try {
      work();
    } catch (const std::exception& exception) {
      error = exception.what();
      print_error(error, rank);
    }
  }
  agree(domain, error.empty(), message);
}

// The checkpoint of a restart in RestartDirName: the root finds it and every
// rank receives it. Every rank evaluates the same agreed input.
data::CheckpointLocation agreed_checkpoint(const lpfx::MpiCallDomain& domain,
                                           const config::SimulationParameters& parameters, int rank) {
  data::CheckpointLocation checkpoint;
  root_agreed(domain, rank, "TPIC startup: no checkpoint to restart from", [&] {
    checkpoint = io::find_restart_checkpoint(config::prepare_output(parameters).checkpoint_dir);
  });
  lpfx::check_mpi_error(lpfx::distributed::host_broadcast<std::byte>(
                            domain, std::as_writable_bytes(std::span{&checkpoint.header, 1})),
                        "TPIC startup: checkpoint broadcast failed");
  broadcast_text(domain, checkpoint.directory, "TPIC startup: checkpoint broadcast failed");
  broadcast_text(domain, checkpoint.extension, "TPIC startup: checkpoint broadcast failed");
  return checkpoint;
}

} // namespace

void report_initialized_memory(const lpfx::MpiCallDomain& domain, int device) {
  RankMemoryReport local{};
  local.device = static_cast<std::uint64_t>(device);
  const auto reports = gather_memory_reports(domain, local, "allocated memory report");
  if (domain.rank() == 0) print_initialized_memory(reports);
}

int run_distributed(int argc, char** argv, std::string_view path, InputOptions options, RunMode mode) {
  lpfx::distributed::DeviceRuntime<lpfx::device::ActiveBackend> runtime;
  try {
    const auto initialized = runtime.initialize(
        argc, argv, {.thread_mode = lpfx::MpiThreadMode::multiple});
    if (!initialized.succeeded()) {
      lpfx::distributed::print_runtime_error(initialized, "TPIC startup: runtime initialization");
      if (runtime.active() && initialized.native_result() != MPI_SUCCESS)
        runtime.abort(1);
      if (runtime.active()) close_runtime(runtime);
      return 1;
    }
    const int rank = runtime.rank(), ranks = runtime.size();
    options.rank = rank;
    options.communicator_size = ranks;
    {
      const auto& domain = runtime.domain();
      // Input bytes cannot detect differently compiled executables in one MPI
      // job. Backend bootstrap is complete; agree storage/wire identities before
      // constructing simulation allocations or registering physics operations.
      enum CommandSlot : std::size_t {
        order_slot, mode_slot, precision_slot, overwrite_slot, mass_matrix_slot, payload_backend_slot,
        mass_apply_slot, restart_slot, command_slots
      };
      const std::array<int, command_slots> local_command{
          static_cast<int>(options.order), static_cast<int>(mode),
          static_cast<int>(config::precision_pair_index(options.precision)),
          int(options.overwrite_output), static_cast<int>(options.mass_matrix),
          communication::payload_backend_id, static_cast<int>(options.mass_apply), int(options.restart)};
      auto root_command = local_command;
      lpfx::check_mpi_error(lpfx::distributed::host_broadcast<int>(domain, root_command),
                            "TPIC startup: command agreement failed");
      const auto agreed = [&](CommandSlot slot) { return root_command[slot] == local_command[slot]; };
      agree(domain, agreed(order_slot) && agreed(mode_slot),
            "TPIC startup: command or particle order differs across ranks");
      agree(domain, agreed(precision_slot),
            "TPIC startup: selected precision pair differs across ranks");
      agree(domain, agreed(overwrite_slot),
            "TPIC startup: --overwrite-output differs across ranks");
      agree(domain, agreed(mass_matrix_slot),
            "TPIC startup: --mass-matrix differs across ranks");
      agree(domain, agreed(payload_backend_slot),
            "TPIC startup: compiled payload backend differs across ranks");
      agree(domain, agreed(mass_apply_slot),
            "TPIC startup: --mass-apply differs across ranks");
      agree(domain, agreed(restart_slot), "TPIC startup: --restart differs across ranks");
      std::istringstream snapshot{agreed_input(domain, path, rank)};
      std::unique_ptr<RunPlan> plan;
      std::string error;
      try {
        // Parsing fails on every rank or on none, so every rank enters the
        // checkpoint agreement or none does.
        auto parameters = config::parse_simulation_parameters(snapshot);
        std::optional<data::CheckpointLocation> checkpoint;
        if (options.restart) checkpoint = agreed_checkpoint(domain, parameters, rank);
        plan = std::make_unique<RunPlan>(RunPlan::prepare(std::move(parameters), options, mode,
            RuntimeDebugOptions::from_environment(), std::move(checkpoint)));
      } catch (const std::exception& exception) { error = exception.what(); }
      if (!error.empty())
        print_error(error, rank);
      agree(domain, error.empty(), "TPIC startup: input admission failed");
      const auto& input = plan->input;
      if (advances(mode)) {
        // The field, particle and scalar files go into SaveDirName, which exists before any rank opens one.
        root_agreed(domain, rank, "TPIC startup: cannot create SaveDirName", [&] {
          std::filesystem::create_directories(input.output.save_dir);
        });
        // Before any rank writes a checkpoint, none of an earlier run may mix with this run's.
        if (input.output.checkpoints_enabled())
          root_agreed(domain, rank, "TPIC startup: RestartDirName holds checkpoints of an earlier run", [&] {
            io::prepare_checkpoint_directory(input.output.checkpoint_dir,
                                             plan->restart ? &plan->restart->checkpoint : nullptr,
                                             options.overwrite_output);
          });
      }
      if (rank == 0)
        print_configuration(input, path, mode, device_backend_name, communication::payload_backend_name,
                            plan->restart);

      const auto& properties = runtime.properties();
      const int selected = runtime.device_ordinal();
      agree(domain, options.mass_matrix != config::MassDepositionKernel::matrix_core ||
                config::dispatch_precision(options.precision, [&]<class P> {
                  // The matrix kernels exist for double mass compute only.
                  if constexpr (std::is_same_v<typename P::Compute::Mass, double>)
                    return ecsim::matrix_core_mass_deposition_available<P>(selected);
                  else
                    return false;
                }),
            "TPIC startup: Matrix-core mass deposition requires a supported FP64 matrix device and compiled kernel image");
      // Reserves and transport allowances are estimates, not allocation limits:
      // report them without vetoing startup. An actual allocation failure surfaces
      // when the simulation owners are constructed.
      std::optional<MemoryEstimate> estimate;
      try {
        estimate = estimate_memory<lpfx::device::ActiveBackend>(*plan, properties);
      } catch (const std::exception& error) {
        // Even a conservative scenario's arithmetic can exceed its planning
        // bounds. Reporting must not prevent actual, smaller allocations.
        print_error("TPIC startup: memory estimate unavailable (" + std::string(error.what()) +
                    "); proceeding with actual allocations", rank);
      }
      report_memory_plan(domain, runtime.local_rank(), selected, estimate ? &*estimate : nullptr);
      // RunPlan::prepare admitted the pair; only compiled pairs have a run_simulation instantiation.
      const auto result = config::dispatch_precision(input.precision, [&]<class P> {
        if constexpr (config::precision_pair_compiled_v<P>)
          return run_simulation<P>(*plan, runtime.setup(), runtime.executor());
        else
          return RunResult{};
      });
      if (rank == 0) print_completion(result, input, mode);
    } // All simulation/communication owners are gone before runtime close.
    return close_runtime(runtime);
  } catch (const std::exception& exception) {
    print_error(exception.what(), runtime.rank());
    // A peer may be inside an accepted device exchange. Never let a local
    // failure continue or unwind through live LPFx communication on that peer.
    runtime.abort(1);
  }
}
} // namespace tpic::simulation
