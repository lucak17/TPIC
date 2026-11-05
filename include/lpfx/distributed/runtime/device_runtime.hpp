#ifndef LPFX_DISTRIBUTED_RUNTIME_DEVICE_RUNTIME_HPP
#define LPFX_DISTRIBUTED_RUNTIME_DEVICE_RUNTIME_HPP

#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>

#include <mpi.h>

#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/distributed/runtime/mpi_runtime.hpp>
#include <lpfx/distributed/transport/mpi/agreement.hpp>
#include <lpfx/distributed/transport/mpi/setup.hpp>
#include <lpfx/execution/device/context.hpp>

namespace lpfx::distributed {

enum class RuntimeStage : std::uint8_t {
  none, options, device_discovery, device_selection, stream_creation,
  stream_admission, mpi_initialization, mpi_bootstrap, thread_support,
  call_domain, local_topology, device_mapping, device_properties, device_memory,
  communication_setup, stream_destruction, mpi_finalization
};

[[nodiscard]] constexpr std::string_view
runtime_stage_name(const RuntimeStage stage) noexcept {
  constexpr std::array names{
      "none", "options", "device discovery", "device selection",
      "stream creation", "stream admission", "MPI initialization",
      "MPI bootstrap", "MPI thread support", "MPI call domain",
      "node-local topology", "device mapping", "device properties",
      "device memory", "communication setup", "stream destruction",
      "MPI finalization"};
  const auto index = static_cast<std::size_t>(stage);
  return index < names.size() ? names[index] : "unknown";
}

class RuntimeResult final {
public:
  constexpr RuntimeResult() noexcept = default;
  constexpr RuntimeResult(ErrorCode code, RuntimeStage stage,
                          int native = MPI_SUCCESS, int rank = -1) noexcept
      : code_(code), stage_(stage), native_(native), rank_(rank) {}
  [[nodiscard]] constexpr bool succeeded() const noexcept {
    return code_ == ErrorCode::success;
  }
  [[nodiscard]] constexpr ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] constexpr RuntimeStage stage() const noexcept { return stage_; }
  [[nodiscard]] constexpr int native_result() const noexcept { return native_; }
  [[nodiscard]] constexpr int failing_rank() const noexcept { return rank_; }
private:
  ErrorCode code_ = ErrorCode::success;
  RuntimeStage stage_ = RuntimeStage::none;
  int native_ = MPI_SUCCESS;
  int rank_ = -1;
};

// Safe before MPI initialization and after finalization: report only captured
// values, without calling MPI or consuming any backend error state.
inline void print_runtime_error(const RuntimeResult &result,
                                const char *context = "LPFx runtime") noexcept {
  if (result.succeeded()) return;
  const auto stage = runtime_stage_name(result.stage());
  char message[256];
  std::snprintf(message, sizeof(message),
      "lpfx=%s(%u) stage=%.*s native_result=%d failing_rank=%d",
      error_code_name(result.code()), static_cast<unsigned>(result.code()),
      static_cast<int>(stage.size()), stage.data(), result.native_result(), result.failing_rank());
  print_error(message, context);
}

// Ordinals are relative to the devices visible to this process; the runtime
// interprets CUDA/HIP visibility masks and UUIDs. An explicit device and an
// explicit local rank are mutually exclusive; with neither, a single visible
// device is used, otherwise the launcher's local rank selects one.
struct DevicePlacement final {
  int device_ordinal = -1;
  int local_rank = -1;
};

struct DeviceRuntimeOptions final {
  MpiThreadMode thread_mode = MpiThreadMode::funneled;
  DevicePlacement placement{};
};

namespace detail {

struct PreparedDevicePlacement final {
  int device_ordinal = -1;
  int local_rank_hint = -1;
  bool automatic = true;
};

[[nodiscard]] constexpr ErrorCode
try_parse_local_rank(const char *text, int &output) noexcept {
  if (text == nullptr || *text == '\0')
    return ErrorCode::invalid_argument;
  int value = 0;
  for (; *text != '\0'; ++text) {
    if (*text < '0' || *text > '9')
      return ErrorCode::invalid_argument;
    const int digit = *text - '0';
    if (value > (INT_MAX - digit) / 10)
      return ErrorCode::overflow;
    value = value * 10 + digit;
  }
  output = value;
  return ErrorCode::success;
}

// Failure leaves output unchanged. MPI launchers can themselves run within
// Slurm: their job-local ranks win over an inherited scheduler task ID, and
// disagreeing MPI launcher hints are rejected.
[[nodiscard]] inline ErrorCode
try_resolve_device_placement(const int visible, const DevicePlacement options,
                             PreparedDevicePlacement &output) noexcept {
  if (visible <= 0 || options.device_ordinal < -1 || options.local_rank < -1 ||
      (options.device_ordinal >= 0 && options.local_rank >= 0))
    return ErrorCode::invalid_argument;
  PreparedDevicePlacement prepared{};
  if (options.device_ordinal >= 0) {
    if (options.device_ordinal >= visible)
      return ErrorCode::invalid_argument;
    prepared.device_ordinal = options.device_ordinal;
    prepared.automatic = false;
  } else if (options.local_rank >= 0) {
    prepared.local_rank_hint = options.local_rank;
    prepared.device_ordinal = options.local_rank % visible;
  } else if (visible == 1) {
    // A launcher may expose only its assigned GPU, whatever its local rank.
    prepared.device_ordinal = 0;
  } else {
    const std::array<const char *, 3> hints{
        std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"),
        std::getenv("MPI_LOCALRANKID"), std::getenv("SLURM_LOCALID")};
    int local_rank = -1;
    for (std::size_t index = 0; index != hints.size(); ++index) {
      if (hints[index] == nullptr || (index == 2 && local_rank >= 0))
        continue;
      int parsed = -1;
      LPFX_TRY(try_parse_local_rank(hints[index], parsed));
      if (local_rank >= 0 && local_rank != parsed)
        return ErrorCode::invalid_argument;
      local_rank = parsed;
    }
    if (local_rank < 0)
      return ErrorCode::invalid_argument;
    prepared.local_rank_hint = local_rank;
    prepared.device_ordinal = local_rank % visible;
  }
  output = prepared;
  return ErrorCode::success;
}

} // namespace detail

// Cold process owner. The stable domain, initial stream and setup are borrowed
// by consumers; they add no wrapper to communication or kernel submission.
// initialize/close are collective main-thread boundaries. Failure after MPI
// starts leaves active()==true so all ranks can explicitly close, or abort on
// a native MPI failure. Destruction never hides a collective finalization.
template <class Backend>
  requires device::DeviceStreamBackend<Backend> &&
           device::DeviceDiscoveryBackend<Backend>
class DeviceRuntime final {
  using Traits = device::BackendTraits<Backend>;
  using Stream = device::DeviceStream<Backend>;
public:
  DeviceRuntime() noexcept = default;
  LPFX_NONMOVABLE(DeviceRuntime);
  ~DeviceRuntime() noexcept {
    if (active()) std::abort();
  }

  [[nodiscard]] RuntimeResult initialize(int &argc, char **&argv,
      const DeviceRuntimeOptions options = {}) noexcept {
    if (active() || ready_ || domain_ || !stream_.empty())
      return {ErrorCode::invalid_state, RuntimeStage::options};
    const auto available = MpiRuntime::can_initialize();
    if (!available.succeeded())
      return {available.code(), RuntimeStage::mpi_initialization,
              available.native_result()};

    const bool valid_mode = options.thread_mode >= MpiThreadMode::single &&
                           options.thread_mode <= MpiThreadMode::multiple;
    // Keep pre-MPI errors local until every rank can participate in rejection.
    RuntimeResult local = valid_mode ? prepare_device(options.placement)
        : RuntimeResult{ErrorCode::invalid_argument, RuntimeStage::options};
    const auto started = mpi_.initialize(argc, argv,
        valid_mode ? options.thread_mode : MpiThreadMode::funneled);
    if (!started.succeeded())
      return {started.code(), RuntimeStage::mpi_initialization,
              started.native_result()};

    int native = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (native == MPI_SUCCESS) native = MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    if (native == MPI_SUCCESS) native = MPI_Comm_size(MPI_COMM_WORLD, &size_);
    if (native != MPI_SUCCESS)
      local = {ErrorCode::backend_failure, RuntimeStage::mpi_bootstrap, native};
    RuntimeResult result = agree(local);
    if (!result.succeeded()) return result;

    MpiThreadMode provided{};
    const bool mapped = lpfx::detail::try_map_thread_mode(
        mpi_.provided_thread_level(), provided);
    result = agree({mapped && provided >= options.thread_mode
                        ? ErrorCode::success : ErrorCode::invalid_state,
                    RuntimeStage::thread_support});
    if (!result.succeeded()) return result;

    domain_.emplace();
    const auto admitted = try_make_mpi_call_domain(MPI_COMM_WORLD,
        provided == MpiThreadMode::single
            ? MpiProcessThreadUse::single_process_thread
            : MpiProcessThreadUse::other_process_threads_may_exist, *domain_);
    result = agree({admitted.code(), RuntimeStage::call_domain,
                    admitted.native_result()});
    if (!result.succeeded()) return result;

    const auto topology = try_query_mpi_local_topology(*domain_, local_);
    result = agree({topology.code(), RuntimeStage::local_topology,
                    topology.native_result()});
    if (!result.succeeded()) return result;
    const bool mapping =
        (!placement_.automatic ||
         placement_.device_ordinal == local_.rank % visible_) &&
        (placement_.local_rank_hint < 0 ||
         placement_.local_rank_hint == local_.rank);
    result = agree({mapping ? ErrorCode::success : ErrorCode::invalid_argument,
                    RuntimeStage::device_mapping});
    if (!result.succeeded()) return result;

    local = device_result(Traits::query_current_device_properties(properties_),
                          RuntimeStage::device_properties);
    if (local.succeeded() && properties_.device_ordinal != placement_.device_ordinal)
      local = {ErrorCode::invalid_state, RuntimeStage::device_properties};
    result = agree(local);
    if (!result.succeeded()) return result;
    result = agree({device::try_validate_explicit_executor(
                        stream_.executor(), placement_.device_ordinal),
                    RuntimeStage::stream_admission});
    if (!result.succeeded()) return result;

    result = agree(device_result(Traits::query_current_device_memory(memory_),
                                  RuntimeStage::device_memory));
    if (!result.succeeded()) return result;
    const auto configured = try_make_communication_setup(*domain_,
        properties_, stream_.executor(), setup_);
    result = agree({configured.code(), RuntimeStage::communication_setup,
                    configured.native_result()});
    ready_ = result.succeeded();
    return result;
  }

  // Consumers, workers and all accepted operations must already be retired.
  [[nodiscard]] RuntimeResult close() noexcept {
    if (!active()) return {};
    if (!mpi_.on_initializing_thread())
      return {ErrorCode::invalid_state, RuntimeStage::mpi_finalization};
    ready_ = false;
    setup_ = {};
    const auto released = stream_.reset();
    domain_.reset(); // MpiCallDomain must disappear before MPI_Finalize.
    const auto result = agree(device_result(released, RuntimeStage::stream_destruction));
    if (result.native_result() != MPI_SUCCESS) return result;
    const auto finalized = mpi_.finalize();
    if (!finalized.succeeded())
      return {finalized.code(), RuntimeStage::mpi_finalization,
              finalized.native_result(), rank_};
    return result;
  }

  [[nodiscard]] bool active() const noexcept { return mpi_.active(); }
  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }
  [[nodiscard]] int local_rank() const noexcept { return local_.rank; }
  [[nodiscard]] int local_size() const noexcept { return local_.size; }
  [[nodiscard]] int device_ordinal() const noexcept { return placement_.device_ordinal; }
  // Borrow only after successful initialization, and release before close().
  [[nodiscard]] MpiCallDomain &domain() noexcept { return *domain_; }
  [[nodiscard]] const CommunicationSetup &setup() const noexcept { return setup_; }
  [[nodiscard]] device::Executor<Backend> executor() const noexcept { return stream_.executor(); }
  [[nodiscard]] const device::DeviceProperties &properties() const noexcept { return properties_; }
  [[nodiscard]] const device::DeviceMemoryInfo &memory_info() const noexcept { return memory_; }
  [[noreturn]] void abort(int code) const noexcept { mpi_.abort(code); }

private:
  [[nodiscard]] static RuntimeResult device_result(
      typename Traits::error_type error, RuntimeStage stage) noexcept {
    return {device::to_error_code<Backend>(error), stage};
  }

  [[nodiscard]] RuntimeResult prepare_device(DevicePlacement options) noexcept {
    auto result = device_result(Traits::query_device_count(visible_),
                                 RuntimeStage::device_discovery);
    if (!result.succeeded()) return result;
    if (visible_ <= 0)
      return {ErrorCode::backend_failure, RuntimeStage::device_discovery};
    const auto placement =
        detail::try_resolve_device_placement(visible_, options, placement_);
    if (placement != ErrorCode::success)
      return {placement, RuntimeStage::device_selection};
    result = device_result(Traits::select_device(placement_.device_ordinal),
                            RuntimeStage::device_selection);
    if (!result.succeeded()) return result;
    // The actual long-lived, nondefault stream also establishes the selected
    // context on runtimes where selecting a device alone is lazy.
    result = device_result(Stream::create(stream_), RuntimeStage::stream_creation);
    if (!result.succeeded()) return result;
    return {device::try_validate_explicit_executor(stream_.executor(),
                                                 placement_.device_ordinal),
            RuntimeStage::stream_admission};
  }

  // Startup/teardown is process-exclusive: no workers or borrowed operations
  // may be published yet/still live, and some ranks may lack a call domain.
  // The lowest failing rank reports for everyone; a native MPI failure
  // dominates so that callers abort instead of closing a communicator whose
  // state may no longer permit collective cleanup.
  [[nodiscard]] RuntimeResult agree(const RuntimeResult local) const noexcept {
    std::array<std::uint64_t, 1> reporter{
        local.succeeded()
            ? std::uint64_t{0}
            : (std::uint64_t{local.native_result() != MPI_SUCCESS} << 32U) |
                  static_cast<std::uint64_t>(std::int64_t{INT_MAX} - rank_)};
    static_cast<void>(detail::agree_contract(MPI_COMM_WORLD, ErrorCode::success,
                                             std::array<std::uint64_t, 0>{},
                                             reporter));
    if (reporter[0] == 0) return {};
    const int first = static_cast<int>(
        std::int64_t{INT_MAX} -
        static_cast<std::int64_t>(reporter[0] & UINT64_C(0xffffffff)));
    if (first < 0) {
      // A native rank/bootstrap query failed; no reliable root is available.
      return {ErrorCode::backend_failure, RuntimeStage::mpi_bootstrap,
              local.native_result() != MPI_SUCCESS ? local.native_result()
                                                   : MPI_ERR_OTHER};
    }
    std::array<int, 3> error{static_cast<int>(local.code()),
                            static_cast<int>(local.stage()), local.native_result()};
    if (MPI_Bcast(error.data(), static_cast<int>(error.size()), MPI_INT, first,
                  MPI_COMM_WORLD) != MPI_SUCCESS)
      std::abort();
    return {static_cast<ErrorCode>(error[0]), static_cast<RuntimeStage>(error[1]),
            error[2], first};
  }

  MpiRuntime mpi_;
  std::optional<MpiCallDomain> domain_;
  Stream stream_;
  CommunicationSetup setup_;
  device::DeviceProperties properties_{};
  device::DeviceMemoryInfo memory_{};
  detail::PreparedDevicePlacement placement_{};
  MpiLocalTopology local_{};
  int visible_ = 0;
  int rank_ = -1;
  int size_ = 0;
  bool ready_ = false;
};

} // namespace lpfx::distributed

#endif
