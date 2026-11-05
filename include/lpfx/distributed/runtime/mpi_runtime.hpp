#ifndef LPFX_DISTRIBUTED_RUNTIME_MPI_RUNTIME_HPP
#define LPFX_DISTRIBUTED_RUNTIME_MPI_RUNTIME_HPP

#include <cstdlib>
#include <thread>

#include <mpi.h>

#include <lpfx/distributed/transport/mpi/call_domain.hpp>

namespace lpfx {

// Optional ownership of MPI's process lifetime; an MpiCallDomain may instead
// borrow an externally initialized runtime. This owner never attaches
// implicitly and never finalizes MPI during exception unwinding.
class MpiRuntime final {
public:
  MpiRuntime() noexcept = default;
  LPFX_NONMOVABLE(MpiRuntime);
  ~MpiRuntime() noexcept {
    if (active_) std::abort();
  }

  [[nodiscard]] static MpiResult can_initialize() noexcept {
    int initialized = 0, finalized = 0;
    int native = MPI_Initialized(&initialized);
    if (native == MPI_SUCCESS) native = MPI_Finalized(&finalized);
    if (native != MPI_SUCCESS) return MpiResult::native(native);
    return initialized == 0 && finalized == 0 ? ErrorCode::success
                                              : ErrorCode::invalid_state;
  }

  [[nodiscard]] MpiResult initialize(int &argc, char **&argv,
                                     MpiThreadMode required) noexcept {
    if (active_) return ErrorCode::invalid_state;
    const auto available = can_initialize();
    if (!available.succeeded()) return available;
    int native_level = MPI_THREAD_SINGLE;
    switch (required) {
    case MpiThreadMode::single: native_level = MPI_THREAD_SINGLE; break;
    case MpiThreadMode::funneled: native_level = MPI_THREAD_FUNNELED; break;
    case MpiThreadMode::serialized: native_level = MPI_THREAD_SERIALIZED; break;
    case MpiThreadMode::multiple: native_level = MPI_THREAD_MULTIPLE; break;
    default: return ErrorCode::invalid_argument;
    }
    const int native = MPI_Init_thread(&argc, &argv, native_level, &provided_);
    if (native != MPI_SUCCESS) return MpiResult::native(native);
    active_ = true;
    thread_ = std::this_thread::get_id();
    return {};
  }

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] int provided_thread_level() const noexcept { return provided_; }
  [[nodiscard]] bool on_initializing_thread() const noexcept {
    return active_ && thread_ == std::this_thread::get_id();
  }

  // Collective, after every communication object and call domain has closed.
  [[nodiscard]] MpiResult finalize() noexcept {
    if (!on_initializing_thread() ||
        detail::process_domain_live.load(std::memory_order_acquire))
      return ErrorCode::invalid_state;
    const int native = MPI_Finalize();
    if (native != MPI_SUCCESS) return MpiResult::native(native);
    active_ = false;
    return {};
  }

  [[noreturn]] void abort(const int code) const noexcept {
    if (active_) static_cast<void>(MPI_Abort(MPI_COMM_WORLD, code));
    std::abort();
  }

private:
  bool active_ = false;
  int provided_ = MPI_THREAD_SINGLE;
  std::thread::id thread_{};
};

struct MpiLocalTopology final {
  int rank = -1;
  int size = 0;
};

// Discover actual node-local placement only after MPI exists. This validates
// an earlier launcher hint; it must not be used to defer device selection.
[[nodiscard]] inline MpiResult try_query_mpi_local_topology(
    const MpiCallDomain &domain, MpiLocalTopology &output) noexcept {
  MpiCallLease lease;
  LPFX_TRY(try_acquire_shared_mpi_call(domain, lease));
  MPI_Comm local = MPI_COMM_NULL;
  // Equal keys preserve bootstrap-rank order. Avoid a fallible local rank
  // query between call admission and this collective.
  const int split = MPI_Comm_split_type(domain.bootstrap_communicator(),
                                        MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                        &local);
  if (split != MPI_SUCCESS) return MpiResult::native(split);
  MpiLocalTopology topology;
  MpiResult result =
      MpiResult::native(MPI_Comm_set_errhandler(local, MPI_ERRORS_RETURN));
  if (result.succeeded())
    result = detail::query_intracommunicator(local, topology.rank, topology.size);
  const int released = MPI_Comm_free(&local);
  if (!result.succeeded()) return result;
  if (released != MPI_SUCCESS) return MpiResult::native(released);
  output = topology;
  return {};
}

} // namespace lpfx

#endif
