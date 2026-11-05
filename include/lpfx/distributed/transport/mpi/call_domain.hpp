#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_CALL_DOMAIN_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_CALL_DOMAIN_HPP

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <thread>

#include <mpi.h>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx {

// Cold MPI boundaries retain the native result for diagnostics without
// widening the library-wide ErrorCode used for control flow.
class MpiResult final {
public:
  constexpr MpiResult() noexcept = default;
  constexpr MpiResult(const ErrorCode code) noexcept : code_(code) {}
  constexpr MpiResult(const ErrorCode code, const int native_result) noexcept
      : code_(code), native_result_(native_result) {}

  // backend_failure unless result is MPI_SUCCESS.
  [[nodiscard]] static constexpr MpiResult native(const int result) noexcept {
    return {result == MPI_SUCCESS ? ErrorCode::success
                                  : ErrorCode::backend_failure,
            result};
  }

  [[nodiscard]] constexpr bool succeeded() const noexcept {
    return code_ == ErrorCode::success;
  }
  [[nodiscard]] constexpr ErrorCode code() const noexcept { return code_; }
  [[nodiscard]] constexpr bool native_succeeded() const noexcept {
    return native_result_ == MPI_SUCCESS;
  }
  [[nodiscard]] constexpr int native_result() const noexcept {
    return native_result_;
  }

private:
  ErrorCode code_ = ErrorCode::success;
  int native_result_ = MPI_SUCCESS;
};

enum class MpiThreadMode : std::uint8_t {
  single = 0,
  funneled = 1,
  serialized = 2,
  multiple = 3,
};

enum class MpiProcessThreadUse : std::uint8_t {
  single_process_thread = 0,
  other_process_threads_may_exist = 1,
};

class MpiCallDomain;
class MpiCallLease;

namespace detail {

// LPFX_TRY operand adapter; the native result is dropped.
[[nodiscard]] constexpr ErrorCode try_error(const MpiResult result) noexcept {
  return result.code();
}

// One consistently configured linked LPFx image shares these identities across
// all translation units. Independently loaded hidden-symbol DSOs are outside
// this ownership contract. Neither object requires dynamic initialization.
inline constinit std::atomic<bool> process_domain_live{false};
inline constinit std::atomic<unsigned int> process_call_gate{0};

[[nodiscard]] inline bool try_map_thread_mode(const int native_level,
                                              MpiThreadMode &output) noexcept {
  if (native_level == MPI_THREAD_SINGLE) {
    output = MpiThreadMode::single;
  } else if (native_level == MPI_THREAD_FUNNELED) {
    output = MpiThreadMode::funneled;
  } else if (native_level == MPI_THREAD_SERIALIZED) {
    output = MpiThreadMode::serialized;
  } else if (native_level == MPI_THREAD_MULTIPLE) {
    output = MpiThreadMode::multiple;
  } else {
    return false;
  }
  return true;
}

// Rank and size of an intracommunicator; invalid_argument otherwise.
[[nodiscard]] inline MpiResult
query_intracommunicator(const MPI_Comm communicator, int &rank,
                        int &size) noexcept {
  int intercommunicator = 0;
  int native = MPI_Comm_test_inter(communicator, &intercommunicator);
  if (native == MPI_SUCCESS) {
    native = MPI_Comm_rank(communicator, &rank);
  }
  if (native == MPI_SUCCESS) {
    native = MPI_Comm_size(communicator, &size);
  }
  if (native != MPI_SUCCESS) {
    return MpiResult::native(native);
  }
  return intercommunicator == 0 && rank >= 0 && size > 0 && rank < size
             ? MpiResult{}
             : MpiResult{ErrorCode::invalid_argument};
}

// Keep the contended wait off the admission path. Its header definition is
// ODR-shared, while the uncontended acquire remains one atomic exchange.
LPFX_NOINLINE inline void wait_for_process_call_gate() noexcept {
  do {
    process_call_gate.wait(1, std::memory_order_relaxed);
  } while (process_call_gate.exchange(1, std::memory_order_acquire) != 0);
}

LPFX_ALWAYS_INLINE void acquire_process_call_gate() noexcept {
  if (process_call_gate.exchange(1, std::memory_order_acquire) != 0) {
    wait_for_process_call_gate();
  }
}

LPFX_ALWAYS_INLINE void release_process_call_gate() noexcept {
  process_call_gate.store(0, std::memory_order_release);
  process_call_gate.notify_one();
}

void seal_mpi_registration(const MpiCallDomain &) noexcept;
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
acquire_mpi_call(const MpiCallDomain &, bool, MpiCallLease &) noexcept;

} // namespace detail

// Exactly one process-local domain may be active. Derived resources borrow its
// stable address, so an active domain is deliberately neither copyable nor
// movable and never initializes or finalizes MPI. The bootstrap communicator
// is a validated intracommunicator whose error handler returns errors.
class MpiCallDomain final {
public:
  constexpr MpiCallDomain() noexcept = default;
  LPFX_NONMOVABLE(MpiCallDomain);
  ~MpiCallDomain() noexcept;

  [[nodiscard]] constexpr bool active() const noexcept { return active_; }
  [[nodiscard]] constexpr MpiThreadMode thread_mode() const noexcept {
    return thread_mode_;
  }
  [[nodiscard]] constexpr int native_thread_level() const noexcept {
    return native_thread_level_;
  }
  [[nodiscard]] std::thread::id initializing_thread() const noexcept {
    return initializing_thread_;
  }
  [[nodiscard]] MPI_Comm bootstrap_communicator() const noexcept {
    return bootstrap_communicator_;
  }
  [[nodiscard]] constexpr int rank() const noexcept { return rank_; }
  [[nodiscard]] constexpr int size() const noexcept { return size_; }
  // The domain admits exactly one communication registration epoch.
  [[nodiscard]] constexpr bool registration_sealed() const noexcept {
    return registration_sealed_;
  }

private:
  bool active_ = false;
  MpiThreadMode thread_mode_ = MpiThreadMode::single;
  int native_thread_level_ = MPI_THREAD_SINGLE;
  int rank_ = 0;
  int size_ = 0;
  std::thread::id initializing_thread_{};
  MPI_Comm bootstrap_communicator_ = MPI_COMM_NULL;
  mutable bool registration_sealed_ = false;

  friend MpiResult try_make_mpi_call_domain(MPI_Comm, MpiProcessThreadUse,
                                            MpiCallDomain &) noexcept;
  friend void detail::seal_mpi_registration(const MpiCallDomain &) noexcept;
};

inline MpiCallDomain::~MpiCallDomain() noexcept {
  if (!active_) {
    return;
  }
  int finalized = 0;
  if (std::this_thread::get_id() != initializing_thread_ ||
      MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0) {
    std::abort();
  }
  active_ = false;
  bootstrap_communicator_ = MPI_COMM_NULL;
  registration_sealed_ = false;
  detail::process_domain_live.store(false, std::memory_order_release);
}

inline void detail::seal_mpi_registration(const MpiCallDomain &domain) noexcept {
  domain.registration_sealed_ = true;
}

inline MpiResult
try_make_mpi_call_domain(const MPI_Comm bootstrap_communicator,
                         const MpiProcessThreadUse process_thread_use,
                         MpiCallDomain &output) noexcept {
  bool expected = false;
  if (output.active_ || !detail::process_domain_live.compare_exchange_strong(
                            expected, true, std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
    return ErrorCode::invalid_state;
  }
  // Releases the process reservation unless the domain is published.
  struct Reservation final {
    bool committed = false;
    LPFX_NONMOVABLE(Reservation);
    Reservation() noexcept = default;
    ~Reservation() noexcept {
      if (!committed) {
        detail::process_domain_live.store(false, std::memory_order_release);
      }
    }
  } reservation;

  int initialized = 0;
  int finalized = 0;
  int native_thread_level = MPI_THREAD_SINGLE;
  int main_thread = 0;
  int native = MPI_Initialized(&initialized);
  if (native == MPI_SUCCESS && initialized != 0) {
    native = MPI_Finalized(&finalized);
  }
  if (native != MPI_SUCCESS) {
    return MpiResult::native(native);
  }
  if (initialized == 0 || finalized != 0) {
    return ErrorCode::invalid_state;
  }
  native = MPI_Query_thread(&native_thread_level);
  if (native != MPI_SUCCESS) {
    return MpiResult::native(native);
  }
  MpiThreadMode thread_mode = MpiThreadMode::single;
  if (!detail::try_map_thread_mode(native_thread_level, thread_mode)) {
    return ErrorCode::backend_failure;
  }
  native = MPI_Is_thread_main(&main_thread);
  if (native != MPI_SUCCESS) {
    return MpiResult::native(native);
  }
  if (main_thread == 0) {
    return ErrorCode::invalid_state;
  }
  if ((thread_mode == MpiThreadMode::single &&
       process_thread_use != MpiProcessThreadUse::single_process_thread) ||
      bootstrap_communicator == MPI_COMM_NULL) {
    return ErrorCode::invalid_argument;
  }

  MPI_Errhandler handler = MPI_ERRHANDLER_NULL;
  native = MPI_Comm_get_errhandler(bootstrap_communicator, &handler);
  if (native != MPI_SUCCESS) {
    return MpiResult::native(native);
  }
  const bool returns_errors = handler == MPI_ERRORS_RETURN;
  native = MPI_Errhandler_free(&handler);
  if (native != MPI_SUCCESS) {
    return MpiResult::native(native);
  }
  if (!returns_errors) {
    return ErrorCode::invalid_argument;
  }
  int rank = 0;
  int size = 0;
  const MpiResult queried =
      detail::query_intracommunicator(bootstrap_communicator, rank, size);
  if (!queried.succeeded()) {
    return queried;
  }

  output.thread_mode_ = thread_mode;
  output.native_thread_level_ = native_thread_level;
  output.rank_ = rank;
  output.size_ = size;
  output.initializing_thread_ = std::this_thread::get_id();
  output.bootstrap_communicator_ = bootstrap_communicator;
  output.registration_sealed_ = false;
  output.active_ = true;
  reservation.committed = true;
  return {};
}

class MpiCallLease final {
public:
  constexpr MpiCallLease() noexcept = default;
  MpiCallLease(const MpiCallLease &) = delete;
  MpiCallLease &operator=(const MpiCallLease &) = delete;
  MpiCallLease(MpiCallLease &&other) noexcept
      : phase_(other.phase_) {
    other.phase_ = Phase::empty;
  }
  MpiCallLease &operator=(MpiCallLease &&) = delete;
  LPFX_ALWAYS_INLINE ~MpiCallLease() noexcept {
    if (phase_ == Phase::gated) {
      detail::release_process_call_gate();
    }
    phase_ = Phase::empty;
  }

  [[nodiscard]] constexpr bool active() const noexcept {
    return phase_ != Phase::empty;
  }

private:
  // One authoritative admission/ownership state. The high active bit keeps
  // the ownership discriminator independent of earlier true predicates in
  // GCC's inlined collective path, avoiding an extra gate-ownership spill.
  enum class Phase : std::uint8_t { empty = 0, active = 0x80, gated = 0x81 };
  Phase phase_ = Phase::empty;

  friend ErrorCode detail::acquire_mpi_call(const MpiCallDomain &, bool,
                                            MpiCallLease &) noexcept;
};

[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
detail::acquire_mpi_call(const MpiCallDomain &domain, const bool disjoint_state,
                         MpiCallLease &output) noexcept {
  if (!domain.active() || output.active()) {
    return ErrorCode::invalid_state;
  }
  const auto mode = domain.thread_mode();
  if ((mode == MpiThreadMode::single || mode == MpiThreadMode::funneled) &&
      std::this_thread::get_id() != domain.initializing_thread()) {
    return ErrorCode::invalid_state;
  }

  const bool requires_gate =
      mode == MpiThreadMode::serialized ||
      (mode == MpiThreadMode::multiple && !disjoint_state);
  if (requires_gate) {
    acquire_process_call_gate();
  }
  output.phase_ =
      requires_gate ? MpiCallLease::Phase::gated : MpiCallLease::Phase::active;
  return ErrorCode::success;
}

// Shared control/registration state is gated for SERIALIZED and MULTIPLE.
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode try_acquire_shared_mpi_call(
    const MpiCallDomain &domain, MpiCallLease &output) noexcept {
  return detail::acquire_mpi_call(domain, false, output);
}

// The caller asserts disjoint mutable native state. MULTIPLE therefore takes
// no LPFx gate; SERIALIZED still serializes entry as required by MPI.
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode try_acquire_disjoint_mpi_call(
    const MpiCallDomain &domain, MpiCallLease &output) noexcept {
  return detail::acquire_mpi_call(domain, true, output);
}

} // namespace lpfx

#endif
