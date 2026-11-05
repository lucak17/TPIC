#ifndef LPFX_DISTRIBUTED_TRANSPORT_XCCL_SESSION_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_XCCL_SESSION_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/error_reporting.hpp>

namespace lpfx::distributed::detail {

// The XCCL native boundary is deliberately vendor-neutral. A concrete NCCL
// or RCCL opt-in supplies one empty adapter with these qualified static calls;
// naming this contract alone does not advertise either vendor as supported.
// initialize() selects the vendor's nonblocking mode with split resource
// sharing; split() derives the blocking operations communicator from it.
// The succeeded/in_progress/valid_communicator predicates are pure queries;
// error_string and last_error serve failure reports only.
template <class Native>
concept XcclNativeAdapter =
    std::is_empty_v<Native> &&
    device::DeviceEventBackend<typename Native::device_backend_type> &&
    device::DeviceDiscoveryBackend<typename Native::device_backend_type> &&
    std::is_trivially_copyable_v<typename Native::unique_id_type> &&
    std::is_trivially_default_constructible_v<
        typename Native::unique_id_type> &&
    std::is_trivially_copyable_v<typename Native::result_type> &&
    std::is_trivially_default_constructible_v<typename Native::result_type> &&
    requires(typename Native::communicator_type &output,
             typename Native::unique_id_type &unique_id,
             const typename Native::communicator_type communicator,
             const typename Native::unique_id_type &input_id,
             typename Native::result_type &async_state,
             const typename Native::result_type result, const int rank_count,
             const int rank, int &runtime_version) {
      {
        Native::null_communicator
      } -> std::convertible_to<typename Native::communicator_type>;
      {
        Native::success_result
      } -> std::convertible_to<typename Native::result_type>;
      {
        Native::in_progress_result
      } -> std::convertible_to<typename Native::result_type>;
      {
        Native::invalid_argument_result
      } -> std::convertible_to<typename Native::result_type>;
      typename std::integral_constant<int, Native::minimum_runtime_version>;
      {
        Native::query_runtime_version(runtime_version)
      } noexcept -> std::same_as<typename Native::result_type>;
      { Native::succeeded(result) } noexcept -> std::same_as<bool>;
      { Native::in_progress(result) } noexcept -> std::same_as<bool>;
      {
        Native::valid_communicator(communicator)
      } noexcept -> std::same_as<bool>;
      {
        Native::get_unique_id(unique_id)
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::initialize(output, rank_count, input_id, rank)
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::split(communicator, output)
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::query_async_error(communicator, async_state)
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::abort(communicator)
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::finalize(communicator)
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::destroy(communicator)
      } noexcept -> std::same_as<typename Native::result_type>;
      { Native::error_string(result) } noexcept -> std::same_as<const char *>;
      {
        Native::last_error(communicator)
      } noexcept -> std::same_as<const char *>;
    };

// P2P payload ports group their transfers; collectives issue one direct
// native operation and do not require this extension.
template <class Native>
concept XcclGroupedNativeAdapter =
    XcclNativeAdapter<Native> &&
    requires(std::byte *receive, const std::byte *send,
             const std::size_t bytes, const int peer,
             const typename Native::communicator_type communicator,
             const typename device::BackendTraits<
                 typename Native::device_backend_type>::stream_type stream) {
      {
        Native::group_start()
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::group_end()
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::template transfer<true>(receive, bytes, peer, communicator,
                                        stream)
      } noexcept -> std::same_as<typename Native::result_type>;
      {
        Native::template transfer<false>(send, bytes, peer, communicator,
                                         stream)
      } noexcept -> std::same_as<typename Native::result_type>;
    };

enum class XcclSubmissionMode : std::uint8_t { direct, grouped };

// The submission check whose failure ends the process, in issue order.
enum class XcclSubmissionCheck : std::uint8_t {
  asynchronous_error, // query and state: an earlier operation's error
  group_start,
  issue,         // the direct native call
  grouped_issue, // the grouped native calls and the group end
};

// Cold report of one failed submission check, before the caller aborts: one
// stderr line with the participant's rank, the check, its native results and
// the vendor's latest warning text. It makes no communicator query, which
// would itself fail on the asynchronous error being reported. Raw codegen
// references call it at the same sites, so a successful epoch's work is
// unchanged.
template <XcclNativeAdapter Native>
LPFX_NOINLINE void report_xccl_submission_failure(
    const XcclSubmissionCheck check, const int rank,
    const typename Native::communicator_type communicator,
    const typename Native::result_type first,
    const typename Native::result_type second) noexcept {
  struct Labels {
    const char *name, *first_name, *second_name; // null: one result
  };
  constexpr Labels labels[]{
      {"asynchronous-error check", "query", "state"},
      {"group start", "result", nullptr},
      {"native issue", "result", nullptr},
      {"grouped issue", "issue", "group_end"},
  };
  const Labels &label = labels[static_cast<unsigned>(check)];
  const char *const latest = Native::last_error(communicator);
  const char *const warning =
      latest != nullptr && latest[0] != '\0' ? latest : "none";
  const lpfx::detail::RankText rank_text{rank};
  if (label.second_name == nullptr) {
    lpfx::detail::print_error_line(
        "LPFx XCCL%s: %s failed: %s=%d (%s) last_warning=%s\n",
        rank_text.text, label.name, label.first_name, static_cast<int>(first),
        lpfx::detail::error_text(Native::error_string(first)), warning);
  } else {
    lpfx::detail::print_error_line(
        "LPFx XCCL%s: %s failed: %s=%d (%s) %s=%d (%s) last_warning=%s\n",
        rank_text.text, label.name, label.first_name, static_cast<int>(first),
        lpfx::detail::error_text(Native::error_string(first)),
        label.second_name, static_cast<int>(second),
        lpfx::detail::error_text(Native::error_string(second)), warning);
  }
}

enum class XcclCommunicatorPhase : std::uint8_t {
  empty = 0,
  initializing = 1,
  splitting = 2,
  ready = 3,
  failed = 4,
  finalizing = 5,
  finalized = 6,
};

// Shared XCCL resource owner for one native participant/device group. The
// nonblocking parent carries initialization, finalization and abort; every
// operation runs on its blocking child, so a submission is enqueued when the
// native call returns and no submission state is shared between ports. MPI
// control performs the rank-symmetric agreement around every lifecycle result
// returned here; this owner contains no MPI dependency or coordination policy.
template <XcclNativeAdapter Native> class XcclSession final {
public:
  using executor_type = device::Executor<typename Native::device_backend_type>;
  using communicator_type = typename Native::communicator_type;
  using unique_id_type = typename Native::unique_id_type;
  using result_type = typename Native::result_type;
  using phase_type = XcclCommunicatorPhase;

  constexpr XcclSession() noexcept = default;
  LPFX_NONMOVABLE(XcclSession);
  ~XcclSession() noexcept {
    // Native lifecycle calls may require rank participation. Destruction must
    // never invent an uncoordinated abort/finalize operation.
    if (!empty()) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return phase_ == phase_type::empty;
  }
  [[nodiscard]] constexpr bool ready() const noexcept {
    return phase_ == phase_type::ready;
  }
  // The operations communicator; read only once ready.
  [[nodiscard]] constexpr communicator_type native_handle() const noexcept {
    return communicator_;
  }

  // Issues one native submission on the operations communicator and returns
  // once it is enqueued on executor's stream. An earlier operation's
  // asynchronous error is fatal here. Every failed check is reported with its
  // native results before the abort.
  template <XcclSubmissionMode Mode = XcclSubmissionMode::direct, class Issue>
    requires std::is_nothrow_invocable_r_v<result_type, Issue &> &&
             (Mode == XcclSubmissionMode::direct ||
              XcclGroupedNativeAdapter<Native>)
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  submit(const executor_type &executor, Issue &&issue) noexcept {
    // Both predicates are pure admission facts; evaluate them together so an
    // inlined port adds no short-circuit branch.
    if ((static_cast<unsigned>(ready()) &
         static_cast<unsigned>(executor.bound())) == 0) {
      return Status::failure(ErrorCode::invalid_state);
    }
    // The epoch has entered its rank-global native definition order. There is
    // no symmetric drain/reset protocol for a local native failure. The
    // blocking communicator is never in progress: the query and its state
    // both succeed unless an earlier operation failed.
    result_type state = Native::in_progress_result;
    const result_type query = Native::query_async_error(communicator_, state);
    if (!Native::succeeded(query) || !Native::succeeded(state)) {
      report_xccl_submission_failure<Native>(
          XcclSubmissionCheck::asynchronous_error, rank_, communicator_, query,
          state);
      std::abort();
    }
    if constexpr (Mode == XcclSubmissionMode::grouped) {
      const result_type started = Native::group_start();
      if (!Native::succeeded(started)) {
        report_xccl_submission_failure<Native>(
            XcclSubmissionCheck::group_start, rank_, communicator_, started,
            Native::success_result);
        std::abort();
      }
      const result_type issued = issue();
      const result_type ended = Native::group_end();
      if (!Native::succeeded(issued) || !Native::succeeded(ended)) {
        report_xccl_submission_failure<Native>(
            XcclSubmissionCheck::grouped_issue, rank_, communicator_, issued,
            ended);
        std::abort();
      }
    } else {
      const result_type issued = issue();
      if (!Native::succeeded(issued)) {
        report_xccl_submission_failure<Native>(
            XcclSubmissionCheck::issue, rank_, communicator_, issued,
            Native::success_result);
        std::abort();
      }
    }
    return Status::success();
  }

  // A failure may still produce a native handle. Retain it in failed state so
  // composition can first agree the failure and then abort all retained
  // handles in the same rank-global order.
  [[nodiscard]] result_type try_initialize(const int rank_count,
                                           const unique_id_type &unique_id,
                                           const int rank) noexcept {
    if (!empty() || rank_count <= 0 || rank < 0 || rank >= rank_count) {
      return Native::invalid_argument_result;
    }
    rank_ = rank;
    communicator_type prepared = Native::null_communicator;
    const result_type result =
        Native::initialize(prepared, rank_count, unique_id, rank);
    if (Native::succeeded(result) || Native::in_progress(result)) {
      if (!Native::valid_communicator(prepared)) {
        std::abort();
      }
      parent_ = prepared;
      if (Native::succeeded(result)) {
        return split();
      }
      phase_ = phase_type::initializing;
    } else if (Native::valid_communicator(prepared)) {
      parent_ = prepared;
      phase_ = phase_type::failed;
    }
    return result;
  }

  // Polls the parent: a completed initialization starts the split, and a
  // completed split publishes the child.
  [[nodiscard]] result_type progress_initialization() noexcept {
    if (phase_ == phase_type::initializing) {
      const result_type result =
          progress<phase_type::initializing, phase_type::splitting>();
      return phase_ == phase_type::splitting ? split() : result;
    }
    return progress<phase_type::splitting, phase_type::ready>();
  }

  // The blocking child finalizes inside the call; the parent then finalizes
  // asynchronously and progress_finalization() polls it.
  [[nodiscard]] result_type try_finalize() noexcept {
    if (phase_ == phase_type::finalized) {
      return Native::success_result;
    }
    if (!ready()) {
      return Native::invalid_argument_result;
    }
    const result_type child = Native::finalize(communicator_);
    const result_type result =
        Native::succeeded(child) ? Native::finalize(parent_) : child;
    phase_ = Native::succeeded(result)     ? phase_type::finalized
             : Native::in_progress(result) ? phase_type::finalizing
                                           : phase_type::failed;
    return result;
  }

  [[nodiscard]] result_type progress_finalization() noexcept {
    return progress<phase_type::finalizing, phase_type::finalized>();
  }

  // Abort consumes every retained native handle, whatever lifecycle phase it
  // is in. It is idempotent for ranks whose failed initialization produced no
  // handle, which lets composition run one symmetric rollback schedule. The
  // parent goes first: it raises the shared abort flag and joins an in-flight
  // split, which may publish a child (also after a failed split) only then.
  [[nodiscard]] result_type abort() noexcept {
    if (empty()) {
      return Native::success_result;
    }
    const result_type parent =
        Native::abort(std::exchange(parent_, Native::null_communicator));
    const communicator_type child =
        std::exchange(communicator_, Native::null_communicator);
    phase_ = phase_type::empty;
    const result_type aborted = Native::valid_communicator(child)
                                    ? Native::abort(child)
                                    : Native::success_result;
    return Native::succeeded(parent) ? aborted : parent;
  }

  // Both handles are consumed once destroy is called, even when the vendor
  // reports an error. A retry could otherwise use an invalid handle and
  // diverge from peers.
  [[nodiscard]] result_type destroy_finalized() noexcept {
    if (phase_ != phase_type::finalized) {
      return Native::invalid_argument_result;
    }
    const communicator_type child =
        std::exchange(communicator_, Native::null_communicator);
    const communicator_type parent =
        std::exchange(parent_, Native::null_communicator);
    phase_ = phase_type::empty;
    const result_type destroyed = Native::destroy(child);
    const result_type parent_destroyed = Native::destroy(parent);
    return Native::succeeded(destroyed) ? parent_destroyed : destroyed;
  }

private:
  template <phase_type Pending, phase_type Complete>
  [[nodiscard]] result_type progress() noexcept {
    if (phase_ == Complete) {
      return Native::success_result;
    }
    if (phase_ != Pending) {
      return Native::invalid_argument_result;
    }
    // A failed query reports its own result, otherwise the parent's state.
    result_type state = Native::in_progress_result;
    const result_type query = Native::query_async_error(parent_, state);
    const result_type result = Native::succeeded(query) ? state : query;
    if (!Native::in_progress(result)) {
      phase_ = Native::succeeded(result) ? Complete : phase_type::failed;
    }
    return result;
  }

  // The vendor writes the child from its own thread; it is read only after
  // the parent poll completes (or the parent abort has joined the split).
  // An immediate success leaves the child null until that poll.
  [[nodiscard]] result_type split() noexcept {
    const result_type result = Native::split(parent_, communicator_);
    if (!Native::succeeded(result) && !Native::in_progress(result)) {
      phase_ = phase_type::failed;
      return result;
    }
    phase_ = phase_type::splitting;
    return Native::in_progress_result;
  }

  communicator_type communicator_ = Native::null_communicator;
  communicator_type parent_ = Native::null_communicator;
  phase_type phase_ = phase_type::empty;
  // Both communicators' vendor rank (the split keeps the parent order), for
  // failure reports. With pointer handles it fits the padding after phase_.
  int rank_ = -1;
};

template <class Session, class Native>
concept XcclSessionProvider =
    XcclNativeAdapter<Native> && requires(Session &session) {
      { session.xccl() } -> std::same_as<XcclSession<Native> &>;
    };

} // namespace lpfx::distributed::detail

#endif
