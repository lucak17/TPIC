#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_P2P_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_P2P_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/detail/fixed_byte.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>
#include <lpfx/distributed/transport/mpi/communicator.hpp>
#include <lpfx/distributed/transport/workspace.hpp>

namespace lpfx::detail {

// Dynamic batches borrow budget-backed request slots. Only protocols which
// audit successful receives retain status storage. All completion failures are
// fail-stop, including MPI_ERR_IN_STATUS; fixed traffic needs no status payload.
struct NoMpiRequestStatuses final {};
template <bool RetainStatuses> struct DynamicMpiRequestStorage final {
  MPI_Request *requests = nullptr;
  [[no_unique_address]]
  std::conditional_t<RetainStatuses, MPI_Status *, NoMpiRequestStatuses> statuses{};
  std::size_t capacity = 0;
  std::size_t active_count = 0;
};

struct SingleMpiRequestStorage final {
  MPI_Request request = MPI_REQUEST_NULL;
};

// Sole request/status auditor for every MPI family. Protocol owners retain
// immutable, cold-validated recipes; this owner supplies their request slots,
// one native poll/wait per driving call, and the exact issued/completed state.
template <std::size_t Extent, bool RetainStatuses = (Extent == std::dynamic_extent)>
class BasicPreparedMpiRequestBatch final {
  static_assert(Extent == std::dynamic_extent || Extent == 1);
  static constexpr bool dynamic = Extent == std::dynamic_extent;
  static_assert(dynamic || !RetainStatuses);

public:
  constexpr BasicPreparedMpiRequestBatch() noexcept = default;
  LPFX_NONMOVABLE(BasicPreparedMpiRequestBatch);
  ~BasicPreparedMpiRequestBatch() noexcept {
    // Binding is retained native ownership even while idle. Every owner must
    // therefore participate in explicit reverse-order reset before destruction.
    if (phase_ != Phase::unbound) {
      std::abort();
    }
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return phase_ == Phase::unbound;
  }
  [[nodiscard]] constexpr bool is_valid() const noexcept {
    if constexpr (dynamic) {
      return phase_ != Phase::unbound && domain_ != nullptr &&
             (storage_.capacity == 0 ||
              (storage_.requests != nullptr && [&] {
                if constexpr (RetainStatuses) return storage_.statuses != nullptr;
                else return true;
              }()));
    } else {
      return phase_ != Phase::unbound && domain_ != nullptr;
    }
  }
  [[nodiscard]] constexpr bool active() const noexcept {
    return phase_ == Phase::issued || phase_ == Phase::completed;
  }
  [[nodiscard]] constexpr bool reusable() const noexcept {
    return phase_ == Phase::idle;
  }
  [[nodiscard]] constexpr std::size_t maximum_work_count() const noexcept {
    if constexpr (dynamic) {
      return storage_.capacity;
    } else {
      return Extent;
    }
  }
  [[nodiscard]] constexpr std::size_t active_work_count() const noexcept {
    if constexpr (dynamic) {
      return storage_.active_count;
    } else {
      return active() ? Extent : 0;
    }
  }

  [[nodiscard]] ErrorCode
  bind(const MpiCallDomain &domain, MPI_Request *const requests,
       MPI_Status *const statuses, const std::size_t capacity) noexcept
    requires(dynamic && RetainStatuses)
  {
    if (capacity != 0 && statuses == nullptr)
      return ErrorCode::invalid_argument;
    LPFX_TRY(bind_slots(domain, requests, capacity));
    storage_.statuses = statuses;
    return ErrorCode::success;
  }

  [[nodiscard]] ErrorCode
  bind(const MpiCallDomain &domain, MPI_Request *const requests,
       const std::size_t capacity) noexcept
    requires(dynamic && !RetainStatuses)
  {
    return bind_slots(domain, requests, capacity);
  }

  [[nodiscard]] ErrorCode bind(const MpiCallDomain &domain) noexcept
    requires(!dynamic)
  {
    if (phase_ != Phase::unbound || !domain.active()) {
      return ErrorCode::invalid_argument;
    }
    storage_.request = MPI_REQUEST_NULL;
    domain_ = &domain;
    phase_ = Phase::idle;
    return ErrorCode::success;
  }

  template <class Issue>
    requires(!dynamic && std::is_nothrow_invocable_r_v<ErrorCode, Issue &,
                                                       std::span<MPI_Request>>)
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start(Issue &&issue) noexcept {
    return start_impl(Extent, std::forward<Issue>(issue));
  }

  // Hard recipe precondition: before this call, the protocol owner has
  // validated every pointer, count, peer, tag, root, datatype, operation, and
  // communicator-dependent bound used by issue. Once issue is entered, any
  // reported/native failure is fail-stop because work may have escaped.
  template <class Issue>
    requires(dynamic && std::is_nothrow_invocable_r_v<ErrorCode, Issue &,
                                                      std::span<MPI_Request>>)
  [[nodiscard]] LPFX_ALWAYS_INLINE Status start(const std::size_t request_count,
                                                Issue &&issue) noexcept {
    return start_impl(request_count, std::forward<Issue>(issue));
  }

  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() noexcept {
    if (phase_ == Phase::completed) {
      return Status::success();
    }
    if (phase_ != Phase::issued) {
      return Status::failure(ErrorCode::invalid_state);
    }
    MpiCallLease lease{};
    if (try_acquire_disjoint_mpi_call(*domain_, lease) != ErrorCode::success) {
      std::abort();
    }
    // Admission above established an issued batch. Fixed-one completion must
    // use its sealed extent, without re-reading the public activity query after
    // native thread/gate calls.
    const int request_count = [&]() noexcept {
      if constexpr (dynamic) {
        return static_cast<int>(storage_.active_count);
      } else {
        return static_cast<int>(Extent);
      }
    }();
    if constexpr (Wait) {
      if (MPI_Waitall(request_count, requests(), statuses()) != MPI_SUCCESS) {
        std::abort();
      }
    } else {
      int finished = 0;
      if (MPI_Testall(request_count, requests(), &finished, statuses()) !=
          MPI_SUCCESS) {
        std::abort();
      }
      if (finished == 0) {
        return Status::pending();
      }
    }
    phase_ = Phase::completed;
    return Status::success();
  }

  // Protocol receive audits use the completed statuses before release. This
  // lends the same program-owned backing without another completion owner.
  [[nodiscard]] std::span<const MPI_Status> completed_statuses() const noexcept
    requires(dynamic && RetainStatuses)
  {
    assert(phase_ == Phase::completed);
    return {storage_.statuses, storage_.active_count};
  }

  [[nodiscard]] Status release() noexcept {
    if (phase_ != Phase::completed) {
      return Status::failure(ErrorCode::invalid_state);
    }
#ifndef NDEBUG
    for (std::size_t index = 0; index != active_work_count(); ++index) {
      if (requests()[index] != MPI_REQUEST_NULL) {
        std::abort();
      }
    }
#endif
    if constexpr (dynamic) {
      storage_.active_count = 0;
    }
    phase_ = Phase::idle;
    return Status::success();
  }
  void reset() noexcept {
    if (phase_ == Phase::issued || phase_ == Phase::completed) {
      std::abort();
    }
    domain_ = nullptr;
    storage_ = {};
    phase_ = Phase::unbound;
  }

private:
  [[nodiscard]] ErrorCode
  bind_slots(const MpiCallDomain &domain, MPI_Request *const requests,
             const std::size_t capacity) noexcept
    requires(dynamic)
  {
    if (phase_ != Phase::unbound || !domain.active() ||
        (capacity != 0 && requests == nullptr) ||
        capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      return ErrorCode::invalid_argument;
    std::fill_n(requests, capacity, MPI_REQUEST_NULL);
    domain_ = &domain;
    storage_.requests = requests;
    storage_.capacity = capacity;
    phase_ = Phase::idle;
    return ErrorCode::success;
  }

  template <class Issue>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status
  start_impl(const std::size_t request_count, Issue &&issue) noexcept {
    if (request_count > maximum_work_count() || phase_ != Phase::idle) {
      return Status::failure(ErrorCode::invalid_state);
    }
    assert(is_valid());
    if (request_count == 0) {
      if constexpr (dynamic) {
        storage_.active_count = 0;
      }
      phase_ = Phase::completed;
      return Status::success();
    }

#ifndef NDEBUG
    for (std::size_t index = 0; index != request_count; ++index) {
      if (requests()[index] != MPI_REQUEST_NULL) {
        std::abort();
      }
    }
#endif

    MpiCallLease lease{};
    if (try_acquire_disjoint_mpi_call(*domain_, lease) != ErrorCode::success) {
      std::abort();
    }
    // Admission is complete and issue failure is fail-stop. Publish the count
    // before the native calls so it need not stay live across the issue loop;
    // the issued phase is still published only after every call succeeds.
    if constexpr (dynamic) {
      storage_.active_count = request_count;
    }
    if (issue(std::span<MPI_Request>{requests(), request_count}) !=
        ErrorCode::success) {
      std::abort();
    }
    phase_ = Phase::issued;
    return Status::pending();
  }

  [[nodiscard]] MPI_Request *requests() noexcept {
    if constexpr (dynamic) {
      return storage_.requests;
    } else {
      return &storage_.request;
    }
  }

  [[nodiscard]] MPI_Status *statuses() noexcept {
    if constexpr (dynamic && RetainStatuses) {
      return storage_.statuses;
    } else {
      return MPI_STATUSES_IGNORE;
    }
  }

  enum class Phase : std::uint8_t {
    unbound = 0,
    // Keep idle distinct from public pending and collective native-phase tags.
    // GCC can otherwise preserve a checked constant through unrelated calls.
    idle = 0x40,
    issued = 2,
    completed = 3,
  };

  const MpiCallDomain *domain_ = nullptr;
  std::conditional_t<dynamic, DynamicMpiRequestStorage<RetainStatuses>,
                     SingleMpiRequestStorage>
      storage_{};
  Phase phase_ = Phase::unbound;
};

using PreparedMpiRequestBatch =
    BasicPreparedMpiRequestBatch<std::dynamic_extent>;
using PreparedMpiRequestBatchNoStatus =
    BasicPreparedMpiRequestBatch<std::dynamic_extent, false>;
using PreparedMpiRequestBatchOne = BasicPreparedMpiRequestBatch<1>;

} // namespace lpfx::detail

namespace lpfx::distributed::detail {

template <class Byte> struct MpiFixedByteChunk final {
  Byte *data = nullptr;
  int count = 0;
  int peer = -1;
  int tag = -1;
};

// Fixed-message MPI P2P for FixedBytePort: immutable Irecv/Isend recipes plus
// the common completion owner. Every message is split once during preparation
// into INT_MAX-byte chunks that keep the message's tag and order; MPI's
// non-overtaking rule supplies matching within a message.
class MpiP2p final {
  using Send = MpiFixedByteChunk<const std::byte>;
  using Receive = MpiFixedByteChunk<std::byte>;

public:
  constexpr MpiP2p() noexcept = default;
  MpiP2p(const MpiP2p &) = delete;
  MpiP2p &operator=(const MpiP2p &) = delete;

  [[nodiscard]] bool empty() const noexcept { return requests_.empty(); }
  [[nodiscard]] bool is_valid() const noexcept { return requests_.is_valid(); }
  [[nodiscard]] bool has_incoming_work() const noexcept {
    return !receives_.empty();
  }
  [[nodiscard]] std::size_t work_count() const noexcept {
    return requests_.maximum_work_count();
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return storage_.capacity_bytes();
  }

  // address maps a family byte pointer to its native (possibly staged) byte.
  template <class Session, class Address>
  [[nodiscard]] ErrorCode
  try_prepare(HostMemoryBudget &budget, Session &session,
              const FixedByteExchange &shape,
              const std::span<const int> route_tags, Address &&address) noexcept {
    if (!session.mpi().has_payload_context()) {
      return ErrorCode::invalid_state;
    }
    const MpiPayloadContext &context = session.mpi().payload();
    std::size_t send_count = 0;
    std::size_t receive_count = 0;
    LPFX_TRY(count_chunks(context, shape.sends, route_tags, send_count));
    LPFX_TRY(count_chunks(context, shape.receives, route_tags, receive_count));
    std::size_t request_count = 0;
    if (!lpfx::detail::checked_add(send_count, receive_count, request_count) ||
        request_count >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return ErrorCode::overflow;
    }
    Workspace::pointers_type arrays{};
    LPFX_TRY(Workspace::try_allocate(
        budget, {receive_count, send_count, request_count}, storage_, arrays));
    fill_chunks(shape.receives, route_tags, address, std::get<0>(arrays));
    fill_chunks(shape.sends, route_tags, address, std::get<1>(arrays));
    const ErrorCode bound =
        requests_.bind(context.domain(), std::get<2>(arrays), request_count);
    if (bound != ErrorCode::success) {
      storage_.reset_capacity();
      return bound;
    }
    receives_ = {std::get<0>(arrays), receive_count};
    sends_ = {std::get<1>(arrays), send_count};
    communicator_ = context.native_handle();
    return ErrorCode::success;
  }

  template <class Executor>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status submit(const Executor &) noexcept {
    struct Issue {
      MpiP2p &owner;

      [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
      operator()(std::span<MPI_Request> requests) const noexcept {
        std::size_t request = 0;
        for (const Receive &chunk : owner.receives_) {
          if (MPI_Irecv(chunk.data, chunk.count, MPI_BYTE, chunk.peer, chunk.tag,
                        owner.communicator_, &requests[request++]) != MPI_SUCCESS) {
            return ErrorCode::backend_failure;
          }
        }
        for (const Send &chunk : owner.sends_) {
          if (MPI_Isend(chunk.data, chunk.count, MPI_BYTE, chunk.peer, chunk.tag,
                        owner.communicator_, &requests[request++]) != MPI_SUCCESS) {
            return ErrorCode::backend_failure;
          }
        }
        return ErrorCode::success;
      }
    };
    return requests_.start(requests_.maximum_work_count(), Issue{*this});
  }

  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() noexcept {
    return requests_.template advance<Wait>();
  }
  [[nodiscard]] Status release() noexcept { return requests_.release(); }
  void reset() noexcept {
    requests_.reset();
    receives_ = {};
    sends_ = {};
    communicator_ = MPI_COMM_NULL;
    storage_.reset_capacity();
  }

private:
  static constexpr std::size_t maximum_chunk_bytes =
      static_cast<std::size_t>(std::numeric_limits<int>::max());

  template <class Message>
  [[nodiscard]] static ErrorCode
  count_chunks(const MpiPayloadContext &context,
               const std::span<const Message> messages,
               const std::span<const int> route_tags,
               std::size_t &output) noexcept {
    for (const Message &message : messages) {
      if ((message.size_bytes != 0 && message.data == nullptr) ||
          message.route >= route_tags.size() ||
          message.peer >= static_cast<std::uint32_t>(context.size()) ||
          message.peer == static_cast<std::uint32_t>(context.rank())) {
        return ErrorCode::invalid_argument;
      }
      const std::size_t chunks =
          message.size_bytes / maximum_chunk_bytes +
          (message.size_bytes % maximum_chunk_bytes != 0);
      if (!lpfx::detail::checked_add(output, chunks, output)) {
        return ErrorCode::overflow;
      }
    }
    return ErrorCode::success;
  }

  template <class Message, class Chunk, class Address>
  static void fill_chunks(const std::span<const Message> messages,
                          const std::span<const int> route_tags,
                          Address &address, Chunk *output) noexcept {
    for (const Message &message : messages) {
      const auto data =
          message.size_bytes == 0 ? nullptr : address(message.data);
      for (std::size_t offset = 0; offset != message.size_bytes;) {
        const std::size_t bytes =
            std::min(maximum_chunk_bytes, message.size_bytes - offset);
        *output++ = {data + offset, static_cast<int>(bytes),
                     static_cast<int>(message.peer), route_tags[message.route]};
        offset += bytes;
      }
    }
  }

  // The bare allocation: receives_/sends_ and the request batch hold the typed
  // views.
  using Workspace = HostWorkspace<Receive, Send, MPI_Request>;
  Workspace::allocation_type storage_{};
  std::span<const Receive> receives_{};
  std::span<const Send> sends_{};
  MPI_Comm communicator_ = MPI_COMM_NULL;
  lpfx::detail::PreparedMpiRequestBatchNoStatus requests_{};
};

} // namespace lpfx::distributed::detail

#endif
