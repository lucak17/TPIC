#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_COMMUNICATOR_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_COMMUNICATOR_HPP

#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/transport/mpi/agreement.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>
#include <lpfx/distributed/transport/mpi/setup.hpp>

namespace lpfx::distributed {
template <bool RetainPayloadContext> class MpiSession;
class MpiFileWriter;
} // namespace lpfx::distributed

namespace lpfx {

enum class MpiContextRole : std::uint8_t { control = 0, payload = 1, file = 2 };

namespace detail {

// The caller holds the shared lease of the communicator's domain.
inline void free_communicator_or_abort(MPI_Comm &communicator) noexcept {
  if (communicator != MPI_COMM_NULL &&
      (MPI_Comm_free(&communicator) != MPI_SUCCESS ||
       communicator != MPI_COMM_NULL)) {
    std::abort();
  }
}

} // namespace detail

// One duplicated communicator owned by an MpiSession or an MpiFileWriter.
// Destruction is local and never frees it: the owner releases it after
// rank-symmetric close or rollback admission.
template <MpiContextRole Role> class BasicMpiContext final {
public:
  static constexpr MpiContextRole role = Role;

  constexpr BasicMpiContext() noexcept = default;
  LPFX_NONMOVABLE(BasicMpiContext);
  ~BasicMpiContext() noexcept {
    if (communicator_ != MPI_COMM_NULL) {
      std::abort();
    }
  }

  [[nodiscard]] bool active() const noexcept {
    return communicator_ != MPI_COMM_NULL;
  }
  [[nodiscard]] MPI_Comm native_handle() const noexcept {
    return communicator_;
  }
  // Borrow only while active.
  [[nodiscard]] const MpiCallDomain &domain() const noexcept {
    return *domain_;
  }
  [[nodiscard]] constexpr int rank() const noexcept { return rank_; }
  [[nodiscard]] constexpr int size() const noexcept { return size_; }
  [[nodiscard]] constexpr int tag_upper_bound() const noexcept {
    return tag_upper_bound_;
  }

private:
  // Duplicates the domain communicator under the caller's shared lease and
  // returns the agreed validation error; a rejected duplicate is freed.
  [[nodiscard]] ErrorCode open(const MpiCallDomain &domain) noexcept {
    MPI_Comm communicator = MPI_COMM_NULL;
    if (MPI_Comm_dup(domain.bootstrap_communicator(), &communicator) !=
            MPI_SUCCESS ||
        communicator == MPI_COMM_NULL) {
      std::abort();
    }
    int rank = 0;
    int size = 0;
    int tag_upper_bound = -1;
    MpiResult local =
        MpiResult::native(MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN));
    if (local.succeeded()) {
      local = detail::query_intracommunicator(communicator, rank, size);
    }
    if (local.succeeded() &&
        (rank != domain.rank() || size != domain.size())) {
      local = ErrorCode::backend_failure;
    }
    if (local.succeeded()) {
      void *attribute = nullptr;
      int present = 0;
      local = MpiResult::native(
          MPI_Comm_get_attr(communicator, MPI_TAG_UB, &attribute, &present));
      // MPI guarantees at least 32767 even without the optional attribute.
      tag_upper_bound = present != 0 && attribute != nullptr
                            ? *static_cast<const int *>(attribute)
                            : 32767;
      if (local.succeeded() && tag_upper_bound < 0) {
        local = ErrorCode::backend_failure;
      }
    }
    const ErrorCode agreed = distributed::detail::agree_or_abort(
        domain.bootstrap_communicator(), local.code());
    if (agreed != ErrorCode::success) {
      detail::free_communicator_or_abort(communicator);
      return agreed;
    }
    domain_ = &domain;
    communicator_ = communicator;
    rank_ = rank;
    size_ = size;
    tag_upper_bound_ = tag_upper_bound;
    return ErrorCode::success;
  }

  // Frees the communicator under the caller's shared lease.
  void close() noexcept {
    detail::free_communicator_or_abort(communicator_);
    domain_ = nullptr;
    rank_ = 0;
    size_ = 0;
    tag_upper_bound_ = -1;
  }

  const MpiCallDomain *domain_ = nullptr;
  MPI_Comm communicator_ = MPI_COMM_NULL;
  int rank_ = 0;
  int size_ = 0;
  int tag_upper_bound_ = -1;

  template <bool> friend class distributed::MpiSession;
  friend class distributed::MpiFileWriter;
};

using MpiControlContext = BasicMpiContext<MpiContextRole::control>;
using MpiPayloadContext = BasicMpiContext<MpiContextRole::payload>;
using MpiFileContext = BasicMpiContext<MpiContextRole::file>;

// Proof that the domain's single registration epoch was agreed. It owns no
// native resource and is consumed exactly once by session activation.
class MpiRegistrationSeal final {
public:
  constexpr MpiRegistrationSeal() noexcept = default;
  MpiRegistrationSeal(const MpiRegistrationSeal &) = delete;
  MpiRegistrationSeal &operator=(const MpiRegistrationSeal &) = delete;
  MpiRegistrationSeal(MpiRegistrationSeal &&other) noexcept
      : domain_(std::exchange(other.domain_, nullptr)) {}
  MpiRegistrationSeal &operator=(MpiRegistrationSeal &&other) noexcept {
    domain_ = std::exchange(other.domain_, nullptr);
    return *this;
  }

  [[nodiscard]] constexpr bool active() const noexcept {
    return domain_ != nullptr;
  }
  [[nodiscard]] constexpr bool
  matches(const MpiCallDomain &domain) const noexcept {
    return domain_ == &domain;
  }
  // Discarding is idempotent; the domain stays sealed.
  constexpr void reset() noexcept { domain_ = nullptr; }

private:
  constexpr explicit MpiRegistrationSeal(const MpiCallDomain &domain) noexcept
      : domain_(&domain) {}

  const MpiCallDomain *domain_ = nullptr;

  friend struct detail::PrivateConstructionAccess;
};

} // namespace lpfx

namespace lpfx::distributed {
namespace detail {
struct NoMpiPayloadContextStorage final {};
} // namespace detail

// The bool is the exact compile-time MPI resource union. The false
// representation has no payload-context object or runtime discriminator.
template <bool RetainPayloadContext> class MpiSession final {
public:
  constexpr MpiSession() noexcept = default;
  LPFX_NONMOVABLE(MpiSession);
  ~MpiSession() noexcept {
    if (!empty()) {
      std::abort();
    }
  }

  [[nodiscard]] bool empty() const noexcept { return !control_.active(); }
  [[nodiscard]] bool is_valid() const noexcept { return control_.active(); }
  [[nodiscard]] bool has_payload_context() const noexcept {
    if constexpr (RetainPayloadContext) {
      return payload_.active();
    } else {
      return false;
    }
  }

  [[nodiscard]] MpiControlContext &control() noexcept { return control_; }
  [[nodiscard]] const MpiControlContext &control() const noexcept {
    return control_;
  }
  [[nodiscard]] MpiPayloadContext &payload() noexcept
    requires(RetainPayloadContext)
  {
    return payload_;
  }
  [[nodiscard]] const MpiPayloadContext &payload() const noexcept
    requires(RetainPayloadContext)
  {
    return payload_;
  }

  [[nodiscard]] MpiSession &mpi() noexcept { return *this; }
  [[nodiscard]] const MpiSession &mpi() const noexcept { return *this; }

  // Consumes the seal. Every rank supplies the same create_payload_context,
  // derived from the registered operation shape. Communicator duplication
  // failure is fail-stop; a rank-local validation rejection is agreed and the
  // duplicates are freed in reverse order.
  [[nodiscard]] ErrorCode try_activate(const CommunicationSetup &setup,
                                       MpiRegistrationSeal &&seal,
                                       const bool create_payload_context) noexcept {
    if (!empty() || !setup.is_valid()) {
      return ErrorCode::invalid_state;
    }
    if (create_payload_context && !RetainPayloadContext) {
      return ErrorCode::invalid_argument;
    }
    const MpiCallDomain &domain = setup.domain();
    if (!seal.matches(domain)) {
      return ErrorCode::invalid_state;
    }
    seal.reset();
    MpiCallLease lease{};
    if (try_acquire_shared_mpi_call(domain, lease) != ErrorCode::success) {
      std::abort();
    }
    LPFX_TRY(control_.open(domain));
    if constexpr (RetainPayloadContext) {
      if (create_payload_context) {
        const ErrorCode opened = payload_.open(domain);
        if (opened != ErrorCode::success) {
          control_.close();
          return opened;
        }
      }
    }
    return ErrorCode::success;
  }

  // Called only after rank-symmetric quiescent close or rollback admission.
  // Contexts are freed in payload-then-control dependency order.
  void deactivate() noexcept {
    if (empty()) {
      return;
    }
    MpiCallLease lease{};
    if (try_acquire_shared_mpi_call(control_.domain(), lease) !=
        ErrorCode::success) {
      std::abort();
    }
    if constexpr (RetainPayloadContext) {
      payload_.close();
    }
    control_.close();
  }

private:
  using payload_storage_type =
      std::conditional_t<RetainPayloadContext, MpiPayloadContext,
                         detail::NoMpiPayloadContextStorage>;

  MpiControlContext control_{};
  [[no_unique_address]] payload_storage_type payload_{};
};

static_assert(sizeof(MpiSession<false>) == sizeof(MpiControlContext));

} // namespace lpfx::distributed

#endif
