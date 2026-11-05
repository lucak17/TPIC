#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_COLLECTIVE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_COLLECTIVE_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/collective/port.hpp>
#include <lpfx/distributed/transport/mpi/communicator.hpp>
#include <lpfx/distributed/transport/mpi/p2p.hpp>

namespace lpfx::distributed::detail {

template <class T> struct MpiCollectiveDatatype;

template <> struct MpiCollectiveDatatype<float> final {
  inline static const MPI_Datatype value = MPI_FLOAT;
};
template <> struct MpiCollectiveDatatype<double> final {
  inline static const MPI_Datatype value = MPI_DOUBLE;
};
template <> struct MpiCollectiveDatatype<std::int8_t> final {
  inline static const MPI_Datatype value = MPI_INT8_T;
};
template <> struct MpiCollectiveDatatype<std::uint8_t> final {
  inline static const MPI_Datatype value = MPI_UINT8_T;
};
template <> struct MpiCollectiveDatatype<std::int16_t> final {
  inline static const MPI_Datatype value = MPI_INT16_T;
};
template <> struct MpiCollectiveDatatype<std::uint16_t> final {
  inline static const MPI_Datatype value = MPI_UINT16_T;
};
template <> struct MpiCollectiveDatatype<std::int32_t> final {
  inline static const MPI_Datatype value = MPI_INT32_T;
};
template <> struct MpiCollectiveDatatype<std::uint32_t> final {
  inline static const MPI_Datatype value = MPI_UINT32_T;
};
template <> struct MpiCollectiveDatatype<std::int64_t> final {
  inline static const MPI_Datatype value = MPI_INT64_T;
};
template <> struct MpiCollectiveDatatype<std::uint64_t> final {
  inline static const MPI_Datatype value = MPI_UINT64_T;
};

template <class T>
concept MpiCollectiveScalar = requires {
  { MpiCollectiveDatatype<T>::value } -> std::convertible_to<MPI_Datatype>;
};

template <reduction_operation Reduction>
[[nodiscard]] MPI_Op mpi_collective_reduction() noexcept {
  if constexpr (Reduction == reduction_operation::sum) {
    return MPI_SUM;
  } else if constexpr (Reduction == reduction_operation::product) {
    return MPI_PROD;
  } else if constexpr (Reduction == reduction_operation::minimum) {
    return MPI_MIN;
  } else {
    static_assert(Reduction == reduction_operation::maximum);
    return MPI_MAX;
  }
}

// Kind, reduction, datatype, in-place mode, and blocking capability are all
// compile-time facts. Count/root are cold-validated native integers.
template <bool Nonblocking, collective_kind Kind, reduction_operation Reduction,
          bool ExactInPlace, MpiCollectiveScalar T, class... Completion>
[[nodiscard]] int invoke_mpi_collective(
    const T *const input, T *const output, const int count, const int root,
    const int local_rank, const MPI_Comm communicator,
    Completion &&...completion) noexcept {
  static_assert(sizeof...(Completion) == (Nonblocking ? 1 : 0));
  const MPI_Datatype datatype = MpiCollectiveDatatype<T>::value;
  if constexpr (Kind == collective_kind::all_reduce) {
    const void *source = ExactInPlace ? MPI_IN_PLACE : input;
    if constexpr (Nonblocking) {
      return MPI_Iallreduce(source, output, count, datatype,
                            mpi_collective_reduction<Reduction>(), communicator,
                            (&completion)...);
    } else {
      return MPI_Allreduce(source, output, count, datatype,
                           mpi_collective_reduction<Reduction>(), communicator);
    }
  } else if constexpr (Kind == collective_kind::reduce) {
    const void *source =
        ExactInPlace && local_rank == root ? MPI_IN_PLACE : input;
    if constexpr (Nonblocking) {
      return MPI_Ireduce(source, output, count, datatype,
                         mpi_collective_reduction<Reduction>(), root,
                         communicator, (&completion)...);
    } else {
      return MPI_Reduce(source, output, count, datatype,
                        mpi_collective_reduction<Reduction>(), root,
                        communicator);
    }
  } else if constexpr (Kind == collective_kind::broadcast) {
    if constexpr (Nonblocking) {
      return MPI_Ibcast(output, count, datatype, root, communicator,
                        (&completion)...);
    } else {
      return MPI_Bcast(output, count, datatype, root, communicator);
    }
  } else if constexpr (Kind == collective_kind::all_gather) {
    static_assert(!ExactInPlace);
    if constexpr (Nonblocking) {
      return MPI_Iallgather(input, count, datatype, output, count, datatype,
                            communicator, (&completion)...);
    } else {
      return MPI_Allgather(input, count, datatype, output, count, datatype,
                           communicator);
    }
  } else {
    static_assert(Kind == collective_kind::reduce_scatter);
    static_assert(!ExactInPlace);
    if constexpr (Nonblocking) {
      return MPI_Ireduce_scatter_block(
          input, output, count, datatype,
          mpi_collective_reduction<Reduction>(), communicator,
          (&completion)...);
    } else {
      return MPI_Reduce_scatter_block(
          input, output, count, datatype,
          mpi_collective_reduction<Reduction>(), communicator);
    }
  }
}

struct NoMpiCollectiveRequest final {};

// MPI collective call policy for CollectivePort: the payload context and,
// when nonblocking, one prepared request. Blocking calls complete in submit.
template <collective_kind Kind, reduction_operation Reduction,
          MpiCollectiveScalar T, bool ExactInPlace, bool Nonblocking>
class MpiCollectiveCall final {
public:
  using value_type = T;

  [[nodiscard]] bool empty() const noexcept {
    if constexpr (Nonblocking) {
      return context_ == nullptr && requests_.empty();
    } else {
      return context_ == nullptr;
    }
  }

  // The family has established count relationships and rank-specific access;
  // only native integer bounds and rank consistency remain.
  [[nodiscard]] static constexpr bool
  valid_recipe(const CollectivePreparedShape &shape, const int rank,
               const int rank_count, const bool require_native_work) noexcept {
    return rank >= 0 && rank_count > 0 && rank < rank_count &&
           shape.rank == static_cast<std::uint32_t>(rank) &&
           shape.rank_count == static_cast<std::uint32_t>(rank_count) &&
           (!require_native_work || shape.has_native_work()) &&
           shape.native_count <=
               static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
           (shape.root == no_collective_root ||
            shape.root <=
                static_cast<std::uint32_t>(std::numeric_limits<int>::max()));
  }

  [[nodiscard]] bool
  is_valid(const CollectivePreparedShape &shape) const noexcept {
    bool bound = context_ != nullptr && context_->active();
    if constexpr (Nonblocking) {
      bound = bound && requests_.is_valid();
    }
    return bound &&
           valid_recipe(shape, context_->rank(), context_->size(), true);
  }

  template <class Session, class Setup, class Executor>
    requires requires(Session &session) {
      { session.mpi().is_valid() } noexcept -> std::same_as<bool>;
      { session.mpi().has_payload_context() } noexcept -> std::same_as<bool>;
      { session.mpi().payload() } noexcept -> std::same_as<MpiPayloadContext &>;
    }
  [[nodiscard]] ErrorCode try_prepare(Session &session, const Setup &,
                                      const Executor &) noexcept {
    auto &mpi = session.mpi();
    if (!mpi.is_valid() || !mpi.has_payload_context()) {
      return ErrorCode::invalid_state;
    }
    if constexpr (Nonblocking) {
      LPFX_TRY(requests_.bind(mpi.payload().domain()));
    }
    context_ = &mpi.payload();
    return ErrorCode::success;
  }

  // operands() is evaluated after thread admission; delaying operand/count
  // reads avoids carrying eager snapshots across the thread/gate calls.
  template <std::uint32_t Root, class Operands, class Executor>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status submit(Operands &&operands,
                                                 const Executor &) noexcept {
    static_assert(Root == no_collective_root || std::in_range<int>(Root));
    constexpr int root = Root == no_collective_root ? -1 : static_cast<int>(Root);
    if constexpr (Nonblocking) {
      // A named one-shot callable preserves force-inline through NVCC's host
      // lowering, which drops attributes attached directly to lambda operators.
      // The context is read inside the issue, after request admission.
      struct NativeIssue final {
        const MpiCollectiveCall &call;
        Operands &operands;
        [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
        operator()(std::span<MPI_Request> requests) const noexcept {
          const auto args = operands();
          const MpiPayloadContext *const context = call.context_;
          const int native =
              invoke_mpi_collective<true, Kind, Reduction, ExactInPlace>(
                  args.input, args.output, static_cast<int>(args.count), root,
                  context->rank(), context->native_handle(), requests[0]);
          return native == MPI_SUCCESS ? ErrorCode::success
                                       : ErrorCode::backend_failure;
        }
      };
      return requests_.start(NativeIssue{*this, operands});
    } else {
      MpiCallLease lease{};
      if (try_acquire_disjoint_mpi_call(context_->domain(), lease) !=
          ErrorCode::success) {
        std::abort();
      }
      const auto args = operands();
      if (invoke_mpi_collective<false, Kind, Reduction, ExactInPlace>(
              args.input, args.output, static_cast<int>(args.count), root,
              context_->rank(), context_->native_handle()) != MPI_SUCCESS) {
        std::abort();
      }
      return Status::success();
    }
  }

  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() noexcept {
    if constexpr (Nonblocking) {
      return requests_.template advance<Wait>();
    } else {
      std::abort();
    }
  }
  [[nodiscard]] Status release() noexcept {
    if constexpr (Nonblocking) {
      return requests_.release();
    } else {
      return Status::success();
    }
  }
  void reset() noexcept {
    if constexpr (Nonblocking) {
      if (requests_.active()) {
        std::abort();
      }
      requests_.reset();
    }
    context_ = nullptr;
  }

private:
  const MpiPayloadContext *context_ = nullptr;
  [[no_unique_address]] std::conditional_t<
      Nonblocking, lpfx::detail::PreparedMpiRequestBatchOne,
      NoMpiCollectiveRequest> requests_{};
};

} // namespace lpfx::distributed::detail

#endif
