#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_AGREEMENT_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_AGREEMENT_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>

#include <mpi.h>

#include <lpfx/core/attributes.hpp>
#include <lpfx/distributed/transport/mpi/setup.hpp>

// Every agreement is one rank-symmetric collective. A native failure leaves
// peers without a common outcome and is therefore fatal.
namespace lpfx::distributed::detail {

// A coordinator exposes domain() and native_handle(): a CommunicationSetup
// (validated here) or an MPI context (whose activity is a caller precondition).
template <class Coordinator>
[[nodiscard]] LPFX_ALWAYS_INLINE bool
admit_shared_call(const Coordinator &coordinator, MpiCallLease &lease) noexcept {
  if constexpr (std::same_as<Coordinator, CommunicationSetup>) {
    if (!coordinator.is_valid()) {
      return false;
    }
  } else {
    assert(coordinator.active());
  }
  return try_acquire_shared_mpi_call(coordinator.domain(), lease) ==
         ErrorCode::success;
}

// Runs call(communicator) under a shared lease; invalid_state when the
// coordinator cannot be admitted.
template <class Coordinator, class Call>
[[nodiscard]] MpiResult with_shared_call(const Coordinator &coordinator,
                                         Call &&call) noexcept {
  MpiCallLease lease{};
  if (!admit_shared_call(coordinator, lease)) {
    return ErrorCode::invalid_state;
  }
  return call(coordinator.native_handle());
}

// Agrees the largest ErrorCode.
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
agree_or_abort(const MPI_Comm communicator, const ErrorCode local) noexcept {
  const auto input = static_cast<std::uint32_t>(local);
  std::uint32_t agreed = 0;
  if (MPI_Allreduce(&input, &agreed, 1, MPI_UINT32_T, MPI_MAX, communicator) !=
          MPI_SUCCESS ||
      agreed > static_cast<std::uint32_t>(ErrorCode::backend_failure)) {
    std::abort();
  }
  return static_cast<ErrorCode>(agreed);
}

template <class Coordinator>
  requires(!std::same_as<Coordinator, MPI_Comm>)
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
agree_or_abort(const Coordinator &coordinator, const ErrorCode local) noexcept {
  MpiCallLease lease{};
  if (!admit_shared_call(coordinator, lease)) {
    std::abort();
  }
  return agree_or_abort(coordinator.native_handle(), local);
}

// One MAX reduction of {local error, maxima, each invariant and its
// complement}. Returns the agreed error, then invalid_argument when any
// invariant differs across ranks; maxima receive the agreed maxima.
template <std::size_t Invariants, std::size_t Maxima>
[[nodiscard]] ErrorCode
agree_contract(const MPI_Comm communicator, const ErrorCode local,
               const std::array<std::uint64_t, Invariants> &invariants,
               std::array<std::uint64_t, Maxima> &maxima) noexcept {
  std::array<std::uint64_t, 1 + Maxima + 2 * Invariants> words{};
  words[0] = static_cast<std::uint64_t>(local);
  for (std::size_t index = 0; index != Maxima; ++index) {
    words[1 + index] = maxima[index];
  }
  for (std::size_t index = 0; index != Invariants; ++index) {
    words[1 + Maxima + 2 * index] = invariants[index];
    words[2 + Maxima + 2 * index] = ~invariants[index];
  }
  if (MPI_Allreduce(MPI_IN_PLACE, words.data(), static_cast<int>(words.size()),
                    MPI_UINT64_T, MPI_MAX, communicator) != MPI_SUCCESS ||
      words[0] > static_cast<std::uint64_t>(ErrorCode::backend_failure)) {
    std::abort();
  }
  if (words[0] != static_cast<std::uint64_t>(ErrorCode::success)) {
    return static_cast<ErrorCode>(words[0]);
  }
  for (std::size_t index = 0; index != Invariants; ++index) {
    if (words[1 + Maxima + 2 * index] != ~words[2 + Maxima + 2 * index]) {
      return ErrorCode::invalid_argument;
    }
  }
  for (std::size_t index = 0; index != Maxima; ++index) {
    maxima[index] = words[1 + index];
  }
  return ErrorCode::success;
}

template <class Coordinator, std::size_t Invariants, std::size_t Maxima>
  requires(!std::same_as<Coordinator, MPI_Comm>)
[[nodiscard]] ErrorCode
agree_contract(const Coordinator &coordinator, const ErrorCode local,
               const std::array<std::uint64_t, Invariants> &invariants,
               std::array<std::uint64_t, Maxima> &maxima) noexcept {
  MpiCallLease lease{};
  if (!admit_shared_call(coordinator, lease)) {
    return ErrorCode::invalid_state;
  }
  return agree_contract(coordinator.native_handle(), local, invariants,
                        maxima);
}

// Broadcasts bytes from root after agreeing their size and root.
template <class Coordinator>
[[nodiscard]] ErrorCode broadcast_bytes(const Coordinator &coordinator,
                                        const std::span<std::byte> bytes,
                                        const int root) noexcept {
  const ErrorCode local =
      root < 0 || root >= coordinator.size() ||
              (!bytes.empty() && bytes.data() == nullptr)
          ? ErrorCode::invalid_argument
          : (bytes.size() > static_cast<std::size_t>(
                                std::numeric_limits<int>::max())
                 ? ErrorCode::overflow
                 : ErrorCode::success);
  return with_shared_call(coordinator, [&](const MPI_Comm communicator) noexcept {
    std::array<std::uint64_t, 0> maxima{};
    const ErrorCode agreed = agree_contract(
        communicator, local,
        std::array<std::uint64_t, 2>{
            static_cast<std::uint64_t>(bytes.size()),
            static_cast<std::uint64_t>(root < 0 ? 0 : root)},
        maxima);
    if (agreed != ErrorCode::success) {
      return MpiResult{agreed};
    }
    if (MPI_Bcast(bytes.data(), static_cast<int>(bytes.size()), MPI_BYTE, root,
                  communicator) != MPI_SUCCESS) {
      std::abort();
    }
    return MpiResult{};
  }).code();
}

} // namespace lpfx::distributed::detail

#endif
