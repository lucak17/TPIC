#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_REGISTRATION_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_REGISTRATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <tuple>

#include <mpi.h>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/registration/image.hpp>
#include <lpfx/distributed/transport/mpi/agreement.hpp>
#include <lpfx/distributed/transport/mpi/communicator.hpp>
#include <lpfx/distributed/transport/workspace.hpp>

namespace lpfx::distributed::detail {

// The length Allgather writes entries [1,P]. Convert those entries in place;
// no second rank-size array or full local encoded image is necessary.
[[nodiscard]] inline ErrorCode
try_registration_prefix_offsets(const std::span<std::uint64_t> offsets,
                                std::size_t &total) noexcept {
  if (offsets.size() < 2 || offsets.data() == nullptr)
    return ErrorCode::invalid_argument;
  offsets[0] = 0;
  for (std::size_t rank = 1; rank != offsets.size(); ++rank) {
    if (offsets[rank] == 0)
      return ErrorCode::invalid_argument;
    if (!lpfx::detail::checked_add(offsets[rank - 1], offsets[rank],
                                   offsets[rank]))
      return ErrorCode::overflow;
  }
  return lpfx::detail::checked_integral_cast(offsets.back(), total)
             ? ErrorCode::success
             : ErrorCode::overflow;
}

// Pure layout of one Allgatherv window of at most limit bytes starting at
// base; the caller supplies a common limit on all ranks.
[[nodiscard]] inline ErrorCode try_registration_gather_window(
    const std::span<const std::uint64_t> offsets, const std::uint64_t base,
    const std::uint64_t limit, const std::span<int> counts,
    const std::span<int> displacements, std::uint64_t &end) noexcept {
  if (offsets.size() < 2 || offsets.data() == nullptr ||
      counts.size() != offsets.size() - 1 ||
      displacements.size() != counts.size() || counts.data() == nullptr ||
      displacements.data() == nullptr || offsets[0] != 0 ||
      base >= offsets.back() || limit == 0 ||
      limit > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    return ErrorCode::invalid_argument;
  // Subtract before adding so the final partial window cannot overflow.
  end = base + std::min(limit, offsets.back() - base);
  for (std::size_t rank = 0; rank != counts.size(); ++rank) {
    if (offsets[rank] > offsets[rank + 1])
      return ErrorCode::invalid_argument;
    const auto begin = std::max(base, offsets[rank]);
    const auto finish = std::min(end, offsets[rank + 1]);
    counts[rank] = 0;
    displacements[rank] = 0;
    if (begin < finish &&
        (!lpfx::detail::checked_integral_cast(finish - begin, counts[rank]) ||
         !lpfx::detail::checked_integral_cast(begin - base,
                                              displacements[rank])))
      return ErrorCode::overflow;
  }
  return ErrorCode::success;
}

// Rank image directory (offsets, counts, displacements) and gathered images.
class MpiRegistrationExchange final {
public:
  [[nodiscard]] ErrorCode prepare_directory(HostMemoryBudget &budget,
                                            const int rank_count) noexcept {
    if (rank_count <= 0)
      return ErrorCode::invalid_argument;
    const auto ranks = static_cast<std::size_t>(rank_count);
    return directory_.try_prepare(budget, {ranks + 1, ranks, ranks});
  }

  [[nodiscard]] ErrorCode
  gather_lengths(const MPI_Comm communicator,
                 const std::uint64_t local_bytes) noexcept {
    if (MPI_Allgather(&local_bytes, 1, MPI_UINT64_T,
                      directory_.data<0>() + 1, 1, MPI_UINT64_T,
                      communicator) != MPI_SUCCESS)
      std::abort();
    return try_registration_prefix_offsets(directory_.span<0>(), total_bytes_);
  }

  [[nodiscard]] ErrorCode prepare_payload(HostMemoryBudget &budget) noexcept {
    if (total_bytes_ == 0)
      return ErrorCode::invalid_state;
    return payload_.try_prepare(budget, {total_bytes_});
  }

  [[nodiscard]] std::span<std::byte>
  local_image(const std::size_t rank) noexcept {
    const auto offsets = directory_.span<0>();
    if (rank + 1 >= offsets.size() || payload_.size<0>() != total_bytes_)
      return {};
    return {payload_.data<0>() + static_cast<std::size_t>(offsets[rank]),
            static_cast<std::size_t>(offsets[rank + 1] - offsets[rank])};
  }

  void gather_payload(const MPI_Comm communicator) noexcept {
    const auto offsets = directory_.span<0>();
    for (std::uint64_t base = 0; base < offsets.back();) {
      std::uint64_t end{};
      if (try_registration_gather_window(
              offsets, base,
              static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
              directory_.span<1>(), directory_.span<2>(),
              end) != ErrorCode::success ||
          MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE,
                         payload_.data<0>() + static_cast<std::size_t>(base),
                         directory_.data<1>(), directory_.data<2>(), MPI_BYTE,
                         communicator) != MPI_SUCCESS)
        std::abort();
      base = end;
    }
  }

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return payload_.span<0>();
  }
  [[nodiscard]] std::span<const std::uint64_t> offsets() const noexcept {
    return directory_.span<0>();
  }

private:
  HostWorkspace<std::uint64_t, int, int> directory_{};
  HostWorkspace<std::byte> payload_{};
  std::size_t total_bytes_{};
};

} // namespace lpfx::distributed::detail

namespace lpfx {

// Gathers every rank's registration image of plans (canonical channel order)
// and validates them before the seal consumes the domain's registration
// epoch. Native session construction comes after return, when every transient
// admission has been released.
template <class... Plans>
[[nodiscard]] ErrorCode
try_seal_mpi_registration(const distributed::CommunicationSetup &setup,
                          const std::tuple<const Plans *...> &plans,
                          HostMemoryBudget &budget,
                          MpiRegistrationSeal &output) noexcept {
  if (output.active() || !setup.is_valid() ||
      setup.domain().registration_sealed())
    return ErrorCode::invalid_state;
  const MpiCallDomain &domain = setup.domain();
  MpiCallLease lease{};
  if (try_acquire_shared_mpi_call(domain, lease) != ErrorCode::success)
    return ErrorCode::invalid_state;
  const MPI_Comm communicator = domain.bootstrap_communicator();
  const auto agree = [communicator](const ErrorCode local) noexcept {
    return distributed::detail::agree_or_abort(communicator, local);
  };

  distributed::detail::RegistrationImageLayout layout{};
  distributed::detail::MpiRegistrationExchange exchange{};
  ErrorCode local =
      distributed::detail::try_layout_registration_image(plans, layout);
  std::uint64_t local_bytes{};
  if (local == ErrorCode::success &&
      !detail::checked_integral_cast(layout.image_bytes, local_bytes))
    local = ErrorCode::overflow;
  if (local == ErrorCode::success)
    local = exchange.prepare_directory(budget, domain.size());
  LPFX_TRY(agree(local));

  local = exchange.gather_lengths(communicator, local_bytes);
  if (local == ErrorCode::success)
    local = exchange.prepare_payload(budget);
  if (local == ErrorCode::success)
    local = distributed::detail::encode_registration_image(
        plans, layout,
        exchange.local_image(static_cast<std::size_t>(domain.rank())));
  LPFX_TRY(agree(local));

  exchange.gather_payload(communicator);
  LPFX_TRY(agree(distributed::detail::validate_registration_images(
      {exchange.bytes(), exchange.offsets()})));

  detail::seal_mpi_registration(domain);
  output = detail::PrivateConstructionAccess::create<MpiRegistrationSeal>(domain);
  return ErrorCode::success;
}

} // namespace lpfx

#endif
