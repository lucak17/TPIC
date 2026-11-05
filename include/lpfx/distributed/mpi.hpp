#ifndef LPFX_DISTRIBUTED_MPI_HPP
#define LPFX_DISTRIBUTED_MPI_HPP

// The MPI component supplies native MPI compile/link requirements. Device
// operands additionally use the configured device component and its native
// CUDA/HIP compiler; host consumers need only ordinary C++ and MPI.
#include <array>
#include <cstddef>
#include <cstdint>

#include <lpfx/api/distributed/communication.hpp>
#include <lpfx/api/distributed/particles.hpp>
#include <lpfx/distributed/composition/mpi.hpp>
#include <lpfx/distributed/runtime/host_control.hpp>
#include <lpfx/distributed/runtime/mpi_file_writer.hpp>
#include <lpfx/distributed/transport/mpi/agreement.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>
#include <lpfx/distributed/transport/mpi/particle.hpp>
#include <lpfx/distributed/transport/mpi/setup.hpp>
#include <lpfx/distributed/transport/routes.hpp>
#include <lpfx/distributed/transport/stage.hpp>

namespace lpfx::distributed::detail {
template <> struct ObjectPreparation<CommunicationSetup> final {
  template <class Topology, class Hops>
  [[nodiscard]] static ErrorCode agree_hops(const CommunicationSetup &setup,
      const Topology &topology, std::uint64_t maximum, Hops &hops) noexcept {
    return try_agree_mpi_particle_hop_bound(setup, topology, maximum, hops);
  }
  template <std::size_t Size>
  [[nodiscard]] static ErrorCode agree_words(const CommunicationSetup &setup,
      const std::array<std::uint64_t, Size> &words) noexcept {
    std::array<std::uint64_t, 0> maxima{};
    return agree_contract(setup, ErrorCode::success, words, maxima);
  }
  [[nodiscard]] static ErrorCode agree_error(const CommunicationSetup &setup,
                                            ErrorCode local) noexcept {
    return agree_or_abort(setup, local);
  }
};
} // namespace lpfx::distributed::detail

#endif
