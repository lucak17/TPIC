#ifndef LPFX_DISTRIBUTED_COMPOSITION_ENTRY_HPP
#define LPFX_DISTRIBUTED_COMPOSITION_ENTRY_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <lpfx/backend/traits.hpp>

namespace lpfx::distributed::detail {

enum class CommunicationLoweringIdentity : std::uint64_t {
  mpi_host = UINT64_C(1),
  mpi_staged = UINT64_C(2),
  mpi_direct = UINT64_C(3),
  nccl = UINT64_C(4),
  rccl = UINT64_C(5),
  mpi_host_blocking_collective = UINT64_C(7),
  mpi_staged_blocking_collective = UINT64_C(8),
  mpi_direct_nonblocking_collective = UINT64_C(9),
};

// Distinguishes device runtimes in the cross-rank communication identity.
// Test backends specialize it.
template <class Backend> struct CommunicationDeviceRuntimeIdentity;
template <> struct CommunicationDeviceRuntimeIdentity<void> {
  static constexpr std::uint64_t value = 0;
};
template <> struct CommunicationDeviceRuntimeIdentity<device::CudaBackend> {
  static constexpr std::uint64_t value = UINT64_C(0x4355444100000001);
};
template <> struct CommunicationDeviceRuntimeIdentity<device::HipBackend> {
  static constexpr std::uint64_t value = UINT64_C(0x4849500000000001);
};

// Native resources an operation needs when it has native work on some rank.
struct CommunicationRequirements final {
  bool mpi_payload = false;
  bool xccl = false;
};

// What a lowering provides: the family program for (route, space, family),
// its native resource (the MPI payload context, or the XCCL session of
// XcclNative) and its compile-time identity and tag extent.
template <class Program, CommunicationLoweringIdentity Identity,
          std::size_t TagExtent, class XcclNative = void>
struct LoweredProgram {
  using program_type = Program;
  using xccl_native_type = XcclNative;
  static constexpr bool mpi_payload = std::is_void_v<XcclNative>;
  static constexpr CommunicationLoweringIdentity lowering_identity = Identity;
  static constexpr std::size_t tag_extent = TagExtent;
};

// Transport opt-ins specialize this for the bindings they lower.
template <class Binding> struct CommunicationEntryLowering;

// Defined by the MPI opt-in for XcclNative = void and by the XCCL opt-in for
// its native adapter.
template <bool RetainMpiPayload, class XcclNative> class CommunicationSession;

} // namespace lpfx::distributed::detail

#endif
