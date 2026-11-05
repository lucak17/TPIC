#ifndef LPFX_DISTRIBUTED_TRANSPORT_ROUTES_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_ROUTES_HPP

namespace lpfx::distributed::transport {

// Transport routes are compile-time selection only. They own no runtime
// state and do not imply that the corresponding native component is enabled;
// opt-in transport headers provide the supported entry lowerings.
struct MpiHost final {};
struct MpiStaged final {};

template <class Ordering> struct MpiDirect final {
  using ordering_type = Ordering;
};

// Opt-in completion policy for native MPI collectives. Bare MPI routes use
// conventional blocking collectives; only collective lowerings recognize this
// wrapper.
template <class Route> struct NonblockingMpiCollectives final {
  using route_type = Route;
};

struct Nccl final {};
struct Rccl final {};

// Opt-in completion policy for the NCCL/RCCL routes: a fixed exchange or a
// collective completes once its work is enqueued on the binding's executor.
// Work enqueued later on that executor is ordered after it; other streams wait
// on an event recorded there. Particle migration completes as on the bare
// route.
template <class Route> struct StreamOrderedCompletion final {};

} // namespace lpfx::distributed::transport

#endif
