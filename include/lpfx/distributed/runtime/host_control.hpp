#ifndef LPFX_DISTRIBUTED_RUNTIME_HOST_CONTROL_HPP
#define LPFX_DISTRIBUTED_RUNTIME_HOST_CONTROL_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include <mpi.h>

#include <lpfx/distributed/transport/mpi/call_domain.hpp>
#include <lpfx/distributed/transport/mpi/collective.hpp>

namespace lpfx::distributed {
namespace detail {

template <class T>
concept HostControlScalar = MpiCollectiveScalar<T> ||
    std::same_as<T, char> || std::same_as<T, std::byte>;

template <HostControlScalar T>
[[nodiscard]] MPI_Datatype host_control_datatype() noexcept {
  if constexpr (std::same_as<T, char>) return MPI_CHAR;
  else if constexpr (std::same_as<T, std::byte>) return MPI_BYTE;
  else return MpiCollectiveDatatype<T>::value;
}

// A null base is valid only when empty; count is the native element count.
template <class T>
[[nodiscard]] bool valid_host_control_span(std::span<T> values) noexcept {
  return values.empty() || (values.data() != nullptr &&
      values.size() <= std::numeric_limits<std::size_t>::max() / sizeof(T));
}

[[nodiscard]] inline bool host_control_count(std::size_t count) noexcept {
  return count <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

template <class T, class U>
[[nodiscard]] bool host_control_overlap(std::span<T> left,
                                        std::span<U> right) noexcept {
  if (left.empty() || right.empty()) return false;
  const auto a = reinterpret_cast<std::uintptr_t>(left.data());
  const auto b = reinterpret_cast<std::uintptr_t>(right.data());
  return a <= b ? b - a < left.size_bytes() : a - b < right.size_bytes();
}

// Issues call(bootstrap communicator) under the shared lease.
template <class Call>
[[nodiscard]] MpiResult host_control_call(const MpiCallDomain &domain,
                                          Call &&call) noexcept {
  MpiCallLease lease;
  LPFX_TRY(try_acquire_shared_mpi_call(domain, lease));
  return MpiResult::native(call(domain.bootstrap_communicator()));
}

} // namespace detail

// Blocking, allocation-free operations on the domain's bootstrap
// intracommunicator. All spans borrow HOST-accessible memory; accepting a
// pointer does not prove its residence. No device pointer, automatic copy or
// implicit GPU synchronization is supported. Callers collectively match
// operation, datatype, count and root. Validation is local; after a rank
// reports a bad argument the application must coordinate failure/abort rather
// than let peers enter an unmatched collective. No operation may race another
// bootstrap collective or run under an existing shared MPI lease. Prepared
// device collectives remain the device-data API.
[[nodiscard]] inline MpiResult host_barrier(const MpiCallDomain& domain) noexcept {
  return detail::host_control_call(domain, [](const MPI_Comm communicator) noexcept {
    return MPI_Barrier(communicator);
  });
}

template <detail::MpiCollectiveScalar T, reduction_operation Operation>
[[nodiscard]] MpiResult host_all_reduce(const MpiCallDomain& domain,
    std::span<const T> input, std::span<T> output) noexcept {
  if (!detail::valid_host_control_span(input) ||
      !detail::valid_host_control_span(output) ||
      input.size() != output.size() || !detail::host_control_count(input.size()) ||
      (input.data() != output.data() && detail::host_control_overlap(input, output)))
    return ErrorCode::invalid_argument;
  return detail::host_control_call(domain, [&](const MPI_Comm communicator) noexcept {
    const void* source = !input.empty() && input.data() == output.data()
        ? MPI_IN_PLACE : input.data();
    return MPI_Allreduce(source, output.data(), static_cast<int>(input.size()),
        detail::MpiCollectiveDatatype<T>::value,
        detail::mpi_collective_reduction<Operation>(), communicator);
  });
}

template <detail::MpiCollectiveScalar T, reduction_operation Operation>
[[nodiscard]] MpiResult host_reduce(const MpiCallDomain& domain,
    std::span<const T> input, std::span<T> output, int root = 0) noexcept {
  if (!detail::valid_host_control_span(input) ||
      !detail::valid_host_control_span(output) ||
      !detail::host_control_count(input.size()) || root < 0 ||
      root >= domain.size() ||
      (domain.rank() == root && (output.size() != input.size() ||
          (input.data() != output.data() && detail::host_control_overlap(input, output)))))
    return ErrorCode::invalid_argument;
  return detail::host_control_call(domain, [&](const MPI_Comm communicator) noexcept {
    const void* source = domain.rank() == root && !input.empty() &&
        input.data() == output.data() ? MPI_IN_PLACE : input.data();
    return MPI_Reduce(source, output.data(), static_cast<int>(input.size()),
        detail::MpiCollectiveDatatype<T>::value,
        detail::mpi_collective_reduction<Operation>(), root, communicator);
  });
}

template <detail::HostControlScalar T>
[[nodiscard]] MpiResult host_broadcast(const MpiCallDomain& domain,
    std::span<T> values, int root = 0) noexcept {
  if (!detail::valid_host_control_span(values) ||
      !detail::host_control_count(values.size()) || root < 0 ||
      root >= domain.size())
    return ErrorCode::invalid_argument;
  return detail::host_control_call(domain, [&](const MPI_Comm communicator) noexcept {
    return MPI_Bcast(values.data(), static_cast<int>(values.size()),
        detail::host_control_datatype<T>(), root, communicator);
  });
}

// The root output has exactly input.size() * communicator_size elements and is
// disjoint from the local input. Non-root output is ignored and may be empty.
template <detail::HostControlScalar T>
[[nodiscard]] MpiResult host_gather(const MpiCallDomain& domain,
    std::span<const T> input, std::span<T> output, int root = 0) noexcept {
  const auto ranks = static_cast<std::size_t>(domain.size());
  if (!detail::valid_host_control_span(input) ||
      !detail::valid_host_control_span(output) ||
      !detail::host_control_count(input.size()) || root < 0 ||
      root >= domain.size() ||
      (domain.rank() == root &&
       (input.size() > std::numeric_limits<std::size_t>::max() / ranks ||
        output.size() != input.size() * ranks ||
        detail::host_control_overlap(input, output))))
    return ErrorCode::invalid_argument;
  return detail::host_control_call(domain, [&](const MPI_Comm communicator) noexcept {
    return MPI_Gather(input.data(), static_cast<int>(input.size()),
        detail::host_control_datatype<T>(), output.data(),
        static_cast<int>(input.size()), detail::host_control_datatype<T>(), root,
        communicator);
  });
}

} // namespace lpfx::distributed

#endif
