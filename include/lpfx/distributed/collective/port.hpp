#ifndef LPFX_DISTRIBUTED_COLLECTIVE_PORT_HPP
#define LPFX_DISTRIBUTED_COLLECTIVE_PORT_HPP

#include <cstddef>
#include <cstdint>

#include <lpfx/backend/handles.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/descriptor/collective.hpp>
#include <lpfx/distributed/detail/fixed_byte.hpp>

namespace lpfx::distributed::detail {

// Cold backend-neutral preparation facts for one rank. Kind, reduction,
// scalar, memory space and exact-alias mode stay compile-time facts of the
// port type.
struct CollectivePreparedShape final {
  std::size_t input_count = 0;
  std::size_t output_count = 0;
  std::size_t native_count = 0;
  std::uint32_t rank = no_collective_root;
  std::uint32_t rank_count = 0;
  std::uint32_t root = no_collective_root;

  [[nodiscard]] constexpr bool has_native_work() const noexcept {
    return native_count != 0;
  }
};

// Only the native count survives in the epoch recipe; root stays a descriptor
// constant at the port call.
struct CollectiveEpoch final {
  std::size_t native_count = 0;

  [[nodiscard]] constexpr bool has_native_work() const noexcept {
    return native_count != 0;
  }
};

struct EmptyCollectiveAccess final {};

// The rank-specific operand access, resolved once. An exact in-place
// operation presents the same bytes in both regions so direct ordering
// policies see one coalescible read/write batch.
struct CollectiveLocalAccess final {
  ByteRegion<const std::byte> read{};
  ByteRegion<std::byte> write{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return read.empty() && write.empty();
  }
};

template <class T> struct CollectiveBuffers final {
  const T *input = nullptr;
  T *output = nullptr;
};

// Cold affinity validation shared by MPI and XCCL device ports. It checks the
// executor even for zero work; empty regions need no pointer query.
template <device::DeviceDiscoveryBackend Backend, class Setup, class T>
[[nodiscard]] ErrorCode try_validate_device_collective_affinity(
    const Setup &setup, const CollectivePreparedShape &shape,
    const CollectiveBuffers<T> buffers,
    const device::Executor<Backend> &executor) noexcept {
  if (!setup.is_valid() || !setup.has_device_affinity()) {
    return ErrorCode::invalid_argument;
  }
  LPFX_TRY(setup.template try_verify_device_affinity<Backend>(executor));
  if (shape.input_count != 0) {
    LPFX_TRY(setup.template try_verify_device_pointer_affinity<Backend>(
        buffers.input));
  }
  if (shape.output_count != 0 && static_cast<const void *>(buffers.output) !=
                                     static_cast<const void *>(buffers.input)) {
    return setup.template try_verify_device_pointer_affinity<Backend>(
        buffers.output);
  }
  return ErrorCode::success;
}

} // namespace lpfx::distributed::detail

#endif
