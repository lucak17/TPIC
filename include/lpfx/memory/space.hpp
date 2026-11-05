#ifndef LPFX_MEMORY_SPACE_HPP
#define LPFX_MEMORY_SPACE_HPP

#include <cstddef>

#include <lpfx/backend/traits.hpp>

namespace lpfx {

struct HostSpace final {};

template <device::DeviceBackend Backend> struct DeviceSpace final {
  using backend_type = Backend;
};

template <class Space> struct DefaultAllocator;

template <class Space> struct AllocatorTraits;

template <device::DeviceAllocationBackend Backend>
struct AllocatorTraits<DeviceSpace<Backend>> final {
  static constexpr std::size_t minimum_data_alignment =
      device::BackendTraits<Backend>::minimum_device_allocation_alignment;
};

} // namespace lpfx

#endif
