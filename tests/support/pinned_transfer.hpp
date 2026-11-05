#ifndef TPIC_TESTS_SUPPORT_PINNED_TRANSFER_HPP
#define TPIC_TESTS_SUPPORT_PINNED_TRANSFER_HPP

#include <lpfx/backend/backend.hpp>
#include <lpfx/memory/native_allocation.hpp>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tpic::test {

// Validation fixtures may be ordinary CPU values; the actual transfer endpoint
// uses native pinned storage and remains owned through checked completion.
inline void transfer_bytes(void* device, void* host, std::size_t bytes,
                           lpfx::device::ActiveExecutor executor, bool upload) {
  using Traits = lpfx::device::ActiveBackendTraits;
  using Pinned = lpfx::device::PinnedHostAllocation<lpfx::device::ActiveBackend>;
  if (bytes == 0) return;
  Pinned staging;
  if (!Traits::succeeded(Pinned::create(bytes, staging)))
    throw std::runtime_error("validation pinned allocation failed");
  if (upload) std::memcpy(staging.raw_base(), host, bytes);
  const auto copied = upload
      ? Traits::copy_host_to_device_async(device, staging.raw_base(), bytes, executor.stream())
      : Traits::copy_device_to_host_async(staging.raw_base(), device, bytes, executor.stream());
  if (!Traits::succeeded(Traits::synchronize_stream(executor.stream()))) std::abort();
  if (!Traits::succeeded(copied))
    throw std::runtime_error("validation pinned transfer failed");
  if (!upload) std::memcpy(host, staging.raw_base(), bytes);
}

inline void download_bytes(void* host, const void* device, std::size_t bytes,
                           lpfx::device::ActiveExecutor executor) {
  transfer_bytes(const_cast<void*>(device), host, bytes, executor, false);
}

inline void upload_bytes(void* device, const void* host, std::size_t bytes,
                         lpfx::device::ActiveExecutor executor) {
  transfer_bytes(device, const_cast<void*>(host), bytes, executor, true);
}

} // namespace tpic::test

#endif // TPIC_TESTS_SUPPORT_PINNED_TRANSFER_HPP
