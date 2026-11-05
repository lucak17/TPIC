#ifndef LPFX_TESTS_SUPPORT_DEVICE_MOCK_BACKEND_HPP
#define LPFX_TESTS_SUPPORT_DEVICE_MOCK_BACKEND_HPP

#include <lpfx/backend/traits.hpp>
#include <lpfx/data/particle/schema.hpp>
#include <lpfx/memory/host/budget.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace lpfx::device {

struct MockBackend final {};

struct MockStream final {
  std::uintptr_t value{};
  friend constexpr bool operator==(MockStream, MockStream) noexcept = default;
};

struct MockEvent final {
  std::uintptr_t value{};
  friend constexpr bool operator==(MockEvent, MockEvent) noexcept = default;
};

enum class MockError : std::uint8_t { success, not_ready, failure };

struct MockPositionTag final {};
struct MockKeyTag final {};
using MockPosition = lpfx::Attribute<MockPositionTag, double>;
using MockKey = lpfx::Attribute<MockKeyTag, std::uint32_t>;
using MockSchema = lpfx::ParticleSchema<MockPosition, MockKey>;

template <> struct BackendTraits<MockBackend> final {
  using stream_type = MockStream;
  using event_type = MockEvent;
  using error_type = MockError;

  static constexpr stream_type null_stream{};
  static constexpr event_type null_event{};
  static constexpr error_type success_error = error_type::success;
  static constexpr std::size_t minimum_device_allocation_alignment = 256;

  // Large enough for the deliberately descriptor-rich fused 3-D field
  // communication fixture (all 26 directions, including supplemental rim
  // fan-out) while remaining a fixed, allocation-free test arena.
  alignas(256) inline static std::array<std::byte, 131072> device_storage{};
  alignas(256) inline static std::array<std::byte, 131072> pinned_storage{};
  struct PinnedRange {
    std::byte *base;
    std::size_t bytes;
  };
  struct DeferredCopy {
    void *destination;
    const void *source;
    std::size_t bytes;
    stream_type stream;
  };
  inline static std::array<PinnedRange, 256> pinned_ranges{};
  inline static std::array<DeferredCopy, 256> deferred_copies{};
  inline static bool enforce_pinned_endpoints{}, defer_host_copies{};
  inline static std::uint32_t unpinned_endpoints{};
  inline static HostMemoryBudget *observed_pinned_budget{};
  inline static std::size_t device_offset{};
  inline static std::size_t pinned_offset{};
  inline static std::uint32_t device_allocations{};
  inline static std::uint32_t device_deallocations{};
  inline static std::uint32_t device_allocation_attempts{};
  inline static std::uint32_t device_deallocation_attempts{};
  inline static std::uint32_t pinned_allocations{};
  inline static std::uint32_t pinned_deallocations{};
  inline static std::uint32_t pinned_deallocation_attempts{};
  inline static std::uint32_t event_creations{};
  inline static std::uint32_t event_destructions{};
  inline static std::uint32_t event_destruction_attempts{};
  inline static std::uint32_t event_records{};
  inline static std::array<std::uint32_t, 64> event_record_d2h_counts{};
  inline static std::uint32_t event_queries{};
  inline static std::uint32_t event_synchronizations{};
  inline static std::uint32_t stream_event_waits{};
  inline static stream_type last_waiting_stream{};
  inline static event_type last_waited_event{};
  inline static std::uint32_t stream_synchronizations{};
  inline static stream_type last_synchronized_stream{};
  inline static std::uint32_t device_identity_queries{};
  inline static std::uint32_t stream_device_queries{};
  inline static std::uint32_t device_pointer_queries{};
  inline static std::int32_t current_device_ordinal = 2;
  inline static std::int32_t stream_device_ordinal = 2;
  inline static std::int32_t device_pointer_ordinal = 2;
  inline static DeviceIdentity current_device_identity = []() constexpr {
    DeviceIdentity identity{};
    identity.bytes[0] = 0x4d;
    identity.bytes[15] = 0x02;
    return identity;
  }();
  inline static std::uint32_t copy_calls{};
  inline static std::uint32_t host_to_device_copies{};
  inline static std::uint32_t device_to_host_copies{};
  inline static std::uint32_t device_to_device_copies{};
  inline static std::size_t host_to_device_bytes{};
  inline static std::size_t device_to_host_bytes{};
  inline static std::uint32_t fill_calls{};
  inline static std::size_t fill_bytes{};
  inline static bool fail_next_allocation{};
  inline static bool fail_next_device_deallocation{};
  inline static bool fail_next_pinned_deallocation{};
  inline static bool fail_next_event{};
  inline static bool fail_next_event_destruction{};
  inline static bool fail_next_event_record{};
  inline static bool fail_next_event_query{};
  inline static bool fail_next_event_synchronization{};
  inline static bool fail_next_stream_event_wait{};
  inline static bool fail_next_stream_synchronization{};
  inline static bool fail_next_device_identity_query{};
  inline static bool fail_next_stream_device_query{};
  inline static bool fail_next_device_pointer_query{};
  inline static bool fail_next_copy{};
  inline static bool fail_next_fill{};
  // Zero preserves fail-next semantics; a positive value permits deterministic
  // partial multi-copy epoch failures after that many successful calls.
  inline static std::uint32_t successful_copies_before_failure{};
  inline static bool event_ready{};

  static void reset() noexcept {
    device_storage.fill(std::byte{});
    pinned_storage.fill(std::byte{});
    pinned_ranges = {};
    deferred_copies = {};
    enforce_pinned_endpoints = defer_host_copies = false;
    unpinned_endpoints = 0;
    last_stream_priority = 0;
    observed_pinned_budget = nullptr;
    device_offset = 0;
    pinned_offset = 0;
    device_allocations = 0;
    device_deallocations = 0;
    device_allocation_attempts = 0;
    device_deallocation_attempts = 0;
    pinned_allocations = 0;
    pinned_deallocations = 0;
    pinned_deallocation_attempts = 0;
    event_creations = 0;
    event_destructions = 0;
    event_destruction_attempts = 0;
    event_records = 0;
    event_record_d2h_counts = {};
    event_queries = 0;
    event_synchronizations = 0;
    stream_event_waits = 0;
    last_waiting_stream = null_stream;
    last_waited_event = null_event;
    stream_synchronizations = 0;
    last_synchronized_stream = null_stream;
    device_identity_queries = 0;
    stream_device_queries = 0;
    device_pointer_queries = 0;
    current_device_ordinal = 2;
    stream_device_ordinal = 2;
    device_pointer_ordinal = 2;
    current_device_identity = []() constexpr {
      DeviceIdentity identity{};
      identity.bytes[0] = 0x4d;
      identity.bytes[15] = 0x02;
      return identity;
    }();
    copy_calls = 0;
    host_to_device_copies = 0;
    device_to_host_copies = 0;
    device_to_device_copies = 0;
    host_to_device_bytes = 0;
    device_to_host_bytes = 0;
    fill_calls = 0;
    fill_bytes = 0;
    fail_next_allocation = false;
    fail_next_device_deallocation = false;
    fail_next_pinned_deallocation = false;
    fail_next_event = false;
    fail_next_event_destruction = false;
    fail_next_event_record = false;
    fail_next_event_query = false;
    fail_next_event_synchronization = false;
    fail_next_stream_event_wait = false;
    fail_next_stream_synchronization = false;
    fail_next_device_identity_query = false;
    fail_next_stream_device_query = false;
    fail_next_device_pointer_query = false;
    fail_next_copy = false;
    fail_next_fill = false;
    successful_copies_before_failure = 0;
    event_ready = false;
  }

  [[nodiscard]] static bool fail_requested_copy() noexcept {
    if (!fail_next_copy) {
      return false;
    }
    if (successful_copies_before_failure != 0) {
      --successful_copies_before_failure;
      return false;
    }
    fail_next_copy = false;
    return true;
  }

  [[nodiscard]] static constexpr bool
  succeeded(const error_type error) noexcept {
    return error == error_type::success;
  }

  [[nodiscard]] static constexpr bool
  not_ready(const error_type error) noexcept {
    return error == error_type::not_ready;
  }

  [[nodiscard]] static constexpr error_type
  create_stream(stream_type &stream) noexcept {
    stream.value = 1;
    return success_error;
  }

  [[nodiscard]] static constexpr error_type
  destroy_stream(const stream_type) noexcept {
    return success_error;
  }

  [[nodiscard]] static error_type
  synchronize_stream(const stream_type stream) noexcept {
    ++stream_synchronizations;
    last_synchronized_stream = stream;
    if (std::exchange(fail_next_stream_synchronization, false))
      return error_type::failure;
    for (auto &copy : deferred_copies) {
      if (copy.bytes != 0 && copy.stream == stream) {
        std::memcpy(copy.destination, copy.source, copy.bytes);
        copy = {};
      }
    }
    return success_error;
  }

  [[nodiscard]] static error_type
  allocate_device(std::byte *&allocation, const std::size_t bytes) noexcept {
    ++device_allocation_attempts;
    constexpr std::size_t alignment = minimum_device_allocation_alignment;
    const std::size_t aligned_offset =
        (device_offset + alignment - 1) / alignment * alignment;
    if (std::exchange(fail_next_allocation, false) ||
        aligned_offset > device_storage.size() ||
        bytes > device_storage.size() - aligned_offset) {
      return error_type::failure;
    }
    allocation = device_storage.data() + aligned_offset;
    device_offset = aligned_offset + bytes;
    ++device_allocations;
    return success_error;
  }

  [[nodiscard]] static error_type deallocate_device(std::byte *const) noexcept {
    ++device_deallocation_attempts;
    if (std::exchange(fail_next_device_deallocation, false)) {
      return error_type::failure;
    }
    ++device_deallocations;
    return success_error;
  }

  [[nodiscard]] static error_type
  allocate_pinned_host(std::byte *&allocation,
                       const std::size_t bytes) noexcept {
    if (std::exchange(fail_next_allocation, false) ||
        bytes > pinned_storage.size() - pinned_offset) {
      return error_type::failure;
    }
    allocation = pinned_storage.data() + pinned_offset;
    pinned_offset += bytes;
    bool registered = false;
    for (auto &range : pinned_ranges) {
      if (range.base == nullptr) {
        range = {allocation, bytes};
        registered = true;
        break;
      }
    }
    if (!registered) std::abort();
    ++pinned_allocations;
    return success_error;
  }

  [[nodiscard]] static error_type
  deallocate_pinned_host(std::byte *const base) noexcept {
    ++pinned_deallocation_attempts;
    if (std::exchange(fail_next_pinned_deallocation, false)) {
      return error_type::failure;
    }
    for (auto &range : pinned_ranges) {
      if (range.base != base) continue;
      if (observed_pinned_budget != nullptr &&
          observed_pinned_budget->used() < range.bytes)
        std::abort();
      const auto first = reinterpret_cast<std::uintptr_t>(base);
      for (const auto &copy : deferred_copies) {
        const auto source = reinterpret_cast<std::uintptr_t>(copy.source);
        const auto destination = reinterpret_cast<std::uintptr_t>(copy.destination);
        if (copy.bytes != 0 &&
            ((source >= first && source - first < range.bytes) ||
             (destination >= first && destination - first < range.bytes)))
          std::abort();
      }
      range = {};
      break;
    }
    ++pinned_deallocations;
    return success_error;
  }

  [[nodiscard]] static bool host_endpoint_is_pinned(
      const void *pointer, std::size_t width, std::size_t pitch = 0,
      std::size_t rows = 1) noexcept {
    if (width == 0 || rows == 0) return true;
    std::size_t preceding = 0, extent = 0;
    if (!lpfx::detail::checked_multiply(rows - 1, pitch, preceding) ||
        !lpfx::detail::checked_add(preceding, width, extent)) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    for (const auto &range : pinned_ranges) {
      const auto first = reinterpret_cast<std::uintptr_t>(range.base);
      if (range.base != nullptr && address >= first &&
          address - first <= range.bytes &&
          extent <= range.bytes - (address - first)) return true;
    }
    return false;
  }

  [[nodiscard]] static bool check_host_endpoint(
      const void *pointer, std::size_t width, std::size_t pitch = 0,
      std::size_t rows = 1) noexcept {
    if (!enforce_pinned_endpoints ||
        host_endpoint_is_pinned(pointer, width, pitch, rows)) return true;
    ++unpinned_endpoints;
    return false;
  }

  static error_type copy_host_endpoint(void *destination, const void *source,
      std::size_t bytes, stream_type stream) noexcept {
    if (bytes == 0) return success_error;
    if (defer_host_copies) {
      for (auto &copy : deferred_copies) {
        if (copy.bytes == 0) {
          copy = {destination, source, bytes, stream};
          return success_error;
        }
      }
      std::abort();
    }
    std::memcpy(destination, source, bytes);
    return success_error;
  }

  [[nodiscard]] static error_type
  copy_host_to_device_async(void *const destination, const void *const source,
                            const std::size_t bytes,
                            const stream_type stream) noexcept {
    ++copy_calls;
    ++host_to_device_copies;
    host_to_device_bytes += bytes;
    if (!check_host_endpoint(source, bytes) || fail_requested_copy()) {
      return error_type::failure;
    }
    return copy_host_endpoint(destination, source, bytes, stream);
  }

  [[nodiscard]] static error_type
  copy_device_to_host_async(void *const destination, const void *const source,
                            const std::size_t bytes,
                            const stream_type stream) noexcept {
    ++copy_calls;
    ++device_to_host_copies;
    device_to_host_bytes += bytes;
    if (!check_host_endpoint(destination, bytes) || fail_requested_copy()) {
      return error_type::failure;
    }
    return copy_host_endpoint(destination, source, bytes, stream);
  }

  [[nodiscard]] static error_type
  copy_device_to_device_async(void *const destination, const void *const source,
                              const std::size_t bytes,
                              const stream_type) noexcept {
    ++copy_calls;
    ++device_to_device_copies;
    if (fail_requested_copy()) {
      return error_type::failure;
    }
    std::memcpy(destination, source, bytes);
    return success_error;
  }

  [[nodiscard]] static error_type
  memset_device_async(void *const destination, const unsigned char value,
                      const std::size_t bytes, const stream_type) noexcept {
    ++fill_calls;
    fill_bytes += bytes;
    if (std::exchange(fail_next_fill, false)) {
      return error_type::failure;
    }
    std::memset(destination, static_cast<int>(value), bytes);
    return success_error;
  }

  [[nodiscard]] static error_type create_event(event_type &event) noexcept {
    if (std::exchange(fail_next_event, false)) {
      return error_type::failure;
    }
    event.value = 1;
    ++event_creations;
    return success_error;
  }

  [[nodiscard]] static error_type destroy_event(const event_type) noexcept {
    ++event_destruction_attempts;
    if (std::exchange(fail_next_event_destruction, false)) {
      return error_type::failure;
    }
    ++event_destructions;
    return success_error;
  }

  [[nodiscard]] static error_type record_event(const event_type,
                                               const stream_type) noexcept {
    if (event_records < event_record_d2h_counts.size())
      event_record_d2h_counts[event_records] = device_to_host_copies;
    ++event_records;
    event_ready = false;
    if (std::exchange(fail_next_event_record, false)) {
      return error_type::failure;
    }
    return success_error;
  }

  [[nodiscard]] static error_type query_event(const event_type) noexcept {
    ++event_queries;
    if (std::exchange(fail_next_event_query, false)) {
      return error_type::failure;
    }
    return event_ready ? success_error : error_type::not_ready;
  }

  [[nodiscard]] static error_type
  stream_wait_event(const stream_type stream, const event_type event) noexcept {
    ++stream_event_waits;
    last_waiting_stream = stream;
    last_waited_event = event;
    return std::exchange(fail_next_stream_event_wait, false)
               ? error_type::failure
               : success_error;
  }

  [[nodiscard]] static error_type synchronize_event(const event_type) noexcept {
    ++event_synchronizations;
    if (std::exchange(fail_next_event_synchronization, false)) {
      return error_type::failure;
    }
    event_ready = true;
    return success_error;
  }

  [[nodiscard]] static constexpr error_type
  query_current_device_properties(DeviceProperties &properties) noexcept {
    properties = DeviceProperties{2, 32, 256, 65535, 12};
    return success_error;
  }

  [[nodiscard]] static error_type
  query_current_device_identity(DeviceIdentity &identity,
                                std::int32_t &device_ordinal) noexcept {
    ++device_identity_queries;
    if (std::exchange(fail_next_device_identity_query, false)) {
      return error_type::failure;
    }
    identity = current_device_identity;
    device_ordinal = current_device_ordinal;
    return success_error;
  }

  [[nodiscard]] static error_type
  query_stream_device_ordinal(const stream_type,
                              std::int32_t &device_ordinal) noexcept {
    ++stream_device_queries;
    if (std::exchange(fail_next_stream_device_query, false)) {
      return error_type::failure;
    }
    device_ordinal = stream_device_ordinal;
    return success_error;
  }

  [[nodiscard]] static error_type
  query_device_pointer_ordinal(const void *const pointer,
                               std::int32_t &device_ordinal) noexcept {
    ++device_pointer_queries;
    if (std::exchange(fail_next_device_pointer_query, false)) {
      return error_type::failure;
    }
    device_ordinal = pointer == nullptr ? -1 : device_pointer_ordinal;
    return success_error;
  }

  // Role members required by the backend concepts (traits.hpp).
  [[nodiscard]] static constexpr const char *
  native_error_name(const error_type error) noexcept {
    return error == error_type::success     ? "mockSuccess"
           : error == error_type::not_ready ? "mockNotReady"
                                            : "mockFailure";
  }

  [[nodiscard]] static constexpr const char *
  native_error_string(const error_type error) noexcept {
    return error == error_type::success     ? "mock success"
           : error == error_type::not_ready ? "mock operation not ready"
                                            : "mock failure";
  }

  [[nodiscard]] static constexpr bool
  runtime_unavailable(const error_type) noexcept {
    return false;
  }

  [[nodiscard]] static constexpr bool
  is_default_stream(const stream_type) noexcept {
    return false;
  }

  [[nodiscard]] static constexpr error_type
  query_stream_nonblocking(const stream_type, bool &nonblocking) noexcept {
    nonblocking = true;
    return success_error;
  }

  // Optional priority role: two levels, greatest (numerically lowest) first.
  inline static int last_stream_priority{};
  [[nodiscard]] static error_type create_stream(stream_type &stream,
                                                const int priority) noexcept {
    last_stream_priority = priority;
    return create_stream(stream);
  }

  [[nodiscard]] static constexpr error_type
  query_stream_priority_range(int &least, int &greatest) noexcept {
    least = 0;
    greatest = -1;
    return success_error;
  }

  [[nodiscard]] static error_type
  create_timing_event(event_type &event) noexcept {
    return create_event(event);
  }

  [[nodiscard]] static constexpr error_type
  event_elapsed_time(float &milliseconds, const event_type,
                     const event_type) noexcept {
    milliseconds = 0.0F;
    return success_error;
  }

  [[nodiscard]] static error_type copy_host_to_device_2d_async(
      void *const destination, const std::size_t destination_pitch,
      const void *const source, const std::size_t source_pitch,
      const std::size_t width, const std::size_t rows,
      const stream_type stream) noexcept {
    ++copy_calls;
    ++host_to_device_copies;
    host_to_device_bytes += width * rows;
    if (!check_host_endpoint(source, width, source_pitch, rows) ||
        fail_requested_copy()) {
      return error_type::failure;
    }
    for (std::size_t row = 0; row != rows; ++row) {
      static_cast<void>(copy_host_endpoint(
          static_cast<std::byte *>(destination) + row * destination_pitch,
          static_cast<const std::byte *>(source) + row * source_pitch, width,
          stream));
    }
    return success_error;
  }

  [[nodiscard]] static error_type copy_device_to_host_2d_async(
      void *const destination, const std::size_t destination_pitch,
      const void *const source, const std::size_t source_pitch,
      const std::size_t width, const std::size_t rows,
      const stream_type stream) noexcept {
    ++copy_calls;
    ++device_to_host_copies;
    device_to_host_bytes += width * rows;
    if (!check_host_endpoint(destination, width, destination_pitch, rows) ||
        fail_requested_copy()) {
      return error_type::failure;
    }
    for (std::size_t row = 0; row != rows; ++row) {
      static_cast<void>(copy_host_endpoint(
          static_cast<std::byte *>(destination) + row * destination_pitch,
          static_cast<const std::byte *>(source) + row * source_pitch, width,
          stream));
    }
    return success_error;
  }

  [[nodiscard]] static constexpr error_type
  query_host_device_max_copy_pitch(std::size_t &max_pitch_bytes) noexcept {
    max_pitch_bytes = std::size_t{1} << 30U;
    return success_error;
  }

  [[nodiscard]] static constexpr error_type
  query_device_count(int &count) noexcept {
    count = 1;
    return success_error;
  }

  [[nodiscard]] static constexpr error_type select_device(const int) noexcept {
    return success_error;
  }

  [[nodiscard]] static constexpr error_type
  query_current_device_memory(DeviceMemoryInfo &memory) noexcept {
    memory = DeviceMemoryInfo{device_storage.size(), device_storage.size()};
    return success_error;
  }
};

} // namespace lpfx::device

static_assert(lpfx::device::DeviceBackend<lpfx::device::MockBackend>);
static_assert(lpfx::device::DeviceAllocationBackend<lpfx::device::MockBackend>);
static_assert(
    lpfx::device::PinnedHostAllocationBackend<lpfx::device::MockBackend>);
static_assert(lpfx::device::DeviceStreamBackend<lpfx::device::MockBackend>);
static_assert(
    lpfx::device::DeviceStreamPriorityBackend<lpfx::device::MockBackend>);
static_assert(lpfx::device::DeviceCopyBackend<lpfx::device::MockBackend>);
static_assert(lpfx::device::DeviceEventBackend<lpfx::device::MockBackend>);
static_assert(lpfx::device::DeviceDiscoveryBackend<lpfx::device::MockBackend>);
// Launches are emulated by per-family launch specializations, so the mock is
// deliberately not a complete native runtime.
static_assert(
    !lpfx::device::DeviceKernelLaunchBackend<lpfx::device::MockBackend>);
static_assert(!lpfx::device::DeviceRuntimeBackend<lpfx::device::MockBackend>);

#endif
