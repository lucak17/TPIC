#ifndef LPFX_MEMORY_NATIVE_ALLOCATION_HPP
#define LPFX_MEMORY_NATIVE_ALLOCATION_HPP

#include <bit>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <utility>

#include <lpfx/backend/traits.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/result.hpp>
#include <lpfx/memory/host/budget.hpp>

namespace lpfx::device {

enum class NativeMemory { device, pinned_host };

// Fixed-capacity raw storage from the device runtime: device memory or pinned
// host memory. It owns no stream, device ordinal or budget admission; the
// current-device discipline is a caller precondition. Destruction and reset
// are valid only after every user has quiesced.
template <class Backend, NativeMemory Kind>
  requires(Kind == NativeMemory::device ? DeviceAllocationBackend<Backend>
                                        : PinnedHostAllocationBackend<Backend>)
class NativeAllocation final {
public:
  using traits_type = BackendTraits<Backend>;
  using error_type = typename traits_type::error_type;

  constexpr NativeAllocation() noexcept = default;

  NativeAllocation(NativeAllocation &&other) noexcept
      : base_(std::exchange(other.base_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}

  NativeAllocation &operator=(NativeAllocation &&other) noexcept {
    if (this != &other) {
      destroy_or_abort();
      base_ = std::exchange(other.base_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  ~NativeAllocation() noexcept { destroy_or_abort(); }

  // output must be empty. Zero bytes publish the empty owner without a native
  // call; a failed allocation leaves output unchanged.
  [[nodiscard]] static error_type create(const std::size_t bytes,
                                         NativeAllocation &output) noexcept {
    if (!output.empty()) {
      std::abort();
    }
    if (bytes == 0) {
      return traits_type::success_error;
    }
    std::byte *base = nullptr;
    const error_type error = native_allocate(base, bytes);
    if (traits_type::succeeded(error)) {
      output.base_ = base;
      output.bytes_ = bytes;
    }
    return error;
  }

  // A native free may report an earlier asynchronous failure after consuming
  // the allocation, so the owner is disarmed first and never frees twice.
  [[nodiscard]] error_type reset() noexcept {
    if (base_ == nullptr) {
      return traits_type::success_error;
    }
    bytes_ = 0;
    return native_free(std::exchange(base_, nullptr));
  }

  [[nodiscard]] constexpr std::byte *raw_base() noexcept { return base_; }
  [[nodiscard]] constexpr const std::byte *raw_base() const noexcept {
    return base_;
  }
  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    return bytes_;
  }
  [[nodiscard]] constexpr bool empty() const noexcept {
    return base_ == nullptr;
  }

private:
  [[nodiscard]] static error_type native_allocate(std::byte *&base,
                                                  const std::size_t bytes) noexcept {
    if constexpr (Kind == NativeMemory::device) {
      return traits_type::allocate_device(base, bytes);
    } else {
      return traits_type::allocate_pinned_host(base, bytes);
    }
  }

  [[nodiscard]] static error_type native_free(std::byte *const base) noexcept {
    if constexpr (Kind == NativeMemory::device) {
      return traits_type::deallocate_device(base);
    } else {
      return traits_type::deallocate_pinned_host(base);
    }
  }

  void destroy_or_abort() noexcept {
    if (base_ != nullptr && !traits_type::succeeded(reset())) {
      std::abort();
    }
  }

  std::byte *base_ = nullptr;
  std::size_t bytes_ = 0;
};

template <DeviceAllocationBackend Backend>
using DeviceAllocation = NativeAllocation<Backend, NativeMemory::device>;

// Budget admission belongs to the owner that uses the staging bytes.
template <PinnedHostAllocationBackend Backend>
using PinnedHostAllocation = NativeAllocation<Backend, NativeMemory::pinned_host>;

// Pinned host storage whose payload starts at a multiple of Alignment; device
// allocation alignment guarantees do not apply. The native owner keeps the
// runtime pointer and raw_base() addresses the payload. An admitted
// allocation returns its lease only after the native free.
template <PinnedHostAllocationBackend Backend, std::size_t Alignment>
  requires(std::has_single_bit(Alignment))
class AlignedPinnedHostAllocation final {
public:
  constexpr AlignedPinnedHostAllocation() noexcept = default;

  AlignedPinnedHostAllocation(AlignedPinnedHostAllocation &&other) noexcept
      : admission_(std::move(other.admission_)),
        native_(std::move(other.native_)),
        base_(std::exchange(other.base_, nullptr)) {}

  AlignedPinnedHostAllocation &
  operator=(AlignedPinnedHostAllocation &&other) noexcept {
    if (this != &other) {
      reset_capacity();
      admission_ = std::move(other.admission_);
      native_ = std::move(other.native_);
      base_ = std::exchange(other.base_, nullptr);
    }
    return *this;
  }

  // Native bytes needed for a payload of bytes.
  [[nodiscard]] static constexpr bool
  try_allocation_bytes(const std::size_t bytes,
                       std::size_t &output) noexcept {
    if (bytes == 0) {
      output = 0;
      return true;
    }
    return lpfx::detail::checked_add(bytes, Alignment - 1, output);
  }

  // Unadmitted. output must be empty; recoverable failures leave it unchanged.
  [[nodiscard]] static ErrorCode
  create(const std::size_t bytes,
         AlignedPinnedHostAllocation &output) noexcept {
    return create(bytes, HostMemoryLease{}, output);
  }

  // admission must hold exactly try_allocation_bytes(bytes) at an alignment
  // <= Alignment, or be inactive when bytes == 0. Returns
  // Result<AlignedPinnedHostAllocation>, deduced because Result needs a
  // complete type.
  [[nodiscard]] static auto
  allocate(HostMemoryLease admission, const std::size_t bytes) noexcept {
    using Output = Result<AlignedPinnedHostAllocation>;
    std::size_t native_bytes = 0;
    if (!try_allocation_bytes(bytes, native_bytes)) {
      return Output::from_error(ErrorCode::overflow);
    }
    if (!admission.matches(native_bytes, Alignment)) {
      return Output::from_error(ErrorCode::invalid_argument);
    }
    AlignedPinnedHostAllocation allocation{};
    LPFX_TRY_RESULT(create(bytes, std::move(admission), allocation), Output);
    return Output::from_value(std::move(allocation));
  }

  [[nodiscard]] constexpr std::byte *raw_base() noexcept { return base_; }
  [[nodiscard]] constexpr const std::byte *raw_base() const noexcept {
    return base_;
  }
  [[nodiscard]] constexpr std::size_t capacity_bytes() const noexcept {
    return empty() ? 0 : native_.capacity_bytes() - (Alignment - 1);
  }
  [[nodiscard]] constexpr std::size_t allocation_bytes() const noexcept {
    return native_.capacity_bytes();
  }
  [[nodiscard]] static constexpr std::size_t alignment() noexcept {
    return Alignment;
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return base_ == nullptr; }

  void reset_capacity() noexcept {
    if (!BackendTraits<Backend>::succeeded(native_.reset())) {
      std::abort();
    }
    base_ = nullptr;
    admission_.release();
  }

private:
  [[nodiscard]] static ErrorCode
  create(const std::size_t bytes, HostMemoryLease &&admission,
         AlignedPinnedHostAllocation &output) noexcept {
    if (!output.empty()) {
      std::abort();
    }
    std::size_t native_bytes = 0;
    if (!try_allocation_bytes(bytes, native_bytes)) {
      return ErrorCode::overflow;
    }
    if (bytes == 0) {
      return ErrorCode::success;
    }
    PinnedHostAllocation<Backend> native{};
    if (!BackendTraits<Backend>::succeeded(
            PinnedHostAllocation<Backend>::create(native_bytes, native))) {
      return ErrorCode::allocation_failed;
    }
    void *aligned = native.raw_base();
    std::size_t available = native_bytes;
    if (std::align(Alignment, bytes, aligned, available) == nullptr) {
      std::abort();
    }
    output.admission_ = std::move(admission);
    output.native_ = std::move(native);
    output.base_ = static_cast<std::byte *>(aligned);
    return ErrorCode::success;
  }

  // Declaration order frees the native bytes before the lease is returned.
  HostMemoryLease admission_{};
  PinnedHostAllocation<Backend> native_{};
  std::byte *base_ = nullptr;
};

} // namespace lpfx::device

#endif
