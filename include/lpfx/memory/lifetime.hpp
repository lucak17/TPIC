#ifndef LPFX_MEMORY_LIFETIME_HPP
#define LPFX_MEMORY_LIFETIME_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/memory/space.hpp>

namespace lpfx {

// Opt-in for trivially copyable class types whose arrays may start lifetime
// in raw storage.
template <class T>
inline constexpr bool enable_implicit_lifetime_attribute = false;

template <class T>
concept LifetimeStartEligible =
    requires { sizeof(T); } && std::is_object_v<T> && !std::is_array_v<T> &&
    std::same_as<T, std::remove_cv_t<T>> && std::is_trivially_copyable_v<T> &&
    std::is_trivially_destructible_v<T> &&
    (std::is_arithmetic_v<T> || std::same_as<T, std::byte> ||
     enable_implicit_lifetime_attribute<T>);

namespace detail {

template <class T>
concept LifetimeViewElement = LifetimeStartEligible<std::remove_const_t<T>>;

} // namespace detail

// Device arrays become readable only through a completed copy, fill or
// construction kernel owned by a concrete operation.
template <class T, class Space>
inline constexpr bool array_range_start_supported = false;

template <LifetimeStartEligible T>
inline constexpr bool array_range_start_supported<T, HostSpace> = true;

template <LifetimeStartEligible T, device::DeviceBackend Backend>
inline constexpr bool array_range_start_supported<T, DeviceSpace<Backend>> =
    true;

// A live array in one address space. Only host views dereference; a device
// view is a host-side descriptor whose owner is responsible for lifetime and
// transfers.
template <detail::LifetimeViewElement T, class Space> class BufferView final {
public:
  using element_type = T;
  using value_type = std::remove_const_t<T>;
  using space_type = Space;

  constexpr BufferView() noexcept = default;

  template <class U>
    requires std::same_as<T, const U>
  constexpr BufferView(const BufferView<U, Space> other) noexcept
      : base_(other.data()), count_(other.size()) {}

  [[nodiscard]] constexpr T *data() const noexcept { return base_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return count_; }

  [[nodiscard]] auto raw_base() const noexcept {
    using Byte =
        std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;
    return reinterpret_cast<Byte *>(base_);
  }

  // Hard precondition: index < size().
  [[nodiscard]] constexpr T &operator[](const std::size_t index) const noexcept
    requires std::same_as<Space, HostSpace>
  {
    return base_[index];
  }

private:
  constexpr BufferView(T *const base, const std::size_t count) noexcept
      : base_(base), count_(count) {}

  T *base_ = nullptr;
  std::size_t count_ = 0;

  friend struct detail::PrivateConstructionAccess;
};

static_assert(sizeof(BufferView<int, HostSpace>) ==
                  sizeof(int *) + sizeof(std::size_t) &&
              std::is_trivially_copyable_v<BufferView<int, HostSpace>>);

namespace detail {

// Hard precondition: base addresses count live elements.
template <class T, class Space = HostSpace>
[[nodiscard]] constexpr BufferView<T, Space>
make_buffer_view(T *const base, const std::size_t count) noexcept {
  return PrivateConstructionAccess::create<BufferView<T, Space>>(base, count);
}

// Starts the lifetimes of count T objects in raw host storage and keeps its
// bytes. Hard preconditions: base is aligned for T and addresses
// count * sizeof(T) bytes. Returns nullptr when count == 0.
template <LifetimeStartEligible T>
[[nodiscard]] inline T *start_host_array_lifetime(std::byte *const base,
                                                  const std::size_t count) noexcept {
  if (count == 0) {
    return nullptr;
  }
  void *const refreshed = std::memmove(base, base, count * sizeof(T));
  return std::launder(static_cast<T *>(refreshed));
}

// Checked form: an overflowing, null or misaligned range returns false and
// leaves output unchanged.
template <LifetimeStartEligible T>
[[nodiscard]] inline bool
try_start_host_array_lifetime(std::byte *const base, const std::size_t count,
                              T *&output) noexcept {
  std::size_t bytes = 0;
  if (!checked_byte_count<T>(count, bytes) ||
      (count != 0 &&
       (base == nullptr ||
        reinterpret_cast<std::uintptr_t>(base) % alignof(T) != 0))) {
    return false;
  }
  output = start_host_array_lifetime<T>(base, count);
  return true;
}

} // namespace detail

} // namespace lpfx

#endif
