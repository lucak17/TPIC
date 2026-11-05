#ifndef TPIC_DEVICE_DEVICE_VALIDATION_HPP
#define TPIC_DEVICE_DEVICE_VALIDATION_HPP

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/memory/device/failure_flag.hpp>

namespace tpic::device {

// A nonzero flag means a producer kernel found a numerical failure;
// host_wait_check() and ready() turn it into an exception with the caller's
// message. LPFx owns the flag storage, capture, completion and reuse.
template <class Backend> class DeviceValidation {
  using Executor = lpfx::device::Executor<Backend>;
  using Flag = lpfx::device::DeviceFailureFlag<Backend>;

public:
  DeviceValidation() {
    lpfx::check_error(Flag::create(flag_), "TPIC device validation: storage preparation failed");
  }

  LPFX_ALWAYS_INLINE void clear(Executor executor) {
    lpfx::check_error(flag_.clear(executor), "TPIC device validation: clear failed");
  }
  [[nodiscard]] LPFX_ALWAYS_INLINE unsigned* data() noexcept { return flag_.data(); }
  LPFX_ALWAYS_INLINE void capture(Executor executor) {
    lpfx::check_error(flag_.capture(executor), "TPIC device validation: capture failed");
  }
  LPFX_ALWAYS_INLINE void host_wait_check(const char* message) {
    unsigned value = 0;
    lpfx::check_error(flag_.read(value), "TPIC device validation: completion failed");
    lpfx::require(value == 0, message);
  }
  // False while the captured flag is still in flight; throws with message
  // when a producer set it.
  [[nodiscard]] LPFX_ALWAYS_INLINE bool ready(const char* message) {
    unsigned value = 0;
    const auto status = flag_.try_read(value);
    if (status.is_pending()) return false;
    lpfx::check_error(status, "TPIC device validation: completion query failed");
    lpfx::require(value == 0, message);
    return true;
  }
  LPFX_ALWAYS_INLINE void device_wait(Executor executor) const {
    lpfx::check_error(flag_.wait_on(executor), "TPIC device validation: consumer dependency failed");
  }

private:
  Flag flag_;
};

} // namespace tpic::device

#endif // TPIC_DEVICE_DEVICE_VALIDATION_HPP
