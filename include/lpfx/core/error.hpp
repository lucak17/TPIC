#ifndef LPFX_CORE_ERROR_HPP
#define LPFX_CORE_ERROR_HPP

#include <cstdint>
#include <type_traits>

#include <lpfx/core/attributes.hpp>

namespace lpfx {

// Errors are intentionally coarse.  Backend-specific values belong in logs or
// diagnostics at the call boundary, not in every status object.
enum class ErrorCode : std::uint8_t {
  success = 0,
  invalid_argument = 1,
  overflow = 2,
  capacity_exceeded = 3,
  allocation_failed = 4,
  invalid_state = 5,
  backend_failure = 6
};

// A status is the complete control-flow result.  Native error payloads and
// source metadata are deliberately excluded from this hot-path value.
class Status final {
public:
  constexpr Status() noexcept = default;

  [[nodiscard]] LPFX_HOST_DEVICE static constexpr Status success() noexcept {
    return {};
  }

  [[nodiscard]] LPFX_HOST_DEVICE static constexpr Status pending() noexcept {
    return Status{ErrorCode::success, Kind::pending};
  }

  [[nodiscard]] LPFX_HOST_DEVICE static constexpr Status accepted() noexcept {
    // Acceptance always transfers a conclusion obligation to the caller.
    return Status{ErrorCode::success, Kind::accepted};
  }

  [[nodiscard]] LPFX_HOST_DEVICE static constexpr Status
  failure(const ErrorCode code) noexcept {
    return Status{normalize_failure(code), Kind::failure};
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr ErrorCode code() const noexcept {
    return code_;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool is_success() const noexcept {
    return kind_ == Kind::success || kind_ == Kind::accepted;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool is_pending() const noexcept {
    return kind_ == Kind::pending;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool is_failure() const noexcept {
    return kind_ == Kind::failure;
  }

  [[nodiscard]] LPFX_HOST_DEVICE constexpr bool
  requires_conclusion() const noexcept {
    return kind_ == Kind::accepted;
  }

private:
  enum class Kind : unsigned char {
    success = 0,
    pending = 1,
    failure = 2,
    accepted = 3
  };

  LPFX_HOST_DEVICE constexpr Status(const ErrorCode code,
                                    const Kind kind) noexcept
      : code_(code), kind_(kind) {}

  [[nodiscard]] LPFX_HOST_DEVICE static constexpr ErrorCode
  normalize_failure(const ErrorCode code) noexcept {
    // Preserve the failure invariant even when a caller forwards success by
    // mistake; no secondary validation API is needed.
    return code == ErrorCode::success ? ErrorCode::invalid_state : code;
  }

  ErrorCode code_ = ErrorCode::success;
  Kind kind_ = Kind::success;
};

static_assert(sizeof(ErrorCode) == 1);
static_assert(std::is_standard_layout_v<Status>);
static_assert(std::is_trivially_copyable_v<Status>);
static_assert(sizeof(Status) == 2);
static_assert(alignof(Status) == 1);

namespace detail {

// LPFX_TRY operand adapter; core/result.hpp adds the Result overload.
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr ErrorCode
try_error(const ErrorCode error) noexcept {
  return error;
}

} // namespace detail

} // namespace lpfx

// Returns the failure of `expression` (an ErrorCode or a Result, evaluated
// once) from the enclosing ErrorCode-returning function.
#define LPFX_TRY(expression)                                                   \
  do {                                                                         \
    if (const ::lpfx::ErrorCode lpfx_try_error_ =                              \
            ::lpfx::detail::try_error(expression);                             \
        lpfx_try_error_ != ::lpfx::ErrorCode::success) {                       \
      return lpfx_try_error_;                                                  \
    }                                                                          \
  } while (false)

// As LPFX_TRY, returning <result type>::from_error(failure) instead.
#define LPFX_TRY_RESULT(expression, ...)                                       \
  do {                                                                         \
    if (const ::lpfx::ErrorCode lpfx_try_error_ =                              \
            ::lpfx::detail::try_error(expression);                             \
        lpfx_try_error_ != ::lpfx::ErrorCode::success) {                       \
      return __VA_ARGS__::from_error(lpfx_try_error_);                         \
    }                                                                          \
  } while (false)

#endif
