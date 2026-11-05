#ifndef LPFX_CORE_ERROR_REPORTING_HPP
#define LPFX_CORE_ERROR_REPORTING_HPP

#include <cstddef>
#include <cstdio>

#include <lpfx/core/error.hpp>
#include <lpfx/core/result.hpp>

#if LPFX_HAS_EXCEPTIONS
#include <stdexcept>
#include <string>
#endif

namespace lpfx {

[[nodiscard]] LPFX_HOST_DEVICE constexpr const char *
error_code_name(ErrorCode code) noexcept {
  switch (code) {
  case ErrorCode::success:
    return "success";
  case ErrorCode::invalid_argument:
    return "invalid_argument";
  case ErrorCode::overflow:
    return "overflow";
  case ErrorCode::capacity_exceeded:
    return "capacity_exceeded";
  case ErrorCode::allocation_failed:
    return "allocation_failed";
  case ErrorCode::invalid_state:
    return "invalid_state";
  case ErrorCode::backend_failure:
    return "backend_failure";
  }
  return "unknown";
}

[[nodiscard]] LPFX_HOST_DEVICE constexpr const char *
status_name(Status status) noexcept {
  if (status.is_failure())
    return "failure";
  if (status.is_pending())
    return "pending";
  if (status.requires_conclusion())
    return "accepted";
  return "success";
}

namespace detail {

[[nodiscard]] LPFX_HOST_DEVICE constexpr const char *
error_text(const char *text) noexcept {
  return text != nullptr ? text : "unavailable";
}

// " rank <rank>" for a non-negative rank, otherwise empty.
struct RankText final {
  LPFX_HOST_DEVICE explicit RankText(const int rank) noexcept {
    if (rank < 0)
      return;
    char digits[10];
    int count = 0;
    auto value = static_cast<unsigned>(rank);
    do {
      digits[count++] = static_cast<char>('0' + value % 10U);
      value /= 10U;
    } while (value != 0U);
    const char prefix[] = " rank ";
    int length = 0;
    for (; prefix[length] != '\0'; ++length)
      text[length] = prefix[length];
    while (count != 0)
      text[length++] = digits[--count];
  }

  char text[17]{};
};

// One formatted line per call: stderr on the host, printf in device code.
// Reporting never throws, terminates, synchronizes or consumes error state.
template <class... Args>
LPFX_HOST_DEVICE void print_error_line(const char *format,
                                       Args... args) noexcept {
#if LPFX_DEVICE_PASS
  ::printf(format, args...);
#else
  std::fprintf(stderr, format, args...);
#endif
}

} // namespace detail

LPFX_HOST_DEVICE inline void print_error(const char *message,
                                         const char *context = "LPFx",
                                         int rank = -1) noexcept {
  detail::print_error_line("%s%s: %s\n", detail::error_text(context),
                           detail::RankText{rank}.text,
                           detail::error_text(message));
}

LPFX_HOST_DEVICE inline void print_error(ErrorCode code,
                                         const char *context = "LPFx",
                                         int rank = -1) noexcept {
  if (code == ErrorCode::success)
    return;
  detail::print_error_line("%s%s: lpfx=%s(%u)\n", detail::error_text(context),
                           detail::RankText{rank}.text, error_code_name(code),
                           static_cast<unsigned>(code));
}

LPFX_HOST_DEVICE inline void print_error(Status status,
                                         const char *context = "LPFx",
                                         int rank = -1) noexcept {
  if (status.is_success())
    return;
  detail::print_error_line(
      "%s%s: lpfx=%s(%u) status=%s\n", detail::error_text(context),
      detail::RankText{rank}.text, error_code_name(status.code()),
      static_cast<unsigned>(status.code()), status_name(status));
}

// Cold, allocation-free host formatting; each error layer adds an overload.
// The return value follows snprintf: required characters excluding the
// terminator, or a negative formatting error. A null buffer requires size 0.
inline int format_error(char *buffer, std::size_t size, ErrorCode code,
                        const char *context) noexcept {
  return std::snprintf(buffer, size, "%s: lpfx=%s(%u)",
                       detail::error_text(context), error_code_name(code),
                       static_cast<unsigned>(code));
}

inline int format_error(char *buffer, std::size_t size, Status status,
                        const char *context) noexcept {
  return std::snprintf(buffer, size, "%s: lpfx=%s(%u) status=%s",
                       detail::error_text(context),
                       error_code_name(status.code()),
                       static_cast<unsigned>(status.code()),
                       status_name(status));
}

#if LPFX_HAS_EXCEPTIONS

namespace detail {

// Throws Exception with format_error(buffer, size, error..., context), found
// by argument-dependent lookup for the error layers declared elsewhere.
template <class Exception, class... Error>
[[noreturn]] LPFX_NOINLINE void throw_error(const char *context,
                                            const Error &...error) {
  char buffer[512];
  const int required = format_error(buffer, sizeof(buffer), error..., context);
  if (required < 0)
    throw Exception("LPFx error formatting failed");
  if (static_cast<std::size_t>(required) < sizeof(buffer))
    throw Exception(buffer);
  std::string message(static_cast<std::size_t>(required), '\0');
  format_error(message.data(), message.size() + 1, error..., context);
  throw Exception(message);
}

} // namespace detail

// Explicit host exception adapters. Success neither allocates nor formats.
template <class Exception = std::runtime_error>
inline void require(bool condition, const char *message) {
  if (!condition)
    throw Exception(detail::error_text(message));
}

template <class Exception = std::runtime_error>
inline void check_error(ErrorCode code, const char *context) {
  if (code != ErrorCode::success)
    detail::throw_error<Exception>(context, code);
}

// Accepted work retains its conclusion obligation.
template <class Exception = std::runtime_error>
inline void check_error(Status status, const char *context) {
  if (!status.is_success())
    detail::throw_error<Exception>(context, status);
}

template <class Exception = std::runtime_error, class T>
inline void check_error(const Result<T> &result, const char *context) {
  if (!result.has_value())
    detail::throw_error<Exception>(context, result.error());
}

// A progress poll permits pending, but must not silently accept ownership of a
// new conclusion obligation. Neither helper drives the operation itself.
[[nodiscard]] inline bool check_progress(Status status, const char *context) {
  if (status.is_pending())
    return false;
  if (status.is_success() && !status.requires_conclusion())
    return true;
  detail::throw_error<std::runtime_error>(context, status);
}

inline void check_acceptance(Status status, const char *context) {
  if (!status.requires_conclusion())
    detail::throw_error<std::runtime_error>(context, status);
}

#endif

} // namespace lpfx

#endif // LPFX_CORE_ERROR_REPORTING_HPP
