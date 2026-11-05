#ifndef LPFX_DISTRIBUTED_RUNTIME_ERROR_REPORTING_HPP
#define LPFX_DISTRIBUTED_RUNTIME_ERROR_REPORTING_HPP

#include <cstdio>

#include <lpfx/core/error_reporting.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>

namespace lpfx {

// Only captured values are inspected. No MPI call is safe to assume here:
// failures can occur before initialization or while finalizing the runtime.
inline int format_error(char *buffer, std::size_t size, const MpiResult result,
                        const char *context) noexcept {
  return std::snprintf(buffer, size, "%s: lpfx=%s(%u) native_result=%d",
                       detail::error_text(context),
                       error_code_name(result.code()),
                       static_cast<unsigned>(result.code()),
                       result.native_result());
}

#if LPFX_HAS_EXCEPTIONS
inline void check_mpi_error(const MpiResult result, const char *context) {
  if (!result.succeeded())
    detail::throw_error<std::runtime_error>(context, result);
}
#endif

} // namespace lpfx

#endif
