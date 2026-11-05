#ifndef LPFX_PROFILING_HPP
#define LPFX_PROFILING_HPP

#include <lpfx/config.hpp>

// Keep even SDK includes and C++ guard declarations out of disabled builds.
#if LPFX_CONFIG_PROFILING_NVTX || LPFX_CONFIG_PROFILING_ROCTX
#include <cstdint>

#include <lpfx/core/attributes.hpp>

#if LPFX_CONFIG_PROFILING_NVTX
#include <nvtx3/nvToolsExt.h>
#elif LPFX_CONFIG_PROFILING_ROCTX_LEGACY
#include <roctracer/roctx.h>
#else
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace lpfx::profiling {

#if LPFX_CONFIG_PROFILING_NVTX
namespace detail {

// Stable across processes, threads and call sites, without a color registry.
// A wide RGB space avoids the frequent collisions of a small cyclic palette.
// Keep channels in [48, 239] for visibility on light and dark timelines.
constexpr std::uint32_t color_for_name(const char *name) noexcept {
  std::uint32_t hash = 2166136261U;
  for (; *name; ++name)
    hash = (hash ^ static_cast<unsigned char>(*name)) * 16777619U;
  // Avalanche the string hash before distributing its bytes to RGB channels.
  hash = (hash ^ (hash >> 16U)) * 0x85ebca6bU;
  hash = (hash ^ (hash >> 13U)) * 0xc2b2ae35U;
  hash ^= hash >> 16U;
  const auto channel = [](std::uint32_t value) constexpr {
    return 48U + ((value & 255U) * 192U >> 8U);
  };
  return 0xff000000U | (channel(hash >> 16U) << 16U) |
         (channel(hash >> 8U) << 8U) | channel(hash);
}

constexpr nvtxEventAttributes_t event_attributes(const char *name,
                                                 std::uint32_t argb) noexcept {
  nvtxEventAttributes_t event{};
  event.version = NVTX_VERSION;
  event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event.colorType = NVTX_COLOR_ARGB;
  event.color = argb;
  event.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event.message.ascii = name;
  return event;
}

} // namespace detail

struct Backend final {
  static void push(const char *name, std::uint32_t argb) noexcept {
    const auto event = detail::event_attributes(name, argb);
    (void)nvtxRangePushEx(&event);
  }
  static void push(const char *name) noexcept {
    push(name, detail::color_for_name(name));
  }
  static void pop() noexcept { (void)nvtxRangePop(); }
  static void mark(const char *name) noexcept {
    const auto event =
        detail::event_attributes(name, detail::color_for_name(name));
    nvtxMarkEx(&event);
  }
};
#else
struct Backend final {
  static void push(const char *name) noexcept { (void)roctxRangePush(name); }
  // ROCTx ranges do not carry color attributes.
  static void push(const char *name, std::uint32_t) noexcept { push(name); }
  static void pop() noexcept { (void)roctxRangePop(); }
  static void mark(const char *name) noexcept { roctxMark(name); }
};
#endif

// Host-only, thread-local lexical range. No state or runtime dispatch; a guard
// cannot escape its thread's range stack. No device completion is implied.
class ScopedRange final {
public:
  explicit ScopedRange(const char *name) noexcept { Backend::push(name); }
  ScopedRange(const char *name, std::uint32_t argb) noexcept {
    Backend::push(name, argb);
  }
  ~ScopedRange() noexcept { Backend::pop(); }
  LPFX_NONMOVABLE(ScopedRange);
};

} // namespace lpfx::profiling

#define LPFX_DETAIL_PROFILE_JOIN_IMPL(a, b) a##b
#define LPFX_DETAIL_PROFILE_JOIN(a, b) LPFX_DETAIL_PROFILE_JOIN_IMPL(a, b)
// One scope macro per source line in a given lexical scope.
#define LPFX_DETAIL_PROFILE_UNIQUE(prefix)                                     \
  LPFX_DETAIL_PROFILE_JOIN(prefix, __LINE__)

#define LPFX_PROFILE_SCOPE(name)                                               \
  [[maybe_unused]] ::lpfx::profiling::ScopedRange LPFX_DETAIL_PROFILE_UNIQUE(  \
      lpfx_profile_scope_){name}
#define LPFX_PROFILE_SCOPE_COLOR(name, argb)                                   \
  [[maybe_unused]] ::lpfx::profiling::ScopedRange LPFX_DETAIL_PROFILE_UNIQUE(  \
      lpfx_profile_scope_){name, argb}
#define LPFX_PROFILE_FUNCTION() LPFX_PROFILE_SCOPE(__func__)
#define LPFX_PROFILE_MARK(name) ::lpfx::profiling::Backend::mark(name)

#else

// Discard tokens, rather than calling a no-op function: arguments may be costly
// or unavailable, and this must remain zero-cost at -O0 as well as in Release.
#define LPFX_PROFILE_SCOPE(name) ((void)0)
#define LPFX_PROFILE_SCOPE_COLOR(name, argb) ((void)0)
#define LPFX_PROFILE_FUNCTION() ((void)0)
#define LPFX_PROFILE_MARK(name) ((void)0)

#endif

// CrayPat's recording state, independent of LPFX_PROFILING_BACKEND. On the
// process's main thread it switches recording for every thread of the process;
// on any other thread, for the calling thread only.
#if LPFX_CONFIG_ENABLE_CRAYPAT
#include <pat_api.h>

#define LPFX_PROFILE_RECORD(on)                                                \
  ((void)::PAT_record((on) ? PAT_STATE_ON : PAT_STATE_OFF))
#else
#define LPFX_PROFILE_RECORD(on) ((void)0)
#endif
#endif
