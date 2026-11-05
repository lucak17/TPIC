#include <lpfx/backend/backend.hpp>

#include <cstddef>
#include <atomic>
#include <cstdio>
#include <limits>

// Link wrapping intercepts application/LPFx runtime copy submissions, excluding
// driver calls inside shared libraries. Tests share one object; application
// instrumentation is enabled only by TPIC_VALIDATE_PINNED_TRANSFERS.
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
#define TPIC_NATIVE(name) cuda##name
#define TPIC_WRAP(name) __wrap_cuda##name
#define TPIC_REAL(name) __real_cuda##name
using NativePointerAttributes = cudaPointerAttributes;
#else
#define TPIC_NATIVE(name) hip##name
#define TPIC_WRAP(name) __wrap_hip##name
#define TPIC_REAL(name) __real_hip##name
using NativePointerAttributes = hipPointerAttribute_t;
#endif

namespace {
std::atomic<std::size_t> uploads{}, downloads{};
struct AuditReport {
  ~AuditReport() {
    const auto h2d = uploads.load(std::memory_order_relaxed);
    const auto d2h = downloads.load(std::memory_order_relaxed);
    if (h2d != 0 || d2h != 0)
      std::fprintf(stderr, "TPIC pinned transfer audit: %zu H2D and %zu D2H endpoints checked\n", h2d, d2h);
  }
} report;

bool pinned(const void* pointer, std::size_t bytes) {
  if (bytes == 0) return true;
  if (pointer == nullptr) return false;
  NativePointerAttributes attributes{};
  const auto query = [&](const void* address) {
    if (TPIC_NATIVE(PointerGetAttributes)(&attributes, address) != TPIC_NATIVE(Success)) {
      static_cast<void>(TPIC_NATIVE(GetLastError)());
      return false;
    }
    return attributes.type == TPIC_NATIVE(MemoryTypeHost);
  };
  return query(pointer) && query(static_cast<const std::byte*>(pointer) + bytes - 1);
}

bool endpoint(void* destination, const void* source, std::size_t bytes,
              TPIC_NATIVE(MemcpyKind) kind) {
  if (bytes == 0) return true;
  if (kind == TPIC_NATIVE(MemcpyHostToDevice)) uploads.fetch_add(1, std::memory_order_relaxed);
  else if (kind == TPIC_NATIVE(MemcpyDeviceToHost)) downloads.fetch_add(1, std::memory_order_relaxed);
  const bool valid = kind == TPIC_NATIVE(MemcpyHostToDevice) ? pinned(source, bytes)
      : kind == TPIC_NATIVE(MemcpyDeviceToHost) ? pinned(destination, bytes) : true;
  if (!valid) std::fputs("TPIC transfer audit: pageable native copy endpoint\n", stderr);
  return valid;
}
} // namespace

extern "C" TPIC_NATIVE(Error_t) TPIC_REAL(MemcpyAsync)(
    void*, const void*, std::size_t, TPIC_NATIVE(MemcpyKind), TPIC_NATIVE(Stream_t));
extern "C" TPIC_NATIVE(Error_t) TPIC_WRAP(MemcpyAsync)(
    void* destination, const void* source, std::size_t bytes,
    TPIC_NATIVE(MemcpyKind) kind, TPIC_NATIVE(Stream_t) stream) {
  if (!endpoint(destination, source, bytes, kind)) return TPIC_NATIVE(ErrorInvalidValue);
  return TPIC_REAL(MemcpyAsync)(destination, source, bytes, kind, stream);
}

extern "C" TPIC_NATIVE(Error_t) TPIC_REAL(Memcpy2DAsync)(
    void*, std::size_t, const void*, std::size_t, std::size_t, std::size_t,
    TPIC_NATIVE(MemcpyKind), TPIC_NATIVE(Stream_t));
extern "C" TPIC_NATIVE(Error_t) TPIC_WRAP(Memcpy2DAsync)(
    void* destination, std::size_t destination_pitch, const void* source,
    std::size_t source_pitch, std::size_t width, std::size_t rows,
    TPIC_NATIVE(MemcpyKind) kind, TPIC_NATIVE(Stream_t) stream) {
  const auto pitch = kind == TPIC_NATIVE(MemcpyHostToDevice) ? source_pitch : destination_pitch;
  if (rows != 0 && width != 0) {
    if ((pitch != 0 && rows - 1 > (std::numeric_limits<std::size_t>::max() - width) / pitch) ||
        !endpoint(destination, source, (rows - 1) * pitch + width, kind))
      return TPIC_NATIVE(ErrorInvalidValue);
  }
  return TPIC_REAL(Memcpy2DAsync)(destination, destination_pitch, source, source_pitch,
                                 width, rows, kind, stream);
}
