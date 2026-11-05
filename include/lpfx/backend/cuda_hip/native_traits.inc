// BackendTraits body shared by CUDA and HIP. Included only by
// cuda/backend.hpp (LPFX_NATIVE_HIP 0) and hip/backend.hpp (LPFX_NATIVE_HIP 1).

#include <cstddef>
#include <cstdint>

#if LPFX_NATIVE_HIP
#include <hip/hip_runtime_api.h>
#define LPFX_NATIVE(name) hip##name
#define LPFX_NATIVE_BACKEND HipBackend
#define LPFX_NATIVE_DEVICE_PROP hipDeviceProp_t
#define LPFX_NATIVE_POINTER_ATTRIBUTES hipPointerAttribute_t
#define LPFX_NATIVE_MAX_PITCH hipDeviceAttributeMaxPitch
#if defined(HIP_VERSION_MAJOR) &&                                              \
    (HIP_VERSION_MAJOR > 6 ||                                                  \
     (HIP_VERSION_MAJOR == 6 && defined(HIP_VERSION_MINOR) &&                  \
      HIP_VERSION_MINOR >= 1))
#define LPFX_NATIVE_STREAM_GET_DEVICE 1
#else
#define LPFX_NATIVE_STREAM_GET_DEVICE 0
#endif
#else
#include <cuda_runtime_api.h>
#define LPFX_NATIVE(name) cuda##name
#define LPFX_NATIVE_BACKEND CudaBackend
#define LPFX_NATIVE_DEVICE_PROP cudaDeviceProp
#define LPFX_NATIVE_POINTER_ATTRIBUTES cudaPointerAttributes
#define LPFX_NATIVE_MAX_PITCH cudaDevAttrMaxPitch
#if CUDART_VERSION >= 12080
#define LPFX_NATIVE_STREAM_GET_DEVICE 1
#else
#define LPFX_NATIVE_STREAM_GET_DEVICE 0
#endif
#endif

#include <lpfx/backend/traits.hpp>

namespace lpfx::device {

template <> struct BackendTraits<LPFX_NATIVE_BACKEND> final {
  using stream_type = LPFX_NATIVE(Stream_t);
  using event_type = LPFX_NATIVE(Event_t);
  using error_type = LPFX_NATIVE(Error_t);

  static constexpr stream_type null_stream = nullptr;
  static constexpr event_type null_event = nullptr;
  static constexpr error_type success_error = LPFX_NATIVE(Success);
  static constexpr std::size_t minimum_device_allocation_alignment = 256;

  [[nodiscard]] static constexpr bool
  succeeded(const error_type error) noexcept {
    return error == LPFX_NATIVE(Success);
  }

  [[nodiscard]] static const char *
  native_error_name(const error_type error) noexcept {
    const char *const name = LPFX_NATIVE(GetErrorName)(error);
    return name != nullptr ? name : "unknown";
  }

  [[nodiscard]] static const char *
  native_error_string(const error_type error) noexcept {
    const char *const description = LPFX_NATIVE(GetErrorString)(error);
    return description != nullptr ? description : "unknown";
  }

  [[nodiscard]] static constexpr bool
  runtime_unavailable(const error_type error) noexcept {
#if LPFX_NATIVE_HIP
    return error == hipErrorNoDevice || error == hipErrorInsufficientDriver;
#else
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
           error == cudaErrorSystemDriverMismatch;
#endif
  }

  [[nodiscard]] static error_type query_device_count(int &count) noexcept {
    int queried = 0;
    const error_type error = LPFX_NATIVE(GetDeviceCount)(&queried);
    if (succeeded(error)) {
      count = queried;
    }
    return error;
  }

  [[nodiscard]] static error_type
  query_current_device_memory(DeviceMemoryInfo &memory) noexcept {
    DeviceMemoryInfo queried{};
    const error_type error =
        LPFX_NATIVE(MemGetInfo)(&queried.free_bytes, &queried.total_bytes);
    if (succeeded(error)) {
      memory = queried;
    }
    return error;
  }

  [[nodiscard]] static bool
  is_default_stream(const stream_type stream) noexcept {
    return stream == null_stream || stream == LPFX_NATIVE(StreamLegacy) ||
           stream == LPFX_NATIVE(StreamPerThread);
  }

  [[nodiscard]] static error_type
  query_stream_nonblocking(const stream_type stream,
                           bool &nonblocking) noexcept {
    unsigned flags = 0;
    const error_type error = LPFX_NATIVE(StreamGetFlags)(stream, &flags);
    if (succeeded(error)) {
      nonblocking = (flags & LPFX_NATIVE(StreamNonBlocking)) != 0;
    }
    return error;
  }

  [[nodiscard]] static error_type create_stream(stream_type &stream) noexcept {
    return LPFX_NATIVE(StreamCreateWithFlags)(&stream,
                                              LPFX_NATIVE(StreamNonBlocking));
  }

  [[nodiscard]] static error_type create_stream(stream_type &stream,
                                                const int priority) noexcept {
    return LPFX_NATIVE(StreamCreateWithPriority)(
        &stream, LPFX_NATIVE(StreamNonBlocking), priority);
  }

  [[nodiscard]] static error_type
  query_stream_priority_range(int &least, int &greatest) noexcept {
    return LPFX_NATIVE(DeviceGetStreamPriorityRange)(&least, &greatest);
  }

  [[nodiscard]] static error_type select_device(const int ordinal) noexcept {
    return LPFX_NATIVE(SetDevice)(ordinal);
  }

  [[nodiscard]] static error_type
  destroy_stream(const stream_type stream) noexcept {
    return LPFX_NATIVE(StreamDestroy)(stream);
  }

  [[nodiscard]] static error_type
  synchronize_stream(const stream_type stream) noexcept {
    return LPFX_NATIVE(StreamSynchronize)(stream);
  }

  // A failed allocation also sets the per-thread error slot, which the next
  // launch bracket would report as an earlier launch failure. The caller
  // already receives the error, so clear the slot unless another error was
  // pending there before the allocation.
  static void consume_allocation_error(const error_type pending) noexcept {
    if (succeeded(pending) || not_ready(pending))
      static_cast<void>(LPFX_NATIVE(GetLastError)());
  }

  [[nodiscard]] static error_type
  allocate_device(std::byte *&allocation, const std::size_t bytes) noexcept {
    const error_type pending = LPFX_NATIVE(PeekAtLastError)();
    void *native_allocation = nullptr;
    const error_type error = LPFX_NATIVE(Malloc)(&native_allocation, bytes);
    if (succeeded(error))
      allocation = static_cast<std::byte *>(native_allocation);
    else
      consume_allocation_error(pending);
    return error;
  }

  [[nodiscard]] static error_type
  deallocate_device(std::byte *const allocation) noexcept {
    return LPFX_NATIVE(Free)(allocation);
  }

  [[nodiscard]] static error_type
  allocate_pinned_host(std::byte *&allocation,
                       const std::size_t bytes) noexcept {
    const error_type pending = LPFX_NATIVE(PeekAtLastError)();
    void *native_allocation = nullptr;
#if LPFX_NATIVE_HIP
    const error_type error =
        hipHostMalloc(&native_allocation, bytes, hipHostMallocDefault);
#else
    const error_type error = cudaMallocHost(&native_allocation, bytes);
#endif
    if (succeeded(error))
      allocation = static_cast<std::byte *>(native_allocation);
    else
      consume_allocation_error(pending);
    return error;
  }

  [[nodiscard]] static error_type
  deallocate_pinned_host(std::byte *const allocation) noexcept {
#if LPFX_NATIVE_HIP
    return hipHostFree(allocation);
#else
    return cudaFreeHost(allocation);
#endif
  }

  [[nodiscard]] static error_type
  copy_host_to_device_async(void *const destination, const void *const source,
                            const std::size_t bytes,
                            const stream_type stream) noexcept {
    return LPFX_NATIVE(MemcpyAsync)(destination, source, bytes,
                                    LPFX_NATIVE(MemcpyHostToDevice), stream);
  }

  [[nodiscard]] static error_type
  copy_device_to_host_async(void *const destination, const void *const source,
                            const std::size_t bytes,
                            const stream_type stream) noexcept {
    return LPFX_NATIVE(MemcpyAsync)(destination, source, bytes,
                                    LPFX_NATIVE(MemcpyDeviceToHost), stream);
  }

  [[nodiscard]] static error_type
  copy_device_to_device_async(void *const destination, const void *const source,
                              const std::size_t bytes,
                              const stream_type stream) noexcept {
    return LPFX_NATIVE(MemcpyAsync)(destination, source, bytes,
                                    LPFX_NATIVE(MemcpyDeviceToDevice), stream);
  }

  [[nodiscard]] static error_type copy_host_to_device_2d_async(
      void *const destination, const std::size_t destination_pitch,
      const void *const source, const std::size_t source_pitch,
      const std::size_t width_bytes, const std::size_t rows,
      const stream_type stream) noexcept {
    return LPFX_NATIVE(Memcpy2DAsync)(destination, destination_pitch, source,
                                      source_pitch, width_bytes, rows,
                                      LPFX_NATIVE(MemcpyHostToDevice), stream);
  }

  [[nodiscard]] static error_type copy_device_to_host_2d_async(
      void *const destination, const std::size_t destination_pitch,
      const void *const source, const std::size_t source_pitch,
      const std::size_t width_bytes, const std::size_t rows,
      const stream_type stream) noexcept {
    return LPFX_NATIVE(Memcpy2DAsync)(destination, destination_pitch, source,
                                      source_pitch, width_bytes, rows,
                                      LPFX_NATIVE(MemcpyDeviceToHost), stream);
  }

  [[nodiscard]] static error_type
  query_host_device_max_copy_pitch(std::size_t &max_pitch_bytes) noexcept {
    int ordinal = 0;
    error_type error = LPFX_NATIVE(GetDevice)(&ordinal);
    if (!succeeded(error)) {
      return error;
    }
    int native_pitch = 0;
    error = LPFX_NATIVE(DeviceGetAttribute)(&native_pitch,
                                            LPFX_NATIVE_MAX_PITCH, ordinal);
    if (!succeeded(error)) {
      return error;
    }
    if (native_pitch < 0) {
      return LPFX_NATIVE(ErrorInvalidValue);
    }
    max_pitch_bytes = static_cast<std::size_t>(native_pitch);
    return success_error;
  }

  [[nodiscard]] static error_type
  memset_device_async(void *const destination, const unsigned char value,
                      const std::size_t bytes,
                      const stream_type stream) noexcept {
    return LPFX_NATIVE(MemsetAsync)(destination, static_cast<int>(value), bytes,
                                    stream);
  }

  [[nodiscard]] static error_type create_event(event_type &event) noexcept {
    return LPFX_NATIVE(EventCreateWithFlags)(&event,
                                             LPFX_NATIVE(EventDisableTiming));
  }

  [[nodiscard]] static error_type
  create_timing_event(event_type &event) noexcept {
    return LPFX_NATIVE(EventCreateWithFlags)(&event, LPFX_NATIVE(EventDefault));
  }

  [[nodiscard]] static error_type
  event_elapsed_time(float &milliseconds, const event_type start,
                     const event_type stop) noexcept {
    float queried = 0;
    const error_type error =
        LPFX_NATIVE(EventElapsedTime)(&queried, start, stop);
    if (succeeded(error)) {
      milliseconds = queried;
    }
    return error;
  }

  [[nodiscard]] static error_type
  destroy_event(const event_type event) noexcept {
    return LPFX_NATIVE(EventDestroy)(event);
  }

  [[nodiscard]] static error_type
  record_event(const event_type event, const stream_type stream) noexcept {
    return LPFX_NATIVE(EventRecord)(event, stream);
  }

  [[nodiscard]] static error_type
  stream_wait_event(const stream_type stream, const event_type event) noexcept {
    return LPFX_NATIVE(StreamWaitEvent)(stream, event, 0);
  }

  [[nodiscard]] static error_type query_event(const event_type event) noexcept {
    return LPFX_NATIVE(EventQuery)(event);
  }

  [[nodiscard]] static constexpr bool
  not_ready(const error_type error) noexcept {
    return error == LPFX_NATIVE(ErrorNotReady);
  }

  [[nodiscard]] static error_type
  synchronize_event(const event_type event) noexcept {
    return LPFX_NATIVE(EventSynchronize)(event);
  }

  // Consumes the per-thread launch-error slot. Device launch leaves call this
  // immediately after each kernel so a failure cannot be attributed to a later
  // launch in the same submission.
  [[nodiscard]] static error_type take_last_launch_error() noexcept {
#if LPFX_NATIVE_HIP
    const error_type error = hipGetLastError();
    // HIP query calls can leave their expected not-ready result in the same
    // thread-local slot. It is a polling status, never a launch failure.
    // Normalize it here without adding another HIP call to every event poll.
    return not_ready(error) ? success_error : error;
#else
    return cudaGetLastError();
#endif
  }

  [[nodiscard]] static error_type
  query_current_device_properties(DeviceProperties &properties) noexcept {
    int device_ordinal = 0;
    error_type error = LPFX_NATIVE(GetDevice)(&device_ordinal);
    if (!succeeded(error)) {
      return error;
    }

    LPFX_NATIVE_DEVICE_PROP native_properties{};
    error = LPFX_NATIVE(GetDeviceProperties)(&native_properties, device_ordinal);
    if (!succeeded(error)) {
      return error;
    }

    properties = DeviceProperties{
        static_cast<std::int32_t>(device_ordinal),
        static_cast<std::uint32_t>(native_properties.warpSize),
        static_cast<std::uint32_t>(native_properties.maxThreadsPerBlock),
        static_cast<std::uint32_t>(native_properties.maxGridSize[0]),
        static_cast<std::uint32_t>(native_properties.multiProcessorCount)};
    return error;
  }

  [[nodiscard]] static error_type
  query_current_device_identity(DeviceIdentity &identity,
                                std::int32_t &device_ordinal) noexcept {
    int native_ordinal = 0;
    error_type error = LPFX_NATIVE(GetDevice)(&native_ordinal);
    if (!succeeded(error)) {
      return error;
    }

#if LPFX_NATIVE_HIP
    hipDevice_t device{};
    error = hipDeviceGet(&device, native_ordinal);
    if (!succeeded(error)) {
      return error;
    }
    hipUUID uuid{};
    error = hipDeviceGetUuid(&uuid, device);
    if (!succeeded(error)) {
      return error;
    }
#else
    cudaDeviceProp native_properties{};
    error = cudaGetDeviceProperties(&native_properties, native_ordinal);
    if (!succeeded(error)) {
      return error;
    }
    const cudaUUID_t &uuid = native_properties.uuid;
#endif

    DeviceIdentity queried{};
    for (std::size_t index = 0; index != DeviceIdentity::byte_count; ++index) {
      queried.bytes[index] = static_cast<std::uint8_t>(uuid.bytes[index]);
    }
    identity = queried;
    device_ordinal = static_cast<std::int32_t>(native_ordinal);
    return error;
  }

  [[nodiscard]] static error_type
  query_stream_device_ordinal(const stream_type stream,
                              std::int32_t &device_ordinal) noexcept {
#if LPFX_NATIVE_STREAM_GET_DEVICE
#if LPFX_NATIVE_HIP
    hipDevice_t native_ordinal{};
#else
    int native_ordinal = 0;
#endif
    const error_type error =
        LPFX_NATIVE(StreamGetDevice)(stream, &native_ordinal);
#else
    // Older runtimes (CUDA < 12.8, HIP < 6.1) lack a stream-device query. The
    // stream's association is then a documented hard precondition; the cold
    // check still proves that the calling thread kept setup's device current.
    static_cast<void>(stream);
    int native_ordinal = 0;
    const error_type error = LPFX_NATIVE(GetDevice)(&native_ordinal);
#endif
    if (succeeded(error)) {
      device_ordinal = static_cast<std::int32_t>(native_ordinal);
    }
    return error;
  }

  [[nodiscard]] static error_type
  query_device_pointer_ordinal(const void *const pointer,
                               std::int32_t &device_ordinal) noexcept {
    LPFX_NATIVE_POINTER_ATTRIBUTES attributes{};
    const error_type error =
        LPFX_NATIVE(PointerGetAttributes)(&attributes, pointer);
    if (!succeeded(error)) {
      return error;
    }
    device_ordinal = attributes.type == LPFX_NATIVE(MemoryTypeDevice) ||
                             attributes.type == LPFX_NATIVE(MemoryTypeManaged)
                         ? static_cast<std::int32_t>(attributes.device)
                         : std::int32_t{-1};
    return error;
  }
};

} // namespace lpfx::device

#undef LPFX_NATIVE_STREAM_GET_DEVICE
#undef LPFX_NATIVE_MAX_PITCH
#undef LPFX_NATIVE_POINTER_ATTRIBUTES
#undef LPFX_NATIVE_DEVICE_PROP
#undef LPFX_NATIVE_BACKEND
#undef LPFX_NATIVE
#undef LPFX_NATIVE_HIP
