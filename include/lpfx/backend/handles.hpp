#ifndef LPFX_BACKEND_HANDLES_HPP
#define LPFX_BACKEND_HANDLES_HPP

#include <cstdlib>
#include <utility>

#include <lpfx/backend/traits.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>

namespace lpfx::device {

// A device executor is a non-owning launch destination. The application or a
// higher-level prepared owner is responsible for the native stream lifetime.
template <DeviceBackend Backend> class Executor final {
public:
  using backend_type = Backend;
  using traits_type = BackendTraits<Backend>;
  using stream_type = typename traits_type::stream_type;

  constexpr explicit Executor(const stream_type stream) noexcept
      : stream_(stream) {}

  [[nodiscard]] constexpr bool bound() const noexcept {
    return stream_ != traits_type::null_stream;
  }

  [[nodiscard]] constexpr stream_type stream() const noexcept {
    return stream_;
  }

private:
  stream_type stream_;
};

namespace detail {

template <class Backend> struct NativeStream final {
  using type = typename BackendTraits<Backend>::stream_type;
  static constexpr type null = BackendTraits<Backend>::null_stream;
  [[nodiscard]] static auto destroy(const type stream) noexcept {
    return BackendTraits<Backend>::destroy_stream(stream);
  }
};

template <class Backend> struct NativeEvent final {
  using type = typename BackendTraits<Backend>::event_type;
  static constexpr type null = BackendTraits<Backend>::null_event;
  [[nodiscard]] static auto destroy(const type event) noexcept {
    return BackendTraits<Backend>::destroy_event(event);
  }
};

// Sole owner of one native handle. A failed destroy during destruction or
// move assignment aborts: the runtime may already have consumed the handle.
template <class Backend, class Native> class UniqueNativeHandle {
public:
  using traits_type = BackendTraits<Backend>;
  using error_type = typename traits_type::error_type;
  using native_type = typename Native::type;

  constexpr UniqueNativeHandle() noexcept = default;

  UniqueNativeHandle(UniqueNativeHandle &&other) noexcept
      : handle_(std::exchange(other.handle_, Native::null)) {}

  UniqueNativeHandle &operator=(UniqueNativeHandle &&other) noexcept {
    if (this != &other) {
      destroy_or_abort();
      handle_ = std::exchange(other.handle_, Native::null);
    }
    return *this;
  }

  ~UniqueNativeHandle() noexcept { destroy_or_abort(); }

  [[nodiscard]] constexpr native_type native_handle() const noexcept {
    return handle_;
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return !(handle_ != Native::null);
  }

  // A destroy may surface an earlier asynchronous error after consuming the
  // handle, so the handle is released first and never retried.
  [[nodiscard]] error_type reset() noexcept {
    if (empty()) {
      return traits_type::success_error;
    }
    return Native::destroy(std::exchange(handle_, Native::null));
  }

protected:
  // output must be empty. A failed create leaves it unchanged.
  template <class Create>
  [[nodiscard]] static error_type create_handle(UniqueNativeHandle &output,
                                                const Create create) noexcept {
    if (!output.empty()) {
      std::abort();
    }
    native_type handle = Native::null;
    const error_type error = create(handle);
    if (traits_type::succeeded(error)) {
      output.handle_ = handle;
    }
    return error;
  }

private:
  void destroy_or_abort() noexcept {
    if (!traits_type::succeeded(reset())) {
      std::abort();
    }
  }

  native_type handle_ = Native::null;
};

} // namespace detail

enum class StreamPriority { normal, high };

// Setup-time stream ownership. Submission borrows it through the transparent
// Executor, which neither allocates nor adds synchronization. The application
// must retire borrowed operations before reset() or destruction.
template <DeviceStreamBackend Backend>
class DeviceStream final
    : public detail::UniqueNativeHandle<Backend, detail::NativeStream<Backend>> {
  using base_type =
      detail::UniqueNativeHandle<Backend, detail::NativeStream<Backend>>;

public:
  using traits_type = BackendTraits<Backend>;
  using stream_type = typename traits_type::stream_type;
  using error_type = typename traits_type::error_type;

  // Creates a nonblocking stream on the calling thread's selected device. Equal
  // priority limits give a normal-priority stream on devices without levels.
  template <StreamPriority Priority = StreamPriority::normal>
    requires(Priority == StreamPriority::normal ||
             DeviceStreamPriorityBackend<Backend>)
  [[nodiscard]] static error_type create(DeviceStream &output) noexcept {
    return base_type::create_handle(
        output, [](stream_type &stream) noexcept -> error_type {
          if constexpr (Priority == StreamPriority::normal) {
            return traits_type::create_stream(stream);
          } else {
            int least = 0, greatest = 0;
            const error_type queried =
                traits_type::query_stream_priority_range(least, greatest);
            return traits_type::succeeded(queried)
                       ? traits_type::create_stream(stream, greatest)
                       : queried;
          }
        });
  }

  [[nodiscard]] constexpr Executor<Backend> executor() const noexcept {
    return Executor<Backend>{this->native_handle()};
  }
};

// One reusable, timing-disabled completion marker. It carries no state
// machine or stored error; the submitting operation owns those semantics.
template <DeviceEventBackend Backend>
class DeviceEvent final
    : public detail::UniqueNativeHandle<Backend, detail::NativeEvent<Backend>> {
  using base_type =
      detail::UniqueNativeHandle<Backend, detail::NativeEvent<Backend>>;

public:
  using traits_type = BackendTraits<Backend>;
  using event_type = typename traits_type::event_type;
  using error_type = typename traits_type::error_type;

  [[nodiscard]] static error_type create(DeviceEvent &output) noexcept {
    return base_type::create_handle(
        output, [](event_type &event) noexcept -> error_type {
          return traits_type::create_event(event);
        });
  }

  [[nodiscard]] error_type record(const Executor<Backend> executor) noexcept {
    return traits_type::record_event(this->native_handle(), executor.stream());
  }

  // The producer must have issued record() before this call. The enclosing
  // operation owns affinity, record generation and reuse discipline; waiting
  // only inserts a dependency into the consumer stream and never blocks host
  // progress or reports the producer's device validation result.
  [[nodiscard]] error_type
  wait_on(const Executor<Backend> executor) const noexcept {
    return traits_type::stream_wait_event(executor.stream(),
                                          this->native_handle());
  }

  [[nodiscard]] error_type query() const noexcept {
    return traits_type::query_event(this->native_handle());
  }

  [[nodiscard]] error_type synchronize() const noexcept {
    return traits_type::synchronize_event(this->native_handle());
  }

  // Exactly one native observation of the last record(): success, pending
  // (only when !Wait) or backend_failure. The caller owns the recorded
  // generation, empty-event policy and failure handling.
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance() const noexcept {
    const error_type error = Wait ? synchronize() : query();
    if constexpr (!Wait) {
      if (traits_type::not_ready(error)) {
        return Status::pending();
      }
    }
    return traits_type::succeeded(error)
               ? Status::success()
               : Status::failure(ErrorCode::backend_failure);
  }
};

} // namespace lpfx::device

#endif
