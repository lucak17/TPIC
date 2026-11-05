#ifndef LPFX_API_PARTICLE_DEVICE_PARTICLES_HPP
#define LPFX_API_PARTICLE_DEVICE_PARTICLES_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

#include <lpfx/api/particle/host_particles.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/device/storage.hpp>
#include <lpfx/data/particle/device/transfer.hpp>
#include <lpfx/memory/lifetime.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx::particle {

template <lpfx::detail::ParticleSchemaType Schema,
          lpfx::device::DeviceAllocationBackend Backend>
using device = Particles<Schema, DeviceSpace<Backend>>;

// The native reusable transfers with self-owned staging, without a
// caller-owned budget.
template <lpfx::detail::ParticleSchemaType Schema, class Backend>
[[nodiscard]] auto
make_upload(Particles<Schema, DeviceSpace<Backend>> &destination) noexcept {
  return make_device_particle_upload(nullptr, destination);
}

template <class Backend, lpfx::detail::ParticleSchemaType Schema>
[[nodiscard]] auto
make_download(Particles<Schema, HostSpace> &destination) noexcept {
  return make_device_particle_download<Schema, Backend>(nullptr, destination);
}

// A nonzero initial size is uploaded before returning.
template <lpfx::detail::ParticleSchemaType Schema, class Backend>
  requires lpfx::detail::DeviceParticleTransferBackend<
      Backend, lpfx::detail::DeviceCopyDirection::to_device>
[[nodiscard]] Result<device<Schema, Backend>>
make(const options configuration, const ParticleValues<Schema> &initial,
     const lpfx::device::Executor<Backend> executor) noexcept {
  using Output = Result<device<Schema, Backend>>;
  if (configuration.size > configuration.capacity || !executor.bound()) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  if (configuration.capacity > std::numeric_limits<std::uint32_t>::max()) {
    return Output::from_error(ErrorCode::capacity_exceeded);
  }
  std::uint32_t capacity{};
  LPFX_TRY_RESULT(
      try_prepare_particle_capacity(
          static_cast<std::uint32_t>(configuration.capacity), capacity),
      Output);
  PreparedDeviceParticleStorage<Schema, Backend> storage{};
  LPFX_TRY_RESULT(try_prepare_device_particle_storage(capacity, storage),
                  Output);
  auto made = make_device_particles(storage);
  if (!made.has_value() || configuration.size == 0) {
    return made;
  }
  // One-shot pinned image of only the initialized records, constructed in
  // place; the copies complete before the size is published.
  using Layout = PreparedParticleLayout<Schema, HostSpace>;
  using Image = lpfx::device::AlignedPinnedHostAllocation<
      Backend, Layout::allocation_alignment>;
  using Traits = lpfx::device::BackendTraits<Backend>;
  const std::size_t size = configuration.size;
  Layout layout{};
  LPFX_TRY_RESULT(try_prepare_particle_layout(size, layout), Output);
  Image image{};
  LPFX_TRY_RESULT(Image::create(layout.total_bytes(), image), Output);
  lpfx::device::DeviceEvent<Backend> event{};
  if (!Traits::succeeded(lpfx::device::DeviceEvent<Backend>::create(event))) {
    return Output::from_error(ErrorCode::backend_failure);
  }
  const auto bases = layout.bases(image.raw_base());
  lpfx::for_each_attribute<Schema>([&](const auto attribute) noexcept {
    constexpr std::size_t I = decltype(attribute)::value;
    using T = typename Schema::template value_type<I>;
    T *const values =
        lpfx::detail::start_host_array_lifetime<T>(bases[I], size);
    for (std::size_t index = 0; index != size; ++index) {
      lpfx::detail::host_particle_store(values, index,
                                        initial.template get<I>());
    }
  });
  const auto destination =
      lpfx::detail::DeviceParticlesAccess::capacity_view(made.value());
  const ErrorCode copied =
      lpfx::for_each_attribute<Schema>([&](const auto attribute) noexcept {
        constexpr std::size_t I = decltype(attribute)::value;
        return lpfx::device::to_error_code<Backend>(
            Traits::copy_host_to_device_async(destination.template get<I>(),
                                              bases[I],
                                              size * Schema::widths[I],
                                              executor.stream()));
      });
  if (copied != ErrorCode::success ||
      !Traits::succeeded(event.record(executor)) ||
      !Traits::succeeded(event.synchronize())) {
    std::abort();
  }
  lpfx::detail::DeviceParticlesAccess::publish_size(
      made.value(), static_cast<std::uint32_t>(size));
  return made;
}

template <lpfx::detail::ParticleSchemaType Schema, class Backend>
  requires lpfx::detail::DeviceParticleTransferBackend<
               Backend, lpfx::detail::DeviceCopyDirection::to_device> &&
           lpfx::detail::ParticleValuesFactory<Schema>::default_supported
[[nodiscard]] Result<device<Schema, Backend>>
make(const options configuration,
     const lpfx::device::Executor<Backend> executor) noexcept {
  return make<Schema>(configuration, ParticleValues<Schema>{}, executor);
}

// Discard needs no device-copy or stream-synchronization capability. The
// executor must still be bound, and all uses of the old allocation must already
// be complete. Device growth replaces one feature at a time. Allocation failure
// leaves an exclusive pending mutation; retry the same resize to resume it.
template <lpfx::detail::ParticleSchemaType Schema,
          lpfx::device::DeviceAllocationBackend Backend,
          ParticleResizeMode Mode>
  requires(Mode == ParticleResizeMode::discard ||
           (lpfx::device::DeviceCopyBackend<Backend> &&
            lpfx::device::DeviceStreamBackend<Backend>))
[[nodiscard]] ErrorCode
resize_capacity(Particles<Schema, DeviceSpace<Backend>> &owner,
                const std::size_t required, resize_policy<Mode>,
                const lpfx::device::Executor<Backend> executor) noexcept {
  if ((!owner.valid() && !owner.growth_pending()) || !executor.bound())
    return ErrorCode::invalid_argument;
  if (required <= owner.capacity() && !owner.growth_pending()) {
    if constexpr (Mode == ParticleResizeMode::discard)
      owner.clear();
    return ErrorCode::success;
  }
  if (required > std::numeric_limits<std::uint32_t>::max())
    return ErrorCode::capacity_exceeded;
  std::uint32_t capacity{};
  LPFX_TRY(try_prepare_particle_capacity(static_cast<std::uint32_t>(required),
                                         capacity));
  return owner.template grow_in_place<Mode>(
      capacity, Mode == ParticleResizeMode::preserve ? owner.size() : 0U,
      executor);
}

template <lpfx::detail::ParticleSchemaType Schema,
          lpfx::device::DeviceAllocationBackend Backend>
  requires lpfx::device::DeviceCopyBackend<Backend> &&
           lpfx::device::DeviceStreamBackend<Backend>
[[nodiscard]] ErrorCode
resize_capacity(Particles<Schema, DeviceSpace<Backend>> &owner,
                const std::size_t required,
                const lpfx::device::Executor<Backend> executor) noexcept {
  return resize_capacity(owner, required, preserve, executor);
}

} // namespace lpfx::particle

#endif
