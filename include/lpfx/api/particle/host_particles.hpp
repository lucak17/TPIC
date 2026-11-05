#ifndef LPFX_API_PARTICLE_HOST_PARTICLES_HPP
#define LPFX_API_PARTICLE_HOST_PARTICLES_HPP

#include <cstddef>
#include <utility>

#include <lpfx/api/particle/schema.hpp>
#include <lpfx/data/particle/capacity.hpp>
#include <lpfx/data/particle/host/storage.hpp>

namespace lpfx::particle {

template <lpfx::detail::ParticleSchemaType Schema>
using host = Particles<Schema, HostSpace>;

// Capacity reserve is resolved during creation, never retained as owner state.
template <lpfx::detail::ParticleSchemaType Schema>
[[nodiscard]] Result<host<Schema>>
make(const options configuration,
     const ParticleValues<Schema> &initial) noexcept {
  using Output = Result<host<Schema>>;
  if (configuration.size > configuration.capacity) {
    return Output::from_error(ErrorCode::invalid_argument);
  }
  std::size_t capacity{};
  LPFX_TRY_RESULT(try_prepare_particle_capacity(configuration.capacity, capacity),
                  Output);
  PreparedParticleStorage<Schema> plan{};
  LPFX_TRY_RESULT(try_prepare_particle_layout(capacity, plan), Output);
  return make_host_particles(nullptr, plan, configuration.size, initial);
}

template <lpfx::detail::ParticleSchemaType Schema>
  requires lpfx::detail::ParticleValuesFactory<Schema>::default_supported
[[nodiscard]] Result<host<Schema>> make(const options configuration) noexcept {
  return make<Schema>(configuration, ParticleValues<Schema>{});
}

// Ensure space for at least required records, retaining existing capacity when
// it fits. The caller retires all consumers before resizing and refreshes any
// allocation-bound views/workspaces after growth. Discard publishes size zero
// even when no allocation is needed; it does not zero-initialize spare records.
template <lpfx::detail::ParticleSchemaType Schema,
          ParticleResizeMode Mode = ParticleResizeMode::preserve>
[[nodiscard]] ErrorCode resize_capacity(Particles<Schema, HostSpace> &owner,
                                        const std::size_t required,
                                        resize_policy<Mode> = {}) noexcept {
  if (!owner.valid())
    return ErrorCode::invalid_argument;
  if (required <= owner.capacity()) {
    if constexpr (Mode == ParticleResizeMode::discard)
      owner.clear();
    return ErrorCode::success;
  }
  std::size_t capacity{};
  LPFX_TRY(try_prepare_particle_capacity(required, capacity));
  auto candidate = make_grown_host_particles<Mode>(
      owner, capacity,
      Mode == ParticleResizeMode::preserve ? owner.size() : 0);
  LPFX_TRY(candidate);
  owner = std::move(candidate).value();
  return ErrorCode::success;
}

} // namespace lpfx::particle

#endif
