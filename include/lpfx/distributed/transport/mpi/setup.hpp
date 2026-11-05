#ifndef LPFX_DISTRIBUTED_TRANSPORT_MPI_SETUP_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_MPI_SETUP_HPP

#include <cstdint>

#include <mpi.h>

#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/properties.hpp>
#include <lpfx/backend/traits.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>

namespace lpfx::distributed {

// Concrete non-owning setup for every production communication. It borrows one
// process domain and its bootstrap communicator and optionally records the
// single physical device selected by this Communication. Device facts are
// validated once, when the setup is made.
class CommunicationSetup final {
public:
  constexpr CommunicationSetup() noexcept = default;

  [[nodiscard]] constexpr bool configured() const noexcept {
    return configured_;
  }
  [[nodiscard]] bool is_valid() const noexcept {
    return configured_ && domain_->active();
  }

  [[nodiscard]] constexpr bool has_device_properties() const noexcept {
    return has_device_properties_;
  }
  [[nodiscard]] constexpr bool has_device_affinity() const noexcept {
    return has_device_affinity_;
  }

  // Borrow only while configured.
  [[nodiscard]] MpiCallDomain &domain() const noexcept { return *domain_; }
  // The domain's bootstrap communicator.
  [[nodiscard]] MPI_Comm native_handle() const noexcept {
    return domain_->bootstrap_communicator();
  }
  [[nodiscard]] constexpr int rank() const noexcept { return rank_; }
  [[nodiscard]] constexpr int size() const noexcept { return size_; }
  [[nodiscard]] constexpr const device::DeviceProperties &
  device_properties() const noexcept {
    return device_properties_;
  }
  [[nodiscard]] constexpr std::int32_t device_ordinal() const noexcept {
    return device_properties_.device_ordinal;
  }
  [[nodiscard]] constexpr const device::DeviceIdentity &
  device_identity() const noexcept {
    return device_identity_;
  }

  template <device::DeviceDiscoveryBackend Backend>
  [[nodiscard]] ErrorCode try_verify_current_device_affinity() const noexcept {
    if (!is_valid() || !has_device_affinity_) {
      return ErrorCode::invalid_state;
    }
    device::DeviceIdentity identity{};
    std::int32_t ordinal = -1;
    if (!device::BackendTraits<Backend>::succeeded(
            device::BackendTraits<Backend>::query_current_device_identity(
                identity, ordinal))) {
      return ErrorCode::backend_failure;
    }
    return identity.valid() && identity == device_identity_ &&
                   ordinal == device_properties_.device_ordinal
               ? ErrorCode::success
               : ErrorCode::invalid_argument;
  }

  template <device::DeviceDiscoveryBackend Backend>
  [[nodiscard]] ErrorCode try_verify_device_affinity(
      const device::Executor<Backend> &executor) const noexcept {
    LPFX_TRY(try_verify_current_device_affinity<Backend>());
    return try_verify_stream_affinity(executor);
  }

  template <device::DeviceDiscoveryBackend Backend>
  [[nodiscard]] ErrorCode
  try_verify_device_pointer_affinity(const void *const pointer) const noexcept {
    if (!is_valid() || !has_device_affinity_) {
      return ErrorCode::invalid_state;
    }
    if (pointer == nullptr) {
      return ErrorCode::invalid_argument;
    }
    std::int32_t pointer_ordinal = -1;
    if (!device::BackendTraits<Backend>::succeeded(
            device::BackendTraits<Backend>::query_device_pointer_ordinal(
                pointer, pointer_ordinal))) {
      return ErrorCode::backend_failure;
    }
    return pointer_ordinal == device_properties_.device_ordinal
               ? ErrorCode::success
               : ErrorCode::invalid_argument;
  }

  // The executor must be bound to the recorded device.
  template <device::DeviceDiscoveryBackend Backend>
  [[nodiscard]] ErrorCode try_verify_stream_affinity(
      const device::Executor<Backend> &executor) const noexcept {
    if (!executor.bound()) {
      return ErrorCode::invalid_argument;
    }
    std::int32_t stream_ordinal = -1;
    if (!device::BackendTraits<Backend>::succeeded(
            device::BackendTraits<Backend>::query_stream_device_ordinal(
                executor.stream(), stream_ordinal))) {
      return ErrorCode::backend_failure;
    }
    return stream_ordinal == device_properties_.device_ordinal
               ? ErrorCode::success
               : ErrorCode::invalid_argument;
  }

private:
  constexpr CommunicationSetup(MpiCallDomain &domain,
                               const bool has_device_properties,
                               const device::DeviceProperties properties,
                               const bool has_device_affinity,
                               const device::DeviceIdentity identity) noexcept
      : domain_(&domain), rank_(domain.rank()), size_(domain.size()),
        device_properties_(properties), device_identity_(identity),
        has_device_properties_(has_device_properties),
        has_device_affinity_(has_device_affinity), configured_(true) {}

  MpiCallDomain *domain_ = nullptr;
  int rank_ = 0;
  int size_ = 0;
  device::DeviceProperties device_properties_{};
  device::DeviceIdentity device_identity_{};
  bool has_device_properties_ = false;
  bool has_device_affinity_ = false;
  bool configured_ = false;

  friend struct lpfx::detail::PrivateConstructionAccess;
};

namespace detail {

[[nodiscard]] constexpr bool
valid_device_properties(const device::DeviceProperties &properties) noexcept {
  return properties.device_ordinal >= 0 && properties.subgroup_width != 0 &&
         properties.maximum_threads_per_block != 0 &&
         properties.maximum_grid_size_x != 0 &&
         properties.multiprocessor_count != 0;
}

} // namespace detail

// The setup borrows the domain's bootstrap communicator.
[[nodiscard]] inline MpiResult
try_make_communication_setup(MpiCallDomain &domain,
                             CommunicationSetup &output) noexcept {
  if (output.configured() || !domain.active()) {
    return ErrorCode::invalid_state;
  }
  output = lpfx::detail::PrivateConstructionAccess::create<CommunicationSetup>(
      domain, false, device::DeviceProperties{}, false,
      device::DeviceIdentity{});
  return {};
}

[[nodiscard]] inline MpiResult try_make_communication_setup(
    MpiCallDomain &domain, const device::DeviceProperties properties,
    CommunicationSetup &output) noexcept {
  if (output.configured() || !domain.active()) {
    return ErrorCode::invalid_state;
  }
  if (!detail::valid_device_properties(properties)) {
    return ErrorCode::invalid_argument;
  }
  output = lpfx::detail::PrivateConstructionAccess::create<CommunicationSetup>(
      domain, true, properties, false, device::DeviceIdentity{});
  return {};
}

// Records the current device and verifies that executor's stream belongs to it.
template <device::DeviceDiscoveryBackend Backend>
[[nodiscard]] MpiResult try_make_communication_setup(
    MpiCallDomain &domain, const device::DeviceProperties properties,
    const device::Executor<Backend> &executor,
    CommunicationSetup &output) noexcept {
  if (output.configured() || !domain.active()) {
    return ErrorCode::invalid_state;
  }
  if (!detail::valid_device_properties(properties) || !executor.bound()) {
    return ErrorCode::invalid_argument;
  }
  device::DeviceIdentity identity{};
  std::int32_t ordinal = -1;
  if (!device::BackendTraits<Backend>::succeeded(
          device::BackendTraits<Backend>::query_current_device_identity(
              identity, ordinal))) {
    return ErrorCode::backend_failure;
  }
  if (!identity.valid() || ordinal != properties.device_ordinal) {
    return ErrorCode::invalid_argument;
  }
  const auto candidate =
      lpfx::detail::PrivateConstructionAccess::create<CommunicationSetup>(
          domain, true, properties, true, identity);
  LPFX_TRY(candidate.try_verify_stream_affinity(executor));
  output = candidate;
  return {};
}

} // namespace lpfx::distributed

#endif
