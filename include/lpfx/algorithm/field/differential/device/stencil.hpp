#ifndef LPFX_ALGORITHM_FIELD_DIFFERENTIAL_DEVICE_STENCIL_HPP
#define LPFX_ALGORITHM_FIELD_DIFFERENTIAL_DEVICE_STENCIL_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/meta.hpp>
#include <lpfx/core/private_construction.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/backend/operation_registration.hpp>
#include <lpfx/memory/space.hpp>
#include <lpfx/algorithm/field/differential/device/work.hpp>
#include <lpfx/algorithm/field/differential/operators.hpp>

namespace lpfx::detail {

template <class Backend, class Accumulator, class Binding, class Work,
          class SpacingScalar>
concept DeviceStencilLaunchable =
    device::DeviceBackend<Backend> &&
    requires(const Binding binding, const Work work,
             const DifferentialSpacing3D<SpacingScalar> spacing,
             const typename device::BackendTraits<Backend>::stream_type stream,
             bool &accepted) {
      {
        DeviceStencilLaunch<Backend>::template enqueue<Accumulator>(
            binding, work, spacing, stream, accepted)
      } noexcept
          -> std::same_as<typename device::BackendTraits<Backend>::error_type>;
    };

template <class Op, class Backend, class Accumulator, class SpacingScalar,
          class XComponent, class YComponent, class ZComponent,
          class... Targets>
concept DeviceStencilOperands =
    StencilOperands<Op, Accumulator, DeviceSpace<Backend>, XComponent,
                    YComponent, ZComponent, Targets...> &&
    DeviceStencilLaunchable<
        Backend, Accumulator,
        DeviceStencilBinding<
            StencilComponents<XComponent, YComponent, ZComponent>,
            StencilComponents<Targets...>>,
        PreparedDeviceStencilWork<Op, typename XComponent::index_policy>,
        SpacingScalar>;

template <std::floating_point Accumulator, class... Sources, class... Targets,
          class Op, CartesianIndexPolicy IndexPolicy,
          std::floating_point SpacingScalar, device::DeviceBackend Backend>
  requires DeviceStencilLaunchable<
      Backend, Accumulator,
      DeviceStencilBinding<StencilComponents<Sources...>,
                           StencilComponents<Targets...>>,
      PreparedDeviceStencilWork<Op, IndexPolicy>, SpacingScalar>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_stencil(
    const StencilComponents<Sources...> sources,
    const StencilComponents<Targets...> targets,
    const PreparedDeviceStencilWork<Op, IndexPolicy> work,
    const DifferentialSpacing3D<SpacingScalar> spacing,
    const device::Executor<Backend> executor) noexcept {
  using Binding = DeviceStencilBinding<StencilComponents<Sources...>,
                                       StencilComponents<Targets...>>;
  static_assert(packed_trivially_copyable<Binding, Sources..., Targets...>);
  static_assert(packed_trivially_copyable<DifferentialSpacing3D<SpacingScalar>,
                                          SpacingScalar, SpacingScalar,
                                          SpacingScalar>);
  if (work.empty()) {
    return ErrorCode::success;
  }
  if (!executor.bound()) {
    return ErrorCode::invalid_state;
  }
  bool accepted = false;
  const auto error = DeviceStencilLaunch<Backend>::template enqueue<
      Accumulator>(Binding{sources, targets}, work, spacing, executor.stream(),
                   accepted);
  return device::finish_device_enqueue<Backend>(error, accepted);
}

} // namespace lpfx::detail

namespace lpfx {

// Enqueues one stencil over previously prepared work; repeated epochs reuse
// the work without revalidation. Views and spacing are borrowed until later
// work on the stream completes. Hard preconditions: layouts describe the
// prepared cell extents, every source sample is addressable, sources do not
// overlap targets, target interiors are pairwise disjoint (one launch writes
// them concurrently), and each inverse spacing stays finite and positive as
// Accumulator. Nothing is retained.

template <std::floating_point Accumulator, device::DeviceBackend Backend,
          class EdgeXComponent, class EdgeYComponent, class EdgeZComponent,
          class FaceXComponent, class FaceYComponent, class FaceZComponent,
          std::floating_point SpacingScalar>
  requires detail::DeviceStencilOperands<
      detail::CurlEdgeToFace, Backend, Accumulator, SpacingScalar,
      EdgeXComponent, EdgeYComponent, EdgeZComponent, FaceXComponent,
      FaceYComponent, FaceZComponent>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_curl_edge_to_face(
    const EdgeXComponent edge_x, const EdgeYComponent edge_y,
    const EdgeZComponent edge_z, const FaceXComponent face_x,
    const FaceYComponent face_y, const FaceZComponent face_z,
    const PreparedDeviceCurlEdgeToFaceWork<
        typename EdgeXComponent::index_policy>
        work,
    const DifferentialSpacing3D<SpacingScalar> spacing,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_stencil<Accumulator>(
      detail::make_stencil_components(edge_x, edge_y, edge_z),
      detail::make_stencil_components(face_x, face_y, face_z), work, spacing,
      executor);
}

template <std::floating_point Accumulator, device::DeviceBackend Backend,
          class FaceXComponent, class FaceYComponent, class FaceZComponent,
          class EdgeXComponent, class EdgeYComponent, class EdgeZComponent,
          std::floating_point SpacingScalar>
  requires detail::DeviceStencilOperands<
      detail::CurlFaceToEdge, Backend, Accumulator, SpacingScalar,
      FaceXComponent, FaceYComponent, FaceZComponent, EdgeXComponent,
      EdgeYComponent, EdgeZComponent>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode enqueue_device_curl_face_to_edge(
    const FaceXComponent face_x, const FaceYComponent face_y,
    const FaceZComponent face_z, const EdgeXComponent edge_x,
    const EdgeYComponent edge_y, const EdgeZComponent edge_z,
    const PreparedDeviceCurlFaceToEdgeWork<
        typename FaceXComponent::index_policy>
        work,
    const DifferentialSpacing3D<SpacingScalar> spacing,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_stencil<Accumulator>(
      detail::make_stencil_components(face_x, face_y, face_z),
      detail::make_stencil_components(edge_x, edge_y, edge_z), work, spacing,
      executor);
}

template <std::floating_point Accumulator, device::DeviceBackend Backend,
          class EdgeXComponent, class EdgeYComponent, class EdgeZComponent,
          class NodeComponent, std::floating_point SpacingScalar>
  requires detail::DeviceStencilOperands<
      detail::DivergenceEdgeToNode, Backend, Accumulator, SpacingScalar,
      EdgeXComponent, EdgeYComponent, EdgeZComponent, NodeComponent>
[[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode
enqueue_device_divergence_edge_to_node(
    const EdgeXComponent edge_x, const EdgeYComponent edge_y,
    const EdgeZComponent edge_z, const NodeComponent node,
    const PreparedDeviceDivergenceEdgeToNodeWork<
        typename EdgeXComponent::index_policy>
        work,
    const DifferentialSpacing3D<SpacingScalar> spacing,
    const device::Executor<Backend> executor) noexcept {
  return detail::enqueue_device_stencil<Accumulator>(
      detail::make_stencil_components(edge_x, edge_y, edge_z),
      detail::make_stencil_components(node), work, spacing, executor);
}

} // namespace lpfx

#if LPFX_NATIVE_DEVICE_OPERATIONS
#include <lpfx/algorithm/field/differential/device/cuda_hip.hpp>
#endif

#endif
