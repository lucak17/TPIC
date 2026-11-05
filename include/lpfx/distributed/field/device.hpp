#ifndef LPFX_DISTRIBUTED_FIELD_DEVICE_HPP
#define LPFX_DISTRIBUTED_FIELD_DEVICE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/handles.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/core/memory_intrinsics.hpp>
#include <lpfx/data/field/view.hpp>
#include <lpfx/distributed/field/lowering.hpp>
#include <lpfx/memory/native_allocation.hpp>

namespace lpfx::distributed::detail {

inline constexpr std::uint32_t field_device_block_threads = 256;
inline constexpr std::uint32_t field_device_maximum_blocks_per_work = 2520;
static_assert(field_device_maximum_blocks_per_work <=
              std::numeric_limits<std::uint16_t>::max());

template <class Coordinate> struct DeviceFieldSecondBox final {
  Coordinate lower_x, lower_y, lower_z;
};
struct NoFieldSecondBox final {};

template <class FieldView, std::size_t Component>
using field_device_component_view_t =
    // HIP diagnoses std::declval as a host call when this alias is instantiated
    // from a device function. The pointer expression is likewise unevaluated,
    // while resolving only the host/device component() member.
    decltype(static_cast<const FieldView *>(nullptr)
                 ->template component<Component>());

template <class View>
inline constexpr bool field_device_needs_component_dispatch =
    []<std::size_t... I>(std::index_sequence<I...>) {
      using Element =
          typename field_device_component_view_t<View, 0>::element_type;
      return (!std::same_as<Element, typename field_device_component_view_t<
                                         View, I>::element_type> ||
              ...);
    }(std::make_index_sequence<View::component_count>{});

struct NoDeviceFieldView final {};
template <class View>
using DeviceFieldLaunchView =
    std::conditional_t<View::component_count == 1, View, NoDeviceFieldView>;

// Multi-component launches keep the selected component's binding alongside its
// geometry. The complete field pack is a kernel type, not a potentially huge
// kernel argument. Scalar launches pass their view inline.
template <lpfx::detail::CartesianIndexPolicy IndexPolicy,
          std::size_t Components, bool Dispatch = (Components > 1)>
struct DeviceFieldComponentBinding final {
  std::byte *core;
  typename IndexPolicy::stride_type stride_y, stride_z;
  std::uint16_t component;
};
template <lpfx::detail::CartesianIndexPolicy IndexPolicy,
          std::size_t Components>
  requires(Components > 1)
struct DeviceFieldComponentBinding<IndexPolicy, Components, false> final {
  std::byte *core;
  typename IndexPolicy::stride_type stride_y, stride_z;
};
template <lpfx::detail::CartesianIndexPolicy IndexPolicy, bool Dispatch>
struct DeviceFieldComponentBinding<IndexPolicy, 1, Dispatch> final {};

template <lpfx::detail::CartesianIndexPolicy IndexPolicy, bool Direct = true,
          std::size_t Components = 1, bool Dispatch = (Components > 1)>
struct DeviceFieldBoxWork final {
  static_assert(Components > 0 && Components <= UINT16_MAX);
  using coordinate_type = typename IndexPolicy::coordinate_type;
  using count_type = typename IndexPolicy::count_type;
  using binding_type =
      DeviceFieldComponentBinding<IndexPolicy, Components, Dispatch>;
  static constexpr bool needs_component_dispatch = Dispatch;
  static constexpr std::size_t component_count = Components;
  static constexpr bool has_direct_routes = Direct;
  using destination_type =
      std::conditional_t<Direct, DeviceFieldSecondBox<coordinate_type>,
                         NoFieldSecondBox>;

  coordinate_type source_lower_x;
  coordinate_type source_lower_y;
  coordinate_type source_lower_z;
  [[no_unique_address]] destination_type destination;
  count_type extent_x;
  count_type extent_y;
  count_type sample_count;
  std::size_t image_offset_bytes;
  std::uint32_t first_block;
  std::uint32_t block_count;
  FieldRuntimeRouteClass route;
  [[no_unique_address]] binding_type binding;
};

struct DeviceFieldLaunchBucket final {
  std::uint32_t work_offset{};
  std::uint32_t work_count{};
  std::uint32_t block_count{};
  // Setup classifies the complete bucket once. A nonzero value is the common
  // blocks-per-work count. Launching that case as a rectangular grid maps
  // blockIdx.y directly to the descriptor and blockIdx.x to its local block,
  // with no lookup or power-of-two restriction. Zero selects the general grid.
  std::uint16_t uniform_blocks_per_work{};
  // Every block covers at most one sample per thread unless setup had to cap
  // at field_device_maximum_blocks_per_work. Keeping this as prepared
  // launch metadata lets the common kernel contain no grid-stride loop.
  bool single_pass{};
};

template <class Work>
[[nodiscard]] constexpr DeviceFieldLaunchBucket make_field_device_launch_bucket(
    const Work *const work, const std::uint32_t work_offset,
    const std::uint32_t work_count, const std::uint32_t block_count) noexcept {
  DeviceFieldLaunchBucket bucket{work_offset, work_count, block_count};
  if (work_count == 0) {
    return bucket;
  }

  const std::uint32_t uniform_blocks = work[work_offset].block_count;
  // CUDA's portable grid-y floor is 65,535; current field buckets are much
  // smaller, but keep the classifier safe if their bounds grow later.
  bool uniform = uniform_blocks != 0 &&
                 uniform_blocks <= std::numeric_limits<std::uint16_t>::max() &&
                 work_count <= std::numeric_limits<std::uint16_t>::max();
  bool single_pass = true;
  for (std::uint32_t index = 0; index != work_count; ++index) {
    const Work &item = work[work_offset + index];
    const std::uint64_t expected_first =
        static_cast<std::uint64_t>(index) * uniform_blocks;
    uniform = uniform && item.block_count == uniform_blocks &&
              static_cast<std::uint64_t>(item.first_block) == expected_first;
    single_pass =
        single_pass && static_cast<std::uint64_t>(item.sample_count) <=
                           static_cast<std::uint64_t>(item.block_count) *
                               field_device_block_threads;
  }
  uniform =
      uniform && static_cast<std::uint64_t>(block_count) ==
                     static_cast<std::uint64_t>(work_count) * uniform_blocks;

  // Encode rectangular mapping only for the single-pass kernel so the launch
  // layer has exactly three reachable variants. Capped work retains its
  // grid-stride fallback even when several capped descriptors are uniform.
  if (uniform && single_pass) {
    bucket.uniform_blocks_per_work = static_cast<std::uint16_t>(uniform_blocks);
  }
  bucket.single_pass = single_pass;
  return bucket;
}

template <class Work>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr Work
select_field_device_box_work(const Work *const work,
                             const std::uint32_t work_count,
                             const std::uint32_t native_block) noexcept {
  // Preparation publishes descriptors in strictly increasing first-block
  // order. The block and lookup path are wave-uniform, allowing GPU compilers
  // to scalarize descriptor loads without shared storage or a block barrier.
  // Logarithmic search avoids a component/direction-count-dependent scan.
  std::uint32_t lower = 0;
  std::uint32_t upper = work_count;
  while (lower != upper) {
    const std::uint32_t middle = lower + (upper - lower) / 2;
    const Work candidate = work[middle];
    if (native_block < candidate.first_block) {
      upper = middle;
    } else if (native_block - candidate.first_block < candidate.block_count) {
      return candidate;
    } else {
      lower = middle + 1;
    }
  }
  return {};
}

template <bool UniformBlocks, class Work>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr Work
select_field_device_box_work(const Work *const work,
                             const std::uint32_t work_count,
                             const std::uint32_t native_block,
                             const std::uint32_t native_work) noexcept {
  // A prepared launch covers exactly the sum of its nonempty work block
  // spans, so every native block has one descriptor. Kernels rely on that
  // invariant and need no defensive empty-work branch after selection.
  if constexpr (UniformBlocks) {
    static_cast<void>(work_count);
    static_cast<void>(native_block);
    return work[native_work];
  } else {
    return select_field_device_box_work(work, work_count, native_block);
  }
}

template <bool UniformBlocks, class Work>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint32_t
field_device_box_local_block(const Work work, const std::uint32_t native_block,
                             const std::uint32_t native_work) noexcept {
  if constexpr (UniformBlocks) {
    static_cast<void>(work);
    static_cast<void>(native_work);
    return native_block;
  } else {
    return native_block - work.first_block;
  }
}

template <bool UniformBlocks, class Work>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint64_t
field_device_box_block_first_sample(const Work work,
                                    const std::uint32_t native_block,
                                    const std::uint32_t native_work) noexcept {
  return static_cast<std::uint64_t>(field_device_box_local_block<UniformBlocks>(
             work, native_block, native_work)) *
         field_device_block_threads;
}

// Preparation proves the mathematical sum is representable, including boxes
// crossing zero. Unsigned addition avoids signed overflow even for a uint64
// offset wider than INT64_MAX. C++20 integral conversion yields the destination
// value congruent modulo 2^N, so the final conversion recovers that proven sum
// without a per-sample sign/magnitude decision.
template <lpfx::detail::CartesianCoordinate Coordinate,
          lpfx::detail::CartesianUnsignedWord Offset>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr Coordinate
field_device_advance_coordinate(const Coordinate lower,
                                const Offset offset) noexcept {
  using UnsignedCoordinate = std::make_unsigned_t<Coordinate>;
  return static_cast<Coordinate>(static_cast<UnsignedCoordinate>(lower) +
                                 static_cast<UnsignedCoordinate>(offset));
}

template <bool Destination, class Work,
          lpfx::detail::CartesianCoordinate Coordinate>
LPFX_DEVICE LPFX_ALWAYS_INLINE void field_device_box_sample_coordinates(
    const Work work, const typename Work::count_type sample, Coordinate &i,
    Coordinate &j, Coordinate &k) noexcept {
  using Count = typename Work::count_type;
  const Count xy = work.extent_x * work.extent_y;
  const Count z = sample / xy;
  const Count xy_index = sample - z * xy;
  const Count y = xy_index / work.extent_x;
  const Count x = xy_index - y * work.extent_x;
  if constexpr (Destination && Work::has_direct_routes) {
    i = field_device_advance_coordinate(work.destination.lower_x, x);
    j = field_device_advance_coordinate(work.destination.lower_y, y);
    k = field_device_advance_coordinate(work.destination.lower_z, z);
  } else {
    i = field_device_advance_coordinate(work.source_lower_x, x);
    j = field_device_advance_coordinate(work.source_lower_y, y);
    k = field_device_advance_coordinate(work.source_lower_z, z);
  }
}

// Device execution images start at this alignment inside the (natively
// 256-byte aligned) execution allocation. With a single element type T every
// piece spans count * sizeof(T) bytes, so every image_offset_bytes is a
// multiple of sizeof(T) and each image sample is a naturally aligned T.
// Mixed element types (Work::needs_component_dispatch) keep the byte stream.
inline constexpr std::size_t field_device_image_alignment = 16;

// Compile-time selection of typed image samples: provably aligned, and a
// scalar whose load/store moves exactly its object representation. Other
// element types keep the byte-representation copy.
template <class T, class Work>
inline constexpr bool field_device_typed_image =
    !Work::needs_component_dispatch && std::is_arithmetic_v<T> &&
    !std::same_as<T, bool> && alignof(T) <= field_device_image_alignment;

template <class Backend> struct DeviceFieldPassLaunch;

#if defined(__CUDACC__) || defined(__HIPCC__)

template <class T, class Work>
[[nodiscard]] LPFX_DEVICE LPFX_ALWAYS_INLINE T *
field_device_image_sample(std::byte *const image, const Work work,
                          const typename Work::count_type sample) noexcept {
  static_assert(field_device_typed_image<std::remove_const_t<T>, Work>);
  return lpfx::detail::assume_aligned_pointer<alignof(T)>(
      reinterpret_cast<T *>(image + work.image_offset_bytes) + sample);
}

template <field_operation_kind Kind, FieldSemanticPass Pass,
          FieldRuntimeRouteClass Route, class View, class Work>
LPFX_DEVICE LPFX_ALWAYS_INLINE void
execute_field_device_sample(const View field, const Work work,
                            const typename Work::count_type sample,
                            std::byte *const image) noexcept {
  using T = typename View::value_type;
  using Coordinate = typename View::coordinate_type;
  using Traits = lpfx::detail::FieldPassTraits<Kind, Pass>;
  constexpr auto action = Traits::template action<Route>;
  constexpr auto source_kind = Traits::template source<Route>;
  Coordinate i{}, j{}, k{};
  if constexpr (field_device_typed_image<T, Work>) {
    // One naturally aligned T per sample: the same bits, loaded before the
    // destination action, in the same work order as the byte path.
    T value;
    if constexpr (source_kind == FieldSourceRequirement::field) {
      field_device_box_sample_coordinates<false>(work, sample, i, j, k);
      value = field(i, j, k);
    } else {
      value = *field_device_image_sample<const T>(image, work, sample);
    }
    if constexpr (action == FieldWorkAction::pack) {
      *field_device_image_sample<T>(image, work, sample) = value;
    } else {
      field_device_box_sample_coordinates<true>(work, sample, i, j, k);
      if constexpr (action == FieldWorkAction::overwrite) {
        field(i, j, k) = value;
      } else {
        static_assert(action == FieldWorkAction::add);
        static_assert(std::same_as<T, float> || std::same_as<T, double>);
#if defined(__CUDA_ARCH__)
        static_assert(!std::same_as<T, double> || __CUDA_ARCH__ >= 600,
                      "Double field addition requires CUDA sm_60 or newer");
#endif
        static_cast<void>(atomicAdd(&field(i, j, k), value));
      }
    }
    return;
  }
  const std::byte *source{};
  if constexpr (source_kind == FieldSourceRequirement::field) {
    field_device_box_sample_coordinates<false>(work, sample, i, j, k);
    source = reinterpret_cast<const std::byte *>(&field(i, j, k));
  } else {
    source = image + work.image_offset_bytes + sample * sizeof(T);
  }
  if constexpr (action == FieldWorkAction::pack) {
    lpfx::detail::copy_constant_size_representation<sizeof(T)>(
        image + work.image_offset_bytes + sample * sizeof(T), source);
  } else {
    field_device_box_sample_coordinates<true>(work, sample, i, j, k);
    if constexpr (action == FieldWorkAction::overwrite) {
      lpfx::detail::copy_constant_size_representation<sizeof(T)>(
          reinterpret_cast<std::byte *>(&field(i, j, k)), source);
    } else {
      static_assert(action == FieldWorkAction::add);
      static_assert(std::same_as<T, float> || std::same_as<T, double>);
#if defined(__CUDA_ARCH__)
      static_assert(!std::same_as<T, double> || __CUDA_ARCH__ >= 600,
                    "Double field addition requires CUDA sm_60 or newer");
#endif
      T value;
      lpfx::detail::copy_constant_size_representation<sizeof(T)>(
          reinterpret_cast<std::byte *>(&value), source);
      static_cast<void>(atomicAdd(&field(i, j, k), value));
    }
  }
}

template <field_operation_kind Kind, FieldSemanticPass Pass,
          FieldRuntimeRouteClass Route, bool SinglePass, bool Uniform,
          class View, class Work>
LPFX_DEVICE LPFX_ALWAYS_INLINE void execute_field_device_box(
    const View field, const Work work, const std::uint32_t block,
    const std::uint32_t native_work, std::byte *const image) noexcept {
  using Count = typename Work::count_type;
  const std::uint64_t first =
      field_device_box_block_first_sample<Uniform>(work, block, native_work) +
      threadIdx.x;
  if constexpr (SinglePass) {
    if (first < work.sample_count) {
      execute_field_device_sample<Kind, Pass, Route>(
          field, work, static_cast<Count>(first), image);
    }
  } else {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(work.block_count) * blockDim.x;
    for (std::uint64_t sample = first; sample < work.sample_count;
         sample += stride) {
      execute_field_device_sample<Kind, Pass, Route>(
          field, work, static_cast<Count>(sample), image);
    }
  }
}

template <field_operation_kind Kind, FieldSemanticPass Pass, bool SinglePass,
          bool Uniform, class View, class Work>
LPFX_DEVICE LPFX_ALWAYS_INLINE void execute_field_device_routes(
    const View field, const Work selected, const std::uint32_t block,
    const std::uint32_t native_work, std::byte *const remote,
    std::byte *const snapshot) noexcept {
  using Traits = lpfx::detail::FieldPassTraits<Kind, Pass>;
  // Selection is descriptor/block-uniform and occurs outside the sample loop.
  // No local/remote launch split, runtime action interpreter or per-sample
  // route branch is needed.
  if constexpr (Traits::template route_is_supported<
                    FieldRuntimeRouteClass::self_direct>) {
    if (selected.route == FieldRuntimeRouteClass::self_direct) {
      execute_field_device_box<Kind, Pass, FieldRuntimeRouteClass::self_direct,
                               SinglePass, Uniform>(field, selected, block,
                                                    native_work, nullptr);
      return;
    }
  }
  if constexpr (Traits::template route_is_supported<
                    FieldRuntimeRouteClass::self_snapshot>) {
    if (selected.route == FieldRuntimeRouteClass::self_snapshot) {
      execute_field_device_box<Kind, Pass,
                               FieldRuntimeRouteClass::self_snapshot,
                               SinglePass, Uniform>(field, selected, block,
                                                    native_work, snapshot);
      return;
    }
  }
  execute_field_device_box<Kind, Pass, FieldRuntimeRouteClass::remote,
                           SinglePass, Uniform>(field, selected, block,
                                                native_work, remote);
}

template <field_operation_kind Kind, FieldSemanticPass Pass, bool SinglePass,
          bool Uniform, class View, class Work>
__global__
__launch_bounds__(field_device_block_threads) void field_device_box_kernel(
    const View field, const Work *const work, const std::uint32_t count,
    std::byte *const remote, std::byte *const snapshot) {
  const auto block = static_cast<std::uint32_t>(blockIdx.x);
  const auto native_work = static_cast<std::uint32_t>(blockIdx.y);
  const Work selected =
      select_field_device_box_work<Uniform>(work, count, block, native_work);
  execute_field_device_routes<Kind, Pass, SinglePass, Uniform>(
      field, selected, block, native_work, remote, snapshot);
}

template <field_operation_kind Kind, FieldSemanticPass Pass, bool SinglePass,
          bool Uniform, class ComponentView, class Work>
LPFX_DEVICE LPFX_ALWAYS_INLINE void execute_field_device_bound_component(
    const Work selected, const std::uint32_t block,
    const std::uint32_t native_work, std::byte *const remote,
    std::byte *const snapshot) noexcept {
  const auto field = lpfx::detail::PrivateConstructionAccess::create<
      ComponentView3D<typename ComponentView::element_type,
                      DeviceSpace<typename ComponentView::backend_type>,
                      typename ComponentView::location_type,
                      typename ComponentView::index_policy>>(
          reinterpret_cast<typename ComponentView::element_type *>(selected.binding.core),
          selected.binding.stride_y, selected.binding.stride_z);
  execute_field_device_routes<Kind, Pass, SinglePass, Uniform>(
      field, selected, block, native_work, remote, snapshot);
}


template <field_operation_kind Kind, FieldSemanticPass Pass, bool SinglePass,
          bool Uniform, class FieldView, std::size_t Component = 1, class Work>
LPFX_DEVICE LPFX_ALWAYS_INLINE void dispatch_field_device_components(
    const Work selected, const std::uint32_t block,
    const std::uint32_t native_work, std::byte *const remote,
    std::byte *const snapshot) noexcept {
  static_assert(FieldView::component_count == Work::component_count);
  if constexpr (Component == FieldView::component_count) {
    // Preparation proves the ordinal valid. The first element type is the
    // fallback after testing every other distinct type; no bounds guard is
    // needed. Equal element types share a body even at different staggerings:
    // geometry, core pointers and strides already belong to the selected work.
    execute_field_device_bound_component<
        Kind, Pass, SinglePass, Uniform,
        field_device_component_view_t<FieldView, 0>>(
        selected, block, native_work, remote, snapshot);
  } else {
    using View = field_device_component_view_t<FieldView, Component>;
    using Element = typename View::element_type;
    constexpr bool seen = []<std::size_t... Earlier>(
                              std::index_sequence<Earlier...>) {
      return (std::same_as<Element, typename field_device_component_view_t<
                                       FieldView, Earlier>::element_type> || ...);
    }(std::make_index_sequence<Component>{});
    if constexpr (seen) {
      dispatch_field_device_components<Kind, Pass, SinglePass, Uniform, FieldView,
                                      Component + 1>(
          selected, block, native_work, remote, snapshot);
    } else {
      const bool matches = [&]<std::size_t... Candidates>(
                               std::index_sequence<Candidates...>) {
        return ((std::same_as<Element, typename field_device_component_view_t<
                                          FieldView, Candidates>::element_type> &&
                 selected.binding.component == Candidates) || ...);
      }(std::make_index_sequence<FieldView::component_count>{});
      if (matches) {
        execute_field_device_bound_component<Kind, Pass, SinglePass, Uniform, View>(
            selected, block, native_work, remote, snapshot);
      } else {
        dispatch_field_device_components<Kind, Pass, SinglePass, Uniform, FieldView,
                                        Component + 1>(
            selected, block, native_work, remote, snapshot);
      }
    }
  }
}

template <field_operation_kind Kind, FieldSemanticPass Pass, bool SinglePass,
          bool Uniform, class FieldView, std::size_t Component = 1, class Work>
LPFX_DEVICE LPFX_ALWAYS_INLINE void execute_field_device_components(
    const Work selected, const std::uint32_t block,
    const std::uint32_t native_work, std::byte *const remote,
    std::byte *const snapshot) noexcept {
  if constexpr (SinglePass) {
    // Mapping and bounds do not depend on the selected element type. Expose
    // their common value before dispatch so typed bodies reuse the same work.
    const std::uint64_t first =
        field_device_box_block_first_sample<Uniform>(selected, block, native_work) +
        threadIdx.x;
    if (first >= selected.sample_count) return;
  }
  dispatch_field_device_components<Kind, Pass, SinglePass, Uniform, FieldView, Component>(
      selected, block, native_work, remote, snapshot);
}

template <field_operation_kind Kind, FieldSemanticPass Pass, bool SinglePass,
          bool Uniform, class FieldView, class Work>
__global__ __launch_bounds__(field_device_block_threads)
void field_device_components_kernel(const Work *const work,
    const std::uint32_t count, std::byte *const remote,
    std::byte *const snapshot) {
  const auto block = static_cast<std::uint32_t>(blockIdx.x);
  const auto native_work = static_cast<std::uint32_t>(blockIdx.y);
  const Work selected =
      select_field_device_box_work<Uniform>(work, count, block, native_work);
  execute_field_device_components<Kind, Pass, SinglePass, Uniform, FieldView>(
      selected, block, native_work, remote, snapshot);
}

template <class Backend> struct DeviceFieldPassLaunch final {
  using traits_type = device::BackendTraits<Backend>;

  template <field_operation_kind Kind, FieldSemanticPass Pass, bool SinglePass,
            bool Uniform, class View, class Work>
  LPFX_ALWAYS_INLINE static void
  launch(const DeviceFieldLaunchView<View> field, const Work *const work,
         const DeviceFieldLaunchBucket bucket,
         const typename traits_type::stream_type stream,
         std::byte *const remote, std::byte *const snapshot) noexcept {
    const dim3 grid =
        Uniform ? dim3{bucket.uniform_blocks_per_work, bucket.work_count, 1}
                : dim3{bucket.block_count, 1, 1};
    static_assert(View::component_count == Work::component_count);
    if constexpr (View::component_count == 1) {
      field_device_box_kernel<Kind, Pass, SinglePass, Uniform>
          <<<grid, field_device_block_threads, 0, stream>>>(
              field.template component<0>(), work, bucket.work_count, remote, snapshot);
    } else {
      field_device_components_kernel<Kind, Pass, SinglePass, Uniform, View>
          <<<grid, field_device_block_threads, 0, stream>>>(
              work, bucket.work_count, remote, snapshot);
    }
  }

  template <field_operation_kind Kind, FieldSemanticPass Pass, class View,
            class Work>
  [[nodiscard]] LPFX_ALWAYS_INLINE static ErrorCode
  execute(const DeviceFieldLaunchView<View> field, const Work *const work,
          const DeviceFieldLaunchBucket bucket,
          const typename traits_type::stream_type stream,
          std::byte *const remote, std::byte *const snapshot) noexcept {
    if (bucket.uniform_blocks_per_work != 0) {
      launch<Kind, Pass, true, true, View>(field, work, bucket, stream, remote,
                                           snapshot);
    } else if (bucket.single_pass) {
      launch<Kind, Pass, true, false, View>(field, work, bucket, stream, remote,
                                            snapshot);
    } else {
      launch<Kind, Pass, false, false, View>(field, work, bucket, stream,
                                             remote, snapshot);
    }
    return device::to_error_code<Backend>(traits_type::take_last_launch_error());
  }
};

#endif

template <class Pattern, class Owner, class Executor>
  requires lpfx::detail::FieldOwnerMatchesOperationSpec<
               Owner, typename Pattern::spec_type, typename Pattern::index_policy> &&
           std::same_as<typename Owner::space_type,
                        DeviceSpace<typename Executor::backend_type>>
class DeviceFieldExecution final {
public:
  using pattern_type = Pattern;
  using owner_type = Owner;
  using spec_type = typename Pattern::spec_type;
  using backend_type = typename Executor::backend_type;
  using traits_type = device::BackendTraits<backend_type>;
  using view_type = lpfx::detail::mutable_field_owner_view_t<Owner>;
  using work_type =
      DeviceFieldBoxWork<typename Pattern::index_policy,
                         field_execution_has_direct_routes<spec_type>,
                         spec_type::component_count,
                         field_device_needs_component_dispatch<view_type>>;
  using layout_type = FieldExecutionLayout<spec_type>;
  using schedule_type = typename layout_type::schedule_type;
  static constexpr bool asynchronous = true;
  static_assert(traits_type::minimum_device_allocation_alignment >=
                alignof(work_type));
  static_assert(traits_type::minimum_device_allocation_alignment >=
                field_device_image_alignment);

  DeviceFieldExecution() noexcept = default;
  LPFX_NONMOVABLE(DeviceFieldExecution);

  // Validates the plan against the owner, binds every work record and
  // uploads them on the executor's stream. executor must outlive the
  // prepared state.
  template <class Plan>
    requires std::same_as<typename Plan::pattern_type, pattern_type>
  [[nodiscard]] ErrorCode prepare(HostMemoryBudget &budget, const Plan &plan,
                                  Owner &owner, const Executor &executor,
                                  FieldExecutionImages &images) noexcept {
    if (!allocation_.empty() || executor_ != nullptr || !executor.bound()) {
      return ErrorCode::invalid_state;
    }
    const pattern_type &pattern = plan.pattern();
    layout_type layout{};
    LPFX_TRY((prepare_field_execution_layout<work_type,
                                             field_device_image_alignment>(
        pattern, layout)));
    FieldComponentGeometries<Pattern> geometries{};
    LPFX_TRY(prepare_field_owner_geometry(pattern, owner, geometries));
    using Staging = device::AlignedPinnedHostAllocation<
        backend_type, field_host_allocation_alignment<work_type>>;
    std::size_t allocation_bytes = 0;
    HostMemoryRequirement requirement{};
    if (!Staging::try_allocation_bytes(layout.work_bytes, allocation_bytes) ||
        !requirement.try_add(allocation_bytes, Staging::alignment()))
      return ErrorCode::overflow;
    auto admitted = budget.try_admit(requirement);
    LPFX_TRY(admitted);
    // Destruction order keeps the native allocation charged until after free.
    auto admission = std::move(admitted).value();
    Staging staging{};
    LPFX_TRY(Staging::create(layout.work_bytes, staging));
    std::array<DeviceFieldLaunchBucket, schedule_type::pass_count> buckets{};
    const auto field = owner.view();
    std::array<std::span<work_type>, layout_type::bucket_count> work{};
    std::array<std::size_t, layout_type::bucket_count> cursors{};
    // Component spans are adjacent slices of one descriptor array. Launch
    // buckets traverse the full pass, including component boundaries.
    const std::span<work_type> staged_work{
        start_field_work_array<work_type>(staging.raw_base(), 0,
                                         layout.work_bytes / sizeof(work_type)),
        layout.work_bytes / sizeof(work_type)};
    for (std::size_t i = 0; i != work.size(); ++i) {
      work[i] = staged_work.subspan(layout.offsets[i] / sizeof(work_type),
                                   layout.counts[i]);
    }
    LPFX_TRY(schedule_type::try_for_each_active_pass(
        [&]<FieldSemanticPass Pass>(
            std::integral_constant<FieldSemanticPass, Pass>) noexcept {
          auto &pass = buckets[schedule_type::template pass_index<Pass>];
          constexpr auto first = layout_type::template bucket_index<Pass, 0>;
          if (!lpfx::detail::checked_integral_cast(
                  layout.offsets[first] / sizeof(work_type), pass.work_offset))
            return ErrorCode::overflow;
          for (std::size_t component = 0; component != spec_type::component_count;
               ++component) {
            std::uint32_t count{};
            if (!lpfx::detail::checked_integral_cast(layout.counts[first + component], count) ||
                !lpfx::detail::checked_add(pass.work_count, count, pass.work_count))
              return ErrorCode::overflow;
          }
          return lpfx::detail::try_for_each_field_work<Pass>(
              pattern, layout.schedule,
              [&](const auto component, const std::size_t d, const auto &piece,
                  const PreparedFieldRoute route) noexcept {
                constexpr auto bucket =
                    layout_type::template bucket_index<Pass, component.value>;
                return bind_box<Pass, component.value>(
                    pattern, geometries, d, piece, route, layout, field,
                    pass.block_count, work[bucket][cursors[bucket]++]);
              });
        }));
    for (auto &bucket : buckets) {
      if (bucket.block_count > static_cast<std::uint32_t>(INT32_MAX))
        return ErrorCode::capacity_exceeded;
      bucket = make_field_device_launch_bucket(staged_work.data(), bucket.work_offset,
                                               bucket.work_count,
                                               bucket.block_count);
    }
    device::DeviceAllocation<backend_type> allocation{};
    if (!traits_type::succeeded(device::DeviceAllocation<backend_type>::create(
            layout.total_bytes, allocation))) {
      return ErrorCode::allocation_failed;
    }
    if (layout.work_bytes != 0) {
      const auto copied = traits_type::copy_host_to_device_async(
          allocation.raw_base(), staging.raw_base(), layout.work_bytes,
          executor.stream());
      // The temporary descriptor bytes cannot be released until this cold
      // representation upload has completed, including an immediate error.
      if (!traits_type::succeeded(
              traits_type::synchronize_stream(executor.stream()))) {
        std::abort();
      }
      LPFX_TRY(device::to_error_code<backend_type>(copied));
    }
    device::DeviceEvent<backend_type> completion{};
    if (layout.work_bytes != 0) {
      LPFX_TRY(device::to_error_code<backend_type>(
          device::DeviceEvent<backend_type>::create(completion)));
    }
    images = {field_execution_image(allocation.raw_base(), layout, 0),
              field_execution_image(allocation.raw_base(), layout, 1)};
    outgoing_ = images.outgoing.data();
    incoming_ = images.incoming.data();
    snapshot_ = field_execution_image(allocation.raw_base(), layout, 2).data();
    buckets_ = buckets;
    if constexpr (view_type::component_count == 1)
      field_ = field;
    executor_ = &executor;
    completion_ = std::move(completion);
    allocation_ = std::move(allocation);
    return ErrorCode::success;
  }

  template <FieldSemanticPass Pass>
  [[nodiscard]] LPFX_ALWAYS_INLINE ErrorCode execute() noexcept {
    static_assert(schedule_type::template pass_is_active<Pass>);
    const auto bucket = buckets_[schedule_type::template pass_index<Pass>];
    if (bucket.work_count != 0) {
      const auto error = DeviceFieldPassLaunch<backend_type>::template execute<
          spec_type::kind, Pass, view_type>(
          field_,
          reinterpret_cast<const work_type *>(allocation_.raw_base()) +
              bucket.work_offset,
          bucket, executor_->stream(),
          Pass == FieldSemanticPass::source ? outgoing_ : incoming_, snapshot_);
      if (error != ErrorCode::success)
        return error;
    }
    return ErrorCode::success;
  }

  [[nodiscard]] ErrorCode record_completion() noexcept {
    return completion_.empty()
               ? ErrorCode::success
               : device::to_error_code<backend_type>(
                     completion_.record(*executor_));
  }
  template <bool Wait>
  [[nodiscard]] LPFX_ALWAYS_INLINE Status advance_apply() noexcept {
    if (completion_.empty())
      return Status::success();
    return completion_.template advance<Wait>();
  }
  [[nodiscard]] ErrorCode reset() noexcept {
    // The enclosing family program proves stream quiescence before teardown.
    const auto event_error = completion_.reset();
    const auto allocation_error = allocation_.reset();
    executor_ = nullptr;
    field_ = {};
    buckets_ = {};
    snapshot_ = nullptr;
    outgoing_ = nullptr;
    incoming_ = nullptr;
    return traits_type::succeeded(event_error) &&
                   traits_type::succeeded(allocation_error)
               ? ErrorCode::success
               : ErrorCode::backend_failure;
  }
  [[nodiscard]] std::size_t workspace_bytes() const noexcept {
    return allocation_.capacity_bytes();
  }

private:
  template <FieldSemanticPass Pass, std::size_t Component, class Piece>
  [[nodiscard]] static ErrorCode
  bind_box(const Pattern &pattern,
           const FieldComponentGeometries<Pattern> &geometries,
           const std::size_t d, const Piece &piece,
           const PreparedFieldRoute route, const layout_type &layout,
           const view_type &field, std::uint32_t &block_cursor,
           work_type &output) noexcept {
    lpfx::detail::FieldRouteBoxes<typename work_type::coordinate_type> boxes{};
    FieldExecutionBoxShape source{}, destination{};
    LPFX_TRY((prepare_field_route_shapes<Pass, Component>(
        pattern, geometries, d, piece, route, boxes, source, destination)));
    const auto box =
        Pass == FieldSemanticPass::source ? boxes.source : boxes.destination;
    CartesianExtent3<typename work_type::count_type> extents{};
    typename work_type::count_type samples{};
    LPFX_TRY(lpfx::detail::try_cartesian_box_shape(box, extents, samples));
    std::size_t count{};
    if (!lpfx::detail::checked_integral_cast(samples, count))
      return ErrorCode::overflow;
    if (route.route != FieldRuntimeRouteClass::self_direct) {
      const std::size_t image =
          lpfx::detail::field_image_index<Pass>(route.route);
      const std::size_t bytes =
          Pass == FieldSemanticPass::source ? source.bytes : destination.bytes;
      if (route.image_offset_bytes > layout.image_bytes[image] ||
          bytes > layout.image_bytes[image] - route.image_offset_bytes) {
        return ErrorCode::invalid_argument;
      }
    }
    std::uint64_t blocks = count / field_device_block_threads +
                           (count % field_device_block_threads != 0);
    if (blocks > field_device_maximum_blocks_per_work) {
      blocks = field_device_maximum_blocks_per_work;
    }
    const auto block_count = static_cast<std::uint32_t>(blocks);
    const auto source_lower =
        work_type::has_direct_routes ? boxes.source.lower() : box.lower();
    typename work_type::destination_type destination_lower{};
    if constexpr (work_type::has_direct_routes) {
      const auto lower = boxes.destination.lower();
      destination_lower = {lower.x(), lower.y(), lower.z()};
    }
    output = {source_lower.x(), source_lower.y(),
              source_lower.z(), destination_lower,
              extents.x(),      extents.y(),
              samples,          route.image_offset_bytes,
              block_cursor,     block_count,
              route.route,      {}};
    if constexpr (spec_type::component_count > 1) {
      const auto view = field.template component<Component>();
      output.binding.core = reinterpret_cast<std::byte *>(view.core_data());
      output.binding.stride_y = view.stride_y();
      output.binding.stride_z = view.stride_z();
      if constexpr (work_type::needs_component_dispatch)
        output.binding.component = static_cast<std::uint16_t>(Component);
    }
    return lpfx::detail::checked_add(block_cursor, block_count, block_cursor)
               ? ErrorCode::success
               : ErrorCode::overflow;
  }

  device::DeviceAllocation<backend_type> allocation_{};
  device::DeviceEvent<backend_type> completion_{};
  const Executor *executor_ = nullptr;
  [[no_unique_address]] DeviceFieldLaunchView<view_type> field_{};
  std::array<DeviceFieldLaunchBucket, schedule_type::pass_count> buckets_{};
  std::byte *snapshot_ = nullptr;
  std::byte *outgoing_ = nullptr;
  std::byte *incoming_ = nullptr;
};

} // namespace lpfx::distributed::detail

#endif
