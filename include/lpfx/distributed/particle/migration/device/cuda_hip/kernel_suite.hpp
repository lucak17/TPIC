#ifndef LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_CUDA_HIP_KERNEL_SUITE_HPP
#define LPFX_DISTRIBUTED_PARTICLE_MIGRATION_DEVICE_CUDA_HIP_KERNEL_SUITE_HPP

#include <cstdint>
#include <type_traits>
#include <utility>

#include <lpfx/backend/cuda_hip/runtime.hpp>
#include <lpfx/backend/cuda_hip/subgroup.hpp>
#include <lpfx/backend/traits.hpp>
#if defined(__CUDACC__)
#include <lpfx/backend/cuda/backend.hpp>
#elif defined(__HIPCC__)
#include <lpfx/backend/hip/backend.hpp>
#endif
#include <lpfx/core/attributes.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/data/particle/device/cuda_hip/classification.hpp>
#include <lpfx/data/particle/device/view.hpp>
#include <lpfx/distributed/particle/migration/device/kernel_suite.hpp>
#include <lpfx/distributed/particle/wire/codec.hpp>

#if defined(__CUDACC__) || defined(__HIPCC__)

namespace lpfx::detail {

static_assert(std::is_same_v<std::uint32_t, unsigned int>);
static_assert(PackedDeviceParticleClassificationSlot::maximum_peer_count ==
              particle_migration_maximum_peer_count);
static_assert(PackedDeviceParticleClassificationSlot::resident_bucket ==
              device_particle_migration_resident_bucket);

__device__ inline void publish_device_particle_migration_failure(
    std::uint32_t *const failure_word,
    const DeviceParticleMigrationFailureWord failure) noexcept {
  static_cast<void>(atomicCAS(
      failure_word,
      static_cast<std::uint32_t>(DeviceParticleMigrationFailureWord::none),
      static_cast<std::uint32_t>(failure)));
}

// One thread per (category, sub-bucket): each 16-lane logical warp scans one
// category's sub-bucket counts in uint64, exactly the serial prefix. 16-lane
// segments tile both 32-lane warps and 64-lane wavefronts, and every lane of
// the block reaches the shuffles, so the full-warp primitive is converged.
inline constexpr std::uint32_t device_particle_finalize_threads =
    PackedDeviceParticleClassificationSlot::main_bucket_count *
    PackedDeviceParticleClassificationSlot::sub_bucket_count;
static_assert(device_particle_finalize_threads %
                  device::subgroup::maximum_warp_size ==
              0);

// The mover has already classified and reserved every particle's sub-bucket
// ordinal; only fixed-size metadata is inspected. This sole summary producer
// defines the complete output, including failed cases: sub-offsets of every
// addressable category up to the first rejected one, closing prefixes and
// counts below it, zero counts from it onward, and failure = none only for an
// unrejected table whose total equals source_count. A translation unit that
// instantiates no launch leaves this internal kernel unused.
[[maybe_unused]] static __global__ __launch_bounds__(
    device_particle_finalize_threads) void finalize_device_particle_registered_classification_kernel(
    const std::uint32_t source_count, const std::uint8_t peer_count,
    const DeviceParticleRegisteredClassificationSummary *const registration,
    std::uint32_t *const sub_offsets,
    DeviceParticleMigrationSummary *const summary) {
  constexpr std::uint32_t sub_count =
      PackedDeviceParticleClassificationSlot::sub_bucket_count;
  constexpr std::uint32_t categories =
      PackedDeviceParticleClassificationSlot::main_bucket_count;
  static_assert(categories == particle_migration_maximum_peer_count + 2 &&
                categories * sub_count == device_particle_finalize_threads);
  __shared__ std::uint64_t totals[categories];
  __shared__ std::uint32_t rejected;
  const std::uint32_t thread = threadIdx.x;
  const std::uint32_t category = thread / sub_count;
  const std::uint32_t sub = thread % sub_count;
  if (registration->failure != 0 ||
      peer_count > particle_migration_maximum_peer_count) {
    if (thread < device_particle_migration_bucket_count)
      summary->counts[thread] = 0;
    if (thread == 0)
      summary->failure = static_cast<std::uint32_t>(
          DeviceParticleMigrationFailureWord::invalid_record);
    return;
  }

  const std::uint64_t count = registration->counts[category][sub];
  std::uint64_t inclusive = count;
#pragma unroll
  for (std::uint32_t delta = 1; delta != sub_count; delta *= 2) {
    const std::uint64_t lower =
        device::subgroup::warp_shuffle_up<sub_count>(inclusive, delta);
    if (sub >= delta)
      inclusive += lower;
  }
  if (sub == sub_count - 1)
    totals[category] = inclusive;
  if (thread == 0)
    rejected = categories;
  __syncthreads();
  if (sub == sub_count - 1 &&
      (inclusive > source_count ||
       (category < particle_migration_maximum_peer_count &&
        category >= peer_count && inclusive != 0)))
    atomicMin(&rejected, category);
  __syncthreads();

  const std::uint32_t first_rejected = rejected;
  if (category < device_particle_migration_bucket_count) {
    std::uint32_t *const offsets = sub_offsets + category * (sub_count + 1);
    if (category <= first_rejected)
      offsets[sub] = static_cast<std::uint32_t>(inclusive - count);
    if (sub == 0) {
      const auto total = static_cast<std::uint32_t>(totals[category]);
      if (category < first_rejected)
        offsets[sub_count] = total;
      summary->counts[category] = category < first_rejected ? total : 0U;
    }
  }
  if (thread == 0) {
    std::uint64_t total{};
    for (std::uint32_t index = 0; index != categories; ++index)
      total += totals[index];
    summary->failure = static_cast<std::uint32_t>(
        first_rejected == categories && total == source_count
            ? DeviceParticleMigrationFailureWord::none
            : DeviceParticleMigrationFailureWord::invalid_record);
  }
}

// Registration reserved one unique ordinal and admission proved every prefix
// fits uint32, so the destination is a direct addition.
[[nodiscard]] __device__ LPFX_ALWAYS_INLINE std::uint32_t
registered_device_particle_slot(const std::uint64_t word,
                                const std::uint32_t bucket,
                                const std::uint32_t *const sub_offsets) {
  using Slot = PackedDeviceParticleClassificationSlot;
  constexpr std::uint32_t stride = Slot::sub_bucket_count + 1;
  return sub_offsets[bucket * stride + Slot::sub_bucket(word)] +
         static_cast<std::uint32_t>(Slot::ordinal(word));
}

template <class Writer>
__device__ LPFX_ALWAYS_INLINE void write_device_particle_registered_record(
    const std::uint32_t source, const std::uint64_t word,
    const std::uint32_t *const sub_offsets, std::uint32_t *const failure_word,
    const Writer &writer) {
  const auto bucket = static_cast<std::uint32_t>(word & UINT64_C(31));
  if (bucket < 2)
    return;
  const auto peer = static_cast<std::uint32_t>(bucket - 2);
  const auto slot = registered_device_particle_slot(word, peer, sub_offsets);
  if (!writer.write_peer(source, peer, slot))
    publish_device_particle_migration_failure(
        failure_word, DeviceParticleMigrationFailureWord::write_failure);
}

template <class Writer>
__global__ __launch_bounds__(
    device_particle_migration_block_threads) void write_device_particle_registered_buckets_kernel(
    const std::uint32_t source_count, const std::uint64_t *const slots,
    const std::uint32_t *const sub_offsets, std::uint32_t *const failure_word,
    Writer writer) {
  const std::uint32_t source =
      blockIdx.x * device_particle_migration_block_threads + threadIdx.x;
  if (source < source_count)
    write_device_particle_registered_record(source, slots[source], sub_offsets,
                                            failure_word, writer);
}

// Each pipeline classifies a record once and registers its bucket with a
// unique ordinal; later kernels consume that packed word.
template <class Classifier, device::DeviceBackend Backend>
__global__ __launch_bounds__(
    device_particle_migration_block_threads) void classify_device_particle_kernel(
    const std::uint32_t source_count,
    const DeviceParticleRegisteredClassificationView<Backend> registration,
    Classifier classifier) {
  const std::uint32_t source =
      blockIdx.x * device_particle_migration_block_threads + threadIdx.x;
  if (source >= source_count)
    return;
  std::uint8_t decision{};
  if (!classifier(source, decision)) {
    static_cast<void>(registration.invalidate());
    return;
  }
  static_cast<void>(registration.register_particle(source, decision));
}

template <ParticleSchemaType Schema, device::DeviceBackend Backend,
          class Protocol>
struct DeviceInitialParticleClassifier final {
  BasicParticleView<Schema, DeviceSpace<Backend>, true> source;
  Protocol protocol;

  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool
  operator()(const std::uint32_t index, std::uint8_t &bucket) const noexcept {
    return protocol.classify_initial(source, index, bucket) ==
           ErrorCode::success;
  }
};

template <ParticleWireContractType Contract, class Protocol>
struct DeviceReceivedParticleClassifier final {
  ConstDeviceParticlePeerWireImageView<Contract> source;
  Protocol protocol;
  std::uint64_t round;

  [[nodiscard]] __device__ LPFX_ALWAYS_INLINE bool
  operator()(const std::uint32_t index, std::uint8_t &bucket) const noexcept {
    ParticleWireCanonicalPosition<Contract> position{};
    const auto record = source.resolve(index);
    return load_particle_canonical_position(record.image, record.lane,
                                            position) &&
           protocol.classify_received(round, index, position, bucket) ==
               ErrorCode::success;
  }
};

// Each thread consumes one immutable slot: final records decode into their
// reserved arrival slot, transit records are copied into their peer packet.
template <ParticleWireContractType Contract, device::DeviceBackend Backend,
          class CodecContext>
__global__ __launch_bounds__(
    device_particle_migration_block_threads) void process_device_particle_registered_received_kernel(
    const std::uint32_t source_count, const std::uint64_t *const slots,
    const std::uint32_t *const sub_offsets,
    const ConstDeviceParticlePeerWireImageView<Contract> source,
    const BasicParticleView<typename Contract::schema_type,
                            DeviceSpace<Backend>, false> final_destination,
    const DeviceParticlePeerWireImageView<Contract> transit_destination,
    std::uint32_t *const failure_word, const CodecContext codec_context) {
  const std::uint32_t index =
      blockIdx.x * device_particle_migration_block_threads + threadIdx.x;
  if (index >= source_count)
    return;
  const std::uint64_t word = slots[index];
  const auto bucket = static_cast<std::uint32_t>(word & UINT64_C(31));
  if (bucket == 0)
    return;
  if (bucket == 1) {
    const auto slot = registered_device_particle_slot(
        word, device_particle_migration_resident_bucket, sub_offsets);
    const auto record = source.resolve(index);
    if (!decode_particle_record(record.image, record.lane, codec_context,
                                final_destination, slot))
      publish_device_particle_migration_failure(
          failure_word, DeviceParticleMigrationFailureWord::write_failure);
  } else {
    write_device_particle_registered_record(
        index, word, sub_offsets, failure_word,
        DeviceTransitParticleWireWriter<Contract>{source,
                                                  transit_destination});
  }
}

template <device::DeviceKernelLaunchBackend Backend>
struct CudaHipDeviceParticleKernelSuite {
  using traits_type = device::BackendTraits<Backend>;
  using error_type = typename traits_type::error_type;
  using stream_type = typename traits_type::stream_type;

  [[nodiscard]] LPFX_ALWAYS_INLINE static error_type
  enqueue_registered_initial_admission(
      const std::uint32_t source_count, const std::uint8_t peer_count,
      const DeviceParticleRegisteredClassificationSummary *const registration,
      std::uint32_t *const sub_offsets,
      DeviceParticleMigrationSummary *const summary,
      const stream_type stream) noexcept {
    finalize_device_particle_registered_classification_kernel<<<
        1, device_particle_finalize_threads, 0, stream>>>(
        source_count, peer_count, registration, sub_offsets, summary);
    return traits_type::take_last_launch_error();
  }

  template <ParticleSchemaType Schema, class Protocol>
    requires std::is_trivially_copyable_v<Protocol>
  [[nodiscard]] LPFX_ALWAYS_INLINE static error_type
  enqueue_initial_classification(
      const std::uint32_t source_count,
      const BasicParticleView<Schema, DeviceSpace<Backend>, true> source,
      const DeviceParticleRegisteredClassificationView<Backend> registration,
      const Protocol protocol, const stream_type stream) noexcept {
    const DeviceInitialParticleClassifier<Schema, Backend, Protocol>
        classifier{source, protocol};
    return launch<classify_device_particle_kernel<decltype(classifier),
                                                  Backend>>(
        source_count, stream, registration, classifier);
  }

  template <ParticleWireContractType Contract, class Protocol>
    requires std::is_trivially_copyable_v<Protocol>
  [[nodiscard]] LPFX_ALWAYS_INLINE static error_type
  enqueue_received_classification(
      const std::uint64_t round, const std::uint32_t source_count,
      const std::uint8_t peer_count,
      const ConstDeviceParticlePeerWireImageView<Contract> source,
      const DeviceParticleRegisteredClassificationView<Backend> registration,
      DeviceParticleMigrationSummary *const summary, const Protocol protocol,
      const stream_type stream) noexcept {
    const DeviceReceivedParticleClassifier<Contract, Protocol> classifier{
        source, protocol, round};
    const error_type error =
        launch<classify_device_particle_kernel<decltype(classifier),
                                               Backend>>(
            source_count, stream, registration, classifier);
    if (!traits_type::succeeded(error))
      return error;
    return enqueue_registered_initial_admission(
        source_count, peer_count, registration.summary(),
        registration.offsets(), summary, stream);
  }

  template <ParticleWireContractType Contract, class CodecContext>
    requires std::is_trivially_copyable_v<CodecContext>
  [[nodiscard]] LPFX_ALWAYS_INLINE static error_type
  enqueue_registered_initial_pack(
      const std::uint32_t source_count, const std::uint64_t *const decisions,
      const std::uint32_t *const sub_offsets,
      const BasicParticleView<typename Contract::schema_type,
                              DeviceSpace<Backend>, true>
          source,
      const DeviceParticlePeerWireImageView<Contract> destination,
      std::uint32_t *const failure_word, const CodecContext codec_context,
      const stream_type stream) noexcept {
    const auto writer = make_device_initial_particle_wire_writer<Contract>(
        source, codec_context, destination);
    return launch<write_device_particle_registered_buckets_kernel<
        decltype(writer)>>(source_count, stream, decisions, sub_offsets,
                           failure_word, writer);
  }

  template <ParticleWireContractType Contract, class CodecContext>
    requires std::is_trivially_copyable_v<CodecContext>
  [[nodiscard]] LPFX_ALWAYS_INLINE static error_type
  enqueue_received_processing(
      const std::uint32_t source_count, const std::uint64_t *const decisions,
      const std::uint32_t *const sub_offsets,
      const ConstDeviceParticlePeerWireImageView<Contract> source,
      const BasicParticleView<typename Contract::schema_type,
                              DeviceSpace<Backend>, false>
          final_destination,
      const DeviceParticlePeerWireImageView<Contract> transit_destination,
      std::uint32_t *const failure_word, const CodecContext codec_context,
      const stream_type stream) noexcept {
    return launch<process_device_particle_registered_received_kernel<
        Contract, Backend, CodecContext>>(
        source_count, stream, decisions, sub_offsets, source,
        final_destination, transit_destination, failure_word, codec_context);
  }

private:
  // The kernel is a non-type template argument: every instantiation is a
  // direct native launch.
  template <auto Kernel, class... Arguments>
  [[nodiscard]] LPFX_ALWAYS_INLINE static error_type
  launch(const std::uint32_t source_count, const stream_type stream,
         Arguments &&...arguments) noexcept {
    if (source_count == 0)
      return traits_type::success_error;
    constexpr std::uint32_t block_threads =
        device_particle_migration_block_threads;
    const std::uint32_t block_count =
        source_count / block_threads +
        static_cast<std::uint32_t>(source_count % block_threads != 0);
    Kernel<<<block_count, block_threads, 0, stream>>>(
        source_count, std::forward<Arguments>(arguments)...);
    return traits_type::take_last_launch_error();
  }
};

#if defined(__CUDACC__)
template <>
struct DeviceParticleKernelSuite<device::CudaBackend> final
    : CudaHipDeviceParticleKernelSuite<device::CudaBackend> {};
#endif

#if defined(__HIPCC__)
template <>
struct DeviceParticleKernelSuite<device::HipBackend> final
    : CudaHipDeviceParticleKernelSuite<device::HipBackend> {};
#endif

} // namespace lpfx::detail

#endif

#endif
