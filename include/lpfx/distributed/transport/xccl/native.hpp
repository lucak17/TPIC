#ifndef LPFX_DISTRIBUTED_TRANSPORT_XCCL_NATIVE_HPP
#define LPFX_DISTRIBUTED_TRANSPORT_XCCL_NATIVE_HPP

#include <lpfx/config.hpp>

// NCCL and RCCL deliberately share their C API. Configuration selects one
// vendor, and its public opt-in alone installs that route's entry lowerings.
// This native leaf owns no communicator, event, dispatch table or runtime state.
#if LPFX_CONFIG_ENABLE_NCCL && LPFX_CONFIG_ENABLE_RCCL
#error LPFX_XCCL_REQUIRES_EXACTLY_ONE_VENDOR
#elif LPFX_CONFIG_ENABLE_NCCL || LPFX_CONFIG_ENABLE_RCCL

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <lpfx/core/attributes.hpp>
#include <lpfx/distributed/composition/entry.hpp>
#include <lpfx/distributed/descriptor/collective.hpp>
#include <lpfx/distributed/transport/routes.hpp>
#include <lpfx/distributed/transport/xccl/session.hpp>

#if LPFX_CONFIG_ENABLE_NCCL
#if !LPFX_CONFIG_DEVICE_BACKEND_CUDA
#error LPFX_NCCL_REQUIRES_CUDA_BACKEND
#endif
#include <lpfx/backend/cuda/backend.hpp>
#include <nccl.h>
#else
#if !LPFX_CONFIG_DEVICE_BACKEND_HIP
#error LPFX_RCCL_REQUIRES_HIP_BACKEND
#endif
#include <lpfx/backend/hip/backend.hpp>
#include <rccl/rccl.h>
#endif

namespace lpfx::distributed::detail {

#if LPFX_CONFIG_ENABLE_NCCL
using XcclVendorRoute = transport::Nccl;
using XcclVendorBackend = device::CudaBackend;
inline constexpr CommunicationLoweringIdentity xccl_vendor_lowering =
    CommunicationLoweringIdentity::nccl;
#else
using XcclVendorRoute = transport::Rccl;
using XcclVendorBackend = device::HipBackend;
inline constexpr CommunicationLoweringIdentity xccl_vendor_lowering =
    CommunicationLoweringIdentity::rccl;
#endif

template <class T>
concept XcclCollectiveScalar =
    std::same_as<T, float> || std::same_as<T, double> ||
    std::same_as<T, std::int8_t> || std::same_as<T, std::uint8_t> ||
    std::same_as<T, std::int32_t> || std::same_as<T, std::uint32_t> ||
    std::same_as<T, std::int64_t> || std::same_as<T, std::uint64_t>;

struct XcclNativeCalls final {
  using device_backend_type = XcclVendorBackend;
  using stream_type = typename device::BackendTraits<XcclVendorBackend>::stream_type;
  using communicator_type = ncclComm_t;
  using unique_id_type = ncclUniqueId;
  using result_type = ncclResult_t;

  static constexpr communicator_type null_communicator = nullptr;
  static constexpr result_type success_result = ncclSuccess;
  static constexpr result_type in_progress_result = ncclInProgress;
  static constexpr result_type invalid_argument_result = ncclInvalidArgument;
  static constexpr int minimum_runtime_version = LPFX_CONFIG_XCCL_MINIMUM_VERSION;

  [[nodiscard]] static result_type query_runtime_version(int &version) noexcept {
    return ncclGetVersion(&version);
  }

  [[nodiscard]] static constexpr bool
  succeeded(const result_type result) noexcept {
    return result == success_result;
  }

  [[nodiscard]] static constexpr bool
  in_progress(const result_type result) noexcept {
    return result == in_progress_result;
  }

  [[nodiscard]] static constexpr bool
  valid_communicator(const communicator_type communicator) noexcept {
    return communicator != null_communicator;
  }

  [[nodiscard]] static result_type
  get_unique_id(unique_id_type &output) noexcept {
    return ncclGetUniqueId(&output);
  }

  [[nodiscard]] static result_type initialize(
      communicator_type &output, const int rank_count,
      const unique_id_type &unique_id, const int rank) noexcept {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    config.splitShare = 1;
    return ncclCommInitRankConfig(&output, rank_count, unique_id, rank, &config);
  }

  // The child shares the parent's resources and abort flag; its own blocking
  // configuration makes every operation launch inside the native call. Equal
  // keys keep the parent rank order. The vendor writes output from its own
  // thread once the parent's split job completes.
  [[nodiscard]] static result_type split(const communicator_type parent,
                                         communicator_type &output) noexcept {
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    return ncclCommSplit(parent, 0, 0, &output, &config);
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE static result_type query_async_error(
      const communicator_type communicator, result_type &state) noexcept {
    return ncclCommGetAsyncError(communicator, &state);
  }

  // Failure reports only.
  [[nodiscard]] static const char *
  error_string(const result_type result) noexcept {
    return ncclGetErrorString(result);
  }

  // Failure reports only: the text of the vendor's latest warning, kept
  // process-wide whether or not NCCL_DEBUG prints it.
  [[nodiscard]] static const char *
  last_error(const communicator_type communicator) noexcept {
    return ncclGetLastError(communicator);
  }

  [[nodiscard]] static result_type
  abort(const communicator_type communicator) noexcept {
    return ncclCommAbort(communicator);
  }

  [[nodiscard]] static result_type
  finalize(const communicator_type communicator) noexcept {
    return ncclCommFinalize(communicator);
  }

  [[nodiscard]] static result_type
  destroy(const communicator_type communicator) noexcept {
    return ncclCommDestroy(communicator);
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE static result_type group_start() noexcept {
    return ncclGroupStart();
  }

  [[nodiscard]] LPFX_ALWAYS_INLINE static result_type group_end() noexcept {
    return ncclGroupEnd();
  }

  template <bool Receive>
  [[nodiscard]] LPFX_ALWAYS_INLINE static result_type transfer(
      std::conditional_t<Receive, std::byte *, const std::byte *> data,
      const std::size_t bytes, const int peer,
      const communicator_type communicator, const stream_type stream) noexcept {
    if constexpr (Receive) {
      return ncclRecv(data, bytes, ncclUint8, peer, communicator, stream);
    } else {
      return ncclSend(data, bytes, ncclUint8, peer, communicator, stream);
    }
  }

  template <collective_kind Kind, reduction_operation Reduction,
            XcclCollectiveScalar T, bool ExactInPlace, std::uint32_t Root>
    requires(valid_reduction_operation<Reduction> &&
             (Kind == collective_kind::all_reduce ||
              Kind == collective_kind::reduce ||
              Kind == collective_kind::broadcast ||
              Kind == collective_kind::all_gather ||
              Kind == collective_kind::reduce_scatter) &&
             ((Kind != collective_kind::broadcast &&
               Kind != collective_kind::all_gather) ||
              Reduction == reduction_operation::sum) &&
             (!ExactInPlace ||
              (Kind != collective_kind::all_gather &&
               Kind != collective_kind::reduce_scatter)) &&
             ((Kind != collective_kind::reduce &&
               Kind != collective_kind::broadcast) ||
              Root <= static_cast<std::uint32_t>(
                          std::numeric_limits<int>::max())))
  [[nodiscard]] LPFX_ALWAYS_INLINE static result_type submit(
      const std::size_t count, const T *const input, T *const output,
      const communicator_type communicator, const stream_type stream) noexcept {
    constexpr ncclDataType_t datatype = native_datatype<T>();
    if constexpr (Kind == collective_kind::all_reduce) {
      return ncclAllReduce(input, output, count, datatype,
                           native_reduction<Reduction>(), communicator, stream);
    } else if constexpr (Kind == collective_kind::reduce) {
      return ncclReduce(input, output, count, datatype,
                        native_reduction<Reduction>(), static_cast<int>(Root),
                        communicator, stream);
    } else if constexpr (Kind == collective_kind::broadcast) {
      // The semantic operand exposes its one broadcast buffer as output on
      // every rank, including the root. Its input pointer is intentionally
      // not part of the native broadcast contract.
      return ncclBroadcast(output, output, count, datatype,
                           static_cast<int>(Root), communicator, stream);
    } else if constexpr (Kind == collective_kind::all_gather) {
      return ncclAllGather(input, output, count, datatype, communicator, stream);
    } else {
      return ncclReduceScatter(input, output, count, datatype,
                               native_reduction<Reduction>(), communicator,
                               stream);
    }
  }

private:
  template <XcclCollectiveScalar T>
  [[nodiscard]] static constexpr ncclDataType_t native_datatype() noexcept {
    if constexpr (std::same_as<T, float>) {
      return ncclFloat32;
    } else if constexpr (std::same_as<T, double>) {
      return ncclFloat64;
    } else if constexpr (std::same_as<T, std::int8_t>) {
      return ncclInt8;
    } else if constexpr (std::same_as<T, std::uint8_t>) {
      return ncclUint8;
    } else if constexpr (std::same_as<T, std::int32_t>) {
      return ncclInt32;
    } else if constexpr (std::same_as<T, std::uint32_t>) {
      return ncclUint32;
    } else if constexpr (std::same_as<T, std::int64_t>) {
      return ncclInt64;
    } else {
      return ncclUint64;
    }
  }

  template <reduction_operation Reduction>
  [[nodiscard]] static constexpr ncclRedOp_t native_reduction() noexcept {
    if constexpr (Reduction == reduction_operation::sum) {
      return ncclSum;
    } else if constexpr (Reduction == reduction_operation::product) {
      return ncclProd;
    } else if constexpr (Reduction == reduction_operation::minimum) {
      return ncclMin;
    } else {
      return ncclMax;
    }
  }
};

// The one configured vendor route and its stream-ordered completion variant;
// nccl.hpp and rccl.hpp only select them.
template <class Transport> struct XcclTransportConfiguration;
template <> struct XcclTransportConfiguration<XcclVendorRoute> {
  using native_type = XcclNativeCalls;
  static constexpr CommunicationLoweringIdentity lowering_identity =
      xccl_vendor_lowering;
  static constexpr bool completes_on_stream = false;
};
template <>
struct XcclTransportConfiguration<
    transport::StreamOrderedCompletion<XcclVendorRoute>>
    final : XcclTransportConfiguration<XcclVendorRoute> {
  static constexpr bool completes_on_stream = true;
};

} // namespace lpfx::distributed::detail

#endif
#endif
