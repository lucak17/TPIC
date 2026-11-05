#ifndef LPFX_DISTRIBUTED_RUNTIME_MPI_FILE_WRITER_HPP
#define LPFX_DISTRIBUTED_RUNTIME_MPI_FILE_WRITER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>

#include <mpi.h>

#include <lpfx/core/attributes.hpp>
#include <lpfx/core/checked_arithmetic.hpp>
#include <lpfx/core/error.hpp>
#include <lpfx/distributed/transport/mpi/agreement.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>
#include <lpfx/distributed/transport/mpi/collective.hpp>
#include <lpfx/distributed/transport/mpi/communicator.hpp>
#include <lpfx/distributed/transport/mpi/setup.hpp>

namespace lpfx::distributed {

// One rank's block of a C-ordered array whose elements hold `width` values:
// the array shape in the file, the block's offset and extent there, and the
// shape of the local memory array whose origin holds the block. Slowest axis
// first, counted in elements.
template <std::size_t Rank> struct MpiArrayBlock final {
  std::array<std::uint64_t, Rank> file_shape{}, file_offset{}, extent{},
      memory_shape{};
  std::uint64_t width = 1;
};

enum class MpiFileCreation : std::uint8_t {
  exclusive = 0, // fails on every rank when the file exists
  replace = 1,   // truncates an existing file to the written size
};

namespace detail {

// Error classes, unlike error codes, compare across ranks; 0 is success.
[[nodiscard]] inline std::uint64_t mpi_error_class(const int native) noexcept {
  int error_class = MPI_SUCCESS;
  if (native != MPI_SUCCESS &&
      MPI_Error_class(native, &error_class) != MPI_SUCCESS)
    error_class = MPI_ERR_OTHER;
  return static_cast<std::uint64_t>(error_class);
}

} // namespace detail

// Collective writes of shared files on a duplicate of the domain communicator.
// try_open and close run on every rank, on the initializing thread. Between
// them the writes of one rank are sequential, from any thread, and every rank
// issues them in the same order. MULTIPLE is required: under SERIALIZED the
// write's lease would hold the process gate through blocking file I/O.
class MpiFileWriter final {
public:
  constexpr MpiFileWriter() noexcept = default;
  LPFX_NONMOVABLE(MpiFileWriter);

  // Agreed; invalid_state on every rank unless every rank runs MULTIPLE.
  [[nodiscard]] ErrorCode try_open(const CommunicationSetup &setup) noexcept {
    if (context_.active() || !setup.is_valid()) return ErrorCode::invalid_state;
    const MpiCallDomain &domain = setup.domain();
    MpiCallLease lease{};
    if (try_acquire_shared_mpi_call(domain, lease) != ErrorCode::success)
      std::abort();
    LPFX_TRY(detail::agree_or_abort(
        domain.bootstrap_communicator(),
        domain.thread_mode() == MpiThreadMode::multiple
            ? ErrorCode::success
            : ErrorCode::invalid_state));
    return context_.open(domain);
  }

  // Frees the communicator; no write is in progress.
  void close() noexcept {
    if (!context_.active()) return;
    MpiCallLease lease{};
    if (try_acquire_shared_mpi_call(context_.domain(), lease) !=
        ErrorCode::success)
      std::abort();
    context_.close();
  }

  // Collective: every rank passes the same path, creation, head and tail
  // sizes, element type, width and file shape, and its own block; the blocks
  // tile the file array. Rank 0 writes head at offset 0 and tail after the
  // array, every rank its block in between. The result is agreed; its native
  // value is the largest MPI error class. A closed writer returns
  // invalid_state without a collective call.
  template <detail::MpiCollectiveScalar T, std::size_t Rank>
    requires(Rank != 0)
  [[nodiscard]] MpiResult
  host_write_array(const char *path, const MpiFileCreation creation,
                   const std::span<const std::byte> head,
                   const std::span<const std::byte> tail,
                   const MpiArrayBlock<Rank> &block,
                   const std::span<const T> memory) noexcept {
    namespace checked = ::lpfx::detail;
    if (!context_.active()) return ErrorCode::invalid_state;
    // The file communicator serves one call at a time.
    MpiCallLease lease{};
    if (try_acquire_disjoint_mpi_call(context_.domain(), lease) !=
        ErrorCode::success)
      std::abort();
    const MPI_Comm communicator = context_.native_handle();

    // MPI takes int dimensions and counts, and the file size is an
    // MPI_Offset. The width is the fastest axis of both subarrays.
    std::array<int, Rank + 1> sizes{}, subsizes{}, starts{}, memory_sizes{};
    const std::array<int, Rank + 1> origin{};
    int head_count = 0, tail_count = 0;
    bool valid = path != nullptr && block.width != 0 &&
                 checked::checked_integral_cast(block.width, sizes[Rank]) &&
                 checked::checked_integral_cast(head.size(), head_count) &&
                 checked::checked_integral_cast(tail.size(), tail_count) &&
                 (head.empty() || head.data() != nullptr) &&
                 (tail.empty() || tail.data() != nullptr);
    subsizes[Rank] = memory_sizes[Rank] = sizes[Rank];
    std::uint64_t payload = block.width * sizeof(T), values = block.width;
    bool empty = false;
    for (std::size_t axis = 0; valid && axis != Rank; ++axis) {
      valid = block.file_shape[axis] != 0 &&
              block.extent[axis] <= block.memory_shape[axis] &&
              block.file_offset[axis] <= block.file_shape[axis] &&
              block.extent[axis] <=
                  block.file_shape[axis] - block.file_offset[axis] &&
              checked::checked_integral_cast(block.file_shape[axis],
                                             sizes[axis]) &&
              checked::checked_integral_cast(block.extent[axis],
                                             subsizes[axis]) &&
              checked::checked_integral_cast(block.file_offset[axis],
                                             starts[axis]) &&
              checked::checked_integral_cast(block.memory_shape[axis],
                                             memory_sizes[axis]) &&
              checked::checked_multiply(payload, block.file_shape[axis],
                                        payload) &&
              checked::checked_multiply(values, block.memory_shape[axis],
                                        values);
      empty = empty || block.extent[axis] == 0;
    }
    constexpr auto offset_limit =
        static_cast<std::uint64_t>(std::numeric_limits<MPI_Offset>::max());
    valid = valid && memory.size() == values &&
            payload <= offset_limit - head.size() - tail.size();

    // Types are built before the first agreement, so a local failure stops
    // every rank. An empty block writes nothing through the element type.
    const MPI_Datatype element = detail::MpiCollectiveDatatype<T>::value;
    MPI_Datatype file_type = element, memory_type = element;
    int native = MPI_SUCCESS;
    if (valid && !empty) {
      native = MPI_Type_create_subarray(Rank + 1, sizes.data(), subsizes.data(),
                                        starts.data(), MPI_ORDER_C, element,
                                        &file_type);
      if (native == MPI_SUCCESS) native = MPI_Type_commit(&file_type);
      if (native == MPI_SUCCESS)
        native = MPI_Type_create_subarray(Rank + 1, memory_sizes.data(),
                                          subsizes.data(), origin.data(),
                                          MPI_ORDER_C, element, &memory_type);
      if (native == MPI_SUCCESS) native = MPI_Type_commit(&memory_type);
    }
    const auto free_types = [&]() noexcept {
      for (MPI_Datatype *type : {&memory_type, &file_type})
        if (*type != element) MPI_Type_free(type);
    };

    std::uint64_t path_hash = 14695981039346656037ULL; // FNV-1a
    for (const char *character = path; character != nullptr && *character != '\0';
         ++character)
      path_hash = (path_hash ^ static_cast<unsigned char>(*character)) *
                  1099511628211ULL;
    std::array<std::uint64_t, Rank + 6> invariants{
        static_cast<std::uint64_t>(creation), head.size(), tail.size(),
        block.width, sizeof(T), path_hash};
    for (std::size_t axis = 0; axis != Rank; ++axis)
      invariants[6 + axis] = block.file_shape[axis];
    std::array<std::uint64_t, 0> none{};
    const ErrorCode admitted = detail::agree_contract(
        communicator,
        !valid                  ? ErrorCode::invalid_argument
        : native != MPI_SUCCESS ? ErrorCode::backend_failure
                                : ErrorCode::success,
        invariants, none);
    if (admitted != ErrorCode::success) {
      free_types();
      return admitted;
    }

    MPI_File file = MPI_FILE_NULL;
    const int opened = MPI_File_open(
        communicator, path,
        MPI_MODE_WRONLY | MPI_MODE_CREATE |
            (creation == MpiFileCreation::exclusive ? MPI_MODE_EXCL : 0),
        MPI_INFO_NULL, &file);
    // The largest error class and whether any rank opened the file.
    std::array<std::uint64_t, 2> open_step{detail::mpi_error_class(opened),
                                           opened == MPI_SUCCESS ? 1U : 0U};
    static_cast<void>(detail::agree_contract(communicator, ErrorCode::success,
                                             none, open_step));
    if (open_step[0] != 0) {
      // A file open on some ranks only cannot be closed collectively.
      if (open_step[1] != 0) std::abort();
      free_types();
      return MpiResult{ErrorCode::backend_failure,
                       static_cast<int>(open_step[0])};
    }

    // Every rank makes every call from here on; a rank that failed joins the
    // remaining collectives without data.
    int first_error = MPI_SUCCESS;
    const auto step = [&first_error](const int result) noexcept {
      if (first_error == MPI_SUCCESS) first_error = result;
    };
    const bool root = context_.rank() == 0;
    const auto head_bytes = static_cast<MPI_Offset>(head.size());
    const auto tail_offset = head_bytes + static_cast<MPI_Offset>(payload);
    if (creation == MpiFileCreation::replace)
      step(MPI_File_set_size(file,
                             tail_offset + static_cast<MPI_Offset>(tail.size())));
    // Collective even for rank 0's bytes: hints that confine file access to
    // aggregators forbid independent writes.
    step(MPI_File_write_at_all(file, 0, head.data(), root ? head_count : 0,
                               MPI_BYTE, MPI_STATUS_IGNORE));
    step(MPI_File_write_at_all(file, tail_offset, tail.data(),
                               root ? tail_count : 0, MPI_BYTE,
                               MPI_STATUS_IGNORE));
    step(MPI_File_set_view(file, head_bytes, element, file_type, "native",
                           MPI_INFO_NULL));
    step(MPI_File_write_all(file, memory.data(),
                            first_error == MPI_SUCCESS && !empty ? 1 : 0,
                            memory_type, MPI_STATUS_IGNORE));
    step(MPI_File_close(&file));
    free_types();
    std::array<std::uint64_t, 1> written{detail::mpi_error_class(first_error)};
    static_cast<void>(detail::agree_contract(communicator, ErrorCode::success,
                                             none, written));
    if (written[0] != 0)
      return MpiResult{ErrorCode::backend_failure,
                       static_cast<int>(written[0])};
    return {};
  }

private:
  MpiFileContext context_{};
};

} // namespace lpfx::distributed

#endif
