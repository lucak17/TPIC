#ifndef TPIC_TESTS_SUPPORT_MPI_FIELD_FILES_HPP
#define TPIC_TESTS_SUPPORT_MPI_FIELD_FILES_HPP

#include <lpfx/distributed/runtime/mpi_file_writer.hpp>
#include <lpfx/distributed/transport/mpi/call_domain.hpp>
#include <lpfx/distributed/transport/mpi/setup.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/config/output_selection.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/io/async_output_writer.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>

namespace tpic::test {

// MPI at MULTIPLE, initialized first and finalized after every member below.
struct MpiThreadMultiple {
  MpiThreadMultiple(int& argc, char**& argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS ||
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) != MPI_SUCCESS)
      std::abort();
  }
  ~MpiThreadMultiple() { MPI_Finalize(); }
  MpiThreadMultiple(const MpiThreadMultiple&) = delete;
  MpiThreadMultiple& operator=(const MpiThreadMultiple&) = delete;
};

// The call domain, the setup and the open field file writer of a host test
// that writes VTK fields, the way the run's transport registry holds them.
class MpiFieldFiles : MpiThreadMultiple {
public:
  MpiFieldFiles(int& argc, char**& argv) : MpiThreadMultiple(argc, argv) {
    if (!lpfx::try_make_mpi_call_domain(MPI_COMM_WORLD,
            lpfx::MpiProcessThreadUse::other_process_threads_may_exist, domain).succeeded() ||
        !lpfx::distributed::try_make_communication_setup(domain, setup).succeeded() ||
        files.try_open(setup) != lpfx::ErrorCode::success)
      std::abort();
  }
  ~MpiFieldFiles() { files.close(); }

  [[nodiscard]] int rank() const noexcept { return domain.rank(); }
  [[nodiscard]] int ranks() const noexcept { return domain.size(); }

  // Runs a test body; a failure on any rank aborts every rank, since its
  // peers may be inside a collective write.
  template <class Body> int run(const char* name, Body&& body) noexcept {
    try {
      body();
      return 0;
    } catch (const std::exception& failure) {
      std::fprintf(stderr, "%s rank %d: %s\n", name, rank(), failure.what());
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }

  lpfx::MpiCallDomain domain;
  lpfx::distributed::CommunicationSetup setup;
  lpfx::distributed::MpiFileWriter files;
};

// Configures and starts a writer with the engines its selection names, as the
// run does; VTK fields write through files.
inline void configure(io::AsyncOutputWriter& writer, const config::PreparedOutput& prepared,
                      const config::PreparedDomain& domain, lpfx::distributed::MpiFileWriter& files,
                      config::PrecisionMetadata precision = config::precision_metadata(config::PrecisionSelection{
                          config::StoragePrecision::double_precision, config::ComputePrecision::double_precision})) {
  const config::PreparedOutputSelection selection(prepared);
  writer.configure(selection, domain, io::make_output_engine(selection, domain, std::move(precision), files));
  writer.start([]() noexcept {});
}

} // namespace tpic::test

#endif // TPIC_TESTS_SUPPORT_MPI_FIELD_FILES_HPP
