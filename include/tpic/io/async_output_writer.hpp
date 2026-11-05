#ifndef TPIC_IO_ASYNC_OUTPUT_WRITER_HPP
#define TPIC_IO_ASYNC_OUTPUT_WRITER_HPP

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <lpfx/execution/host/fence.hpp>
#include <lpfx/execution/host/worker.hpp>

#include <tpic/io/output_engine.hpp>
#include <tpic/data/output/output_snapshot.hpp>

namespace tpic::io {

// One persistent host worker serializes one owned snapshot at a time. The
// worker never receives an executor or a device owner. One controller thread at
// a time calls the public methods. Destruction drains an accepted piece but
// neither publishes it nor reports its failure.
//
// configure -> start -> submit -> host_wait_piece -> rank-common agreement -> complete_snapshot
//                                                                 -> submit next epoch
//
// host_wait_piece rethrows write failures on the controller thread; the
// simulation's distributed abort path handles them. complete_snapshot
// publishes the manifest on rank zero and releases the slot on every rank.
// The agreement is deliberately outside this class and must occur at the same
// epoch boundary on all ranks, regardless of local completion times. A
// snapshot that publishes nothing (VTK fields alone) needs no agreement; its
// controller may complete it as late as just before the next submit.
class AsyncOutputWriter {
public:
  AsyncOutputWriter() = default;
  AsyncOutputWriter(const AsyncOutputWriter&) = delete;
  AsyncOutputWriter& operator=(const AsyncOutputWriter&) = delete;

  // The one place that checks what the engine's writes rely on: an enabled
  // channel, an output directory and a prepared domain. The engine
  // (make_output_engine, the checkpoint engine) obeys OutputEngine's ownership
  // and MPI contract.
  void configure(const config::PreparedOutputSelection& selection,
                 const config::PreparedDomain& domain, std::unique_ptr<OutputEngine> engine);
  // Starts the worker once, after configure; an unconfigured writer starts none.
  // setup runs once on the worker before its first piece. The engines' MPI
  // collectives may progress the rank's device transfers there, so a device
  // build binds the rank's device; a failure must not return.
  template <class Setup> void start(Setup setup) {
    if (configured_)
      worker_.start([this](std::uint64_t&& generation) noexcept { run_piece(generation); }, std::move(setup));
  }
  // Rejects a second snapshot until complete_snapshot releases the first;
  // no hidden queue or additional full-snapshot storage is allocated.
  void submit(data::OwnedOutputSnapshot&& snapshot);
  void host_wait_piece();
  // Returns the owned data after publication so the next capture can reuse
  // their allocations. The worker cannot access them after this handoff.
  data::OwnedOutputSnapshot complete_snapshot();

  [[nodiscard]] bool pending() const noexcept { return snapshot_.has_value(); }
  // Available only after completion/publication, when the writer is quiescent.
  // The engine's latest field and particle paths; empty for a disabled channel.
  [[nodiscard]] const std::string& fields_path() const;
  [[nodiscard]] const std::string& particles_path() const;

private:
  void validate(const data::OwnedOutputSnapshot& snapshot) const;
  // Worker program: checks the epoch order and calls engine_->write_piece.
  void run_piece(std::uint64_t generation) noexcept;

  std::unique_ptr<OutputEngine> engine_;
  std::vector<data::OutputArrayDescriptor> arrays_;
  std::size_t node_count_ = 0;
  std::size_t cell_count_ = 0;
  std::size_t species_count_ = 0;
  bool fields_enabled_ = false;
  bool particles_enabled_ = false;
  int last_cycle_ = -1;
  double last_time_ = 0.0;
  bool final_written_ = false;
  // A reusable value slot avoids allocating a wrapper for every output epoch;
  // moving the snapshot transfers its host buffers without copying their data.
  std::optional<data::OwnedOutputSnapshot> snapshot_;
  std::exception_ptr failure_;
  lpfx::HostFence completed_fence_;
  std::uint64_t generation_ = 0;
  bool configured_ = false;
  bool root_ = false;
  // Last member: destructor joins before any worker-visible state is destroyed.
  lpfx::HostWorker<std::uint64_t> worker_;
};

} // namespace tpic::io

#endif // TPIC_IO_ASYNC_OUTPUT_WRITER_HPP
