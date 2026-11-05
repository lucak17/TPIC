#ifndef TPIC_IO_CHECKPOINT_HPP
#define TPIC_IO_CHECKPOINT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <variant>

#include <tpic/config/domain.hpp>
#include <tpic/config/output.hpp>
#include <tpic/data/output/checkpoint.hpp>
#include <tpic/io/output_engine.hpp>

namespace tpic::io {

// Checkpoints. <RestartDirName>/checkpoint_t<cycle>_n<ranks>/ holds one piece
// per writer rank, checkpoint_t<cycle>_r<rank>.bp (ADIOS2 BP5) or .h5 (HDF5).
// A piece is written under a hidden name, synced and renamed, so a checkpoint
// is complete when its directory lists the pieces of all ranks. Nothing here
// calls MPI: the callers agree on the results.

// The engine of the checkpoint writer, for snapshots with checkpoint_present.
// At the start of its piece the root removes the checkpoints older than the
// newest complete one and the set-aside ones; their errors are warnings.
[[nodiscard]] std::unique_ptr<OutputEngine> make_checkpoint_engine(
    const config::PreparedOutput& output, const config::PreparedDomain& domain);

// Root: the newest complete checkpoint of directory and its header. No
// complete checkpoint, a listing error or an unreadable header throws.
[[nodiscard]] data::CheckpointLocation find_restart_checkpoint(const std::string& directory);
// Root, after admission and before any rank writes a checkpoint: sets aside
// (renames abandoned_<name>) the checkpoints a run would mix its own with:
// for a restart those newer than its checkpoint, for a fresh run all of them,
// which it refuses without overwrite.
void prepare_checkpoint_directory(const std::string& directory, const data::CheckpointLocation* restart,
                                  bool overwrite);

// Every rank: the pieces of the checkpoint that hold the cells of domain, and
// the records of each species in them.
[[nodiscard]] data::PreparedRestart prepare_restart(data::CheckpointLocation checkpoint,
                                                    const config::PreparedDomain& domain);

// Restart reads. Values arrive in the stored types, which the header names.
using RestoredValues = std::variant<std::span<float>, std::span<double>>;

// One stored field component (data::checkpoint_field_names) of the restart,
// written into the local interior of a host field image: E at nodes, B at
// cells. origin addresses local (0,0,0); strides count values.
struct FieldDestination {
  std::variant<float*, double*> origin;
  std::size_t stride_y{}, stride_z{};
};
void read_restart_fields(const data::PreparedRestart& restart, const config::PreparedDomain& domain,
                         const std::array<FieldDestination, data::checkpoint_field_names.size()>& fields);

// The records of species s of the restart, in chunks of at most the staging
// capacity: deliver(count) runs when the first count records of the staging
// hold the next chunk, the cell keys rank-local to domain. A staging of
// restart_staging_records records bounds the startup memory of a restart.
inline constexpr std::size_t restart_staging_records = std::size_t{1} << 20;
struct ParticleStaging {
  std::array<RestoredValues, 3> fraction, velocity;
  RestoredValues charge;
  std::span<std::int32_t> cell;
  std::span<std::uint64_t> id;
};
void read_restart_particles(const data::PreparedRestart& restart, const config::PreparedDomain& domain,
                            std::size_t species, const ParticleStaging& staging,
                            const std::function<void(std::size_t)>& deliver);

} // namespace tpic::io

#endif // TPIC_IO_CHECKPOINT_HPP
