#ifndef TPIC_IO_OUTPUT_ENGINE_HPP
#define TPIC_IO_OUTPUT_ENGINE_HPP

#include <memory>
#include <string>

#include <tpic/config/domain.hpp>
#include <tpic/config/output_selection.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/output/output_snapshot.hpp>

namespace lpfx::distributed {
class MpiFileWriter;
} // namespace lpfx::distributed

namespace tpic::io {

// Serialization engines borrow immutable views of an owned host snapshot. Engines do
// not create threads or access device owners. write_piece executes on the writer
// worker; the VTK field engine's is collective over the ranks through the field
// file writer, so every rank writes the same snapshot sequence, and no other engine
// calls MPI. publish runs on the root controller, after rank agreement for a
// snapshot that publishes.
// Construction is filesystem-free. A routed engine exposes the latest field
// snapshot via manifest_path() (a manifest, or the first present array's file
// of VTK fields), and the particle manifest separately.
//
// Precondition, enforced once by AsyncOutputWriter and not re-checked here:
// the selection has an enabled channel, an output directory and a prepared
// domain; each snapshot has the configured shape and only enabled channels;
// cycles and times increase up to one final snapshot; the root publishes each
// completed piece before the next write_piece, and never after a failed one.
class OutputEngine {
public:
  virtual ~OutputEngine() = default;
  virtual void write_piece(data::OutputSnapshotView snapshot) = 0;
  virtual void publish() = 0;
  [[nodiscard]] virtual const std::string& manifest_path() const noexcept = 0;
  // A channel engine has one manifest. A routed engine exposes both channels.
  [[nodiscard]] virtual const std::string& particles_path() const noexcept { return manifest_path(); }
};

// Routes each enabled channel to the engine its selection names. VTK fields
// write through field_files, which TransportRegistry opens on every rank
// before the first write.
[[nodiscard]] std::unique_ptr<OutputEngine> make_output_engine(
    const config::PreparedOutputSelection& selection, const config::PreparedDomain& domain,
    config::PrecisionMetadata precision, lpfx::distributed::MpiFileWriter& field_files);
// Single-channel engines; the channel is explicit, the selection supplies the
// directory, overwrite policy, field arrays and species count.
[[nodiscard]] std::unique_ptr<OutputEngine> make_vtk_field_engine(
    const config::PreparedOutputSelection& selection, const config::PreparedDomain& domain,
    config::PrecisionMetadata precision, lpfx::distributed::MpiFileWriter& field_files);
[[nodiscard]] std::unique_ptr<OutputEngine> make_vtk_particle_engine(
    const config::PreparedOutputSelection& selection,
    const config::PreparedDomain& domain, config::PrecisionMetadata precision);
[[nodiscard]] std::unique_ptr<OutputEngine> make_adios2_engine(
    data::OutputChannel channel, const config::PreparedOutputSelection& selection,
    const config::PreparedDomain& domain, config::PrecisionMetadata precision);
[[nodiscard]] std::unique_ptr<OutputEngine> make_hdf5_engine(
    data::OutputChannel channel, const config::PreparedOutputSelection& selection,
    const config::PreparedDomain& domain, config::PrecisionMetadata precision);

} // namespace tpic::io

#endif // TPIC_IO_OUTPUT_ENGINE_HPP
