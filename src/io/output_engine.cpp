#include <tpic/io/output_engine.hpp>

#include <stdexcept>

namespace tpic::io {
namespace {

std::unique_ptr<OutputEngine> make_channel_engine(
    data::OutputChannel channel, config::OutputEngineKind engine,
    const config::PreparedOutputSelection& selection, const config::PreparedDomain& domain,
    config::PrecisionMetadata precision, lpfx::distributed::MpiFileWriter& field_files) {
  switch (engine) {
  case config::OutputEngineKind::Default:
  case config::OutputEngineKind::Vtk:
    return channel == data::OutputChannel::fields
        ? make_vtk_field_engine(selection, domain, precision, field_files)
        : make_vtk_particle_engine(selection, domain, precision);
  case config::OutputEngineKind::Adios2:
#if defined(TPIC_HAS_ADIOS2) && TPIC_HAS_ADIOS2
    return make_adios2_engine(channel, selection, domain, precision);
#else
    throw std::invalid_argument("TPIC output: ADIOS2 output requires a build with TPIC_ENABLE_ADIOS2=ON");
#endif
  case config::OutputEngineKind::Hdf5:
#if defined(TPIC_HAS_HDF5) && TPIC_HAS_HDF5
    return make_hdf5_engine(channel, selection, domain, precision);
#else
    throw std::invalid_argument("TPIC output: HDF5 output requires a build with TPIC_ENABLE_HDF5=ON");
#endif
  }
  throw std::invalid_argument("TPIC output: unknown output engine");
}

// One worker owns the complete host snapshot and serializes the enabled
// channels in order. Channel views borrow its storage, so selecting different
// backends never allocates another full field or particle image. The two
// written flags make publish cover exactly the channels of the last piece.
class RoutedOutputEngine final : public OutputEngine {
public:
  RoutedOutputEngine(const config::PreparedOutputSelection& selection,
                     const config::PreparedDomain& domain, config::PrecisionMetadata precision,
                     lpfx::distributed::MpiFileWriter& field_files) {
    const auto& output = selection.output();
    if (output.fields_enabled())
      fields_ = make_channel_engine(data::OutputChannel::fields, output.field_engine,
                                    selection, domain, precision, field_files);
    if (output.particles_enabled())
      particles_ = make_channel_engine(data::OutputChannel::particles, output.particle_engine,
                                       selection, domain, precision, field_files);
  }

  void write_piece(data::OutputSnapshotView snapshot) override {
    fields_written_ = particles_written_ = false;
    if (snapshot.fields_present) {
      fields_->write_piece(snapshot.select(data::OutputChannel::fields));
      fields_written_ = true;
    }
    if (snapshot.particles_present) {
      particles_->write_piece(snapshot.select(data::OutputChannel::particles));
      particles_written_ = true;
    }
  }

  void publish() override {
    if (fields_written_) fields_->publish();
    if (particles_written_) particles_->publish();
    fields_written_ = particles_written_ = false;
  }

  const std::string& manifest_path() const noexcept override {
    return fields_ ? fields_->manifest_path() : empty_;
  }
  const std::string& particles_path() const noexcept override {
    return particles_ ? particles_->manifest_path() : empty_;
  }

private:
  std::unique_ptr<OutputEngine> fields_, particles_;
  std::string empty_;
  bool fields_written_ = false, particles_written_ = false;
};

} // namespace

std::unique_ptr<OutputEngine> make_output_engine(
    const config::PreparedOutputSelection& selection, const config::PreparedDomain& domain,
    config::PrecisionMetadata precision, lpfx::distributed::MpiFileWriter& field_files) {
  return std::make_unique<RoutedOutputEngine>(selection, domain, precision, field_files);
}

} // namespace tpic::io
