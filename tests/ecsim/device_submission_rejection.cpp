// These are actual public device API calls, but reject on host arguments (an
// unbound executor, an empty cell map) before touching a native stream, and
// clear a reused diagnostic. They can run on a CUDA/HIP build without a GPU.
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/ecsim/deposition/current_deposition_device.hpp>

#include <iostream>
#include <stdexcept>

namespace {
template <class Precision> void reused_diagnostic_rejected_geometry() {
  using Backend = lpfx::device::ActiveBackend;
  tpic::data::MagneticNodeOwner<Precision, Backend> magnetic;
  tpic::data::CurrentOwner<Precision, Backend> current;
  lpfx::ResidentParticleRead<typename tpic::data::ParticleSchema<Precision>::type, Backend> resident;
  const lpfx::device::Executor<Backend> executor{lpfx::device::BackendTraits<Backend>::null_stream};
  const tpic::data::ParticleCellMap geometry{};
  const auto reject = [&](const auto& input) {
    lpfx::device::DeviceSubmissionDiagnostic diagnostic{
        lpfx::device::DeviceSubmissionStage::completion, 700, "previous-error", "previous-message"};
    const auto result = tpic::ecsim::enqueue_resident_current<Precision>(
        resident, magnetic, current, input, 1., 0.5, nullptr, executor, &diagnostic);
    if (result != lpfx::ErrorCode::invalid_argument || diagnostic.has_native_error() ||
        diagnostic.native_code != 0 || diagnostic.native_name != nullptr ||
        diagnostic.native_message != nullptr)
      throw std::runtime_error("resident current early rejection retained a previous diagnostic");
  };
  reject(geometry);
}
} // namespace

int main() {
  try {
    reused_diagnostic_rejected_geometry<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>();
    reused_diagnostic_rejected_geometry<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>();
    reused_diagnostic_rejected_geometry<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "TPIC reused submission diagnostic rejection tests passed\n";
}
