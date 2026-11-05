#ifndef TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_DEPOSITION_DEVICE_HPP
#define TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_DEPOSITION_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/api/field/field.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/launch.hpp>
#include <lpfx/memory/native_allocation.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/device/block_threads.hpp>
#include <tpic/config/numerics_options.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/data/mass_matrix/compact_mass_matrix_device.hpp>
#include <tpic/ecsim/deposition/deposition_admission_device.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_scalar_device.hpp>
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
#include <tpic/ecsim/mass_matrix/mass_matrix_mma_device.hpp>
#elif LPFX_CONFIG_DEVICE_BACKEND_HIP
#include <tpic/ecsim/mass_matrix/mass_matrix_mfma_device.hpp>
#endif
#include <tpic/ecsim/particles/sorted_cell_device.hpp>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tpic::ecsim {

namespace detail {

// Admission shared by the three mass entry points; it never reads the particle
// owner, whose mapping may be unpublished. Contract of every entry point: the
// coefficients are accumulated without the volume factor, with alpha and the
// macrocharge factor both from beta_over_c = (q/m)*dt/(2*c) (the input requires
// c = 1), in Compute::Mass. Stream: executor, on which migration has completed
// (not needed by the resident entry point), the nodal B replicas are coherent,
// and mass and *invalid have been cleared. A nonzero *invalid invalidates the
// whole result; overflow of an atomic sum is caught when the assembled
// coefficients are applied. The caller runs the mass-halo addition and copy,
// periodic self routes included, before the matrix is applied.
template <class Precision, class MagneticOwner, class Backend, class Diagnostic>
[[nodiscard]] lpfx::ErrorCode check_mass_deposition_inputs(
    const MagneticOwner& magnetic, const data::CompactMassMatrix<Precision, Backend>& mass,
    data::ParticleCellMap cells, double beta_over_c,
    unsigned* invalid, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic) {
  static_assert(std::is_same_v<Backend, typename MagneticOwner::space_type::backend_type>);
  static_assert(VectorFieldAt<MagneticOwner, lpfx::Node>);
  if (!executor.bound() || invalid == nullptr || cells.count <= 0)
    return lpfx::ErrorCode::invalid_argument;
  if (!magnetic.initialized() || !mass.initialized())
    return lpfx::ErrorCode::invalid_state;
  const auto layout = mass.layout();
  if (!field_cells_equal(magnetic, cells.nx, cells.ny, cells.nz) ||
      layout.nx - 1 != cells.nx || layout.ny - 1 != cells.ny || layout.nz - 1 != cells.nz ||
      !std::isfinite(beta_over_c)) return lpfx::ErrorCode::invalid_argument;
  using Compute = typename Precision::Compute::Mass;
  if (!std::isfinite(config::convert<Compute>(beta_over_c))) return lpfx::ErrorCode::overflow;
  return lpfx::device::begin_device_submission<Backend>(diagnostic);
}

template <class Precision, config::MassDepositionKernel Kernel,
          class MagneticOwner, class Backend, class Diagnostic>
[[nodiscard]] lpfx::ErrorCode launch_sorted_mass_matrix(
    ConstParticleReadView<Precision, Backend> particles, ParticleBins bins,
    const MagneticOwner& magnetic, data::CompactMassMatrix<Precision, Backend>& mass,
    data::ParticleCellMap cells, double beta_over_c,
    unsigned* invalid, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic) {
  using Compute = typename Precision::Compute::Mass;
  // Matrix launches own four cells/block on either backend. Scalar wave64
  // uses two; its grid stride covers remaining bins without a device query.
#if LPFX_CONFIG_DEVICE_BACKEND_HIP
  using Launch = std::conditional_t<Kernel == config::MassDepositionKernel::matrix_core,
      MassMfma4Launch, SortedCellLaunch>;
#else
  using Launch = SortedCellLaunch;
#endif
  static_assert(Kernel == config::MassDepositionKernel::scalar ||
                Kernel == config::MassDepositionKernel::matrix_core);
  using Particles = ConstParticleReadView<Precision, Backend>;
  using MagneticView = decltype(magnetic.view());
  constexpr auto kernel = [] {
    if constexpr (Kernel == config::MassDepositionKernel::matrix_core) {
#if LPFX_CONFIG_DEVICE_BACKEND_CUDA
      return kernels::sorted_mass_matrix_mma_kernel<Precision, Particles, MagneticView>;
#elif LPFX_CONFIG_DEVICE_BACKEND_HIP
      return kernels::sorted_mass_matrix_mfma4_kernel<Precision, Particles, MagneticView>;
#endif
    } else {
      return kernels::sorted_mass_matrix_scalar_kernel<Precision, Particles, MagneticView>;
    }
  }();
  return lpfx::device::launch<Launch, kernel, lpfx::device::EmptyWork::launch,
      lpfx::device::LaunchPreflight::admitted>(executor, diagnostic, bins.view.bin_count,
      particles, magnetic.view(), mass.data(), mass.layout(),
      cells, config::convert<Compute>(beta_over_c), bins, invalid);
}

} // namespace detail

// Zeroes the whole matrix, ghosts included. Stream: executor, once before the
// first species deposits; the caller clears its invalid flag separately.
template <class Precision, class Backend, class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_clear_mass_matrix(
    data::CompactMassMatrix<Precision, Backend>& mass,
    lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  if (!executor.bound()) return lpfx::ErrorCode::invalid_argument;
  if (!mass.initialized()) return lpfx::ErrorCode::invalid_state;
  using Traits = lpfx::device::BackendTraits<Backend>;
  const auto begun = lpfx::device::begin_device_submission<Backend>(diagnostic);
  if (begun != lpfx::ErrorCode::success) return begun;
  return lpfx::device::finish_device_submission<Backend>(
      Traits::memset_device_async(
          mass.data(), 0, mass.storage_bytes(), executor.stream()),
      executor, diagnostic);
}

// Adds the records [first, first + count). Contract at
// detail::check_mass_deposition_inputs; the caller captures *invalid next.
template <class Precision, class ParticleOwner, class MagneticOwner, class Backend, class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_unsorted_mass_matrix(
    const ParticleOwner& particles, const MagneticOwner& magnetic,
    data::CompactMassMatrix<Precision, Backend>& mass, data::ParticleCellMap cells,
    double beta_over_c, std::uint32_t first, std::uint32_t count,
    unsigned* invalid, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  static_assert(std::is_same_v<Backend, typename ParticleOwner::backend_type>);
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  const auto range = ecsim::check_completed_particle_range(particles, first, count);
  if (range != lpfx::ErrorCode::success) return range;
  const auto checked = detail::check_mass_deposition_inputs<Precision>(
      magnetic, mass, cells, beta_over_c, invalid, executor, diagnostic);
  if (checked != lpfx::ErrorCode::success) return checked;
  using Compute = typename Precision::Compute::Mass;
  const auto beta = config::convert<Compute>(beta_over_c);
  return lpfx::device::launch<lpfx::device::KernelLaunch<device::particle_block_threads>,
      kernels::unsorted_mass_matrix_kernel<
      Precision, decltype(particles.cview()), decltype(magnetic.view())>,
      lpfx::device::EmptyWork::skip, lpfx::device::LaunchPreflight::admitted>(
      executor, diagnostic, count, particles.cview(), magnetic.view(), mass.data(),
      mass.layout(), cells, beta, first, count, invalid);
}

// Adds only the sorted prefix of a completed receipt; the caller adds the tail
// with enqueue_unsorted_mass_matrix and keeps the receipt through the last bin
// reader, the mover included. The scalar kernel narrows each contribution to
// Mass; the matrix-core kernel keeps FP64 cell sums and narrows once at the
// cell flush. Contract at detail::check_mass_deposition_inputs.
template <class Precision, config::MassDepositionKernel Kernel = config::MassDepositionKernel::scalar,
          class ParticleOwner, class MagneticOwner, class Backend,
          class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_sorted_mass_matrix(
    const ParticleOwner& particles,
    const lpfx::DeviceSortSubmission<typename data::ParticleSchema<Precision>::type, Backend>& receipt,
    const MagneticOwner& magnetic, data::CompactMassMatrix<Precision, Backend>& mass,
    data::ParticleCellMap cells, double beta_over_c, unsigned* invalid,
    lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  static_assert(std::is_same_v<Backend, typename ParticleOwner::backend_type>);
  static_assert(std::is_same_v<decltype(receipt.resident_particle_view()),
                               ConstParticleReadView<Precision, Backend>>,
                "completed receipts and resident capabilities launch one read-view specialization");
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  const auto range = ecsim::check_completed_particle_range(particles, 0, 0);
  if (range != lpfx::ErrorCode::success) return range;
  const auto checked = detail::check_mass_deposition_inputs<Precision>(
      magnetic, mass, cells, beta_over_c, invalid, executor, diagnostic);
  if (checked != lpfx::ErrorCode::success) return checked;
  ecsim::ParticleBins bins;
  const auto admitted = ecsim::device_wait_particle_bins(
      particles, receipt, cells, executor, bins);
  if (admitted != lpfx::ErrorCode::success) return admitted;
  return detail::launch_sorted_mass_matrix<Precision, Kernel>(
      receipt.resident_particle_view(), bins, magnetic, mass,
      cells, beta_over_c, invalid, executor, diagnostic);
}

// Adds the retained prefix of a partial sort while routing proceeds. The
// caller registers its reader event with the sort owner before any append or
// allocation replacement. Contract at detail::check_mass_deposition_inputs.
template <class Precision, config::MassDepositionKernel Kernel = config::MassDepositionKernel::scalar,
          class MagneticOwner, class Backend,
          class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic>
[[nodiscard]] lpfx::ErrorCode enqueue_resident_mass_matrix(
    const lpfx::ResidentParticleRead<typename data::ParticleSchema<Precision>::type, Backend>& residents,
    const MagneticOwner& next_b, data::CompactMassMatrix<Precision, Backend>& mass,
    data::ParticleCellMap cells, double beta_over_c,
    unsigned* invalid, lpfx::device::Executor<Backend> executor,
    Diagnostic diagnostic = {}) {
  lpfx::device::reset_device_submission_diagnostic(diagnostic);
  const auto checked = detail::check_mass_deposition_inputs<Precision>(
      next_b, mass, cells, beta_over_c, invalid, executor, diagnostic);
  if (checked != lpfx::ErrorCode::success) return checked;
  ecsim::ParticleBins bins;
  const auto admitted = ecsim::device_wait_resident_particle_bins(
      residents, cells, executor, bins);
  if (admitted != lpfx::ErrorCode::success) return admitted;
  return detail::launch_sorted_mass_matrix<Precision, Kernel>(
      residents.resident_particle_view(), bins, next_b, mass,
      cells, beta_over_c, invalid, executor, diagnostic);
}

#if LPFX_CONFIG_DEVICE_BACKEND_CUDA || LPFX_CONFIG_DEVICE_BACKEND_HIP
template <class Precision>
[[nodiscard]] bool matrix_core_mass_deposition_available(int device) noexcept {
  using Backend = lpfx::device::ActiveBackend;
  using ParticleView = ConstParticleReadView<Precision, Backend>;
  using MagneticView =
      decltype(std::declval<const data::MagneticNodeOwner<Precision, Backend>&>().view());
  return ecsim::matrix_core_mass_kernel_available<Precision, ParticleView, MagneticView>(device);
}
#endif

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MASS_MATRIX_MASS_MATRIX_DEPOSITION_DEVICE_HPP
