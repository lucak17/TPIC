#ifndef TPIC_DIAGNOSTICS_GAUSS_RESIDUAL_DEVICE_HPP
#define TPIC_DIAGNOSTICS_GAUSS_RESIDUAL_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <lpfx/distributed/collective_operand.hpp>
#include <tpic/communication/fields/field_exchange.hpp>
#include <tpic/config/boundary_conditions.hpp>
#include <tpic/config/physics.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/data/output/gauss_sample.hpp>
#include <tpic/data/particles/particle_cell.hpp>
#include <tpic/diagnostics/output_moment_deposition_device.hpp>
#include <tpic/diagnostics/scalar_diagnostics_device.hpp>
#include <tpic/ecsim/deposition/moment_field_device.hpp>
#include <tpic/ecsim/fields/field_boundary_device.hpp>
#include <tpic/ecsim/fields/node_cell_curl.hpp>
#include <tpic/ecsim/fields/node_smoothing_device.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <utility>

namespace tpic::diagnostics {

namespace kernels {

// Per local cell h = divC(E)/2. Finishing stores r = stash + h with block
// partials of r^2, (r - first)^2 and r^2 on wall cells; beginning: stash = h.
template <bool Begin, bool Finish, class Reduction, class ElectricView, class CellView>
__global__ __launch_bounds__(detail::ScalarLaunch::threads) void gauss_field_partials_kernel(
    ElectricView electric, CellView stash, CellView first, CellView residual, Reduction* contributions,
    ecsim::NodeCellSpacing<Reduction> spacing, ecsim::WallFaces faces, bool set_first) {
  using Index = data::GaussScalarIndex;
  const auto s = stash.template component<0>();
  const auto f = first.template component<0>();
  const auto out = residual.template component<0>();
  const unsigned nx = faces.cells[0], ny = faces.cells[1];
  Reduction value[data::gauss_field_scalar_count]{};
  lpfx::device::for_each_grid_index(nx * ny * unsigned(faces.cells[2]), [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, nx, ny);
    const int i = at.i, j = at.j, k = at.k;
    const Reduction half =
        Reduction(0.5) * ecsim::node_cell_divergence<Reduction>(electric, i, j, k, spacing);
    if constexpr (Finish) {
      const Reduction r = s(i, j, k) + half;
      out(i, j, k) = r;
      value[Index::residual] += r * r;
      if (set_first) {
        f(i, j, k) = r;
      } else {
        const Reduction drift = r - f(i, j, k);
        value[Index::drift] += drift * drift;
      }
      if (faces.boundary_count(i, j, k) + faces.boundary_count(i + 1, j + 1, k + 1))
        value[Index::wall] += r * r;
    }
    if constexpr (Begin) s(i, j, k) = half;
  });
  if constexpr (Finish) detail::write_scalar_block_partials(value, contributions);
}

// Per local cell c = 4*pi times the mean of the eight corner densities: the
// stash becomes h - c, with block partials of c^2.
template <class Reduction, class NodeView, class CellView>
__global__ __launch_bounds__(detail::ScalarLaunch::threads) void gauss_charge_partials_kernel(
    NodeView charge, CellView stash, Reduction* contributions, ecsim::WallFaces faces) {
  const auto q = charge.template component<0>();
  const auto s = stash.template component<0>();
  const unsigned nx = faces.cells[0], ny = faces.cells[1];
  Reduction value[1]{};
  lpfx::device::for_each_grid_index(nx * ny * unsigned(faces.cells[2]), [&](const unsigned index) {
    const auto at = device::decode_x_fastest(index, nx, ny);
    const int i = at.i, j = at.j, k = at.k;
    const Reduction cell = std::numbers::pi_v<Reduction> / 2 *
        (q(i, j, k) + q(i + 1, j, k) + q(i, j + 1, k) + q(i + 1, j + 1, k) +
         q(i, j, k + 1) + q(i + 1, j, k + 1) + q(i, j + 1, k + 1) + q(i + 1, j + 1, k + 1));
    s(i, j, k) -= cell;
    value[0] += cell * cell;
  });
  detail::write_scalar_block_partials(value, contributions);
}

} // namespace kernels

// The Gauss-law residual of the committed states (docs/user/gauss_law.md).
// A sample begun at cycle n stores per local cell divC(E^n)/2 - c, where c is
// 4*pi times the cell mean of the charge density of the particles at x^{n+1/2}:
// dual-volume scaled, zero on PEC wall nodes (their image charge is opposite)
// and smoothed like the solver's current. Finishing at n + 1 adds divC(E^{n+1})/2,
// so the residual is centred at (n + 1/2)*dt for every theta; residual holds the
// per-cell r of the last finished sample. Everything runs in order on one
// executor. charge and smooth are registered even when disabled; the cell
// fields exist only when enabled.
template <class Precision, class Backend> class GaussResidual {
  using Reduction = typename Precision::Compute::Reduction;
  using Schema = data::GaussFieldSchema<Precision>;
  using Traits = lpfx::device::BackendTraits<Backend>;
  using Allocation = lpfx::device::DeviceAllocation<Backend>;
  using Pinned = lpfx::device::PinnedHostAllocation<Backend>;
  using Executor = lpfx::device::Executor<Backend>;
  using NodeOwner = data::DeviceFieldOwner<typename Schema::Node, Backend>;
  using CellOwner = data::GaussCellOwner<Precision, Backend>;
  using Index = data::GaussScalarIndex;

public:
  GaussResidual(const config::PreparedDomain& domain, const config::PreparedPhysics& physics,
                const config::SimulationParameters& parameters, bool enabled,
                const lpfx::device::DeviceProperties& properties, Executor executor)
      : charge(data::make_zero_field<typename Schema::Node>(domain, executor)),
        smooth(data::make_zero_field<typename Schema::Node>(domain, executor)),
        executor_(executor) {
    if (!enabled) return;
    residual = data::make_zero_field<typename Schema::Cell>(domain, executor);
    stash = data::make_zero_field<typename Schema::Cell>(domain, executor);
    first = data::make_zero_field<typename Schema::Cell>(domain, executor);
    lpfx::check_error(ecsim::try_make_node_cell_spacing(domain.spacing, spacing_),
                      "TPIC Gauss diagnostics: invalid domain");
    cells_ = data::prepare_particle_cell_map(domain);
    moment_plan_ = ecsim::prepare_moment_field_plan(domain);
    plans_ = ecsim::prepare_field_boundary_plans<ecsim::WallQuantity::charge>(domain, physics);
    inverse_cell_volume_ = domain.inverse_cell_volume;
    dt_ = parameters.dt;
    passes_ = physics.smoothing_enabled ? parameters.smooth_niter : 0;
    alpha_ = config::convert<Reduction>(parameters.smooth);
    operator_ = parameters.ecsim_smoothing_operator;
    ranks_ = domain.field_neighborhood.topology().input().communicator_size();
    blocks_ = detail::ScalarLaunch::blocks_at_least_one(domain.interior_cell_count,
        static_cast<unsigned>(std::min<std::uint64_t>(data::contribution_count,
            std::min<std::uint64_t>(4ULL * properties.multiprocessor_count, properties.maximum_grid_size_x))));
    // Global cells, and those touching a closed face (right, left per axis).
    double inner = 1;
    global_cells_ = 1;
    for (int axis = 0; axis < 3; ++axis) {
      int walls = 0;
      for (int face = 2 * axis; face < 2 * axis + 2; ++face)
        walls += config::field_boundary_rule(physics.field_boundary[face]).closed;
      global_cells_ *= double(domain.global_cells[axis]);
      inner *= double(std::max(domain.global_cells[axis] - walls, 0));
    }
    wall_cells_ = global_cells_ - inner;
    lpfx::device::check_device_error<Backend>(
        Allocation::create(data::gauss_scalar_count * sizeof(Reduction), values_),
        "TPIC Gauss diagnostics: device allocation failed");
    lpfx::device::check_device_error<Backend>(
        Allocation::create(data::gauss_scalar_count * data::contribution_count * sizeof(Reduction), scratch_),
        "TPIC Gauss diagnostics: device allocation failed");
    lpfx::device::check_device_error<Backend>(
        Pinned::create(data::gauss_scalar_count * sizeof(Reduction), host_values_),
        "TPIC Gauss diagnostics: pinned allocation failed");
  }

  GaussResidual(const GaussResidual&) = delete;
  GaussResidual& operator=(const GaussResidual&) = delete;

  ~GaussResidual() noexcept {
    // Work accepted on the executor must retire before its fields are released.
    if (work_pending_ && !Traits::succeeded(Traits::synchronize_stream(executor_.stream())))
      std::abort();
  }

  [[nodiscard]] bool enabled() const noexcept { return residual.initialized(); }
  [[nodiscard]] auto collective_operand() {
    return lpfx::distributed::device_collective_in_place<Backend>(
        values(), enabled() ? data::gauss_scalar_count : 0);
  }

  // Enqueues the field kernel on the committed E: finishing reduces the field
  // slots, beginning stores divC(E)/2 for the sample's charge to complete and
  // zeroes the density before the sample's depositions.
  template <class ElectricOwner> void enqueue_field(const ElectricOwner& electric, bool begin, bool finish) {
    const auto& cells = moment_plan_.faces.cells;
    lpfx::require(enabled() && (begin || finish) && electric.initialized() &&
                      ecsim::field_cells_equal(electric, cells[0], cells[1], cells[2]),
                  "TPIC Gauss diagnostics: invalid field kernel request");
    const bool set_first = finish && !first_set_;
    const auto launch = [&]<bool Begin, bool Finish>() {
      lpfx::device::launch_or_throw<kernels::gauss_field_partials_kernel<
          Begin, Finish, Reduction, decltype(electric.view()), decltype(stash.view())>>(
          "TPIC Gauss diagnostics: field kernel failed", executor_,
          lpfx::device::LaunchGrid{blocks_, detail::ScalarLaunch::threads}, electric.view(), stash.view(),
          first.view(), residual.view(), scratch(), spacing_, moment_plan_.faces, set_first);
    };
    work_pending_ = true;
    if (begin && finish) launch.template operator()<true, true>();
    else if (begin) launch.template operator()<true, false>();
    else launch.template operator()<false, true>();
    if (begin)
      lpfx::check_error(ecsim::enqueue_clear_moment_field(charge, executor_),
                        "TPIC Gauss diagnostics: charge clear failed");
    if (!finish) return;
    enqueue_reduction(Index::residual, unsigned(data::gauss_field_scalar_count));
    first_set_ = true;
  }

  // Adds the density of one species of the committed state; *invalid flags a
  // malformed record and must be checked before complete_charge.
  template <class ParticleOwner> void enqueue_deposit(const ParticleOwner& particles, double qom, unsigned* invalid) {
    lpfx::check_error(enqueue_deposit_output_moments<Precision>(
                          particles, charge, cells_, inverse_cell_volume_, qom,
                          std::uint16_t(data::OutputMomentBit::rho), invalid, executor_),
                      "TPIC Gauss diagnostics: charge deposition failed");
  }

  // Completes the begun sample: shared-node sum, dual-volume scale, wall image
  // values, the solver's smoothing, then the stash subtracts the cell charge and
  // the charge slot holds its squares.
  template <class ChargeSum, class ChargeGhosts, class SmoothGhosts>
  void complete_charge(ChargeSum& sum, ChargeGhosts& charge_ghosts, SmoothGhosts& smooth_ghosts) {
    constexpr const char* rejected = "TPIC Gauss diagnostics: charge exchange rejected";
    constexpr const char* failed = "TPIC Gauss diagnostics: charge exchange failed";
    lpfx::require(enabled(), "TPIC Gauss diagnostics: disabled");
    work_pending_ = true;
    communication::complete_field_exchange(sum, rejected, failed);
    lpfx::check_error(ecsim::enqueue_scale_moment_dual_volume(charge, moment_plan_, executor_),
                      "TPIC Gauss diagnostics: dual-volume scale failed");
    NodeOwner* current = &charge;
    NodeOwner* next = &smooth;
    for (int pass = 0; pass < passes_; ++pass) {
      project(*current);
      if (current == &charge) communication::complete_field_exchange(charge_ghosts, rejected, failed);
      else communication::complete_field_exchange(smooth_ghosts, rejected, failed);
      lpfx::check_error(ecsim::enqueue_wall_ghosts(*current, plans_, executor_),
                        "TPIC Gauss diagnostics: charge ghost closure failed");
      project(*current);
      lpfx::check_error(ecsim::enqueue_node_smoothing<Precision, Reduction>(
                            *current, *next, operator_, alpha_, nullptr, executor_),
                        "TPIC Gauss diagnostics: charge smoothing failed");
      std::swap(current, next);
    }
    project(*current);
    lpfx::device::launch_or_throw<kernels::gauss_charge_partials_kernel<
        Reduction, decltype(current->view()), decltype(stash.view())>>(
        "TPIC Gauss diagnostics: charge kernel failed", executor_,
        lpfx::device::LaunchGrid{blocks_, detail::ScalarLaunch::threads}, current->view(), stash.view(),
        scratch() + Index::charge * data::contribution_count, moment_plan_.faces);
    enqueue_reduction(Index::charge, 1);
  }

  // The global sums of the sample begun at cycle, as root mean squares. The
  // in-place sum of one rank is the identity; the download is the host join.
  template <class Collective> [[nodiscard]] data::GaussSample host_blocking_finish(Collective& collective, int cycle) {
    lpfx::require(enabled() && cycle >= 0, "TPIC Gauss diagnostics: invalid sample");
    if (ranks_ > 1) {
      lpfx::check_acceptance(collective.begin(), "TPIC Gauss diagnostics: global sum rejected");
      lpfx::check_error(collective.wait(), "TPIC Gauss diagnostics: global sum failed");
      lpfx::check_error(collective.conclude(), "TPIC Gauss diagnostics: global sum failed");
    }
    lpfx::device::check_device_error<Backend>(
        Traits::copy_device_to_host_async(host_values_.raw_base(), values(),
                                          data::gauss_scalar_count * sizeof(Reduction), executor_.stream()),
        "TPIC Gauss diagnostics: download failed");
    lpfx::device::check_device_error<Backend>(Traits::synchronize_stream(executor_.stream()),
                                              "TPIC Gauss diagnostics: download completion failed");
    work_pending_ = false;
    data::GaussSample sample{cycle, (cycle + 0.5) * dt_};
    for (std::size_t slot = 0; slot < data::gauss_scalar_count; ++slot) {
      Reduction sum;
      std::memcpy(&sum, host_values_.raw_base() + slot * sizeof(Reduction), sizeof(sum));
      const double cells = slot == Index::wall ? wall_cells_ : global_cells_;
      sample.rms[slot] = cells > 0 ? std::sqrt(double(sum) / cells) : 0.0;
      lpfx::require(std::isfinite(sample.rms[slot]), "TPIC Gauss diagnostics: nonfinite global sum");
    }
    return sample;
  }

  NodeOwner charge, smooth;
  CellOwner residual;

private:
  Reduction* values() { return reinterpret_cast<Reduction*>(values_.raw_base()); }
  Reduction* scratch() { return reinterpret_cast<Reduction*>(scratch_.raw_base()); }
  void project(NodeOwner& field) {
    lpfx::check_error(ecsim::enqueue_constrained_rows(field, ecsim::detail::NoRowInput{},
                                                      plans_.constrained_rows, executor_),
                      "TPIC Gauss diagnostics: charge wall rows failed");
  }
  void enqueue_reduction(std::size_t slot, unsigned rows) {
    lpfx::device::launch_or_throw<device::kernels::reduce_block_partials_kernel<Reduction>>(
        "TPIC Gauss diagnostics: reduction failed", executor_,
        lpfx::device::LaunchGrid{rows, detail::ScalarLaunch::threads},
        scratch() + slot * data::contribution_count, values() + slot, unsigned(data::contribution_count),
        blocks_, rows);
  }

  CellOwner stash, first;
  Executor executor_;
  bool first_set_ = false, work_pending_ = false;
  data::ParticleCellMap cells_{};
  ecsim::MomentFieldPlan moment_plan_{};
  ecsim::FieldBoundaryPlans plans_{};
  ecsim::NodeCellSpacing<Reduction> spacing_{};
  config::EcsimSmoothingOperator operator_{};
  Reduction alpha_{};
  double inverse_cell_volume_ = 0, dt_ = 0, global_cells_ = 0, wall_cells_ = 0;
  int passes_ = 0;
  unsigned ranks_ = 1, blocks_ = 0;
  Allocation values_, scratch_;
  Pinned host_values_;
};

} // namespace tpic::diagnostics

#endif // TPIC_DIAGNOSTICS_GAUSS_RESIDUAL_DEVICE_HPP
