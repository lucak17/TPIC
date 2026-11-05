#ifndef TPIC_DIAGNOSTICS_OUTPUT_MOMENT_DEPOSITION_DEVICE_HPP
#define TPIC_DIAGNOSTICS_OUTPUT_MOMENT_DEPOSITION_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/output/output_moment_schema.hpp>
#include <tpic/data/particles/particle_fractions.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/particles/cic.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace tpic::diagnostics {
namespace detail {

// Adds the first Count selected amounts, each times weight, at node. The stored
// scalar is the compute type, so a finite amount times a CIC weight in (0, 1]
// is finite: the kernel's amount check covers every stored product.
template <std::size_t Count, class View, class Compute>
__device__ LPFX_ALWAYS_INLINE void add_output_moment_components(View moments, data::GridIndex node,
    const Compute (&amount)[data::output_moment_component_count], Compute weight,
    std::uint16_t selected) {
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    static_assert((std::is_same_v<std::remove_cvref_t<decltype(moments.template component<I>()(0, 0, 0))>,
                                  Compute> && ...),
                  "output moments accumulate in the deposition compute type");
    (((selected & (1U << I)) != 0
          ? (atomicAdd(&moments.template component<I>()(node.i, node.j, node.k), amount[I] * weight),
             void()) : void()), ...);
  }(std::make_index_sequence<Count>{});
}

} // namespace detail

namespace kernels {

template <class Precision, bool Pressure, class ParticleView, class MomentView>
__global__ __launch_bounds__(lpfx::device::default_block_threads) void deposit_output_moments_kernel(
    ParticleView particles, MomentView moments, data::ParticleCellMap cells,
    typename Precision::Compute::Reduction inverse_volume,
    typename Precision::Compute::Reduction inverse_qom, std::uint16_t selected, unsigned* invalid) {
  using Particles = data::ParticleSchema<Precision>;
  using Compute = typename Precision::Compute::Reduction;
  using Component = data::OutputMomentComponent;
  using Bit = data::OutputMomentBit;
  constexpr std::size_t deposited_components = MomentView::component_count == 1 ? 1 : Pressure ?
      data::output_moment_component_count : data::output_pressure_first_component;
  lpfx::device::for_each_grid_index(particles.size(), [&](const unsigned index) {
    const auto key = particles.template get<typename Particles::Cell>()[index];
    if (key < 0 || key >= cells.count) {
      atomicExch(invalid, 1U);
      return;
    }
    const auto cell = cells.decode(key);
    if (!cells.contains_interior(cell.i, cell.j, cell.k)) {
      atomicExch(invalid, 1U);
      return;
    }
    const auto fraction = data::load_fractions<Precision>(particles, index);
    const Compute rx = config::convert<Compute>(fraction.x);
    const Compute ry = config::convert<Compute>(fraction.y);
    const Compute rz = config::convert<Compute>(fraction.z);
    const Compute q = config::convert<Compute>(particles.template get<typename Particles::Charge>()[index]);
    if (!data::in_unit_cell(fraction.x, fraction.y, fraction.z)) {
      atomicExch(invalid, 1U);
      return;
    }
    // Every selected amount carries the charge, and a velocity is loaded only
    // when a selected amount uses it, so a nonfinite q, u, v or w makes some
    // amount nonfinite: the single amount check below rejects it.
    Compute u{}, v{}, w{};
    if (selected & Bit::uses_u)
      u = config::convert<Compute>(particles.template get<typename Particles::U>()[index]);
    if (selected & Bit::uses_v)
      v = config::convert<Compute>(particles.template get<typename Particles::V>()[index]);
    if (selected & Bit::uses_w)
      w = config::convert<Compute>(particles.template get<typename Particles::W>()[index]);
    const Compute charge = q * inverse_volume;
    Compute amount[data::output_moment_component_count]{};
    if (selected & Bit::rho) amount[Component::rho] = charge;
    if (selected & Bit::jx) amount[Component::jx] = charge*u;
    if (selected & Bit::jy) amount[Component::jy] = charge*v;
    if (selected & Bit::jz) amount[Component::jz] = charge*w;
    if constexpr (Pressure) {
      const Compute mass = charge * inverse_qom;
      if (selected & Bit::pxx) amount[Component::pxx] = mass*u*u;
      if (selected & Bit::pxy) amount[Component::pxy] = mass*u*v;
      if (selected & Bit::pxz) amount[Component::pxz] = mass*u*w;
      if (selected & Bit::pyy) amount[Component::pyy] = mass*v*v;
      if (selected & Bit::pyz) amount[Component::pyz] = mass*v*w;
      if (selected & Bit::pzz) amount[Component::pzz] = mass*w*w;
    }
    bool finite = true;
    #pragma unroll
    for (std::size_t component = 0; component < deposited_components; ++component)
      finite = finite && ::isfinite(amount[component]);
    if (!finite) {
      atomicExch(invalid, 1U);
      return;
    }
    const auto weights = ecsim::make_cic_weights<Compute>(rx, ry, rz);
    #pragma unroll
    for (int corner = 0; corner < ecsim::cic_corner_count; ++corner) {
      const Compute weight = weights.value[corner];
      if (weight != Compute(0))
        detail::add_output_moment_components<deposited_components>(moments,
            ecsim::cic_corner(cell, corner), amount, weight, selected);
    }
  });
}

} // namespace kernels

// Output-only rho = sum(q W)/V (the snapshot writes 4*pi*rho), physical
// J = sum(q v W)/V, and raw P_ij = sum((q/qom) v_i v_j W)/V. Each particle is
// weighted by its mass q/qom at deposition, not each species at write time, so
// summing species yields the physical total P. A one-component owner receives
// rho alone.
//
// The entire authoritative owner is traversed, so unsorted, full-sort and
// partial-sort modes share this path. Migration and sorting must be complete.
// Caller clears this reusable owner and its persistent unsigned error flag on
// executor, checks the flag before shared-node addition, then applies physical
// dual-volume correction once. Only selected moments are evaluated/deposited.
// cells is the particle cell map prepared once from the local domain. No
// allocation or particle download occurs.
template <class Precision, class ParticleOwner, class MomentOwner, class Backend>
[[nodiscard]] lpfx::ErrorCode enqueue_deposit_output_moments(
    const ParticleOwner& particles, MomentOwner& moments,
    const data::ParticleCellMap& cells, double inverse_cell_volume, double qom, std::uint16_t selected,
    unsigned* invalid, lpfx::device::Executor<Backend> executor) {
  static_assert(std::is_same_v<Backend, typename ParticleOwner::backend_type> &&
                std::is_same_v<Backend, typename MomentOwner::space_type::backend_type>);
  static_assert(MomentOwner::component_count == data::output_moment_component_count ||
                MomentOwner::component_count == 1);
  static_assert(ecsim::FieldAt<MomentOwner, lpfx::Node>);
  using Compute = typename Precision::Compute::Reduction;
  if (!executor.bound() || invalid == nullptr || (selected & ~data::all_output_moments) != 0 ||
      (MomentOwner::component_count == 1 && (selected & ~data::OutputMomentBit::rho) != 0))
    return lpfx::ErrorCode::invalid_argument;
  if (!particles.valid() || !moments.initialized()) return lpfx::ErrorCode::invalid_state;
  if (particles.size() > static_cast<unsigned>(std::numeric_limits<int>::max()))
    return lpfx::ErrorCode::overflow;
  if (cells.count <= 0 || moments.cells().x() != cells.nx ||
      moments.cells().y() != cells.ny || moments.cells().z() != cells.nz ||
      !std::isfinite(qom) || qom == 0 ||
      !std::isfinite(inverse_cell_volume) || inverse_cell_volume <= 0)
    return lpfx::ErrorCode::invalid_argument;
  const Compute inverse_volume = config::convert<Compute>(inverse_cell_volume);
  const Compute inverse_qom = Compute(1) / config::convert<Compute>(qom);
  if (!std::isfinite(inverse_volume) || inverse_volume <= 0 ||
      !std::isfinite(inverse_qom) || inverse_qom == 0)
    return lpfx::ErrorCode::overflow;
  const auto begun = lpfx::device::begin_device_submission<Backend>();
  if (begun != lpfx::ErrorCode::success) return begun;
  if (selected == 0) return lpfx::ErrorCode::success;
  using ParticleView = decltype(particles.cview());
  using MomentView = decltype(moments.view());
  if constexpr (MomentOwner::component_count != 1)
    if (selected >> data::output_pressure_first_component)
      return lpfx::device::launch<lpfx::device::KernelLaunch<>,
          kernels::deposit_output_moments_kernel<Precision, true, ParticleView, MomentView>,
          lpfx::device::EmptyWork::skip, lpfx::device::LaunchPreflight::admitted>(
          executor, particles.size(), particles.cview(), moments.view(), cells,
          inverse_volume, inverse_qom, selected, invalid);
  return lpfx::device::launch<lpfx::device::KernelLaunch<>,
      kernels::deposit_output_moments_kernel<Precision, false, ParticleView, MomentView>,
      lpfx::device::EmptyWork::skip, lpfx::device::LaunchPreflight::admitted>(
      executor, particles.size(), particles.cview(), moments.view(), cells,
      inverse_volume, inverse_qom, selected, invalid);
}

} // namespace tpic::diagnostics

#endif // TPIC_DIAGNOSTICS_OUTPUT_MOMENT_DEPOSITION_DEVICE_HPP
