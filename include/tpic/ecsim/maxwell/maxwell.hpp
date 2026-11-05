#ifndef TPIC_ECSIM_MAXWELL_MAXWELL_HPP
#define TPIC_ECSIM_MAXWELL_MAXWELL_HPP

#include <lpfx/data/field/component.hpp>
#include <tpic/config/domain.hpp>
#include <tpic/config/physics.hpp>
#include <tpic/ecsim/fields/field_boundary.hpp>
#include <tpic/ecsim/fields/stencil_region.hpp>
#include <tpic/ecsim/gmres/gmres.hpp>


namespace tpic::ecsim {

// Solver storage is independent of authoritative E/B/J storage: SolverStorage
// (ecsim/gmres/gmres.hpp). Shared with the memory estimate so field plans
// match the owners used by Maxwell.
template <class Precision> struct SolverFieldSchema {
  using Node = lpfx::field::schema<
      lpfx::field::component<"sx", SolverStorage<Precision>, lpfx::field::node>,
      lpfx::field::component<"sy", SolverStorage<Precision>, lpfx::field::node>,
      lpfx::field::component<"sz", SolverStorage<Precision>, lpfx::field::node>>;
  using Cell = lpfx::field::schema<
      lpfx::field::component<"cx", SolverStorage<Precision>, lpfx::field::cell>,
      lpfx::field::component<"cy", SolverStorage<Precision>, lpfx::field::cell>,
      lpfx::field::component<"cz", SolverStorage<Precision>, lpfx::field::cell>>;
};

// Construction-only numerical inputs; MaxwellOperator copies what it uses.
struct MaxwellParameters {
  const config::PreparedDomain& domain;
  const config::PreparedPhysics& physics;
  double c, dt, theta;
  int smooth_iterations;
  double smooth_alpha;
  config::EcsimSmoothingOperator smooth_operator;
};

// The run-constant plans of the field operators: a pure function of the
// domain and the physics. MaxwellOperator builds them once at construction;
// enqueue functions take the member they need and kernels receive it by value.
struct FieldOperatorPlans {
  FieldBoundaryPlans boundary;
  StencilRegionGeometry nodes; // Interior nodes: cells + 1 per axis.
  StencilRegionGeometry cells; // Interior cells.
};

[[nodiscard]] inline FieldOperatorPlans prepare_field_operator_plans(
    const config::PreparedDomain& domain, const config::PreparedPhysics& physics) {
  FieldOperatorPlans plans{prepare_field_boundary_plans(domain, physics), {}, {}};
  constexpr const char* failed = "TPIC fields: operator plan preparation failed";
  const auto* cells = domain.local_cells.data();
  lpfx::check_error(try_make_stencil_region_geometry(
      cells[0] + 1, cells[1] + 1, cells[2] + 1, plans.nodes), failed);
  lpfx::check_error(try_make_stencil_region_geometry(
      cells[0], cells[1], cells[2], plans.cells), failed);
  return plans;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_MAXWELL_MAXWELL_HPP
