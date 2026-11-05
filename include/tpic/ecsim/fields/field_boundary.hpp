#ifndef TPIC_ECSIM_FIELDS_FIELD_BOUNDARY_HPP
#define TPIC_ECSIM_FIELDS_FIELD_BOUNDARY_HPP

#include <lpfx/core/error.hpp>
#include <lpfx/core/error_reporting.hpp>
#include <tpic/config/physics.hpp>
#include <tpic/ecsim/fields/field_checks.hpp>
#include <tpic/ecsim/fields/wall_faces.hpp>

#include <cstdint>
#include <limits>

namespace tpic::ecsim {

// The wall plan of one component location and ghost depth.
struct FieldBoundaryPlan {
  WallFaces faces;
  int extent[3]{};
  int face_count[3]{};
  int multiplier[3][6]{};
  unsigned constrained[6]{};
};

// The quantity a wall plan closes: the E-like node vectors, the B-like cell
// vectors or the nodal charge density, one component with the signs and
// constraint of a tangential E (div E is odd at a PEC wall, even at a PMC one).
enum class WallQuantity { electric, magnetic, charge };

// The per-call owner check of a prepared plan: the owner is initialized, has
// the plan's cells, has its components at Location and at least halo ghost
// layers on every side.
template <class Location, class Owner>
[[nodiscard]] lpfx::ErrorCode check_field_boundary_owner(
    const Owner& field, const FieldBoundaryPlan& plan, int halo) noexcept {
  static_assert(FieldAt<Owner, Location>);
  if (!field.initialized()) return lpfx::ErrorCode::invalid_state;
  if (!field_cells_equal(field, plan.faces.cells[0], plan.faces.cells[1], plan.faces.cells[2]))
    return lpfx::ErrorCode::invalid_argument;
  return has_halo(field, halo) ? lpfx::ErrorCode::success : lpfx::ErrorCode::invalid_argument;
}

namespace detail {

// The owner-independent part of a wall plan: a pure function of the domain,
// the physics, the closed quantity and the ghost depth.
template <WallQuantity Quantity>
[[nodiscard]] lpfx::ErrorCode try_prepare_field_boundary(
    const config::PreparedDomain& domain, const config::PreparedPhysics& physics, int halo,
    FieldBoundaryPlan& plan) noexcept {
  plan = {};
  const auto checked = try_prepare_wall_faces(domain, plan.faces);
  if (checked != lpfx::ErrorCode::success) return checked;
  constexpr bool node = Quantity != WallQuantity::magnetic;
  constexpr int components = Quantity == WallQuantity::charge ? 1 : 3;
  constexpr auto limit = std::uint64_t(std::numeric_limits<int>::max());
  for (int axis = 0; axis < 3; ++axis) {
    const auto extent = std::uint64_t(plan.faces.cells[axis]) + node;
    if (extent + 2 * halo > limit) return lpfx::ErrorCode::overflow;
    plan.extent[axis] = int(extent);
    for (int side = 0; side < 2; ++side) {
      if (!(side ? plan.faces.upper[axis] : plan.faces.lower[axis])) continue;
      // PreparedPhysics uses right, left for each axis, unlike this side loop.
      const int face = 2 * axis + (side == 0);
      const auto& rule = config::field_boundary_rule(physics.field_boundary[face]);
      if (!rule.closed) return lpfx::ErrorCode::invalid_argument;
      for (int component = 0; component < components; ++component) {
        const bool normal = Quantity != WallQuantity::charge && component == axis;
        const auto ghost_parity =
            node ? (normal ? rule.electric_normal : rule.electric_tangential)
                 : (normal ? rule.magnetic_normal : rule.magnetic_tangential);
        plan.multiplier[component][face] = ghost_parity == config::GhostParity::odd ? -1 : 1;
        if (rule.constrained ==
            (normal ? config::WallComponents::normal : config::WallComponents::tangential))
          plan.constrained[face] |= 1U << component;
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    const int first = axis == 0 ? 1 : 0;
    const int second = axis == 2 ? 1 : 2;
    const auto a = std::uint64_t(plan.extent[first]) + 2 * halo;
    const auto b = std::uint64_t(plan.extent[second]) + 2 * halo;
    if (a > limit / b) return lpfx::ErrorCode::overflow;
    plan.face_count[axis] = int(a * b);
  }
  return lpfx::ErrorCode::success;
}

} // namespace detail

// The three wall plans of one local domain, built once by MaxwellOperator and,
// with charge node plans, the Gauss diagnostic; kernels receive the selected
// plan by value.
struct FieldBoundaryPlans {
  FieldBoundaryPlan node_ghosts;     // Node ghosts, depth 1.
  FieldBoundaryPlan cell_ghosts;     // Cell ghosts, depth 1.
  FieldBoundaryPlan constrained_rows; // The constrained node rows.
};

template <WallQuantity NodeQuantity = WallQuantity::electric>
[[nodiscard]] FieldBoundaryPlans prepare_field_boundary_plans(
    const config::PreparedDomain& domain, const config::PreparedPhysics& physics) {
  FieldBoundaryPlans plans;
  constexpr const char* failed = "TPIC fields: wall plan preparation failed";
  lpfx::check_error(detail::try_prepare_field_boundary<NodeQuantity>(
      domain, physics, 1, plans.node_ghosts), failed);
  lpfx::check_error(detail::try_prepare_field_boundary<WallQuantity::magnetic>(
      domain, physics, 1, plans.cell_ghosts), failed);
  lpfx::check_error(detail::try_prepare_field_boundary<NodeQuantity>(
      domain, physics, 0, plans.constrained_rows), failed);
  return plans;
}

} // namespace tpic::ecsim

#endif // TPIC_ECSIM_FIELDS_FIELD_BOUNDARY_HPP
