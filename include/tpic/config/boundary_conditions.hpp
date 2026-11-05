#ifndef TPIC_CONFIG_BOUNDARY_CONDITIONS_HPP
#define TPIC_CONFIG_BOUNDARY_CONDITIONS_HPP

// The one place that defines the boundary conditions of fields and particles.
// config::PreparedPhysics holds the kind of every face; the rule tables below
// say what a kind means, and no other file names a rule.
//
// Fields. A face is periodic when its PERIODIC key is set; otherwise the deck
// code bcEMface selects a wall kind. A perfect electric conductor holds the
// tangential E at zero. A perfect magnetic conductor, its dual and a
// mirror-symmetry plane of the fields, holds the tangential B and the normal E
// at zero. Both are lossless: with these ghost signs the node curl of B and
// the cell curl of E stay adjoint on the wall, which the energy conservation
// of the scheme needs.
// A closed wall is described completely by its row: the two wall kernels
// (ecsim/fields/field_boundary_device.hpp) apply its ghost signs and zero its
// electric rows, and no solver code names the kind. The moment closure and the
// dual-volume scaling are the same for every wall. A new closed wall is one
// enumerator before periodic, one row, the code list below and the deck
// documentation; its ghost signs must keep the two curls adjoint.
// A wall that is not closed (a prescribed value, an outgoing wave) also needs
// solver code in ecsim::MaxwellOperator: its rows of the operator image
// (combine_image), its right-hand-side term (complete_rhs) and its magnetic
// ghosts (close_magnetic_ghosts and the next-cell closure).
//
// Particles. A face is periodic when its PERIODIC_P key is set; otherwise the
// deck code bcPface must be particle_mirror_code. The response of a wall is
// device code: the wall arm of communication::classify_particle_point, inlined
// in the mover kernels. Another kind needs its response there, a kind per face
// in communication::ParticleMigrationGeometry (one periodic flag per axis) and,
// when it removes particles, a discard destination in
// communication::ParticleDestinationFlagDecoder and a population check in
// simulation::LocalSpeciesActions that allows it.

#include <array>
#include <cstddef>
#include <string_view>

namespace tpic::config {

// The wall kinds precede periodic; their order is the deck code bcEMface.
enum class FieldBoundary { perfect_electric_conductor, perfect_magnetic_conductor, open, periodic };
enum class ParticleBoundary { mirror, periodic };
// The sign of a ghost value relative to its interior image.
enum class GhostParity { odd, even };
// Components of a vector on a wall, relative to the wall normal.
enum class WallComponents { none, tangential, normal };

struct FieldBoundaryRule {
  // The word of the startup report.
  std::string_view name;
  // Ghosts are plus or minus the interior image and rows are zeroed: the two
  // wall kernels suffice.
  bool closed;
  // Node vectors closed by the wall kernel are E-like (E, theta, solver
  // vectors, the smoothed current), cell vectors B-like (B, curl E).
  GhostParity electric_tangential, electric_normal, magnetic_tangential, magnetic_normal;
  // Electric rows held at zero on the wall nodes: in every solver vector, after
  // every smoothing pass, in the right-hand side and in the committed field.
  // A component with an odd ghost sign must be constrained: smoothing is then
  // self-adjoint in the dual-volume inner product.
  WallComponents constrained;
  // When not empty: --initialize and --run reject the kind with this text.
  std::string_view run_rejection;
};

// One row per FieldBoundary, in enumerator order.
inline constexpr auto field_boundary_rules = std::to_array<FieldBoundaryRule>({
    {"PEC", true, GhostParity::odd, GhostParity::even, GhostParity::even, GhostParity::odd,
     WallComponents::tangential, ""},
    {"PMC", true, GhostParity::even, GhostParity::odd, GhostParity::odd, GhostParity::even,
     WallComponents::normal, ""},
    {"open", false, GhostParity::even, GhostParity::even, GhostParity::even, GhostParity::even,
     WallComponents::none, "TPIC fields: open electromagnetic boundaries are not implemented"},
    {"periodic", false, GhostParity::even, GhostParity::even, GhostParity::even, GhostParity::even,
     WallComponents::none, ""},
});
static_assert(field_boundary_rules.size() == static_cast<std::size_t>(FieldBoundary::periodic) + 1);
// The deck codes of the wall kinds, for the error of an unknown code.
inline constexpr std::string_view field_boundary_codes =
    "0 perfect electric conductor, 1 perfect magnetic conductor or 2 open";

[[nodiscard]] constexpr const FieldBoundaryRule& field_boundary_rule(FieldBoundary kind) noexcept {
  return field_boundary_rules[static_cast<std::size_t>(kind)];
}

struct ParticleBoundaryRule {
  // The word of the startup report.
  std::string_view name;
};

// One row per ParticleBoundary, in enumerator order.
inline constexpr auto particle_boundary_rules =
    std::to_array<ParticleBoundaryRule>({{"mirror"}, {"periodic"}});
static_assert(particle_boundary_rules.size() ==
              static_cast<std::size_t>(ParticleBoundary::periodic) + 1);
// The deck code bcPface of the mirror, the one wall kind admitted.
inline constexpr int particle_mirror_code = 1;

[[nodiscard]] constexpr const ParticleBoundaryRule& particle_boundary_rule(
    ParticleBoundary kind) noexcept {
  return particle_boundary_rules[static_cast<std::size_t>(kind)];
}

} // namespace tpic::config

#endif // TPIC_CONFIG_BOUNDARY_CONDITIONS_HPP
