#ifndef TPIC_COMMUNICATION_PARTICLES_PARTICLE_BOUNDARIES_HPP
#define TPIC_COMMUNICATION_PARTICLES_PARTICLE_BOUNDARIES_HPP

#include <lpfx/distributed/plan/particle_destination.hpp>
#include <tpic/communication/particles/particle_geometry.hpp>
#include <tpic/data/particles/particle_fractions.hpp>
#include <tpic/data/particles/particle_schema.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace tpic::communication {

using ParticleDestination = lpfx::distributed::ParticleDestination3D<std::uint32_t>;

// True for a finite floating-point value and for every integral value.
template <class Scalar>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool finite_or_integral(
    Scalar value) noexcept {
  if constexpr (std::is_floating_point_v<Scalar>) {
    constexpr Scalar maximum = std::numeric_limits<Scalar>::max();
    return value >= -maximum && value <= maximum;
  } else
    return true;
}

[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::int64_t positive_cell_modulo(
    std::int64_t cell, std::int64_t extent) noexcept {
  const auto remainder = cell % extent;
  return remainder < 0 ? remainder + extent : remainder;
}

[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool particle_cell_in_halo(
    const ParticleMigrationGeometry& geometry, int axis, std::int64_t cell) noexcept {
  return cell >= -config::PreparedDomain::particle_halo &&
         cell < std::int64_t(geometry.cells.extent(axis)) + config::PreparedDomain::particle_halo;
}

[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr std::uint32_t particle_axis_owner(
    const ParticleMigrationGeometry& geometry, int axis, std::uint32_t cell) noexcept {
  const auto offset = static_cast<std::uint32_t>(geometry.global_offset[axis]);
  if (cell >= offset && cell < offset + static_cast<std::uint32_t>(geometry.cells.extent(axis)))
    return geometry.owner_coordinate[axis];
  const auto split = static_cast<std::uint32_t>(geometry.enlarged_partition_end[axis]);
  const auto base = static_cast<std::uint32_t>(geometry.partition_base[axis]);
  return cell < split ? cell / (base + 1) :
      static_cast<std::uint32_t>(geometry.partition_remainder[axis]) +
          (cell - split) / base;
}

template <class Position>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr bool valid_canonical_particle_position(
    const ParticleMigrationGeometry& geometry, const Position& position) noexcept {
  return position.positive_endpoint_mask == 0 &&
      position.global_cell_x < static_cast<std::uint32_t>(geometry.global_cells[0]) &&
      position.global_cell_y < static_cast<std::uint32_t>(geometry.global_cells[1]) &&
      position.global_cell_z < static_cast<std::uint32_t>(geometry.global_cells[2]) &&
      position.fraction_x >= 0 && position.fraction_x < 1 &&
      position.fraction_y >= 0 && position.fraction_y < 1 &&
      position.fraction_z >= 0 && position.fraction_z < 1;
}

// Coordinate widths occupy the bits above coordinate_bits; Cartesian owner
// coordinates the bits below. A signed-int process count bounds the combined
// coordinate widths to at most 33 bits, including non-power-of-two extents.
struct ParticleDestinationFlagDecoder {
  static constexpr std::uint64_t invalid_flag = UINT64_MAX;
  // Owner coordinates occupy bits [0, 54), the x coordinate width bits
  // [54, 59) and the y coordinate width bits [59, 64).
  static constexpr int coordinate_bits = 54;
  static constexpr int x_width_shift = coordinate_bits;
  static constexpr int y_width_shift = 59;
  static constexpr int width_mask = 31;
  static constexpr int maximum_coordinate_bits = 33;
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr bool decode(
      std::uint64_t flag, ParticleDestination& destination) noexcept {
    const unsigned x_bits = static_cast<unsigned>((flag >> x_width_shift) & width_mask);
    const unsigned y_bits = static_cast<unsigned>(flag >> y_width_shift);
    if (flag == invalid_flag || x_bits + y_bits > maximum_coordinate_bits) return false;
    const auto payload = flag & ((UINT64_C(1) << coordinate_bits) - 1);
    const auto x = static_cast<std::uint32_t>(payload & ((UINT64_C(1) << x_bits) - 1));
    const auto y = static_cast<std::uint32_t>((payload >> x_bits) &
        ((UINT64_C(1) << y_bits) - 1));
    const auto z = payload >> (x_bits + y_bits);
    constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
    if (z > maximum) return false;
    lpfx::CartesianIndex3<std::uint32_t> owner;
    if (lpfx::try_make_cartesian_index3(x, y, static_cast<std::uint32_t>(z), owner) !=
        lpfx::ErrorCode::success) return false;
    destination = ParticleDestination::owner(owner);
    return true;
  }
};

template <class Precision> struct ParticleMigrationPoint {
  std::int64_t cell[3]{};
  typename Precision::Storage::Position fraction[3]{};
  typename Precision::Storage::Velocity velocity[3]{};
};

// Feed the original moved coordinates here BEFORE encoding a ParticleCellMap key.
// Precondition: every fraction lies in [0, 1) and every velocity is finite. The
// mover's push produces such a point; classify_stored_particle checks a stored
// record. Distributed owners reject excessive flights before mutation because
// the routing hop bound (particle_migration_hops) covers only the admitted
// halo-2 flight. A whole-domain owner has no routing constraint, so fold its
// finite coordinates first.
// The point is updated in place; on an invalid flag it is partly updated and
// both callers discard it.
template <class Precision>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE std::uint64_t classify_particle_point(
    const ParticleMigrationGeometry& geometry,
    ParticleMigrationPoint<Precision>& point) noexcept {
  using Position = typename Precision::Storage::Position;
  if (!is_local_particle_geometry(geometry))
    for (int axis = 0; axis < 3; ++axis)
      if (!communication::particle_cell_in_halo(geometry, axis, point.cell[axis]))
        return ParticleDestinationFlagDecoder::invalid_flag;

  std::uint32_t owner[3];
  for (int axis = 0; axis < 3; ++axis) {
    const std::int64_t extent = geometry.global_cells[axis];
    std::int64_t global_cell = point.cell[axis] + geometry.global_offset[axis];
    if (geometry.periodic[axis]) {
      if (global_cell < 0 || global_cell >= extent)
        global_cell = communication::positive_cell_modulo(global_cell, extent);
      // With remote periodic peers retain the sender's unwrapped halo key.
      // Its canonical global destination is encoded later without rebuilding
      // an unrepresentable sender-local index near the opposite domain edge.
      if (global_cell >= geometry.global_offset[axis] &&
          global_cell < std::int64_t(geometry.global_offset[axis]) + geometry.cells.extent(axis))
        point.cell[axis] = global_cell - geometry.global_offset[axis];
    } else if (global_cell < 0 || global_cell >= extent) {
      global_cell = communication::positive_cell_modulo(global_cell, 2 * extent);
      if (global_cell >= extent) {
        global_cell = 2 * extent - global_cell - 1;
        point.fraction[axis] = Position(1) - point.fraction[axis];
        point.velocity[axis] = -point.velocity[axis];
        if (point.fraction[axis] == Position(1)) {
          if (global_cell + 1 < extent) {
            ++global_cell;
            point.fraction[axis] = Position(0);
          } else {
            // The exact upper endpoint of the last cell moves inward to
            // 1 - epsilon/2, keeping fractions half-open in [0, 1).
            constexpr Position inward_endpoint = Position(1) -
                std::numeric_limits<Position>::epsilon() / Position(2);
            point.fraction[axis] = inward_endpoint;
          }
        }
      }
      point.cell[axis] = global_cell - geometry.global_offset[axis];
    }
    if (!communication::particle_cell_in_halo(geometry, axis, point.cell[axis]))
      return ParticleDestinationFlagDecoder::invalid_flag;
    owner[axis] = communication::particle_axis_owner(geometry, axis,
                                             static_cast<std::uint32_t>(global_cell));
  }
  return std::uint64_t(owner[0]) |
      (std::uint64_t(owner[1]) << geometry.owner_x_bits) |
      (std::uint64_t(owner[2]) << (geometry.owner_x_bits + geometry.owner_y_bits)) |
      (std::uint64_t(geometry.owner_x_bits) << ParticleDestinationFlagDecoder::x_width_shift) |
      (std::uint64_t(geometry.owner_y_bits) << ParticleDestinationFlagDecoder::y_width_shift);
}

// Startup classification of a stored record: it checks the key, the charge,
// the fractions and the velocities, which the point helper assumes, and
// stores the record only after every axis passes the boundary checks.
template <class Precision, class View>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE std::uint64_t classify_stored_particle(
    const ParticleMigrationGeometry& geometry, View view, std::uint32_t index) noexcept {
  using Schema = data::ParticleSchema<Precision>;
  const auto key = view.template get<typename Schema::Cell>()[index];
  if (key < 0 || key >= geometry.cells.count ||
      !communication::finite_or_integral(view.template get<typename Schema::Charge>()[index]))
    return ParticleDestinationFlagDecoder::invalid_flag;
  const auto cell = geometry.cells.decode(key);
  const auto fraction = data::load_fractions<Precision>(view, index);
  ParticleMigrationPoint<Precision> point{{cell.i, cell.j, cell.k},
      {fraction.x, fraction.y, fraction.z},
      {view.template get<typename Schema::U>()[index],
       view.template get<typename Schema::V>()[index],
       view.template get<typename Schema::W>()[index]}};
  if (!data::in_unit_cell(fraction.x, fraction.y, fraction.z))
    return ParticleDestinationFlagDecoder::invalid_flag;
  for (int axis = 0; axis < 3; ++axis)
    if (!communication::finite_or_integral(point.velocity[axis]))
      return ParticleDestinationFlagDecoder::invalid_flag;
  const auto flag = classify_particle_point(geometry, point);
  if (flag == ParticleDestinationFlagDecoder::invalid_flag) return flag;
  view.template get<typename Schema::Cell>()[index] = geometry.cells.encode(
      static_cast<int>(point.cell[0]), static_cast<int>(point.cell[1]),
      static_cast<int>(point.cell[2]));
  view.template get<typename Schema::Rx>()[index] = point.fraction[0];
  view.template get<typename Schema::Ry>()[index] = point.fraction[1];
  view.template get<typename Schema::Rz>()[index] = point.fraction[2];
  view.template get<typename Schema::U>()[index] = point.velocity[0];
  view.template get<typename Schema::V>()[index] = point.velocity[1];
  view.template get<typename Schema::W>()[index] = point.velocity[2];
  return flag;
}

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_PARTICLES_PARTICLE_BOUNDARIES_HPP
