#ifndef TPIC_IO_DETAIL_OUTPUT_LAYOUT_HPP
#define TPIC_IO_DETAIL_OUTPUT_LAYOUT_HPP

#include <tpic/config/domain.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/data/output/output_snapshot.hpp>
#include <tpic/data/particles/particle_cell.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <variant>

namespace tpic::io::detail {

// The particle arrays of every format, named once, in schema order. VTK orders
// them around its Points/PointData sections; dataset formats visit them in this
// order. The id is written as stored; the other three are converted from the
// records while they are written.
enum class ParticleQuantity { position, velocity, charge, id };
struct ParticleArray {
  const char* name;
  int components;
  ParticleQuantity quantity;
};

inline constexpr ParticleArray particle_position{"position", 3, ParticleQuantity::position};
inline constexpr ParticleArray particle_velocity{"velocity", 3, ParticleQuantity::velocity};
inline constexpr ParticleArray particle_charge{"charge", 1, ParticleQuantity::charge};
inline constexpr ParticleArray particle_id{"id", 1, ParticleQuantity::id};

inline constexpr std::size_t particle_array_count = 4;

template<class Visitor>
void visit_particle_array_descriptors(Visitor&& visit) {
  visit(particle_position);
  visit(particle_velocity);
  visit(particle_charge);
  visit(particle_id);
}

// Records per write: bounds every engine's conversion scratch at three doubles
// per record, so no engine holds a second image of the population.
inline constexpr std::size_t particle_chunk_records = std::size_t{1} << 20;

// Physical placement of one rank's cells.
struct ParticleGeometry {
  data::ParticleCellMap cells{};
  std::array<int, 3> offset{};
  std::array<double, 3> spacing{};
};

[[nodiscard]] inline ParticleGeometry prepare_particle_geometry(const config::PreparedDomain& domain) {
  return {data::prepare_particle_cell_map(domain), domain.global_offset, domain.spacing};
}

// Writes the values of records [first, first + out.size() / components) of a
// converted array into out, components interleaved per record. A position is
// (offset + cell + fraction) * spacing evaluated in double; velocities and
// charge widen to double. A cell key outside the interior, a fraction outside
// [0, 1) or a nonfinite value throws, so an invalid record never reaches a file.
inline void convert_particles(const ParticleArray& array, const data::ParticleRecords& records,
                              const ParticleGeometry& geometry, std::size_t first,
                              std::span<double> out) {
  const auto components = static_cast<std::size_t>(array.components);
  const std::size_t count = out.size() / components;
  const auto invalid = [] {
    throw std::invalid_argument("TPIC particle output: invalid particle coordinate or nonfinite value");
  };
  const auto widen = [&](const data::ParticleValues& values, std::size_t component) {
    std::visit([&](auto stored) {
      for (std::size_t i = 0; i < count; ++i) {
        const auto value = config::convert<double>(stored[first + i]);
        if (!std::isfinite(value)) invalid();
        out[i * components + component] = value;
      }
    }, values);
  };
  switch (array.quantity) {
  case ParticleQuantity::position: {
    for (std::size_t axis = 0; axis < 3; ++axis) widen(records.fraction[axis], axis);
    // Sorted records share keys: decode only when the key changes.
    std::int32_t key = 0;
    std::array<int, 3> cell{};
    for (std::size_t i = 0; i < count; ++i) {
      if (i == 0 || records.cell[first + i] != key) {
        key = records.cell[first + i];
        if (key < 0 || key >= geometry.cells.count) invalid();
        const auto decoded = geometry.cells.decode(key);
        if (geometry.cells.interior_bin(decoded.i, decoded.j, decoded.k) < 0) invalid();
        cell = {decoded.i, decoded.j, decoded.k};
      }
      for (std::size_t axis = 0; axis < 3; ++axis) {
        double& value = out[3 * i + axis];
        if (!(value >= 0 && value < 1)) invalid();
        value = (double(geometry.offset[axis]) + cell[axis] + value) * geometry.spacing[axis];
      }
    }
    return;
  }
  case ParticleQuantity::velocity:
    for (std::size_t axis = 0; axis < 3; ++axis) widen(records.velocity[axis], axis);
    return;
  case ParticleQuantity::charge:
    widen(records.charge, 0);
    return;
  case ParticleQuantity::id:
    break;
  }
  throw std::logic_error("TPIC particle output: ids are written as stored");
}

} // namespace tpic::io::detail

#endif // TPIC_IO_DETAIL_OUTPUT_LAYOUT_HPP
