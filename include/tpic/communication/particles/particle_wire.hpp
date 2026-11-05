#ifndef TPIC_COMMUNICATION_PARTICLES_PARTICLE_WIRE_HPP
#define TPIC_COMMUNICATION_PARTICLES_PARTICLE_WIRE_HPP

#include <lpfx/api/distributed/particle_wire.hpp>
#include <lpfx/config.hpp>
#include <lpfx/distributed/particle/wire/contract.hpp>
#include <lpfx/distributed/plan/particle_destination.hpp>
#include <tpic/communication/particles/particle_boundaries.hpp>
#include <tpic/communication/particles/particle_geometry.hpp>
#include <tpic/data/particles/particle_schema.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tpic::communication {

struct ParticleReceivedOwner {
  ParticleMigrationGeometry geometry;
  template <class Position>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE bool operator()(
      std::uint64_t, std::uint32_t, const Position& position,
      ParticleDestination& destination) const noexcept {
    if (!communication::valid_canonical_particle_position(geometry, position)) return false;
    lpfx::CartesianIndex3<std::uint32_t> owner;
    if (lpfx::try_make_cartesian_index3(
            communication::particle_axis_owner(geometry, 0, position.global_cell_x),
            communication::particle_axis_owner(geometry, 1, position.global_cell_y),
            communication::particle_axis_owner(geometry, 2, position.global_cell_z), owner) !=
        lpfx::ErrorCode::success) return false;
    destination = ParticleDestination::owner(owner);
    return true;
  }
};

struct ParticleScalarCodec {
  static constexpr std::uint64_t particle_wire_id = 0x5450494353434101; // TPICSCA 1
  template <class T>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr bool try_encode(
      const ParticleMigrationGeometry&, T value, T& wire) noexcept {
    if (!communication::finite_or_integral(value)) return false;
    wire = value;
    return true;
  }
  // Kept on receive as a transport-corruption guard for scalar payloads.
  template <class T>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr bool try_decode(
      const ParticleMigrationGeometry& geometry, T wire, T& value) noexcept {
    return try_encode(geometry, wire, value);
  }
};

template <class Precision> struct ParticlePositionCodec {
  static constexpr std::uint64_t particle_wire_id = 0x54504943504f5301; // TPICPOS 1
  template <class View, class Position>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static bool try_encode(
      const ParticleMigrationGeometry& geometry, View view, std::size_t index,
      Position& position) noexcept {
    using Schema = data::ParticleSchema<Precision>;
    const auto key = view.template get<typename Schema::Cell>()[index];
    if (key < 0 || key >= geometry.cells.count) return false;
    const auto cell = geometry.cells.decode(key);
    const int local[3]{cell.i, cell.j, cell.k};
    std::uint32_t global[3];
    for (int axis = 0; axis < 3; ++axis) {
      std::int64_t coordinate = std::int64_t(local[axis]) + geometry.global_offset[axis];
      if (geometry.periodic[axis] &&
          (coordinate < 0 || coordinate >= geometry.global_cells[axis]))
        coordinate = communication::positive_cell_modulo(coordinate, geometry.global_cells[axis]);
      if (coordinate < 0 || coordinate >= geometry.global_cells[axis]) return false;
      global[axis] = static_cast<std::uint32_t>(coordinate);
    }
    position = {global[0], global[1], global[2],
        view.template get<typename Schema::Rx>()[index],
        view.template get<typename Schema::Ry>()[index],
        view.template get<typename Schema::Rz>()[index], 0};
    return communication::valid_canonical_particle_position(geometry, position);
  }
};

// The rebuilders run only on final arrival, and LPFx decodes a received
// record only after ParticleReceivedOwner has accepted its canonical position
// (classify_received assigns the resident bucket; the host and the device
// receive paths decode resident records only). They therefore assume a
// canonical position.

// Fractions already travel in canonical routing; final arrival reuses that
// representation without transmitting a second copy as scalar attributes.
template <unsigned Axis> struct RebuildParticleFraction {
  static_assert(Axis < 3);
  static constexpr std::uint64_t particle_wire_id = 0x5450494346524101 + Axis; // TPICFRA 1-3
  template <class Position, class Scalar>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr bool try_rebuild(
      const ParticleMigrationGeometry&, const Position& position,
      Scalar& fraction) noexcept {
    static_assert(std::is_same_v<Scalar, decltype(position.fraction_x)>);
    const Scalar fractions[3]{position.fraction_x, position.fraction_y, position.fraction_z};
    fraction = fractions[Axis];
    return true;
  }
};

// Transit records remain canonical wire records; intermediate ranks must not
// try to fit them in local halos. The interior test stays: the owner check
// accepts any canonical cell of this rank's owner coordinate.
struct RebuildLocalCell {
  static constexpr std::uint64_t particle_wire_id = 0x5450494343454c01; // TPICCEL 1
  template <class Position>
  [[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE static constexpr bool try_rebuild(
      const ParticleMigrationGeometry& geometry,
      const Position& position, std::int32_t& key) noexcept {
    const auto i = std::int64_t(position.global_cell_x) - geometry.global_offset[0];
    const auto j = std::int64_t(position.global_cell_y) - geometry.global_offset[1];
    const auto k = std::int64_t(position.global_cell_z) - geometry.global_offset[2];
    if (i < 0 || i >= geometry.cells.nx || j < 0 || j >= geometry.cells.ny ||
        k < 0 || k >= geometry.cells.nz) return false;
    key = geometry.cells.encode(static_cast<int>(i), static_cast<int>(j), static_cast<int>(k));
    return true;
  }
};

// TPIC migrates particles only on DeviceRoute (MpiDirect, NCCL or RCCL), whose
// device codecs produce and consume the images: device builds use the aligned
// wire layout. Builds without a device backend keep the packed layout (host
// codecs). The choice follows the build-wide LPFx configuration, not the
// compiling translation unit, so host TUs of a device build (for example
// the memory estimate) size the same contract the device TUs register. All ranks
// run one build, so every rank registers the same descriptor version.
#if (defined(LPFX_CONFIG_DEVICE_BACKEND_CUDA) && LPFX_CONFIG_DEVICE_BACKEND_CUDA) || \
    (defined(LPFX_CONFIG_DEVICE_BACKEND_HIP) && LPFX_CONFIG_DEVICE_BACKEND_HIP)
inline constexpr lpfx::ParticleWireLayout particle_migration_wire_layout =
    lpfx::ParticleWireLayout::aligned;
#else
inline constexpr lpfx::ParticleWireLayout particle_migration_wire_layout =
    lpfx::ParticleWireLayout::packed;
#endif

template <class Precision> struct ParticleMigrationWire {
  using Hop = lpfx::AdaptiveParticleHopPolicy<
      0x54504943484f5001, lpfx::ParticleWireIdentity<0x54504943484f5002>>; // TPICHOP 1, 2
  using Routing = lpfx::CanonicalParticleRouting<
      lpfx::ParticleWireScalar<std::uint32_t, lpfx::field::wire_traits<std::uint32_t>::identity>,
      lpfx::ParticleWireScalar<typename Precision::Storage::Position,
          lpfx::field::wire_traits<typename Precision::Storage::Position>::identity>,
      ParticlePositionCodec<Precision>, lpfx::ParticleWireIdentity<0x54504943454e4401>, // TPICEND 1
      lpfx::ParticleEndpointMaskEncoding::invariant_zero_omitted>;
  using Contract = lpfx::distributed::particle_wire<typename data::ParticleSchema<Precision>::type,
      lpfx::ParticleWireEndianness::little, Hop, Routing,
      lpfx::distributed::particle_scalars<ParticleScalarCodec>,
      particle_migration_wire_layout>;
};

} // namespace tpic::communication

#endif // TPIC_COMMUNICATION_PARTICLES_PARTICLE_WIRE_HPP
