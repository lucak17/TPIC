#ifndef TPIC_CONFIG_PRECISION_HPP
#define TPIC_CONFIG_PRECISION_HPP

#include <lpfx/core/attributes.hpp>

#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace tpic::config {

// A run selects one storage struct and one compute struct. Storage::X is the
// type quantity X is stored in; Compute::X is the arithmetic type that
// produces X. Every struct lists every member; the two families are
// independent and any storage struct combines with any compute struct.
enum class StoragePrecision {
  double_precision, single_position, single_position_velocity,
  single_particles_moments_fields_double_mass, single_particles_moments_fields
};
enum class ComputePrecision {
  double_precision, single_position, single_position_velocity,
  single_particles_moments_fields_double_mass, single_particles_moments_fields
};
inline constexpr unsigned precision_level_count = 5;

struct StorageDouble {
  static constexpr StoragePrecision id = StoragePrecision::double_precision;
  static constexpr std::string_view name = "double";
  using Position = double;
  using Velocity = double;
  using Weight = double;
  using Electric = double;
  using Magnetic = double;
  using Current = double;
  using Mass = double;
};

struct StorageSinglePosition {
  static constexpr StoragePrecision id = StoragePrecision::single_position;
  static constexpr std::string_view name = "single-position";
  using Position = float;
  using Velocity = double;
  using Weight = double;
  using Electric = double;
  using Magnetic = double;
  using Current = double;
  using Mass = double;
};

struct StorageSinglePositionVelocity {
  static constexpr StoragePrecision id = StoragePrecision::single_position_velocity;
  static constexpr std::string_view name = "single-position-velocity";
  using Position = float;
  using Velocity = float;
  using Weight = double;
  using Electric = double;
  using Magnetic = double;
  using Current = double;
  using Mass = double;
};

struct StorageSingleParticlesMomentsFieldsDoubleMass {
  static constexpr StoragePrecision id = StoragePrecision::single_particles_moments_fields_double_mass;
  static constexpr std::string_view name = "single-particles-moments-fields-double-mass";
  using Position = float;
  using Velocity = float;
  using Weight = double;
  using Electric = float;
  using Magnetic = float;
  using Current = float;
  using Mass = double;
};

struct StorageSingleParticlesMomentsFields {
  static constexpr StoragePrecision id = StoragePrecision::single_particles_moments_fields;
  static constexpr std::string_view name = "single-particles-moments-fields";
  using Position = float;
  using Velocity = float;
  using Weight = double;
  using Electric = float;
  using Magnetic = float;
  using Current = float;
  using Mass = float;
};

// SolverReduction types the GMRES dot products, norms, Hessenberg state and
// residual; Reduction types the diagnostic and centering sums and the output
// moments.
struct ComputeDouble {
  static constexpr ComputePrecision id = ComputePrecision::double_precision;
  static constexpr std::string_view name = "double";
  using Position = double;
  using Velocity = double;
  using Electric = double;
  using Magnetic = double;
  using Current = double;
  using Mass = double;
  using SolverReduction = double;
  using Reduction = double;
};

struct ComputeSinglePosition {
  static constexpr ComputePrecision id = ComputePrecision::single_position;
  static constexpr std::string_view name = "single-position";
  using Position = float;
  using Velocity = double;
  using Electric = double;
  using Magnetic = double;
  using Current = double;
  using Mass = double;
  using SolverReduction = double;
  using Reduction = double;
};

struct ComputeSinglePositionVelocity {
  static constexpr ComputePrecision id = ComputePrecision::single_position_velocity;
  static constexpr std::string_view name = "single-position-velocity";
  using Position = float;
  using Velocity = float;
  using Electric = double;
  using Magnetic = double;
  using Current = double;
  using Mass = double;
  using SolverReduction = double;
  using Reduction = double;
};

struct ComputeSingleParticlesMomentsFieldsDoubleMass {
  static constexpr ComputePrecision id = ComputePrecision::single_particles_moments_fields_double_mass;
  static constexpr std::string_view name = "single-particles-moments-fields-double-mass";
  using Position = float;
  using Velocity = float;
  using Electric = float;
  using Magnetic = float;
  using Current = float;
  using Mass = double;
  using SolverReduction = double;
  using Reduction = double;
};

struct ComputeSingleParticlesMomentsFields {
  static constexpr ComputePrecision id = ComputePrecision::single_particles_moments_fields;
  static constexpr std::string_view name = "single-particles-moments-fields";
  using Position = float;
  using Velocity = float;
  using Electric = float;
  using Magnetic = float;
  using Current = float;
  using Mass = float;
  using SolverReduction = double;
  using Reduction = double;
};

// The template argument of every kernel, schema and owner.
template <class StorageStruct, class ComputeStruct> struct Precision {
  using Storage = StorageStruct;
  using Compute = ComputeStruct;
};

// The one conversion between two precision roles. Equal types return the
// value itself: no cast is instantiated.
template <class To, class From>
[[nodiscard]] LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE constexpr To convert(From value) noexcept {
  static_assert(std::is_arithmetic_v<To> && std::is_arithmetic_v<From>);
  if constexpr (std::is_same_v<To, From>) return value;
  else return static_cast<To>(value);
}

#ifndef TPIC_DEFAULT_STORAGE_PRECISION_ID
#define TPIC_DEFAULT_STORAGE_PRECISION_ID 0
#endif
#ifndef TPIC_DEFAULT_COMPUTE_PRECISION_ID
#define TPIC_DEFAULT_COMPUTE_PRECISION_ID 0
#endif
// Bit storage * precision_level_count + compute is set for each pair the build compiles.
#ifndef TPIC_PRECISION_PAIR_MASK
#define TPIC_PRECISION_PAIR_MASK ((1U << (precision_level_count * precision_level_count)) - 1U)
#endif
static_assert(static_cast<unsigned>(TPIC_DEFAULT_STORAGE_PRECISION_ID) < precision_level_count &&
              static_cast<unsigned>(TPIC_DEFAULT_COMPUTE_PRECISION_ID) < precision_level_count);

struct PrecisionSelection {
  StoragePrecision storage = static_cast<StoragePrecision>(TPIC_DEFAULT_STORAGE_PRECISION_ID);
  ComputePrecision compute = static_cast<ComputePrecision>(TPIC_DEFAULT_COMPUTE_PRECISION_ID);
  friend constexpr bool operator==(const PrecisionSelection&, const PrecisionSelection&) = default;
};

[[nodiscard]] constexpr unsigned precision_pair_index(PrecisionSelection selection) noexcept {
  return static_cast<unsigned>(selection.storage) * precision_level_count +
         static_cast<unsigned>(selection.compute);
}
[[nodiscard]] constexpr bool precision_pair_compiled(PrecisionSelection selection) noexcept {
  return ((static_cast<unsigned>(TPIC_PRECISION_PAIR_MASK) >> precision_pair_index(selection)) & 1U) != 0;
}
template <class P>
inline constexpr bool precision_pair_compiled_v =
    precision_pair_compiled({P::Storage::id, P::Compute::id});

// The switches over the levels: each calls f.template operator()<Struct>().
template <class F> constexpr decltype(auto) dispatch_storage(StoragePrecision storage, F&& f) {
  switch (storage) {
  case StoragePrecision::double_precision:
    return std::forward<F>(f).template operator()<StorageDouble>();
  case StoragePrecision::single_position:
    return std::forward<F>(f).template operator()<StorageSinglePosition>();
  case StoragePrecision::single_position_velocity:
    return std::forward<F>(f).template operator()<StorageSinglePositionVelocity>();
  case StoragePrecision::single_particles_moments_fields_double_mass:
    return std::forward<F>(f).template operator()<StorageSingleParticlesMomentsFieldsDoubleMass>();
  case StoragePrecision::single_particles_moments_fields:
    return std::forward<F>(f).template operator()<StorageSingleParticlesMomentsFields>();
  }
  throw std::invalid_argument("TPIC precision: invalid storage precision");
}
template <class F> constexpr decltype(auto) dispatch_compute(ComputePrecision compute, F&& f) {
  switch (compute) {
  case ComputePrecision::double_precision:
    return std::forward<F>(f).template operator()<ComputeDouble>();
  case ComputePrecision::single_position:
    return std::forward<F>(f).template operator()<ComputeSinglePosition>();
  case ComputePrecision::single_position_velocity:
    return std::forward<F>(f).template operator()<ComputeSinglePositionVelocity>();
  case ComputePrecision::single_particles_moments_fields_double_mass:
    return std::forward<F>(f).template operator()<ComputeSingleParticlesMomentsFieldsDoubleMass>();
  case ComputePrecision::single_particles_moments_fields:
    return std::forward<F>(f).template operator()<ComputeSingleParticlesMomentsFields>();
  }
  throw std::invalid_argument("TPIC precision: invalid compute precision");
}
// Calls f.template operator()<Precision<Storage, Compute>>() for the selected pair.
template <class F> constexpr decltype(auto) dispatch_precision(PrecisionSelection selection, F&& f) {
  return dispatch_storage(selection.storage, [&]<class Storage> {
    return dispatch_compute(selection.compute, [&]<class Compute> {
      return f.template operator()<Precision<Storage, Compute>>();
    });
  });
}

[[nodiscard]] constexpr std::string_view storage_precision_name(StoragePrecision storage) {
  return dispatch_storage(storage, [&]<class Storage> { return Storage::name; });
}
[[nodiscard]] constexpr std::string_view compute_precision_name(ComputePrecision compute) {
  return dispatch_compute(compute, [&]<class Compute> { return Compute::name; });
}
[[nodiscard]] constexpr StoragePrecision parse_storage_precision(std::string_view name) {
  for (unsigned level = 0; level < precision_level_count; ++level)
    if (storage_precision_name(static_cast<StoragePrecision>(level)) == name)
      return static_cast<StoragePrecision>(level);
  throw std::invalid_argument("TPIC precision: --storage-precision must be double, single-position, "
                              "single-position-velocity, single-particles-moments-fields-double-mass or "
                              "single-particles-moments-fields");
}
[[nodiscard]] constexpr ComputePrecision parse_compute_precision(std::string_view name) {
  for (unsigned level = 0; level < precision_level_count; ++level)
    if (compute_precision_name(static_cast<ComputePrecision>(level)) == name)
      return static_cast<ComputePrecision>(level);
  throw std::invalid_argument("TPIC precision: --compute-precision must be double, single-position, "
                              "single-position-velocity, single-particles-moments-fields-double-mass or "
                              "single-particles-moments-fields");
}

// The matched pair of a compute level: the default build compiles it and --precision
// selects it. Storage is the same level, except that
// single-particles-moments-fields-double-mass keeps the mass matrix stored in
// FP32 (storage single-particles-moments-fields): GMRES reads the matrix on
// every iteration, and FP64 mass storage nearly doubles the solve time.
[[nodiscard]] constexpr PrecisionSelection matched_precision(ComputePrecision compute) noexcept {
  return {compute == ComputePrecision::single_particles_moments_fields_double_mass
              ? StoragePrecision::single_particles_moments_fields
              : static_cast<StoragePrecision>(compute),
          compute};
}
[[nodiscard]] constexpr PrecisionSelection parse_precision(std::string_view name) {
  for (unsigned level = 0; level < precision_level_count; ++level)
    if (compute_precision_name(static_cast<ComputePrecision>(level)) == name)
      return matched_precision(static_cast<ComputePrecision>(level));
  throw std::invalid_argument("TPIC precision: --precision must be double, single-position, "
                              "single-position-velocity, single-particles-moments-fields-double-mass or "
                              "single-particles-moments-fields");
}

// The name is one token, "<storage>/<compute>"; output metadata records it and
// the roles, and tools read the name back.
struct PrecisionMetadata {
  std::string name;
  std::string roles;
};

[[nodiscard]] inline std::string precision_name(PrecisionSelection selection) {
  return std::string(storage_precision_name(selection.storage)) + '/' +
         std::string(compute_precision_name(selection.compute));
}

[[nodiscard]] inline PrecisionMetadata precision_metadata(PrecisionSelection selection) {
  return dispatch_precision(selection, [&]<class P> {
    using Storage = typename P::Storage;
    using Compute = typename P::Compute;
    std::string roles;
    const auto role = [&]<class T>(std::string_view key) {
      static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
      roles += key;
      roles += std::is_same_v<T, float> ? "=float32 " : "=float64 ";
    };
    roles += "storage: ";
    role.template operator()<typename Storage::Position>("position");
    role.template operator()<typename Storage::Velocity>("velocity");
    role.template operator()<typename Storage::Weight>("weight");
    role.template operator()<typename Storage::Electric>("E");
    role.template operator()<typename Storage::Magnetic>("B");
    role.template operator()<typename Storage::Current>("J");
    role.template operator()<typename Storage::Mass>("mass");
    roles += "cell=int32 ID=uint64; compute: ";
    role.template operator()<typename Compute::Position>("position");
    role.template operator()<typename Compute::Velocity>("velocity");
    role.template operator()<typename Compute::Electric>("E");
    role.template operator()<typename Compute::Magnetic>("B");
    role.template operator()<typename Compute::Current>("J");
    role.template operator()<typename Compute::Mass>("mass");
    role.template operator()<typename Compute::SolverReduction>("solver-reduction");
    role.template operator()<typename Compute::Reduction>("reduction");
    roles += "output-moment-accumulation=float64";
    return PrecisionMetadata{precision_name(selection), std::move(roles)};
  });
}

} // namespace tpic::config

#endif // TPIC_CONFIG_PRECISION_HPP
