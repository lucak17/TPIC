#ifndef TPIC_INITIALIZATION_CASES_CASE_INTERFACE_HPP
#define TPIC_INITIALIZATION_CASES_CASE_INTERFACE_HPP

// The standard shape of a simulation case. A case is one struct, declared in
// cases/<name>.hpp and implemented in src/initialization/cases/<name>.cpp:
//
//   struct MyCase {
//     static constexpr std::array<std::string_view, 1> names{"MyCase"};
//     explicit MyCase(const config::SimulationParameters& params);
//     FieldVector electric(const config::SimulationParameters& p, double x, double y, double z) const;
//     FieldVector magnetic(const config::SimulationParameters& p, double x, double y, double z) const;
//     std::string_view particle_label() const;
//     using Profile = MyProfile;
//     Profile profile(std::size_t species) const noexcept;
//     bool operator==(const MyCase&) const = default;
//   };
//
// names: the values of the deck key Case that select the case, each also
//   listed in config::case_names.
// Constructor: admission of the deck values (with the checks of
//   initial_case.hpp) and every derived coefficient, stored as members. It is
//   the only member that throws.
// electric, magnetic: the analytic seed fields at a physical position, halo
//   positions included. magnetic excludes the uniform background, which
//   initial_magnetic adds once; a case that needs the total field adds
//   p.b0x, p.b0y and p.b0z itself.
// particle_label: the loading named in the startup report.
// Profile: the device loading of one species, a trivially copyable kernel
//   argument (UniformProfile when the loading is uniform) with
//     void evaluate(ParticleProfileSample& sample, const ParticleLoadPlan& plan,
//                   double x, double y, double z, const int cell[3]) const noexcept;
//     void add_velocity_seed(double x, double y, double z,
//                            double& u, double& v, double& w) const noexcept;
//     ParticleSupport support() const noexcept;
//   The deck key QuietStart, not the profile, places the particles: at the
//   subcell centres or at three uniform offsets. evaluate receives the
//   species' uniform Maxwellian in sample and the stored cell of the
//   particle, after the endpoint carry. add_velocity_seed runs after
//   v = drift + thermal * normal. The drift evaluate returns depends on the
//   stored position and cell: velocity centering evaluates it again there and
//   keeps its mean. The seed is a perturbation of zero mean, which centering
//   removes with the sampled one. support is the host's region of the load;
//   ParticleSupport{} loads every subcell.
//
// A new case adds its two files, its include and type in initial_case.hpp,
// its source in CMakeLists.txt and its names in config::case_names. New deck
// keys are members of config::SimulationParameters read by the input parser.

#include <lpfx/core/attributes.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/initialization/particles/particle_load_plan.hpp>
#include <tpic/initialization/particles/particle_support.hpp>

#include <array>

namespace tpic::initialization {

using FieldVector = std::array<double, 3>;

struct ParticleProfileSample {
  // Internal density includes rhoINIT/(4*pi), but no charge sign or volume.
  // Keep the TwoStream multiplier separate to preserve q = (q_factor*rho)*seed.
  double density{};
  double density_multiplier = 1.0;
  double drift[3]{};
  double thermal[3]{};
};

// The species Maxwellian, uniform in space.
struct UniformProfile {
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void evaluate(ParticleProfileSample&, const ParticleLoadPlan&, double, double,
                                 double, const int*) const noexcept {}
  LPFX_HOST_DEVICE LPFX_ALWAYS_INLINE void add_velocity_seed(double, double, double, double&, double&,
                                          double&) const noexcept {}
  [[nodiscard]] ParticleSupport support() const noexcept { return {}; }
  bool operator==(const UniformProfile&) const = default;
};

} // namespace tpic::initialization

#endif // TPIC_INITIALIZATION_CASES_CASE_INTERFACE_HPP
