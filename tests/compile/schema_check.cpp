#include <lpfx/backend/backend.hpp>
#include <lpfx/api/field/field.hpp>
#include <lpfx/api/particle/device_particles.hpp>
#include <tpic/communication/transport_routes.hpp>

#include <tpic/config/precision.hpp>
#include <tpic/communication/mass_halo/mass_halo_device.hpp>
#include <tpic/communication/particles/initial_particle_classification_device.hpp>
#include <tpic/communication/particles/particle_migration_policy.hpp>
#include <tpic/data/fields/field_storage.hpp>
#include <tpic/data/particles/particle_schema.hpp>
#include <tpic/config/numerics_options.hpp>
#include <tpic/ecsim/fields/cell_to_node_device.hpp>
#include <tpic/ecsim/deposition/current_deposition_device.hpp>
#include <tpic/ecsim/fields/field_boundary_device.hpp>
#include <tpic/ecsim/deposition/moment_field_device.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_apply_device.hpp>
#include <tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp>
#include <tpic/ecsim/fields/node_cell_curl_device.hpp>
#include <tpic/ecsim/fields/node_smoothing_device.hpp>
#include <tpic/ecsim/maxwell/maxwell.hpp>
#include <tpic/ecsim/particles/particle_mover_device.hpp>
#include <tpic/ecsim/particles/particle_sort_device.hpp>
#include <tpic/initialization/particles/particle_centering_device.hpp>
#include <tpic/initialization/particles/particle_loading_device.hpp>

#include <cstddef>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tpic::compile_check {

// Compile actual device factories for each schema. This object has no main and
// is never executed.
template <class Precision>
void instantiate_device_owners(lpfx::device::ActiveExecutor executor) {
  using Fields = data::FieldSchema<Precision>;
  using Particles = data::ParticleSchema<Precision>;
  const auto shape = lpfx::field::options{.cells = {2, 2, 2}, .halo = 1};
  [[maybe_unused]] auto e = lpfx::field::make<typename Fields::E>(shape, executor);
  [[maybe_unused]] auto b = lpfx::field::make<typename Fields::BCell>(shape, executor);
  [[maybe_unused]] auto bn = lpfx::field::make<typename Fields::BNode>(shape, executor);
  [[maybe_unused]] auto j = lpfx::field::make<typename Fields::Current>(shape, executor);
  [[maybe_unused]] auto p = lpfx::particle::make<typename Particles::type>(
      lpfx::particle::options{.capacity = 16}, executor);
}

template void instantiate_device_owners<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(lpfx::device::ActiveExecutor);
template void instantiate_device_owners<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(lpfx::device::ActiveExecutor);
template void instantiate_device_owners<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(lpfx::device::ActiveExecutor);

// Keep the numerical launch contract compiled even when runtime tests are
// disabled. These are the same public low-level owners returned by uploads.
template <class Precision> struct MagneticInterpolation {
  using Cell = data::MagneticCellOwner<Precision, lpfx::device::ActiveBackend>;
  using Node = data::MagneticNodeOwner<Precision, lpfx::device::ActiveBackend>;
  static lpfx::ErrorCode enqueue(const Cell& cell, Node& node,
                                lpfx::device::ActiveExecutor executor) {
    return ecsim::enqueue_magnetic_cell_to_node<Precision>(cell, node, nullptr, executor);
  }
};
template struct MagneticInterpolation<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
template struct MagneticInterpolation<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>;
template struct MagneticInterpolation<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>;

// Maxwell scratch stays in solver precision even when physical B is float.
// Compile both curl directions and the split regions without a runtime
// allocation or a new test executable.
template <class Precision> struct MaxwellStencils {
  using Fields = data::FieldSchema<Precision>;
  using Compute = typename Precision::Compute::Electric;
  using Cx = lpfx::field::component<"curl_x", Compute, lpfx::field::cell>;
  using Cy = lpfx::field::component<"curl_y", Compute, lpfx::field::cell>;
  using Cz = lpfx::field::component<"curl_z", Compute, lpfx::field::cell>;
  template <class Schema> using Owner = data::DeviceFieldOwner<Schema, lpfx::device::ActiveBackend>;
  using CellVector = Owner<lpfx::field::schema<Cx, Cy, Cz>>;
  using NodeVector = Owner<typename Fields::E>;
  using Magnetic = Owner<typename Fields::BCell>;

  static void enqueue(NodeVector& node, CellVector& cell, const Magnetic& magnetic,
                      const config::PreparedDomain& domain,
                      const config::PreparedPhysics& physics,
                      lpfx::device::ActiveExecutor executor) {
    ecsim::NodeCellSpacing<Compute> spacing;
    (void)ecsim::try_make_node_cell_spacing({1.0, 1.0, 1.0}, spacing);
    const auto plans = ecsim::prepare_field_operator_plans(domain, physics);
    (void)ecsim::enqueue_node_cell_curl<Precision>(node, cell, plans.cells, spacing, executor);
    (void)ecsim::enqueue_node_cell_curl<Precision>(cell, node, plans.nodes, spacing, executor);
    (void)ecsim::enqueue_node_cell_curl<Precision, ecsim::StencilRegion::strict_interior>(
        cell, node, plans.nodes, spacing, executor);
    (void)ecsim::enqueue_node_cell_curl<Precision, ecsim::StencilRegion::shell>(
        cell, node, plans.nodes, spacing, executor);
    (void)ecsim::enqueue_node_cell_curl<Precision>(magnetic, node, plans.nodes, spacing,
                                                   executor);
  }

  static void smooth(const NodeVector& source, NodeVector& target,
                     lpfx::device::ActiveExecutor executor, unsigned* invalid) {
    (void)ecsim::enqueue_node_smoothing<Precision>(source, target,
        config::EcsimSmoothingOperator::Face7, Compute(0.5), invalid, executor);
    (void)ecsim::enqueue_node_smoothing<Precision>(source, target,
        config::EcsimSmoothingOperator::Binomial27, Compute(0.5), invalid, executor);
  }

  static void boundaries(NodeVector& node, const NodeVector& input, Magnetic& magnetic,
                         CellVector& curl, const config::PreparedDomain& domain,
                         const config::PreparedPhysics& physics,
                         lpfx::device::ActiveExecutor executor) {
    const auto plans = ecsim::prepare_field_boundary_plans(domain, physics);
    (void)ecsim::enqueue_wall_ghosts(node, plans, executor);
    (void)ecsim::enqueue_wall_ghosts(magnetic, plans, executor);
    (void)ecsim::enqueue_wall_ghosts(curl, plans, executor);
    (void)ecsim::enqueue_constrained_rows(node, ecsim::detail::NoRowInput{},
        plans.constrained_rows, executor);
    (void)ecsim::enqueue_constrained_rows(node, input, plans.constrained_rows, executor);
  }
};
template struct MaxwellStencils<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
template struct MaxwellStencils<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>;
template struct MaxwellStencils<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>;

template <class Precision>
void instantiate_velocity_centering(const lpfx::device::DeviceProperties& properties,
                                    lpfx::device::ActiveExecutor executor) {
  initialization::VelocityCentering<Precision, lpfx::device::ActiveBackend>
      workspace(16, properties, executor);
  [[maybe_unused]] auto operand = workspace.collective_operand();
}
template void instantiate_velocity_centering<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(
    const lpfx::device::DeviceProperties&, lpfx::device::ActiveExecutor);
template void instantiate_velocity_centering<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(
    const lpfx::device::DeviceProperties&, lpfx::device::ActiveExecutor);
template void instantiate_velocity_centering<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(
    const lpfx::device::DeviceProperties&, lpfx::device::ActiveExecutor);

template <class Precision>
void instantiate_particle_loading(const initialization::ParticleLoadPlan& plan,
                                  lpfx::device::ActiveExecutor executor) {
  [[maybe_unused]] auto particles = initialization::make_initial_particles<Precision>(
      plan, config::PreparedDomain{}, initialization::InitialCase{}, 0, executor);
}
template void instantiate_particle_loading<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(
    const initialization::ParticleLoadPlan&, lpfx::device::ActiveExecutor);
template void instantiate_particle_loading<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(
    const initialization::ParticleLoadPlan&, lpfx::device::ActiveExecutor);
template void instantiate_particle_loading<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(
    const initialization::ParticleLoadPlan&, lpfx::device::ActiveExecutor);

template <class Precision> struct ParticleHandoff {
  using Backend = lpfx::device::ActiveBackend;
  using Executor = lpfx::device::ActiveExecutor;
  using Owner = lpfx::Particles<typename data::ParticleSchema<Precision>::type,
      lpfx::DeviceSpace<Backend>>;
  static auto describe(Owner& particles, const config::PreparedDomain& domain,
      communication::ParticleMigrationGeometry geometry,
      lpfx::DeviceParticleClassificationSlots<Backend>& slots, Executor executor) {
    namespace d = lpfx::distributed;
    return d::operations(d::migrate<"particles">(particles, domain.particle_topology,
        communication::particle_migration_policy<Precision>(geometry),
        d::on<communication::DeviceRoute>(executor), d::using_flags(slots),
        d::traffic_seed(1, d::arrivals_up_to_owner), d::resident_access<false>));
  }
  using Communication = lpfx::distributed::communication_for<decltype(describe(
      std::declval<Owner&>(), std::declval<const config::PreparedDomain&>(),
      std::declval<communication::ParticleMigrationGeometry>(),
      std::declval<lpfx::DeviceParticleClassificationSlots<Backend>&>(), std::declval<Executor>()))>;
  using Migration = std::remove_reference_t<decltype(
      std::declval<Communication&>().template get<"particles">())>;
  static void publish(const communication::ParticleMigrationGeometry& geometry,
      Migration& migration, Owner& particles, Executor executor) {
    (void)communication::enqueue_initial_particle_classification<Precision>(
        migration, particles, geometry, executor);
  }
};
template struct ParticleHandoff<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
template struct ParticleHandoff<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>;
template struct ParticleHandoff<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>;

// Compile the local deposition and the Jhat clear and assembly operations.
template <class Precision> struct CurrentDeposition {
  using Fields = data::FieldSchema<Precision>;
  using Particles = lpfx::Particles<typename data::ParticleSchema<Precision>::type,
      lpfx::DeviceSpace<lpfx::device::ActiveBackend>>;
  template <class Schema> using Owner = data::DeviceFieldOwner<Schema, lpfx::device::ActiveBackend>;
  static void enqueue(const Particles& particles,
      const Owner<typename Fields::BNode>& magnetic,
      const Owner<typename Fields::E>& electric,
      Owner<typename Fields::Current>& current, const config::PreparedDomain& domain,
      unsigned* invalid, lpfx::device::ActiveExecutor executor) {
    (void)ecsim::enqueue_clear_moment_field(current, executor);
    (void)ecsim::enqueue_unsorted_current<Precision>(particles, magnetic,
        current, data::prepare_particle_cell_map(domain), domain.inverse_cell_volume, 0.25, 0, particles.size(), invalid,
        executor);
    const auto moment_plan = ecsim::prepare_moment_field_plan(domain);
    (void)ecsim::enqueue_scale_moment_dual_volume(current, moment_plan, executor);
    (void)ecsim::enqueue_fill_moment_neumann_ghost(current, moment_plan, executor);
    data::CompactMassMatrix<Precision, lpfx::device::ActiveBackend> mass(domain);
    (void)ecsim::enqueue_clear_mass_matrix(mass, executor);
    (void)ecsim::enqueue_unsorted_mass_matrix<Precision>(particles, magnetic,
        mass, data::prepare_particle_cell_map(domain), 0.25, 0, particles.size(), invalid,
        executor);
    ecsim::WallFaces dual_volume;
    (void)ecsim::try_prepare_wall_faces(domain, dual_volume);
    ecsim::StencilRegionGeometry nodes;
    const auto* cells = domain.local_cells.data();
    (void)ecsim::try_make_stencil_region_geometry(cells[0] + 1, cells[1] + 1, cells[2] + 1, nodes);
    using Solver = typename Precision::Compute::Electric;
    const auto inverse_volume = static_cast<Solver>(domain.inverse_cell_volume);
    (void)ecsim::enqueue_mass_matrix_apply<Precision>(mass, electric, current,
        nodes, dual_volume, inverse_volume, invalid, executor);
    (void)ecsim::enqueue_mass_matrix_apply<Precision, ecsim::StencilRegion::strict_interior>(
        mass, electric, current, nodes, dual_volume, inverse_volume, invalid, executor);
    (void)ecsim::enqueue_mass_matrix_apply<Precision, ecsim::StencilRegion::shell>(
        mass, electric, current, nodes, dual_volume, inverse_volume, invalid, executor);
  }
};
template struct CurrentDeposition<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
template struct CurrentDeposition<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>;
template struct CurrentDeposition<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>;

// Keep the sort receipt consumers compiled.
template <class Precision> struct SortedParticles {
  using Backend = lpfx::device::ActiveBackend;
  using Fields = data::FieldSchema<Precision>;
  using Sort = ecsim::FullParticleSort<Precision, Backend>;
  template <class Schema> using Owner = data::DeviceFieldOwner<Schema, Backend>;

  static void enqueue(typename Sort::Owner& particles,
      typename ParticleHandoff<Precision>::Migration& migration,
      const communication::ParticleMigrationGeometry& geometry,
      const Owner<typename Fields::BNode>& magnetic, const Owner<typename Fields::E>& electric,
      Owner<typename Fields::Current>& current,
      data::CompactMassMatrix<Precision, Backend>& mass, const config::PreparedDomain& domain,
      const lpfx::device::DeviceProperties& properties, unsigned* invalid,
      lpfx::device::ActiveExecutor executor) {
    Sort ordering;
    ordering.initialize(particles, domain, properties, executor);
    ordering.rebind_to_owner(particles, properties, executor);
    ordering.enqueue(particles, domain, executor);
    ordering.host_wait_complete();
    (void)ecsim::enqueue_sorted_current<Precision>(particles, ordering.receipt(), magnetic,
        current, data::prepare_particle_cell_map(domain), domain.inverse_cell_volume, 0.25, invalid,
        executor);
    (void)ecsim::enqueue_sorted_mass_matrix<Precision>(particles, ordering.receipt(), magnetic,
        mass, data::prepare_particle_cell_map(domain), 0.25, invalid, executor);
    const ecsim::ParticleMoverParameters<Precision> mover{0.25, 0.25, 0.125, {1, 1, 1}};
    (void)ecsim::enqueue_migrating_particle_mover<Precision>(&ordering.receipt(), magnetic,
        electric, particles, migration, geometry, mover, invalid, executor);
    (void)ecsim::enqueue_migrating_particle_mover<Precision>(
        static_cast<const typename Sort::Receipt*>(nullptr), magnetic, electric, particles,
        migration, geometry, mover, invalid, executor);
    (void)ecsim::enqueue_local_particle_mover<Precision>(&ordering.receipt(), magnetic, electric,
        particles, geometry, mover, invalid, executor);
    (void)ecsim::enqueue_local_particle_mover<Precision>(
        static_cast<const typename Sort::Receipt*>(nullptr), magnetic, electric, particles,
        geometry, mover, invalid, executor);
    ordering.host_wait_retired(executor);
    ordering.rebind_to_owner(particles, properties, executor);
    ordering.enqueue(particles, domain, executor);
    ordering.host_wait_complete();
    ordering.host_wait_retired(executor);
  }
};
template struct SortedParticles<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
template struct SortedParticles<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>;
template struct SortedParticles<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>;

// Keep the selected LPFx mass transport compiled for each Mass storage type without
// adding a separate test executable or changing production precision dispatch.
template <class Precision>
void instantiate_mass_halo(const config::PreparedDomain& domain,
    const lpfx::distributed::CommunicationSetup& setup,
    lpfx::HostMemoryBudget& registration_budget,
    lpfx::device::ActiveExecutor executor) {
  data::CompactMassMatrix<Precision, lpfx::device::ActiveBackend> mass(domain);
  using Storage =
      communication::MassHaloStorage<Precision, lpfx::device::ActiveBackend>;
  Storage storage(domain, mass);
  storage.prepare(setup, executor);
  auto add_binding = lpfx::distributed::bind<communication::DeviceRoute>(
      storage.plan().template phase<true>().message_plan(), storage.template buffers<true>(), executor);
  auto ghost_binding = lpfx::distributed::bind<communication::DeviceRoute>(
      storage.plan().template phase<false>().message_plan(), storage.template buffers<false>(), executor);
  lpfx::distributed::Communication<decltype(add_binding), decltype(ghost_binding)> owner;
  lpfx::HostMemoryBudget operation_budget;
  if (!operation_budget.try_set_limit(std::numeric_limits<std::size_t>::max()) ||
      !operation_budget.seal())
    throw std::runtime_error("mass schema operation budget failed");
  if (lpfx::distributed::try_prepare_communication(
          owner, setup, registration_budget, operation_budget,
          std::move(add_binding), std::move(ghost_binding)) != lpfx::ErrorCode::success)
    throw std::runtime_error("mass schema communication failed");
  using Plan = typename Storage::plan_type;
  auto &add = owner.template get<typename Plan::template phase_type<true>::definition_type>();
  auto &ghosts = owner.template get<typename Plan::template phase_type<false>::definition_type>();
  storage.begin(add);
  while (storage.progress(add, ghosts) != communication::HaloProgress::complete) {}
  storage.conclude();
  if (!owner.close().is_success())
    throw std::runtime_error("mass schema communication close failed");
  storage.close();
}
template void instantiate_mass_halo<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>(const config::PreparedDomain&,
    const lpfx::distributed::CommunicationSetup&, lpfx::HostMemoryBudget&,
    lpfx::device::ActiveExecutor);
template void instantiate_mass_halo<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>(const config::PreparedDomain&,
    const lpfx::distributed::CommunicationSetup&, lpfx::HostMemoryBudget&,
    lpfx::device::ActiveExecutor);
template void instantiate_mass_halo<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>(const config::PreparedDomain&,
    const lpfx::distributed::CommunicationSetup&, lpfx::HostMemoryBudget&,
    lpfx::device::ActiveExecutor);

static_assert(std::is_same_v<tpic::config::StorageSingleParticlesMomentsFields::Electric, float>);
static_assert(std::is_same_v<tpic::config::StorageSingleParticlesMomentsFields::Magnetic, float>);
static_assert(std::is_same_v<tpic::config::StorageSinglePositionVelocity::Magnetic, double>);
static_assert(std::is_same_v<tpic::config::StorageSingleParticlesMomentsFieldsDoubleMass::Mass, double>);
static_assert(std::is_same_v<tpic::config::ComputeSingleParticlesMomentsFieldsDoubleMass::Mass, double>);
static_assert(std::is_same_v<tpic::config::ComputeSingleParticlesMomentsFields::SolverReduction, double>);
static_assert(data::ParticleSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>::type::attribute_count == 9);

} // namespace tpic::compile_check
