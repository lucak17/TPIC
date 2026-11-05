#include <tpic/config/domain.hpp>
#include <tpic/config/precision.hpp>
#include <tpic/config/simulation_parameters.hpp>
#include <device/mock_backend.hpp>
#include <lpfx/distributed/field/host.hpp>
#include <lpfx/distributed/transport/routes.hpp>
#include <lpfx/domain/cartesian/direction_mask.hpp>
#include "field_plan_fixture.hpp"
#include <tpic/communication/fields/field_exchange.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/data/fields/field_storage.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace d = lpfx::distributed;
using Domain = tpic::config::PreparedDomain;
using Flow = lpfx::FieldFlow;
using Pass = lpfx::FieldSemanticPass;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void success(lpfx::ErrorCode error, const char* message) {
  require(error == lpfx::ErrorCode::success, message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument& error) {
    require(std::string(error.what()).starts_with("TPIC field fixture:"),
            "halo preparation failure has no TPIC context");
    return;
  }
  throw std::runtime_error("invalid field domain accepted");
}

tpic::config::SimulationParameters parameters(bool distributed, bool physical) {
  tpic::config::SimulationParameters p;
  p.grid.nxc = distributed ? 7 : 4;
  p.grid.nyc = distributed ? 5 : 3;
  p.grid.nzc = distributed ? 8 : 2;
  p.topology.xlen = distributed ? 2 : 1;
  p.topology.ylen = distributed ? 2 : 1;
  p.topology.zlen = distributed ? 3 : 1;
  p.topology.periodic_y = !physical;
  p.topology.periodic_z = !physical;
  // Opposite particle periodicity catches accidental PARTICLE topology use.
  p.topology.periodic_x_particles = false;
  p.topology.periodic_y_particles = physical;
  p.topology.periodic_z_particles = physical;
  return p;
}

template <class Schema> auto make_field(const Domain& domain) {
  auto result = lpfx::field::make<Schema>(lpfx::field::options{
      .cells = {domain.local_cells[0], domain.local_cells[1], domain.local_cells[2]},
      .halo = Domain::field_halo});
  require(result.has_value(), "host field allocation failed");
  return std::move(result).value();
}

template <class Owner, class Function> void samples(Owner& owner, Function function) {
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    const auto visit = [&]<std::size_t C>() {
      const auto view = owner.view().template component<C>();
      const auto n = view.layout().interior_extents();
      for (int z = -1; z <= static_cast<int>(n.z()); ++z)
        for (int y = -1; y <= static_cast<int>(n.y()); ++y)
          for (int x = -1; x <= static_cast<int>(n.x()); ++x) {
            const lpfx::CartesianDirection3 direction{
                static_cast<std::int8_t>(x < 0 ? -1 : x >= static_cast<int>(n.x())),
                static_cast<std::int8_t>(y < 0 ? -1 : y >= static_cast<int>(n.y())),
                static_cast<std::int8_t>(z < 0 ? -1 : z >= static_cast<int>(n.z()))};
            function(view(x, y, z), C, x, y, z, direction);
          }
    };
    (visit.template operator()<I>(), ...);
  }(std::make_index_sequence<Owner::component_count>{});
}

double sample_value(const Domain& domain, int epoch, std::size_t component,
                    int x, int y, int z) {
  std::array q{x + domain.global_offset[0], y + domain.global_offset[1],
               z + domain.global_offset[2]};
  const auto periodic = domain.field_neighborhood.topology().input().periodicity();
  const std::array flags{periodic.x, periodic.y, periodic.z};
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (flags[axis]) q[axis] = (q[axis] % domain.global_cells[axis] +
                              domain.global_cells[axis]) % domain.global_cells[axis];
  return epoch * 100000.0 + component * 10000.0 + 100.0 * q[2] + 10.0 * q[1] + q[0] + .125;
}

bool ghost(lpfx::CartesianDirection3 direction) {
  return direction.x || direction.y || direction.z;
}

// Test-only transport harness. LPFx's internal host execution runs the prepared
// pack/copy schedules; immutable message images replace MPI between logical
// ranks. Production code uses public LPFx object requests.
template <class Schema> void exchange_scenario(
    const tpic::config::SimulationParameters& p, bool break_shared_coherence = false) {
  using Owner = lpfx::field::host<Schema>;
  using Plan = decltype(tpic::test::field_copy_plan(
      std::declval<Owner&>(), std::declval<const Domain&>()));
  using Execution = d::detail::HostFieldExecution<typename Plan::pattern_type, Owner>;
  static_assert(lpfx::cartesian_neighbor_mask_count(Plan::spec_type::transport_mask) == 26);
  static_assert(Plan::descriptor_type::kind == d::field_operation_kind::copy_to_ghost);
  const int ranks = tpic::config::configured_process_count(p.topology);
  lpfx::HostMemoryBudget budget;
  require(budget.try_set_limit(16 * 1024 * 1024) && budget.seal(), "execution budget failed");
  std::vector<Domain> domains;
  auto plans = std::make_unique<Plan[]>(ranks);
  auto owners = std::make_unique<Owner[]>(ranks);
  auto executions = std::make_unique<Execution[]>(ranks);
  std::vector<d::detail::FieldExecutionImages> images(ranks);
  for (int rank = 0; rank < ranks; ++rank) {
    domains.push_back(tpic::config::prepare_domain(p.grid, p.topology, rank, ranks));
    owners[rank] = make_field<Schema>(domains.back());
    plans[rank] = tpic::test::field_copy_plan(owners[rank], domains.back());
    require(d::bind<d::transport::MpiHost>(
                plans[rank], owners[rank], lpfx::SerialExecutor{})
                .valid(),
            "public host field binding invalid");
    success(executions[rank].prepare(budget, plans[rank], owners[rank],
                                   lpfx::SerialExecutor{}, images[rank]), "host exchange preparation failed");
  }
  const auto prepared_bytes = budget.used();
  for (int epoch = 0; epoch < 3; ++epoch) {
    for (int rank = 0; rank < ranks; ++rank)
      samples(owners[rank], [&](auto& value, auto c, int x, int y, int z, auto direction) {
        // Deliberately inconsistent shared replicas in a separate copy-only
        // characterization prove the operation does not silently reconcile them.
        const double rank_marker = break_shared_coherence ? 1000.0 * rank : 0.0;
        value = static_cast<std::remove_reference_t<decltype(value)>>(ghost(direction) ? -9999.0
            : sample_value(domains[rank], epoch, c, x, y, z) + rank_marker);
      });
    for (int rank = 0; rank < ranks; ++rank)
      success(executions[rank].template execute<Pass::source>(), "source pack failed");
    for (int rank = 0; rank < ranks; ++rank)
      for (std::size_t index = 0; index < 26; ++index) {
        const auto neighbor = domains[rank].field_neighborhood.topology().neighbor_by_index(index);
        if (!neighbor.has_peer() || neighbor.is_self_peer()) continue;
        const auto peer = neighbor.peer_rank();
        const auto outgoing = plans[rank].pattern().direction(neighbor.direction());
        const auto incoming = plans[peer].pattern().direction(
            lpfx::opposite_cartesian_direction(neighbor.direction()));
        const auto bytes = outgoing.template message_bytes<Flow::outbound>();
        require(bytes == incoming.template message_bytes<Flow::inbound>(),
                "nonreciprocal halo message sizes");
        if (bytes) std::memcpy(images[peer].incoming.data() +
            incoming.template image_offset_bytes<Flow::inbound>(),
            images[rank].outgoing.data() + outgoing.template image_offset_bytes<Flow::outbound>(), bytes);
      }
    for (int rank = 0; rank < ranks; ++rank) {
      success(executions[rank].template execute<Pass::overwrite>(), "ghost overwrite failed");
      samples(owners[rank], [&](const auto& value, auto c, int x, int y, int z, auto direction) {
        const auto& domain = domains[rank];
        bool written = !ghost(direction);
        auto source_rank = static_cast<Domain::Rank>(rank);
        if (ghost(direction)) {
          const auto neighbor = domain.field_neighborhood.topology().neighbor(direction);
          written = neighbor.has_peer();
          if (written) source_rank = neighbor.peer_rank();
        }
        const double marker = break_shared_coherence ? 1000.0 * source_rank : 0.0;
        const double expected = written ? sample_value(domain, epoch, c, x, y, z) + marker : -9999.0;
        require(value == expected, "halo differs from global-coordinate oracle or changed physical/shared samples");
      });
    }
    require(budget.used() == prepared_bytes, "exchange allocated during a repeated epoch");
  }
  for (int rank = 0; rank < ranks; ++rank) success(executions[rank].reset(), "exchange reset failed");
  require(budget.used() == 0, "exchange storage did not release its budget");
}

template <class Schema> void device_binding() {
  using Backend = lpfx::device::MockBackend;
  constexpr lpfx::device::Executor<Backend> executor{lpfx::device::MockStream{1}};
  const auto p = parameters(false, false);
  const auto domain = tpic::config::prepare_domain(p.grid, p.topology, 0, 1);
  auto host = make_field<Schema>(domain);
  auto device = tpic::data::upload_field(host, executor);
  const auto plan = tpic::test::field_copy_plan<d::transport::MpiStaged>(device, domain, executor);
  auto binding = d::bind<d::transport::MpiStaged>(plan, device, executor);
  require(binding.valid() && &binding.resource() == &device && &binding.plan() == &plan,
          "public binding does not borrow the uploaded low-level owner and plan");
}

template <class Precision> void precision() {
  using Fields = tpic::data::FieldSchema<Precision>;
  const auto schemas = [&]<class Schema>() {
    exchange_scenario<Schema>(parameters(false, false));
    exchange_scenario<Schema>(parameters(true, false));
    exchange_scenario<Schema>(parameters(false, true));
    exchange_scenario<Schema>(parameters(true, true));
    auto minimum = parameters(false, false);
    minimum.grid.nxc = minimum.grid.nyc = minimum.grid.nzc = 1;
    exchange_scenario<Schema>(minimum);
    device_binding<Schema>();
  };
  schemas.template operator()<typename Fields::E>();
  schemas.template operator()<typename Fields::BCell>();
  schemas.template operator()<typename Fields::BNode>();
  exchange_scenario<typename Fields::BNode>(parameters(true, false), true);
}

void invalid_domains() {
  using Schema = tpic::data::FieldSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>::BCell;
  const auto valid = tpic::config::prepare_domain(parameters(false, false).grid,
                                               parameters(false, false).topology, 0, 1);
  auto owner = make_field<Schema>(valid);
  const auto prepare = [&](const Domain& domain) {
    static_cast<void>(tpic::test::field_copy_plan(owner, domain));
  };
  rejects([&] { prepare(Domain{}); });
  const auto p = parameters(true, true);
  auto bad = tpic::config::prepare_domain(p.grid, p.topology, 0, 12);
  bad.field_neighborhood = {};
  rejects([&] { prepare(bad); });
}

void blocking_operation_conclusion() {
  using Status = lpfx::Status;
  struct Exchange {
    Status accepted = Status::accepted();
    Status waited = Status::success();
    Status concluded = Status::success();
    int begins{}, waits{}, conclusions{};
    Status begin() { ++begins; return accepted; }
    Status wait() { ++waits; return waited; }
    Status conclude() { ++conclusions; return concluded; }
  };
  const auto complete = [](Exchange& exchange) {
    tpic::communication::complete_field_exchange(
        exchange, "test exchange rejected", "test exchange failed");
  };
  Exchange good;
  complete(good);
  require(good.begins == 1 && good.waits == 1 && good.conclusions == 1,
          "blocking exchange did not complete the accepted operation exactly once");
  const auto failure = [&](Exchange& exchange, const char* context,
                           int waits, int conclusions) {
    try { complete(exchange); }
    catch (const std::runtime_error& error) {
      require(std::string(error.what()).starts_with(std::string(context) + ": lpfx="), "blocking exchange lost caller context");
      require(exchange.waits == waits && exchange.conclusions == conclusions,
              "blocking exchange concluded unaccepted or failed live work");
      return;
    }
    throw std::runtime_error("blocking exchange ignored rejected completion");
  };
  Exchange rejected;
  rejected.accepted = Status::failure(lpfx::ErrorCode::invalid_state);
  failure(rejected, "test exchange rejected", 0, 0);
  Exchange no_conclusion;
  no_conclusion.accepted = Status::success();
  failure(no_conclusion, "test exchange rejected", 0, 0);
  Exchange wait_failed;
  wait_failed.waited = Status::failure(lpfx::ErrorCode::backend_failure);
  failure(wait_failed, "test exchange failed", 1, 0);
  Exchange wait_pending;
  wait_pending.waited = Status::pending();
  failure(wait_pending, "test exchange failed", 1, 0);
  Exchange conclude_failed;
  conclude_failed.concluded = Status::failure(lpfx::ErrorCode::backend_failure);
  failure(conclude_failed, "test exchange failed", 1, 1);
}

} // namespace

int main() {
  try {
    precision<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>();
    precision<tpic::config::Precision<tpic::config::StorageSinglePosition, tpic::config::ComputeDouble>>();
    precision<tpic::config::Precision<tpic::config::StorageSingleParticlesMomentsFields, tpic::config::ComputeDouble>>();
    invalid_domains();
    blocking_operation_conclusion();
    std::cout << "TPIC field copy halo plans, host exchange, and device bindings passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
