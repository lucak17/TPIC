#ifndef TPIC_TESTS_COMMUNICATION_FIELD_PLAN_FIXTURE_HPP
#define TPIC_TESTS_COMMUNICATION_FIELD_PLAN_FIXTURE_HPP

#include <lpfx/api/distributed/requests.hpp>
#include <lpfx/distributed/transport/routes.hpp>
#include <tpic/config/domain.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tpic::test {

// The host oracle simulates several ranks in one process. Infer the production
// field definition, then prepare its geometry without an MPI registration.
// Native tests use the complete public object preparation transaction.
template <class Route = lpfx::distributed::transport::MpiHost, class Owner,
          class Executor = lpfx::SerialExecutor>
auto field_copy_plan(Owner& owner, const config::PreparedDomain& domain, Executor executor = {}) {
  namespace d = lpfx::distributed;
  auto request = d::copy_ghosts<"fixture">(owner,
      d::fields(domain.field_neighborhood, d::uniform_halos).template on<Route>(executor));
  using Request = decltype(request);
  typename Request::plan_type plan;
  const auto geometry = [&](const auto& partition, auto& result) noexcept {
    result.cell_extents = partition.local_cells();
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((result.component_halos[I] = owner.template component_halo<I>()), ...);
    }(std::make_index_sequence<Owner::component_count>{});
    return lpfx::ErrorCode::success;
  };
  if (d::try_prepare_field_plan(typename Request::definition_type{},
          domain.field_neighborhood, geometry, plan) != lpfx::ErrorCode::success)
    throw std::invalid_argument("TPIC field fixture: invalid domain geometry");
  return plan;
}

} // namespace tpic::test

#endif // TPIC_TESTS_COMMUNICATION_FIELD_PLAN_FIXTURE_HPP
