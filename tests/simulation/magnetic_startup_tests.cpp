// Host magnetic startup: B-cell halo exchange, cell-to-node average and B-node
// halo exchange through LPFx's prepared host schedule, compared exactly with
// an oracle written from the contracts over 3 grids x 8 periodicity masks x 2
// datasets. Oracle contracts: a halo region with any non-periodic missing
// neighbour stays unchanged (cells keep their samples, nodes stay zero);
// otherwise it copies the periodic image. A node is the left-associated,
// x-fastest sum of its eight cells, each load promoted first, times 1/8.
#include <tpic/config/simulation_parameters.hpp>
#include <tpic/config/domain.hpp>
#include "../communication/field_plan_fixture.hpp"
#include <tpic/config/precision.hpp>
#include <tpic/data/fields/field_schema.hpp>
#include <tpic/ecsim/fields/cell_to_node.hpp>

#include <lpfx/distributed/field/host.hpp>
#include <lpfx/execution/host/serial.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using Fields = tpic::data::FieldSchema<tpic::config::Precision<tpic::config::StorageDouble, tpic::config::ComputeDouble>>;
using Index = std::array<int, 3>;
std::size_t fixtures = 0;
std::size_t samples = 0;
std::size_t retained_physical_cells = 0;
std::size_t zero_physical_nodes = 0;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void success(lpfx::ErrorCode error, const char* message) {
  require(error == lpfx::ErrorCode::success, message);
}

double sample(int component, Index index, bool cancellation) {
  const auto [i, j, k] = index;
  if (cancellation) {
    // Alternating magnitudes detect a reordered eight-point reduction. Values
    // are exactly representable, so exact comparison is meaningful.
    constexpr std::array<double, 16> values{
        0x1p60, 1.0, -0x1p60, 3.0, -5.0, 0x1p58, 7.0, -0x1p58,
        11.0, -13.0, 17.0, 0x1p54, -19.0, -0x1p54, 23.0, -29.0};
    const auto offset = static_cast<std::size_t>(i + 1 + 3 * (j + 1) +
                                                7 * (k + 1) + 5 * component);
    return values[offset % values.size()];
  }
  // A nonperiodic polynomial makes periodic wrapping observable; physical
  // ghosts must keep their original unwrapped samples.
  const double x = i + .5, y = j + .5, z = k + .5;
  return 1000.0 * component + x * x + 3.0 * y * y + 5.0 * z * z +
         7.0 * x * y + 11.0 * y * z + 13.0 * x * z + .125;
}

struct Fixture {
  Index cells;
  std::array<bool, 3> periodic;
  bool cancellation;
};

// Source of a halo entry on one rank: the periodic image, or nothing when an
// out-of-range axis is not periodic.
bool periodic_source(const Fixture& f, Index& index, int extent_extra) {
  bool available = true;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    const int n = f.cells[axis];
    const int extent = n + extent_extra;
    if (index[axis] < 0 || index[axis] >= extent) {
      if (!f.periodic[axis]) available = false;
      else index[axis] += index[axis] < 0 ? n : -n;
    }
  }
  return available;
}

double expected_cell(const Fixture& f, int component, Index index) {
  auto source = index;
  return sample(component, periodic_source(f, source, 0) ? source : index, f.cancellation);
}

double expected_node(const Fixture& f, int component, Index index) {
  if (!periodic_source(f, index, 1)) return 0.0;
  const auto cell = [&](int di, int dj, int dk) {
    return static_cast<double>(expected_cell(f, component,
        {index[0] + di, index[1] + dj, index[2] + dk}));
  };
  double sum = cell(-1, -1, -1);
  sum += cell(0, -1, -1);
  sum += cell(-1, 0, -1);
  sum += cell(0, 0, -1);
  sum += cell(-1, -1, 0);
  sum += cell(0, -1, 0);
  sum += cell(-1, 0, 0);
  sum += cell(0, 0, 0);
  return sum * 0.125;
}

bool outside_physical_boundary(Index index, const Index& extents,
                               const std::array<bool, 3>& periodic) {
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (!periodic[axis] && (index[axis] < 0 || index[axis] >= extents[axis]))
      return true;
  return false;
}

template <class Schema>
void exchange_self(lpfx::field::host<Schema>& owner,
                   const tpic::config::PreparedDomain& domain) {
  auto plan = tpic::test::field_copy_plan(owner, domain);
  lpfx::HostMemoryBudget budget;
  require(budget.try_set_limit(1024 * 1024) && budget.seal(), "halo budget");
  using Plan = decltype(plan);
  // Test-only execution of LPFx's real prepared schedule. The production
  // application uses inferred requests in its complete registration.
  lpfx::distributed::detail::HostFieldExecution<typename Plan::pattern_type,
      lpfx::field::host<Schema>> execution;
  lpfx::distributed::detail::FieldExecutionImages images;
  success(execution.prepare(budget, plan, owner, lpfx::SerialExecutor{}, images),
          "prepare LPFx self exchange");
  require(images.incoming.empty() && images.outgoing.empty(),
          "one-rank fixture unexpectedly needs remote transport");
  success(execution.template execute<lpfx::FieldSemanticPass::source>(),
          "execute LPFx source pass");
  success(execution.template execute<lpfx::FieldSemanticPass::overwrite>(),
          "execute LPFx overwrite pass");
}

template <class Schema>
auto make_field(const Index& cells) {
  auto made = lpfx::field::make<Schema>(lpfx::field::options{
      .cells = {cells[0], cells[1], cells[2]}, .halo = 1});
  require(made.has_value(), "LPFx field allocation");
  return std::move(made).value();
}

template <class View>
void compare_component(View actual, const Fixture& f, int component, bool nodal,
                       const std::string& label) {
  const Index extents{f.cells[0] + nodal, f.cells[1] + nodal, f.cells[2] + nodal};
  for (int k = -1; k <= extents[2]; ++k)
    for (int j = -1; j <= extents[1]; ++j)
      for (int i = -1; i <= extents[0]; ++i) {
        const Index index{i, j, k};
        const double value = actual(i, j, k);
        const double oracle = nodal ? expected_node(f, component, index)
                                    : expected_cell(f, component, index);
        if (!std::isfinite(value) || value != oracle) {
          std::ostringstream message;
          message.precision(17);
          message << label << (nodal ? " Bnode" : " Bcell") << " component " << component
                  << " at (" << i << ", " << j << ", " << k << "): TPIC=" << value
                  << ", oracle=" << oracle;
          throw std::runtime_error(message.str());
        }
        ++samples;
        if (outside_physical_boundary(index, extents, f.periodic)) {
          if (nodal) {
            require(value == 0.0, "startup changed a physical Bnode ghost");
            ++zero_physical_nodes;
          } else {
            require(value == sample(component, index, f.cancellation),
                    "startup replaced an analytic physical Bcell ghost");
            ++retained_physical_cells;
          }
        }
      }
}

void run(const Fixture& f, unsigned periodic_mask) {
  tpic::config::SimulationParameters parameters;
  parameters.grid.nxc = f.cells[0];
  parameters.grid.nyc = f.cells[1];
  parameters.grid.nzc = f.cells[2];
  parameters.topology.periodic_x = f.periodic[0];
  parameters.topology.periodic_y = f.periodic[1];
  parameters.topology.periodic_z = f.periodic[2];
  // Deliberately disagree with particle topology: this is a FIELD exchange.
  parameters.topology.periodic_x_particles = !f.periodic[0];
  parameters.topology.periodic_y_particles = !f.periodic[1];
  parameters.topology.periodic_z_particles = !f.periodic[2];
  const auto domain = tpic::config::prepare_domain(parameters.grid, parameters.topology, 0, 1);
  auto cell = make_field<Fields::BCell>(f.cells);
  auto node = make_field<Fields::BNode>(f.cells);
  const auto populate = [&]<std::size_t Component>() {
    const auto destination = cell.view().template component<Component>();
    for (int k = -1; k <= f.cells[2]; ++k)
      for (int j = -1; j <= f.cells[1]; ++j)
        for (int i = -1; i <= f.cells[0]; ++i)
          destination(i, j, k) = sample(static_cast<int>(Component), {i, j, k}, f.cancellation);
  };
  populate.template operator()<0>();
  populate.template operator()<1>();
  populate.template operator()<2>();

  exchange_self(cell, domain);
  const auto derive = [&]<std::size_t Component>() {
    const auto source = cell.view().template component<Component>();
    const auto destination = node.view().template component<Component>();
    for (int k = 0; k <= f.cells[2]; ++k)
      for (int j = 0; j <= f.cells[1]; ++j)
        for (int i = 0; i <= f.cells[0]; ++i)
          destination(i, j, k) = tpic::ecsim::cell_to_node_average(source, i, j, k);
  };
  derive.template operator()<0>();
  derive.template operator()<1>();
  derive.template operator()<2>();
  exchange_self(node, domain);

  const std::string label = "cells=" + std::to_string(f.cells[0]) + "," +
      std::to_string(f.cells[1]) + "," + std::to_string(f.cells[2]) +
      " periodic-mask=" + std::to_string(periodic_mask) +
      (f.cancellation ? " cancellation" : " polynomial");
  const auto check = [&]<std::size_t Component>() {
    compare_component(cell.view().template component<Component>(), f,
                      static_cast<int>(Component), false, label);
    compare_component(node.view().template component<Component>(), f,
                      static_cast<int>(Component), true, label);
  };
  check.template operator()<0>();
  check.template operator()<1>();
  check.template operator()<2>();
  ++fixtures;
}

} // namespace

int main() {
  try {
    for (const auto cells : {Index{2, 2, 2}, Index{2, 3, 4}, Index{5, 4, 3}})
      for (unsigned mask = 0; mask < 8; ++mask)
        for (const bool cancellation : {false, true})
          run({cells, {bool(mask & 1), bool(mask & 2), bool(mask & 4)}, cancellation}, mask);
    require(fixtures == 48 && retained_physical_cells != 0 && zero_physical_nodes != 0,
            "physical ghost checks were not exercised");
    std::cout << fixtures << " startup fixtures and " << samples
              << " scalar samples matched the oracle exactly; "
              << retained_physical_cells << " analytic physical cell ghosts and "
              << zero_physical_nodes << " zero physical node ghosts checked\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
