#ifndef TPIC_SIMULATION_ACTIONS_MAGNETIC_STARTUP_DEVICE_HPP
#define TPIC_SIMULATION_ACTIONS_MAGNETIC_STARTUP_DEVICE_HPP

#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/error_reporting.hpp>
#include <tpic/communication/fields/field_exchange.hpp>
#include <tpic/ecsim/fields/cell_to_node_device.hpp>
#include <tpic/ecsim/fields/field_boundary_device.hpp>

namespace tpic::simulation {

// Cold, rank-uniform startup after every field, species and collective is
// registered. The two halo operations bind these owners and this executor with
// all-26 copy-to-ghost plans; plans, owners and communication outlive the call.
// Errors enter the run's collective fatal path. MPI blocks for both halos;
// NCCL/RCCL return at enqueue, and an all-physical topology has no halo work,
// so the caller synchronizes the executor before publishing C2N.
// A fresh run keeps the analytic wall ghosts of B (walls null); restored B
// closes them, as the advance that wrote it did.
template <class Precision, class CellOwner, class NodeOwner,
          class CellExchange, class NodeExchange, class Backend>
void complete_initial_magnetic_field(
    CellOwner& cell, NodeOwner& node,
    CellExchange& cell_exchange, NodeExchange& node_exchange,
    lpfx::device::Executor<Backend> executor, const ecsim::FieldBoundaryPlans* walls) {
  communication::complete_field_exchange(cell_exchange,
      "TPIC startup: magnetic halo exchange rejected", "TPIC startup: magnetic halo exchange failed");
  if (walls)
    lpfx::check_error(ecsim::enqueue_wall_ghosts(cell, *walls, executor),
                      "TPIC startup: magnetic wall closure failed");
  lpfx::check_error(ecsim::enqueue_magnetic_cell_to_node<Precision>(cell, node, nullptr, executor),
                    "TPIC startup: magnetic C2N launch failed");
  communication::complete_field_exchange(node_exchange,
      "TPIC startup: magnetic halo exchange rejected", "TPIC startup: magnetic halo exchange failed");
}

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_ACTIONS_MAGNETIC_STARTUP_DEVICE_HPP
