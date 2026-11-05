#ifndef TPIC_SIMULATION_EXECUTION_ACTIVE_BACKEND_DEVICE_HPP
#define TPIC_SIMULATION_EXECUTION_ACTIVE_BACKEND_DEVICE_HPP

// The device backend of this build, named once for the device classes of
// simulation/. Host-testable utilities (execution streams, stage timing)
// keep their Backend parameter and are instantiated with this backend.
#include <lpfx/backend/cuda_hip/kernel.hpp>
#include <lpfx/backend/backend.hpp>
#include <lpfx/backend/handles.hpp>

namespace tpic::simulation {

using Backend = lpfx::device::ActiveBackend;
using Executor = lpfx::device::Executor<Backend>;
using Event = lpfx::device::DeviceEvent<Backend>;

} // namespace tpic::simulation

#endif // TPIC_SIMULATION_EXECUTION_ACTIVE_BACKEND_DEVICE_HPP
