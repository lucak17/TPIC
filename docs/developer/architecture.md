# Architecture and ownership

TPIC defines the ECSIM method, its data layouts and its execution order. LPFx,
bundled in the tree, supplies the typed device owners, streams and events, host
workers, fences and channels, registered communication and particle migration.
The [engineering standards](standards.md) state the rules this structure
follows, the [header map](headers.md) lists every file, and the
[kernel inventory](kernels.md) lists every device kernel.

## Directory layout

| Path | Holds |
| --- | --- |
| `include/tpic/<module>/` | Every production header, by module and topic. |
| `src/<module>/` | The translation units of a module, named after the header they implement. |
| `include/lpfx/` | The LPFx headers, an unchanged copy of the LPFx revision recorded in `cmake/dependencies.json`. |
| `cmake/lpfx/` | The LPFx build support, pulled in by `add_subdirectory(cmake/lpfx)` in `CMakeLists.txt`. `CMakeLists.txt`, `LPFxOptions.cmake` and `LPFxTargets.cmake` are adapted to this tree and are not in the lock; the other files are unchanged LPFx files. |
| `cmake/` | Build options (`TpicOptions.cmake`), the device targets (`TpicDevice.cmake`), the test-suite functions (`TpicTests.cmake`) and the LPFx lock `dependencies.json`. |
| `tests/<module>/` | Host tests (`*_tests.cpp`), device tests (`*_device.cpp`), golden tests and integration drivers; `tests/compile/` holds the structure checks, `tests/support/` the shared test helpers, and `tests/data/golden/` the golden tables. |
| `tests/support/device/mock_backend.hpp` | The LPFx mock device backend of the host tests, an LPFx file recorded in the lock. |
| `tools/` | Benchmark and run-comparison drivers, the explicit electrostatic PIC run, the dependency inventory, the post-processing scripts and the `tpic_tools` Python package. |
| `input/` | Input decks. |
| `performance/scaling/` | The cluster scaling decks and scripts. |

### The bundled LPFx

`cmake/dependencies.json` records the LPFx revision and the SHA-256 of every
copied file: the files of `include/lpfx/`, the unchanged files of `cmake/lpfx/`
and `tests/support/device/mock_backend.hpp`. `python3
tools/dependency_inventory.py --check` reports every copied file that differs
from its record, is missing or is not recorded, and needs no LPFx checkout. `python3
tools/dependency_inventory.py --refresh --lpfx <checkout>`, with the checkout
named explicitly, runs once its files are copied into the tree: it
compares each copied file with the file committed at the checkout's `HEAD`,
requires every `include/lpfx/` file of that commit to be present, and only
then writes the lock with that revision. It copies nothing itself. The rule
that the copy changes only this way is in the [standards](standards.md#the-lpfx-copy).

### Modules

| Module | Responsibility | CMake target |
| --- | --- | --- |
| `device` | Block sizes, thread indexing, the cross-block reducer and the device validation flag | `TPIC::model` (headers) |
| `data` | Field and particle schemas and storage, the compact mass-matrix layout, neutral output records | `TPIC::model` |
| `config` | Precision structs, species limits, input-deck parsing, run-option enumerations, domain, physics, particle and output preparation | `TPIC::config` |
| `communication` | Field exchanges, the compact mass halo and particle migration over LPFx, and the transport routes | `TPIC::communication` |
| `ecsim` | Deposition, field operators, GMRES, mass matrices, Maxwell and the field update, particle algorithms | `TPIC::ecsim` |
| `initialization` | Simulation cases, analytic fields, host field images, particle loading and velocity centering | `TPIC::initialization`, `TPIC::device_initialization` |
| `diagnostics` | Scalar reductions, the Gauss-law residual, output-moment deposition and the asynchronous field and particle snapshots | header-only, used by `TPIC::simulation` |
| `io` | Output engines (VTK, ADIOS2, HDF5), the asynchronous disk writer and the scalar CSV | `TPIC::io` |
| `simulation` | Planning, execution resources, timing, species state, transport, the actions, reporting and the driver with the runtime | `TPIC::simulation_support` (host), `TPIC::simulation` (device) |
| `app` | Command line and entry point (`src/app/main.cpp`) | `tpic` |

Headers follow a layer order: `device`, `data` and `config` at the bottom,
`communication`, then `ecsim`, then `initialization`, `diagnostics` and `io`,
then `simulation` and finally the application. A header includes only its own
tree and the layers below it that its row lists; the exact table, with its two
file-level exceptions, is in [standards](standards.md#layer-table) and is
checked by `tests/compile/module_dependencies.py` on `include/tpic/` and `src/`.

## From the command line to the simulation

`main` (`src/app/main.cpp`) parses the command into `simulation::InputOptions`
and a `RunMode`. `--check-input` prepares the host input (`prepare_input`) and
prints its report without initializing MPI and without a GPU. `--initialize`
and `--run` call:

1. `simulation::run_distributed` (`src/simulation/run_distributed.cpp`) owns the
   LPFx runtime: it bootstraps MPI and the device, agrees on the command
   options and the input bytes across ranks (`agree`), parses the agreed bytes,
   builds the `RunPlan`, agrees on matrix-core support, and gathers and prints
   the memory plan before any owner exists.
2. `dispatch_precision` (`config/precision.hpp`) turns the `PrecisionSelection`
   into the pair type `Precision<Storage, Compute>` of one storage struct and
   one compute struct.
3. `run_simulation<Precision>` (`simulation/driver/simulation_device.hpp`) is
   instantiated once per compiled pair by `src/simulation/simulation_dispatch.cpp`,
   which the build compiles once for every pair. It selects the particle order
   (`Partial`) and the `MassDepositionKernel`, constructs
   `Simulation<Precision, Partial, Kernel>` and its `Pipeline` and calls
   `Pipeline::execute`, which returns the `RunResult`.

Each switch runs once per run; every compiled pair has its own translation unit
with four `Simulation` instantiations (order and kernel selections; two when the
pair's mass compute is single, which has no matrix kernel). Below this point
nothing selects a precision, order or kernel at run time.

The species count is a run-time value. Every `Simulation` is compiled for the
capacity `config::max_real_species` (CMake option `TPIC_MAX_SPECIES`, default
10), because each species' migration is a distinct LPFx operation type, and a
run uses the `ns` species of its input. A species of the capacity beyond `ns`
owns no particle storage, classification slots, sort workspace, validation
flag, stream, event, worker thread or timer; its migration is registered with an
empty owner and no traffic, and is never begun. `StreamExecutors`,
`Pipeline` and `DepositionLedger` take the species count of the run at
construction and bound every species loop by it.

With `--restart`, `run_distributed` has the root find the newest complete
checkpoint (`io::find_restart_checkpoint`) and broadcasts its location and
header before the plan; `RunPlan::prepare` then rejects every change the
checkpoint cannot continue, reads which pieces of the checkpoint hold the
rank's cells (`io::prepare_restart`) and sizes the particle plan from their
records. After admission, and before any rank can write a checkpoint, the root
sets aside the checkpoints the run would mix its own with
(`io::prepare_checkpoint_directory`). When the run advances, the root creates
`SaveDirName` after admission; no engine creates it.

`RunPlan::prepare` is the only factory of a run plan: it derives every record
once from the parameters and options and freezes them before owners are
allocated. Unsupported requests, the field-boundary rules of `--initialize` and
`--run` and the output backends of an advancing run are checked only there;
everything that takes `const RunPlan&` trusts it. `prepare_input` resolves the
option values, including the Maxwell solve schedule (`full`, `split` or
`smoothed`, stored in `PreparedInput::maxwell_schedule`). The data builders
`prepare_particle_cell_map` and `prepare_mass_matrix_layout` copy the counts
that `config::prepare_domain` checked. Tests build records by hand through
`tests/support/test_builders.hpp` (`fill_domain_counts`).

## Phases of `simulation/`

`simulation/` is cut by phase of the run; a directory includes only itself and
the directories before it in this order.

| Directory | Holds |
| --- | --- |
| `planning/` | `InputOptions`, `RunMode`, `PreparedInput`, `RunPlan`: host preparation, compiled into `TPIC::simulation_support`. |
| `execution/` | `ExecutionLayout` (every cardinality), the epoch, the deposition vocabulary, `ExecutionStreams` and `StreamExecutors`, `DepositionLedger` and the backend aliases. |
| `timing/` | `TimingRegion` table, `StageTiming` and the running statistics. |
| `state/` | `Species<Precision, Partial>` and `visit_species`. |
| `transport/` | `TransportRegistry`. |
| `actions/` | `MoverActions`, `FieldAdvanceActions`, `LocalSpeciesActions`, `DepositionActions`, `FieldSolveActions`, `DiagnosticsActions`, `OutputActions` and `complete_initial_magnetic_field`: the device and native work, without events, fences, timing or validations, except that `OutputActions` keeps the epoch it captures and orders and waits for its own snapshot transfers and checkpoint copy, and `LocalSpeciesActions::retire_after` hands the mover's event to the species sorter. |
| `reporting/` | `RunReporter`, the startup report with `RunResult`, and the memory estimate. |
| `driver/` | `run_distributed`, `Simulation` (the physics owner) and its dispatch, and `Pipeline` (the runtime: every thread program, wait, record, fence, validation and timing scope). |

The structure classes are in `tpic::simulation`; `detail` holds file-private
helpers only.

## Runtime lifetime

`run_simulation` constructs, and destroys in reverse:

1. The `ExecutionStreams` registry, which admits the borrowed control executor.
2. `StreamExecutors<Backend, Ns>`, which creates every other stream in role order.
3. `Simulation`, the physics owner. In declaration order: the migration
   geometry, the field image, the magnetic node field, the current and the
   compact mass matrix, the species pack, velocity centering, the field-solve,
   diagnostics and output actions, the `TransportRegistry`, the mover and
   local-species actions. Built late in `prepare()`, behind `std::unique_ptr`:
   the field-advance and deposition actions, which resolve registered operations.
4. `Pipeline`, the runtime: what main publishes, one object per worker (its
   thread, timer and the events, fences, validations and handoffs it
   publishes), main's timer and the output and checkpoint writers.

The `Simulation` constructor uploads the initial field image (E before B),
allocates the zero node B and current and initialises each species in ascending
index on the control stream (this fixes the loading and random-number order).
The `Pipeline` constructor creates the events of the run's species, checks the
stream count against `ExecutionLayout::streams` and seals the registry.
`Pipeline::execute` then runs:

- `Simulation::prepare`: `TransportRegistry::prepare` registers every operation
  from one declaration recipe, then the two late actions are built.
- `initialize`: synchronises the magnetic field, centers velocities, migrates
  and orders the particles through `LocalSpeciesActions`, checks the
  population, starts the advancing workers and runs the bootstrap advance, which
  deposits the moments of the initial state; `state_ready_event_` is recorded.
  The two solve workers are started after the memory report.
- `run_cycles` (an advancing run only): the cycle loop (below), then one final
  solve epoch without the field solve.
- `close`: drain every stream, retire each species' ordering in index order,
  close and join the workers, close the transport.

A failure after communication is registered never unwinds: `execute`, worker
setup and every worker failure path end in `Simulation::abort`.

The initial node-centred B is not sampled from the analytic profile:
`complete_initial_magnetic_field` (`simulation/actions/magnetic_startup_device.hpp`)
exchanges the cell B halos, interpolates the eight surrounding cell samples to
each node and exchanges the node halos.

A restart starts from the state of its checkpoint, cycle n: the `Simulation`
constructor overwrites the interiors of the field image with the checkpoint's
E and B and loads each species' records through `make_restored_particles`
(`simulation/state/species_device.hpp`) instead of the analytic load. Startup
closes the B wall ghosts before cell-to-node, as the advance that wrote B^n
did, skips velocity centering, and starts the bootstrap and the cycle loop at
n; `RunPlan::first_cycle` is the warm-up cycle of the timing tables and the
reporter.

The particle plan (`config/particle_plan.hpp`) marks a fixed local population
when there is one rank, no test-particle source and only periodic or mirror
particle faces. Such a population uses exactly its active count as initial
capacity; with full or unsorted ordering it also bypasses migration
(`bypass_local_migration`): startup uses a boundary-only pass and the timestep
the local mover. Partial ordering, the default, keeps the migration.

## Streams, events and workers

`ExecutionLayout` (`simulation/execution/execution_layout.hpp`) holds every
cardinality of a run; the owners and the memory estimate both use it, and the
owners tie their enumerations and members to it by `static_assert`.

**Streams.** `StreamExecutors` in
`simulation/execution/execution_streams.hpp` creates the roles in a fixed order.
`ExecutionStreams` accepts only an explicit nonblocking stream on the selected
device and rejects null and default streams; after `seal()` it performs no
stream query.

| Order | Role | Used for |
| --- | --- | --- |
| 1 | `control` | The borrowed runtime executor: startup, the main thread's joins, the committed-state event. |
| 2 | `current` | Current clear, deposition and assembly. |
| 3 | `mass` | Mass-matrix clear, deposition and halo. |
| 4 | `diagnostics` | Scalar diagnostics, first lane; the Gauss-law residual. |
| 5 | `diagnostics_helper` | Scalar diagnostics, second lane. |
| 6 | `solver` | Right-hand side, GMRES, particle field. |
| 7 | `solver_interior` | Interior curl of the split Maxwell apply. |
| 8 | `solver_mass` | Mass-matrix apply of the split Maxwell apply. |
| 9 | `route` | Particle migration routing; the only high-priority stream. |
| 10 | `move` | Particle mover. |
| 11 | `field_update` | Next fields, the next-cell halo, node closure and commit. |
| 12 | `output_capture` | Snapshot capture; created only when an advancing run enables field or particle output or checkpoints (`snapshots_enabled`), otherwise an alias of `diagnostics`. |
| 13 | `output_transfer` | Snapshot download; same condition, otherwise an alias of `control`. |
| 14 | `local`, one per species | Migration steps of the species. |
| 15 | `sort`, one per species | Sort of the species; an alias of its `local` stream when the order is unsorted. |

`run_simulation` builds this stream table once. The `Simulation` constructor,
`prepare()` and `TransportRegistry::describe` bind named entries to owners,
actions and registered operations; `Pipeline` passes the per-call entries and
places every scope, validation and event on the entry of the work it orders.

**Workers.** `Pipeline` runs `ns + 6` persistent host workers in two
admissions per cycle. Each is one object of `Pipeline` (`mover_`,
`field_advance_`, `routing_`, `deposition_`, `local_species_[s]`,
`field_solve_`, `diagnostics_output_`): a `Pipeline::Worker` base with the
name, the species, the `done_fence`, the timer and its thread, extended by the
events, fences, channels, validations and handoffs that worker publishes. The
thread is an `lpfx::HostWorker<Epoch>` (one thread, a one-epoch mailbox, setup
once) that `Pipeline::start(worker, program)` starts: for every epoch
`Worker::submit` sends, `Pipeline::run` runs the worker's program
(`run_mover`, `run_local_species`, ...) inside its timing cycle, then signals
`done_fence`. The worker-to-program mapping is written where the workers are
started. Submission and joins loop over `advance_workers()` and
`all_workers()`. The dependency map at the top of `pipeline.hpp` lists every
worker's streams, waits and publications.

| Admission | Workers | Count |
| --- | --- | --- |
| Advance (`advance_workers()`) | `mover`, `field_advance`, `routing`, `deposition`, then `local_species[s]` | `ns + 4` |
| Solve | `field_solve`, `diagnostics_output` | 2 |
| `io::AsyncOutputWriter` | The output and the checkpoint disk writers, owned by `Pipeline` | 1 each, started with the solve workers when configured |

Every thread's setup, the writer threads' included, selects the device
(`Pipeline::setup_thread`); main opens the scalar file before it starts the
diagnostics-output worker that writes it. A setup failure terminates the
process through `Simulation::abort`, so there is no setup handshake. The printed
timing tables name the workers; the profile zones are `TPIC::<action or step>`.
Host fences (`lpfx::HostFence`) order what the threads may read. The threads
borrow each other's worker members, so `close` and the `Pipeline` destructor
close and join every worker thread before any member is destroyed. A dependency
polling loop yields every 64 unsuccessful polls (`yield_every_64`); the timing
collection yields on every poll.

**Events.** Device events order what the GPU reads and writes; they are created,
recorded and waited through the checked `lpfx::device::create_event`,
`record_event` and `wait_event` of `lpfx/backend/error_reporting.hpp`.

| Event | Recorded | Waited | Orders |
| --- | --- | --- | --- |
| `state_ready_event_` | On `control` after startup and after each advance join. | On `solver`; on `diagnostics`, `diagnostics_helper` and `output_capture`. | The solve reads the committed state. |
| `field_solve.particle_field_ready_event` | On `solver` by the field-solve worker after the particle field. | Once on `move`, before the species loop; on `field_update`. | Mover and field advance read the solved field. |
| `field_solve.deposition_ready_event` | On `solver` after `particle_field_ready_event`; on `control` at bootstrap. | By the moment clear on `current` and `mass`. | Moments are cleared only after the solve consumed them. |
| `field_advance.next_magnetic_event` | On `field_update`. | By every deposit on its quantity stream, except at bootstrap. | Deposits read the next magnetic field. |
| `field_advance.committed_event` | On `field_update`. | On `control` by the main thread. | The committed state. |
| `mover.moved_event[s]` | On `move`, in species order. | By the field commit, last species only (FIFO on `move` implies the others); by the local worker of `s` on `local`. | Commit and migration follow the mover. |
| `local_species[s].owner_ready_event` | By the local worker on the `sort` or `local` stream. | By completed-path deposits; on `control` in species order. | Deposits and the commit join read the settled owner. |
| `deposition.resident_done_event[s][q]`, `local_species[s].resident_release_done_event` | Partial order only, after a resident deposit and after the readers are sealed. | By the local worker of `s`. | Resident readers release before the owner is finalised. |
| `deposition.mass_interior_event` | On `mass` after the additive mass scatter; its fence follows the native call of the ghost phase. | On `solver` before the solve. | The solve reads the interior mass records while the ghost phase is in flight. |
| `deposition.assembled_event[q]` | After assembly, on the quantity stream. | Current on `solver`; mass by the solve's first apply before its first read of mass ghosts, on `solver_mass` (on `solver` when smoothing); on `control` at startup when the run does not advance or has no cycle. | The solve reads assembled moments. |
| `diagnostics_output.sources_read_event` and the snapshot source events | By the diagnostics-output worker on epochs that read the state (scalars or a Gauss sample), and by the snapshots. | On `field_update`, then `move`, before the next advance; on `output_capture` before a Gauss residual is packed. | Field advance and mover do not overwrite what diagnostics and output still read; the capture packs a finished residual. |

No event orders the next mover after the deposits: every mover waits
`field_solve.particle_field_ready_event`, recorded after the solve waited both
`assembled_event`, each recorded after every deposit of its quantity.

The event count printed at startup is `ExecutionLayout::dependency_events`; the
validation flags (one per species plus `ExecutionLayout::shared_validations`)
are cleared and captured on the producing stream.

## One timestep

At cycle `n` the committed state is `E^n`, `B^n`, the deposition positions and
the velocities `v^n`. The mover finalises the velocity and then advances the
position once; there is no leading drift and no stored per-particle response
matrix. Startup deposits the current and mass for the first cycle, and each
later advance prepares them for the next solve. The main thread runs, per cycle:

1. `start_solve` submits the two solve workers. The field-solve worker
   waits `state_ready_event_`, prepares the right-hand side, joins the
   assembled current, finishes the right-hand side (here when not smoothing),
   joins the mass interior (and finishes the right-hand side here when
   smoothing), solves with `FieldSolveActions::host_blocking_solve` (a switch
   over the three schedules), whose first apply joins the assembled mass before
   it reads the mass ghosts, prepares the particle field and records
   `particle_field_ready_event` then `deposition_ready_event`. The
   diagnostics-output worker waits `state_ready_event_` on its three streams,
   launches the scalar diagnostics, the reads of a Gauss sample and the
   snapshot capture, and after the solver's native interval collects the
   diagnostics, finishes a Gauss sample begun at the preceding solve before
   completing the charge of one begun at this solve, and finishes the capture. While the
   solve runs, the main thread joins the preceding advance, whose assembly the
   solve waits for.
2. `host_wait_solved` waits for the solve, then the main thread enqueues the
   source-release waits on `field_update` and `move`.
3. `start_advance` admits the next epoch and submits the `ns + 4` advancing
   workers: the field-advance worker advances E and B, publishes the first
   native submission of the cell halo, closes the cells, interpolates the
   nodes, records `next_magnetic_event` and commits after all movers are
   submitted; the mover enqueues each species on `move` in index order; each
   local worker migrates and orders its species; the routing worker progresses
   the migration routes in species order; the deposition worker clears,
   deposits in `DepositionLedger` order and assembles current, then mass.
4. `host_wait_advanced` joins the commit and the settled owners on `control`,
   and `state_ready_event_` is recorded.
5. `host_blocking_finish_solve`: on snapshot cycles that publish, after the
   output piece, the advance's host epoch and a host barrier; then the
   diagnostics-output worker's completion.

Native communication follows one rank-common order in each cycle: deposition
assembly (current, then both mass phases; the solve may start before the ghost
phase completes), the Maxwell solve, the diagnostics collective, the Gauss
collective of a finishing sample and the charge exchanges of a beginning one,
the output-moment halos, then the field advance and particle routing of the next
advance. Local readiness selects numerical work only; it never changes this
order. Partial order is a sorted resident prefix followed by the unsorted
arrivals of the last migration; the mover makes the bins stale, so every reader
of an ordering receipt retires before the next compaction, capacity growth or
sort. The final advance builds no moments, and the final solve epoch runs
without the field solve.

`host_blocking_timing_boundary` is the only place that drains every stream
inside the loop: at the end of the first cycle, after its report (without a
drain) and after the last cycle.

## Communication

`TransportRegistry<Precision, Ns, Partial>`
(`simulation/transport/transport_registry_device.hpp`) owns the LPFx
communication object, the mass-halo storage and the output writer's
`lpfx::distributed::MpiFileWriter`, which `prepare` opens and `close` frees
after the workers are joined. One declaration recipe registers the fixed
operations (`B.cell`, `B.node`, `J.add`, `J.ghosts`, `centering`, the
`solver.*` fields, `B.next.cell`, `gmres`, `diagnostics`, the `gauss.*` charge
exchanges and `gauss`, `output.moments`, `mass.add`, `mass.ghosts`) and one
particle migration per species, each bound to its owner, route and stream.
Consumers reach them through `TransportRegistry::get<Key>()`.

In the MPI build every operation except `gmres` binds to `DeviceRoute`
(`MpiDirect<StreamOrderedMpiOrdering>`); `gmres` binds to `GmresRoute`
(`MpiStaged`, pinned host staging). NCCL and RCCL builds bind both aliases to
`StreamOrderedCompletion<Nccl>` (or `<Rccl>`). The aliases are defined in
`communication/transport_routes.hpp`.

On the NCCL/RCCL route an exchange or collective completes once its work is
enqueued on the bound executor: `wait()` and `progress()` return without a host
wait, and the `complete_*` helpers return at enqueue. On the MPI route the host
waits for the pack and the MPI requests; a field's unpack is then only enqueued
on the bound executor, and no stream synchronize follows the receives. Every
consumer therefore runs on the operation's executor or waits on a device event
recorded there after `conclude`; a host read joins that stream first. Hand-offs
between operations on different executors rely on NCCL (RCCL) serializing one
communicator's operations in submission order. Particle migration completes as
on the bare vendor route: the host observes each round.

The compact mass halo (`communication/mass_halo/`) is TPIC's own message:
`MassHaloPlan::prepare` fixes the directions and stencils of both phases once;
`MassHaloStorage` packs (`enqueue_pack`) and scatters (`enqueue_unpack`) them
in ascending canonical direction order.

## Species pack

`Species<Precision, Partial>` (`simulation/state/species_device.hpp`) holds one
species' particle owner, classification slots and its ordering owner
(`PartialParticleSort` or `FullParticleSort`). The species index is not part of
the type: the pack is `TransportRegistry::SpeciesPack`, a
`std::array<Species<Precision, Partial>, Ns>` of the compiled capacity, of which
the run initializes its first `ns` entries. Only the migration request, which
LPFx keys by species, takes the index as a template argument
(`Species::migration_request<Index, Route>`). The deposition path indexes the
array directly; the mover and the local-species actions reach the typed
migration through `visit_species` (`simulation/state/species_device.hpp`), a
compare chain over at most `config::max_real_species` entries.

## Backend and launches

The device classes of `simulation/` name the build's backend once, in
`simulation/execution/active_backend_device.hpp`, and every kernel is launched
through LPFx from an enqueue function or an owner class, as the
[kernel inventory](kernels.md) and the [standards](standards.md#enqueue-functions-and-launches)
state.

## Ownership conventions

Use plain members for always-present values and resources whose construction
fits their enclosing owner's lifetime. An `lpfx::HostWorker` is a plain member
whose thread `start()` launches. Use `std::unique_ptr` with `std::make_unique`
for polymorphic engines and for persistent objects constructed conditionally or
after their enclosing owner: the field-advance and deposition actions, snapshot
capture resources, sorters, registered adapters and output engines. Construct
them at startup or on first use, with explicit replacement when resources
change, such as a sorter after particle storage is reallocated. The owning
pointer keeps the object's address stable until replacement or destruction;
objects borrowed by callbacks or asynchronous operations keep a stable address.
References, pointers and views borrow already-owned state and never extend or
manage its lifetime. Shared ownership requires independent owners that actually
share a lifetime; TPIC has none.

`std::optional` represents an absent value only where absence is part of the
value's meaning and has a concrete benefit; it is not the default for delayed
resource ownership:

- `DepositionLedger::next` returns a job or no job without a heap allocation;
- sort receipts and resident-read slots occupy inline slots acquired and retired
  each epoch; they borrow particle storage rather than own resources;
- the asynchronous writer's inline snapshot slot accepts and returns a moved
  host value each output epoch, reusing its buffers.

Timing progress scopes also use inline storage: their nonmovable factory result
spans successive progress calls without allocating each epoch. These per-epoch
slots stay inline, not heap owners. A non-obvious exception is explained where
it is declared.

Declare workers after all state their callbacks borrow, or join them in the
owner's destructor, and drain and join them before that state is destroyed;
construction-failure paths included: a constructor failure closes and joins the
workers already started.

The memory estimate counts the worker objects with the rest of the runtime's
host state, and counts every separately allocated object on its own:
`sizeof` of a `std::unique_ptr` does not include its pointee. Device and host
payload are accounted separately. Allocator overhead, thread states and thread
stacks are outside the reported object-storage subtotal.

## Errors

Errors are reported through the three channels and the single validation
boundary of the [standards](standards.md#errors-and-the-validation-boundary).

## Include and API surface

The host entry points are `config/simulation_parameters.hpp` (parser),
`config/domain.hpp`, `config/output.hpp`, `config/output_selection.hpp`,
`config/numerics_options.hpp`, `initialization/prepared_initialization.hpp`,
and `simulation/planning/run_plan.hpp`, with the neutral `data` records and the
`io` writers; `tests/CMakeLists.txt` compiles each of them, with further host
headers, alone in its own translation unit. `simulation/driver/run_distributed.hpp`
declares `run_distributed`, the device entry point of the application, and
`simulation/driver/simulation_device.hpp` defines `run_simulation` for a caller
that already owns the LPFx runtime. A header's presence in `include/` does not
make every type in it part of the API, and the project promises no binary ABI.

`config/precision.hpp` owns the precision structs, `convert` and their
dispatch; `config/limits.hpp` the species limits; `config/domain.hpp`
`config::PreparedDomain`. `config/numerics_options.hpp` holds `ParticleOrder`,
`MassDepositionKernel` and `MassApplyMode` with their names and parsers. Host
libraries link in the order `TPIC::model` < `TPIC::config` <
`TPIC::initialization` and `TPIC::io` < `TPIC::simulation_support`, and
`TPIC::io` links `LPFx::mpi` for the shared field files; the device library
`TPIC::simulation` holds `src/simulation/run_distributed.cpp` and the dispatch
unit of every compiled pair. The interface target `tpic_test_support` adds the
in-tree `tests/support/` to the include path of the tests: the test builders,
the golden-table helpers and the LPFx mock backend.

## Adding a case

Each case is one struct with the standard interface of
[case_interface.hpp](../../include/tpic/initialization/cases/case_interface.hpp)
(names, admitting constructor, analytic `electric` and `magnetic`,
`particle_label`, the device `Profile`). `harris_sheet` holds `GEMHarris` and
its five variants; every other case has its own files, `Uniform` only a header.
A case is added in these steps:

1. Declare the struct in `include/tpic/initialization/cases/<name>.hpp` and
   implement it in `src/initialization/cases/<name>.cpp`.
2. List the source in `tpic_initialization` in `CMakeLists.txt`.
3. Include the header in `initialization/cases/initial_case.hpp` and add the
   type to `InitialCase`.
4. Add each of its `names` to `config::case_names`
   (`config/simulation_parameters.hpp`).
5. Make every deck key it reads a member of `config::SimulationParameters`,
   read by the input parser (`src/config/simulation_parameters.cpp`), and give
   the key its row in [input_keys.md](../user/input_keys.md);
   `tests/compile/documentation.py` checks that table against the parser.
6. A deck shipped in `input/` or `performance/scaling/` gets its row in
   `tests/data/golden/input_decks.txt`, which `tests/simulation/input_decks.cpp`
   writes with `--print`; the case gets its row in [cases](../user/cases.md).
