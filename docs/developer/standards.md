# Engineering standards

These are the standing rules for every change to TPIC. The [architecture](architecture.md)
describes the structure they protect, the [kernel inventory](kernels.md) lists the
device kernels, the [change review](change_review.md) fixes how a change is
qualified, and [testing](testing.md) describes the test suites.

## The seven rules

| Rule | Statement |
| --- | --- |
| R1 Performance does not get worse | The data- and task-parallel structure (streams and executors, events, host workers, the runtime), the overlap between stages and the communication order stay unless a change is measured at least as fast. A kernel may change to remove redundancy or duplication when no kernel becomes slower (duration, occupancy, registers, shared memory, memory traffic, atomics) and the numerical results stay equivalent: the ECSIM algorithm, its accuracy and its energy conservation are unchanged, and every change of floating-point operation order, rounding or narrowing point or FMA contraction is named and justified. |
| R2 No redundancy, no bloat | An abstraction exists only where it actually helps. One fact is written once: a contract at its callee, a constant beside its schema, a cardinality in `ExecutionLayout`. |
| R3 Compile-time resolution | Templates, `constexpr` and static dispatch. No run-time dispatch, virtual call, stored function pointer or extra indirection on a hot path. The only virtual interfaces are `io::OutputEngine` with the `io::detail::CheckpointEngine::write_file` hook, which run on the writer workers, and the startup-only `io::detail::CheckpointReader`. |
| R4 Clear names | Every name says what the thing is or does, unambiguously, with the vocabulary of the [naming rules](#naming). |
| R5 Modularity and one shape | One shape for every recurring thing: one launch helper, one enqueue signature, one event helper, one error prefix, one layer table. |
| R6 No compatibility layers | Any API, function, type, file, kernel, command-line option, input key, printed text or profile zone may change or go when that brings clarity or performance. A change removes the old spelling; it adds no alias, forwarding overload or test-only entry point. Every reader of a changed name or text (tests, tools, decks, documents) changes in the same change. Output-file array and metadata names are a data contract and change only with the output format. |
| R7 No other code base named | Nothing in the repository names or refers to another code base. Only LPFx and real third-party dependencies are named; the ECSIM method is cited by its published paper. A numerical rule is stated as TPIC's own contract in the [numerical reference](numerical_reference.md). `tests/compile/foreign_references.py` enforces the rule on every tracked or untracked unignored file except `LICENSE`, itself, `include/lpfx/` and `cmake/lpfx/`; the attribution notice, title and output directory of the comparison deck in `input/reconnection_comparison/`, and its file name and output directory in the golden deck tables, are its listed exceptions. |

## Layer table

A header includes headers of its own directory tree and of the layers listed for
it, directly or transitively. [`module_dependencies.py`](../../tests/compile/module_dependencies.py)
holds this table and checks it on the include graph of `include/tpic/` and `src/`.

| Layer | Directory | May include |
| --- | --- | --- |
| 1 | `device/` | no TPIC header |
| 1 | `data/` | `device/`; `config/precision.hpp`, `config/limits.hpp`, `config/domain.hpp` |
| 1 | `config/` | `data/output/output_moment_quantities.hpp`, `data/output/output_snapshot.hpp` and the `data/output/checkpoint.hpp` it includes, from `config/output.hpp` and `config/output_selection.hpp` only |
| 2 | `communication/` | `config/`, `device/`, `data/` |
| 3 | `ecsim/` | `config/`, `device/`, `data/`, `communication/` |
| 4 | `initialization/` | `config/`, `device/`, `data/` |
| 4 | `diagnostics/` | `config/`, `device/`, `data/`; `ecsim/particles/cic.hpp`, the `ecsim/fields/` curl, boundary and smoothing headers (`cell_to_node`, `field_boundary`, `field_boundary_device`, `field_checks`, `node_cell_curl`, `node_smoothing`, `node_smoothing_device`, `stencil_region`, `wall_faces`), `ecsim/deposition/moment_field_device.hpp` and `communication/fields/field_exchange.hpp` |
| 4 | `io/` | `config/`, `data/` |
| 5 | `simulation/` | every layer below |
| 6 | `src/app` | `simulation/`, `config/` |

A source `src/<module>/X.cpp` includes the headers its module's row allows.
Inside `simulation/` a directory includes only itself and the directories before
it: `planning` < `execution` < `timing` < `state` < `transport` < `actions` <
`reporting` < `driver`. Four further rules complete the table;
`module_dependencies.py` checks the first three.

| Rule | Reason |
| --- | --- |
| Nothing reachable from `driver/pipeline.hpp` is a `*_device.hpp`, includes `<mpi.h>` or lies in `io/`. | The runtime stays host-compilable and is tested with a substitute physics owner. |
| No header outside `simulation/driver/` names `Simulation`. | An action sees only what it is handed. |
| `data/` and `ecsim/` contain no `getenv`. | Environment policy belongs to planning (`simulation/planning/run_plan.hpp`). |
| Every standard name used has a direct standard include, and no standard include is unused. | Headers stay self-contained. |

A rule that names a path or pattern that does not exist fails the check, so
the table follows every move in the same change.

## Naming

| What | Rule | Example |
| --- | --- | --- |
| Directories | Lower snake case. The name is a layer or a phase of the run, never a mechanism. | `simulation/planning/`, `ecsim/mass_matrix/` |
| Files | Snake case of the primary type or operation the file defines. `src/<module>/X.cpp` implements `include/tpic/<module>/.../X.hpp`. Two groups implement an interface declared elsewhere: `src/simulation/simulation_dispatch.cpp`, compiled once per precision pair (the explicit instantiation of `run_simulation` in `simulation/driver/simulation_device.hpp`) and the format engines `src/io/vtk_field_engine.cpp`, `vtk_particle_engine.cpp`, `adios2_engine.cpp` and `hdf5_engine.cpp` (`io/output_engine.hpp`, for the last two also `io/detail/checkpoint_engine.hpp`). Basenames are unique within a module. | `scalar_diagnostics_device.hpp` holds `ScalarDiagnostics`; `src/config/simulation_parameters.cpp` implements `config/simulation_parameters.hpp`. |
| Device suffix | `X_device.hpp` needs the device compiler and includes `lpfx/backend/cuda_hip/kernel.hpp` first and once. `X.hpp` is host-includable: it includes no `*_device.hpp` and holds no `__global__`, `__device__`, `__shared__` or launch statement. A test named `*_tests.cpp` is host code; a test named `*_device.cpp` holds device code. No file of `include/tpic/` or `src/` contains a `__CUDACC__`/`__HIPCC__` switch: backend plumbing belongs in LPFx. The only vendor selections are the LPFx build macros `LPFX_CONFIG_DEVICE_BACKEND_CUDA` and `LPFX_CONFIG_DEVICE_BACKEND_HIP`, which pick the cuRAND or hipRAND include of `particle_loading_device.hpp`, the matrix-core mass kernel and its launch shape (`mass_matrix_deposition_device.hpp`, `mass_matrix_mma_device.hpp`, `mass_matrix_mfma_device.hpp`), the particle wire layout of a device build (`particle_wire.hpp`) and the printed backend name (`src/simulation/run_distributed.cpp`), and the device-pass tests of the CUDA rounding intrinsics (`__CUDA_ARCH__` in `async_field_snapshot_device.hpp` and `current_deposition_device.hpp`) and of the matrix-core bodies (`mass_matrix_mma_device.hpp`, `mass_matrix_mfma_layout.hpp`). | `node_smoothing.hpp` (weights, preparation) and `node_smoothing_device.hpp` (kernel, enqueue). |
| Include guards | `#ifndef`/`#define`/`#endif`, never `#pragma once`. The guard is the uppercase path below `include/` with separators and dots replaced by underscores; test headers use the `TPIC_TESTS_` prefix. | `TPIC_SIMULATION_ACTIONS_MOVER_ACTIONS_DEVICE_HPP` |
| Namespaces | `tpic::<top-level directory>`. No namespace per subdirectory and no re-export of a TPIC name at namespace scope. Kernels live in `tpic::<module>::kernels`. `detail` holds names used only by the declaring file and its `_device` twin; a `detail/` directory is private to its module. | `lpfx::device::KernelLaunch`, `tpic::simulation::Simulation`, `tpic::ecsim::kernels::sorted_current_kernel` |
| Types | PascalCase nouns with fixed suffix meanings. `Prepared*`: validated host record built from the input. `*Plan`: immutable host description of something to allocate, register or launch. `*Actions`: the device and native work of one calculation; it owns that calculation's data and borrows the owners, takes `unsigned* invalid` and, unless its stream is bound at construction, the executor; it never records or waits an event, opens a timing scope, places a validation, owns a thread, reads an epoch or calls another action. Two exceptions keep work with its owner: `OutputActions` keeps the epoch it captures and orders and waits for its own snapshot transfers and checkpoint copy, and `LocalSpeciesActions::retire_after` hands the mover's event to the species sorter, whose retirement waits for it. Only `Pipeline` sequences actions. `*Operator`: a numerics owner. | `PreparedDomain`, `MassHaloPlan`, `MoverActions`, `DiagnosticsActions`, `MaxwellOperator` |
| Functions | Verb first, fixed verbs. `prepare_`: host, once, may throw. `make_`: allocates and returns an owner. `enqueue_`: submits device work and returns `lpfx::ErrorCode` (the `enqueue_*` members of owner classes that report by exception, and `enqueue_next_fields`, throw instead). `launch_`: one launch sequence shared by two or more enqueue functions. `begin_`, `progress_`, `conclude_`: a multi-step exchange. `record_`, `wait_`: events. A call that the pipeline or an action makes names its side first. `host_` holds or acts on the calling host thread: `host_wait_` waits for work already submitted, `host_sync_` drains device streams, `host_blocking_` runs an operation to completion on every route (`host_blocking_solve`), `host_poll_` spins, `host_signal` wakes host waiters. `complete_` names no side because the route decides it: it finishes LPFx communication in its executor's order, so work enqueued later on that executor sees the result; a host-observed route (MPI) also blocks the calling thread, a stream-ordered route (NCCL, RCCL) returns at enqueue, and a host read or another stream joins that executor first (`complete_field_exchange`). `device_record` and `device_wait_` only order a stream. `progress_` and `ready` never block. A bare verb on an action or device owner only enqueues device work. `require_`: a throwing precondition. No prefix that repeats the class, no `_impl` in a public namespace. | `prepare_input`, `make_initial_particles`, `enqueue_sorted_current`, `launch_sorted_current`, `begin_assembly`, `require_species_count` |
| Kernels | `<verb>_<object>_kernel` in `<module>::kernels`, with the stem of the enqueue function that launches it. An algorithm variant is part of the name when implementations share an operation; a backend label alone does not identify the algorithm. Parameters come in the order inputs, outputs, run-constant parameters, work range (counts, bins, grid shapes, region geometry), `unsigned* invalid`; parameters of the same or convertible types keep their relative order. Meaning is carried by enumerations and named structs (`TransferDirection`, `GmresIterationStatus`), not by `bool` flags or array slots. | `magnetic_cell_to_node_kernel`, `sorted_mass_matrix_scalar_kernel`, `sorted_mass_matrix_mma_kernel`, `sorted_mass_matrix_mfma4_kernel` |
| Enqueue functions | `enqueue_<stem>` for the kernel `<stem>_kernel`. An enqueue function that selects among several kernels is named after the operation. | `enqueue_fill_moment_neumann_ghost` launches `fill_moment_neumann_ghost_kernel`; `enqueue_sorted_mass_matrix` selects among three kernels. |
| Template parameters | PascalCase, in this order: `Precision`, compile-time selectors, deduced owner types, `Backend`, `Diagnostic`. A `bool` selector is a predicate; anything else is a scoped enumeration. Owner types are named by role, never a bare `Owner` where two owners meet. | `enqueue_magnetic_cell_to_node<Precision, CellOwner, NodeOwner, Backend>`, `config::MassDepositionKernel Kernel` |
| Constants | Every layout or wire literal is spelled once, at its schema. Block sizes are named beside `lpfx::device::KernelLaunch`. | `particle_wire_id` in `communication/particles/particle_wire.hpp`; `default_block_threads` |
| External spellings | Printed text, profile zone names, command-line values and input keys have one canonical spelling and no alias. Zone names are `TPIC::<action or step>`; roles are printed by name. Errors start with `TPIC <module or topic>: `. | `--mass-matrix matrix-core`, `TPIC::Mover` |

## Enqueue functions and launches

A kernel is launched only by an enqueue function, through `lpfx::device::launch`,
or by a member of an owner class that reports by exception, through
`lpfx::device::launch_or_throw`; both are defined in `lpfx/backend/cuda_hip/kernel.hpp`
beside the host-includable `lpfx/backend/launch.hpp`. A free enqueue function,
`enqueue_next_fields` excepted, has this shape:

```text
template <class Precision, selectors..., owner types..., class Backend
          [, class Diagnostic = lpfx::device::DisabledDeviceSubmissionDiagnostic]>
[[nodiscard]] lpfx::ErrorCode enqueue_<stem>(
    read-only owners, written owners, const Plan& prepared_once, scalars, range,
    unsigned* invalid, lpfx::device::Executor<Backend> executor [, Diagnostic diagnostic = {}]);
```

The body has a fixed order: compile-time constraints on the owner and backend
types; per-call host checks, each returning an `lpfx::ErrorCode` (executor
bound, owner initialised, counts in range, extent and halo checks that guard
memory for a caller-supplied owner); the empty-range decision
(`lpfx::device::EmptyWork`); then `return lpfx::device::launch<Shape, kernel>(...)`.

An enqueue function never allocates, records or waits on an event,
synchronises a stream, reads back or re-derives a plan, and a free one other
than `enqueue_next_fields` never throws. The caller
converts the result once with `lpfx::check_error(code, "TPIC <action>: ...")`.
Ordering (wait, timing scope, validation clear, enqueue, validation capture,
record) stays in `Pipeline`. Events are created, recorded and waited
through `lpfx::device::create_event`, `lpfx::device::record_event` and `lpfx::device::wait_event`.

## Errors and the validation boundary

There are three reporting channels.

| Where | How an error is reported |
| --- | --- |
| Host preparation and owner constructors | A standard exception whose text starts with `TPIC <module or topic>: ` (in `config`, through `config/detail/errors.hpp`; file errors of `io` through `output_error`, `TPIC <topic>: <operation> <path>`). |
| Enqueue functions | `lpfx::ErrorCode` as the return value. |
| Orchestration | `lpfx::check_error` or `lpfx::require` at the call site. |

LPFx's `core/error_reporting.hpp` and `backend/error_reporting.hpp` adapters
report portable results and native device failures; `lpfx::require` handles
application predicates. TPIC supplies operation, rank, species and cycle
context and does not decode backend errors. Its own collective helpers are the
startup `agree` and `root_agreed` in `src/simulation/run_distributed.cpp` and
`Simulation::abort`. Checks do not synchronise or retire work: use
`lpfx::check_acceptance` at ownership transfer and `lpfx::check_progress` for
pending or completed polls. Kernels publish validation flags; the host consumer
reports them.

After communication is registered no exception unwinds: `Pipeline::execute`,
worker setup and every worker failure path end in `Simulation::abort`
(`simulation/driver/simulation_device.hpp`). Every failure path prints its
message once on standard error through `simulation::print_error`
(`simulation/reporting/startup_report.hpp`), with ` (rank N)` appended when the
rank is known; a message without the prefix is printed after `TPIC: `. Each fact
is validated once, where it enters:

| Fact | Validated in | Trusted by |
| --- | --- | --- |
| Input values | The parser or the `prepare_*` function that first sees them (`config::require_species_count`, `config::require_canonical_partition`). | Everything that takes a `Prepared*` record. |
| Values that can differ between ranks | The agreements in `run_distributed` (`src/simulation/run_distributed.cpp`). | Every later layer. |
| Run admission | Option values in `prepare_input`; unsupported requests, field-boundary rules and output backends in `RunPlan::prepare`. | Everything that takes `const RunPlan&`. |
| Run-constant device preconditions | The `prepare_*` function or the owner's constructor, once. | The enqueue functions. |
| Per-call facts | The enqueue function. | The kernel. |
| Device-side validity | The kernel sets `unsigned* invalid` with `atomicExch`. One `device::DeviceValidation` owns the flag; it is cleared on the producing stream before the first producer, captured after the last and checked on the consuming host thread. | The next consumer. |

Tests obtain valid records through `tests/support/test_builders.hpp`, not
through validating entry points in production headers.

## Host and device headers, compile time

| Kind | Holds | May include |
| --- | --- | --- |
| `X.hpp` | `LPFX_HOST_DEVICE` arithmetic, plan structs, enumerations, `prepare_*` functions, `constexpr` size rules. | Never a `*_device.hpp`. |
| `X_device.hpp` | `__global__` and `__device__` functions, launches, device owners. | `lpfx/backend/cuda_hip/kernel.hpp` first; its host twin; device headers of lower layers. |

Sizes that both an owner and the memory estimate need live in the host twin
(`ecsim/gmres/gmres.hpp`, `ecsim/maxwell/maxwell.hpp`, the `data/output`
headers, `simulation/execution/execution_layout.hpp`,
`simulation/execution/deposition_ledger.hpp`). Host-only translation units
stay host-compiled: `TPIC::config`, `TPIC::initialization`, `TPIC::io` and
`TPIC::simulation_support`; `--check-input` needs no GPU and never initializes
MPI.

`Precision` is the pair `Precision<Storage, Compute>` (`config/precision.hpp`):
`Storage::X` is the type quantity X is stored in, `Compute::X` the arithmetic
type that produces it. A value changes type only through `convert<To>(value)`,
which returns the value itself for equal types; no expression mixes two
floating types implicitly.

What selects device code or an LPFx type is a template parameter: `Precision`
on every kernel, schema, owner and conversion point; the kernel selectors
(`StencilRegion`, `MassDepositionKernel`, `Partial`, `Diagnose`,
`MaxwellApplySchedule`, the cell source of the particle sample, the axis of the
wall kernels, the GMRES normalize tail); `KernelLaunch` and `SortedCellLaunch`;
`Backend` in `device/`, `data/`, `communication/`, `ecsim/`, `diagnostics/`, the
initialisation device headers and the host-testable utilities (`ExecutionStreams`,
`StageTiming`, `DeviceValidation`, which host tests instantiate with a substitute
backend); `Ns` wherever a typed LPFx operation is named (the compiled species
capacity; the species count of a run is a run-time value); and
`Simulation<Precision, Partial, Kernel>`, selected by one switch per run.
`Pipeline` takes its physics owner as a template parameter, so that host tests
substitute one. What only organises host bookkeeping is plain code: the species
index is not part of `Species`, the device classes of `simulation/` use the one
backend alias of `execution/active_backend_device.hpp` (`Backend`, `Executor`
and `Event` of `lpfx::device::ActiveBackend`), and timing tables, reports and
`RunResult` are compiled once in `TPIC::simulation_support`.

## Kernel changes

Kernel deduplication preserves rounding, compiler contraction, launch layout,
register pressure and memory access patterns. Each storage profile and backend
is validated; reduced source size alone is not evidence of improvement. The
[change review](change_review.md) fixes the checks of a kernel change, and the
[numerical reference](numerical_reference.md) the arithmetic a kernel keeps.

## Change boundaries

Keep resident-reader protection before compaction or growth, the canonical
mass-add order, the rank-common communication order and explicit retirement
after accepted failures. The distributed failure path aborts; closing host
channels cannot cancel native MPI. Registration and lifecycle stay in the
driver; precision, order, kernel and species selection happen once, at the
dispatch. Do not add run-time precision dispatch inside kernels or GMRES matrix
products. See the [output architecture](output_architecture.md) for the output
path, the [numerical reference](numerical_reference.md) before changing
arithmetic and [testing](testing.md) for the test suites.

## The LPFx copy

`include/lpfx/`, the unchanged files of `cmake/lpfx/` and
`tests/support/device/mock_backend.hpp` change only through LPFx: a change is
made in LPFx, the files of that LPFx revision are copied in, and
`python3 tools/dependency_inventory.py --refresh --lpfx <checkout>` records the
revision in `cmake/dependencies.json`. The three adapted files
`cmake/lpfx/CMakeLists.txt`, `LPFxOptions.cmake` and `LPFxTargets.cmake` are
TPIC's own. Host workers, mailboxes, executors and the other runtime
infrastructure come from LPFx; what LPFx lacks is added to LPFx, not written in
TPIC. The [architecture](architecture.md#the-bundled-lpfx)
describes the lock.

## Comments

A comment states what the code cannot state.

| Write | Do not write |
| --- | --- |
| The contract of each enqueue function in at most five lines: the stream it uses, what must be complete before it reads, what it writes, which halo or event the caller handles next. | History, plans, removed designs, or words such as "previously" and "compatibility". |
| The reason for every ordering constraint, at the statement that enforces it: why this wait precedes this launch, why this member is declared last. | Provenance of any kind (R7). |
| Narrowing and rounding points, FMA-contraction choices, units of scalar arguments, the meaning of each validation flag. A statement that must not be reordered or unified is tagged `R1:`. | A restatement of the next line or of the type. |
| At a duplication kept for a measured performance reason, one line naming the other copies and the measurement. | The same contract in two places; it is written at the callee. |
| One line per kernel: iteration model, launch geometry, validation flag. | A number that duplicates a constant; cite the constant. |

## Checks

Three scripts in `tests/compile/` run as tests in every preset (`host`, `cuda`, `hip`).
`module_dependencies.py` (suite `tpic_architecture`) enforces the layer table
(every header in a row), the `src/` rows, the phase order inside `simulation/`
(every header in a phase directory), the first three further rules and the
absence of stale rules and unresolved includes.
`foreign_references.py` (suite `tpic_architecture`) enforces R7 and the
provenance words of the comment rules. `documentation.py` (suite
`tpic_documentation`) checks that the file of every local link of the maintained documents
exists and that [input_keys.md](../user/input_keys.md) lists each key the
input parser reads exactly once and no other key. Two compile steps and one
test enforce parts of other rules: `tests/CMakeLists.txt` compiles the host
entry points and further host headers alone, each in its own translation unit;
`tests/compile/schema_check.cpp` (device builds, `cmake/TpicDevice.cmake`) holds
static assertions on the precision structs and schemas; the host test
`device_submission_rejection` (device builds,
`tests/ecsim/device_submission_rejection.cpp`) checks that enqueue functions
reject invalid host arguments before they touch a stream.
`tools/dependency_inventory.py --check` verifies the LPFx copy rule; it is run
by hand and is not a test. Everything else is a convention, kept by review: the
standard-include rule, the naming rows and include guards, the enqueue shape,
the error channels, the comment rules, R1 to R6 beyond the checked parts, and
the [change classes](change_review.md).
