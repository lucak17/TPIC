# Gauss-law residual diagnostic

This document describes how TPIC computes the Gauss-law residual. What the check
measures, why it is not zero, and how to enable and monitor it are in
[Gauss-law check](../user/gauss_law.md); the keys, `gauss.csv` columns and the
`Gauss` field array are in [output](../user/output.md).

## 1. Computed quantities and cadence

A sample is *begun* at cycle `n` and *finished* at cycle `n + 1`. Per local
cell, with the solver's node-to-cell divergence `divC`
(`ecsim::node_cell_divergence` in `ecsim/fields/node_cell_curl.hpp`):

- begin, state `n`: `s = 0.5*divC(E^n) - c`. Here `c` is the cell charge:
  `4*pi` times the 8-node average of `S(rho)`, where `rho` is deposited from
  the particles at `x^{n+1/2}`, gets wall image values, and passes through the
  run's smoothing `S`.
- finish, state `n + 1`: `r = s + 0.5*divC(E^{n+1})`.

`r` is the residual at `t = (n + 1/2)*dt`; it uses only committed fields and
contains no `theta` ([Gauss-law check](../user/gauss_law.md)). The four sums
of a sample are kept in the slot order of `data::GaussScalarIndex`
(`data/output/gauss_sample.hpp`), which is also the `gauss.csv` column order:
`residual` (`r^2`), `drift` (`(r - r_first)^2`, with `r_first` the first
finished sample of this process, so a restart starts its own baseline), `wall`
(`r^2` over the cells touching a physical wall) and `charge` (`c^2`).

**Cadence.** `config::PreparedOutput::should_sample_gauss(cycle, final)`
(`src/config/output.cpp`) begins a sample at cycle `n` when
`DiagnosticGaussCycle = K > 0` and `n % K == 0`, or, with `FieldOutputTag`
`Gauss`, when `n + 1` is a field snapshot cycle or `n + 1 == final`. `K <= 0`
disables the cadence (the default is 0). The pipeline asks only for
`first_cycle <= n < ncycles`, with absolute cycle numbers like the other
cadences, and the sample finishes at `n + 1 <= ncycles`, so the final epoch
finishes a sample begun at `ncycles - 1`. Any `K >= 1` works:

- With `K = 1`, every epoch finishes sample `n - 1` and begins sample `n`
  (section 2 gives the order).
- `K > ncycles` samples cycle 0 only.
- A restart samples the multiples of `K` from its first cycle.
- The final state cannot begin a sample, because it has no following state.

## 2. Per-epoch timeline

The work runs on the diagnostics/output worker, on the `diagnostics` stream.
`Epoch` (`simulation/execution/execution_layout.hpp`) carries `gauss_begin` and
`gauss_finish`; `reads_state()` is `scalars || gauss_begin || gauss_finish`.
`run_diagnostics_output` (`simulation/driver/pipeline.hpp`) runs:

| Step | When | Work | Reads state n |
| --- | --- | --- | --- |
| A | after `state_ready_event_` and the scalar kernels, finish epochs | `read_gauss_state` → `enqueue_field`: `r = s + h` with `h = 0.5*divC(E^n)`; partials of `r^2`, `(r-r_first)^2`, wall `r^2`; sets `r_first` on the first finish; reduces into slots 0..2 | E |
| B | same, begin epochs | the same kernel writes `s = h` (one fused kernel when `K = 1`), then clears `charge` | E |
| C | begin epochs | `enqueue_deposit` of every species into `charge` (rho only), flagging malformed records in `gauss_validation` | particles |
| D | after A-C | `device_record(sources_read_event)`: field update and mover may overwrite state n | |
| E | after `field_solve.done_fence` (R1) and the scalar collect, finish epochs | `host_blocking_collect_gauss(cycle - 1)`: all-reduce `"gauss"` (4 values, skipped on one rank), download, host sync; rank 0 writes the `gauss.csv` row | |
| F | after E, begin epochs | `gauss_validation.host_wait_check`, then `complete_gauss_charge` (section 6) | |

Why this order:

- A runs before B, and E before F, so `K = 1` never overwrites the stash or
  slot 3 before the finished sample has read them.
- All communication (E, F) runs after the field solve's native work (rule R1 of
  the [standards](standards.md)) and before `diagnosed_fence`, like the scalar
  collective.
- The final epoch has no solve: it runs A and E only, without the R1 wait.
- A field capture that packs the residual (`FieldOutputTag` `Gauss`, finish
  epoch) waits for `sources_read_event` on `output_capture`.

`run_cycles` sets `.gauss_begin = should_sample_gauss(cycle, ncycles)` and
`.gauss_finish = solve_epoch_.gauss_begin` (the previous solve), also on the
final epoch. `count_samples` adds each finish to `RunResult::gauss_samples`.

## 3. Keys

- `OutputConfig::gauss_cycle` (`config/simulation_parameters.hpp`), read from
  `DiagnosticGaussCycle` in `src/config/simulation_parameters.cpp`, default 0.
- `PreparedOutput::gauss_cycle` and `output_gauss` (`config/output.hpp`,
  `src/config/output.cpp`): the `FieldOutputTag` token `Gauss` sets
  `output_gauss` only when `FieldOutputCycle > 0`, and it makes
  `fields_enabled()` true.
- The diagnostic is enabled when the run advances and `gauss_cycle > 0 ||
  output_gauss` (`DiagnosticsActions` constructor).
- The `src/app/main.cpp` help text, the [input keys](../user/input_keys.md) row
  and every deck in `input/` list `DiagnosticGaussCycle`.

## 4. Data and owners

`diagnostics::GaussResidual<Precision, Backend>` in
`include/tpic/diagnostics/gauss_residual_device.hpp` is constructed from
`PreparedDomain`, `PreparedPhysics`, `SimulationParameters`, `enabled`, the
device properties and the `diagnostics` executor. Its fields use
`data::GaussFieldSchema<Precision>` (`data/output/gauss_sample.hpp`), one
component in `Compute::Reduction`. `ScalarLaunch` and
`write_scalar_block_partials` come from `scalar_diagnostics_device.hpp`.

| Member | Type | Purpose |
| --- | --- | --- |
| `charge` | node field (`"q"`) | deposit target and smoothing ping |
| `smooth` | same schema | smoothing pong |
| `residual` | cell field (`"s"`) | `r` of the last finished sample, packed by `FieldOutputTag = Gauss` |
| `stash`, `first` | same cell schema | `s` between begin and finish; `r_first` |
| `values_` / `host_values_` | `gauss_scalar_count` (4) `Reduction`, device / pinned | sums in slot order |
| `scratch_` | `4 * data::contribution_count` `Reduction` | block partials |
| `cells_` | `data::ParticleCellMap` | deposition |
| `moment_plan_` | `ecsim::MomentFieldPlan` | dual-volume scale; its `faces` give the local cells and walls |
| `plans_` | `ecsim::FieldBoundaryPlans` from `prepare_field_boundary_plans<WallQuantity::charge>` | wall rule for a scalar (section 5) |
| `spacing_` | `ecsim::NodeCellSpacing<Reduction>` | `divC` |
| `passes_`, `alpha_`, `operator_` | smoothing; `passes_` is 0 unless `physics.smoothing_enabled` | from `SmoothNiter`, `Smooth`, `ECSimSmoothingOperator` |
| `global_cells_`, `wall_cells_` | double | rms denominators: all global cells, and those touching a closed face |
| `first_set_` | bool | the first finish fills `first` |

**Allocation.** LPFx registration requires initialized owners
(`lpfx/distributed/field/lowering.hpp`). The two registered node fields,
`charge` and `smooth`, therefore always exist, like the output moments: two
`Reduction` values per node and ghost node, 16 B in FP64. The cell fields,
scratch and values are allocated only when enabled, and `enabled()` is
`residual.initialized()`. A disabled run registers a zero-count collective
operand, as `ScalarDiagnostics` does: `collective_operand()` returns
`lpfx::distributed::device_collective_in_place<Backend>(values(), enabled() ? gauss_scalar_count : 0)`.
The destructor synchronizes the stream when work is pending.

**Methods.** `enqueue_field(electric, begin, finish)` (steps A, B),
`enqueue_deposit(particles, qom, invalid)` (step C, through
`enqueue_deposit_output_moments` with the `rho` mask on a one-component owner),
`complete_charge(sum, charge_ghosts, smooth_ghosts)` (section 6) and
`host_blocking_finish(collective, cycle)`, which returns
`data::GaussSample{cycle, (cycle + 0.5)*dt, rms}` with `rms = sqrt(sum / N)`,
`N` being `global_cells_` or, for the wall slot, `wall_cells_` (0 when there
are none), and rejects a nonfinite value.

## 5. Kernels and walls

Both kernels are in `diagnostics::kernels` and grid-stride over local cells
with `ScalarLaunch` blocks (listed in [kernels](kernels.md)):

- `gauss_field_partials_kernel<Begin, Finish, Reduction, ElectricView, CellView>` with the arguments
  `(electric, stash, first, residual, contributions, spacing, faces, set_first)`
  - `h = 0.5*divC(E)`.
  - With `Finish`: `r = stash + h`, stored in `residual`. Accumulates `r^2`,
    the drift term (`(r - first)^2`, or `first = r` and 0 when `set_first`) and
    the wall `r^2` for cells whose lower or upper corner node lies on a wall
    face.
  - With `Begin`: `stash = h`, written after the read, so one kernel serves
    `K = 1`.
  - One `write_scalar_block_partials` call for 3 components.
- `gauss_charge_partials_kernel<Reduction, NodeView, CellView>(charge, stash, contributions, faces)`
  - `c = 4*pi * (8-node average)`, `stash -= c`, partials of `c^2`.
- The reduction into the value slots reuses
  `device::kernels::reduce_block_partials_kernel` with an output offset.

The diagnostic shares the solver's code paths, which take any component count:
`ecsim::FieldAt` (`ecsim/fields/field_checks.hpp`), node smoothing, wall
ghosts, `enqueue_constrained_rows`, and the output-moment deposition, which
accepts a one-component owner (rho only) or the full moment set.

**Walls.** The wall rules are those of
[numerical reference](numerical_reference.md#boundaries-and-migration); their
effect on the check is in [Gauss-law check](../user/gauss_law.md). The charge has no column in the
rule table (`config::FieldBoundaryRule`): its wall plans are those of a
tangential E component (`WallQuantity::charge` in
`ecsim/fields/field_boundary.hpp`), because div E is odd at a PEC wall and even at
a PMC wall:

- **PEC (CP image):** the charge is zero on every node with a PEC face through
  it, and its ghosts are odd.
- **PMC (P image):** the dual-volume-doubled wall value, and even ghosts.

## 6. Charge completion (step F)

`GaussResidual::complete_charge` runs, in this order, after the deposit flag has
been checked. The smoothing passes take no flag (`nullptr`), because
`host_blocking_finish` rejects any nonfinite residual.

1. `complete_field_exchange` on `"gauss.charge"`: shared-node sum.
2. `enqueue_scale_moment_dual_volume(charge, moment_plan_)`.
3. Each smoothing pass:
   - refresh the current owner: constrained rows (PEC wall nodes set to 0),
     then `complete_field_exchange` on `"gauss.charge.ghosts"` or
     `"gauss.smooth"`, then `enqueue_wall_ghosts(current, plans_)`, then rows
     again;
   - `enqueue_node_smoothing<Precision, Reduction>(current, next, operator, alpha, nullptr)`;
   - swap.
4. Rows on the result.
5. `gauss_charge_partials_kernel` on the result, then the reduction into slot 3.

This mirrors `MaxwellOperator::complete_smoothing` and
`complete_electric_refresh`, with the charge plans. Without smoothing it runs
steps 1, 2, 4 and 5, and the kernel needs no ghosts, because the node-to-cell
divergence reads local nodes only.

## 7. Transport registration

`simulation/transport/transport_registry_device.hpp`: `Resources` holds
`diagnostics::GaussResidual<Precision, Backend>& gauss;` and `describe` adds

```cpp
const auto gauss = fields.template on<Route>(r.executors.diagnostics);
d::sum_shared<"gauss.charge">(r.gauss.charge, gauss),
d::copy_ghosts<"gauss.charge.ghosts">(r.gauss.charge, gauss),
d::copy_ghosts<"gauss.smooth">(r.gauss.smooth, gauss),
d::all_reduce<"gauss">(r.gauss.collective_operand(), d::sum, d::on<Route>(r.executors.diagnostics)),
```

The same owner under `sum_shared` and `copy_ghosts` follows `"J.add"` and
`"J.ghosts"`. `Simulation::prepare` (`simulation/driver/simulation_device.hpp`)
passes `.gauss = diagnostics.gauss()`, and the output actions receive
`diagnostics.gauss().residual`.

## 8. Actions and worker

`DiagnosticsActions` (`simulation/actions/diagnostics_actions_device.hpp`) owns
`GaussResidual gauss_` and `io::GaussWriter gauss_writer_`:

| Method | Contract |
| --- | --- |
| `gauss()` | owner access for registration |
| `read_gauss_state(begin, finish, invalid)` | steps A-C: enqueue only (timing row `GaussDiagnostics`, profiler zone `TPIC::GaussDiagnostics`) |
| `host_blocking_collect_gauss(cycle)` | step E: collective, download and host sync; rank 0 writes the row; returns the `data::GaussSample` (timing row `GaussCollect`, zone `TPIC::GaussCollect`) |
| `complete_gauss_charge()` | step F (zone `TPIC::GaussCharge`); MPI blocks for its exchanges, NCCL/RCCL return at enqueue |
| `open_scalars(restart)`, `close_gauss()` | open `gauss.csv` on rank 0 with the scalar files when enabled; close it and return its path (empty when off) |

`DiagnosticsOutputWorker` holds `Validation gauss_validation` and
`data::GaussSample gauss_sample`. `ExecutionLayout::shared_validations` is 6,
one of them the Gauss charge. The header contract comment of `pipeline.hpp`
names the Gauss work.

## 9. Output, reporting and memory

**`gauss.csv`.** `io::GaussWriter` in `io/diagnostics_writer.hpp` and
`src/io/diagnostics_writer.cpp` sits next to `DiagnosticsWriter`; both use the
CSV core `io::detail::CycleCsvFile`: path, open and overwrite, restart (keep the
rows with a cycle below the restart cycle, add the restart comment), the
precision header, increasing cycles and times, and the row buffer. The file has
header comments defining each column and the time `(n + 1/2)*dt`, then the
columns `cycle,time,residual_rms,drift_rms,wall_residual_rms,charge_rms`. The
diagnostic leaves `diagnostics.csv` untouched.

**Field output.** The `FieldOutputTag` token `Gauss` writes `residual` with the
field snapshots, as the only cell-located snapshot array: VTK CellData, ADIOS2
and HDF5 datasets of cell shape, the Fides `cell_set`, and `Center="Cell"` in
the XDMF that `tools/postprocessing/hdf5_fields_to_xdmf.py` writes. It is absent
before the first finished sample.

**Run log.** Rank 0 prints one line per finished sample with
`RunReporter::gauss` (`simulation/reporting/run_reporter.hpp`,
`src/simulation/run_reporter.cpp`), from the main thread after
`host_wait_solved()`, as for the energy lines, the final epoch included:
`  Gauss residual: cycle n t = ...  rms = ...  drift = ...  wall = ...  charge = ...`.

**Startup and end report** (`src/simulation/startup_report.cpp`). The label
"Gauss residual (CSV, log)" shows the cadence, and "before each field snapshot"
for `FieldOutputTag` `Gauss`; the end report lists `RunResult::gauss_samples`
("Gauss-law rows") and `gauss_path` ("Gauss-law file").

**Memory estimate** (`simulation/reporting/memory_estimator.hpp`). The two
Gauss node fields count in `field_device_bytes` (always allocated) and their
halo payload in the bound channels; the three cell fields, scratch and values
count in `diagnostics_device_bytes`, the values also in the pinned diagnostics
bytes, only when enabled; a selected `Gauss` array counts in
`output_device_bytes`.

## 10. Tests

| Test | Checks |
| --- | --- |
| `tests/diagnostics/scalar_diagnostics_device.cpp` (`gauss`) | An independent host evaluation of section 1 in double. One rank with PEC, PMC or mixed walls on every face; smoothing off, face7 or binomial27; double storage and single particle, moment and field storage. Two chained samples, the second begun where the first finishes, compared cell by cell and slot by slot within `2e-11`, relative above magnitude 1. |
| `tests/config/output_tests.cpp`, `simulation_parameters_tests.cpp`, `resolved_parameters_tests.cpp` | the predicate (cadence, `FieldOutputTag` `Gauss`, disabled cadences), parsing and defaults, the golden `resolved_parameters.txt` |
| `tests/io/checkpoint_tests.cpp` | `gauss.csv` open, write, a refused repeated cycle and restart |
| `tests/io/vtk_field_output_tests.cpp`, `tests/io/test_field_reader.py` | the `Gauss` cell array (absent at cycle 0) and its reader |
| `tests/simulation/pipeline_tests.cpp` | the fake diagnostics count `read_gauss_state`, `complete_gauss_charge` and `host_blocking_collect_gauss` for `K = 1`, `K = 3` (a sample finishing in the final state) and a run with scalars only at its ends, and check that a sample finishes before the next begins |
| `tests/simulation/memory_estimator_tests.cpp` | the bytes an enabled diagnostic adds: values, cells and the staged `Gauss` array |

The suites are described in [testing](testing.md).

## 11. Cost per sample

- Begin: one charge deposition (one atomic per CIC corner, all species), one
  shared-node exchange, one halo exchange per smoothing pass, two cell passes.
- Finish: one cell pass, a 4-value all-reduce and one host sync of the
  `diagnostics` stream.

`K = 1` adds about one rho-only deposition per cycle. With the diagnostic off,
a run does no Gauss work; only the two node fields stay allocated.
