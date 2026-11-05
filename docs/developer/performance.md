# Performance measurement

## Paired runs and comparisons

`tools/benchmark_simulation.py --help` describes paired runs: it compares two
existing TPIC binaries (`--baseline`, `--candidate`) on frozen copies of a
one-rank input, sequentially, in separate output directories, never overwriting
the source input, with stage profiling off. Timings use matched cycle indices;
repeated-run means, rather than individual timesteps, are the samples of the
paired comparison. `tools/compare_simulation_outputs.py --help` describes the
output comparison ([testing](testing.md#comparing-two-runs)); it reads the
precision pair from the `Precision` line of the startup report.

Keep the commands, the input and executable hashes, precision, order, mass
kernel, schedule, GPU and transport of every measurement. A candidate passes the
numerical comparison before performance conclusions are drawn.

Compare repeated ordinary and snapshot cycles separately. Exclude warm-up from
steady-state statistics and use the maximum-rank loop time for distributed
runs. Console reporting and final-state output occupy different timing
intervals: the loop totals include cycle reporting and exclude final-state
diagnostics and output.

The whole-run interval starts at bootstrap, so it includes the initial current
and mass deposition; the steady interval starts after the first cycle. Each
advance builds the moments for the next solve, and the final advance builds
none because no solve follows it. A per-cycle interval therefore does not
contain "its own" deposition, and paired comparisons use whole-run intervals.

Stage profiling (`TPIC_PROFILE_STAGES=1` in the environment) observes streams
and events; it does not justify adding synchronization. Synchronized deposition
diagnostics (`TPIC_DIAGNOSTIC_SYNC_DEPOSITION=1`) deliberately alter timing.
Both are read once by the run plan (`simulation/planning/run_plan.hpp`).
One-GPU self halos cannot demonstrate remote network latency hiding. Compare
full and split schedules and scalar and matrix-core kernels only on their
supported configurations.

## Stage timing rows

With `TPIC_PROFILE_STAGES=1` each worker records into its own `StageTiming`
instance (`simulation/timing/stage_timing.hpp`). The regions, their device event
slots and their print order are one table in
`simulation/timing/timing_regions.hpp`; the statistics are printed by
`src/simulation/timing_regions.cpp`. Every cycle prints a
`Detailed timing cycle <n> (...) host wall ms:` line and a
`Device stream elapsed ms (inclusive; overlapping):` line, and the run ends with
tables for all cycles, ordinary cycles and output cycles; the last two groups
exclude the first cycle. A row is printed only when its region recorded a
sample.

The advancing workers head each per-cycle table with
`Worker timing epoch <generation> destination <cycle> role <role> species <s>`
and each final table with `Worker timing role <role> species <s>`, where
`<role>` is `mover`, `field_advance`, `routing`, `deposition` or
`local_species`. The startup memory plan reports the runtime's host storage on
the line `Runtime host / timing bytes`.

A host value is the wall time of a scope on the submitting thread, including
submission, waits, MPI and I/O. A device value is the time between a start and a
stop event recorded on one stream. Device intervals of different streams
overlap, nested scopes include their children, and an interval includes the
time its stream spent in queued waits. The rows are not exclusive kernel time
and do not sum.

| Row | Recorded by | Meaning |
| --- | --- | --- |
| `CycleTotal` | Main, `control` stream | The whole cycle body: the solve, source release, the next advance and the committed-state publication. |
| `FieldSolve` | Solver worker, `solver` stream | The whole solver job: the wait for the committed state, right-hand side, waits for assembled current and mass, the GMRES solve and the particle field. It encloses the four rows below. |
| `RhsFields` | Solver worker | Old-field terms of the right-hand side, submitted before the assembled current is needed. |
| `FinishRhs` | Solver worker | Completion of the right-hand side with the assembled current, including current smoothing when enabled. |
| `SolveMaxwell` | Solver worker | The complete GMRES solve with every matrix-vector product, its halo exchanges and the joins of the `solver_interior` and `solver_mass` streams. There is no row per interior or shell application. |
| `ParticleField` | Solver worker | The particle-facing electric field, the capture of the solve's validation flag and the publication to the advance. |
| `FieldDiagnostics` | Diagnostics/output worker, `diagnostics` stream | Launch of the field-energy reduction. |
| `ParticleDiagnostics` | Diagnostics/output worker | One sample per species on the lane that species uses: with two lanes, `diagnostics_helper` for even indices and `diagnostics` for odd indices. |
| `DiagnosticsCollect` | Diagnostics/output worker | The diagnostics collective and scalar readback. It starts after the solver's native interval. |
| `GaussDiagnostics` | Diagnostics/output worker, `diagnostics` stream | The Gauss field kernel and, for a beginning sample, the charge deposition of every species. |
| `GaussCollect` | Diagnostics/output worker, host only | The Gauss collective and readback of a finishing sample, then the charge exchanges, smoothing and charge kernel of a beginning one. It starts after the solver's native interval. |
| `OutputDiagnostics` | Diagnostics/output worker, host only | Writing the scalar sample. |
| `OutputFields` | Diagnostics/output worker, host only | Launch of the snapshot capture. |
| `OutputMoments` | Diagnostics/output worker | One sample per species with selected moments: its moment deposition (launched under `OutputFields` for the first such species), the flag check and the shared-node exchange of its output moments; packing follows outside the row. |
| `OutputCapturePublish` | Diagnostics/output worker, host only | Handing a completed host transfer to the disk writer, after joining a previous snapshot that publishes nothing. |
| `OutputWriterWait` | Diagnostics/output worker, host only | Waiting for a pending host transfer and its handoff, then the local disk piece of a snapshot that publishes. |
| `Checkpoint` | Diagnostics/output worker, host only | Joining the pending checkpoint write before a particle or checkpoint capture, the checkpoint capture, and, on epochs without a snapshot, the wait for its transfer and its handoff. |
| `ClearMoments` | Deposition worker | Two samples: clearing the current buffer on `current` and the mass matrix on `mass`, after the solver has released them. |
| `CurrentDeposition` | Deposition worker, `current` stream | Current deposition of one species. Resident, tail and whole-owner phases are separate samples. |
| `MassDeposition` | Deposition worker, `mass` stream | Mass deposition of one species, with the same phases. |
| `CurrentAssembly` | Deposition worker | Current assembly, from the start of the shared-node sum to its completion. |
| `MassAssembly` | Deposition worker | The additive and ghost phases of the compact mass halo. |
| `Mover` | Mover worker, `move` stream | Mover submission of one species. |
| `ParticleMigration` | Local and routing workers | Migration of one species: the local stage on its local stream and the routing stage on `route`. |
| `ParticleSort` | Local worker | The sort of one species on its sort stream. |
| `FieldAdvance` | Field worker, `field_update` stream | The next-field update. |
| `FieldCommit` | Field worker, `field_update` stream | The wait for the last mover, the commit of the next cell and node magnetic fields and the boundary projection of E. |

Every region has a production scope; the table holds no region that no scope
opens.

## Loop statistics

The loop summary (`ECSIM Main Loop Complete`, printed by `RunReporter`) gives
the total and steady-state times as the maximum over the ranks, the steady state
excluding the first cycle. The cycle-time statistics (`Std. dev. per cycle`, the
`Ordinary after warm-up` and `Output after warm-up` rows) are rank-0 samples.

## Profiler zones

Host scopes are NVTX ranges on CUDA and ROCTX ranges on HIP when the build sets
`LPFX_PROFILING_BACKEND` to `NVTX` or `ROCTX` (default `NONE`, which compiles
the scopes away). Zones are named `TPIC::<action or step>`, for example
`TPIC::Mover`, `TPIC::FieldSolve`, `TPIC::GMRES::Iteration` and
`TPIC::Initialize::ParticleSort`. Most stage timing regions have a zone of the
same name; `TPIC::GaussCharge`, `TPIC::Output::WritePiece` and
`TPIC::Output::PublishSnapshot` name work inside larger regions, and instant
marks such as `TPIC::DiagnosticSourcesRead` and `TPIC::SteadyStateBegin` label
events.

## CrayPat

With `LPFX_ENABLE_CRAYPAT=ON`, an executable run under `pat_run` records only
the steady-state interval of the loop summary: `main` switches CrayPat recording
off, and `run_cycles` switches it on after the warm-up cycle's timing boundary
and off after the closing one. The switch is made on the main thread, so it
covers every worker thread, and every table of `pat_report` covers the cycles
that the steady-state time covers. A run of one cycle records nothing. Without
`pat_run` the executable runs normally. Load `perftools-base` before
configuring; the executable then needs the same perftools install at run time:

```sh
module load perftools-base
cmake -S . -B build/craypat <options of the profiled build> -DLPFX_ENABLE_CRAYPAT=ON
srun ... pat_run -w -g mpi,hip build/craypat/tpic --run deck.inp
```

## Launch configurations

The launch constants of every kernel are listed in the
[kernel inventory](kernels.md); a change to a kernel body or a launch
configuration is qualified as described in [change review](change_review.md).

## Compilation

For compilation changes, record clean build wall time, peak compiler memory,
object and executable size and which translation units rebuild after a small
edit. Keep one translation unit per precision pair
(`src/simulation/simulation_dispatch.cpp`, compiled once per pair with its
`Simulation` instantiations) and the typed species registration unless
measurements justify changing their specialization boundaries.

Cluster scaling campaigns are described in [scaling](../../performance/scaling/README.md).
