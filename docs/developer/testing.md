# Testing

Run from `TPIC/`:

```sh
cmake --preset host
cmake --build --preset host -j 4
ctest --preset host
```

Building and the presets are described in [build and run](../user/build_and_run.md).
Presets build in `build/<preset>`. Use
`ctest --test-dir build/host -R '^tpic_output$' --output-on-failure` for the
output checks, or `ctest --test-dir build/host -R '^tpic_simulation$' -V` to see
each planning and pipeline check. Native fixtures stay directly runnable under
`build/host/tests/`, for example `build/host/tests/tpic_test_run_plan`.

`python3 tools/dependency_inventory.py --check` verifies the bundled LPFx copy
against `cmake/dependencies.json`. It is a manual check, run by hand, and no
CTest suite runs it; the lock is described in
[architecture](architecture.md#the-bundled-lpfx).

## Suites

CTest exposes 10 host suites, plus 5 device suites in CUDA and HIP builds.
Integration suites add up to four more; configuration rejects more than 20
suites (`tpic_register_test_suites` in `cmake/TpicTests.cmake`) and prints the
count. Related checks share a suite while keeping separate executables and
process lifetimes. The output names each check with its result and logs, and the
suite fails if any check fails. Every check has its own timeout (60 s unless
stated); MPI launcher arguments and process counts are preserved. A suite whose
checks all return skip code 77 is skipped; a mixed suite reports passed and
skipped counts separately. Device suites run serially, so that they do not
compete for one GPU.

| Suite | Checks |
| --- | --- |
| `tpic_documentation` | `documentation`: `tests/compile/documentation.py` |
| `tpic_architecture` | `module_dependencies`, `foreign_references` |
| `tpic_tools` | `test_runs`, `test_suite_runner`, `test_whistler_turbulence` (`tests/support/`) |
| `tpic_config` | `simulation_parameters`, `domain`, `physics`, `particle_plan`, `output`, `output_selection` (golden), `precision`, `resolved_parameters` (golden) |
| `tpic_initialization` | `recipe`, `field_profile` (golden), `field_image`, `field_upload` |
| `tpic_numerics` | `mass_mfma_layout`, `stencil_region`, `cell_to_node`, `cic` |
| `tpic_communication` | `field_exchange`, `field_halo`, `mass_halo_plan` |
| `tpic_output` | `checkpoint`, `async_output_writer`, `vtk_field_output`, `vtk_field_output_3` (three ranks), `dataset_engine` (ADIOS2 or HDF5 builds), `adios_paraview_reader` (ADIOS2 builds with `pvpython`), `field_reader` |
| `tpic_simulation` | `execution_streams`, `deposition_ledger`, `stage_timing`, `pipeline`, `device_validation`, `magnetic_startup`, `memory_estimator`, `admission`, `prepared_input`, `run_plan`, `run_options`, `moment_submission_error`, `prepared_plans`, `kernel_records`, `input_decks` (golden); in device builds also `device_submission_rejection`, which compiles device templates and exercises host checks without a GPU |
| `tpic_cli` | `check_input`: the `tpic` executable's `--help` and `--check-input` modes and exit codes (`tests/config/check_input.cmake`); an enabled output channel whose engine the build lacks exits 2 |
| `tpic_device_initialization` | `particle_loading`, `particle_centering`, `field_upload` |
| `tpic_device_runtime` | `magnetic_startup`, `stage_timing`, `execution_streams` |
| `tpic_device_diagnostics` | `scalar_diagnostics`, `async_field_snapshot`, `async_particle_snapshot` |
| `tpic_device_numerics` | `sorted_mass`, `sorted_current_mover`, `gmres`, `maxwell_overlap`, `mass_apply`, `particle_sort` |
| `tpic_device_communication` | `particle_migration`, `mass_halo` (one rank) |
| `tpic_integration_transport` | `particle_migration_2rank` and `mass_halo_2rank` when 2 is a selected rank count; `mass_halo_8rank`, `mass_halo_12rank` |
| `tpic_integration_overlap` | `tests/simulation/overlap_integration.py` |
| `tpic_integration_mass_apply` | `tests/simulation/mass_apply_integration.py` |
| `tpic_integration_snapshot_overlap` | `tests/simulation/snapshot_overlap_integration.py`, registered only with `TPIC_INTEGRATION_BASELINE` |

The test build requires a Python 3 interpreter. Host tests cover parsing,
preparation, immutable plans, geometry, initialization, writer ownership and the
runtime `Pipeline` with a substitute physics owner. `tests/CMakeLists.txt` also
compiles a list of public host headers alone, each in its own translation unit,
and checks during configuration that the module targets exist. Test-only LPFx
mocks come through `tpic_test_support`, separate from the production target
dependencies. Tests that need prepared records build them through
`tests/support/test_builders.hpp` (for example `fill_domain_counts` for a domain
built by hand) instead of validating entry points in production headers. The
golden-table helpers are in `tests/support/golden_data.hpp`, the integration
checks in `tests/support/integration_checks.py` and
`tests/support/snapshot_checks.py`. The `tpic_tools` cases cover job exit
status, logs and timeout cleanup (`test_runs.py`), the suite runner's
pass/skip/fail accounting (`test_suite_runner.py`), and the whistler-turbulence
analyzer on synthetic data (`test_whistler_turbulence.py`).

### Structure and documentation checks

The rules these scripts enforce are stated in the [engineering standards](standards.md).

- [module_dependencies.py](../../tests/compile/module_dependencies.py) checks the
  [layer table](standards.md#layer-table), the phase order inside
  `simulation/`, the runtime isolation, the `Simulation`-name rule and the
  `getenv` rule on the transitive include graph of `include/` and `src/`. A rule
  that names a path that does not exist fails.
- [foreign_references.py](../../tests/compile/foreign_references.py) scans every
  tracked or unignored file except `LICENSE` and the script itself. It rejects
  text that names another code base or justifies a rule by one; an allow list
  names, per file pattern, the only contexts in which denied words may appear,
  and the licence and attribution notice at the top of the comparison deck in
  `input/reconnection_comparison/` is allowed verbatim.
- [documentation.py](../../tests/compile/documentation.py) checks that every
  local link of `README.md`, `AGENTS.md`, `docs/README.md`, `performance/scaling/README.md`
  and every document in `docs/user` and `docs/developer` resolves (anchors are
  not checked), and that [input keys](../user/input_keys.md) lists each key that
  `src/config/simulation_parameters.cpp` reads exactly once, and no other.

## Golden tests

Golden tests compare TPIC with stored tables of values that TPIC produced, kept
as text in `tests/data/golden/`. Each test takes the source root as its
argument; the tests that read decks read every `.inp` file under `input/` and
`performance/scaling/`, except the copies a scaling campaign leaves in its `results_*`
folders.

- `resolved_parameters` (`tpic_config`): every resolved `SimulationParameters`
  member but `unknown_keys`, as exact hexadecimal doubles, for each deck in
  `input/` and `performance/scaling/` and each synthetic deck in
  `tests/data/synthetic_decks.txt` (grammar, canonical values, rejections,
  output channels, 1 to 10 species). Table `resolved_parameters.txt`. The test
  also checks that decks named `bad_*` are rejected and the resolved output
  channel methods. The grammar rejections (unknown case names, trailing text,
  booleans other than `0`/`1`/`true`/`false`) and the listing of unknown keys
  are unit tests in `tests/config/simulation_parameters_tests.cpp`.
- `input_decks` (`tpic_simulation`): host admission and warning count after
  preparing each deck in `input/` and `performance/scaling/`, without the build-dependent
  warning of checkpoints in a build that has no checkpoint backend, so the table
  holds for every build. Table `input_decks.txt`.
- `output_selection` (`tpic_config`): output-tag masks of 2000 seeded tag
  fixtures (`mt19937` seed 8128, 1 to 10 species), table
  `output_tag_masks.txt`, plus 33 hand-derived edge cases of the tag grammar.
- `field_profile` (`tpic_initialization`): B and E at six points for each case
  name and parameter variant, table `field_profiles.txt` (tolerance `3e-14`,
  relative above magnitude 1). Analytic checks need no table: divergence-free
  magnetic fields, the seeded Harris names against the X-point expression
  written with its constants (bit for bit unless the compiler fuses
  multiply-add), periodic seeded modes, the whistler dispersion residual, zero
  amplitudes leaving the background, and host field images of 1 and 12 logical
  ranks that sample the profile at nodes and cell centres, halos included.

`magnetic_startup` (`tpic_simulation`) runs the magnetic startup sequence on
host fields with LPFx's host halo schedule (cell halo exchange, cell-to-node
average, node halo exchange) over 48 fixtures and compares it exactly with an
oracle written from the halo and cell-to-node contracts.

A test reports a missing table by name. A deck added to `input/`, `performance/scaling/` or
the synthetic decks needs its entries. A golden test started with `--print`
after its source root writes the entries of its table to standard output,
without the `#` comment lines that the reader skips; an intended behaviour
change regenerates the affected table this way, and the change and the table
diff are reviewed together.

## Output tests

Writer and reader tests use files emitted by the production C++ writer plus an
independently authored binary fixture. The writer fixture
(`tpic_write_field_fixture`), like every host test that writes VTK fields, is an
MPI program. NumPy adapter cases of `test_field_reader.py` skip when the
optional analysis dependency is absent; core decoding still runs. With the
analysis extras installed, the loading paths of the two-stream, Weibel, whistler
and Taylor-Green analyzers, and the node filter of the whistler-turbulence
analyzer, read actual writer output.

Host tests cover the exact VTK bytes, the shared field files on one and three
ranks (including the `Gauss` cell array, absent at cycle 0), snapshot ownership,
bounded admission, worker failure propagation, destruction draining, particle
data and ids, optional-backend admission and dataset readback. Device capture
tests cover event ownership and particle reconstruction. The memory estimate
reports the pinned particle images, allocated on capture, and the owned field
values separately from the startup allocations; hypothetical particle growth is
excluded from the startup total and never limits allocation.

The checkpoint test writes the pieces of a two-rank checkpoint with each built
backend and restores them on other rank counts and process grids: every record
must arrive once, in its cell, with its stored values, and every field interior
value must be the checkpoint's. It also covers a first checkpoint that creates
`RestartDirName` without a warning, discovery, set-aside, retention and the
continuation of `diagnostics.csv` and `gauss.csv`.

Native backend readback (`dataset_engine`) covers every enabled field and
particle backend combination, with concurrent rank-local writers, simultaneous
and disjoint output cadences, empty species and UInt64 ids above `2^53`. It
verifies that dataset payloads and manifests contain only their selected channel
and that inactive channels keep their latest paths. A bounded ADIOS2 fixture
writes field vectors larger than BP5's 4 MiB deferred threshold and particle
arrays of more than one write chunk across three epochs, including an
empty-to-nonempty particle transition. It overwrites reclaimed buffers and
particle records before readback to verify completed-file ownership and
per-file definitions; these larger files are deleted after the check. The
three-precision device capture fixture (`async_particle_snapshot`) verifies that
the lent records keep their storage types and the stored values after the
released sources are overwritten, that images grow only with the population,
and that the conversion rejects invalid records. Tests without a GPU or MPI
execution do not establish runtime overlap or distributed failure behaviour;
those require the deployed device and transport path.

When `pvpython` is available, ADIOS2 builds run
`tests/io/test_adios_paraview_reader.py` in `tpic_output`. It opens ten
production-writer field pieces through ParaView's Fides reader, one at a time:
six with mixed fields, two with only vectors and two with only scalars. It
checks geometry, physical time, exact field values, vector components, automatic
magnitude selection and interleaved VTK storage without component-conversion
warnings; missing Fides reader support is a skipped case. Native C++ readback
separately validates the interleaved payload and the embedded Fides metadata.

The diagnostics device test compares two chained Gauss samples with an
independent host evaluation; the pipeline test checks the Gauss epochs, their
source releases, and that a finishing sample is collected before the next one
completes its charge. The Gauss tests are listed in
[Gauss diagnostic](gauss_diagnostic.md#10-tests).

The relevant suites: writer and backend checks in `tpic_output`, admission in
`tpic_simulation`, output tag parsing in `tpic_config`, snapshot capture in
`tpic_device_diagnostics` (device builds). The optional library switches are in
[build and run](../user/build_and_run.md).

## Device and integration tiers

Configure the CUDA or HIP toolchain and run CTest. Device tests use skip code 77
when there is no usable driver or device; a skip is not a numerical pass.
Integration is enabled explicitly:

```sh
cmake --preset cuda -DLPFX_DEVICE_ARCHITECTURES=120 \
  -DTPIC_TEST_INTEGRATION=ON -DTPIC_BUILD_ALL_PRECISION_PAIRS=ON \
  '-DTPIC_INTEGRATION_RANKS=1;2' \
  -DTPIC_INTEGRATION_LAUNCHER=mpiexec
cmake --build --preset cuda -j 2
ctest --test-dir build/cuda --output-on-failure -L integration
```

`TPIC_TEST_INTEGRATION` needs a device test build and every storage level with
double compute, hence `TPIC_BUILD_ALL_PRECISION_PAIRS=ON`.
`TPIC_INTEGRATION_RANKS` accepts 1 and 2. `TPIC_INTEGRATION_TIMEOUT` (default
90 s) bounds each MPI job; each integration script has a 3600 s case timeout.
Integration suites use separate output directories under
`integration-results/` of the test build and keep commands, executable hashes,
logs and results. `TPIC_INTEGRATION_BASELINE`, the path of an existing baseline
executable, additionally registers the snapshot comparison.

`overlap_integration.py` defines eight runs per rank count plus the one- and
ten-species one-rank endpoints: 18 cases for ranks 1 and 2, repeated for each
storage level with double compute (90 runs for the five storage levels). With
`--mass-matrix matrix-core` the unsorted cases are recorded as skipped, four per
pair. `mass_apply_integration.py` runs six full/split pairs per rank count, and
`snapshot_overlap_integration.py` runs eight snapshot cases. The mass-halo
oracle also runs on 8 and 12 ranks (`mass_halo_8rank`, `mass_halo_12rank`),
which need that many MPI slots.

`-DTPIC_VALIDATE_PINNED_TRANSFERS=ON` (default `OFF`) links the application
with a validation-only audit of the host pointers passed to host/device copies.
It wraps the CUDA or HIP runtime `MemcpyAsync` and `Memcpy2DAsync` calls at link
time, so it requires a device build and a linker that supports `--wrap`, and it
does not see copies made inside shared vendor libraries. Test builds compile the
same audit object (`tests/support/pinned_transfer_audit.cpp`) independently of
this option, into the `particle_loading`, `particle_centering` and
`scalar_diagnostics` device tests.

## Numerical gates

Cover representative interactions among all precision and order modes,
supported mass kernels, full, split and smoothed Maxwell schedules, wall
conditions, one and two ranks, species endpoints and final, zero-cycle and
selected output. `overlap_integration.py` checks the field files of every run,
and species totals must match the sum of their species arrays. Particle loading
is seeded per rank, so runs on different rank counts are not compared.

The matrix-path device test `tests/ecsim/sorted_mass_device.cpp` compares each
sorted mass-matrix coefficient with the unsorted scalar deposition (scalar
kernel) or with host FP64 cell sums (matrix-core kernel) within
`3e-12 * max|coefficient| + 1e-15` for FP64 mass storage and
`3e-5 * max|coefficient| + 1e-15` for FP32 mass storage, the maximum taken over
the expected coefficients.

### Baseline tolerances

Baseline comparisons in `snapshot_overlap_integration.py` and
`mass_apply_integration.py` compare each precision pair with a baseline of the
same pair; different pairs are not required to produce identical trajectories.
CTest passes `TPIC_INTEGRATION_BASELINE` to `snapshot_overlap` only; the
`mass_apply` comparison runs when the script is started by hand with
`--baseline`, and its full/split comparison uses the same bounds. Pairs with
double field storage use `rtol=3e-10, atol=3e-12` and allow a
normalized energy-drift difference of `1e-12`. Single field storage uses
`rtol=3e-5`, `atol=3e-7` and `1e-6`. Other comparisons state their own bounds;
thresholds are not relaxed to make a refactor pass.

GPU changes also require the applicable memcheck and racecheck runs and runtime
failure tests. CUDA, HIP and each transport are qualified separately, and only
results of the binary under test qualify it; the change classes are in
[change review](change_review.md).

## Comparing two runs

`tools/compare_simulation_outputs.py baseline candidate --report report.json`
compares two completed run directories without changing either. Each directory
must contain `simulation.out` and `output/diagnostics.csv`; matching VTK field
files are compared when present, read in bounded chunks with the Python
standard library. It checks the completed cycles, GMRES iterations and particle
counts of the logs, and requires the precision profile of the `Precision` line
of both startup reports and of both CSV headers to agree. A value passes when
`|candidate - baseline| <= atol + rtol * max(|candidate|, |baseline|)`. The
defaults (`--field-rtol`, `--diagnostic-rtol` `3e-10`; `--field-atol`,
`--diagnostic-atol` `3e-12`) follow the FP64 integration checks; other precision
profiles need explicitly chosen thresholds, which the script never infers.
`--energy-drift-atol` additionally bounds the difference between the two
normalized energy-drift histories. Timing comparisons are in
[performance](performance.md).
