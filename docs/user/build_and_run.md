# Build and run

Commands below run from the TPIC root directory. LPFx is bundled in `include/lpfx` and `cmake/lpfx` and is
built with TPIC; nothing else has to be fetched ([architecture](../developer/architecture.md) describes the copy
and its lock).

## Prerequisites

| Requirement | Minimum | Where it is checked |
| --- | --- | --- |
| CMake | 3.24 | `cmake_minimum_required` in [CMakeLists.txt](../../CMakeLists.txt) and [CMakePresets.json](../../CMakePresets.json) |
| C++ compiler | C++20 | `cxx_std_20` on every TPIC target |
| MPI | any implementation that provides `MPI_THREAD_MULTIPLE` | `find_package(MPI)` at configure time; the thread level at startup |
| CUDA toolkit with cuRAND (CUDA builds) | C++20 device compilation: CUDA 12; `sm_120` needs CUDA 12.8 | not checked by CMake beyond `find_package(CUDAToolkit)` |
| ROCm with HIP and hipRAND (HIP builds) | C++20 device compilation | `find_package(hip)` and `find_package(hiprand)` |
| NCCL (optional) | 2.18.1 | [cmake/lpfx/CMakeLists.txt](../../cmake/lpfx/CMakeLists.txt), again at runtime |
| RCCL (optional) | 2.18.3 | as NCCL |
| ADIOS2 (optional) | 2.10, CXX component | [CMakeLists.txt](../../CMakeLists.txt) |
| HDF5 (optional) | C component | [CMakeLists.txt](../../CMakeLists.txt) |
| Python 3 (tests and post-processing only) | an interpreter for the test suite; 3.10 for the tools in `tools/` | `find_package(Python3)` in `tests/CMakeLists.txt`; [pyproject.toml](../../pyproject.toml) |
| NVTX or ROCTX (optional) | headers of the toolkit | `LPFX_PROFILING_BACKEND` |
| CrayPat (optional) | `pat_api.h` and the API stub library of perftools | `LPFX_ENABLE_CRAYPAT`, from `CRAYPAT_ROOT` |

Running a simulation needs no Python: the `tpic` executable does not use it. CMake looks for a Python 3
interpreter only to register the test suites (`TPIC_BUILD_TESTS=ON`, the default); with
`-DTPIC_BUILD_TESTS=OFF` the build needs no Python either. The post-processing tools and analyzers in
`tools/` need Python 3.10; their analysis extra (`numpy`, `matplotlib`, `scipy`) is described in
[analysis](analysis.md).

## Presets

[CMakePresets.json](../../CMakePresets.json) defines three configure, build and test presets. Each builds into
`build/<preset>/` with the Unix Makefiles generator, `CMAKE_BUILD_TYPE=Release`, `TPIC_BUILD_TESTS=ON` and
`LPFX_ENABLE_MPI=ON`; every build links MPI. Machine-specific settings belong in an untracked
`CMakeUserPresets.json`.

| Preset | Selects | Builds |
| --- | --- | --- |
| `host` | `TPIC_BUILD_DEVICE=OFF`, `LPFX_DEVICE_BACKEND=NONE` | the host input checker and the portable tests; no simulation |
| `cuda` | `LPFX_DEVICE_BACKEND=CUDA`, `TPIC_COMMUNICATION_BACKEND=MPI`, `LPFX_ENABLE_DEVICE_AWARE_MPI=ON` | the simulation and the device tests |
| `hip` | `LPFX_DEVICE_BACKEND=HIP`, otherwise as `cuda` | as `cuda` |

```sh
cmake --preset host
cmake --build --preset host -j 4
ctest --preset host
```

The host executable `build/host/tpic` checks decks without GPU or MPI startup and writes no file; it performs no
timestep.

### Configuring without presets

These commands set the options of the presets (`TPIC_BUILD_DEVICE` is `ON` by default) and give the same CMake
caches; the CUDA one also enables ADIOS2 and HDF5:

```sh
# CUDA (--preset cuda, plus ADIOS2 and HDF5)
cmake -S . -B build/cuda -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTPIC_BUILD_TESTS=ON \
  -DLPFX_DEVICE_BACKEND=CUDA -DLPFX_DEVICE_ARCHITECTURES=120 \
  -DLPFX_ENABLE_MPI=ON -DLPFX_ENABLE_DEVICE_AWARE_MPI=ON \
  -DTPIC_COMMUNICATION_BACKEND=MPI \
  -DTPIC_ENABLE_ADIOS2=ON -DTPIC_ENABLE_HDF5=ON
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure

# Host input checker and portable tests (--preset host)
cmake -S . -B build/host -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DTPIC_BUILD_DEVICE=OFF -DTPIC_BUILD_TESTS=ON -DLPFX_DEVICE_BACKEND=NONE -DLPFX_ENABLE_MPI=ON
```

For HIP use the CUDA command with `-B build/hip`, `-DLPFX_DEVICE_BACKEND=HIP` and
`-DLPFX_DEVICE_ARCHITECTURES=gfx90a`. Every build requires `LPFX_ENABLE_MPI=ON`; a device build requires
`LPFX_DEVICE_BACKEND`, `LPFX_DEVICE_ARCHITECTURES` and, with the MPI payload backend,
`LPFX_ENABLE_DEVICE_AWARE_MPI=ON`; the host build requires `TPIC_BUILD_DEVICE=OFF`. `TPIC_ENABLE_ADIOS2`,
`TPIC_ENABLE_HDF5` and NCCL or RCCL through `TPIC_COMMUNICATION_BACKEND`
([device builds](#device-builds)) are optional. [Build options](#build-options) lists every option.

## Device builds

A device build needs the architecture of the GPU it runs on:

```sh
cmake --preset cuda -DLPFX_DEVICE_ARCHITECTURES=120
cmake --build --preset cuda -j 4
ctest --preset cuda

# On a ROCm installation:
cmake --preset hip -DLPFX_DEVICE_ARCHITECTURES=gfx90a
cmake --build --preset hip -j 4
```

`LPFX_DEVICE_ARCHITECTURES` takes a `;`-separated list (`"80;90"`); the standard `CMAKE_CUDA_ARCHITECTURES` or
`CMAKE_HIP_ARCHITECTURES` is accepted instead, and a conflicting pair is a configure error. A device build
without an architecture stops at configure time.

| Backend | Value | GPUs | `--mass-matrix matrix-core` |
| --- | --- | --- | --- |
| CUDA | `80` | A100 | FP64 `m8n8k4` |
| CUDA | `86` | RTX 30xx, A40 | FP64 `m8n8k4` |
| CUDA | `89` | RTX 40xx, L40S | FP64 `m8n8k4` |
| CUDA | `90` | H100, GH200 | FP64 `m8n8k4` |
| CUDA | `120` | RTX 50xx | FP64 `m8n8k4` |
| HIP | `gfx90a` | MI210, MI250, MI250X | FP64 MFMA4 |
| HIP | `gfx942` | MI300A, MI300X, MI325X | FP64 MFMA4 |
| HIP | `gfx950` | MI350X, MI355X | FP64 MFMA4 |

The CUDA matrix-core kernel needs compute capability 8.0 or higher; the HIP one needs a wave64 device of
`gfx90a`, `gfx940`, `gfx941`, `gfx942` or `gfx950`. [Support](support.md) lists the platforms TPIC is tested on.

The presets assert that MPI accepts device pointers (`LPFX_ENABLE_DEVICE_AWARE_MPI=ON`); compilation cannot
establish it, so use them only with a GPU-aware MPI deployment. To move device payloads with NCCL instead:

```sh
cmake --preset cuda -B build/cuda-nccl \
  -DLPFX_DEVICE_ARCHITECTURES=120 \
  -DTPIC_COMMUNICATION_BACKEND=NCCL -DLPFX_ENABLE_NCCL=ON \
  -DLPFX_ENABLE_DEVICE_AWARE_MPI=OFF
cmake --build build/cuda-nccl -j 4
```

For HIP use `TPIC_COMMUNICATION_BACKEND=RCCL` and `LPFX_ENABLE_RCCL=ON`. `NCCL_ROOT` or `RCCL_ROOT` locates the
library. MPI stays in use for startup and host control. Use one build directory per backend and toolchain.

## Build options

| Option | Default | Effect |
| --- | --- | --- |
| `TPIC_BUILD_DEVICE` | `ON` (presets set it) | build the CUDA/HIP simulation; `OFF` builds the host input checker only |
| `TPIC_BUILD_TESTS` | `ON` | build the CTest suites |
| `TPIC_ENABLE_ADIOS2` | `OFF` | rank-local ADIOS2 BP5 field, particle and checkpoint output |
| `TPIC_ENABLE_HDF5` | `OFF` | rank-local HDF5 field, particle and checkpoint output |
| `TPIC_COMMUNICATION_BACKEND` | `MPI` | device payload transport: `MPI`, `NCCL` (CUDA) or `RCCL` (HIP) |
| `TPIC_MAX_SPECIES` | `10` | the largest `ns` a deck may request |
| `TPIC_PARTICLE_MIGRATION_HOPS` | `1` | particle migration routing rounds per species and cycle |
| `TPIC_BUILD_ALL_PRECISION_PAIRS` | `OFF` | compile all 25 storage/compute pairs instead of the five matched ones |
| `TPIC_PRECISION_PAIRS` | empty | an explicit `;`-list of compiled pairs, such as `double/double;single-position/double` |
| `TPIC_DEFAULT_STORAGE_PRECISION`, `TPIC_DEFAULT_COMPUTE_PRECISION` | `double` | the levels used without `--precision`; the pair must be compiled |
| `TPIC_TEST_INTEGRATION` | `OFF` | register the GPU/MPI integration suites; [testing](../developer/testing.md) gives their `TPIC_INTEGRATION_*` settings |
| `TPIC_VALIDATE_PINNED_TRANSFERS` | `OFF` | audit host/device copy pointers of the executable (device build, GNU `--wrap` linker) |
| `LPFX_ENABLE_MPI` | `OFF` (`ON` in the presets) | MPI host control and shared field files; required `ON` |
| `LPFX_ENABLE_OPENMP` | `OFF` | build the LPFx OpenMP host-execution component; no TPIC target links it |
| `LPFX_DEVICE_BACKEND` | `NONE` | `CUDA` or `HIP` (set by the presets) |
| `LPFX_DEVICE_ARCHITECTURES` | empty | device architectures, required for a device build |
| `LPFX_ENABLE_DEVICE_AWARE_MPI` | `OFF` (`ON` in the device presets) | assert that MPI accepts device pointers; required with `TPIC_COMMUNICATION_BACKEND=MPI` |
| `LPFX_ENABLE_NCCL`, `LPFX_ENABLE_RCCL` | `OFF` | build the vendor payload component |
| `LPFX_TEST_USE_MPIEXEC` | `ON` with `LPFX_ENABLE_MPI` | launch the native test executables through `MPIEXEC_EXECUTABLE` |
| `LPFX_PROFILING_BACKEND` | `NONE` | `NVTX` or `ROCTX` trace annotations ([performance](../developer/performance.md)) |
| `LPFX_ENABLE_CRAYPAT` | `OFF` | CrayPat records only the steady-state cycles ([performance](../developer/performance.md#craypat)) |
| `LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT` | `20` | extra particle capacity over the planned count at allocation and growth |
| `LPFX_PARTICLE_CAPACITY_MULTIPLE` | `64` | particle capacity rounding (a power of two) |

The ADIOS2 and HDF5 switches work with every preset; `CMAKE_PREFIX_PATH` finds libraries outside the default
search path. The deck keys `WriteMethod`, `FieldWriteMethod`, `ParticlesWriteMethod` and `RestartWriteMethod`
select the engines ([output](output.md)); an engine that is not built rejects `--run`.

```sh
cmake --preset cuda -B build/cuda-output -DLPFX_DEVICE_ARCHITECTURES=120 \
  -DTPIC_ENABLE_ADIOS2=ON -DTPIC_ENABLE_HDF5=ON
cmake --build build/cuda-output -j 4
```

A run with fewer species than `TPIC_MAX_SPECIES` allocates particle, sort and communication buffers only for the
species it uses; a larger capacity only lengthens compilation. The test suite and its golden tables assume the
default. One migration round covers every admitted deck: on a decomposed run a particle moves at most two cells
beyond its rank in one step, and every split axis has at least two cells per rank. A farther flight aborts the
run, so a larger `TPIC_PARTICLE_MIGRATION_HOPS` only adds rounds.

## Commands

```sh
build/host/tpic --check-input input/uniformStartup.inp
mpiexec -n 1 build/cuda/tpic --initialize input/uniformStartup.inp
mpiexec -n 1 build/cuda/tpic --run input/uniformStartup.inp
```

Exactly one command and one input file are required:

| Command | Effect |
| --- | --- |
| `--check-input INPUT` | Host admission of the deck for one rank, without GPU or MPI startup and without output. |
| `--initialize INPUT` | Distributed GPU startup (fields, particles, communication), then exit without timesteps or output. Device build only. |
| `--run INPUT` | Startup, then `ncycles` timesteps with output. Device build only. |
| `--help` | Print the usage text and the compiled default precision. |

| Option | Values | Default | Commands |
| --- | --- | --- | --- |
| `--order` | `full`, `partial`, `unsorted` ([particle ordering](#particle-ordering)) | `partial` | all |
| `--precision` | `double`, `single-position`, `single-position-velocity`, `single-particles-moments-fields-double-mass`, `single-particles-moments-fields` ([precision](#precision)) | the build default (`double/double`) | all |
| `--storage-precision`, `--compute-precision` | the same five levels, selected separately | `TPIC_DEFAULT_STORAGE_PRECISION`, `TPIC_DEFAULT_COMPUTE_PRECISION` | all |
| `--mass-matrix` | `scalar`, `matrix-core` ([numerical options](#numerical-options)) | `scalar` | all |
| `--mass-apply` | `full`, `split` ([numerical options](#numerical-options)) | `full` | all |
| `--overwrite-output` | none: replace the files this run emits ([output](output.md)) | existing files are refused | all |
| `--restart` | none: continue from the newest complete checkpoint in `RestartDirName` ([output](output.md)) | a fresh start | `--initialize`, `--run` |
| `--rank N` | the partition `--check-input` checks, below `XLEN*YLEN*ZLEN` | `0` | `--check-input` |

Any other argument that starts with `-`, an option without its value or with a value outside its list, a second
input file, `--rank` with `--initialize` or `--run`, and `--restart` with `--check-input` are command-line errors.
`--initialize` and `--run` in a host build are errors.

`--check-input` takes the process count from `XLEN*YLEN*ZLEN`; rank 0 holds the largest partition. It checks the
deck, the case admission, the particle plan, the command-line selections and that the engine of every enabled
output channel is compiled in (a missing one is an unsupported request). It does not check the GPU (device
memory, matrix-core support of the device), the run-time rejection of open field walls or a restart, which
`--initialize` and `--run` check, nor the output and checkpoint directories, which only `--run` checks.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | `--check-input` admitted the deck; `--initialize` or `--run` completed; `--help` |
| `1` | invalid command line or input, printed as one `TPIC <topic>: <message>` line on standard error (`TPIC options:` for the command line, followed by the usage when no mode or input file is given); with `--initialize` and `--run`, any failure (input, admission, startup, existing output files, a solver that does not converge, a device error): each rank that detects it prints that line with ` (rank N)` appended, and all ranks stop through `MPI_Abort` with error code 1 |
| `2` | `--check-input` only: the deck is valid but holds unsupported requests, each printed on an `Unsupported:` line |

A host build reports `--mass-matrix matrix-core` as unsupported (it has no device kernel), so `--check-input`
exits with code 2 for it. An unsupported request in `--initialize` or `--run` is a failure with code 1.

### Particle ordering

| `--order` | Behaviour |
| --- | --- |
| `unsorted` | particle kernels without cell reordering |
| `full` | sort the complete local population by cell after migration |
| `partial` | sort and deposit the particles that stay on the rank while migration proceeds, then deposit the arrivals unsorted (default) |

Every command defaults to `partial`, so a default check validates the plan a default run executes. A one-rank run
with `full` or `unsorted` order and no particle-removing face skips local migration.

### Precision

A run selects a storage level and a compute level. Storage is the type each quantity is kept in; compute is the
arithmetic type that produces it. `--precision LEVEL` selects the matched pair: the same level for both, except
that `single-particles-moments-fields-double-mass` keeps `single-particles-moments-fields` storage, so the mass
matrix is computed in FP64 and stored in FP32. GMRES reads the mass matrix on every iteration, and FP32 mass
storage halves those reads. `--storage-precision` and `--compute-precision` select the levels separately; an
unmatched pair needs a build with `TPIC_BUILD_ALL_PRECISION_PAIRS=ON` or a `TPIC_PRECISION_PAIRS` list that
names it, because the default build compiles the five matched pairs. A pair that is not compiled is an
unsupported request.

| Level | FP32 | FP64 |
| --- | --- | --- |
| `double` | nothing | everything |
| `single-position` | particle fractions and displacements | everything else |
| `single-position-velocity` | fractions, displacements and velocities | everything else |
| `single-particles-moments-fields-double-mass` | fractions, displacements, velocities, J, E and B | mass matrix, particle charge, solver and diagnostic reductions |
| `single-particles-moments-fields` | fractions, displacements, velocities, J, mass matrix, E and B | particle charge, solver and diagnostic reductions |

As a storage level a row names what is stored in FP32; as a compute level, the arithmetic that produces those
quantities: the mover's drift and velocity update, the current and mass deposition, the field solve and update.
The GMRES dot products and its residual `rhs - A*x` are FP64 at every level, so the solver tolerance is reached
with FP32 field arithmetic too. Cell indices are int32 and particle IDs uint64. Snapshot values are Float64
(particle `id` and `species` UInt64; checkpoints keep the stored types); conversion cannot restore precision lost
in stored state. Scientific tolerances must be qualified for each pair.

### Numerical options

`--mass-matrix scalar` is the default deposition without matrix cores. `--mass-matrix matrix-core` selects the
FP64 `m8n8k4` inline PTX kernel on CUDA compute capability 8.0 and higher, or a four-quadrant FP64 MFMA4 kernel on
the HIP wave64 devices listed under [device builds](#device-builds). It needs FP64 mass compute (the compute
levels `double`, `single-position`, `single-position-velocity` and
`single-particles-moments-fields-double-mass`) and `full` or `partial` order; in `partial` order the arrivals use
the scalar deposition. Startup rejects it unless the device supports it and the build holds a kernel image for
the device's architecture. Both matrix kernels accumulate the cell sums in FP64 and convert to mass storage once
per cell.

`--mass-apply full|split` selects the field-solver schedule. `split` applies the interior mass rows while the
input halo exchange proceeds, then the shell. With active smoothing both values run the smoothed full-domain
schedule; the report prints the request and the effective schedule (`Mass apply: split; effective: smoothed
full-domain`). [The numerical reference](../developer/numerical_reference.md) gives the algebra of both options.

## Ranks and GPUs

The MPI size must equal `XLEN*YLEN*ZLEN` ([decomposition](inputs.md#decomposition)). Run one rank per GPU. Each
rank selects its device before MPI starts:

- a launcher that exposes exactly one GPU to the process gives that GPU;
- otherwise the device is the node-local rank modulo the visible GPU count, with the local rank taken from
  `OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID` or `SLURM_LOCALID` (the MPI launcher values win over
  `SLURM_LOCALID`; disagreeing launcher values reject startup), and checked against the MPI node-local rank.

With `TPIC_COMMUNICATION_BACKEND=MPI` ranks may share a GPU; NCCL and RCCL need a distinct GPU per node-local
rank. MPI must provide `MPI_THREAD_MULTIPLE`, and with the MPI payload backend it must accept device pointers
(for Cray MPICH: `MPICH_GPU_SUPPORT_ENABLED=1` and `MPICH_MAX_THREAD_SAFETY=multiple`). Every rank reads the deck,
and startup stops when the bytes differ from those of rank 0. All ranks must receive the same command-line
options and run executables of the same build. Each rank runs host worker threads besides its main thread (the
*Memory plan* prints their count): give every rank several CPU cores close to its GPU.

A generic launch on two nodes with four GPUs each, for a deck with `XLEN*YLEN*ZLEN = 8`:

```sh
srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=8 \
  build/cuda/tpic --run input/gaussianExpansion3D.inp
```

[Cluster scaling](../../performance/scaling/README.md) holds the Slurm scripts, per-system settings and CPU
bindings; known platform issues are in [support](support.md).

## First run

From the TPIC root, after a device build:

```sh
build/cuda/tpic --check-input input/uniformStartup.inp
mpiexec -n 1 build/cuda/tpic --run input/uniformStartup.inp
```

[uniformStartup.inp](../../input/uniformStartup.inp) is a periodic 7 x 3 x 2 box with three species of
2 x 2 x 2 particles per cell (1008 particles) and three timesteps at `dt = 0.125`, `th = 0.5`. The check prints
`TPIC host input check: passed`, the plan of each species, the face rules, the output plan
(`fields=disabled, particles=disabled`), the *Numerical resolution* section and the note
`Linear ECSIM uses GMRES restart=30 and maximum iterations=100`, and exits with code 0.

The run prints the configuration report, the memory plan, one block per cycle and the final state (sections
below). Every output cadence of the deck is 0, so it creates the empty directory `data` (`SaveDirName`, relative to the
working directory) and ends with `No output files written.` With `th = 0.5` the reported energy change stays at
rounding level. To see output, set `FieldOutputCycle = 1` and `DiagnosticsOutputCycle = 1` in a copy of the deck:
`data` then receives `diagnostics.csv` and one `.vti` file per array and cycle (`E`, `B`, and `rho` and `J`
for the total and for every species), and *Simulation complete* names the last files. A second run into the same
directory stops with `use --overwrite-output or a fresh output directory`. [Output](output.md) describes the
files and their readers. For two ranks set `XLEN = 2` and launch `mpiexec -n 2`; the seven x cells split 4 + 3.

## Console report

`--check-input` prints the host input check: the case, its particle recipe and the particle positions, the
precision pair and its storage and compute roles, the particle order, the mass-matrix kernel and schedule, the
rank's local cells and particle counts per species, a lower bound of the particle device bytes, the field and
particle face rules, smoothing and GMRES settings, the output plan, the *Numerical resolution* section, then every
`Note:`, `Warning:` and `Unsupported:` line.

`--initialize` and `--run` print on rank 0:

| Section | Content |
| --- | --- |
| *Simulation setup* | input file, `SimulationName` and case, device and payload backends, process grid, cells, domain and cell size, `dt`, `th`, cycles, final time, `c` |
| *Precision and particles* | precision pair and roles, mass kernel and schedule, particle order, per-species counts, `qom`, `rhoINIT`, thermal speeds, drifts and velocity centring |
| *Boundaries and solver* | the field and particle rule of every face, smoothing, GMRES tolerance, restart length and iteration limit |
| *Numerical resolution* | see below |
| *Output* | directory, overwrite policy, the cadence of every channel, engines, checkpoints, the restart source |
| *Notes and warnings* | every `Note:` and `Warning:` line, including the unknown deck keys |
| *Memory plan* | per rank: planned device owners, transport and runtime allowances, host output buffers, worker and stream counts; per physical GPU: measured free memory and the combined plan of the ranks that share it |
| *GPU memory after initialization* | measured used, free and total memory per rank, including other processes and driver allocations |

The memory reports are informational: the plan excludes hypothetical particle growth, an estimate never rejects
a run, and an unavailable estimate does not prevent the real allocations.

During `--run` each cycle prints its header `ECSIM Cycle N  |  t = ...`, the field, kinetic and total energy with
the change relative to cycle 0 (every `DiagnosticEnergyCycle` cycles, before that cycle's solve), the GMRES
iteration count and residual, and the cycle wall time of rank 0. A Gauss-law sample prints a `Gauss residual:`
line ([output](output.md)). The loop ends with *ECSIM Main Loop Complete* (total, steady-state and per-cycle
times; the first cycle is the warm-up and is excluded from the steady-state figures), then *ECSIM Final State*
with the final energies and *Simulation complete*, which lists the particle counts and the last output files.
`--initialize` ends with *Initialization complete*.

*Numerical resolution* compares the time step and the cell with the scales of each species. A length is printed
in cells: the length divided by the cell size along x, y and z, which is the number of cells that resolve it on
each axis. A length that is one number is printed first: `Skin depth c/omega_p = 0.125 = 1.875 x 1.875 x 0.125
cells` is a skin depth of 0.125 that spans 1.875 cells along x and y and an eighth of a cell along z.

| Line | Quantity |
| --- | --- |
| `Light step c*dt` | `c*dt` and its size in cells, then `c*dt*sqrt(sum 1/cell^2)` over the axes with more than one cell |
| `Largest initial \|B\|` | the largest magnitude of the analytic initial field, uniform background included, on a lattice of at most 33 points per axis |
| `omega_p*dt` | `omega_p = sqrt(\|qom\|*rhoINIT)` ([units](inputs.md#units-and-normalization)) |
| `Omega_c*dt` | `Omega_c = \|qom\|*B/c` with that largest `\|B\|` |
| `Skin depth`, `Debye length` | `c/omega_p` and `vth/omega_p`, the latter with `uth`, `vth`, `wth` along x, y, z |
| `Gyroradius` | `vth_perp/Omega_c`, with the thermal speed per direction perpendicular to that field |
| `Thermal step`, `Drift step` | `vth*dt` and `\|v0\|*dt` along each axis |

The section uses the deck values of each species (`rhoINIT`, thermal speeds, drifts): a profile that a case sets
itself, such as the `fluxRope` thermal speeds, is not included, and a species with zero thermal speeds prints no
Debye length, gyroradius or thermal step. The numbers are reported only; admission sets no limit on them.

## Environment variables

| Variable | Effect |
| --- | --- |
| `TPIC_PROFILE_STAGES=1` | every cycle also prints a `Detailed timing cycle` line of host wall and device stream times per region, and the run ends with the *TPIC detailed timing summary (rank-local)*: for all cycles, ordinary cycles and output cycles, one *Host wall* and one *Device stream elapsed* table with Region, Calls, Total, Avg, Min, Max and StdDev in milliseconds |
| `TPIC_DIAGNOSTIC_SYNC_DEPOSITION=1` | runs the deposition synchronously, for debugging overlap |

Host wall scopes nest and device stream rows overlap, so rows are not additive. Profiling records extra device
events; see [performance](../developer/performance.md) before comparing timed runs.
