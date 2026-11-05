# Support

What TPIC implements, where it is limited, and the platforms it runs on.
Building and launching are in [build and run](build_and_run.md).

## Supported scope

| Area | Implemented scope |
| --- | --- |
| Execution | CUDA or HIP device build; the host build checks inputs and has no CPU time step |
| Method | Linear ECSIM with GMRES field solve, `c = 1` |
| Species | 1 to `TPIC_MAX_SPECIES` (default 10) real species per run |
| Precision | Storage and compute levels `double`, `single-position`, `single-position-velocity`, `single-particles-moments-fields-double-mass`, `single-particles-moments-fields` |
| Particle order | `full`, `partial` (default), `unsorted` |
| Mass matrix | `scalar` kernel on every device; `matrix-core` kernel on FP64 matrix hardware (below) |
| Transport | GPU-aware MPI, CUDA/NCCL or HIP/RCCL for device payloads; MPI host control in all cases |
| Particle faces | Periodic or mirror |
| EM faces | Periodic, perfect electric conductor (PEC) or perfect magnetic conductor (PMC), each with optional energy-conserving smoothing |
| Output | CSV diagnostics; asynchronous parallel VTK, ADIOS2 BP5 or HDF5 fields, moments and real particles (ADIOS2 and HDF5 need their build options) |
| Checkpoints | ADIOS2 BP5 or HDF5; restart on any rank count and process grid |

## Known limits

- No CPU time step: `--initialize` and `--run` need a CUDA or HIP build.
- `c = 1` only: any other `c` is an unsupported request.
- Test particles (`nsTestPart`, `TestParticlesOutputCycle`) are not implemented
  and are rejected.
- Particle injection is not implemented: an injecting particle face (`bcPface`
  3 or 4) is rejected; a physical particle face admits only the mirror (1).
- Open EM faces are not implemented: `--check-input` admits them with a note,
  `--initialize` and `--run` reject them, and active energy-conserving
  smoothing rejects them in every mode.
- Species: loading gives every same-sign species the full wave current, so a
  `Whistler` run whose seed carries a wave current with more than one electron
  species, and a `TaylorGreenVortex` run with nonzero
  `TaylorGreenVortexMagneticAmplitude` and more than one electron or ion
  species, are rejected.
- Particle flight: on a decomposed run a particle moves at most two cells
  beyond its rank in one step; a farther flight stops the run, whatever
  `TPIC_PARTICLE_MIGRATION_HOPS` ([build and run](build_and_run.md)).
- `matrix-core` needs `--order full` or `partial`, double mass compute and an
  FP64 matrix device; a host build reports it as `Unsupported:` and
  `--check-input` exits with code 2.
- Memory figures (the startup memory plan and the `--check-input` particle
  bytes) are estimates, not a guarantee that allocation succeeds.
- Device-pointer support of the MPI library, distinct-GPU placement of ranks
  and MPI thread support must hold on the deployed runtime; the build cannot
  establish them. Tests whose ranks share one GPU do not cover overlap between
  remote GPUs or across a network.

## Communication libraries

The NCCL (CUDA) and RCCL (HIP) device payload transports need the minimum
versions in the [prerequisites](build_and_run.md#prerequisites).

## Tested platforms

| GPU | Architecture | System |
| --- | --- | --- |
| NVIDIA RTX 5080 | `sm_120` | Workstation |
| NVIDIA A100 | `sm_80` | MeluXina, Vega |
| NVIDIA GH200 | `sm_90` | |
| AMD MI250X | `gfx90a` | LUMI-G |

## Matrix-core mass kernels

`--mass-matrix matrix-core` assembles the mass matrix with FP64 matrix
instructions.

| Vendor | Kernel | Status |
| --- | --- | --- |
| NVIDIA | FP64 MMA, `sm_80` and newer | Compiles and runs on `sm_80` and newer; its device test checks every coefficient against host FP64 cell sums ([testing](../developer/testing.md)) |
| AMD | FP64 MFMA4 on wave64 | Compiles for `gfx90a`, `gfx942` and `gfx950` and runs on `gfx90a`; not numerically qualified or benchmarked on AMD hardware |

AMD products with FP64 matrix cores, which the MFMA4 kernel targets:

| Products | Architecture |
| --- | --- |
| MI210, MI250, MI250X | `gfx90a` |
| MI300A, MI300X, MI325X | `gfx942` (the kernel also accepts `gfx940` and `gfx941`) |
| MI350X, MI355X | `gfx950` |

MI100 (`gfx908`) and RDNA GPUs have no FP64 matrix cores; runs there use
`--mass-matrix scalar`, the default. Startup rejects `matrix-core` on a device
without them.

## Known platform issues

### Cray MPICH GPU IPC cache (LUMI-G)

- Symptom: a multi-node run stops with `Memory access fault by GPU node-N` on
  a peer GCD in the middle of the run.
- Cause: the default `MPICH_GPU_IPC_CACHE_MAX_SIZE` of 50 evicts the IPC
  mappings of the persistent exchange buffers, and a peer GCD then accesses an
  unmapped address. TPIC opens no peer access or IPC itself; Cray MPICH does.
- Setting: `export MPICH_GPU_IPC_CACHE_MAX_SIZE=1024`, as
  `performance/scaling/mi250x/system.sh` does. Runs with it complete.
- Diagnosis: `MPICH_GPU_IPC_ENABLED=0` disables GPU IPC in Cray MPICH, which
  separates an IPC fault from a TPIC fault; `AMD_SERIALIZE_KERNEL=3`
  serializes kernel launches so the fault is reported at the kernel that
  causes it.
