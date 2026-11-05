# TPIC cluster scaling

Strong- and weak-scaling campaigns of the GEM case, one folder per GPU system.

```text
performance/scaling/
  runScalingCommon.sh      shared launcher and job worker
  scalingStrong.py         plot a strong campaign
  scalingWeak.py           plot a weak campaign
  scalingPlotCommon.py     shared parsing and plotting
  mi250x/                  LUMI-G: 8 MI250X GCDs per node, 64 GB each
    system.sh              ranks per node, Slurm resources, srun command
    selectGPU.sh           exposes one GCD to each rank
    strong/                runTPICStrong.slurm, GEMstrongN<nodes>.inp
    weak/                  runTPICWeak.slurm, GEMweakN<nodes>.inp
  a100/                    MeluXina: 4 A100 per node, 40 GB each
    system.sh
    strong/
    weak/
```

Every job runs one MPI rank per GPU:

```text
tpic --run INPUT --order partial --precision double \
  --mass-matrix matrix-core --mass-apply split
```

## Systems

| Folder | Cluster | GPUs per node | GPU memory | Slurm request (`system.sh`) | CPU binding |
| --- | --- | --- | --- | --- | --- |
| `mi250x` | LUMI-G | 8 GCDs | 64 GB | `--partition=standard-g --gpus-per-node=8` | six cores per rank next to its GCD |
| `a100` | MeluXina | 4 | 40 GB | `--partition=gpu --qos=default --gpus-per-node=4 --cpus-per-task=32` | Slurm default for 32 CPUs per rank |

For Cray MPICH, `mi250x` sets `MPICH_GPU_SUPPORT_ENABLED=1`,
`MPICH_MAX_THREAD_SAFETY=multiple` and `MPICH_GPU_IPC_CACHE_MAX_SIZE=1024`, and
gives each rank one GCD through `selectGPU.sh`. The IPC cache size avoids a
GPU memory fault on a peer GCD at scale
([support](../../docs/user/support.md#known-platform-issues)). `a100`
needs none of these: TPIC selects the GPU of a rank from its node-local rank.

Build with the preset of the backend and the architecture of the GPU, as in
[build and run](../../docs/user/build_and_run.md): `hip` with
`-DLPFX_DEVICE_ARCHITECTURES=gfx90a` for the MI250X, `cuda` with
`-DLPFX_DEVICE_ARCHITECTURES=80` for the A100.

## Test cases

A rank at full load owns 64x64x64 cells of a 10x10x10 domain and four species.
The A100 decks keep the grid and reduce the particles per cell to 40/64 of the
MI250X decks, so both fill the same fraction of the GPU memory.

| | `mi250x` | `a100` |
| --- | --- | --- |
| `npcelx` x `npcely` x `npcelz` per species | 7 x 8 x 8 = 448 | 7 x 8 x 5 = 280 |
| Particles per rank | 469 762 048 | 293 601 280 |
| TPIC startup memory plan per rank | 59.8 GiB of 64 | 37.6 GiB of 40 |

The memory plan is the `Startup plan + allowances` line that every job prints
in its *Memory plan* section, next to the measured free and total GPU memory.
Weak decks hold this load on every rank; strong decks hold it at their
smallest node count and split the same global grid further. Every deck runs
101 cycles, one warm-up cycle and 100 measured ones. No deck writes output,
and `DiagnosticEnergyCycle = 0` keeps energy reductions out of the
steady-state cycles.

Weak scaling, `<system>/weak/GEMweakN<nodes>.inp`:

| GPUs | MPI topology | Grid cells | `mi250x` | `a100` |
| --- | --- | --- | --- | --- |
| 4 | 2x2x1 | 128x128x64 | | N1 |
| 8 | 2x2x2 | 128x128x128 | | N2 |
| 16 | 4x2x2 | 256x128x128 | | N4 |
| 32 | 4x4x2 | 256x256x128 | N4 | N8 |
| 64 | 4x4x4 | 256x256x256 | N8 | N16 |
| 128 | 8x4x4 | 512x256x256 | N16 | N32 |
| 256 | 8x8x4 | 512x512x256 | N32 | N64 |
| 512 | 8x8x8 | 512x512x512 | N64 | N128 |
| 1024 | 16x8x8 | 1024x512x512 | N128 | |
| 2048 | 16x16x8 | 1024x1024x512 | N256 | |
| 4096 | 16x16x16 | 1024x1024x1024 | N512 | |

Strong scaling, `<system>/strong/GEMstrongN<nodes>.inp`:

| | `mi250x` | `a100` |
| --- | --- | --- |
| Grid cells | 256x256x256 | 128x128x128 |
| Domain | 40x40x40 | 20x20x20 |
| Node counts | N8 to N512 | N2 to N128 |
| GPUs | 64 to 4096 | 8 to 512 |
| Cells per rank | 64x64x64 down to 16x16x16 | 64x64x64 down to 16x16x16 |

At equal GPU count both systems use the same topology. The plot scripts
normalize to the smallest plotted node count; use `--min-nodes` to choose it.

## 1. Set the required paths

Run all commands from the `scaling` directory.

```sh
export TPIC_SLURM_ACCOUNT='YOUR_PROJECT_ACCOUNT'
export TPIC_PROFILE='/shared/absolute/path/tpic-runtime-profile.sh'
export TPIC_EXECUTABLE='/shared/absolute/path/TPIC/build/hip/tpic'   # build/cuda/tpic on a100
```

`TPIC_PROFILE` is sourced on the compute nodes and must load the compiler, GPU
runtime (ROCm or CUDA) and GPU-aware MPI modules used to build TPIC. All three
values must be available from the compute nodes, and the executable basename
must be `tpic`.

The profile must not export `OMP_PROC_BIND` or `OMP_PLACES`: TPIC has no
OpenMP region, and a GNU OpenMP runtime linked into the executable pins every
TPIC host thread to one core when they are set.

Put `ulimit -c 0` in the profile: a crashed rank can otherwise write a core
file of tens of GiB, one per rank, into the run folder.

## 2. Check a submission

A dry run validates the paths and inputs and prints the `sbatch` command. It
does not create files or submit jobs.

```sh
./mi250x/strong/runTPICStrong.slurm --dry-run 8
./mi250x/weak/runTPICWeak.slurm --dry-run 4

./a100/strong/runTPICStrong.slurm --dry-run 2
./a100/weak/runTPICWeak.slurm --dry-run 1
```

## 3. Submit jobs

The commands below use `a100`; replace the folder for another system. Start
with the smallest job and check its *Memory plan* and
*GPU memory after initialization* sections:

```sh
./a100/strong/runTPICStrong.slurm 2
./a100/weak/runTPICWeak.slurm 1
```

Two node counts select every available input in that inclusive range:

```sh
./a100/strong/runTPICStrong.slurm 4 32
./a100/weak/runTPICWeak.slurm 2 32
```

Submit all supplied node counts:

```sh
./a100/strong/runTPICStrong.slurm --all
./a100/weak/runTPICWeak.slurm --all
```

Optional overrides:

```sh
./a100/weak/runTPICWeak.slurm --qos test 1
./mi250x/strong/runTPICStrong.slurm \
  --time 00:45:00 --exclude 'nid[0010-0012],nid0020' 16 128
```

`--qos` replaces the QOS of `system.sh` (`default` on `a100`, none on
`mi250x`); use the QOS the site requires for a test job or for a large node
count. The default time limit is 30 minutes for strong and 20 minutes for weak
jobs. `--help` lists every option.

Each submission prints its campaign directory,
`<system>/<kind>/results_tpic_TIMESTAMP/`. It holds a copy of the executable
and of the scripts, `manifest.txt`, one run folder per node count and the job
output.

## 4. Check completion

Replace the path with the directory printed during submission:

```sh
campaign='a100/strong/results_tpic_TIMESTAMP'
grep '^job_N' "$campaign/manifest.txt"

job_ids=$(awk -F= '$1 ~ /^job_N/ { ids = ids sep $2; sep = "," } END { print ids }' \
  "$campaign/manifest.txt")
sacct -j "$job_ids" --format=JobID,JobName%28,State,Elapsed,ExitCode

grep -EH 'Steady-state time:|TPIC_SCALING_COMPLETE' "$campaign"/stdout_N*
```

A run can be plotted once its stdout contains `TPIC_SCALING_COMPLETE`. The
plot scripts do not need the whole campaign: they plot every completed run and
print a `Warning:` line for an interrupted submission, a planned node count
without a completed run, a run they skip and a failure line in the output.

## 5. Plot on the cluster

Activate a cluster Python 3.10+ environment containing Matplotlib, then run
from the `scaling` directory:

```sh
python3 -c 'import sys; assert sys.version_info >= (3, 10); import matplotlib'

python3 scalingStrong.py a100/strong/results_tpic_TIMESTAMP
python3 scalingWeak.py a100/weak/results_tpic_TIMESTAMP
```

The scripts read the ranks per node and the GPU axis label from the campaign
manifest, so the same commands plot every system.

To plot only part of a campaign:

```sh
python3 scalingStrong.py a100/strong/results_tpic_TIMESTAMP \
  --min-nodes 4 --max-nodes 32
```

To leave out specific node counts, for example a completed run slowed by a bad
node, list them in `--skip-nodes`. It takes comma-separated counts, can be
repeated, and applies to every campaign:

```sh
python3 scalingStrong.py a100/strong/results_tpic_TIMESTAMP \
  --skip-nodes 8,32
```

To combine non-overlapping node counts from several campaigns of one system,
repeat `--add-campaign`. Each folder is read with its own manifest, and the
plot is written to the first campaign folder:

```sh
python3 scalingWeak.py a100/weak/results_tpic_TIMESTAMP_A \
  --add-campaign a100/weak/results_tpic_TIMESTAMP_B
```

The scripts use `Steady-state time`, which excludes the first cycle, and write
`strong_scaling_efficiency.png` or `weak_scaling_efficiency.png` inside the
campaign directory.

They also write `<kind>_scaling_time_efficiency.png` and
`<kind>_scaling_time_efficiency.pdf` (`<kind>` is `strong` or `weak`). This
title-free plot shows the parsed execution time in seconds on the left axis
(circles, solid line) and scaling efficiency on the right axis (triangles,
dashed line). The efficiency axis starts at 0% and ends at the greater of 110%
or 1.1 times the maximum efficiency.

Edit the `FontSizes` defaults near the top of `scalingPlotCommon.py` to control
the font sizes for titles, axis labels, legends, ticks and annotations in all
plots. Sizes are in points.

## Production decks at scale

The scaling decks write no output. A production deck from `../input/` run at
these particle counts writes very large particle snapshots: each one holds
every particle of the run, tens of bytes per particle, so terabytes at the
loads above. Check `ParticlesOutputCycle` and the free space of the target file
system before such a run, or set `ParticlesOutputCycle = 0`.

## Adding a node count or a system

A deck is named `GEM<kind>N<nodes>.inp` and its `XLEN*YLEN*ZLEN` must equal the
node count times the ranks per node. Within one folder the decks may differ
only in the topology and, for weak scaling, in `Lx`, `Ly`, `Lz`, `nxc`, `nyc`
and `nzc`, with the cells and the domain length per rank kept fixed; the
launcher rejects anything else. It also requires `Case = GEM` with `B0x = 0`
and a nonzero `HarrisB0x`, a grid that
divides evenly over the topology with at least two cells per rank on each
axis, and zero `FieldOutputCycle`, `ParticlesOutputCycle`,
`RestartOutputCycle` and `DiagnosticsOutputCycle`. A deck added here also
needs its entries in the golden tables of the deck tests
([testing](../../docs/developer/testing.md)).

A system is a folder with a `system.sh` that sets `TASKS_PER_NODE`,
`GPU_LABEL`, `SYSTEM_SBATCH_ARGS`, optionally `SYSTEM_QOS`, and the function
`system_srun`, plus `strong/` and `weak/` folders holding a copy of the
launchers and the decks sized for its GPU memory.
