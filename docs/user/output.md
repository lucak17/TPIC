# Output

A run writes field snapshots, particle snapshots, two scalar CSV logs and, when asked, checkpoints. This page
describes what a user selects and reads. The writer, the epochs and the checkpoint internals are in
[output architecture](../developer/output_architecture.md); every key with its default is in
[input keys](input_keys.md); the analyzers that read the files are in [analysis](analysis.md).

`SaveDirName` (default `data`) is resolved relative to the run working directory; rank 0 creates it when
`--run` starts. `--check-input` and `--initialize` create no simulation output.

## Cadences

| Key | Default | Selects |
| --- | --- | --- |
| `FieldOutputCycle` | 100 | field and moment snapshots every that many cycles |
| `ParticlesOutputCycle` | 0 | particle snapshots every that many cycles |
| `DiagnosticsOutputCycle` | `FieldOutputCycle` | rows of `diagnostics.csv` |
| `DiagnosticEnergyCycle` | 1 | the console energy report; 0 reports only the final state |
| `DiagnosticGaussCycle` | 0 (off) | Gauss-law residual samples, rows of `gauss.csv` |
| `RestartOutputCycle` | 0 (off) | checkpoints ([below](#checkpoints-and-restart)) |

A nonpositive cadence disables its channel. Every enabled snapshot channel is also written at the final state,
whatever its cadence; a zero-cycle run still has a final state. Cycles with neither an energy report nor a
`diagnostics.csv` row skip the energy reductions, except the first state of a fresh run, which sets the initial
energy.

A snapshot at cycle `n` holds the state before that cycle's solve, at time `n*dt`; the final snapshot holds the
completed run. Its metadata records the cycle, the physical time, whether it is the final state, and the
precision profile and roles of the run. E and B are written at the nodes.

## Field arrays and tags

`FieldOutputTag` and `MomentsOutputTag` select the Float64 arrays of a field snapshot. With both tags empty, `E`,
`B`, and the total and per-species `rho` and `J` are written. Setting either tag turns off this default for both,
and pressure is written only when it is named.

Both tags are tokens joined by `+`, without spaces, each matched whole and case-sensitively:

- `FieldOutputTag`: `B`, `E` and `Gauss`. `Gauss` has no effect without a positive `FieldOutputCycle`.
- `MomentsOutputTag`: a quantity `rho`, `J`, `PXX`, `PXY`, `PXZ`, `PYY`, `PYZ` or `PZZ`, alone, followed by
  `_tot`, or followed by a decimal species index. `<name>_tot` writes the total (array `<name>`), `<name><s>`
  writes species `s` (array `<name>_<s>`), and a bare `<name>` writes every species, unless the tag also holds
  an in-range indexed token of the same name: `rho+rho0` writes only `rho_0`. An index at or above `ns`
  selects nothing; an index beyond the signed 32-bit range is an error.

Any other token, an empty token (`B+`, `rho++J`) or a token with spaces rejects the deck during output
preparation, with the key and the token named: `EB`, `none`, `rhoJ`, `rho-1` and `PXX_totally` are errors.

`ParticlesOutputTag` has no effect: a particle snapshot always holds every particle with its `position`,
`velocity`, signed macrocharge `charge`, UInt64 `id` and `species`. Test particles are not implemented
(`TestParticlesOutputCycle > 0` is an unsupported request).

### Moments

Output moments are deposited from the particles of the written state with the node weight `W` and divided by
the node's dual volume `V`, with `q` the signed macrocharge:

- `rho = 4*pi*sum(q*W)/V`: 4*pi times the charge density;
- `J = sum(q*v*W)/V`;
- `Pij = sum((q/qom)*v_i*v_j*W)/V`: the raw mass-weighted second moment, with no bulk flow subtracted.

These moments are separate from the solver's current ([numerical reference](../developer/numerical_reference.md)).

### The Gauss array

`Gauss` is the per-cell Gauss-law residual `r` of `gauss.csv` ([below](#gausscsv)), written as cell data. The
array of the snapshot at cycle `n` is the residual of the sample begun at `n - 1`, centred at `(n - 1/2)*dt`
(its `gauss.csv` row), while the file metadata keeps the snapshot time `n*dt`. Selecting it begins a sample one
cycle before every field snapshot, the final state's included, in addition to the `DiagnosticGaussCycle`
samples. The first snapshot of a run has no `Gauss` array, because no sample has finished at its cycle.

## Engines

`WriteMethod` selects the engine of both snapshot channels; `FieldWriteMethod` and `ParticlesWriteMethod`
override it per channel when nonempty. Without any method, VTK is used, and an explicit channel value
`default` selects VTK even when `WriteMethod` names another engine. Values are case-sensitive; an unknown
value of any of the three keys, or of `RestartWriteMethod`, is an input error even when its channel is
disabled. Requesting an engine the build lacks rejects a run before device startup, only when that channel is
enabled.

| Value | Build | Fields | Particles |
| --- | --- | --- | --- |
| `vtk`, `pvtk`, `default` or none | always | one `<array>_tNNNNNN.vti` per array, written by all ranks together | `particles_tNNNNNN.pvtp` and one `particles_tNNNNNN_rRRRRRR.vtp` per rank |
| `adios2`, `bp`, `bp5` | `TPIC_ENABLE_ADIOS2=ON` | `fields_tNNNNNN_rRRRRRR.bp` per rank, index `fields_tNNNNNN.adios2.json` | `particles_tNNNNNN_rRRRRRR.bp` per rank, index `particles_tNNNNNN.adios2.json` |
| `hdf5`, `h5` | `TPIC_ENABLE_HDF5=ON` | `fields_tNNNNNN_rRRRRRR.h5` per rank, index `fields_tNNNNNN.hdf5.json` | `particles_tNNNNNN_rRRRRRR.h5` per rank, index `particles_tNNNNNN.hdf5.json` |

`NNNNNN` is the cycle and `RRRRRR` the rank, each with at least six digits. Only the channels due at a cycle
are written. For example, VTK fields every ten cycles and ADIOS2 particles every fifty:

```text
FieldWriteMethod=vtk
ParticlesWriteMethod=adios2
FieldOutputCycle=10
FieldOutputTag=B+E
ParticlesOutputCycle=50
```

With `WriteMethod=hdf5` and no channel override, rank 0 at cycle 0 writes `fields_t000000_r000000.h5` and
`particles_t000000_r000000.h5`, indexed by `fields_t000000.hdf5.json` and `particles_t000000.hdf5.json`.

## File layouts

### VTK

A field file is single-piece Float64 ImageData over the whole global node grid, `nxc+1` by `nyc+1` by `nzc+1`
nodes with the end node of periodic axes included, x fastest, vector components interleaved, as raw appended
data with a UInt64 byte count. Its FieldData holds `TimeValue`, `cycle`, `final_state`, `state` (`pre_solve`
or `final`), `precision_profile` and `precision_roles`. Each node is written once: a rank writes the lower node
of each of its cells and the last rank along an axis also the end node, so a node on a rank seam holds the
upper rank's value. A cell array (`Gauss`) is the file's CellData, `nxc` by `nyc` by `nzc` values. A field file
is complete when its collective write returns and has no index.

A particle snapshot is a `.pvtp` manifest that lists one PolyData `.vtp` piece per rank. A piece holds the
species of its rank one after the other: the points `position` (global coordinates) and the point data
`velocity` and `charge` as Float64, `id` and `species` as UInt64.

### ADIOS2 and HDF5

Fields and particles are separate files, one piece per rank, and every rank piece of a snapshot is listed in
the channel's JSON index, which rank 0 writes after all ranks have completed their pieces. Read an index only
after it exists, then load its rank pieces with the HDF5 or ADIOS2 libraries; deduplicate the shared nodal
endpoints when combining ranks.

- Fields: `fields/<name>` with shape `(z, y, x, components)`, a final size-one dimension for scalars, vector
  components interleaved at each node, from the same host values as VTK. Each spatial dimension includes both
  local nodal endpoints; the `Gauss` cell array has the local cells, one sample fewer per axis.
- Particles: `particles/species_<index>/{position,velocity,charge,id}` with shape `(count, 3)` or `(count)`.
  Each species has a count even when empty; HDF5 keeps empty datasets and ADIOS2 omits empty payload
  variables.
- Metadata: global and local mesh extents, spacing, time, rank, channel presence and precision.
- Format versions: HDF5 files and ADIOS2 particle files carry the schema `tpic-output-v1`, ADIOS2 field files
  `tpic-output-v3`, and the JSON indexes `tpic-output-v1`.

The TPIC analyzers read the VTK files only.

## Viewing in ParaView

VTK. Open the `.vti` field files and the `.pvtp` particle manifests. ParaView groups the files of one array,
such as `E_t*.vti`, into one time series timed by their `TimeValue`; `E` and `B` are separate sources, which
**Append Attributes** combines.

HDF5 with XDMF. The `.h5` pieces and the `.hdf5.json` indexes are not a ParaView mesh; an XDMF file describes
the grid and the arrays. With `h5py` installed, from the TPIC root:

```sh
python3 tools/postprocessing/hdf5_fields_to_xdmf.py /path/to/output-directory
```

This writes `fields.xdmf` from the published `fields_t*.hdf5.json` indexes; it points to the HDF5 arrays without
copying or changing them, and covers every published time and rank piece. A single field `.hdf5.json` index
can be passed instead, for one snapshot (`fields_tNNNNNN.xdmf`). Open the `.xdmf` file, select **Xdmf3 Reader S**
if prompted, and click **Apply**; select **Surface** or add a **Slice**, then a field and **Magnitude**, **X**, **Y**
or **Z** for vectors. Keep the descriptor beside its HDF5 files. It does not update itself: rerun the command with
`--overwrite` to include snapshots published later. A `Gauss` array is marked `Center="Cell"`.

ADIOS2 with Fides. Open one `fields_tNNNNNN_rRRRRRR.bp` directory and select **ADIOS2 BP4/5 Directory (using
Fides)** explicitly; in Python use `paraview.simple.FidesReader`. The default selection, including Python's
`OpenDataFile`, may choose the generic ADIOS2 image reader. The `fides/schema` attribute of the piece exposes
the interleaved arrays as vectors (E, B and the currents offer **Magnitude**, **X**, **Y**, **Z**) and lists
`Gauss` as cell data; each piece supplies its origin, spacing and time. These readers accept one file name:
open other rank pieces as separate sources. The field pieces carry no `vtk.xml` for the VTX reader, which does
not support the interleaved layout; the particle `.bp` files hold particle arrays only. The `.adios2.json` file
is a rank index, not a ParaView input.

## Scalar logs

Both logs live in `SaveDirName`, start with `#` comment lines (the precision profile and roles, then the column
definitions) and have one header row.

### diagnostics.csv

Rank 0 writes it when `DiagnosticsOutputCycle` is positive: one row per multiple of it and one for the final
state, with the columns `cycle`, `time`,
`state`, `electric_energy`, `magnetic_energy`, `kinetic_energy`, `total_energy`, `total_energy_change`,
`charge`, `momentum_x`, `momentum_y`, `momentum_z`, then `kinetic_energy_<s>`, `charge_<s>`,
`momentum_x_<s>`, `momentum_y_<s>`, `momentum_z_<s>` per species. Values are globally reduced in double
precision. `total_energy_change` is `total_energy` minus that of the run's first state, in absolute units. The
comment lines give the energy and momentum definitions.

### gauss.csv

A Gauss-law residual sample begins at every multiple of `DiagnosticGaussCycle` below `ncycles` and finishes at
the next cycle (the final state included); each finished sample writes one row and prints one
`Gauss residual` line. A row holds the begin cycle `n`, the time `(n + 1/2)*dt` and four root mean squares over
the global cells:

| Column | RMS of |
| --- | --- |
| `residual_rms` | `r = (divC(E^n) + divC(E^{n+1}))/2 - c` |
| `drift_rms` | `r` minus the run's first finished sample |
| `wall_residual_rms` | `r` over the cells touching a physical wall (0 without walls) |
| `charge_rms` | `c` |

`divC` is the solver's node-to-cell divergence and `c` is `4*pi` times the eight-node mean of the charge density
of the particles at `x^{n+1/2}`, dual-volume scaled, zero on PEC wall nodes and smoothed like the solver's
current. A sample costs one charge deposition of all species. The construction is in
[Gauss diagnostic](../developer/gauss_diagnostic.md).

## Host memory and the writer

Snapshots are written by persistent host worker threads while the simulation advances: one per rank for
fields and particles and one for checkpoints. The particle records of a snapshot are copied from the device
into one pinned host image per species, held once in host RAM and not on the GPU; the writer converts them to
physical positions, velocities and charges while it writes. The images are allocated when particle output or a
checkpoint first occurs, sized by the active particle counts:

| Storage precision | Host bytes per particle |
| --- | --- |
| `double` | 68 |
| `single-position` | 56 |
| `single-position-velocity`, `single-particles-moments-fields-double-mass`, `single-particles-moments-fields` | 44 |

The writer holds one snapshot per rank: slow storage delays the next snapshot rather than growing memory. A
snapshot of VTK fields alone needs no index and is written while the next cycles run.

## MPI-IO and file systems

VTK field files are written collectively through MPI-IO. TPIC sets no MPI-IO hints; they come from the MPI
library's environment (`ROMIO_HINTS`, `MPICH_MPIIO_HINTS` on Cray MPICH, the Open MPI `io` MCA parameters). On
Lustre a file takes the striping of its directory, so stripe `SaveDirName` before a run with a large grid, for
example `lfs setstripe -c <stripes> <SaveDirName>`. With Open MPI on a local file system the default MPI-IO
component writes the rows of each rank one by one; `OMPI_MCA_fcoll=vulcan` or `OMPI_MCA_io=romio341` selects
collective buffering, which writes faster. Particle snapshots and the ADIOS2 and HDF5 pieces are written per
rank without MPI-IO; their file count grows with ranks and snapshots.

## Overwrite protection

A run refuses to replace an existing output file. `--overwrite-output` permits a fresh `--run` to replace the
files it writes; it neither clears the directory nor resumes a run.

## Checkpoints and restart

`RestartOutputCycle` writes a checkpoint of the complete state every that many cycles, never at the first state
of a run; `CallFinalize` (default `true`) also checkpoints the final state when `RestartOutputCycle` is positive. A
checkpoint holds E at the nodes, B at the cells and every particle record, in their stored types; everything
else is rebuilt at restart. Its particle records come from the same host image as particle output, so
checkpoints add no host copy of the particles.

`RestartWriteMethod` selects `adios2` (`bp`, `bp5`) or `hdf5` (`h5`); `default` or an empty value names ADIOS2
when it is built, else HDF5. A build with neither warns and writes no checkpoints; `vtk` is an unsupported
request. A checkpoint is the directory `<RestartDirName>/checkpoint_tNNNNNN_nRRRRRR/` (cycle and rank count;
`RestartDirName` defaults to `data`) with one `checkpoint_tNNNNNN_rRRRRRR.bp` or `.h5` piece per rank, schema
`tpic-checkpoint-v1`. A rank writes its piece under a hidden name, syncs it and renames it, so a checkpoint is
complete when all its pieces are listed. TPIC manages `RestartDirName`: at the start of each checkpoint, rank 0
deletes the checkpoints older than the newest complete one, so the disk holds one complete checkpoint and the
one being written.

`--restart` continues `--run` or `--initialize` from the newest complete checkpoint, on any number of ranks and
any process grid; each rank reads only the records of its own cells. `ncycles` stays the last cycle of the run,
so resubmitting the same deck finishes it; it must lie after the checkpoint cycle. A restart may change the rank
count, the process grid, the compute precision, the numerical options and the outputs. A different mesh,
domain, field or particle boundary, `dt`, `c`, species (`ns`, `qom`) or storage precision is an unsupported
request. Restored velocities are not centred again.

The restarted run replaces the outputs of the run it continues from the checkpoint cycle on. It continues
`diagnostics.csv` from that cycle after a `# restart at cycle N` comment line, with `total_energy_change` still
relative to the first state of the first run; `gauss.csv` continues the same way, and its `drift_rms` takes the
first sample after the restart as its baseline. Checkpoints newer than the restart's are set aside
as `abandoned_<name>` and deleted at the next checkpoint. A fresh run that writes checkpoints refuses a
`RestartDirName` that holds checkpoints of an earlier run, unless `--overwrite-output` sets them aside the same way.
