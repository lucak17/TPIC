# Output architecture

TPIC separates output selection, capture, scheduling and serialization. The same
host snapshot feeds VTK, ADIOS2 BP5 or HDF5. VTK fields are shared files that
all ranks write together through MPI-IO; every other channel writes rank-local
files. A serialization engine changes no deposition, kernel, particle advance
or MPI ordering. Keys, file names, layouts, ParaView steps and restart behaviour
are in [output](../user/output.md); the output tests are listed in
[testing](testing.md#output-tests).

## Design constraints

Serialization engines sit behind one writer facade and receive only host
snapshots: owned field arrays and particle records lent from the pinned capture
images. Disk writes run on a bounded LPFx worker fed by event-driven capture; a
snapshot handoff distinguishes event-record publication from GPU completion,
because the streams involved are driven by different CPU workers. Field
snapshots keep the logical nodal layout of a rank, including the endpoints it
shares with its neighbours. Existing output is protected unless replacement is
requested.

## Classes and ownership

| Class | Contract |
| --- | --- |
| `config::PreparedOutput` | Independent field and particle engine requests, directories, overwrite policy and cadences. Engine availability is checked for each enabled channel before a timestepping run. The moment selection is two arrays indexed like `data::output_moment_quantities`: the selected species and the selected total of each quantity. |
| `config::PreparedOutputSelection` | Immutable field descriptors, deposition masks, physical conversions, packing slots and scratch width, shared by capture, validation and memory accounting. |
| `diagnostics::AsyncFieldSnapshot` | Packs the selected nodal fields and moments, and the cell-located Gauss residual, into a detached device slab, transfers it into pinned memory, then fills reusable host vectors. An array omitted from a capture (the Gauss residual before its first finished sample) reaches the writers with no values, and every engine skips it. |
| `diagnostics::AsyncParticleSnapshot` | Copies each species' committed records as stored, one device-to-host copy per attribute into one pinned image per species, with no kernel and no device staging, and lends the images to the writers: one capture serves particle output and the checkpoint. One transfer event releases the sources and completes the images. |
| `data::OwnedOutputSnapshot` | Cycle, time, final-state status, independent channel presence flags, owned field arrays and per-species particle records borrowed from the capture's pinned images until the writer returns the snapshot. |
| `data::OutputSnapshotView` | Immutable host spans borrowed from the owned snapshot. Channel selection hides the other channel without copying payloads. |
| `io::AsyncOutputWriter` | The single owner of the epoch contract: checks the configure-time preconditions (an enabled channel, a directory, a prepared domain) and each snapshot's shape, enforces increasing cycles and times up to one final snapshot, transfers ownership to one persistent worker, holds the slot until the root has published (a snapshot of VTK fields alone until the controller joins its write), propagates failures and returns buffers for reuse after publication. The field capture rejects nonfinite values before it copies them into the snapshot; the engines reject invalid particle records while they convert them. |
| `io::OutputEngine` | Abstract `write_piece(snapshot)`, `publish()` and `manifest_path()`; `particles_path()` defaults to `manifest_path()` and is overridden by the routed engine, which exposes both channels. Its precondition is the writer's contract, so engines re-check none of it. It receives no device object, executor or simulation object; the VTK field engine alone borrows the field file writer. |
| VTK field and particle engines | Single-channel engines in `src/io/vtk_field_engine.cpp` (one shared `.vti` per array, written through `lpfx::distributed::MpiFileWriter`) and `src/io/vtk_particle_engine.cpp` (`.vtp` pieces and the `.pvtp` manifest). They keep only the VTK format limits: UInt64 appended offsets. |
| `io::detail::DatasetOutputEngine` | One ADIOS2 or HDF5 channel: prepared field and species paths, common metadata and payload visitation, channel file names and manifest publication. |
| `io::detail::CheckpointEngine` | One rank's checkpoint pieces: the root's retention, the cell index of each species' sorted prefix, and the hidden write, sync and rename of the piece. ADIOS2 and HDF5 supply `write_file` and a `CheckpointReader` for restarts. |
| `io::detail::OutputFile` | Checked streaming sink shared by the scalar CSV files, VTK particle files and dataset manifests; it centralizes safe creation, writes, flush and close errors. `write_text_file` writes a complete manifest. |
| Engine factory | `make_output_engine` routes each enabled channel to its engine and passes the channel explicitly to the backend factories (`make_vtk_field_engine`, `make_vtk_particle_engine`, `make_adios2_engine`, `make_hdf5_engine`). Optional libraries are linked only when their build options are on. |
| `simulation::OutputActions` | Combines the channels due at a physical state, prepares the captures and configures the two writers that `Pipeline` owns. |
| `io::detail::CycleCsvFile` | The rank-0 CSV file whose rows start with `cycle,time`: precision header, restart truncation and increasing rows, shared by `io::DiagnosticsWriter` (`diagnostics.csv`) and `io::GaussWriter` (`gauss.csv`). |
| `simulation::DiagnosticsActions` | Launches and collects the scalar diagnostics and the Gauss-law residual. It opens `diagnostics.csv` and `gauss.csv` on the main thread before the diagnostics/output worker starts, and writes and closes them on that worker. |

`data::OwnedOutputSnapshot` and `io::AsyncOutputWriter` are the record and
writer APIs; tests drive engines through the writer. Each engine writes one
channel. VTK field file names follow the selected array names and are the
readers' contract.

### Shared serialization structure

`io/detail/output_layout.hpp` defines the physical geometry and the particle
array descriptors. A particle descriptor binds its name, component count and
quantity once. VTK uses the descriptors in its Points/PointData order; dataset
engines visit the same descriptors in their dataset order. Neither traversal
creates a second payload image. `io/detail/vtk_output.hpp` holds the VTK XML
fragments both VTK engines share: the file prolog, the appended and parallel
array descriptors, the appended-data markers and the FieldData block.

The dataset base prepares field paths with `(z,y,x,component)` shapes and
species paths once. Its typed visitors describe root metadata and the field and
particle datasets. ADIOS2 supplies variable and attribute calls; HDF5 supplies
dataset, group and attribute calls with explicit file types. Format differences
stay in these adapters: HDF5 writes empty datasets, while BP5 represents an
empty species through its count attribute and omits its payload variables.

Output file names and checked streaming operations are shared; VTK XML encoding
and the HDF5 and BP library operations are format-specific. Dataset libraries
write into the private sibling directory of `StagedOutput` and close their file
before commit; their manifest uses `OutputFile` under the same staging policy.
The streaming sink keeps exclusive creation and symlink protection and reports
flush and close failures through the writer's exception path. VTK field files
are not staged: MPI-IO creates each one exclusively, or replaces it under
`--overwrite-output`.

## One output epoch

```mermaid
sequenceDiagram
    participant Main as Pipeline main thread
    participant Obs as Diagnostics/output worker
    participant GPU as Capture and transfer streams
    participant Disk as Persistent CPU writer
    Main->>Obs: Admit committed cycle (due channels)
    Obs->>GPU: Pack selected fields and/or particles
    GPU->>GPU: Record sources read; transfer detached buffers
    Obs-->>Main: Publish source-read event generation
    Main->>GPU: Wait on source-read events before overwriting sources
    Main->>Main: Run the next particle/field advance
    Obs->>GPU: Collect completed host transfer
    Obs->>Disk: Move owned snapshot
    Disk->>Disk: Serialize local pieces; write shared VTK field files collectively
    Disk-->>Obs: Publish completion or stored exception
    alt Snapshot publishes (particles, ADIOS2/HDF5 fields)
        Obs-->>Main: Local piece complete
        Main->>Main: Retire the advance; rank-common agreement
        Main->>Obs: Allow manifest publication
        Obs->>Obs: Root publishes manifest; reclaim host buffers
    else VTK fields alone
        Obs->>Obs: Join the writer at the next capture or final state; reclaim host buffers
    end
```

There are two lifetimes: simulation sources are protected only until packing or
copying has finished reading them; the detached field storage and the lent
particle images stay unchanged until serialization finishes. The record of the
source event is published on the host before another worker queues a wait on
that event. The transfer stream waits on the source event, and CPU code reads
pinned memory only after transfer completion.

### Field slab sizing

Field capture keeps its detached device slab, pinned host slab and owned host
vectors across epochs. A species contribution to a total is rounded to FP64
(`scale * value`) before it is added. On CUDA one fused kernel does both with a
`__dmul_rn` product, also storing the species array when it is selected
(`pack_and_add`), and needs no scratch. On HIP the contribution is packed and
then added in a second launch, because a contraction-free product is not
verified in its ISA; there contribution scratch is needed only when a requested
total includes a species quantity that has no selected individual output slot.
`PreparedOutputSelection::scratch_component_count()` supplies the largest such
width (zero, one scalar component or three vector components), and
`data::field_snapshot_scratch_components<Backend>` turns it into the width the
backend allocates.

For `N` local nodes, `C` selected components and allocated scratch width `S`,
the device slab holds `8*N*(C+S)` bytes and its pinned transfer image `8*N*C`
bytes; a cell array counts the local cells instead of `N`. The reusable owned
host vectors hold a separate field image; capture copies values from pinned
storage into them.

### Particle images

The complete particle snapshot resides in host RAM once. One pinned image per
species holds the committed records as stored: cell fractions, velocities and
charge in their storage types, the rank-local cell key and the id, which is
`ParticleSchema<Precision>::type::record_width` bytes per particle (68 in double
storage). The capture records a readiness event on the capture stream, which
has waited for the committed state; the transfer stream waits on it and issues
one device-to-host copy per attribute. The transfer event recorded after the
copies both releases the simulation sources and makes the images readable.
Images are allocated on capture for the active records and reused at their
observed high-water mark; a growth releases the old image before allocating the
larger one. They are lent to the writer through the snapshot and rewritten only
at the next capture, after the writer has returned the snapshot. Particle owner
capacity reserves no image.

The engines convert the records while they write them, in chunks of
`io::detail::particle_chunk_records` records (`io::detail::convert_particles`):
a position is `(offset + cell + fraction) * spacing` evaluated in double, the
same expression for every format; velocities and charge widen to double; ids
are written as stored. A cell key outside the interior, a fraction outside
[0, 1) or a nonfinite value fails the piece, so an invalid record never reaches
a file. No engine holds a second image of the population.

### Publication conditions

Snapshot publication has five separate conditions: source release, completed
host transfer, completed local disk piece, rank publication and slot reuse. The
diagnostics/output worker releases the physics readers (`diagnosed_fence`)
before it waits for the disk worker or for main's rank agreement. It then
publishes piece-ready, waits for main's publication permission, publishes the
manifest and takes back the reusable host arrays. `OutputActions::publishes` is
false for a snapshot of VTK fields alone: main neither waits for its piece nor
enters the agreement, and the diagnostics/output worker joins its write before
it hands the writer the next capture, and in the final epoch.

One snapshot is admitted at a time per rank. A second submission is rejected
until the caller has waited for the piece, completed rank agreement when the
snapshot publishes, and reclaimed the snapshot. The simulation schedules this
boundary explicitly, so slow output applies bounded backpressure and cannot
accumulate a queue of full field and particle images. Disk writing overlaps the
next advance; `Pipeline` retires an output epoch that publishes before it
admits the next solve.

The worker drains accepted work when destroyed, but destruction cannot report a
failure or publish distributed metadata. Normal shutdown waits for and completes
the final snapshot explicitly. Worker exceptions are rethrown to the
controller; the distributed abort path keeps peers from waiting indefinitely
after a rank fails.

## Engine contracts

Host field snapshots are Float64, x-fastest logical nodes with vector components
interleaved; a cell array (the Gauss residual) holds the rank's local cells. No
ghosts are written. Each rank's snapshot holds both endpoints of its closed
nodal extent, and shared endpoints agree; a VTK field file holds each global
node once. Dataset pieces keep both endpoints. Field unit conversion stays in
capture, outside the engines.

Particles contain every real species, including empty ones. Position and
velocity are interleaved triples; charge is the signed macroparticle charge.
Position is rebuilt from the global cell offset, the local cell index and the
fractional position with the prepared grid spacing. Ids are UInt64 throughout,
including values beyond Float64's exact integer range. Output describes the
committed active population, so ordering changes and migrating counts do not
change the schema.

Dataset engines write separate rank-local files for fields and particles, also
when both channels use the same backend and are due in the same epoch. Each
piece contains only its channel's payload and the metadata of geometry, time,
precision and channel presence.

One persistent CPU writer owns both selected channel engines. It dispatches
non-owning views of the same immutable host snapshot, so selecting different
backends copies no particle population and adds no writer thread. Both pieces
finish before the epoch is eligible for rank agreement and manifest
publication; a serialization failure in either engine prevents publication of
that epoch.

Particle and dataset engines write per rank and call no MPI; there is no global
particle prefix sum. Rank zero publishes a manifest after the application has
agreed that all local pieces are complete.

VTK field output is collective: every rank writes its nodes of each file through
the `lpfx::distributed::MpiFileWriter` that `TransportRegistry` owns. Only the
output writer thread uses it, and every rank writes the same snapshot sequence
and the arrays in selection order, so the collectives match. A failed write
fails on every rank; it is rethrown to the diagnostics/output worker and ends
the run through the distributed abort path. A field file is complete when its
write returns, so `publish` does nothing and the epoch needs no agreement.

Optional engines are explicit build options: `TPIC_ENABLE_ADIOS2` and
`TPIC_ENABLE_HDF5` require their dependencies at configure time, and selecting
an unavailable engine fails admission only when its channel is enabled.
Disabled output and input checking create no files and start no writer thread.

## ADIOS2 and HDF5 adapters

ADIOS2 uses BP5 and a persistent serial context, created lazily by the CPU
writer. Each piece has a scoped IO object; cleanup releases its variables,
attributes and engine on success or failure before the snapshot can be
reclaimed. Only the context and the prepared metadata persist between files.
Publication performs no ADIOS2 calls, and worker teardown joins before it
destroys the context. HDF5 uses serial rank-local files with file-scoped
handles. Every HDF5 call, cleanup included, is serialized within the process,
so library builds without thread safety are supported.

HDF5 field files carry the format version `tpic-output-v1`, ADIOS2 field files
`tpic-output-v3`, ADIOS2 particle files `tpic-output-v1`. `fields/<name>` has
shape `(z,y,x,components)`, with a size-one component dimension for scalars,
over `(nz+1,ny+1,nx+1)` for the rank's local cell counts or `(nz,ny,nx)` for a
cell array. Both dataset engines consume the interleaved host values shared with
VTK.

ADIOS2 submits field payloads with deferred `Put` and explicit host memory
space; the snapshot stays immutable through `EndStep` and `Close`, so BP5 can
consume large arrays from their buffers. Particle arrays are defined at their
full shape and written chunk by chunk: each chunk is a deferred `Put` of its
selection followed by `PerformDataWrite`, so BP5 writes it from the conversion
scratch before the scratch is refilled. HDF5 writes the same chunks as row
hyperslabs of contiguous datasets. The adapters add no component-separation
buffer, duplicate vector payload or device allocation. ADIOS2 can still buffer
small arrays and allocate internal metadata; the pinned-to-owned copy during
capture stays. The snapshot is reclaimed only after all selected channel writes
finish.

Each ADIOS2 field piece embeds the attribute `fides/schema`, which lists the
fields present in the piece and describes a uniform grid with the rank's
physical origin and spacing. Three small geometry variables store the node
counts as UInt64 `mesh/dimensions` and the physical coordinates as Float64
`mesh/origin` and `mesh/spacing`, each three values in `(x,y,z)` order: 72 bytes
of geometry payload per piece. The schema reads all three through Fides
`array_variable` sources, so its deferred metadata reads complete before the
grid is built. Geometry does not depend on which fields are selected. Basic
Fides arrays read the stored interleaved fields directly; the schema marks the
vectors (E, B, total and species currents) as three-component arrays and lists
the Gauss residual in its `cell_set`. Physical time is read from the ADIOS2
`time` variable. Field pieces omit `vtk.xml`, because the VTX ImageData reader
expects scalar three-dimensional arrays and cannot represent the interleaved
vector layout. Particle pieces carry no field visualization schema.

## Checkpoint engine

Checkpoints are a channel with their own writer: `Pipeline` owns a second
`io::AsyncOutputWriter`, configured with `io::make_checkpoint_engine`. A
checkpoint snapshot (`checkpoint_present`) holds the `data::CheckpointHeader`,
E at the local nodes, B at the local cells, and the particle records lent from
the particle images. The fields are copied host-blocking on the capture stream
before `diagnosed_fence`, interiors only, because `rhs_fields` may be writing
the B ghosts. One particle capture serves both writers, so a particle capture
first joins a checkpoint still being written; the checkpoint writer is joined
only then and in the final epoch, and its write overlaps the cycles in between.
A checkpoint needs no agreement or publication: it is complete when its
directory lists the pieces of all ranks.

A piece is written straight from the particle images, with no sort and no copy:
the longest key-sorted prefix of each species gets a per-cell start index, and
the unsorted tail (the last migration's arrivals in partial order, every record
when unsorted) keeps its rank-local cell keys. A restart reads the prefix of its
own cells in exact spans and filters the tail by key, through a staging block of
`io::restart_staging_records` records. The piece layout is described in
`io/detail/checkpoint_engine.hpp`, the protocol and the retention in
`io/checkpoint.hpp`.

HDF5 calls are serialized: the checkpoint engine takes `hdf5_mutex` per chunk of
2^23 values and lets an output piece that announced itself in `hdf5_waiting` go
first, so an output piece waits for one chunk at most. Each ADIOS2 engine owns
its `adios2::ADIOS`.

## Adding a backend

Implement `OutputEngine`, register its factory and build option, and add its
admission and name resolution. Keep field transformations in capture, convert
particle records with `io::detail::convert_particles`, and consume the snapshot
synchronously in `write_piece`. Release all payload references before
returning. `publish` publishes only completed epochs; the application supplies
the cross-rank agreement.

Use the shared particle descriptors, conversion, geometry and file-name
helpers. A backend with the same dataset contract derives from
`DatasetOutputEngine` and implements the typed metadata and payload visitor
adapters instead of repeating the dataset enumeration. Persistent caches hold
only reusable metadata and bounded working storage; file handles and
completed-epoch registries do not accumulate across outputs. A backend that
also writes checkpoints implements `CheckpointEngine::write_file` and a
`CheckpointReader`.
