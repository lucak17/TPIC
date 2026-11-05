# Numerical reference

This document states the numerical method of the kernels and operators: the
equations, normalisations, accumulation layouts and rounding points. Each
statement names the source that implements it; a change to that source which
alters a statement updates this document in the same change.

TPIC evolves node-centred E and cell-centred B with linear, nonrelativistic
ECSIM. All three components of each vector share their location. This is a
node/cell discretization, not Yee edge-E/face-B staggering. Thin periodic axes
retain three velocity components, eight CIC corners and the full 3-D cell
volume. Admission accepts only `c=1` (see
[`physics.cpp`](../../src/config/physics.cpp)); the equations below keep `c`
explicit to identify the implemented normalization.

[Architecture](architecture.md) describes execution and ownership,
[inputs](../user/inputs.md) the admitted configurations, and
[testing](testing.md) the checks and tolerances. What the Gauss-law check
measures and how to monitor it is in [Gauss's law](../user/gauss_law.md).

## Contracts

These contracts bind every change to the code: a change that breaks one is a
numerical change and needs its own qualification ([change review](change_review.md)).

| Contract | Section |
| --- | --- |
| CIC uses particle cell indices plus local fractions; corner/stencil maps and field pitch are fixed, and particle bins are not field offsets. | [Particle response and CIC](#particle-response-and-cic) |
| Current deposition converts each product to storage before sorted-cell accumulation; scalar current and mass sums use `Storage::Current` and `Storage::Mass`; output moments sum in `Compute::Reduction`. | [Current and mass](#current-and-mass) |
| The CUDA PTX and HIP MFMA4 mass kernels accumulate cells in FP64, narrow once before each global addition and keep the operand factorization `(weight_a * alpha_scaled) * weight_b`. | [Matrix mass kernels](#matrix-mass-kernels) |
| The compact mass layout stores canonical spatial pairs with all nine response-tensor components; halo scatter/add order and boundary algebra are fixed. | [Compact representation and halo](#compact-representation-and-halo) |
| E is nodal and authoritative B is cell-centred; particle B is derived onto nodes; old/new field epochs and the analytic startup ghosts are fixed. | [Maxwell and smoothing](#maxwell-and-smoothing), [Startup](#startup) |
| Full and split Maxwell schedules differ in overlap placement, not in operator algebra; smoothing keeps the S-M-S path; wall projection and physical ghost closure are distinct from transport ghost copies. | [Maxwell and smoothing](#maxwell-and-smoothing), [Boundaries and migration](#boundaries-and-migration) |
| GMRES convergence includes the true-residual checks; reordering arithmetic or reductions needs numerical requalification. | [GMRES and diagnostics](#gmres-and-diagnostics) |
| Generator, seed, sequence and draw order are part of the initial state; a fixed decomposition reproduces the same particles. | [Startup](#startup), [Bounded loads and particle IDs](#bounded-loads-and-particle-ids) |
| Scalar energy uses the physical normalization and owned-node/cell conventions; output moments are physical current/density/raw pressure, separate from the solver response current; Float64 output does not imply Float64 stored state. | [GMRES and diagnostics](#gmres-and-diagnostics) |

Movers gather the old B while the field advance publishes the checked next B
for the next epoch's deposition. The shared `current` and `mass` streams
serialize each quantity, but the deposition ledger orders ready jobs by
readiness, so the species order of the floating-point accumulation can change
between runs; comparisons therefore use tolerances (see
[`deposition_ledger.hpp`](../../include/tpic/simulation/execution/deposition_ledger.hpp)
and [architecture](architecture.md#one-timestep)).

## Particle response and CIC

Particle position is an integer cell plus stored fractions `rx,ry,rz` in `[0,1)`.
Corner `a` has offset `(a&1, (a>>1)&1, (a>>2)&1)`. CIC forms the eight products
with grouping `(wx*wy)*wz`, and gathers in increasing corner order. Already
canonical fractions are used directly, avoiding a second integer/fraction
decomposition. This shared contract is implemented in
[`cic.hpp`](../../include/tpic/ecsim/particles/cic.hpp).

For species `s`, let `beta_s=(q_s/m_s)*dt/2`, gather `B_p` with those weights,
and set `omega=beta_s*B_p/c`. The response matrix is

```text
alpha = 1/(1 + omega.omega) *
 [1+wx*wx, wx*wy+wz, wx*wz-wy;
  wx*wy-wz, 1+wy*wy, wy*wz+wx;
  wx*wz+wy, wy*wz-wx, 1+wz*wz]
```

All nine entries matter: alpha is generally nonsymmetric. Current deposition
and the mover recompute it from the same old B and deposition position. They
keep no global per-particle alpha or hatted-velocity array.

The fused mover in
[`particle_mover_device.hpp`](../../include/tpic/ecsim/particles/particle_mover_device.hpp)
computes:

```text
vhat = alpha * v_old
vbar = vhat + beta_s * alpha * E_particle_theta
v_new = 2*vbar - v_old
x_next = x_dep + dt*v_new
```

`E_particle_theta` is `Etheta` without smoothing and `S(Etheta)` with smoothing.
Old B remains live through every mover even though next B has been prepared.
The velocity update runs in `Compute::Velocity`; the displacement is formed in
`Compute::Position`, converted to `Storage::Position` and added to the stored
cell-relative fraction. The mover checks the key, the charge and the canonical
fractions of the stored record, then the pushed velocity and the range of each
cell shift; a non-finite stored velocity, B or E propagates to the pushed
velocity, and an overflowing displacement to the cell-shift range test. It
keeps fractions canonical and half-open before the next record is published,
and a rejected record keeps all its attributes.

## Current and mass

Let `q_p` denote signed macrocharge, `V=dx*dy*dz`, and `f_g` the relative inverse
nodal dual-volume factor. `f_g` is 2 on a physical face, 4 on an edge and 8 on
a corner, otherwise 1. A physical plane is one whose field topology has no peer
in that direction. After shared-node summation,

```text
Jhat_g = (f_g/V) * sum_p q_p * alpha_p*v_old * W_pg
K_gg'  = sum_species,p q_p*(beta_s/c)*alpha_p*W_pg*W_pg'
(M*E)_g = (f_g/V) * sum_g' K_gg' * E_g'
```

K is volume-free. Current deposition includes `1/V` and receives `f_g` during
assembly; mass receives both factors only during application. Applying them
again in the mass halo would change the operator. See
[`wall_faces.hpp`](../../include/tpic/ecsim/fields/wall_faces.hpp),
the kernels in
[`moment_field_device.hpp`](../../include/tpic/ecsim/deposition/moment_field_device.hpp)
and the current assembly in
[`current_assembly_device.hpp`](../../include/tpic/ecsim/deposition/current_assembly_device.hpp).
The dual-volume scale visits only the shell of the interior-node box (only
boundary nodes have `f_g != 1`) and the Neumann current closure only the ghost
layer; the wall faces and both boxes are prepared once per moment owner
(`MomentFieldPlan`). Raw density is an output quantity and is not deposited
every timestep for the linear Maxwell solve. With smoothing disabled, the RHS
needs current interiors only, so the current ghost copy is skipped after
shared-node assembly and dual-volume scaling.

The mass scale is `beta_s/c`, whereas the electric response in the mover uses
`beta_s`. They agree for the admitted `c=1`; changing either factor is a
numerical change.

### Sorted and unsorted kernels

The current, mass and mover kernels consume LPFx bins in full/partial mode.
A sorted kernel assigns one native warp/wave to a cell in a 128-thread block
(`sorted_cell_threads`). It loads that cell's eight magnetic corners into
shared memory once and reuses them for the cell's particles. The mover also
caches eight electric corners. CUDA warp32 and HIP wave32/wave64 use the native
subgroup abstraction. The current and mass kernels give a warp to every bin,
halo bins included, and reject an occupied halo bin. The sorted mover gives a
warp to interior cells only: halo bins hold no particle exactly when the
interior ranges tile `[0, prefix)`, which the mover checks at the first
interior bin, at each interior row end and at the last interior bin
(`ParticleBins::interior_range`). Metadata of an empty halo bin is therefore
not read by the mover. The per-particle sample (cell source, fraction load,
CIC weights, magnetic gather and response) is shared by these kernels through
[`particle_sample_device.hpp`](../../include/tpic/ecsim/particles/particle_sample_device.hpp);
the mover builds the response after its electric gather, which keeps its
register count down. The fraction load and the `in_unit_cell` predicate live in
[`particle_fractions.hpp`](../../include/tpic/data/particles/particle_fractions.hpp),
which the startup ownership pass and the output-moment kernel also use. Each
kernel keeps its own validation order and reporting form.

Sorted current divides each native subgroup of width `W` into eight corner
groups of `G=W/8` lanes: four lanes per corner on wave32 and eight on wave64.
Each lane keeps three current sums for its corner. A tile contains `W`
particles. Each lane evaluates one particle, then stages its eight CIC weights
and three scaled current values in shared memory. The weights are exactly those
used by the magnetic gather. The shared tile has one row per staged quantity
(11 rows) of `128+G` entries; the padding separates the corner groups' weight
reads across shared memory banks. Each warp owns a disjoint `W`-column span.
Corner `lane/G` consumes sources `batch*G + lane%G` within that span for
batches 0-7, loading only its own weight and the scaled current. Every particle
contributes to each corner exactly once without repeating its weights,
magnetic gather or response calculation. The sums persist across all tiles; a
`G`-wide shuffle reduction leaves each corner's result in its group leader. The
final flush issues 24 current atomics per valid occupied cell. This budget is
per cell, not per particle tile.

Current forms each weighted contribution in `Compute::Current`, converts it to
`Storage::Current`, then adds it to a sum in that storage type. Every
current-deposition product, sorted and unsorted, goes through
`deposit_product`, which rounds the product on its own before
conversion/addition: `__dmul_rn` or `__fmul_rn` on CUDA and a
`#pragma clang fp contract(off)` scope on HIP. No multiply-add contraction
crosses that boundary, even for double storage. Changing particle grouping
changes summation order, so comparisons use tolerances and conservation checks
rather than bitwise equality.

Producer/bin validity, matching particle keys, canonical fractions and the
three response-current components remain checked; a non-finite gathered B or
scaled charge propagates to those components. A checked finite current
multiplied by a canonical weight in `[0,1]` cannot overflow its compute type,
so both backends omit product-finiteness checks unless the storage type is
narrower (`narrowing_may_overflow`). Narrower storage keeps per-product
conversion checks. Final reduction results are always checked in the group
leaders. A failed particle evaluation resets all staged weights and currents
to finite zero; tail lanes also stage zero records. All lanes synchronize after
populating the shared tile and before overwriting it, participate in the final
reduction, and synchronize before reusing the shared cell cache.

Scalar mass distributes the 8x8 corner pairs across lanes (one accumulator
set per lane on wave64, two on warp32), broadcasts particle fractions/response
with shuffles and keeps cell accumulators before flushing the 36 canonical
pairs: at most 324 tensor atomics per occupied cell. Global atomics remain
necessary because adjacent cells contribute to shared nodes/records;
successful particles do not each perform those global flushes. Invalid-data
reporting has separate conditional atomics. The unsorted mass kernel checks
the nine scaled response entries of each particle, which also covers a
non-finite B or charge, and checks each narrowed product only when the mass
storage is narrower than the mass compute type.

Unsorted kernels operate per particle and use global contribution atomics.
Partial mode applies the sorted kernels only to the resident prefix and the
unsorted kernels to arrivals. Output moment deposition
(`enqueue_deposit_output_moments`) takes the particle owner without a sort
receipt, with a per-species deposition mask that selects the requested
quantities.

Sources:
[`current_deposition_device.hpp`](../../include/tpic/ecsim/deposition/current_deposition_device.hpp),
[`mass_matrix_scalar_device.hpp`](../../include/tpic/ecsim/mass_matrix/mass_matrix_scalar_device.hpp),
[`mass_matrix_deposition_device.hpp`](../../include/tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp),
[`sorted_cell_device.hpp`](../../include/tpic/ecsim/particles/sorted_cell_device.hpp).
Scalar mass is available on CUDA and HIP. Each backend also provides a
matrix-instruction mass kernel, described below: inline PTX `mma.sync` on CUDA
and MFMA4 on HIP. Current and mover kernels are scalar on both backends.

The sorted scalar mass accumulation forms `alpha * pair_weight` and adds it to
the accumulator without an explicit rounded multiply, so the compiler may
contract the multiply-add; finite input/response and final-accumulator
validation are kept. With FP32 `Storage::Mass` and FP64 `Compute::Mass` it
forms the product in FP64, narrows each product to FP32, then adds to an FP32
accumulator. That per-product narrowing is a rounding boundary; it is not FP32
response arithmetic or an FP32 fused multiply-add claim. Reordering or
contraction can change last-bit results, so comparisons use numerical
tolerances and conservation checks rather than requiring bitwise equality.

### Matrix mass kernels

`--mass-matrix matrix-core` (`MassDepositionKernel::matrix_core`) selects the
matrix-instruction kernel of the active backend:
[`sorted_mass_matrix_mma_kernel`](../../include/tpic/ecsim/mass_matrix/mass_matrix_mma_device.hpp)
on CUDA and
[`sorted_mass_matrix_mfma4_kernel`](../../include/tpic/ecsim/mass_matrix/mass_matrix_mfma_device.hpp)
on HIP. The tensor-core assembly is published as L. Pennati and S. Markidis,
"Mass matrix assembly on tensor cores for implicit particle-in-cell methods",
J. Comput. Phys. 569, 115442 (2027). Both kernels read the LPFx particle views,
sorted receipts and magnetic caches and write the compact mass layout. One
native warp/wave processes a cell, accumulates all nine tensor components
persistently in FP64 and deposits the 36 canonical spatial pairs once after
the cell's particles have been consumed.

The CUDA kernel uses inline PTX `mma.sync.aligned.m8n8k4.row.col.f64`, with
FP64 register operands and accumulators; no WMMA fragment objects or shared
matrix staging are needed. It uses 128-thread blocks and stages two particles
per lane in 64-particle tiles. Each staged particle has three fractions and
nine scaled response values. Increasing four-particle MMA groups update 18 FP64
accumulators per lane across the complete cell; conversion to mass storage
occurs only at the final cell flush.

Each four-particle MMA computes an 8-by-8 corner product:

```text
A[a,p] = W_pa * (q_p * beta_s/c * alpha_p[component])
B[p,b] = W_pb
C[a,b] += sum_p A[a,p] * B[p,b]
```

On CUDA, lane `l` owns matrix row `l/4` and outputs columns `2*(l%4)` and
the following column. The shared canonical pair definition selects the 36
stored entries; backward reuse keeps the tensor without transposition.
Padding lanes contribute zero.

The HIP kernel uses 256-thread blocks (four wave64 subgroups, one cell each)
and one particle per lane in 64-particle chunks. For each group of four
particles it issues nine `__builtin_amdgcn_mfma_f64_4x4x4f64` instructions
(assembly `v_mfma_f64_4x4x4_4b_f64`), one per tensor component, with operands
`W_a*alpha_scaled` and `W_b`. The four hardware blocks interleaved across the
wave hold the four 4x4 quadrants of the same 8x8 corner product
([`mass_matrix_mfma_layout.hpp`](../../include/tpic/ecsim/mass_matrix/mass_matrix_mfma_layout.hpp)):
lane `l` feeds corner `4*((l/4)%4/2) + l%4` of particle `l/16` and accumulates
the pair `(4*((l/4)%4/2) + l/16, 4*((l/4)%4%2) + l%4)`. The last three builtin
arguments (CBSZ, ABID, BLGP) are zero: on CDNA3/4 the FP64 BLGP field encodes
operand negation, not a lane broadcast, and broadcast controls cannot form
independent 8x8 products. A operands of incomplete groups and rejected
particles are padded with zeros; the instruction is never predicated. Each lane
keeps nine FP64 accumulators and narrows to mass storage once at the cell
flush. The batched 4x4x4 instruction performs 512 FLOPs per wave against 2048
for `16x16x4` in half the instruction cycles, so it has half the peak FP64
matrix rate of the large tile; it is used because its four independent blocks
are exactly the four quadrants of one 8x8 corner product, which a dense
`16x16x4` product is not.

Full ordering applies the matrix kernel to the complete population; partial
ordering applies it to the sorted prefix and uses scalar deposition for the
arrival suffix. Unsorted ordering rejects the selection.

The mass launch is resolved at compile time from the `MassDepositionKernel`
template argument; startup selects the specialization after CLI/rank agreement
and device admission
([`mass_matrix_deposition_device.hpp`](../../include/tpic/ecsim/mass_matrix/mass_matrix_deposition_device.hpp),
with the probe beside each matrix kernel). CUDA requires a device of compute
capability 8.0 or later and a compiled kernel image of that level; the check
includes older PTX images JIT-compiled on a newer device, and older CUDA images
compile a rejecting stub so scalar builds remain usable. HIP requires a wave64
device of architecture `gfx90a`, `gfx940`, `gfx941`, `gfx942` or `gfx950`;
other HIP targets compile a stub that marks the deposition invalid, and startup
rejects the request there. The platforms on which each kernel is qualified are
listed in [support](../user/support.md).

The scalar kernels compute the particle response and the CIC operands in
`Compute::Mass`. The matrix kernels are FP64 and need double mass compute; every
matrix-kernel cell accumulator is FP64, then converts to `Storage::Mass` once
before its global atomic addition. Single mass storage therefore uses FP64 sums
within each cell and FP32 additions across cells. This differs by design from
scalar per-product FP32 narrowing and from the scalar partial tail. A pair of
individually FP32-overflowing contributions can cancel within a matrix-kernel
cell before narrowing, where scalar accumulation rejects them. Tests check the
selected kernel's contract instead of demanding identical overflow flags or
bitwise results between kernels.

All variants validate sorted producer metadata, particle keys/fractions,
finite charge/response and final stored contributions. Their reassociation
also changes FP64 rounding: the matrix kernels group `W_a*alpha_scaled` before
multiplying by `W_b`, while scalar deposition groups the pair weight.
Qualification compares native coefficients, mass-vector products and complete
energy exchange on fixed states; compiling the PTX or MFMA code does not
establish numerical qualification or speed, and hardware reduction order is
qualified on each backend.

### Compact representation and halo

[`mass_matrix_layout.hpp`](../../include/tpic/data/mass_matrix/mass_matrix_layout.hpp)
defines one contiguous 126-scalar record per logical node: 14 spatial offsets
x 9 row-major tensor entries. The canonical offsets, in storage order, are:

```text
0:(0,0,0)     1:(1,0,0)      2:(0,1,0)      3:(0,0,1)
4:(1,1,0)     5:(1,-1,0)     6:(1,0,1)      7:(1,0,-1)
8:(0,1,1)     9:(0,1,-1)    10:(1,-1,1)   11:(1,1,-1)
12:(-1,1,1)  13:(1,1,1)
```

The eight-corner pair table in
[`mass_matrix_stencil.hpp`](../../include/tpic/ecsim/mass_matrix/mass_matrix_stencil.hpp)
is generated from this definition and has 36 canonical pairs, including eight
self pairs. Application reconstructs centre, 13 forward and 13 backward blocks.
Spatial pair symmetry reuses the same nine tensor entries for backward pairs
**without transposing the tensor**. Alpha's antisymmetric part also means the
operator is not a candidate for CG merely because spatial storage is symmetric.

Mass storage has its own one-node halo and x-fastest strides, independent of
LPFx's field pitches. Its coefficient allocation is
`126*N_halo_nodes*sizeof(Mass)`: 1008 bytes per halo node for FP64, 504 for
FP32, before communication buffers.

[`MassHaloStorage`](../../include/tpic/communication/mass_halo/mass_halo_device.hpp)
performs additive assembly followed by a ghost phase. Each phase packs
immutable source values before scattering, and each sends only the stencils
that carry data, fixed per direction by `mass_halo_stencil_mask`. A rank
deposits a pair only in its own cells, so the additive phase drops the stencils
that cross the shared plane toward the neighbour: they are exact zeros there.
The ghost phase copies fully assembled values only into the ghost stencils that
the apply reads, those that point back into the rank on every axis of the
direction. A direction receives what the opposite direction sends. In FP64,
with 26 neighbours of 16x16x32 cells, the two phases send 3.0 MB each way
instead of 6.25 MB. Self-periodic routes use device pack and scatter with the
same snapshot semantics. Every kernel launches one one-dimensional grid that
concatenates the blocks of every direction's send or receive side
(`MassHaloSide::first_block`, `block_count`, each capped at
`lpfx::device::kernel_grid_block_limit`). Additive destinations overlap at
faces, edges and corners: the lowest ordinal whose region holds an entry's
vertex and whose side carries its stencil adds every such direction in
ascending canonical order, which fixes the sum; a direction that does not carry
the stencil would add an exact zero, and records start at +0, so none is -0.
Ghost regions are disjoint, so every other entry is written by exactly one
thread and the grid does not affect values.

The prepared `MessagePlan`s, fixed-message buffers and device allocations
persist across steps. The two phases share one outgoing and one incoming
allocation sized for the larger phase and register in the simulation's shared
`Communication` as `mass.add` and `mass.ghosts`. The ghost phase begins only
after the additive phase has concluded and its ordered scatter has been queued
on the `mass` stream; the events the solve waits on are listed in
[architecture](architecture.md#streams-events-and-workers).

## Maxwell and smoothing

For `delta=c*theta*dt`, `k=4*pi*theta*dt`, the device operator solves

```text
rhs = E_old + delta*curl_C2N(B_old) - k*S(Jhat)
A(v) = v + delta^2*curl_C2N(curl_N2C(v)) + k*S(M*S(v))
A(Etheta) = rhs
E_new = Etheta/theta - (1-theta)*E_old/theta
B_new = B_old - c*dt*curl_N2C(Etheta)
```

Set `S=identity` when smoothing is disabled. Faraday and the committed E update
use unsmoothed `Etheta`; the mover uses the particle-facing smoothed value when
requested. B-node is the arithmetic average of eight surrounding B-cell samples.
Particles gather only after migration has returned them to local cells, and
only at interior cell corners; with Faraday and the E update reading interior
nodes only, the particle-facing electric field and B-node need no ghost
values, only the boundary projection of the electric field.
The N2C/C2N derivatives average four edge differences with a `1/4` factor.
The E update forms the two terms `(1/theta)*Etheta` and
`(-(1-theta)/theta)*E_old` with one narrowing (a copy of `Etheta` when theta is
within `1e-10` of 1); the B update evaluates `curl_N2C(Etheta)` in registers
inside the update kernel and adds it to `B_old` in the same two-term form (a
zero result keeps its sign; a non-finite operand gives Inf or NaN, which raises
the validation flag).
See [`node_cell_curl.hpp`](../../include/tpic/ecsim/fields/node_cell_curl.hpp),
[`node_cell_curl_device.hpp`](../../include/tpic/ecsim/fields/node_cell_curl_device.hpp),
[`cell_to_node.hpp`](../../include/tpic/ecsim/fields/cell_to_node.hpp),
[`maxwell_device.hpp`](../../include/tpic/ecsim/maxwell/maxwell_device.hpp)
and [`field_update_device.hpp`](../../include/tpic/ecsim/maxwell/field_update_device.hpp).

Every matvec projects the trial field once onto the constrained wall rows (PEC
tangential E, PMC normal E) on the `solver` stream, then exchanges its halo and
closes the physical electric ghosts. The halo exchange and ghost closure do not
write interior nodes, so the projection is not repeated. The all-cell N2C curl
reads only local nodes and starts on the `solver_interior` stream while the
input halo is in progress. The strict-interior C2N rows follow on the same
stream. The curl-cell halo starts once N2C has finished; an input halo still in
flight (MPI) shares its network time with it, one completed at begin
(NCCL/RCCL) closes its ghosts and records the mass dependency first, so M never
waits for curl communication. The shell rows wait for the curl-cell halo and
its physical closure on the `solver` stream.

Without smoothing, the mass response runs on the dedicated `solver_mass`
stream, and `--mass-apply` selects one of two compiled schedules
([`maxwell_apply_policy.hpp`](../../include/tpic/ecsim/maxwell/maxwell_apply_policy.hpp)).
`full` (the default) applies M to all local nodes after the input ghosts are
closed. `split` applies the strict-interior rows after the projection and the
shell rows after ghost closure. Each row is written once by the same row
calculation (the apply kernel is specialised on
[`StencilRegion`](../../include/tpic/ecsim/fields/stencil_region.hpp)), so the
schedules differ in overlap placement, not in operator algebra. Events join the
mass and interior-curl results before the final combination and the boundary
rows. With smoothing, either request resolves to the `smoothed` schedule: both
filters and the full-domain mass application run on the `solver` stream
because they share the two scratch owners, and only the curl overlap remains.

Energy-conserving smoothing is active only when `ECSimSmoothing` selects that
mode, `SmoothNiter > 0` and `Smooth < 1`; admitted `Smooth` lies in `[0,1]`.
Each of the requested passes uses one of these stencils
([`node_smoothing_device.hpp`](../../include/tpic/ecsim/fields/node_smoothing_device.hpp)):

- Face7, the seven-point stencil (the node and its six face neighbours):
  `alpha*f + (1-alpha)/6 * sum(face neighbours)`, with `alpha=Smooth`.
- Binomial27, the 27-point tensor product `[1,2,1]^3/64`:
  `(8*center + 4*faces + 2*edges + corners)/64`. `Smooth` gates
  activation but does not change these fixed weights.

Two persistent solver-precision node owners (`smooth_a`, `smooth_b`) alternate
between passes. Intermediate halos use the registered `solver.smooth_a` and
`solver.smooth_b` copy plans. The inner `S(v)` refreshes its final halo before
M reads neighbours. The final particle `S(Etheta)` gets its boundary projection
and no halo refresh.

## Boundaries and migration

Every boundary kind is one rule in
[`boundary_conditions.hpp`](../../include/tpic/config/boundary_conditions.hpp):
its ghost signs, its zeroed electric rows and what `--initialize` and `--run`
reject. Effective periodicity overrides face codes. Physical `bcEMface*=0`
selects the perfect electric conductor (PEC), 1 the perfect magnetic conductor
(PMC), 2 selects open; any other code, periodic or not, is an input error.
`--initialize` and `--run` reject open faces
([`run_plan.cpp`](../../src/simulation/run_plan.cpp)). Physical particle faces
admit mirror code 1 only.

For PEC ghosts, tangential E is odd and normal E even; normal B is odd and
transverse B even. The PMC is the dual: normal E and tangential B are odd,
tangential E and normal B even, so the wall is a mirror-symmetry plane of the
fields. With these signs the node curl of B and the cell curl of E stay adjoint
on the wall in the dual-volume inner product, which exact energy conservation
needs. PEC tangential E and PMC normal E are constrained to zero on interior
wall nodes: exactly the components whose ghost sign is odd. TPIC projects the
RHS and initial E guess into that constrained subspace; Arnoldi, residual and
update operations keep it. Matvec projects the input before its halo and
uses identity values for constrained image rows, so the final projection is
idempotent on the accepted constrained solution. Every smoothing pass is
followed by the same projection, which keeps smoothing self-adjoint in the
dual-volume inner product on PEC and PMC walls.

B-cell and curl-cell physical ghosts use the corresponding magnetic parity.
These physics closures are separate from LPFx neighbour copies, which act only
where peers exist. See
[`field_boundary_device.hpp`](../../include/tpic/ecsim/fields/field_boundary_device.hpp).

General migration compacts residents and appends arriving particles; see
[`particle_boundaries.hpp`](../../include/tpic/communication/particles/particle_boundaries.hpp)
and [`particle_wire.hpp`](../../include/tpic/communication/particles/particle_wire.hpp).
A full-order sort receipt owns the bin epoch through deposition and the final
sorted mover, and retires before migration can compact or replace the particle
allocation
([`particle_sort_device.hpp`](../../include/tpic/ecsim/particles/particle_sort_device.hpp)).

Partial order uses a completed resident-prefix receipt plus an unsorted
arrival tail. While routing is still in progress, the deposition worker
deposits the sorted prefix through a checked resident capability
(`enqueue_resident_current`, `enqueue_resident_mass_matrix`), and the arrivals
are appended only after both resident readers are sealed (the events are in
[architecture](architecture.md#streams-events-and-workers)). The deposition
ledger then accepts the tail range only if it starts where the resident range
ended, so every particle is deposited exactly once per quantity. See
[`local_species_actions_device.hpp`](../../include/tpic/simulation/actions/local_species_actions_device.hpp),
[`deposition_actions_device.hpp`](../../include/tpic/simulation/actions/deposition_actions_device.hpp)
and [`deposition_ledger.hpp`](../../include/tpic/simulation/execution/deposition_ledger.hpp).
A checked receipt alone does not permit concurrent overwrite of particle
attributes.

## GMRES and diagnostics

[`gmres_device.hpp`](../../include/tpic/ecsim/gmres/gmres_device.hpp) keeps
compact vectors, the Krylov basis, Hessenberg state and bounded reduction
scratch on the GPU. Batched projection dots share vector loads; LPFx block
reductions write block partials and one reducer kernel
([`block_reduction_device.hpp`](../../include/tpic/device/block_reduction_device.hpp),
shared with the scalar diagnostics) completes each batch without global
atomics. The Givens update and the restart right-hand side run on one thread
inside the normalization kernel, which also writes the new Krylov vector into
the operator input, so an Arnoldi iteration needs no single-thread launch and
no separate unpack. Single-rank sums skip the collective; remote sums use the
registered `gmres` collective. A persistent pinned two-scalar buffer carries
convergence controls to the host.

The solver uses restart 30 and at most 100 completed Arnoldi steps
(`gmres_restart` and `gmres_max_iterations` in
[`physics.hpp`](../../include/tpic/config/physics.hpp)). The initial
residual is `rhs-A*x`; tolerance is relative to `||rhs||`, using scale 1 for a
zero RHS. Givens estimates can end a restart early, but acceptance always checks
the true residual `||rhs-A*x||/scale` after updating the solution, including the
last allowed iteration. The Krylov basis and the Arnoldi applies use
`Compute::Electric`; the dot products, the Hessenberg state, the solution, the
right-hand side and the residual apply use `Compute::SolverReduction`. When the
Krylov type is narrower, a cycle ends once its estimate has dropped to sixteen
Krylov-type epsilons times the cycle's starting residual and the next restart
refines the solution, so the tolerance is reached in the reduction type. The
solver work fields are stored in the wider of the two types (`SolverStorage`).
The host reads the controls each iteration; this is a GPU-resident solver with
host convergence decisions. Nonconvergence prevents particle advancement and
timestep completion. No preconditioner or automatic Poisson projection is
installed.

[`ScalarDiagnostics`](../../include/tpic/diagnostics/scalar_diagnostics_device.hpp)
computes field energies plus five values per species: kinetic energy, charge
and three momentum components. One species traversal reduces those five values
together. Its scratch bound is independent of population. The reductions use
two lanes with disjoint scratch and result slots: the `diagnostics` stream
carries field energy and the odd-indexed species, and the `diagnostics_helper`
stream carries the even-indexed species
([`execution_streams.hpp`](../../include/tpic/simulation/execution/execution_streams.hpp)).
Particle mass is `q/qom`; field energy uses `V/(8*pi)`, physical nodal
quadrature factors and unique global node ownership to avoid counting
periodic/shared endpoints twice. Cell B energy visits interior cells. Both
lanes read the committed state, which the field advance and the mover overwrite
only after the sources-read event
([`pipeline.hpp`](../../include/tpic/simulation/driver/pipeline.hpp),
[architecture](architecture.md#streams-events-and-workers)). The same sample
serves the console and the scheduled CSV. Output rho/J/pressure are separate
raw committed-state moments, described in [output](../user/output.md).

[`GaussResidual`](../../include/tpic/diagnostics/gauss_residual_device.hpp)
measures Gauss's law where the scheme keeps it: at cell centres, with
`divC = node_cell_divergence`, the divergence of the curl stencil, so
`divC(curl_C2N B)` vanishes and `divC(E)` changes only through
`S(Jhat + M*S(Etheta))`. A sample begun at cycle `n` stores `divC(E^n)/2 - c`
per local cell, where `c` is `4*pi` times the eight-node mean of the charge
density of the particles at `x^{n+1/2}`: deposited like the output rho, summed
on shared nodes, dual-volume scaled, zero on PEC wall nodes (the conductor's
image charge is opposite) and smoothed with the solver's stencil and passes.
Its wall plans are those of a tangential E component: div E is odd at a PEC
wall and even at a PMC wall (`WallQuantity::charge` in
[`field_boundary.hpp`](../../include/tpic/ecsim/fields/field_boundary.hpp)).
The sample finishes at `n + 1` by adding `divC(E^{n+1})/2`: the residual sits at
`(n + 1/2)*dt` for every theta, and its change between samples is the discrete
continuity defect plus the GMRES residual. Four sums (residual, change from the
first sample, wall cells, charge) use one collective on the `diagnostics`
stream. The scheduling, owners and outputs are in
[Gauss diagnostic](gauss_diagnostic.md); what it measures and how to read it
in [Gauss's law](../user/gauss_law.md).

Energy conservation, global charge preservation, discrete continuity, Gauss's
law and magnetic divergence are distinct checks. `Jhat + M*E_particle_theta`
corresponds to current deposited from `vbar=(v_new+v_old)/2`; it does not equal
the output current from `v_old` or a current from `v_new`. With smoothing,
Maxwell couples `S(Jhat+M*S(Etheta))`. Tests use those matching quantities
and node/cell stencils. Adding a projection or changing rounding to improve one
diagnostic changes the method and needs separate qualification.

## Startup

The analytic field images are evaluated in double on the host and narrowed
once to the storage type: E at nodes and authoritative B at cells. Their halos
hold analytic samples at unwrapped coordinates; a ghost position is not wrapped
into the domain before sampling
([`field_image.hpp`](../../include/tpic/initialization/fields/field_image.hpp)).
Particle-facing nodal B is derived on the device: the B-cell halo is exchanged,
cell-to-node interpolation runs, then the B-node halo is exchanged
(`complete_initial_magnetic_field` in
[`magnetic_startup_device.hpp`](../../include/tpic/simulation/actions/magnetic_startup_device.hpp)).
Physical B-node ghosts stay zero at startup; cell ghosts at a missing neighbour
keep their analytic samples.

A restart reads E at the nodes and B at the cells of its checkpoint, state n,
into the image interiors and the particle records as stored. The B wall ghosts
are then closed between the B-cell exchange and the interpolation, as the field
update closes them before it interpolates B^n+1, so the restored B nodes are
those of the run that wrote the checkpoint; velocities are not centred again.
The E ghosts keep the analytic samples: no reader uses them.

Particle loading uses the backend's default XORWOW generator with the seed
`12345 + 100*(rank*TPIC_MAX_SPECIES + species)`
([`prepared_initialization.cpp`](../../src/initialization/prepared_initialization.cpp);
`TPIC_MAX_SPECIES` is the compiled species capacity, `config::max_real_species`)
and the particle's load index as sequence. Each particle draws, in this order,
three uniform subcell offsets (none under `QuietStart = true`, which places
the particle at the subcell centre), one normal pair for `u` and `v`, then one
normal for `w`. The velocity profile is evaluated at the stored, possibly
narrowed, position and at the stored cell, after the endpoint carry: each
component is `drift + thermal*normal`, after which a case may add its velocity
seed. Velocity and charge are narrowed once to their storage types; a
non-finite stored value rejects the load
([`particle_loading_device.hpp`](../../include/tpic/initialization/particles/particle_loading_device.hpp)).
Generator, seed and draw order are part of the initial state.

### Bounded loads and particle IDs

A species loads where its profile's `support()` says
([`particle_support.hpp`](../../include/tpic/initialization/particles/particle_support.hpp)):
a `ParticleSupport` is the ellipsoid `sum_a ((x_a - center_a)/radius_a)^2 <= 1`
over the axes with `radius_a > 0`, or every subcell when no radius is positive.
Every profile has the member function, so a case can bound one species and not
another; only `GaussianExpansionProfile` returns a bounded support, and a
default support takes the dense path (`index -> cell = index/ppc`, x fastest,
`sub = index%ppc`).

The host alone decides which subcells a bounded support holds. A point is
loaded when the sum over the bounded axes of
`((K_a + 1/2)*h_a - center_a)^2/radius_a^2` is at most 1, with `K_a` the global
sub-lattice index and `h_a = d_a/npcel_a`. The coordinate is formed from the
global index, so a point gets the same answer whether a rank's rows or the
global rows evaluate it, and the global count equals the sum of the local ones.
The test uses the subcell centre also under a random start, so the count does
not depend on the draws. A device evaluation is not used because it could
disagree with the host at the boundary (nvcc contracts `r^2 <= r_cut^2` to an
FMA, the host compiler may not), and the planned global count must equal the
loaded population.

`visit_support_rows` visits every row of subcells (y and z fixed) that holds
points of the support, in increasing order, for the rank's box
(`first = global_offset*ppc`, `extent = local_cells*ppc`) or for the whole
domain; a row cuts the ellipsoid in one segment, whose ends start from a
square-root estimate and move subcell by subcell to the exact predicate. Every
rank computes the global count without communication.
`prepare_bounded_particle_plan`
([`prepared_initialization.cpp`](../../src/initialization/prepared_initialization.cpp))
plans the counts again with the local and global points of every bounded
support, right after the case is prepared; the first, dense plan admits the
count arithmetic before any case table is built, dense species keep their
counts, and a restart plans afterwards with the checkpoint's counts. The memory
estimate, capacities, startup report and population check read this plan.

The loader walks the rank's rows with the same function, twice (to size and to
fill), writes each row's `first` and running `end` (both `uint32`) into one
pinned block and copies it to one device allocation for the load only
(`detail::LoadRows`). The kernel maps `index -> row` by a binary search over
`end`; the point is the x-fastest local subcell
`first[row] + index - end[row - 1]`, whose x gives `i = x/npcel_x` and
`sub_x = x%npcel_x`, and likewise y and z. Positions and charges of a bounded quiet load are exactly those of the
dense quiet load inside the support; its velocities are other draws. Inside a
rank the particle order is row-major, not cell-major; the startup sort of
`--order full` or `partial` reorders by cell as for any load.

Particle IDs are `id = first_id + index`, with
`first_id = (sum over s' < s of global_cells*ppc(s')) + preceding_cells*ppc`
([`particle_plan.cpp`](../../src/config/particle_plan.cpp)): the species offset
counts dense candidates, not loaded particles. A rank's range
`[first_id, first_id + local_count)` lies inside its dense range, so ranges are
disjoint across ranks and species; a bounded load leaves unused IDs between
ranks, and for a dense load the offset equals the sum of the earlier species'
global counts. The initial state therefore depends on the decomposition: the
seed on the rank, the sequence on the local index, the IDs on the rank prefix,
and the sample position `origin + (i + r)*d` rounds differently on each rank.

Ranks outside the support plan zero particles; their capacity is LPFx's minimum
quantum, and growth handles arrivals, as on a restarted rank that holds no
record of a species. The migration seed computed from the mean per cell
underestimates the central ranks of a bounded load; LPFx grows the migration
payload on demand, so at worst the first crossings cost one growth.

### Analytic cases

The Harris sheet
([`harris_sheet.hpp`](../../include/tpic/initialization/cases/harris_sheet.hpp),
[`harris_sheet.cpp`](../../src/initialization/cases/harris_sheet.cpp)) is
`Bx = HarrisB0x*tanh((y - yc)/delta)` plus a seed derived from an out-of-plane
potential `A`, `B = (dA/dy, -dA/dx)`. The hump is
`A = A0 * exp(-x^2/wx^2 - y^2/wy^2) * cos(kx*x) * cos(ky*y)` about the sheet
centre and the box-scale mode is `A = (m*HarrisB0x/kx) * cos(kx*x) * cos(ky*y)`
with `kx = 2*pi/Lx`, `ky = 2*pi/Ly` about the box centre. The seed is stored in
the units of the hump expression: the amplitudes of `Bx` and `By` as fractions
of `HarrisB0x` (`A0/(HarrisB0x*wy)` and `A0/(HarrisB0x*wx)`), the widths, and
periods such that the phase is `pi*offset/period/delta` and the slope of the
cosine times the width is `pi/slope_period`. `GEM` and `DoubleGEM` hold the
constants `0.4`, `delta` and `10` there, so their field is the X-point
expression with its operation order; `GEMHarris` derives the same quantities
from its keys, with `A0 = pertHump*HarrisB0x/ky`. The curl along z is
`-(HarrisB0x/delta)*sech^2((y - yc)/delta) - laplacian(A)`, evaluated in closed
form.

With `currentFromAmpere` a species with a nonzero `w0` has the share
`w_s = sign(qom_s)*w0_s / sum(sign(qom)*w0)` of the current, and its particles
load the drift `w_s*c*curl/(sign(qom_s)*rhoINIT_s*shape)` along z, where
`shape` is the cell mean of `sech^2` that scales the charge of the particle. A
particle thus carries `cell_volume/ppc * w_s*c*curl/(4*pi)` at its own
position, whatever the shape of its cell. Admission bounds the drift on every
row of particles of every such species, with a closed form over x
(`|cos| <= 1`, `|r|*exp(-r^2) <= 1/sqrt(2e)`), and rejects a deck whose bound
exceeds `c`. `spatiallyVaryingThermal` replaces `wth` of these particles by
`sqrt((drift^2 + wth^2) - drift^2)` and keeps the deck value when the
difference is below four units of rounding of `drift^2`.

### Velocity centring

Optional velocity centring (`CenterInitialVelocity`) makes the global
charge-weighted mean velocity of a species that of the drift its case loads,
which may differ between particles (the two sheets of `DoubleGEM`, the current
of `fluxRope`): it removes the mean of the sampled part `thermal*normal` and of
the velocity seed, a perturbation of zero mean. It is the last startup step
that changes velocities, before the initial migration and sort. The loaded
drift of a particle is evaluated again at its stored position and subtracted in
place; four global sums follow in the reduction type, the charge and the three
charge-weighted components of what is left; a second pass adds the drift back
and subtracts the quotient of the sums. Within one species the charge-weighted
mean is the mass-weighted one, so the species momentum is the loaded drift's to
rounding, also with variable weights. Each pass forms the velocity in double
and narrows it once, so a centred velocity is within one rounding of the loaded
value minus the mean. A zero charge sum applies no shift
([`particle_centering_device.hpp`](../../include/tpic/initialization/particles/particle_centering_device.hpp)).
