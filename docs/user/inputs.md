# Input decks

TPIC reads a plain-text deck of `key = value` lines. Start from
[input/uniformStartup.inp](../../input/uniformStartup.inp): it is a complete template that lists every supported
general key with its choices, the optional features switched off. The other decks in `input/` set up the cases;
[cases](cases.md) lists them with their semantics and admission rules. [Input keys](input_keys.md) gives the
type, default and meaning of every key, and [build and run](build_and_run.md) the command line.

## Deck syntax

`#` starts a comment, also after a value. Every other non-blank line is `key = value`: the line is split at its
first `=`, and the key and value lose their surrounding whitespace. A line without `=` or with an empty key
rejects the deck with an error that names the line. A repeated key keeps its last value. Keys and values are
case-sensitive.

- A string is the rest of the line; quotes are kept as characters. There is no line continuation, include or
  variable expansion.
- A number must be the whole value: `nxc = 64.5` and `dt = 0.1x` are errors.
- A boolean is `0`, `1`, `true` or `false`.
- A list is separated by whitespace only and each entry must be a whole value: `qom = -64, 1` is an error.
  Entries beyond the required count are ignored.
- A value that does not belong to its key rejects the deck with an error that names the line and the key.
- An unknown key is ignored and listed, with its line, in one startup warning.

`dt`, `ncycles`, `ns` and `Case` are required, and so are the species lists `rhoINIT`, `qom`, `npcelx`,
`npcely`, `npcelz`, `uth`, `vth`, `wth`, `u0`, `v0` and `w0`: each holds at least `ns` entries (`qom` and
`npcel*`: `ns + nsTestPart`). `CenterInitialVelocity` is an optional list; a missing entry is `false`. Every
other key has a default. `ns` lies between 1 and the build option `TPIC_MAX_SPECIES`.

Every scalar key is parsed whatever `Case` selects: an unknown `WhistlerSeedMode` rejects a uniform deck, and an
output method outside `default`, `pvtk`, `vtk`, `adios2`, `bp`, `bp5`, `hdf5`, `h5` (or empty) rejects the deck
at its line even when its channel is disabled. A case checks the values of its own keys only when it is selected.
`FieldOutputTag` and `MomentsOutputTag` are `+`-separated tokens ([output](output.md)); an unknown token rejects
the deck during output preparation.

`--check-input` runs the host admission of a deck: parsing, the domain partition, the physics checks, the case
admission, the particle plan and the output plan. GPU capability, communicator and run admission follow in
`--initialize` and `--run` ([build and run](build_and_run.md#commands)).

## Units and normalization

TPIC integrates the equations in normalized Gaussian units with `c = 1`, the only admitted value (another value
is an unsupported request). A deck fixes its unit system through the values of one species: a species with
`|qom| = 1` and `rhoINIT = 1` has the plasma frequency 1. Lengths are then in `c/omega_p` of that species, times
in `1/omega_p`, velocities in `c`.

| Quantity | Meaning |
| --- | --- |
| `qom` | charge-to-mass ratio with its sign. Ion-normalized decks set `qom = 1` for the ions and `-m_i/m_e` for the electrons (`qom = -256.0 1.0 -256.0 1.0` in `plasmoidChain.inp`); electron-normalized decks set `qom = -1` for the electrons (`whistlerTurbulence3D.inp`). |
| `rhoINIT` | 4π times the magnitude of the species charge density: `omega_p^2 = \|qom\|*rhoINIT`. |
| `uth`, `vth`, `wth` | the standard deviation of the x, y, z velocity components of the loaded Maxwellian, `sqrt(T/m)` in units of `c` |
| `u0`, `v0`, `w0` | the drift velocity in units of `c`; a case may shape it ([cases](cases.md)) |
| `B0x`, `B0y`, `B0z` and the case amplitudes | magnetic field with `Omega_c = \|qom\|*B/c`: in an ion-normalized deck `B0 = Omega_ci/omega_pi`. E has the units of B. |
| `Lx`, `Ly`, `Lz` | box lengths; the cell size is `Lx/nxc`, `Ly/nyc`, `Lz/nzc` |
| `dt` | time step; the run ends at `t = dt*ncycles` |

Every case deck in `input/` except `whistlerTurbulence3D.inp` is ion-normalized (`qom = 1` for the ions) and
measures lengths in `d_i` and times in `1/omega_pi`; `whistlerTurbulence3D.inp` is electron-normalized and uses
`d_e` and `1/omega_pe`. A deck states its normalization in its comments. The energies and moments of
the output files use the same units ([output](output.md)).

## Time step and cells

The field solve is implicit, so neither the light wave nor the plasma frequency limits `dt`: the 2D
Gaussian-expansion deck runs at `c*dt/dx = 5` and `omega_pe*dt = 2.5`. `th` is the time-centring parameter in
`[0.5, 1]`; `th = 0.5` conserves the total energy to the solver tolerance, and `th > 0.5` damps the high
frequencies. `GMREStol` is the solver tolerance relative to the norm of the right-hand side; a solve that does
not converge in 100 iterations stops the run.

Choose `dt` and the cells from the physics the run must resolve, then check them in the *Numerical resolution*
section that `--check-input` prints ([console report](build_and_run.md#console-report)): `omega_p*dt` and
`Omega_c*dt` of each species, the skin depth, Debye length and gyroradius in cells, and the thermal and drift
step of each species in cells.

- A process that the run follows needs several steps per period and several cells per length.
- The cell need not resolve the Debye length: the Gaussian-expansion decks use cells of 10 (2D) and 11 (3D)
  electron Debye lengths, and `whistlerTurbulence3D.inp` uses 2.
- Keep the thermal and drift step of the fastest species below a cell. On a decomposed run a particle may move at
  most two cells beyond its rank in one step; a farther flight aborts the run.

## Decomposition

`XLEN`, `YLEN` and `ZLEN` set the process grid, and their product must equal the number of MPI ranks. Ranks are
numbered x-fastest: rank `r` sits at `(r % XLEN, (r / XLEN) % YLEN, r / (XLEN*YLEN))`. A cell count need not be
divisible by its process count: on an axis with `N` cells and `P` processes every process receives `N/P` cells
(integer division) and the first `N%P` processes receive one more, so rank 0 holds the largest partition. Every
axis must keep at least two cells per process, except that a field-periodic axis with one process may have a
single cell: a 2D run sets `nzc = 1`, `ZLEN = 1` and `PERIODICZ = true`.

Particle loading is reproducible for a fixed decomposition ([particle loading](#particle-loading)); a different
process grid loads different particles. A restart may change the process grid ([output](output.md)).

## Walls

An axis has walls only when its periodic key is false: `PERIODICX`, `PERIODICY`, `PERIODICZ` for the fields and
`PERIODICX_P`, `PERIODICY_P`, `PERIODICZ_P` for the particles, which default to the field value. The decks keep
the two equal.

On a field wall `bcEMface<Side>` (`Xleft`, `Xright`, `Yleft`, `Yright`, `Zleft`, `Zright`) selects:

| Code | Wall | Condition |
| --- | --- | --- |
| `0` | perfect electric conductor (PEC) | tangential E is zero |
| `1` | perfect magnetic conductor (PMC) | tangential B and normal E are zero; a mirror-symmetry plane of the fields |
| `2` | open | `--check-input` admits it with a note; `--initialize` and `--run` reject it |

Both closed walls are lossless, so the total energy stays conserved with them. Any other code rejects the deck,
also on a periodic axis. On a particle wall `bcPface<Side>` must be `1`, the mirror, which reflects the particle;
`0` (exit), `2` (reemission) and `3`, `4` (open with injection) are unsupported requests. The wall rules are in
[numerical_reference.md](../developer/numerical_reference.md#boundaries-and-migration); [support](support.md) lists
the supported combinations.

## Smoothing

`ECSimSmoothing = energy_conserving` smooths the fields of the scheme; `none` (the default) turns it off.
Smoothing is active only when `SmoothNiter > 0` and `Smooth < 1`; with `energy_conserving`, `Smooth` must lie in
`[0, 1]` and `SmoothNiter` must be non-negative. Each pass applies `ECSimSmoothingOperator`:

| Operator | One pass |
| --- | --- |
| `face7` (default) | `Smooth*f + (1 - Smooth)/6 * (sum of the six face neighbours)`; a mode along one axis is scaled by `1 - (1 - Smooth)/3*(1 - cos(k*dx))` |
| `binomial27` | the fixed weights `[1,2,1]^3/64` on the 27-node neighbourhood; `Smooth` only switches it on |

`SmoothNiter` passes act on the deposited current, on the electric field the particles gather and on both sides
of the mass term of the field equation, so the smoothed scheme keeps its energy conservation. On closed walls the
passes use the ghost signs of the wall. An open field wall with active smoothing is an input error. With active
smoothing the solver runs the smoothed full-domain schedule whatever `--mass-apply` requests
([numerical options](build_and_run.md#numerical-options)).

## Particle loading

Every species loads `npcelx*npcely*npcelz` particles in each local cell, one in each subcell of an
`npcelx x npcely x npcelz` split of the cell; a `GaussianExpansion` cutoff loads only the subcells inside it. A
particle of a uniform species carries the charge
`sign(qom) * rhoINIT * cell_volume / (4*pi*npcelx*npcely*npcelz)` and the mass `charge/qom`; a case with a
density profile scales this charge and keeps the particle count. Velocities are `drift + thermal * normal` per
component, with the drift and thermal speeds of the deck or of the case profile at the stored particle position,
plus the velocity seed of the case.

### QuietStart

`QuietStart` places the particles of every case. `true` puts each particle at the centre of its subcell, the same
points for every species, so species with equal `npcel*` sit on one another and their charges cancel point by
point. `false`, the default, draws a random position inside each subcell, a different one for every species.
`GaussianExpansion` and `WhistlerTurbulence` require `true`; `GEM3D.inp`, `GEMHarris.inp`, `doubleGEM.inp`,
`plasmoidChain.inp` and `twoStream1D.inp` also set it, and the scaling decks keep the default.

### Seeds

The generator of species `s` on rank `r` uses the seed `12345 + 100*(r*TPIC_MAX_SPECIES + s)` with the local load
index of the particle as sequence number. Each particle draws three uniform position offsets (none under
`QuietStart`), a normal pair for `u` and `v`, then one normal for `w`. A load therefore repeats for a fixed
backend, build and process grid, and changes when the decomposition changes. The case seeds of the whistler
noise (`WhistlerNoiseSeed`) and of the whistler-turbulence phases (`WhistlerTurbulenceSeed`) are independent of
the decomposition.

### CenterInitialVelocity

`CenterInitialVelocity` holds one boolean per species. For a species set to `true`, startup removes the global
charge-weighted mean of the sampled thermal velocity and of the case's velocity seed, over all ranks, so the mean
velocity of the species equals the drift its case loads (`u0`, `v0`, `w0` for a uniform species). A restart
never centres again.

## Initial fields

The uniform field `(B0x, B0y, B0z)` is added to the magnetic field of every case. A case adds its own analytic E
and B: a Harris sheet, a seed mode, a whistler wave ([cases](cases.md)). The analytic magnetic cell field is
interpolated onto the nodes before the particles use it.
