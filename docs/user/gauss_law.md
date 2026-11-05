# Gauss's law: check and monitoring

TPIC does not enforce Gauss's law: the field solve has no Poisson projection or divergence cleaning, and it sees
the charge only through the implicit current. The code measures how far the fields drift from Gauss's law
instead. This page says what is measured, why it is not zero, and how to switch the measurement on and read
it. The file layouts are in [output.md](output.md#gausscsv); the implementation is in
[gauss_diagnostic.md](../developer/gauss_diagnostic.md).

## What is measured

The residual lives at cell centres, where `B` lives:

$$r_c = \nabla_C\cdot\mathbf E - S\,\bar\rho ,$$

- $\nabla_C\cdot\mathbf E$ is the divergence of the nodal `E` with the same node-to-cell derivative that builds
  the curl. It annihilates the discrete curl of `B` exactly, so Ampère's law turns the residual into a running
  sum of the discrete continuity defect plus the GMRES residual.
- $\bar\rho$ is $4\pi$ times the charge density of the particles, deposited on the nodes and averaged over the
  eight nodes of the cell. $S$ is the solver's smoothing (`ECSimSmoothing`, `ECSimSmoothingOperator`, `Smooth`,
  `SmoothNiter`), applied when it is active; smoothing commutes with the difference and average operators, so
  this is the charge consistent with the smoothed current.
- The sample begun at cycle `n` pairs the charge of the particles at $x^{n+1/2}$ with the mean of
  $\nabla_C\cdot\mathbf E$ over the two committed fields $\mathbf E^n$ and $\mathbf E^{n+1}$, so both sides sit
  at the time $(n + 1/2)\,dt$ for every `th`.

The residual is in the units of $\nabla_C\cdot\mathbf E$.

## Why it is not zero

- **The initial state.** The initial `E` is analytic and the loaded density carries the shape error of the
  deposit and, with random positions, sampling noise, so $r^0 \ne 0$ in general. Conservation is measured by the
  change $r^n - r^0$.
- **The continuity defect.** ECSIM does not make the discrete continuity equation hold, and no particle shape can:
  with CIC weights a particle that moves inside one cell changes the residual of the cells around it even when it
  returns to its start. The defect is second order in `dx` and `dt` and accumulates as a random walk of the
  particle noise.
- **The solver tolerance.** Each step adds the divergence of the accepted GMRES residual divided by `th`; a tight
  `GMREStol` separates this term from the others.
- **Storage rounding.** Profiles that store `E` in single precision add rounding of order
  $\varepsilon\,|\mathbf E|/dx$ per step.

Nothing in the field equation damps the residual, so its change measures what the run accumulates.

## Walls

A PMC wall acts as a mirror and a PEC wall as a mirror with opposite charge. The charge of the residual follows
the same images: it is zero on the nodes of a PEC wall (the output density there holds the doubled plasma
density, for plots) and takes the doubled image value on a PMC wall. The residual over the cells that touch a
physical wall is reported separately, so that reflections at the walls can be told apart from the interior.

## Switching it on

| Key | Effect |
| --- | --- |
| `DiagnosticGaussCycle = N` | a sample begins at every multiple of `N` below `ncycles` and finishes at the next cycle; `0` (the default) disables it |
| `FieldOutputTag = ...+Gauss` | also begins a sample one cycle before every field snapshot and writes the per-cell residual `r` with the snapshot |

Every finished sample writes one row of `gauss.csv` in `SaveDirName` and prints one line in the run log:

```text
  Gauss residual: cycle 100 t = 2.512500e+01  rms = ...  drift = ...  wall = ...  charge = ...
```

## Reading the four numbers

| Column | Meaning | What to watch |
| --- | --- | --- |
| `residual_rms` | rms of `r` over the global cells | the absolute level, including the initial residual |
| `drift_rms` | rms of `r` minus its value at the first finished sample of the run | the accumulated error; a restart takes a new baseline |
| `wall_residual_rms` | rms of `r` over the cells touching a physical wall (0 without walls) | a wall contribution that grows faster than the interior |
| `charge_rms` | rms of the charge term $S\bar\rho$ | the scale: `drift_rms/charge_rms` is the relative Gauss error |

In a healthy run `drift_rms/charge_rms` stays small and grows slowly. Its three parts respond to different
settings: the particle-noise part falls as $1/\sqrt{\text{ppc}}$ with the particles per cell, the
continuity-defect part as $dx^2$, and the solver part with `GMREStol`. Two runs that change one of the three tell
the parts apart.

## From snapshots

`tools/postprocessing/analyze_tgv.py` evaluates the same residual from field snapshots of periodic runs, with the
deck's smoothing, and also checks $\nabla\cdot\mathbf B$. The deck then needs `FieldOutputTag = B+E` (`E` for the
residual alone) and `MomentsOutputTag = rho_tot+J_tot` (setting one tag turns off the default selection of both).
Because the snapshot charge sits half a step after `E`, the tool adds the half-step correction
$2\pi\,dt\,\nabla_C\cdot\mathbf J$ of the output current. It reports the rms of the residual divided by the sum
of the rms of $\nabla_C\cdot\mathbf E$ and of the charge term. It has no wall rule; see [analysis.md](analysis.md).

## Papers

- G. Lapenta, Exactly energy conserving semi-implicit particle in cell formulation, J. Comput. Phys. 334, 349
  (2017).
- Y. Chen and G. Tóth, Gauss's law satisfying energy-conserving semi-implicit particle-in-cell method, J. Comput.
  Phys. 386, 632 (2019): the cell-centre residual and charge.
