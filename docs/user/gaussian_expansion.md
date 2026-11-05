# Gaussian expansion into vacuum

The `GaussianExpansion` case: a neutral, unmagnetized Gaussian plasma (hot electrons, cold ions) released into
vacuum, compared with the exact self-similar solution of the Vlasov equations in the quasineutral limit. Decks:
[`input/gaussianExpansion3D.inp`](../../input/gaussianExpansion3D.inp) (sphere) and
[`input/gaussianExpansion2D.inp`](../../input/gaussianExpansion2D.inp) (cylinder). Analyzer:
[`tools/postprocessing/analyze_gaussianExpansion.py`](../../tools/postprocessing/analyze_gaussianExpansion.py).
The list of all cases is in [cases.md](cases.md).

## Summary

| Item | Value |
| --- | --- |
| Simulated | A neutral Gaussian plasma, Maxwellian electrons and cold ions, `E = B = 0`, expanding into vacuum inside closed walls |
| Exact solution | Self-similar quasineutral expansion (Dorozhkina and Semenov 1998; Kovalev and Bychenkov 2003), separable per Gaussian axis |
| What it tests | Energy conservation of ECSIM with `dx` much larger than the Debye length (no grid heating), ambipolar transfer of electron thermal energy to ion flow, sphericity on a Cartesian grid, `B = 0` in a radial problem, momentum, Gauss-law drift |
| Initial particles | Electrons on the ion positions (pair loading on the `QuietStart` lattice), Gaussian charges, cut at 4.3 widths, global mean velocity removed |
| Production run | `256^3` cells, `L = 14 d_i`, `sigma0 = 0.5 d_i`, `Lambda = sigma0/lambda_De0 = 100`, `dt = 0.25`, 1200 cycles (`t_end = 2.985 tau`), 8 ranks |
| 2D run | `256^2` cells, `L = 12.8 d_i`, the same plasma and time step, one rank |
| Checks | A1-A11, from the diagnostics D1-D14; the analyzer prints A1-A10 |

## Exact solution

### Sphere (three Gaussian axes)

Electrons (species 0) and ions (species 1, charge `Z e`, `Z = 1` in the decks) share the density profile

$$
n_e(\mathbf r, 0) = Z\, n_i(\mathbf r, 0) = n_0 \exp\!\left(-\frac{|\mathbf r - \mathbf r_c|^2}{2\sigma_0^2}\right),
$$

with $\sigma_0$ the rms width per axis. Electrons are Maxwellian with uniform $T_{e0}$; ions are cold or have a
uniform $T_{i0}$. There is no flow, no charge separation and no field. For $\Lambda = \sigma_0/\lambda_{De0} \gg 1$
the profile stays Gaussian, the flow is linear in radius and both species cool adiabatically:

$$
\sigma^2(t) = \sigma_0^2\left(1 + \frac{t^2}{\tau^2}\right), \qquad
\tau = \sigma_0 \sqrt{\frac{M}{Z T_{e0} + T_{i0}}}, \qquad M = m_i + Z m_e ,
$$

$$
n(\mathbf r, t) = n_0\left(\frac{\sigma_0}{\sigma}\right)^3 \exp\!\left(-\frac{r^2}{2\sigma^2}\right), \qquad
\mathbf u(\mathbf r, t) = \mathbf r\,\frac{\dot\sigma}{\sigma} = \mathbf r\,\frac{t}{t^2 + \tau^2}, \qquad
\frac{T_e(t)}{T_{e0}} = \frac{T_i(t)}{T_{i0}} = \frac{1}{1 + t^2/\tau^2}.
$$

- $M$ includes the inertia of the co-moving electrons; it changes $\tau$ by 0.5 % at $m_i/m_e = 100$.
- The expansion speed $\dot\sigma$ tends to $c_{s0} = \sqrt{(Z T_{e0} + T_{i0})/M}$, while the local sound speed
  decays as $c_{s0}\sigma_0/\sigma$: the flow becomes supersonic.
- $\tau$ depends on $m_i$ and not on $m_e$ (apart from $M$), so results against $t/\tau$ do not depend on the mass
  ratio.

**Derivation.** With Gaussian densities and the flow $\mathbf u = \mathbf r\,\dot\sigma/\sigma$, the quasineutral
field is linear, $E_r = E_c(t)\,r$. For a linear force, a drifting Maxwellian with $T \propto \sigma^{-2}$ solves the
Vlasov equation exactly, and each species obeys $m_s\sigma\ddot\sigma = T_s(t) + q_s E_c\sigma^2$. $Z$ times the
electron equation plus the ion equation eliminates $E_c$:
$M\sigma\ddot\sigma = (Z T_{e0} + T_{i0})\,\sigma_0^2/\sigma^2$, solved by
$\sigma^2 = \sigma_0^2 + (Z T_{e0} + T_{i0})\,t^2/M$.

**Ambipolar field.** From the electron equation, in code units (force per unit mass `qom*E`,
$u_{th,e} = \sqrt{T_{e0}/m_e}$ the deck `uth`):

$$
E_r(r,t) = \frac{(c_{s0}^2 - u_{th,e}^2)\,\sigma_0^2}{\mathrm{qom}_e\, \sigma^4}\; r .
$$

It is small: $E_r(\sigma_0, 0) = 4.95\times10^{-5}$ for both decks, and its energy is of order
$(\lambda_{De}/\sigma)^2 \sim 10^{-4}$ of the electron thermal energy, at or below the per-cell PIC fluctuation
field. Measure it as a shell- and time-averaged radial profile (D9). With radial $\mathbf E(r)$,
$\nabla\times\mathbf E = 0$, so $\mathbf B$ stays zero and the Vlasov-Poisson solution is also a Vlasov-Maxwell
solution.

### Per axis and the 2D cylinder

The solution is separable by axis. The Gaussian axes are those with `GaussianExpansionSigma* > 0`; call their
number $D$. Along a Gaussian axis $a$, with the deck thermal speed $v_{sa}$ of species $s$ and the species mass
$M_s \propto$ `rhoINIT/|qom|`,

$$
\sigma_a^2(t) = \sigma_{a0}^2\left(1 + \frac{t^2}{\tau_a^2}\right), \qquad \tau_a = \frac{\sigma_{a0}}{c_a}, \qquad
c_a^2 = \frac{\sum_s M_s v_{sa}^2}{\sum_s M_s}, \qquad u_a = x_a\,\frac{t}{t^2+\tau_a^2},
$$

and the velocity component $a$ cools as $1/(1 + t^2/\tau_a^2)$. For the decks $c_a = c_{s0}$. Components along a
uniform axis keep their initial temperature: no field acts there. The 2D deck is a cylinder: `SigmaZ = 0` with
`nzc = 1` and z periodic. With equal widths, $x = t/\tau$ and $E_0 = \tfrac32 N (Z T_{e0} + T_{i0})$ (all three
velocity components), only these quantities change with $D$:

| Quantity | 3D sphere | 2D cylinder |
| --- | --- | --- |
| $n(0,t)/n_0$ | $(1 + x^2)^{-3/2}$ | $(1 + x^2)^{-1}$ |
| $T_e/T_{e0}$, Gaussian components | $1/(1 + x^2)$ | $1/(1 + x^2)$ |
| $T_e/T_{e0}$, uniform component | none | $1$ |
| $\Lambda/\Lambda_0$ (centre, Gaussian components) | $(1 + x^2)^{1/4}$ | $(1 + x^2)^{1/2}$ |
| $K_i/E_0$ (cold ions) | $(m_i/M)\,x^2/(1 + x^2)$ | $\tfrac23 (m_i/M)\,x^2/(1 + x^2)$ |

The width, the flow slope, the ion velocity spread and the ambipolar slope are the same in both.

### Energy partition

For cold ions, with $E_0$ the initial kinetic energy,

$$
\frac{K_i}{E_0} = \frac{m_i}{M}\,\frac{x^2}{1+x^2}, \qquad
\frac{K_e}{E_0} = \frac{1}{1+x^2} + \frac{Z m_e}{M}\,\frac{x^2}{1+x^2}, \qquad
W_E \ll K, \qquad W_B \approx 0
$$

in 3D. $K_e$ includes the small electron flow energy. Almost all electron thermal energy ends as ion flow: 89 % at
$t = 3\tau$.

### Ion distribution

Each Gaussian-axis component of the ion velocity is Gaussian with variance

$$
\langle v_a^2 \rangle_i = c_{s0}^2\,\frac{x^2}{1+x^2} + \frac{T_{i0}/m_i}{1+x^2}.
$$

The ion energy distribution over the $D$ Gaussian components is a $\chi^2$ distribution with $D$ degrees of freedom;
in 3D, $dN/d\mathcal E \propto \sqrt{\mathcal E}\, e^{-\mathcal E/T_\mathrm{eff}}$ with
$T_\mathrm{eff} = m_i\langle v_a^2\rangle_i$. Departures appear only in the high-energy tail.

### Validity

- **Quasineutrality** needs $\Lambda \gg 1$. In the bulk $\lambda_{De} \propto \sqrt{T_e/n}$ grows as $\sigma^{1/2}$
  in 3D and stays constant in 2D, so $\Lambda$ grows: 100 at $t = 0$, 177 (3D) and 315 (2D) at $t_\mathrm{end}$.
- **The Gaussian tail is never quasineutral.** The local Debye length grows as $e^{r^2/4\sigma^2}$: 2.7, 9.5 and 55
  times $\lambda_{De0}$ at $r = 2\sigma, 3\sigma, 4\sigma$. A non-neutral ion front forms there, and the fastest ions
  exceed the Maxwellian tail. The Murakami-Basko asymptotic maximum ion energy is
  $2 Z T_{e0}\,W(\Lambda^2/2) = 13.3\,Z T_{e0}$ at $\Lambda = 100$; resolving it needs $\Delta x \lesssim \lambda_{De}$
  near the front, so it is not a pass/fail quantity at the deck resolution.
- **Initial transient.** The pair-loaded start has $\mathbf E = 0$, while the exact solution carries the ambipolar
  field at $t = 0$. The plasma builds it within about $\omega_{pe}^{-1}$ through a relative charge separation of
  order $(\lambda_{De}/\sigma_0)^2 = 10^{-4}$; it may be visible in $W_E(t)$ during the first steps.

## Units

The decks use ion units; the general convention is in [inputs.md](inputs.md).

| Quantity | Unit or relation |
| --- | --- |
| Length, time, velocity | $d_i = c/\omega_{pi}$, $\omega_{pi}^{-1}$, $c$ (`c = 1`) |
| `qom` | ions `+1`, electrons `-m_i/m_e = -100` |
| `rhoINIT` | the peak density of each species; $\omega_{ps}^2 = $ `rhoINIT*\|qom\|`, so $\omega_{pi} = 1$, $\omega_{pe} = 10$ |
| `uth`, `vth`, `wth` | the standard deviation of each velocity component, $\sqrt{T/m}$, so $T_{e0} = m_e u_{th,e}^2$ |
| Particle mass | `q/qom` |
| Output moments | `rho` $= 4\pi\sum qW/V$, `J` $= \sum q\mathbf v W/V$, mass-weighted `P_ij` $= \sum (q/\mathrm{qom})\,v_i v_j W/V$ per species |
| `diagnostics.csv` | kinetic energy $\sum (q/\mathrm{qom})\vert\mathbf v\vert^2/2$, momentum $\sum (q/\mathrm{qom})\,\mathbf v$, charge $\sum q$ |

The analyzer converts the moments: per species and axis, the density is `rho/(4*pi*qom)`, the flow
`u_a = 4*pi*J_a/rho`, and the temperature `P_aa/n - u_a^2` (D7).

## Parameters of the shipped decks

Values that `--tables` prints, or that follow from the decks and the startup report.

| Quantity | 3D deck | 2D deck |
| --- | --- | --- |
| Gaussian axes | x, y, z | x, y |
| $m_i/m_e$; $\omega_{pe}$; $d_e$ | 100; 10; 0.1 | same |
| $T_{e0}/(m_e c^2)$ | $2.5\times10^{-3}$ | same |
| $\lambda_{De0} = u_{th,e}/\omega_{pe}$ | $5\times10^{-3}$ | same |
| $\Lambda$ (start, end) | 100, 177 | 100, 315 |
| $c_{s0}$ (with $M$) | $4.975\times10^{-3}$ | same |
| $\tau$ | 100.50 | same |
| Box, cells | $14^3$, $256^3$ | $12.8^2\times0.05$, $256^2\times1$ |
| $\Delta x$ | 0.0546875 = $10.9\,\lambda_{De0}$ = $\sigma_0/9.14$ | 0.05 = $10\,\lambda_{De0}$ = $\sigma_0/10$ |
| $\omega_{pe}\Delta t$; $u_{th,e}\Delta t/\Delta x$ | 2.5; 0.229 | 2.5; 0.25 |
| $c\Delta t/\Delta x$; $c_{s0}\Delta t/\Delta x$ | 4.57; 0.023 | 5.0; 0.025 |
| $t_\mathrm{end}$ | 300 = $2.985\,\tau$ | same |
| $\sigma(t_\mathrm{end})$ | 1.574 | same |
| Nearest wall; density there / centre | $4.45\,\sigma(t_\mathrm{end})$; $5.1\times10^{-5}$ | $4.07\,\sigma(t_\mathrm{end})$; $2.6\times10^{-4}$ |
| Cutoff radius; $n/n_0$ there | $4.3\,\sigma_0 = 2.15$; $9.7\times10^{-5}$ | same |
| Mass outside the cutoff; width loss | $3.5\times10^{-4}$; 0.10 % | $9.7\times10^{-5}$; 0.04 % |
| Deposition broadening of $\sigma$ ($\Delta x^2/6$ in $\sigma^2$) | 0.10 % | 0.08 % |
| Particles per cell and species | 256 ($8\times8\times4$) | 256 ($16\times16\times1$) |
| Loaded particles per species | 65158368 | 1487048 |
| Ranks; particles per rank | 8 ($2\times2\times2$), one octant each: $1.63\times10^7$ | 1; $2.97\times10^6$ |
| Startup memory plan per rank | 7.1 GiB: particles 1.6, mass matrix 2.1, solver 1.7, fields 0.8 | |

The analyzer removes the deposition broadening from the measured widths and compares with the widths measured at
$t = 0$, which the cutoff narrows.

## Case keys and admission

The seven case keys are all optional ([input_keys.md](input_keys.md) lists their types and defaults):

| Key | Default | Meaning |
| --- | --- | --- |
| `GaussianExpansionCenterX`, `...Y`, `...Z` | `Lx/2`, `Ly/2`, `Lz/2` | the centre; it must be a grid node on every Gaussian axis and is unused on a uniform axis |
| `GaussianExpansionSigmaX`, `...Y`, `...Z` | `0` | the rms width per axis; `0` makes the axis uniform |
| `GaussianExpansionCutoff` | `0` | load only where $\sum_a ((x_a - c_a)/\sigma_a)^2 \le$ `Cutoff^2` over the Gaussian axes; `0` loads every cell |

Every species loads `rhoINIT*exp(-sum_a (x_a - c_a)^2/(2*sigma_a^2))` at the subcell centres. The general keys keep
their meaning: `rhoINIT` is the peak density, `uth`/`vth`/`wth` the thermal speeds, `npcel*` the particles per cell
inside the cutoff.

The constructor ([`gaussian_expansion.cpp`](../../src/initialization/cases/gaussian_expansion.cpp)) admits a deck
when:

1. every width is finite and non-negative and at least one is positive; `Cutoff` is finite and non-negative;
2. on every Gaussian axis, `center/dx` lies within `1e-9*max(1, C)` of an integer `C` with `0 <= C <= nc`; the
   centre used is `C*dx`;
3. pair loading holds: `QuietStart = true`, every species has the same `npcelx`, `npcely` and `npcelz`, and the sum
   of `sign(qom)*rhoINIT` is at most `1e-12` times the sum of `rhoINIT`;
4. every `rhoINIT` is positive and every drift `u0`, `v0`, `w0` is zero.

Not checked, and the deck's choice: walls, periodicity, `B0` (the solution assumes distant walls and `B0 = 0`),
the thermal speeds (warm ions are admitted) and `CenterInitialVelocity` (the decks set it for every species). A
cutoff so small that no subcell centre lies inside it gives the warning
`Species s loads no particles: no subcell centre lies within GaussianExpansionCutoff widths of the centre`.

For $Z > 1$, keep the ion `rhoINIT` equal to the electron value (`rhoINIT` is $4\pi$ times the charge density) and
set the ion `qom` to $Z m_p/m_i$: each electron then carries exactly minus the charge of its ion.

## The decks

### Production run

The production run is the shipped 3D deck,
[`input/gaussianExpansion3D.inp`](../../input/gaussianExpansion3D.inp): the parameters above, 8 ranks with one GPU
each (one LUMI-G node of 8 GCDs), for example

```sh
mpirun -np 8 tpic --run gaussianExpansion3D.inp
```

Launch lines and rank placement are in [build_and_run.md](build_and_run.md). The centre is the corner of all 8
subdomains, so each rank loads one octant and the load is balanced. The deck writes:

| Output | Cadence | Times |
| --- | --- | --- |
| Fields `B+E+Gauss`, moments `rho+rho_tot+J+PXX+PYY+PZZ` | `FieldOutputCycle = 100` | $t/\tau = 0, 0.249, \ldots, 2.985$ (13 snapshots) |
| Particle snapshots | `ParticlesOutputCycle = 400` | $t/\tau = 0, 0.995, 1.99, 2.985$ |
| `diagnostics.csv` | `DiagnosticsOutputCycle = 1` | every cycle |
| Console energy | `DiagnosticEnergyCycle = 10` | |
| `gauss.csv` | `DiagnosticGaussCycle = 10` | |

`--run` writes the cycle-0 snapshots, which the analyzer needs for the widths at $t = 0$. The deck ships with
`RestartOutputCycle = 0`. For a restartable run set `RestartOutputCycle = 400`: checkpoints at cycles 400 and 800,
and with `CallFinalize = 1` the final state; `--restart` continues from the newest complete checkpoint, on any rank
count. Checkpoints need an ADIOS2 or HDF5 build ([output.md](output.md)).

### Quick run

A smoke test: the 3D deck with

```ini
GaussianExpansionCenterX = 4.0
GaussianExpansionCenterY = 4.0
GaussianExpansionCenterZ = 4.0
Lx = 8.0
Ly = 8.0
Lz = 8.0
nxc = 128          # dx = 0.0625 = 12.5 lambda_De0 = sigma0/8
nyc = 128
nzc = 128
ncycles = 800      # t_end = 200 = 1.99 tau
npcelx = 3 3
npcely = 3 3
npcelz = 3 3
```

It loads about $4.6\times10^6$ particles per species. The walls sit at $3.59\,\sigma(t_\mathrm{end})$, where the
density is $1.6\times10^{-3}$ of the centre: enough for a smoke test, not for the A-targets at their full precision.
$\omega_{pe}\Delta t = 2.5$, $u_{th,e}\Delta t/\Delta x = 0.2$, $c\Delta t/\Delta x = 4$.

### Octant run

The problem is mirror symmetric about the three planes through the centre. At such a plane the normal $\mathbf E$
and the tangential $\mathbf B$ vanish: a perfect magnetic conductor (PMC). PMC fields with mirror particles act as
an exact symmetry plane (the image picture of [Gauss's law](gauss_law.md#walls)), so the octant
$x, y, z \ge 0$ represents the full sphere at one eighth of the cells and particles. The recipe is the 3D deck with

```ini
GaussianExpansionCenterX = 0.0   # the corner node
GaussianExpansionCenterY = 0.0
GaussianExpansionCenterZ = 0.0
Lx = 7.0
Ly = 7.0
Lz = 7.0
nxc = 128                        # dx = 0.0546875, as in the production run
nyc = 128
nzc = 128
bcEMfaceXleft = 1                # PMC on the symmetry planes
bcEMfaceYleft = 1
bcEMfaceZleft = 1
bcEMfaceXright = 0               # PEC on the far walls
bcEMfaceYright = 0
bcEMfaceZright = 0
```

and mirror particles on every face. It loads 8144796 particles per species, the count of one production rank. With
the deck's $2\times2\times2$ ranks the loaded octant ($r \le 2.15$) lies on the corner rank alone at $t = 0$;
`XLEN = YLEN = ZLEN = 1` puts the $128^3$ cells and the particles of one production rank on one GPU.

Validate it against the full sphere first. Run the octant at the quick-run geometry (`CenterX/Y/Z = 0.0`,
`Lx = Ly = Lz = 4.0`, `nxc = nyc = nzc = 64`, `npcel* = 3 3`, `ncycles = 800`, the boundaries above) and compare
with the full quick run:

- the analyzer values A2-A8 of both runs agree within their sampling noise;
- the ion density in the first node layers at the symmetry planes matches the full run's density at the same
  distance from the centre at every snapshot; a wrong fold of the deposit at the planes shows as an error of up to
  a factor 2 there.

Two analyzer outputs assume the full Gaussian: A9 does not apply (the outward flow of one octant carries momentum,
which the symmetry planes take), and the peak normalization of the density panel of `<prefix>_profiles` is off by
the factor 8.

### 2D deck

[`input/gaussianExpansion2D.inp`](../../input/gaussianExpansion2D.inp) is the cylinder at the 2D production
resolution on one GPU. It differs from the 3D deck in these lines:

```ini
SimulationName = gaussianExpansion2D
GaussianExpansionCenterX = 6.4    # 128*dx
GaussianExpansionCenterY = 6.4
GaussianExpansionCenterZ = 0.025  # unused: z is uniform
GaussianExpansionSigmaZ = 0.0     # a cylinder, uniform along z
Lx = 12.8
Ly = 12.8
Lz = 0.05                         # one cubic cell
nzc = 1
XLEN = 1                          # XLEN = YLEN = 2 runs check A11
YLEN = 1
ZLEN = 1
PERIODICZ = true
PERIODICZ_P = true
npcelx = 16 16
npcely = 16 16
npcelz = 1 1
DiagnosticEnergyCycle = 100
```

Its parameters are in the table above. The analyzer also prints the electron temperature along the uniform z axis,
whose analytic value stays 1.

## Initial state and its t = 0 checks

TPIC loads the state as follows
([`gaussian_expansion.hpp`](../../include/tpic/initialization/cases/gaussian_expansion.hpp),
[`particle_support.hpp`](../../include/tpic/initialization/particles/particle_support.hpp),
[`particle_centering_device.hpp`](../../include/tpic/initialization/particles/particle_centering_device.hpp);
loading and `QuietStart` in general: [inputs.md](inputs.md)).

- **Positions.** `QuietStart = true` puts the `npcelx*npcely*npcelz` particles of a cell at the subcell centres.
  With the centre on a grid node the lattice is symmetric about it along every Gaussian axis, and no point lies on
  a symmetry plane.
- **Cutoff.** Only subcell centres inside the cutoff ellipsoid load; outside it the vacuum holds no particle. The
  host decides which subcells load from their global indices, so the planned and the loaded counts agree exactly.
- **Charges.** Each particle carries
  `sign(qom)*(dx*dy*dz/ppc)*(rhoINIT/(4*pi))*exp(-sum_a (x_a - c_a)^2/(2*sigma_a^2))`:
  variable weights give uniform statistics down to $10^{-4}\,n_0$. The mass is `q/qom`.
- **Pair loading.** Species with equal `npcel*` sit on the same points; with equal `rhoINIT` their charges are exact
  negatives, so the deposited `rho_tot` vanishes on every node to rounding and $\mathbf E = 0$ satisfies the
  discrete Gauss law at $t = 0$.
- **Velocities.** Each electron component is drawn from $\mathcal N(0, u_{th,e}^2)$, independently of position; the
  ions are at rest. The random draws depend on the rank, so the initial state depends on the decomposition (A11).
- **Zero momentum.** `CenterInitialVelocity = true true` removes, per species, the charge-weighted mean velocity
  summed over all ranks (a global mean, not a per-rank one). Within a species the charge-weighted mean is the
  mass-weighted one, so each species momentum is zero to rounding.
- **Fields.** The case returns $\mathbf E = 0$ and $\mathbf B = 0$ (`B0 = 0`), ghosts included; TPIC applies no
  divergence correction.

Checks of the cycle-0 state:

| Check | Where | Expected |
| --- | --- | --- |
| Loaded particles | startup report, `Particles: ... global=` (both species) | 130316736 (3D), 2974096 (2D): 65158368 and 1487048 per species |
| Net charge | `diagnostics.csv`, `charge` | 0 to rounding |
| $\max\vert\rho_\mathrm{tot}\vert/\max\vert\rho_1\vert$ | cycle-0 `rho_tot` and `rho_1` | rounding level |
| $W_E(0)$, $W_B(0)$ | `diagnostics.csv` | 0 |
| Momentum | `momentum_x/y/z` and per species | rounding level |
| $K_e(0)$ | `kinetic_energy_0` | $\tfrac32 N_e m_e u_{th,e}^2$, with $N_e m_e$ = `charge_0/qom_0` |
| Measured $\sigma_0$ | analyzer line `sigma0 deck / measured` | about $0.9990\,\sigma_0$ (3D), $0.9996\,\sigma_0$ (2D), from the cutoff |
| Ion `rho` profile | D5 at $t = 0$ | $n_0 e^{-r^2/2\sigma_0^2}$ |
| Electron temperature | D7 at $t = 0$ | $u_{th,e}^2 = 2.5\times10^{-3}$ per component within sampling error |
| Gauss residual | first row of `gauss.csv` | not rounding: the sample spans the first step ([Gauss's law](gauss_law.md#why-it-is-not-zero)) |

## Boundaries and solver

| Setting | 3D deck | 2D deck | Reason |
| --- | --- | --- | --- |
| Periodicity (fields, particles) | none | z only | Periodic Gaussian axes would let escaping electrons re-enter on the opposite side |
| EM walls | PEC (`0`) on every face | PEC on x and y | Globally neutral, radial problem: image effects are negligible and there is no radiation to absorb |
| Particle walls | mirror (`1`) on every face | mirror on x and y | The wall density at $t_\mathrm{end}$ is $5.1\times10^{-5}$ (3D) and $2.6\times10^{-4}$ (2D) of the centre |
| `th` | 0.5 | 0.5 | Exact energy conservation holds only for $\theta = 1/2$ |
| Smoothing | `energy_conserving`, `face7`, `Smooth = 0.2`, `SmoothNiter = 6` | same | Lower noise in D7-D9 with the energy identity kept ([inputs.md](inputs.md)) |
| `GMREStol` | `1E-15` | same | Energy is conserved up to the linear-solver tolerance |

- PEC walls carry no Poynting flux and the mirror conserves kinetic energy, so the total energy stays conserved
  with walls. Each bounce takes $2 m v_n$ of momentum; few particles reach the walls.
- Electrons that escape the ambipolar potential are a small fraction at $\Lambda = 100$; the mirror walls return
  them to the plasma.
- GMRES runs with restart 30 and at most 100 iterations; the tolerance is relative to the norm of the right-hand
  side. With $c\Delta t/\Delta x \approx 5$ the vacuum part of the field operator is stiff, and a tolerance near
  the rounding floor needs many iterations: watch the `Maxwell GMRES : converged in N iterations` line. A solve that
  does not converge stops the run with `GMRES did not converge; timestep was not committed`.
- Gauss's law is monitored, not enforced: ECSIM has no mechanism that keeps
  $\nabla\cdot\mathbf E = 4\pi\rho$, and measuring the drift is part of the test (D11, A10;
  [gauss_diagnostic.md](../developer/gauss_diagnostic.md)).

## Diagnostics and the analyzer

Here $x = t/\tau$, $\mathbf d = \mathbf x - \mathbf r_c$, $\hat r = \sqrt{\sum_a (d_a/\sigma_a)^2}$ over the Gaussian
axes, with the measured widths. Grid sums over the box use the dual-volume node weights; D4 and D9 sample
interior nodes only, where the weight is 1.

| # | Quantity | How the analyzer computes it | Compare with |
| --- | --- | --- | --- |
| D1 | Energy budget $K_e$, $K_i$, $W_E$, $W_B$, total | `diagnostics.csv`, every cycle | $E_\mathrm{tot}$ constant; $K_s/E_0$ of the energy partition |
| D2 | Width $\sigma_a(t)$ | ion `rho`: $\sigma_a^2 = \sum m\,d_a^2/\sum m - \Delta x_a^2/6$ | $\sigma_{a0}\sqrt{1+x^2}$ |
| D3 | Sphericity | the widths of D2 against each other | equal |
| D4 | Centre density | mean ion `rho` over $\hat r < 0.2$ | $(1+x^2)^{-D/2}$ times its $t = 0$ value |
| D5 | Self-similar profile | shell mean of ion `rho` in $\hat r$ (shells 0.1 wide), over the peak from the total charge and the widths | $e^{-\hat r^2/2}$ |
| D6 | Ion flow | slope of $u_a = J_a/\rho_i$ against $d_a$, mass-weighted over $\hat r < 1.5$; shell mean of $u_r$ (shells $\Delta x$ wide) | $t/(t^2+\tau^2)$ |
| D7 | Electron temperature | species 0 per axis, $\sum (P_{aa} - n u_a^2)/\sum n$ over $\hat r < 1$, over $u_{th,e}^2$ | $1/(1+x^2)$ on Gaussian axes, 1 on a uniform axis |
| D8 | Ion temperature | the same for species 1, over $c_{s0}^2$ | stays $\ll T_e(t)$ for cold ions; growth is numerical ion heating |
| D9 | Ambipolar field | shell mean of $E_r = \mathbf E\cdot\hat{\mathbf d}$ (shells $2\Delta x$ wide, $r < 3\sigma$) | linear, the slope of the ambipolar field; noisy |
| D10 | Magnetic energy | `magnetic_energy` of `diagnostics.csv`; per component from the `B` snapshots | noise level, not growing |
| D11 | Gauss residual | `residual_rms` and `drift_rms` of `gauss.csv`, over the rms of the cycle-0 ion `rho` ($4\pi\rho_i$; $\rho_\mathrm{tot} \approx 0$) | report |
| D12 | Total momentum | `momentum_x/y/z` of `diagnostics.csv` | about 0 |
| D13 | Ion velocity distribution | particle snapshots: charge-weighted histogram of the ion components, Gaussian axes pooled | Gaussian, std $c_{s0}\,x/\sqrt{1+x^2}$ (cold ions) |
| D14 | Ion energy distribution | particle snapshots: $\sum_a v_a^2/(2 v_\mathrm{std}^2)$, charge-weighted | $\chi^2$ with $D$ components; a tail above it from the ion front |

The electron flow must come out of D7: at $m_i/m_e = 100$, $t = 3\tau$ and $r = \sigma$, $u/v_{th,e}(t) = 0.30$, and
ignoring it would add $u^2/3$, a bias of 3 % at $r = \sigma$ and 7 % at $r = 1.5\sigma$.

Run the analyzer from the repository root:

```sh
python3 tools/postprocessing/analyze_gaussianExpansion.py --run-dir /path/to/run
python3 tools/postprocessing/analyze_gaussianExpansion.py --input input/gaussianExpansion3D.inp --tables
```

It reads the single `GaussianExpansion` deck of `--run-dir` (or `--input`), `diagnostics.csv`, `gauss.csv`, the
field files `rho_<s>`, `J_<s>`, `PXX_<s>`, `PYY_<s>`, `PZZ_<s>`, `E`, `B` and the particle snapshots
(`--no-particles` skips them). It writes into `--output-dir` (default the run directory), each as PNG and PDF:
`<prefix>_overview` (D2, D4, D6-D8, D1), `_width_density`, `_energy_share`, `_density_maps` ($D \ge 2$),
`_density_3d` (`--density-3d`, $D = 3$, needs pyvista), `_profiles` (D5, D6, D9), `_radial_flow`, `_conservation`
(D1, D10-D12), `_energy_gauss`, `_ions` (D13, D14) and `_phase_space`; plus `<prefix>_snapshots.csv` and
`<prefix>_summary.txt` with the derived parameters, the analytic table and the checks A1-A10. The default prefix
is `gaussian_expansion`. The analyzer needs the cycle-0 field snapshot and at least the ion `rho`; other tools are
in [analysis.md](analysis.md).

## Expected values

The solution against $t/\tau$ does not depend on the absolute parameters; `--tables` prints it for a deck. Both
decks, with $m_i/M = 1/1.01$:

| $t/\tau$ | $\sigma/\sigma_0$ | $T_e/T_{e0}$ | $\tau\,u_a/x_a$ | ion $v_a$ std $/c_{s0}$ | 3D $n(0)/n_0$ | 3D $K_i/E_0$ | 3D $K_e/E_0$ | 2D $n(0)/n_0$ | 2D $K_i/E_0$ | 2D $K_e/E_0$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 0.5 | 1.1180 | 0.8000 | 0.4000 | 0.4472 | 0.7155 | 0.1980 | 0.8020 | 0.8000 | 0.1320 | 0.8680 |
| 1 | 1.4142 | 0.5000 | 0.5000 | 0.7071 | 0.3536 | 0.4950 | 0.5050 | 0.5000 | 0.3300 | 0.6700 |
| 2 | 2.2361 | 0.2000 | 0.4000 | 0.8944 | 0.0894 | 0.7921 | 0.2079 | 0.2000 | 0.5281 | 0.4719 |
| 3 | 3.1623 | 0.1000 | 0.3000 | 0.9487 | 0.0316 | 0.8911 | 0.1089 | 0.1000 | 0.5941 | 0.4059 |

In code units ($\tau = 100.50$, $c_{s0} = 4.975\times10^{-3}$, $\sigma_0 = 0.5$), for both decks:

| $t/\tau$ | $t$ | $\sigma$ | $u_a/x_a$ | ion $v_a$ std | $E_r/r$ |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.5000 | 0 | 0 | $9.90\times10^{-5}$ |
| 0.5 | 50.25 | 0.5590 | $3.980\times10^{-3}$ | $2.225\times10^{-3}$ | $6.34\times10^{-5}$ |
| 1 | 100.50 | 0.7071 | $4.975\times10^{-3}$ | $3.518\times10^{-3}$ | $2.48\times10^{-5}$ |
| 2 | 201.00 | 1.1180 | $3.980\times10^{-3}$ | $4.450\times10^{-3}$ | $3.96\times10^{-6}$ |
| 2.985 ($t_\mathrm{end}$) | 300 | 1.5741 | $2.997\times10^{-3}$ | $4.718\times10^{-3}$ | $1.01\times10^{-6}$ |

A run compares with these values evaluated at its measured $\sigma_0$, which the analyzer does.

## Acceptance checks

The analyzer prints `Acceptance checks A1-A10 of docs/user/gaussian_expansion.md`, one line per check with its
value, target and `ok` or `over`. The targets:

| # | Check | Analyzer target |
| --- | --- | --- |
| A1 | Total energy (D1): $\max\vert E - E_0\vert/E_0$ over the run | $\le 10^{-4}$; no secular trend. A monotonic drift is a bug or a non-conserving setting |
| A2 | Width (D2): $\max\vert\sigma_a/\mathrm{analytic} - 1\vert$ per Gaussian axis, all field snapshots | $\le 10^{-2}$ |
| A3 | Sphericity (D3): $\max\vert\sigma_a/\sigma_b - 1\vert$ ($D \ge 2$) | $\le 5\times10^{-3}$ |
| A4 | Flow slope (D6): $\max\vert\mathrm{slope}/\mathrm{analytic} - 1\vert$ for $t \ge \tau/4$ | $\le 3\times10^{-2}$ |
| A5 | Core electron temperature (D7) on the Gaussian axes | $\le 5\times10^{-2}$; it must decrease, a rise is numerical heating |
| A6 | Ion velocity std (D13) per particle snapshot with a nonzero analytic spread ($t > 0$ for cold ions) | $\le 2\times10^{-2}$ |
| A7 | $K_i$ and $K_e$ (D1) against the energy partition, field-snapshot cycles with $t \ge \tau/4$ | $\le 3\times10^{-2}$ |
| A8 | $\max W_B/W_E$ (D10) for $t \ge \tau/4$ | $\le 10^{-2}$, not growing exponentially |
| A9 | Momentum (D12): $\max\vert\mathbf P\vert/(M_i c_{s0})$, $M_i$ the ion mass | $\le 10^{-3}$ |
| A10 | Gauss residual (D11): first and last residual and last drift, over the rms of $4\pi\rho_i$ at $t = 0$ | no fixed target. Slow growth at the plasma edge is expected for ECSIM; growth by orders of magnitude points to a deposition or boundary bug |
| A11 | Decomposition independence: the 2D deck at `XLEN = YLEN = 1` against `2`, or the quick run at 1 against $2\times2\times2$ | holds only statistically: the A1-A10 values agree within their sampling noise, because the velocity draws depend on the rank. The analyzer does not print A11 |

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Expansion off by about $\sqrt2$ in $\tau$ | `uth` set as $\sqrt{2T/m}$ instead of the per-component std $\sqrt{T/m}$, or `rhoINIT`/`qom` not in the units above |
| Admission error `GaussianExpansion centre must be a grid node on every axis with a positive width` | the centre is not an integer multiple of `L/nc` on a Gaussian axis |
| Admission error `pair loading requires ...` | `QuietStart` false, unequal `npcel*`, or a non-neutral `rhoINIT` set |
| Warning `Species s loads no particles` | the cutoff ellipsoid holds no subcell centre |
| $T_e$ flat or rising | energy not conserved: `th` not 0.5 or a non-conserving setting; check D1 first |
| Faster expansion along one axis | a periodicity flag left `true` or a wrong face boundary; check D3 at $t = 0$ |
| Density spike or hole at subdomain boundaries | ghost summation of moments or mass matrices; compare decompositions (A11) |
| Density error at the symmetry planes (octant) | the planes do not act as symmetry planes; see the octant validation |
| `rho_tot` or $W_E$ not at rounding level at $t = 0$ | a loading bug: admission guarantees shared positions and opposite charges |
| $W_B$ grows steadily | $\mathbf E$ develops a curl: deposition anisotropy, a staggering bug in $\mathbf J$ or $\mathbf E$, or Gauss-error feedback |
| Centre-of-mass drift | `CenterInitialVelocity` off, or many wall bounces |
| `GMRES did not converge; timestep was not committed` | the 100-iteration limit: raise `GMREStol` or reduce `dt` |
| Gauss residual grows at the plasma edge | expected: ECSIM does not enforce Gauss's law |
| Analyzer error `the first field snapshot is cycle N` | the data directory lacks the cycle-0 snapshot |
| Analyzer error `no ion density rho_1` | add `rho` to `MomentsOutputTag` |

## Variants and scans

| Variant | Change | Purpose |
| --- | --- | --- |
| $\Lambda$ scan | $\sigma_0 = 0.125, 0.25, 0.5, 1.0$, scaling `L*`, `GaussianExpansionCenter*` and `dt` by the same factor (keep `n*c`, `ncycles`, `uth`) | $\Lambda = 25, 50, 100, 200$, $\tau = 25.1, 50.2, 100.5, 201.0$, $\omega_{pe}\Delta t = 0.625, 1.25, 2.5, 5$. Bulk agreement improves with $\Lambda$; small-$\Lambda$ deviations sit in the tail and front |
| Resolution | `nxc = nyc = nzc = 128, 256, 512` at fixed `L*` ($\sigma_0/\Delta x$ = 4.57, 9.14, 18.3), ranks scaled with the cells | spatial convergence of D2-D7 |
| Time step | `dt = 0.125, 0.25, 0.5` with `ncycles = 2400, 1200, 600` | temporal convergence; GMRES needs more iterations at larger $c\Delta t/\Delta x$ |
| Particles per cell | `npcel*` from $4\times4\times2$ to $8\times8\times4$ and above | noise of D7-D9, ion heating (D8) |
| Warm ions | ion `uth = vth = wth = 0.00158` ($T_{i0} = 0.1\,T_{e0}$) | the solution with $T_{i0}$, $\tau = 95.8$; the analyzer takes the ion thermal speeds from the deck |
| Mass ratio | `qom_e = -25` with `uth_e = 0.025`, or `-400` with `0.1` (same $\Lambda$ and $T_{e0}$) | results against $t/\tau$ unchanged ($\tau = 102.0$, 100.1) |
| Smoothing | `ECSimSmoothing = none` | A1 must hold; compare the noise of D7 and D9 |
| Front energy | $\Delta x \lesssim \lambda_{De0}$ near the edge | towards the Murakami-Basko value $2 Z T_{e0} W(\Lambda^2/2)$: 8.6, 10.9, 13.3, 15.7 $Z T_{e0}$ for $\Lambda$ = 25, 50, 100, 200 |
| Ellipsoid or slab | unequal `GaussianExpansionSigma*`, or only `SigmaX > 0` | the per-axis solution with its own $\tau_a$ |

## References

**Exact solution**

- D. S. Dorozhkina and V. E. Semenov, "Exact solution of Vlasov equations for quasineutral expansion of plasma bunch
  into vacuum," Phys. Rev. Lett. 81, 2691 (1998). Kinetic electrons and ions: any mass ratio, any ratio of electron
  to ion thermal energy, a wide class of distributions and profiles.
- D. S. Dorozhkina and V. E. Semenov, "Exact solution of the problem of quasineutral expansion into vacuum of a
  localized collisionless plasma with cold ions," JETP Lett. 67, 573 (1998), doi:10.1134/1.567728. Kinetic
  electrons, cold fluid ions, a finite mass ratio and a bunch that need not be spherical (the per-axis solution).
- A. V. Baitin and K. M. Kuzanyan, "A self-similar solution for expansion into a vacuum of a collisionless plasma
  bunch," J. Plasma Phys. 59, 83 (1998). The 1D version.
- V. F. Kovalev and V. Yu. Bychenkov, "Analytic solutions to the Vlasov equations for expanding plasmas," Phys. Rev.
  Lett. 90, 185004 (2003).
- V. F. Kovalev, V. Yu. Bychenkov and V. T. Tikhonchuk, "Particle dynamics during adiabatic expansion of a plasma
  bunch," JETP (2002).
- S. Laha et al., "Experimental realization of an exact solution to the Vlasov equations for an expanding plasma,"
  Phys. Rev. Lett. 99, 155001 (2007), arXiv:0705.3056.

**Ion front, charge separation, finite $\Lambda$**

- P. Mora, "Plasma expansion into a vacuum," Phys. Rev. Lett. 90, 185002 (2003).
- P. Mora, "Collisionless expansion of a Gaussian plasma into a vacuum," Phys. Plasmas 12, 112102 (2005).
- M. Murakami and M. M. Basko, "Self-similar expansion of finite-size non-quasi-neutral plasmas into vacuum: relation
  to the problem of ion acceleration," Phys. Plasmas 13, 012105 (2006).
- F. Peano et al., "Ergodic model for the expansion of spherical nanoplasmas," arXiv:0704.3211, and "Dynamics and
  control of the expansion of finite-size plasmas," arXiv:physics/0701124.
- A. V. Gurevich, L. V. Pariiskaya and L. P. Pitaevskii, Sov. Phys. JETP 22, 449 (1966). The planar self-similar
  expansion.

**ECSIM and Gauss's law**

- G. Lapenta, "Exactly energy conserving semi-implicit particle in cell formulation," J. Comput. Phys. 334, 349
  (2017).
- G. Lapenta, D. Gonzalez-Herrero and E. Boella, "Multiple-scale kinetic simulations with the energy conserving
  semi-implicit particle in cell method," J. Plasma Phys. 83(2) (2017).
- Y. Chen and G. Tóth, "Gauss's law satisfying energy-conserving semi-implicit particle-in-cell method," J. Comput.
  Phys. 386, 632 (2019).
- M. Campos Pinto and V. Pagès, "A semi-implicit electromagnetic FEM-PIC scheme with exact energy and charge
  conservation," HAL hal-02521069.
