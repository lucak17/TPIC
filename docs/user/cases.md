# Cases

The deck key `Case` selects one initialization case: the analytic E and B at startup, and how each species is
loaded. This page lists the cases, the decks in [input/](../../input) that run them, and what each case does
with its keys. Deck syntax, particle loading, `QuietStart`, the loading seeds and `CenterInitialVelocity` are
in [inputs](inputs.md); every key, with its type and default, is in [input keys](input_keys.md); the analyzers
are in [analysis](analysis.md).

## Catalog

Cells are `nxc x nyc x nzc`; ranks are `XLEN*YLEN*ZLEN` of the deck. "Outputs" are the outputs the analyzer
reads; `diagnostics.csv` is written whenever `DiagnosticsOutputCycle` (default `FieldOutputCycle`) is positive
([output](output.md)).

| `Case` | Deck in `input/` | Dimension, cells | Ranks | Outputs the analysis needs | Analyzer | Paper (deck header) |
| --- | --- | --- | --- | --- | --- | --- |
| `Uniform` | `uniformStartup.inp` | 3D, 7 x 3 x 2 | 1 | none (quick-start case, field output off) | none | none |
| `Uniform` | `whistlerAnisotropy2D.inp` | 2D, 1024 x 64 x 1 | 1 | `B` every cycle | `analyze_whistler_spectrum.py` | D. Gonzalez-Herrero, E. Boella and G. Lapenta, Comput. Phys. Commun. 229, 162 (2018), section 5 |
| `GEM` | `GEM3D.inp` | 3D, 64 x 64 x 64 | 1 | `B`, `E`; `diagnostics.csv` for the energy plot; `rho_<s>` for the 3D view | `analyze_reconnection.py` | J. Birn et al., J. Geophys. Res. 106, 3715 (2001) |
| `GEMHarris` | `GEMHarris.inp` | 3D, 64 x 64 x 64 | 1 | as `GEM` | `analyze_reconnection.py` | J. Birn et al. (2001) |
| `GEMHarris` | `reconnection_comparison/GEM3D.inp` | 2D, 400 x 200 x 1 | 1 | as `GEM` | `analyze_reconnection.py` | J. Birn et al. (2001) |
| `GEMnoPert` | none | | | as `GEM` | `analyze_reconnection.py` | |
| `DoubleGEM` | `doubleGEM.inp` | 2D, 128 x 64 x 1 | 4 | as `GEM` | `analyze_reconnection.py` | J. Birn et al. (2001) |
| `DoubleGEMnoPert` | none | | | as `GEM` | `analyze_reconnection.py` | |
| `plasmoidChain` | `plasmoidChain.inp` | 3D, 512 x 192 x 128 | 1 | | none | S. Markidis et al., Phys. Plasmas 20, 082105 (2013) |
| `ForceFree` | none | | | | none | |
| `Weibel` | `weibel2D.inp` | 2D, 448 x 448 x 1 | 1 | `B`; `E` for the `Ex` plot; electron `rho`, `PXX`, `PYY`, `PZZ` for the measured moments; `diagnostics.csv` | `analyze_weibel.py` | E. S. Weibel, Phys. Rev. Lett. 2, 83 (1959) |
| `Whistler` | `whistler2D.inp` | 2D, 1024 x 128 x 1 | 1 | `B`; `E` and `J` when written | `analyze_whistler.py` | none |
| `TaylorGreenVortex` | `TaylorGreenVortex3D.inp` | 3D, 120 x 120 x 120 | 1 | `B`, `E`, total and per-species `rho`, `J` and the six pressure components; `diagnostics.csv`; the console log | `analyze_tgv.py` | none |
| `TwoStream` | `twoStream1D.inp` | quasi-1D, 128 x 4 x 1 | 1 | `E`; `diagnostics.csv`; particle snapshots for the phase space and for the explicit comparison (set `ParticlesOutputCycle > 0`) | `analyze_two_stream.py`; `explicit_es_pic.py` with `analyze_explicit_comparison.py` | none |
| `fluxRope` | `fluxRope3D.inp` | 3D, 64 x 64 x 128 | 8 | | none | S. Markidis et al., Plasma Phys. Control. Fusion 56, 064010 (2014) |
| `GaussianExpansion` | `gaussianExpansion3D.inp`, `gaussianExpansion2D.inp` | 3D, 256 x 256 x 256; 2D, 256 x 256 x 1 | 8; 1 | `B`, `E`, per-species `rho`, `J`, `PXX`, `PYY`, `PZZ`; `diagnostics.csv`, `gauss.csv`; particle snapshots | `analyze_gaussianExpansion.py` | D. S. Dorozhkina and V. E. Semenov, JETP Lett. 67, 573 (1998) and Phys. Rev. Lett. 81, 2691 (1998) |
| `WhistlerTurbulence` | `whistlerTurbulence3D.inp` | 3D, 300 x 300 x 300 | 32 | `B`; electron `rho`, `J`, `PXX`, `PYY`, `PZZ`; `diagnostics.csv`; particle snapshots | `analyze_whistler_turbulence.py` | S. P. Gary, O. Chang and J. Wang, ApJ 755, 142 (2012); S. Saito, S. P. Gary, H. Li and Y. Narita, Phys. Plasmas 15, 102305 (2008) |

The shipped decks already write the outputs their analyzer needs, except the particle snapshots of
`twoStream1D.inp`. The decks that state their units use ion units (lengths in `d_i = c/omega_pi`, times in
`1/omega_pi`, `c = 1`) with the mass ratio of their header; `whistlerTurbulence3D.inp` uses electron units. The
scaling decks are in `performance/scaling/` ([scaling](../../performance/scaling/README.md)).

`input/reconnection_comparison/` holds `GEM3D.inp`, a 2D `GEMHarris` run with `currentFromAmpere = true`,
face7 smoothing and the X-point hump of `GEM` written as `GEMHarris` keys, and a second deck that describes the
same setup in the input format of an external code. The two runs are compared with `analyze_reconnection.py`
([analysis](analysis.md)).

## Rules that hold for every case

The uniform field `(B0x, B0y, B0z)` is added to the field of every case. A case's own field and its loading come
from its keys alone; every scalar key is parsed whatever `Case` selects, so an invalid value of any case key
(for example an unknown `WhistlerSeedMode`) rejects a deck of any case. Each case checks its keys when the input is
prepared; a violation is an input error that names the rule. A few combinations that are valid input but not
supported (marked below) are reported as unsupported requests instead; [build and run](build_and_run.md)
gives the exit codes of the two kinds. Every case needs `c = 1`; any other value is an unsupported request.

A case is one struct declared in [include/tpic/initialization/cases](../../include/tpic/initialization/cases)
and implemented in [src/initialization/cases](../../src/initialization/cases); the interface and the steps that
add a case are in
[case_interface.hpp](../../include/tpic/initialization/cases/case_interface.hpp). `harris_sheet` holds
`GEMHarris` and its fixed variants; `Uniform` has a header only.

## Uniform

No seed field: E is zero and B is the uniform `(B0x, B0y, B0z)`. Every species is a uniform Maxwellian with the
drift `(u0, v0, w0)` and the thermal speeds `(uth, vth, wth)` of the deck. `uniformStartup.inp` is the
quick-start case of [build and run](build_and_run.md). `whistlerAnisotropy2D.inp` is a uniform magnetized
plasma, `B0` along x, whose electrons are hotter across the field than along it; it seeds no wave, and the
particle noise excites the waves along the field.

## Harris sheets: GEMHarris, GEM, GEMnoPert, DoubleGEM, DoubleGEMnoPert, plasmoidChain

The sheet amplitude is `HarrisB0x`, not `B0x`, and the half-thickness is `delta`. A single sheet has
`Bx = HarrisB0x*tanh((y - Ly/2)/delta)`. The double sheet (`DoubleGEM`, `DoubleGEMnoPert`) has two sheets of
opposite field at `y = Ly/4` and `y = 3Ly/4`:
`Bx = HarrisB0x*(-1 + tanh((y - Ly/4)/delta) - tanh((y - 3Ly/4)/delta))`; `doubleGEM.inp` is periodic on every
axis.

| `Case` | Sheets | Seed | Reads the `GEMHarris` keys |
| --- | --- | --- | --- |
| `GEMHarris` | one | chosen in the deck | yes |
| `GEM` | one | fixed X-point hump | no |
| `GEMnoPert` | one | none | no |
| `DoubleGEM` | two | fixed X-point hump on each sheet | no |
| `DoubleGEMnoPert` | two | none | no |
| `plasmoidChain` | one | none | no |

Density. In every Harris case but `plasmoidChain` a species receives the sheet density when one of its `u0`,
`v0`, `w0` exceeds `1e-10` in magnitude; any other species is a uniform background, whatever its charge sign.
In `plasmoidChain` the first species of each charge sign receives the sheet density and the later species of
that sign are uniform. A sheet species keeps its particle count; the charge of a particle is scaled by the mean
of `sech^2` at the two y nodes of its cell (the sum of both sheets for the double sheet). In the double sheet
the z drift of every species is multiplied by `-tanh((y - Ly/2)/delta)`, so the sheet current reverses
between the two sheets.

Fixed seed. The X-point hump of `GEM` and `DoubleGEM` has the amplitude `0.4*HarrisB0x`, Gaussian widths
`delta` and the wavenumbers `pi/(10*delta)`, centred on the X line (`x = Lx/2`, `y = Ly/2`; for the double sheet
`(Lx/4, Ly/4)` and, with opposite sign, `(3Lx/4, 3Ly/4)`).

GEMHarris seeds. Two seeds are added to the single sheet, both as fractions of `HarrisB0x` and measured from
the box centre:

- `pertGEM` is one wavelength of the box in x and in y, with the out-of-plane potential
  `(pertGEM*HarrisB0x*Lx/(2*pi)) * cos(2*pi*x/Lx) * cos(2*pi*y/Ly)`.
- `pertHump` is a Gaussian hump with the potential
  `(pertHump*HarrisB0x/ky) * exp(-x^2/wx^2 - y^2/wy^2) * cos(kx*x) * cos(ky*y)`, widths
  `wx = deltaxHump*delta` and `wy = deltayHump*delta`, and wavenumbers `kx = kxHump`, `ky = kyHump`; a negative
  value selects `2*pi/Lx` and `pi/Ly`. The hump is not periodic: choose widths that decay inside the box.

`pertHump = 0.4*pi/10`, `deltaxHump = deltayHump = 1` and `kxHump = kyHump = pi/(10*delta)` give the seed of
`GEM` to rounding. With all keys at their defaults `GEMHarris` equals `GEMnoPert`.

`currentFromAmpere` (GEMHarris only) loads the current of the initial field: every species with a nonzero `w0`
drifts along z with the curl of sheet and seed at the particle position, shared between the species in
proportion to `sign(qom)*w0`, and has no in-plane drift. A particle then carries exactly its share of
`c*curl(B)/(4*pi)`, so its drift differs from a uniform `w0` by the ratio of the sheet density at the particle
to the density of its cell. The drift is the current over the sheet density and grows without bound where a
seed outlasts the sheet: the option serves a hump about as wide as the sheet, and a deck with `pertGEM` or a
wide hump is refused by the drift rule below. `spatiallyVaryingThermal` acts with `currentFromAmpere` only: it
sets `wth` of the drifting species from the pressure `rho*(drift^2 + wth^2)` of their Maxwellian, which returns
the deck `wth` to rounding.

Admission:

- Every Harris case requires a finite `HarrisB0x` and `delta > 0`, and the seeded field must stay finite over
  the box and one halo cell.
- `GEMHarris` requires finite seed keys, positive `deltaxHump` and `deltayHump`, and a nonzero `kyHump` when
  `pertHump` is nonzero.
- With `currentFromAmpere`: `u0 = v0 = 0` for every species, a nonzero sum of `sign(qom)*w0`, and a drift that
  stays at or below `c` on every row of particles.

`analyze_reconnection.py` reads the X lines of `GEM`, `GEMnoPert`, `GEMHarris`, `DoubleGEM` and
`DoubleGEMnoPert`.

## ForceFree

A force-free current sheet: `Bx = ForceFreeB0x*tanh((y - Ly/2)/delta)` and
`Bz = ForceFreeB0z/cosh((y - Ly/2)/delta)`, with uniform loading of every species. It requires finite
`ForceFreeB0x` and `ForceFreeB0z` and `delta > 0`. No deck ships for it.

## Weibel

A seeded transverse magnetic mode on a uniform plasma: `Bz = WeibelSeedAmplitude*cos(k*x)` with
`k = 2*pi*WeibelSeedModeX/Lx`. A nonzero `WeibelSeedGrowthRate` (gamma) adds the electric field of the growing
mode from Faraday's law, `Ey = -(gamma/k)*WeibelSeedAmplitude*sin(k*x)`. The anisotropy is that of the deck
thermal speeds: in `weibel2D.inp` the electron thermal speed along y is twice that along x and z, and the ions
are cold.

Admission: `WeibelSeedModeX >= 0`, positive for a nonzero `WeibelSeedAmplitude`; finite amplitude and growth
rate.

## Whistler

A parallel whistler on `B0 = B0x x`. `WhistlerSeedMode` selects the seed; `dB = WhistlerSeedEpsilon*B0x` and
`k = 2*pi*WhistlerModeX/Lx`:

| `WhistlerSeedMode` | B | E | Electron drift |
| --- | --- | --- | --- |
| `eigenmode` (default) | `By = dB*cos(phase)`, `Bz = -dB*sin(phase)` | `Ey = -(omega/(c*k))*dB*sin(phase)`, `Ez = -(omega/(c*k))*dB*cos(phase)` | the wave current of Ampere's law, `dB*(c*k - omega^2/(c*k))`, over the species `rhoINIT`, added after the Maxwellian |
| `magnetic_only` | as `eigenmode`, without the time phase | zero | none |
| `noise` | modes 1 to N in `By` and `Bz`, each `dB/sqrt(N)` with its own phase per component | zero | none |

The phase is `k*x + WhistlerPhase - omega*WhistlerFieldTimeOffset` for the fields and uses
`WhistlerParticleTimeOffset` for the drift. `omega` is the cold-plasma root
`c^2 k^2/omega^2 = 1 - mu/(omega*(omega - mu*B0x)) - 1/(omega*(omega + B0x))` (`mu` the ratio of the electron
and ion `|qom|` of the first species of each sign) with `WhistlerUseExactColdRoot = true` (default), or the
approximation `B0x*c^2 k^2/(1 + c^2 k^2/mu)` otherwise. The root is resolved even for a zero seed amplitude.
The noise phases are a fixed function of `WhistlerNoiseSeed`, the component and the mode; N is
`WhistlerNoiseModeMax`, or `WhistlerModeX` when that is zero, or 16 when both are zero.

The eigenmode frequency assumes one `qom` per charge sign, ion `qom = 1` and `rhoINIT` summing to 1 over the
electrons and over the ions; with a nonzero seed amplitude, other values print a warning, and the seed is then
not an exact eigenmode.

Admission:

- `B0x > 0`, `B0y = B0z = 0`, `WhistlerModeY = WhistlerModeZ = 0` and `WhistlerSeedEpsilon >= 0`.
- `eigenmode` and `magnetic_only` need `WhistlerModeX > 0`; `noise` needs `WhistlerNoiseModeMax >= 0`, and a
  nonzero seed amplitude admits at most 65536 noise modes.
- `eigenmode` needs an electron and an ion species and a positive `rhoINIT` for every electron species.
- Unsupported request: an eigenmode with a nonzero wave current and more than one electron species.

## TaylorGreenVortex

A kinetic Taylor-Green vortex. Every species receives the bulk flow `u = U*sin(kx*x)*cos(ky*y)*cos(kz*z)`,
`v = -U*cos(kx*x)*sin(ky*y)*cos(kz*z)` with `U = TaylorGreenVortexVelocityAmplitude`, added after its
Maxwellian. `TaylorGreenVortexMagneticAmplitude` (`Bm`) seeds
`Bx = Bm*sin(kx*x)*cos(ky*y)*cos(kz*z)`, `By = -Bm*cos(kx*x)*sin(ky*y)*cos(kz*z)`; the drift that carries its
current `c*curl(B)` is shared between the electron and the ion species as `-mu/(mu + 1)` and `1/(mu + 1)`
(`mu` the ratio of their `|qom|`), each over its `rhoINIT`. `TaylorGreenVortexSeedMotionalE` sets
`E = -u x B`, with the uniform field included. The wavenumbers are `2*pi*mode/L` per axis;
`TaylorGreenVortexMode` is the default of `TaylorGreenVortexModeX`, `ModeY` and `ModeZ`.
`TaylorGreenVortex3D.inp` is a pure flow vortex on `B0 = B0z z`.

Admission:

- `rhoINIT > 0` for every species; modes non-negative.
- With a nonzero velocity or magnetic amplitude: the three modes positive and `ModeX/Lx = ModeY/Ly`.
- A nonzero magnetic amplitude requires an electron and an ion species; more than one species of either sign is
  an unsupported request.

## TwoStream

Two counter-streaming electron beams on an ion background, in a box periodic in x. With a nonzero
`TwoStreamSeedDensityAmplitude` (`A`), both electron species receive the density
`n_e*(1 + A*cos(k*x + TwoStreamSeedPhase))`, `k = 2*pi*TwoStreamSeedModeX/Lx`, by the particle charge at a
fixed particle count, and Gauss's law gives the matching field `Ex = -(n_e*A/k)*sin(k*x + TwoStreamSeedPhase)`,
where `n_e` is the sum of the electron `rhoINIT`. The ions stay uniform. In `twoStream1D.inp` the box is one
electron-scale wavelength, so mode `m` has `k*d_e = m`; the header gives the cold growth rate of the seeded
mode.

Admission:

- Exactly three species: two electron species with equal `rhoINIT`, `qom` and thermal speeds and equal and
  opposite nonzero `u0`, and one ion species with zero `u0` whose `rhoINIT` equals the electron sum; every
  `rhoINIT` positive.
- All `v0` and `w0` zero; all species with the same `npcelx`, `npcely`, `npcelz`.
- x periodic for fields and particles.
- `TwoStreamSeedModeX >= 0` with `2*mode < nxc`, positive for a nonzero amplitude;
  `|TwoStreamSeedDensityAmplitude| < 1`.

## fluxRope

A screw-pinch flux rope along z, centred at `(FluxRopeCenterX, FluxRopeCenterY)` (default the box centre), with
`r^2 = (x - xc)^2 + (y - yc)^2` and `w = FluxRopeWidth`:
`Bx = -FluxRopeB0*(y - yc)/(r^2 + w^2)`, `By = FluxRopeB0*(x - xc)/(r^2 + w^2)`, `Bz = FluxRopeB0`. The density
is uniform. The pressure `p = FluxRopePressureFloor + FluxRopeB0^2 w^2/(2*(r^2 + w^2)^2)` is split between the
electrons (`FluxRopeElectronPressureFraction`) and the ions (the rest); each species takes the isotropic
thermal speed `sqrt(T*|qom|)` of its share `T = fraction*p/n`, `n` the `rhoINIT` sum of its charge sign, in
place of the deck thermal speeds. The electrons carry the axial current `2*FluxRopeB0*w^2/(r^2 + w^2)^2` as a
z drift.

Admission: `FluxRopeWidth > 0`, `FluxRopePressureFloor >= 0`, `FluxRopeElectronPressureFraction` in `[0, 1]`,
a positive total `rhoINIT` for the electron and for the ion species, and fields, pressures and drifts that stay
finite over the box and one halo cell.

## GaussianExpansion

A neutral Gaussian plasma released into vacuum: every species carries the weight
`exp(-sum((x_a - c_a)^2/(2*sigma_a^2)))` over the axes with a positive `GaussianExpansionSigma*` and is uniform
along the others; E and B start at zero, plus the uniform `B0`. A positive `GaussianExpansionCutoff` loads only
the subcells whose centre lies inside `sum(((x_a - c_a)/sigma_a)^2) <= Cutoff^2`, and the vacuum holds no
particle. The setup, the self-similar solution, the runs and the checks are in
[Gaussian expansion](gaussian_expansion.md).

Admission: every width non-negative and one positive; `GaussianExpansionCutoff >= 0`; the centre on a grid node
along every axis with a positive width; `QuietStart = true`, one `npcelx`, `npcely`, `npcelz` for all species, a
vanishing sum of `sign(qom)*rhoINIT`; every `rhoINIT` positive and every drift zero.

## WhistlerTurbulence

Decaying whistler turbulence: 150 helical whistlers of equal magnetic energy on `B0 = B0z z`, with
`<|dB|^2>/B0z^2 = WhistlerTurbulenceEpsilon` and phases fixed by `WhistlerTurbulenceSeed`; E from Faraday's
law, the electrons carry the current of Ampere's law and the ions rest. The seed, the units, the runs and the
analysis are in [whistler turbulence](whistler_turbulence.md).

Admission: `WhistlerTurbulenceEpsilon >= 0`; `B0x = B0y = 0` and `B0z > 0`; fields and particles periodic on
every axis; `QuietStart = true`, one `npcelx`, `npcely`, `npcelz` for all species, a vanishing sum of
`sign(qom)*rhoINIT`; one `qom` for all electron species and a positive electron `rhoINIT` sum.
