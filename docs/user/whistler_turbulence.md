# Decaying whistler turbulence

`Case = WhistlerTurbulence` follows freely decaying whistler turbulence in a homogeneous, periodic, collisionless
electron-proton plasma on a uniform field `B0 = B0z z`. At t = 0 a spectrum of 150 long-wavelength whistlers is
superposed on the plasma; nothing drives it afterwards. The run follows the forward cascade to k d_e ~ 1-4, the
growth of quasi-perpendicular anisotropy (k⊥ ≫ k∥), the spectral break at k⊥ d_e ≈ 1 of [G12] and the electron
heating, mostly parallel, by the Landau resonance. The one parameter of the ensemble is the initial amplitude

ε0 = ⟨|δB(t = 0)|²⟩/B0².

| Item | Where |
| --- | --- |
| Deck | [`input/whistlerTurbulence3D.inp`](../../input/whistlerTurbulence3D.inp) |
| Seed | [`whistler_turbulence.hpp`](../../include/tpic/initialization/cases/whistler_turbulence.hpp), [`whistler_turbulence.cpp`](../../src/initialization/cases/whistler_turbulence.cpp) |
| Seed check | `whistler_turbulence()` in [`tests/initialization/recipe_tests.cpp`](../../tests/initialization/recipe_tests.cpp) |
| Analyzer | [`tools/postprocessing/analyze_whistler_turbulence.py`](../../tools/postprocessing/analyze_whistler_turbulence.py), checked by [`tests/support/test_whistler_turbulence.py`](../../tests/support/test_whistler_turbulence.py) |
| Admission rules, keys | [cases](cases.md#whistlerturbulence), [input keys](input_keys.md) |

The run, step by step: set up the deck (Section 2), check the seed at step 0 (Section 3.8), run the validation
runs V0-V4 (Section 4), run the ε0 ensemble (Section 2.5), analyse it (Section 5) and compare with the papers
(Section 6).

## 1. Units and plasma parameters

The deck uses electron units: c = 1, ω_pe = 1, d_e = c/ω_pe = 1 (`c = 1`, electron `qom = -1`, `rhoINIT = 1`),
times in 1/ω_pe, velocities in c. Fields are given over B0 = `B0z` = |Ω_e|/ω_pe = 1/√5, so a field of the
deck is the value of this document times `B0z`. Energy densities are in units of B0²/8π. With these units
ω_pe² is the sum over the electron species of `rhoINIT*|qom|`, d_e = c/ω_pe and |Ω_e| = |qom_e| B0z/c; the
analyzer derives all three from the deck.

| Quantity | Value | Comment |
| --- | --- | --- |
| m_p/m_e | 1836 | `qom = -1 5.446623093681918e-4` |
| ω_pe/\|Ω_e\| | √5 = 2.236 | \|Ω_e\|/ω_pe = 0.4472 = `B0z` |
| β_e = β_p | 0.1 | β_e = 2 (v_te/c)² (ω_pe/Ω_e)² |
| T_p/T_e | 1 | |
| v_te/c | 0.1 | `uth = vth = wth`: the standard deviation of each velocity component, √(k_B T/m), no √2 |
| v_tp/c | 2.334×10⁻³ | 0.1 √(1/1836) |
| B0 | B0 z | z is the parallel direction (∥) |

Derived scales:

| Scale | Value | Comment |
| --- | --- | --- |
| λ_De | 0.1 d_e | half a cell at Δx = 0.2 d_e |
| ρ_e = v_te/\|Ω_e\| | 0.224 d_e | |
| d_i | 42.85 d_e | the box, 60 d_e, is 1.40 d_i |
| ρ_p | 9.58 d_e | |
| v_A/c | 0.01044 | Ω_p/ω_pp |
| largest whistler phase speed (θ = 0, k d_e = 1) | 0.224 c = 2.24 v_te | \|Ω_e\| d_e/2 |
| Ω_p | 2.44×10⁻⁴ ω_pe | Ω_p t = 0.24 at ω_pe t = 1000: the ions are nearly a static background |
| ω_LH | 0.0213 \|Ω_e\| = 0.0095 ω_pe | 18 of the 150 seeded modes lie below it (Section 3.2) |
| 2π/\|Ω_e\| | 14.05/ω_pe | electron gyroperiod |

Times convert as ω_pe t = √5 |Ω_e| t:

| ω_pe t | 50 | 130 | 300 | 1000 | 4497 |
| --- | --- | --- | --- | --- | --- |
| \|Ω_e\| t | 22.36 | 58.14 | 134.16 | 447.21 | 2011 |
| Used in | [G12] Fig. 3 | [G12] Fig. 3 | [G12] Figs. 3-4 | [G12] Figs. 3, 5 (end of run) | [S08] end of run |

The [G12] Fig. 3 labels 22.36, 58.14, 134.16 and 447.21 are ω_pe t = 50, 130, 300 and 1000 exactly, which
confirms ω_pe/|Ω_e| = √5.

## 2. The deck

### 2.1 Box and grid

| Item | Value | Comment |
| --- | --- | --- |
| Box | `Lx = Ly = Lz = 60` d_e | [G12] uses 51.2 d_e; the seed modes are multiples of k0 = 2π/L = 0.1047/d_e |
| Cells | `nxc = nyc = nzc = 300` | Δx = 0.2 d_e = 2 λ_De |
| Grid Nyquist | k d_e = π/Δx = 15.7 | |
| Cells per wavelength at k d_e = 1, 2, 4 | 31.4, 15.7, 7.9 | [G12] analyses up to k d_e = 4 |
| Boundaries | periodic fields and particles on every axis | the case requires it |
| Ranks | `XLEN = YLEN = 4`, `ZLEN = 2` | 32 ranks of 75 × 75 × 150 cells |

Δx = 0.2 d_e resolves the analysed range k d_e ≤ 4 with at least 7.9 cells per wavelength. [G12] uses
Δx = 0.1 d_e (512³ cells in its 51.2 d_e box), as an explicit scheme needs cells near the Debye length; the
implicit scheme runs at Δx = 2 λ_De, and V3 (Section 4) checks the spectrum against Δx = 0.1 d_e. Spectra are
binned in units of k0, the spacing of the seeded wavevectors.

### 2.2 Time step

`dt = 0.25` (Δt |Ω_e| = 0.112), `ncycles = 4000` to ω_pe t = 1000.

| Constraint | Value at Δt = 0.25 | Requirement |
| --- | --- | --- |
| Steps per electron gyroperiod | 56 | ≳ 20: gyromotion and the cyclotron resonance are resolved |
| ωΔt of the fastest seeded mode (ω = 0.0578 ω_pe) | 0.0145 | ≪ 1 |
| ωΔt of the whistler at k d_e = 4, θ = 0 (ω ≈ 0.42 ω_pe) | 0.105 | ≪ 1 |
| Thermal cell crossing v_te Δt/Δx | 0.125 | |
| Landau-resonant electrons, v∥ = ω/k∥ ≤ 0.224 c | ≤ 0.28 cells per step | < 1 |
| Cyclotron-resonant electrons at k∥ d_e = 0.85, v∥ = (ω - \|Ω_e\|)/k∥ ≈ -0.31 c | 0.38 cells per step | < 1 |
| Suprathermal electrons at 5 v_te | 0.63 cells per step | < 1 |
| cΔt/Δx | 1.25 | admitted by the implicit scheme; light waves play no part here |
| ω_pe Δt | 0.25 | plasma oscillations are resolved, which keeps the heating diagnostics clean |

The electron cyclotron physics, not stability, limits the step.

### 2.3 Implicit settings and smoothing

- **θ = 0.5** (`th = 0.5`). A θ-scheme damps a mode of frequency ω at about γ_num ≈ (θ - 1/2) ω² Δt. At
  θ = 0.6, ω = 0.2 ω_pe and Δt = 0.25 that is 1×10⁻³ ω_pe = 2.2×10⁻³ |Ω_e|, the order of the physical Landau
  damping (|γ| ~ 0.002-0.005 |Ω_e|) that sets the heating and the spectral shape. At θ = 0.5 ECSIM [L17]
  conserves energy exactly and adds no numerical damping.
- **GMRES** (`GMREStol = 1E-15`). The energy conservation of ECSIM is limited by the solver tolerance; electron
  heating is a primary observable, so the deck asks for a tolerance far below 10⁻⁸.
- **Smoothing** (`ECSimSmoothing = energy_conserving`, `ECSimSmoothingOperator = face7`, `Smooth = 0.5`,
  `SmoothNiter = 1`). One energy-conserving face7 pass acts on the solver current and on the field the particles
  see ([inputs](inputs.md) describes the operators and when they are active). A face7 pass gives
  `Smooth*f + (1-Smooth)/6` times the sum of the six face neighbours, so it scales a Fourier mode by

  T(k) = 1 - (1 - S)/3 Σ_a (1 - cos k_a Δx), summed over a = x, y, z;

  along one axis 1 - (1 - S)/3 (1 - cos kΔx). The alternative `binomial27` applies fixed [1,2,1]³/64 weights,
  T = Π_a cos²(k_a Δx/2). At Δx = 0.2 d_e, along an axis, per pass:

  | k d_e | 1 | 2 | 3 | 4 | 15.7 (kΔx = π) |
  | --- | --- | --- | --- | --- | --- |
  | face7, S = 0.5: T | 0.997 | 0.987 | 0.971 | 0.949 | 0.667 |
  | face7, S = 0.5: T² | 0.993 | 0.974 | 0.943 | 0.901 | 0.444 |
  | binomial27: T | 0.990 | 0.961 | 0.913 | 0.848 | 0 |
  | binomial27: T² | 0.980 | 0.923 | 0.833 | 0.720 | 0 |

  The pass removes the grid checkerboard and leaves the analysed range nearly untouched. Stronger smoothing,
  applied every step, changes the dispersion and the damping at k d_e ≳ 2 and steepens the spectrum above the
  break, the observable of [G12]: six face7 passes of `Smooth = 0.2` would scale the magnetic energy at
  k⊥ d_e = 3 and 4 by 0.6 and 0.4. V1 and V3 (Section 4) test the smoothing at k d_e = 2 and 4.
- **Particle shape.** TPIC deposits and interpolates with the linear (CIC) shape.

### 2.4 Particles per cell and memory

The noise floor of the magnetic spectrum scales about as one over the particles per unit volume. [G12] uses
64 particles per cell and species at Δx = 0.1 d_e, 6.4×10⁴ per d_e³. The deck loads `npcelx = npcely = 8`,
`npcelz = 4`: 256 per cell and species, 3.2×10⁴ per d_e³, half the [G12] density, 1.38×10¹⁰ particles in all.

| Per cell and species | `npcelx npcely npcelz` | Per d_e³ | Noise against [G12] | Particles (2 species) | Ranks (64 GiB GCD) | Use |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 8 8 2 | 1.6×10⁴ | 4× higher | 6.9×10⁹ | 16 | ε0 ≥ 0.2, tests |
| 256 | 8 8 4 | 3.2×10⁴ | 2× higher | 1.38×10¹⁰ | 32 | the deck |
| 512 | 8 8 8 | 6.4×10⁴ | equal | 2.76×10¹⁰ | 64 (4 × 4 × 4) | ε0 ≤ 0.05, noise-limited runs |

Each row holds 4.3×10⁸ particles per rank. The startup memory plan is then 56 GiB per rank of a 64 GiB GCD:
45 GiB of device owners, 41 of them for the particles, and an 11 GiB transport allowance. In [G12] the ε0 = 0.02
spectrum reaches the noise near k⊥ d_e ≈ 2, so particles matter most at small ε0.

`QuietStart = true` places every particle at a subcell centre and the electron and the proton of a pair on the
same point, so ρ = 0 at t = 0 ([inputs](inputs.md) describes the loading). The lattice lowers the initial noise
below that of random positions, but phase mixing restores thermal noise within tens of 1/ω_pe; the late-time
noise is that of the V0 control run, not that of the start.

### 2.5 The ε0 ensemble and seeds

`WhistlerTurbulenceEpsilon` sets ε0; the deck runs 0.1.

| ε0 | Purpose | Length (ω_pe t) |
| --- | --- | --- |
| 0 | control run V0: noise floor, numerical heating, f(v) stability | 1000 |
| 0.02, 0.05 | weak, linear-damping-dominated regime | 1000 |
| 0.1 | main run (comparable to [S08] Run I and [G12]); the final checkpoint continues it to ≈ 4500 (\|Ω_e\| t ≈ 2000) | 1000 (4500) |
| 0.2 | strongest anisotropy in [G12]; the V3 resolution test | 1000 |
| 0.5, 1.0 | nonlinear regime; the spectra approach an amplitude-independent limit | 1000 |

`WhistlerTurbulenceSeed` selects the phase realization: every mode takes a phase uniform in [0, 2π) from a hash of
the seed and the mode index. The papers do not give their phases; 2-3 seeds per ε0 measure the spread of slopes
and anisotropy. The seed changes the wave phases only: the Maxwellian draw depends on the rank and the species
([inputs](inputs.md)).

### 2.6 Output

| Output | Deck keys | Cadence | Used in |
| --- | --- | --- | --- |
| `diagnostics.csv`: magnetic, electric, per-species kinetic and total energy | `DiagnosticsOutputCycle = 4` | every 1/ω_pe | 5.1 |
| Fields `B` and `E` | `FieldOutputTag = B+E`, `FieldOutputCycle = 100` | every 25/ω_pe, 41 snapshots, including ω_pe t = 0, 50, 300 and 1000 | 5.4-5.8, 5.10 |
| Electron moments `rho`, `J`, `PXX`, `PYY`, `PZZ` of species 0 | `MomentsOutputTag = rho0+J0+PXX0+PYY0+PZZ0` | with the fields | 5.3 |
| Gauss-law residual, `gauss.csv` | `DiagnosticGaussCycle = 100` | every 25/ω_pe | |
| Console energy | `DiagnosticEnergyCycle = 50` | every 12.5/ω_pe | |
| Particle snapshots: position, velocity, charge and id of every particle | `ParticlesOutputCycle = 400` | every 100/ω_pe, 11 snapshots | 5.9 |
| Checkpoints | `RestartOutputCycle = 1000`, `CallFinalize = 1` | every 250/ω_pe and at the end | continuing a run |

`WriteMethod = pvtk`, the format the analyzer reads. The 13 Float64 node arrays of one field output are 2.8 GB,
about 115 GB per run; a particle snapshot of 1.38×10¹⁰ particles is about 0.9 TB. `FieldOutputCycle = 40`
(every 10/ω_pe, 101 snapshots) adds ω_pe t = 130 of [G12] Fig. 3, gives the 5-snapshot slope average of
Section 5.5 and raises the ω-k Nyquist frequency (Section 5.10), at 2.5 times the field volume. To continue the
ε0 = 0.1 run to ω_pe t = 4500, raise `ncycles` to 18000 and run with `--restart`; checkpoints need a build with
ADIOS2 or HDF5 ([output](output.md) describes the files and checkpoints, [build and run](build_and_run.md) the
command line).

## 3. The seed

TPIC builds the seed from the mode set of [G12] and [C11] and the construction of each mode of [S08]. The fields
are sampled in double precision at t = 0 and the electron flow is evaluated at every particle position.

### 3.1 Mode set: 150 modes

The wavevectors are k = 2π (n_x/L_x, n_y/L_y, n_z/L_z) with n_z ≠ 0 and |n_x|, |n_y|, |n_z| ≤ 3:

| Group | Integer indices | Count |
| --- | --- | --- |
| y-z plane | (0, n_y, n_z), n_y ∈ {0, ±1, ±2, ±3}, n_z ∈ {±1, ±2, ±3} | 42 |
| x-z plane | (n_x, 0, n_z), n_x ∈ {±1, ±2, ±3}, n_z ∈ {±1, ±2, ±3} | 36 |
| Diagonals | (n, ±n, n_z), n ∈ {±1, ±2, ±3}, n_z ∈ {±1, ±2, ±3} | 72 |
| Total | 150 distinct wavevectors | 150 |

[G12] describes the diagonals as the x-z modes rotated by 45° and 135°; a literal rotation of (n_x, 0, n_z) is
no lattice vector of a periodic box, so the lattice diagonals (n, ±n, n_z), with k⊥ = √2 |n| k0, take their
place. `WhistlerTurbulenceModes::for_each_index` lists them; the analyzer rebuilds the same list.

Properties of the seed in the 60 d_e box:

- |k| d_e ∈ [0.105, 0.544], |k∥| d_e ≤ 0.314, k⊥ d_e ≤ 0.444 (the [G12] box: 0.123-0.638, 0.368, 0.521).
- The perpendicular directions are φ = 0°, 45°, 90° and 135° only, so the seed is not gyrotropic (index
  G ≈ √2, Section 5.7).
- Every mode has its -k partner: counter-propagating waves on the same |k| line.
- With equal energy per mode tan²θ_B = 1.44 (y-z group 0.86, x-z 1.00, diagonals 2.00) in any cubic box. [G12]
  quotes "≈ 1.3". An isotropic 3D spectrum has tan²θ_B = 2 (Section 5.6).

### 3.2 Frequencies

Every mode is a whistler with ω > 0 and phase factor exp[i(k·x - ωt)]: k∥ > 0 travels along +B0, k∥ < 0 along
-B0. TPIC takes ω from the full cold-plasma dispersion relation of every species of the deck (Stix S, D, P with
ω_ps² = `rhoINIT*|qom|`, signed Ω_s = `qom*B0z/c`, displacement current included): the root of the
magnetosonic-whistler branch nearest in log to the electron-MHD estimate ω = |Ω_e| |k∥| k d_e²/(1 + k² d_e²).
The startup report prints the range of ω/(ck): 0.0392 to 0.1266 for the deck.

Cold-plasma frequencies of the 21 distinct (|n∥|, n⊥²) classes in the 60 d_e box (rows with n⊥² > 0 hold
8 modes, both signs of k∥ and 4 azimuths; rows with n⊥² = 0 hold 2):

| \|n∥\| | n⊥² | k d_e | θ (deg) | ω/\|Ω_e\| cold | ω/\|Ω_e\| electron MHD | v_ph/c | C∥ = sin²θ/2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0.105 | 0.0 | 0.0113 | 0.0108 | 0.048 | 0 |
| 1 | 1 | 0.148 | 45.0 | 0.0157 | 0.0152 | 0.047 | 0.250 |
| 1 | 2 | 0.181 | 54.7 | 0.0189 | 0.0184 | 0.047 | 0.333 |
| 1 | 4 | 0.234 | 63.4 | 0.0238 | 0.0232 | 0.045 | 0.400 |
| 1 | 8 | 0.314 | 70.5 | 0.0305 | 0.0299 | 0.043 | 0.444 |
| 1 | 9 | 0.331 | 71.6 | 0.0318 | 0.0313 | 0.043 | 0.450 |
| 1 | 18 | 0.456 | 76.7 | 0.0400 | 0.0396 | 0.039 | 0.474 |
| 2 | 0 | 0.209 | 0.0 | 0.0422 | 0.0420 | 0.090 | 0 |
| 2 | 1 | 0.234 | 26.6 | 0.0466 | 0.0465 | 0.089 | 0.100 |
| 2 | 2 | 0.257 | 35.3 | 0.0504 | 0.0504 | 0.088 | 0.167 |
| 2 | 4 | 0.296 | 45.0 | 0.0569 | 0.0570 | 0.086 | 0.250 |
| 2 | 8 | 0.363 | 54.7 | 0.0668 | 0.0671 | 0.082 | 0.333 |
| 2 | 9 | 0.378 | 56.3 | 0.0688 | 0.0692 | 0.081 | 0.346 |
| 2 | 18 | 0.491 | 64.8 | 0.0819 | 0.0829 | 0.075 | 0.409 |
| 3 | 0 | 0.314 | 0.0 | 0.0890 | 0.0898 | 0.127 | 0 |
| 3 | 1 | 0.331 | 18.4 | 0.0928 | 0.0938 | 0.125 | 0.050 |
| 3 | 2 | 0.347 | 25.2 | 0.0963 | 0.0974 | 0.124 | 0.091 |
| 3 | 4 | 0.378 | 33.7 | 0.1025 | 0.1038 | 0.121 | 0.154 |
| 3 | 8 | 0.432 | 43.3 | 0.1126 | 0.1143 | 0.117 | 0.235 |
| 3 | 9 | 0.444 | 45.0 | 0.1148 | 0.1166 | 0.116 | 0.250 |
| 3 | 18 | 0.544 | 54.7 | 0.1293 | 0.1319 | 0.106 | 0.333 |

The range is ω/|Ω_e| = 0.0113-0.129 (ω/ω_pe = 0.0051-0.058). The 18 modes with |n∥| = 1 and n⊥² ≤ 2 lie
below ω_LH = 0.0213 |Ω_e|, where the proton response matters (|u_i|/|u_e| ≈ 0.045 for the parallel mode at
k d_e = 0.105); there the cold frequency, which holds the ions, lies up to 4.5 % above the electron-MHD value.

### 3.3 Fields and electron flow

Per mode n, with k̂ = k/k,

e1 = ẑ × k̂/|ẑ × k̂| (e1 = x̂ if k ∥ ẑ), e2 = k̂ × e1, σ = sign(k∥), ψ = k·x + φ_n.

**Magnetic field**, circularly polarized in the plane perpendicular to k, right-handed about B0:

δB_n = A [e1 cos ψ - σ e2 sin ψ], |δB_n|² = A² everywhere.

In complex form b̂ = (e1 + iσ e2)/√2 with k̂ × b̂ = -iσ b̂: a helical (Beltrami) field, ∇ × δB_n = σk δB_n. At a
fixed point both the n_z = +1 and the n_z = -1 parallel modes rotate counter-clockwise about +z, the electron
gyration sense of a whistler. The cold-plasma eigenvectors of all 150 modes overlap with this helical vector at
≥ 0.99997, and their out-of-plane fraction r_B is 0.492-0.506 (helical value 1/2): ion and thermal corrections
to the magnetic polarization are negligible at t = 0.

**Electric field** from Faraday's law, transverse only:

δE_n = -(ω/ck) k̂ × δB_n = -(ω/ck) A [e2 cos ψ + σ e1 sin ψ].

**Electron flow** from Ampère's law with the displacement current and the ions at rest:

δu_e,n/c = -σ (|Ω_e|/ω_pe) (k d_e) (1 - ω²/(k²c²)) δB_n/B0.

The flow is parallel or antiparallel to δB (a force-free current, J ∥ δB); for k∥ > 0 it is antiparallel. In
the code the flow is -c A σk (1 - v_ph²/c²) b/n_e, with n_e the sum of the electron `rhoINIT`; every electron
species, all of one `qom`, carries the same flow, and the ions carry none.

**Amplitude.** Every mode has the same magnetic energy, A = B0z √(ε0/150) (the startup report prints it:
0.011547 at ε0 = 0.1). Then ⟨|δB|²⟩/B0² = Σ A²/B0² = ε0 exactly, for any phases: distinct wavevectors are
orthogonal, and the two members of a ±k pair carry the opposite-helicity vectors b̂ and b̂*, whose cross term
vanishes as b̂·b̂ = 0.

`tests/initialization/recipe_tests.cpp` checks these formulas in the [G12] box (L = 51.2 d_e, ε0 = 0.1)
against an independent NumPy evaluation with the same phases: B, E and the flow at one point, and on 8³
points the energy densities W_B = 0.1, W_E = 9.588×10⁻⁴ and K_flow,e = 0.016785 ([G12] box values).

### 3.4 Particle loading

1. `QuietStart = true` with equal `npcel*` for all species: an electron and a proton at every subcell centre,
   so ρ = 0, consistent with the transverse E (∇·E = 0).
2. Electron velocities: a Maxwellian of standard deviation `uth`, `vth`, `wth` plus δu_e, summed over the 150
   modes at the particle position.
3. Proton velocities: a Maxwellian of standard deviation v_tp, no flow.
4. `CenterInitialVelocity = true true` shifts each species' loaded velocities so that their mean is that of the
   flow the case loads, zero for the waves: it removes the momentum of the Maxwellian sample and keeps the wave
   flow.

The rms electron flow is 0.22, 0.35, 0.50, 0.70, 1.11 and 1.57 v_te for ε0 = 0.02, 0.05, 0.1, 0.2, 0.5 and 1.0;
at ε0 ≥ 0.5 the bulk flow exceeds the thermal speed (Section 3.7 lists what this does to the temperatures).

### 3.5 What the seed leaves out

For oblique whistlers the electron Lorentz force δu_e × B0 has a component along k. In a real eigenmode a
longitudinal electric field E_L and a small charge separation balance it; the seed, as [S08], leaves out the
electrostatic and density parts ("the system adjusts within a few electron plasma periods"). In the 60 d_e box,
from the cold electron-proton eigenvectors:

- E_L carries 0.0073 ε0, 0.7 % of the magnetic energy (for oblique modes |E_L|² is 0.07 to 13 times |E_T|²);
- the proton fluid flow carries 0.016 ε0;
- the eigenmode density perturbation is rms δn_e/n0 ≈ 0.017 √ε0.

A small electrostatic transient at ω_pe in the first 10-50/ω_pe and a small electron energy offset follow; V2
(Section 4) measures them.

### 3.6 Fields on the grid

- **Staggering.** E is sampled at the nodes and B, with B0 added, at the cell centres, each component at its own
  location.
- **∇·B.** The analytic field is divergence-free; sampled at the cell centres its discrete divergence is of
  relative order (kΔx)²/24 ≤ 5×10⁻⁴ for the seeded modes (kΔx ≤ 0.109).
- **Time.** All fields and the flow are taken at t = 0; ωΔt ≤ 0.0145 for every seeded mode.
- **Initial E.** The implicit solver advances E and B together and needs E(t = 0). The seed loads the Faraday E;
  with E = 0 the electron current would contradict Ampère's law at t = 0 and launch a transient.

### 3.7 Initial energy budget

Phase-averaged, in units of B0²/8π, in the 60 d_e box. The thermal energy per degree of freedom and species is
β/2 = 0.05, so a flow energy K adds K/0.05 to T∥/T0 (one degree of freedom) and K/0.1 to T⊥/T0 (two).

| ε0 | W_B | W_E | K_flow,e | K_flow ∥ | K_flow ⊥ | apparent ΔT∥/T0 | apparent ΔT⊥/T0 | rms \|δu_e\|/v_te |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 0.02 | 1.5×10⁻⁴ | 0.0025 | 0.0007 | 0.0017 | 0.015 | 0.017 | 0.22 |
| 0.05 | 0.05 | 3.9×10⁻⁴ | 0.0061 | 0.0018 | 0.0043 | 0.036 | 0.043 | 0.35 |
| 0.1 | 0.1 | 7.7×10⁻⁴ | 0.0123 | 0.0036 | 0.0086 | 0.073 | 0.086 | 0.50 |
| 0.2 | 0.2 | 1.5×10⁻³ | 0.0245 | 0.0073 | 0.0173 | 0.145 | 0.173 | 0.70 |
| 0.5 | 0.5 | 3.9×10⁻³ | 0.0614 | 0.0182 | 0.0432 | 0.363 | 0.432 | 1.11 |
| 1.0 | 1.0 | 7.7×10⁻³ | 0.123 | 0.0363 | 0.0864 | 0.727 | 0.864 | 1.57 |

K_flow,e = Σ k² d_e² (1 - v_ph²/c²)² A²/B0² ≈ ⟨k² d_e²⟩ W_B = 0.123 ε0: the electron kinetic energy of a whistler
is k² d_e² times its magnetic energy. A "temperature" from the total kinetic energy, as in the papers, therefore
starts too high by 0.73 ε0 (∥) and 0.86 ε0 (⊥), and the excess decays with the waves (Section 5.3 reports both
definitions). The ∥/⊥ split is the phase average: a ±k pair forms a partly standing wave whose component split
depends on the relative phase, so one realization differs by a few percent.

### 3.8 Step-0 checks

The analyzer prints the first block of these checks as "Step-0 checks (Section 3.8)" into its summary; the
startup report gives the mode count, the amplitude and the range of ω/(ck).

| Check | Expected (the deck, ε0 = 0.1) |
| --- | --- |
| ⟨\|δB\|²⟩/B0² of the spectrum and of `diagnostics.csv` | ε0 |
| Unseeded fraction of W_B | round-off: the lattice start holds energy in the seeded modes alone |
| tan²θ_B over all k ≤ k_s | 1.44 (equal energy per seeded mode) |
| Gyrotropy index G in the seeded shell | 1.43 (the seed fills 4 of the 12 azimuth bins; gyrotropic: 0) |
| Helicity σ_m in the seeded shell | 0 to round-off (balanced ±k pairs) |
| Papers' T∥/T_th, T⊥/T_th (T_th = m v_te²) | about 1.073 and 1.086 (Section 3.7), within a few percent |
| Thermal T∥/T_th, T⊥/T_th | about 1 |
| Startup report | 150 modes, amplitude 0.011547, ω/(ck) from 0.0392 to 0.1266 |
| ∇·E and ρ | both about 0: transverse E, pair loading (`gauss.csv`) |
| Total energy | the first row of `diagnostics.csv`, the start of the conservation check |

## 4. Validation runs V0-V4

These runs show that the coarse grid (Δx = 2 λ_De), the implicit step and the smoothing leave the physics
unchanged. Each one is a change of `input/whistlerTurbulence3D.inp`.

### V0: control run

`WhistlerTurbulenceEpsilon = 0`, everything else as the deck, to ω_pe t = 1000.

- **Noise spectrum.** Pass its output directory to the analyzer of every other run as `--noise-dir`: the
  spectra are matched by cycle, so the control needs the same `FieldOutputCycle`. The noise grows from t = 0 and
  saturates; it sets k_s (Section 5.5).
- **Numerical heating.** ΔT∥/T0 and ΔT⊥/T0 at ω_pe t = 1000 well below the smallest heating to be measured
  (≲ 10⁻³-10⁻² for ε0 = 0.02).
- **Energy conservation.** Total-energy drift ≪ 0.02 B0²/8π, the smallest wave energy of the ensemble.
- **f(v) shape.** No artificial tails or core flattening in f(v∥) and f(v_x) between t = 0 and 1000 on a log
  scale (Section 5.9).

### V1: linear whistlers

At small ε0 the 150 modes are independent linear waves, so one run measures the dispersion and the damping of
21 or more (k, θ) classes. The box sets them: k = 2π n/L per axis.

- **Boxes.** Keep Δx = 0.2 d_e (`nxc = Lx/0.2`), set `XLEN`, `YLEN`, `ZLEN` to fit the smaller grid, and choose
  L_x, L_y, L_z so that the classes cover
  k d_e ∈ {0.3, 0.5, 1, 2, 4} and θ ∈ {0°, 30°, 60°, 75°}: L = 20 d_e (100³ cells) spans k d_e = 0.31-1.63 and
  L = 6 d_e (30³ cells) spans 1.05-5.4, θ = 0°-77° in each. A box with L_z = 7.8 d_e puts k∥ d_e = 0.81 at
  θ = 0, the onset of cyclotron damping. Recompute k and θ from the box.
- **Amplitude.** δB/B0 ≈ 10⁻³-3×10⁻³ per mode, ε0 = 150 (δB/B0)² ≈ 1.5×10⁻⁴-1.4×10⁻³, with ≥ 1000 particles per
  cell (for example `npcelx = npcely = 16`, `npcelz = 4`). Trapping in E∥ ends linear Landau damping after
  t ~ 1/ω_b, (ω_b/ω_pe)² ≈ (|Ω_e|/ω_pe)(k∥ d_e)|δE∥/B0|: keep ω_b well below the expected |γ| and fit before
  1/ω_b. Halving the amplitude must leave γ unchanged.
- **Measurement.** Write `B` often (for example `FieldOutputCycle = 8`, every 2/ω_pe) and track the complex
  Fourier coefficient B_k(t) of each seeded mode: ω from the phase (ω = -d arg B_k/dt), γ from d ln|B_k|/dt
  (|B_k|² decays at 2γ). The ω-k spectra of the analyzer (Section 5.10) show the frequencies.
- **Pass.** Against kinetic linear theory at β_e = 0.1 at the actual (k, θ): ω within 1-2 %, γ within about
  10-20 % where |γ| ≳ 10⁻³ |Ω_e|. Features of [S08]: cyclotron damping (θ = 0) is essentially zero up to
  k∥ d_e ≈ 0.7 and switches on abruptly near 0.8; Landau damping (θ > 0) grows smoothly with k and peaks at
  θ ≈ 50-60°; γ = -0.005 |Ω_e| is reached at k∥ d_e ≈ 0.85 by cyclotron damping and already at k d_e ≈ 0.45 by
  Landau damping at θ = 60°.
- Keep the production `th`, `GMREStol` and smoothing: the test exposes numerical damping (Section 2.3) and the
  filter.

### V2: initialization

The first 50/ω_pe of a production run: `ncycles = 200`, `DiagnosticsOutputCycle = 1` (25 samples per plasma
period).

- The step-0 checks of Section 3.8.
- W_E(t) and its oscillation at ω_pe: the transient of the missing E_L, of order 0.0073 ε0 (Section 3.5), and
  the electron energy offset it leaves.

### V3: resolution and time step

At ε0 = 0.2 to ω_pe t = 300, against the production settings:

- (a) `dt = 0.125`, `ncycles = 2400`, `FieldOutputCycle = 200` (the same output times);
- (b) half the particles per cell (`npcelz = 2`);
- (c) Δx = 0.1 against 0.2 d_e in a common smaller box, `Lx = Ly = Lz = 30` at 150³ and at 300³ cells (the same
  150-mode pattern, k0 doubles), both at `dt = 0.125`, at equal particles per volume (256 and 32 per cell, for
  example `npcel* = 8 8 4` and `4 4 2`). What matters is that the two resolutions agree in one setup.

Compare E(k⊥) up to k⊥ d_e = 4, the break and both slopes, tan²θ_B(t) and T∥/T⊥(t). The runs pass when the
differences lie within the seed spread of V4. A high-k slope that depends on Δx or Δt points first at the
smoothing and at θ.

### V4: seeds

At ε0 = 0.1, `WhistlerTurbulenceSeed = 1, 2, 3` to ω_pe t = 300 (`ncycles = 1200`). Their spread in slopes and
anisotropy is the error bar of every other comparison.

## 5. Analysis

```sh
python3 tools/postprocessing/analyze_whistler_turbulence.py --run-dir RUN [--noise-dir CONTROL/data]
```

The analyzer reads the single `WhistlerTurbulence` deck of `--run-dir` (default the current directory) unless
`--input` names it, and the data in its `SaveDirName` unless `--data-dir` names them: `diagnostics.csv`, the
`B` field files, the `rho`, `J`, `PXX`, `PYY`, `PZZ` files of the first electron species when the deck writes
them, and the particle snapshots. It writes each plot as PNG and PDF into `--output-dir` (default the run
directory) under `--prefix` (default `whistler_turbulence`), and `<prefix>_timeseries.csv` (every quantity at
every field snapshot) and `<prefix>_summary.txt` (the step-0 checks, the slopes, the heating and the ω-k peaks,
also printed). `--only PART ...` draws the parts `energy`, `temperatures`, `spectra`, `anisotropy`,
`polarization`, `omega-k`, `vdf` or `render` alone and reads only their data; the timeseries and the summary
are written only when every part but `vdf` and `render` runs. `--render-3d` (needs pyvista) adds volume
renderings of |∇ × δB| and of the electron and ion density fluctuations at `--render-times` (default 0 300 1000).
`--no-particles` skips the particle snapshots. [Analysis](analysis.md) lists the other analyzers.

### 5.0 Conventions

- **Units.** Times ω_pe t, wavenumbers k d_e, fields over B0 = `B0z`, energy densities over B0²/8π,
  velocities over the electron `uth` v_te.
- **Layout.** δB = B - B0 z on the nodes of the periodic box, the repeated periodic end nodes dropped, as
  (x, y, z) arrays with z (along B0) last. B_k = rfftn(δB)/N, so that Σ W|B_k|² = ⟨|δB|²⟩, with the weight
  W = 2 for k_z > 0 and 1 at k_z = 0 and at Nyquist; k_z = k∥ ≥ 0. The box mean of B_z is not subtracted: in a
  periodic box it is conserved and equal to B0.
- **Node average.** TPIC writes B at a node as the mean of its eight cells. That average is the filter
  cos(k_x Δx/2) cos(k_y Δy/2) cos(k_z Δz/2) on every Fourier mode, which the analyzer divides out. The modes it
  attenuates in energy by more than 4 are dropped (along an axis, kΔx > 2π/3); every mode with k d_e ≤ 10.4 is
  kept, so the analysed range is untouched. The average puts all three components on the nodes, so the mixed
  diagnostics (polarization, helicity) need no phase shift.
- **Times.** `--times` (default 0 130 300 1000; [G12] Fig. 3 shows 50 130 300 1000) select the nearest
  snapshots: at the deck's 25/ω_pe cadence ω_pe t = 130 falls on 125.

### 5.1 Energies

From `diagnostics.csv`, in units of B0²/8π: W_B = ⟨|δB|²⟩/B0² (the magnetic energy less that of B0), W_E and the
kinetic energy of each species, plotted in `<prefix>_energy`:

1. W_B(t)/W_B(0) and W_E(t)/W_B(0) ([G12] Fig. 1a);
2. the rate 2γ_eff = d ln W_B/dt in units of |Ω_e| ([G12] Fig. 1c), a second-order Savitzky-Golay derivative over
   `--rate-window` (default 20/ω_pe);
3. the exchange -Δ(W_B + W_E) against ΔK of each species;
4. the total-energy change over W_B(0).

`<prefix>_temperature_energy` shows, over the initial total energy, the energy of δB, the electric energy, the
electron and ion kinetic energies and |E_tot - E0|, beside the thermal temperatures.

### 5.2 What the energy rates show

- **Small ε0 (0.02-0.1).** (dW_B/dt)/W_B is nearly independent of ε0 and of order 2γ with |γ| ~ 0.002 |Ω_e|,
  linear Landau damping of oblique whistlers.
- **Large ε0 (≥ 0.2), early times.** The rate grows with ε0: nonlinear processes.
- **Heating.** The normalized heating rate (dT∥_e/dt)/W_B is nearly independent of ε0 ([G12] Fig. 1d).

### 5.3 Electron temperatures

From the node moments of the first electron species, summed over the nodes, with ρ_m = `rho`/(4π qom):

1. **The papers' definition** (total kinetic energy): T∥ = Σ P_zz/Σ ρ_m and T⊥ = Σ (P_xx + P_yy)/(2 Σ ρ_m),
   over T0 = T(t = 0). This T0 includes the wave flow (Section 3.7), as in [G12] Fig. 1b and [S08] Fig. 5.
2. **Thermal**: the same sums without the bulk flow ρ_m u_i² of each node, u = J/(`rho`/4π), over
   T0 = m v_te².

`<prefix>_temperatures` plots T∥/T0 and T⊥/T0 of both definitions, T∥/T⊥ and the heating rate
d(T∥/T0)/d(|Ω_e| t) over W_B; `<prefix>_electron_temperatures` plots the thermal ones alone. The parallel
direction is the global z: for ε0 ≥ 0.5, where δB ~ B0, the local field direction departs from it.

### 5.4 Magnetic spectra

From P(k) = Σ_j W |B_k,j|² of each snapshot at the `--times`:

| Spectrum | Definition | Normalization |
| --- | --- | --- |
| S(k⊥) | P summed over k∥ and over the annuli k⊥ ∈ [(m - 1/2) k0, (m + 1/2) k0), over k0 | ∫ S dk⊥ = ⟨\|δB\|²⟩/B0² |
| E(k⊥) of [G12] | S/k⊥ (their Eq. 1: ∫ k⊥ E dk⊥ = total) | slopes one lower than those of S |
| S(k∥) | P summed over k_x, k_y at each k_z, over k0 | ∫ S dk∥ = ⟨\|δB\|²⟩/B0² |
| (k_y, k∥) plane at k_x = 0 | \|B_k\|² per Fourier mode, both signs of k∥, log scale over 7 decades | per mode, as [G12] Fig. 3 |

`<prefix>_spectra` draws S(k⊥) and S(k∥) on one scale, with the broken power law fitted to E(k⊥) drawn as S (its
slopes one more), k⊥ d_e = 1, the largest seeded k and, with `--noise-dir`, the control S(k⊥) dotted;
`<prefix>_spectrum_2d` draws the (k_y, k∥) planes. The [G12] slopes refer by their Eq. 1 to E = S/k⊥;
electron-MHD theory and solar-wind spectra refer to shell-summed spectra, S. Break locations and slope
differences do not depend on the convention.

### 5.5 Power-law slopes and the break

The fits use the mean E(k⊥) of the snapshots within `--fit-window` (default ±20/ω_pe) of `--fit-time` (default
300): one snapshot at the deck's 25/ω_pe cadence, five (280-320) at `FieldOutputCycle = 40`.

- **Single ranges**: log E against log k⊥ on [0.4, 0.9] below the break and on [1.2, k_s] above it, with the
  1-σ error of the fit.
- **Broken power law**: continuous, with a free break k_b, on [0.3, k_s]. [G12] finds k_b ≈ 1/d_e for every ε0,
  with steeper slopes above the break.
- **k_s** is `--k-signal` (default 3) or, with `--noise-dir`, the smaller of it and the first k⊥ ≥ 1 where S(k⊥)
  falls below 10 times the S(k⊥) of the control run at the same cycle (the smallest k_s of the fit window).

Add the seed spread (V4) and the change over the fit window to the fit error.

### 5.6 Wavevector anisotropy

tan²θ_B = Σ_k k⊥² |δB_k|² / Σ_k k∥² |δB_k|².

The 3D isotropic value is 2 (k⊥² = k_x² + k_y²), against 1 in 2D; [G12] quotes 1, the 2D value. Divide 3D
values by 2 before comparing with [S08] or any 2D result. The analyzer takes it at every field snapshot over:

| Set | Definition | Purpose |
| --- | --- | --- |
| all | every k ≤ k_s | [G12] Fig. 2 |
| cascaded | the unseeded k ≤ k_s | the energy the cascade moved |
| 0.7-3 | 0.7 ≤ k d_e ≤ 3 | above every seeded mode (largest seeded k d_e = 0.544) |
| 0.3-3 | 0.3 ≤ k d_e ≤ 3 | the range of [S08] (holds seeded modes in 3D) |

`<prefix>_anisotropy` plots the four against time and the cascaded tan²θ_B against the cascaded energy, as
[S08] Fig. 3, to test that the anisotropy grows with the fluctuation energy. The two shells carry no noise mask:
isotropic noise draws them towards 2.

### 5.7 Gyrotropy

The mean energy per Fourier mode against the azimuth φ of k⊥, folded to [0°, 180°) in 12 bins, and the
gyrotropy index G = std/mean over the bins. Normalizing per mode removes the bias of the excess of lattice
points along the axes and the diagonals. Two shells: the seeded shell, from the smallest to the largest
seeded |k| (0.105-0.544 in the deck), and the cascade shell above it, up to twice the largest (0.544-1.088).

- At t = 0, G = 1.43 in the seeded shell: the seed fills 4 of the 12 bins and its per-mode means differ between
  the axis and the diagonal bins.
- An isotropic random field leaves a finite-sample floor of a few percent; the V0 run measures it.
- [C11] and [G12] report a gyrotropic cascade: G in the cascade shell drops to the floor within about 100/ω_pe;
  G left in the seeded shell is the memory of the seed.

### 5.8 Polarization and helicity

In the basis z, k̂⊥ and z × k̂⊥ at each k with k⊥ > 0, the analyzer accumulates, energy-weighted over every
snapshot, in 10° bins of θ and in the |k| shells [seeded min, seeded max), [max, 2 max), [2 max, 4 max):

- C∥ = |δB_z|²/|δB|², the magnetic compressibility;
- r_B = |δB · z × k̂⊥|²/|δB|², the out-of-plane fraction.

Because k · δB_k = 0, C∥ = (1 - r_B) sin²θ, so r_B carries the independent information. Circularly polarized
whistlers have r_B = 1/2 and C∥ = sin²θ/2, drawn in `<prefix>_polarization`; kinetic Alfvén waves at k⊥ ≫ k∥ put
most of δB⊥ along z × k̂⊥, r_B > 1/2. At any instant a ±k pair forms a partly standing wave whose component split
depends on the relative phase (the total energy does not); summing over all snapshots, as [S08] averages over
the run, removes that bias. [S08] finds agreement with linear theory in the shells k d_e ≈ 0.3, 0.5 and 1.0, and
a total C∥ in [0.3, 0.6] in 2D.

The normalized magnetic helicity σ_m = Σ W k H_k / Σ W |B_k|², with H_k from the Coulomb-gauge A_k =
i k × B_k/k², in the seeded and the cascade shells. A whistler has σ_m = +1 when its phase travels along +B0 and
-1 along -B0, so σ_m measures the imbalance between counter-propagating whistlers: 0 for the balanced seed, and
it stays near 0 in a balanced run.

### 5.9 Electron distributions

From every particle snapshot, in the lab frame, in units of v_te over ±6 v_te in 121 bins, in
`<prefix>_vdf`:

- f(v∥) and f(v_x), each of unit area, on a log scale with the Maxwellian ([G12] Fig. 5). With growing ε0 the
  subthermal core of f(v∥) flattens and the suprathermal tails of both grow.
- δf = [f(v_x, v∥, t) - f(v_x, v∥, 0)]/f(0, 0, 0) at the last snapshot ([S08] Fig. 6): lobes along ±v∥ (Landau
  resonance) and outer contours along constant |v| (pitch-angle scattering by the cyclotron resonance).

f(v∥) and δf mark the largest whistler Landau speed v∥ = |Ω_e| d_e/2 = 0.224 c = 2.24 v_te: the plateau forms inside
|v∥| ≲ 2.2 v_te. The cyclotron resonance of parallel modes, v∥ = (ω - |Ω_e|)/k∥, is |v∥| = 0.31 c (3.1 v_te) at
k∥ d_e = 0.85 and 0.22 c at k∥ d_e = 1: the resonance enters the thermal population there, hence the abrupt
onset of cyclotron damping near k∥ d_e ≈ 0.8.

### 5.10 ω-k spectra

Frequencies identify the modes directly, as the ω-k spectra of [K14] in 2D. The analyzer keeps B_k(t) of the
modes with k⊥ d_e ≤ 4 in the k∥ planes nearest `--omega-k-kpar` (default 0.1 0.2 0.3 0.4, in the deck the planes
k∥ = n k0, 0.105-0.419) and in those of `--omega-k-bins`, and transforms them in time over the snapshots of each
window: `--omega-k-window` (default 500/ω_pe) long, centred on each of `--omega-k-times` (default 250 500 750).
The transform takes no window (a rectangular window) and pads with zeros 4 times, which smooths the curves
without refining the resolution; a window needs at least 8 evenly spaced snapshots. The result is the energy of
each k⊥ shell (width k0, both signs of k∥) per unit ω; ω > 0 travels towards +k∥, along B0, so the two signs
separate counter-propagating waves (consistent with σ_m).

| Field cadence Δt_out | Nyquist π/Δt_out | Resolution 2π/(N Δt_out), 500/ω_pe window |
| --- | --- | --- |
| 25/ω_pe (the deck, N = 21) | 0.281 \|Ω_e\| | 0.027 \|Ω_e\| |
| 10/ω_pe (`FieldOutputCycle = 40`, N = 51) | 0.702 \|Ω_e\| | 0.028 \|Ω_e\| |

`<prefix>_omega_k` maps ω against k⊥ in each plane and window, each shell over its peak, down to 0.1 of it;
`<prefix>_omega_power` plots the energy against ω in the (k⊥, k∥) bins of `--omega-k-bins` (default (0.5, 0.2),
(1, 0.2), (1, 0.4), (2, 0.3)). Both draw the cold electron whistler ω = |Ω_e| k∥ k d_e²/(1 + k² d_e²) (ions at
rest, no displacement current; it lies within 4.5 % of the seed frequencies of Section 3.2) at ±ω, and a
whistler above π/Δt_out at its alias, dotted. The cold whistler peaks at |Ω_e| k∥ d_e/2 (at k d_e = 1), so at
the deck's cadence every whistler of the default planes lies below the Nyquist frequency and aliases appear
only for k∥ d_e > 0.56. The summary prints the Nyquist frequency, the resolution and, per bin, the whistler
frequency against the peak |ω| of each window. Compare the peak with the dispersion and its width with 2|γ|.

## 6. Comparison with the papers and theory

### 6.1 Targets from the papers

Values from the text, or read approximately from figures (≈). [G12] and [S08] use their own boxes, 51.2 d_e in
3D for [G12].

| # | Quantity | Paper result | Source |
| --- | --- | --- | --- |
| 1 | Forward cascade | gyrotropic, quasi-perpendicular, for every ε0 | [G12], [C11] |
| 2 | tan²θ_B(t) | rapid rise until \|Ω_e\| t ≈ 100 (ω_pe t ≈ 220), then a slower rise driven by oblique damping | [G12] Fig. 2 |
| 3 | tan²θ_B against ε0 | grows for 0.02 ≤ ε0 ≤ 0.20, falls for ε0 = 0.5 and 1.0; 3D > 2D at equal ε0 (mind the factor 2) | [G12] Fig. 2 |
| 4 | 2D tan²θ_B (0.3-3/d_e, isotropic = 1) | maxima ≈ 7 (ε = 0.1), ≈ 4.5 (0.05), ≈ 1.8 (0.02); cascaded energy and anisotropy rise, then slowly fall | [S08] Fig. 2b |
| 5 | 2D spectrum of δB(k_x, k_y) | four-pointed star at \|Ω_e\| t = 447 (ε0 = 0.2) | [G12] Fig. 3 |
| 6 | Break in E(k⊥) | k⊥ d_e ≈ 1 for every ε0 | [G12] Fig. 4 |
| 7 | Slopes of E(k⊥) below / above the break at ω_pe t = 300 (≈, runs assigned from the figure labels) | ε0 = 1.0 and 0.5: ≈ -2.7 / -4.4; 0.2: ≈ -2.8 / -4.7; 0.1: ≈ -3.2 / -5.0; 0.05: ≈ -4.1 / -5.1; 0.02: ≈ -4.6 / -5.2; shallower at larger ε0 | [G12] Fig. 4 |
| 8 | ε0 = 0.5 against 1.0 | nearly identical spectra | [G12] Fig. 4 |
| 9 | Noise | the ε0 = 0.02 spectrum reaches the noise near k⊥ d_e ≈ 2 | [G12] Fig. 4 |
| 10 | Field decay rate | weakly ε0-dependent at small ε0; 2\|γ\| with \|γ\| ≈ 0.002 \|Ω_e\| at late times; grows with ε0 at large ε0, early times | [G12] Fig. 1a, c |
| 11 | Heating | parallel ≫ perpendicular; T∥/T⊥ ≈ 1.6 (3D) against ≈ 1.3 (2D) at ε0 = 0.1, late times; (dT∥/dt)/W_B weakly dependent on ε0 | [G12] Fig. 1b, d |
| 12 | 2D heating (≈) | E∥,kin/E0B from 0.05 to ≈ 0.079 / 0.064 / 0.056 and E⊥/E∥ ≈ 0.77 / 0.86 / 0.93 at \|Ω_e\| t = 2000 for ε = 0.1 / 0.05 / 0.02; perpendicular heating keeps growing at late times | [S08] Fig. 5 |
| 13 | f(v) at \|Ω_e\| t = 447 | heating grows with ε0; parallel > perpendicular; subthermal flattening and suprathermal tails grow with ε0 | [G12] Fig. 5 |
| 14 | δf(v) (2D) | parallel lobes (Landau); outer contours along constant energy (cyclotron pitch-angle scattering) | [S08] Fig. 6 |
| 15 | Polarization (2D) | component ratios follow linear whistler theory in the shells k d_e ≈ 0.3, 0.5 and 1.0; total C∥ in [0.3, 0.6] | [S08] Fig. 4 |
| 16 | 2D reduced spectra (ε = 0.1) | α_y ≈ 4.5 (perpendicular), α_x ≈ 9.0 (parallel) at \|Ω_e\| t = 2011; α_x = 6.8 at \|Ω_e\| t = 447 | [S08] Fig. 1 |

### 6.2 Theory

- **Linear damping.** [G12] notes that oblique Landau damping scales as γ ∝ -k² to and beyond k d_e ≈ 1, so linear
  damping alone predicts no break at k⊥ d_e ≈ 1; they attribute the break to nonlinear wave-particle (and
  possibly wave-wave) effects. Late-time removal of energy near θ ≈ 50-60°, the peak of Landau damping, gives
  the four-pointed star of [G12] Fig. 3; cyclotron damping at θ ≈ 0, k∥ d_e ≳ 0.8 truncates the k∥ extent.
  [S08] reads the steepening of the parallel spectrum (α_x = 6.8 → 9.0 between |Ω_e| t = 447 and 2011) as
  cyclotron damping.
- **Cascade phenomenology**, for the shell-summed S (exponents as commonly quoted; the papers fix the exact
  spectral convention):

  | Model | Prediction |
  | --- | --- |
  | Strong electron-MHD turbulence, k d_e < 1 [B99, CL04, CL09] | S(k) ∝ k^(-7/3) |
  | Strong electron-MHD turbulence, scale-dependent anisotropy [CL04] | k∥ ∝ k⊥^(1/3) |
  | Weak whistler turbulence, k d_e < 1 [GB03] | S(k⊥) ∝ k⊥^(-5/2) |
  | 3D electron MHD, k d_e > 1 [Sh09], as quoted by [G12] | shallower, about k^(-5/3) |

  The spectra of [G12] are steeper and steepen further above k⊥ d_e ≈ 1, with ε0-dependent slopes: there is no
  universal spectrum. The ε0 = 0.5 and 1.0 spectra coincide, the regime where an electron-MHD comparison means
  most.
- **Cascade times** of [S08], in units of 1/|Ω_e|, with the band energy |δB(k)|² ≈ k S(k):

  |Ω_e| τ⊥(k⊥) ≃ 2π [k∥ d_e/(k⊥ d_e)³] B0²/|δB(k⊥)|², |Ω_e| τ∥(k∥) ≃ [4π/(k∥ d_e)²] B0²/|δB(k∥)|².

  [S08] finds |Ω_e| τ ≳ 10⁵, far longer than the runs: neither ensemble reaches a steady state, so simulations
  compare at matched times, never "at saturation". τ⊥ ≪ τ∥ explains the perpendicular anisotropy, and both
  scale as 1/|δB|², which explains why the anisotropy grows with ε0 in the weak regime.
- **Observations.** [G12] relates the break and the steeper high-k range to the electron-scale breaks of
  solar-wind magnetic spectra [Sa09, Sa10, A09]; [S08] relates the compressibility to that of short-wavelength
  solar-wind turbulence. The quantities to compare do not depend on the convention: the break in d_e, the slope
  difference across it and C∥.

## 7. Pitfalls

1. **tan²θ_B is 2 for isotropy in 3D.** Halve it before comparing with 2D.
2. **E(k⊥) or S(k⊥).** The slopes differ by exactly 1; state which one, and compare break locations and slope
   differences.
3. **The papers' temperature includes the flow.** At t = 0 it is too high by 0.73 ε0 (∥) and 0.86 ε0 (⊥) in the
   deck's box. Use the thermal definition for physics and the papers' one for figure comparison.
4. **Noise.** Fit only where S > 10 times the V0 noise (`--noise-dir`). Noise is isotropic: it pulls tan²θ_B
   towards 2 and the slopes towards flat.
5. **Smoothing and θ > 0.5** damp and steepen the spectrum at k d_e ≳ 2, where the break physics lies. V1 and V3
   quantify both.
6. **Polarization snapshots** carry the standing-wave interference of ±k pairs; the analyzer sums over every
   snapshot.
7. **No steady state.** Compare at matched times (ω_pe t = 50, 130, 300, 1000) and report the time of every
   result; ω_pe t = 130 needs `FieldOutputCycle = 40`.
8. **Seeds.** The papers give no phases: 2-3 seeds per ε0, at least for the main run, ε0 = 0.1.
9. **Local and global parallel direction.** The analyzer measures T∥ along z; for ε0 ≥ 0.5 the local field
   direction departs from it.
10. **Thermal-speed convention.** `uth` is √(k_B T/m); with √(2 k_B T/m) β would change by 2. Check β_e = 0.1.
11. **Run length.** The 3D runs of [G12] end at ω_pe t = 1000, the 2D runs of [S08] at ≈ 4500: do not compare
    late 2D numbers with 3D numbers at ω_pe t = 1000.
12. **Box.** The deck's 60 d_e box seeds k d_e = 0.105-0.544 against 0.123-0.638 in [G12], with ω = 0.0113-0.129
    against 0.0153-0.163 |Ω_e| and 18 against 10 modes below ω_LH ([G12] box values); quantities tied to the
    seeded k differ, the break in d_e and the slope differences do not.

## Papers

- **[G12]** S. P. Gary, O. Chang and J. Wang, *Forward cascade of whistler turbulence: three-dimensional
  particle-in-cell simulations*, ApJ 755, 142 (2012). The 3D template.
- **[S08]** S. Saito, S. P. Gary, H. Li and Y. Narita, *Whistler turbulence: particle-in-cell simulations*,
  Phys. Plasmas 15, 102305 (2008). The 2D3V predecessor: the construction of each mode and several diagnostics.
- **[C11]** O. Chang, S. P. Gary and J. Wang, *Whistler turbulence forward cascade: three-dimensional
  particle-in-cell simulations*, Geophys. Res. Lett. 38, L22102 (2011). The first 3D run; its "Run 3" initial
  spectrum is that of [G12].
- **[L17]** G. Lapenta, *Exactly energy conserving semi-implicit particle in cell formulation*, J. Comput. Phys.
  334, 349 (2017). ECSIM.
- **[K14]** H. Karimabadi, V. Roytershteyn et al., Phys. Plasmas 21, 062308 (2014). ω-k spectra of 2D
  turbulence.
- **[B99]** D. Biskamp, E. Schwarz, A. Zeiler, A. Celani and J. F. Drake, *Electron magnetohydrodynamic
  turbulence*, Phys. Plasmas 6, 751 (1999).
- **[CL04]** J. Cho and A. Lazarian, *The anisotropy of electron magnetohydrodynamic turbulence*, ApJ 615, L41
  (2004).
- **[CL09]** J. Cho and A. Lazarian, *Simulations of electron magnetohydrodynamic turbulence*, ApJ 701, 236
  (2009).
- **[GB03]** S. Galtier and A. Bhattacharjee, *Anisotropic weak whistler wave turbulence in electron
  magnetohydrodynamics*, Phys. Plasmas 10, 3065 (2003).
- **[Sh09]** D. Shaikh, *Whistler wave cascades in solar wind plasma*, MNRAS 395, 2292 (2009).
- **[Sa09]** F. Sahraoui, M. L. Goldstein, P. Robert and Yu. V. Khotyaintsev, Phys. Rev. Lett. 102, 231102
  (2009).
- **[Sa10]** F. Sahraoui, M. L. Goldstein, G. Belmont, P. Canu and L. Rezeau, Phys. Rev. Lett. 105, 131101
  (2010).
- **[A09]** O. Alexandrova, J. Saur, C. Lacombe, A. Mangeney, J. Mitchell, S. J. Schwartz and P. Robert,
  Phys. Rev. Lett. 103, 165003 (2009).
