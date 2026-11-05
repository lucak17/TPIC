# Input keys

The table lists every key the input parser reads, one row each:
[src/config/simulation_parameters.cpp](../../src/config/simulation_parameters.cpp) reads them and
[simulation_parameters.hpp](../../include/tpic/config/simulation_parameters.hpp) holds the defaults. Any other
key is ignored and listed in one startup warning. The value grammar (numbers, booleans, lists, comments) is in
[deck syntax](inputs.md#deck-syntax), and the units in [units and normalization](inputs.md#units-and-normalization).

A case key acts only in the case named in its meaning; [cases](cases.md) describes each case and its admission
rules. Lengths, times, velocities and fields are in the units of the deck. A species list holds at least `ns`
values; `qom` and the `npcel*` lists hold `ns + nsTestPart`. An engine value is empty or one of `default`,
`pvtk`, `vtk`, `adios2`, `bp`, `bp5`, `hdf5` and `h5` ([output](output.md)).

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `dt` | real | required | time step; positive |
| `ncycles` | integer | required | timesteps of `--run`, non-negative; with `--restart` the absolute last cycle |
| `th` | real | `1.0` | time-centring parameter in `[0.5, 1]`; `0.5` conserves the total energy |
| `Smooth` | real | `1.0` | `face7` centre weight of one smoothing pass, in `[0, 1]`; smoothing acts only below 1 |
| `SmoothNiter` | integer | `6` | smoothing passes, non-negative; `0` turns smoothing off |
| `ECSimSmoothing` | `none` or `energy_conserving` | `none` | switches energy-conserving smoothing on ([smoothing](inputs.md#smoothing)) |
| `ECSimSmoothingOperator` | `face7` or `binomial27` | `face7` | stencil of one smoothing pass |
| `SaveDirName` | string | `data` | directory of the field, particle and scalar output files; empty selects `data` |
| `RestartDirName` | string | `data` | directory of the checkpoints, read by `--restart`; empty selects `data` |
| `ns` | integer | required | number of species, from 1 to `TPIC_MAX_SPECIES` |
| `nsTestPart` | integer | `0` | test-particle species; a positive value is an unsupported request |
| `B0x` | real | `0.0` | x component of the uniform magnetic field added to every case; `Whistler` requires it positive, `WhistlerTurbulence` zero |
| `B0y` | real | `0.0` | y component of the uniform field; zero in `Whistler` and `WhistlerTurbulence` |
| `B0z` | real | `0.0` | z component of the uniform field; zero in `Whistler`, positive in `WhistlerTurbulence` |
| `delta` | real | `0.5` | current-sheet half-thickness of the Harris-family cases and `ForceFree`; positive |
| `HarrisB0x` | real | `0.0` | Harris-family sheet field `Bx = HarrisB0x*tanh((y - Ly/2)/delta)` |
| `pertGEM` | real | `0.0` | `GEMHarris`: amplitude, in units of `HarrisB0x`, of the seed with one wavelength of the box in x and y |
| `pertHump` | real | `0.0` | `GEMHarris`: amplitude, in units of `HarrisB0x`, of the Gaussian hump seed |
| `deltaxHump` | real | `8.0` | `GEMHarris`: hump width along x in units of `delta`; positive |
| `deltayHump` | real | `4.0` | `GEMHarris`: hump width along y in units of `delta`; positive |
| `kxHump` | real | `-1.0` | `GEMHarris`: hump wavenumber along x; a negative value selects `2*pi/Lx` |
| `kyHump` | real | `-1.0` | `GEMHarris`: hump wavenumber along y; a negative value selects `pi/Ly`; nonzero with a hump |
| `currentFromAmpere` | boolean | `false` | `GEMHarris`: the species with nonzero `w0` carry the current of the initial field |
| `spatiallyVaryingThermal` | boolean | `false` | `GEMHarris` with `currentFromAmpere`: sets `wth` of the drifting species from the pressure of their Maxwellian, which returns the deck `wth` to rounding |
| `ForceFreeB0x` | real | `0.0` | `ForceFree`: `Bx = ForceFreeB0x*tanh((y - Ly/2)/delta)` |
| `ForceFreeB0z` | real | `0.0` | `ForceFree`: `Bz = ForceFreeB0z/cosh((y - Ly/2)/delta)` |
| `FluxRopeB0` | real | `0.1` | `fluxRope`: axial field `Bz` and amplitude of the azimuthal field `FluxRopeB0*r/(r^2 + a^2)` |
| `FluxRopeWidth` | real | `1.0` | `fluxRope`: core radius `a`; positive |
| `FluxRopePressureFloor` | real | `9.0e-4` | `fluxRope`: pressure added to the pressure-balance profile; non-negative |
| `FluxRopeElectronPressureFraction` | real | `0.1` | `fluxRope`: electron share of the pressure, in `[0, 1]` |
| `WeibelSeedModeX` | integer | `0` | `Weibel`: x mode number `m` of the seed `Bz = A*cos(k*x)`, `k = 2*pi*m/Lx`; positive with a seed |
| `WeibelSeedAmplitude` | real | `0.0` | `Weibel`: seed amplitude `A` |
| `WeibelSeedGrowthRate` | real | `0.0` | `Weibel`: growth rate `gamma` of the seed field `Ey = -(gamma/(c*k))*A*sin(k*x)`; `0` seeds no E |
| `TwoStreamSeedModeX` | integer | `0` | `TwoStream`: x mode number of the density seed; below `nxc/2`, positive with a seed |
| `TwoStreamSeedDensityAmplitude` | real | `0.0` | `TwoStream`: electron density factor `1 + A*cos(k*x + phase)` with the matching `Ex`; magnitude below 1 |
| `TwoStreamSeedPhase` | real | `0.0` | `TwoStream`: seed phase in radians |
| `WhistlerModeX` | integer | `0` | `Whistler`: x mode number of the wave, `k = 2*pi*m/Lx`; positive for `eigenmode` and `magnetic_only` |
| `WhistlerModeY` | integer | `0` | `Whistler`: must be 0 (parallel propagation) |
| `WhistlerModeZ` | integer | `0` | `Whistler`: must be 0 |
| `WhistlerSeedEpsilon` | real | `0.0` | `Whistler`: seed amplitude in units of `B0x`; non-negative |
| `WhistlerPhase` | real | `0.0` | `Whistler`: phase of the wave in radians |
| `WhistlerUseExactColdRoot` | boolean | `true` | `Whistler` eigenmode: the exact cold-plasma root; `false` uses `omega = B0x*(c*k)^2/(1 + (c*k)^2/mu)` with the mass ratio `mu` |
| `WhistlerSeedMode` | `eigenmode`, `magnetic_only` or `noise` | `eigenmode` | `Whistler`: wave fields with the electron flow, the magnetic field alone, or magnetic noise over many modes |
| `WhistlerNoiseSeed` | uint64 | `12345` | `Whistler` noise: seed of the mode phases |
| `WhistlerNoiseModeMax` | integer | `0` | `Whistler` noise: highest mode; `0` selects `WhistlerModeX`, or 16 when that is 0 too |
| `WhistlerFieldTimeOffset` | real | `0.0` | `Whistler` eigenmode: the fields start at the wave phase of this time |
| `WhistlerParticleTimeOffset` | real | `0.0` | `Whistler` eigenmode: the electron flow starts at the wave phase of this time |
| `TaylorGreenVortexMode` | integer | `1` | `TaylorGreenVortex`: default of the three per-axis modes |
| `TaylorGreenVortexModeX` | integer | `TaylorGreenVortexMode` | `TaylorGreenVortex`: x mode number, `kx = 2*pi*m/Lx`; with a seed positive and `kx = ky` |
| `TaylorGreenVortexModeY` | integer | `TaylorGreenVortexMode` | `TaylorGreenVortex`: y mode number |
| `TaylorGreenVortexModeZ` | integer | `TaylorGreenVortexMode` | `TaylorGreenVortex`: z mode number |
| `TaylorGreenVortexVelocityAmplitude` | real | `0.0` | `TaylorGreenVortex`: bulk flow `u = U*sin(kx*x)*cos(ky*y)*cos(kz*z)`, `v = -U*cos(kx*x)*sin(ky*y)*cos(kz*z)` |
| `TaylorGreenVortexMagneticAmplitude` | real | `0.0` | `TaylorGreenVortex`: magnetic seed of the same pattern, carried by one electron and one ion species |
| `TaylorGreenVortexSeedMotionalE` | boolean | `false` | `TaylorGreenVortex`: start with `E = -u x B`, uniform field included |
| `GaussianExpansionSigmaX` | real | `0.0` | `GaussianExpansion`: rms width along x; `0` makes the axis uniform |
| `GaussianExpansionSigmaY` | real | `0.0` | `GaussianExpansion`: rms width along y |
| `GaussianExpansionSigmaZ` | real | `0.0` | `GaussianExpansion`: rms width along z |
| `GaussianExpansionCutoff` | real | `0.0` | `GaussianExpansion`: load only the subcells within this many widths of the centre; `0` loads every subcell |
| `WhistlerTurbulenceEpsilon` | real | `0.0` | `WhistlerTurbulence`: mean of `dB^2/B0z^2` of the 150-mode seed; non-negative |
| `WhistlerTurbulenceSeed` | uint64 | `1` | `WhistlerTurbulence`: seed of the mode phases |
| `c` | real | `1.0` | speed of light; another value is an unsupported request |
| `Lx` | real | `10.0` | box length along x |
| `Ly` | real | `10.0` | box length along y |
| `Lz` | real | `10.0` | box length along z |
| `nxc` | integer | `64` | global cells along x |
| `nyc` | integer | `64` | global cells along y |
| `nzc` | integer | `64` | global cells along z |
| `FluxRopeCenterX` | real | `0.5*Lx` | `fluxRope`: x position of the rope axis |
| `FluxRopeCenterY` | real | `0.5*Ly` | `fluxRope`: y position of the rope axis |
| `GaussianExpansionCenterX` | real | `0.5*Lx` | `GaussianExpansion`: x of the centre; a grid node when the x width is positive |
| `GaussianExpansionCenterY` | real | `0.5*Ly` | `GaussianExpansion`: y of the centre; a grid node when the y width is positive |
| `GaussianExpansionCenterZ` | real | `0.5*Lz` | `GaussianExpansion`: z of the centre; a grid node when the z width is positive |
| `XLEN` | integer | `1` | processes along x ([decomposition](inputs.md#decomposition)) |
| `YLEN` | integer | `1` | processes along y |
| `ZLEN` | integer | `1` | processes along z |
| `PERIODICX` | boolean | `true` | periodic fields along x; `false` puts field walls on both x faces |
| `PERIODICY` | boolean | `true` | periodic fields along y |
| `PERIODICZ` | boolean | `true` | periodic fields along z |
| `PERIODICX_P` | boolean | `PERIODICX` | periodic particles along x; `false` puts particle walls on both x faces |
| `PERIODICY_P` | boolean | `PERIODICY` | periodic particles along y |
| `PERIODICZ_P` | boolean | `PERIODICZ` | periodic particles along z |
| `Case` | case name | required | the initial state: `Uniform`, `GEM`, `GEMnoPert`, `GEMHarris`, `DoubleGEM`, `DoubleGEMnoPert`, `plasmoidChain`, `ForceFree`, `Weibel`, `Whistler`, `TaylorGreenVortex`, `TwoStream`, `fluxRope`, `GaussianExpansion` or `WhistlerTurbulence` ([cases](cases.md)) |
| `QuietStart` | boolean | `false` | particles at the subcell centres instead of random points ([QuietStart](inputs.md#quietstart)) |
| `WriteMethod` | engine | empty (`pvtk`) | engine of the field and particle snapshots |
| `FieldWriteMethod` | engine | empty (`WriteMethod`) | engine of the field and moment snapshots |
| `ParticlesWriteMethod` | engine | empty (`WriteMethod`) | engine of the particle snapshots |
| `RestartWriteMethod` | engine | empty (ADIOS2 if built, else HDF5, else none) | engine of the checkpoints: `adios2` or `hdf5` |
| `SimulationName` | string | empty | name printed in the configuration report |
| `GMREStol` | real | `1e-3` | GMRES tolerance relative to the right-hand-side norm, in `(0, 1)` |
| `FieldOutputCycle` | integer | `100` | cadence of the field and moment snapshots in cycles; `0` or less disables them |
| `ParticlesOutputCycle` | integer | `0` | cadence of the particle snapshots; `0` or less disables them |
| `TestParticlesOutputCycle` | integer | `0` | test-particle output; a positive value is an unsupported request |
| `RestartOutputCycle` | integer | `0` | checkpoint cadence; `0` or less disables checkpoints |
| `CallFinalize` | boolean | `true` | with `RestartOutputCycle > 0`, also checkpoint the final state |
| `DiagnosticsOutputCycle` | integer | `FieldOutputCycle` | cadence of the rows of `diagnostics.csv`; `0` or less disables the file |
| `DiagnosticEnergyCycle` | integer | `1` | cadence of the console energy report; `0` or less reports only the final state |
| `DiagnosticGaussCycle` | integer | `0` | cadence of the Gauss-law residual (`gauss.csv` and a console line); `0` or less disables it |
| `FieldOutputTag` | tags | empty | `B`, `E` and `Gauss` joined by `+`; with an empty `MomentsOutputTag` too, the snapshots hold `B`, `E` and `rho` and `J` of the total and of every species |
| `ParticlesOutputTag` | string | empty | no effect: a particle snapshot always holds position, velocity, charge and ID |
| `MomentsOutputTag` | tags | empty | `rho`, `J`, `PXX`, `PXY`, `PXZ`, `PYY`, `PYZ`, `PZZ` joined by `+`: `<name>` every species, `<name>_tot` the total, `<name><s>` species `s` |
| `bcEMfaceXright` | integer | `0` | field wall at the upper x face: `0` PEC, `1` PMC, `2` open ([walls](inputs.md#walls)) |
| `bcEMfaceXleft` | integer | `0` | field wall at the lower x face |
| `bcEMfaceYright` | integer | `0` | field wall at the upper y face |
| `bcEMfaceYleft` | integer | `0` | field wall at the lower y face |
| `bcEMfaceZright` | integer | `0` | field wall at the upper z face |
| `bcEMfaceZleft` | integer | `0` | field wall at the lower z face |
| `bcPfaceXright` | integer | `1` | particle wall at the upper x face: `1` mirror, the only admitted code |
| `bcPfaceXleft` | integer | `1` | particle wall at the lower x face |
| `bcPfaceYright` | integer | `1` | particle wall at the upper y face |
| `bcPfaceYleft` | integer | `1` | particle wall at the lower y face |
| `bcPfaceZright` | integer | `1` | particle wall at the upper z face |
| `bcPfaceZleft` | integer | `1` | particle wall at the lower z face |
| `rhoINIT` | list of `ns` reals | required | 4π times the charge density of each species, non-negative; a case with a profile scales it |
| `qom` | list of `ns + nsTestPart` reals | required | charge-to-mass ratio of each species, nonzero |
| `npcelx` | list of `ns + nsTestPart` integers | required | particles per cell along x, positive |
| `npcely` | list of `ns + nsTestPart` integers | required | particles per cell along y, positive |
| `npcelz` | list of `ns + nsTestPart` integers | required | particles per cell along z, positive |
| `uth` | list of `ns` reals | required | thermal speed (standard deviation) of the x velocity, non-negative |
| `vth` | list of `ns` reals | required | thermal speed of the y velocity |
| `wth` | list of `ns` reals | required | thermal speed of the z velocity |
| `u0` | list of `ns` reals | required | x drift velocity |
| `v0` | list of `ns` reals | required | y drift velocity |
| `w0` | list of `ns` reals | required | z drift velocity |
| `CenterInitialVelocity` | list of booleans | `false` for every species | remove the sampled mean velocity of each species ([CenterInitialVelocity](inputs.md#centerinitialvelocity)) |
