# Analysis

The Python tools in [tools/](../../tools) read the VTK output of a run ([output](output.md)) and compare it with
the theory of its case ([cases](cases.md)). Each analyzer prints a summary and writes plots and tables; `--help`
lists every option.

## Installing

From the TPIC root, install the `tpic-tools` package of [pyproject.toml](../../pyproject.toml) with its analysis
extras (NumPy, Matplotlib, SciPy):

```sh
python3 -m pip install -e '.[analysis]'
```

The package is [tools/tpic_tools](../../tools/tpic_tools): `inputs` (deck reader), `runs` and `vtk` (field file
decoder) use only the standard library; `fields` (field reader) and `particles` (particle piece reader) need
NumPy. The scripts also run directly from a source checkout when the dependencies are installed. The 3D views
(`--density-3d`, `--render-3d`, `--render-cycle`, `--render-particles`, `--particle-cycle`) need `pyvista`, and
`hdf5_fields_to_xdmf.py` needs `h5py`; neither is part of the extras.

## Conventions

- Every analyzer takes `--run-dir` (default `.`), the directory the run is launched from, and reads the deck
  there (its file name below, or the only matching deck) unless `--input` names it. The data are the deck's
  `SaveDirName` under the run directory unless `--data-dir` names them; the whistler, whistler-spectrum and
  reconnection analyzers and `explicit_es_pic.py` have no `--data-dir`.
- The analyzers read the TPIC VTK field files, one file per array and cycle, as `(z, y, x)` arrays over the
  global nodes (the global cells for `Gauss`); the two-stream, explicit-comparison, Gaussian-expansion,
  reconnection and whistler-turbulence analyzers also read the VTK particle pieces. ADIOS2 and HDF5 output is
  not an input to them.
- They read the periodicity of each axis from `PERIODICX`, `PERIODICY` and `PERIODICZ`. In a mean over an axis,
  a periodic axis counts its end node, the image of its first, once, and an axis with walls counts its two wall
  nodes as half a cell each. Two-stream, whistler, whistler spectrum and Weibel transform along x and average
  over y and z, so they need a periodic x; reconnection needs a periodic x; Taylor-Green and whistler turbulence
  need a box periodic along every axis.
- Plots are PNG files, most with a PDF of the same name beside them. Fonts, colours and line styles are set by
  constants at the top of each script.

## Tools

| Tool (`tools/`) | Case, deck | Outputs it needs | Products in the run directory |
| --- | --- | --- | --- |
| `postprocessing/analyze_two_stream.py` | `TwoStream`, `twoStream1D.inp` | `E`; `diagnostics.csv`; particle snapshots for the phase space | `two_stream_mode_growth.png` and `.csv`, with `_figure`, `_energy`, `_phase_space` |
| `postprocessing/analyze_weibel.py` | `Weibel`, `weibel2D.inp` | `B`; `E`; electron `rho`, `PXX`, `PYY`, `PZZ`; `diagnostics.csv` | `weibel_growth`, `weibel_anisotropy`, `weibel_bz_space_time`, `weibel_ex_space_time` |
| `postprocessing/analyze_tgv.py` | `TaylorGreenVortex`, `TaylorGreenVortex3D.inp` | `B`, `E`, `rho`, `J`, per-species `rho`, `J`, pressure; `diagnostics.csv`; the console log | `tgv_analysis.png`, `.csv`, `_summary.csv`, `_output_scan.csv` |
| `postprocessing/analyze_whistler.py` | `Whistler`, `whistler2D.inp` | `B`; `E` and `J` when written | `whistler_analysis.png` |
| `postprocessing/analyze_whistler_spectrum.py` | `Uniform`, `whistlerAnisotropy2D.inp` | `B` at a fixed cadence | `whistler_spectrum.png` and `.csv`, `whistler_waves`, `whistler_travelling`, `whistler_crossing` |
| `postprocessing/analyze_whistler_turbulence.py` | `WhistlerTurbulence`, `whistlerTurbulence3D.inp` | `B`; electron `rho`, `J`, `PXX`, `PYY`, `PZZ`; `diagnostics.csv`; particle snapshots | `whistler_turbulence_*` plots, `_timeseries.csv`, `_summary.txt` |
| `postprocessing/analyze_reconnection.py` | Harris sheets, `GEM3D.inp`, `GEMHarris.inp`, `doubleGEM.inp`, `reconnection_comparison/GEM3D.inp` | `B`, `E`; `diagnostics.csv`; `rho_<s>` or particle snapshots for the 3D views | `reconnection_rate.png` and `.csv`, with `_flux_rate`, `_energy` |
| `postprocessing/analyze_gaussianExpansion.py` | `GaussianExpansion`, `gaussianExpansion3D.inp`, `gaussianExpansion2D.inp` | `B`, `E`, per-species `rho`, `J`, `PXX`, `PYY`, `PZZ`; `diagnostics.csv`, `gauss.csv`; particle snapshots | `gaussian_expansion_*` plots, `_snapshots.csv`, `_summary.txt` |
| `explicit_es_pic.py` | quasi-1D decks, `twoStream1D.inp` | the particle snapshot of cycle 0 | `explicit_pic_energy.csv`, `explicit_pic_particles.npz` |
| `postprocessing/analyze_explicit_comparison.py` | as `explicit_es_pic.py` | `diagnostics.csv`; particle snapshots; the two files of `explicit_es_pic.py` | `explicit_comparison_energy`, `_distribution`, `_phase_space` |
| `postprocessing/hdf5_fields_to_xdmf.py` | any | HDF5 field output | `fields.xdmf` in the output directory |

## Two-stream

`analyze_two_stream.py` measures the seeded `Ex` Fourier mode of a cold symmetric two-stream run, fits its growth
over `--fit-start` to `--fit-end` (in `omega_pe*t`) and compares it with the cold theory. It looks for the deck in
the run directory, then in `input/`. Beside the main plot it writes the mode history with the cold theory alone
(`_figure`), the kinetic, electric and total energy of `diagnostics.csv` over the initial total energy
(`_energy`) and, when the run holds particle snapshots, the electron phase space `(x, vx)` coloured by beam
(`_phase_space`), by default at `t = 0`, half the run and the last snapshot, or at the `--phase-space-times`.

## Weibel

`analyze_weibel.py` compares the growth of the seeded `Bz` mode `WeibelSeedModeX`, averaged over y and z, with
the kinetic Weibel rate (E. S. Weibel, Phys. Rev. Lett. 2, 83 (1959)), with cold ions. The rate uses the second
moments of the electron velocities measured at each snapshot when the deck writes electron `rho`, `PXX`, `PYY`
and `PZZ` (they include the bulk flow of the growing currents), and the deck temperatures otherwise. It
subtracts the standing light wave that a magnetic-only seed excites before taking the local growth rate. The
plots are the growth up to saturation, the anisotropy `<v_y^2>/<v_x^2>`, `<v_z^2>/<v_x^2>` with the field
energies, and the space-time plots of `Bz` and, when the deck writes `E`, of `Ex`. Times are in `omega_pe*t`.

## Taylor-Green vortex

`analyze_tgv.py` runs layered checks on a `TaylorGreenVortex` run: resolution and time scales from the deck, the
analytic `t = 0` fields and species flows, `div(B)`, Gauss-law and Ampere residuals, the energy of
`diagnostics.csv`, magnetic and current shell spectra and the energy in the vortex modes, the electron pressure
anisotropy (after subtracting the bulk flow reconstructed from `J` and `rho`), and a Weibel-like growth fit with
a local kinetic estimate. It also scans the console log of the run (the newest `*.out` of the run directory, or
`--out-log`). `--from-csv` redoes the fit and the plot from a saved time series.

## Whistler

`analyze_whistler.py` analyses the seeded parallel whistler of `whistler2D.inp` (`B0 = B0x x`, `k = kx x`): it
fits the phase and the growth of `b_R = By - i*Bz` in the seeded mode over `--fit-start` to `--fit-end` (in
`omega_pi*t`), compares the frequency with the cold whistler relation, and, when `E` and `J` are written,
checks their ratios to `B` against the eigenmode.

## Whistler spectrum

`analyze_whistler_spectrum.py` transforms `By` and `Bz`, averaged over y and z, in x and in time, with no window
in time, and compares the spectrum of the unseeded `whistlerAnisotropy2D.inp` run with the cold electron
whistler relation (D. Gonzalez-Herrero, E. Boella and G. Lapenta, Comput. Phys. Commun. 229, 162 (2018),
section 5); the ridge of every mode in `--ridge-k` goes to the CSV. The waves plot shows the power of
`By - i*s*Bz` (`s` the sign of `B0x`) with the four cold waves along the field; the travelling plot follows the
whistlers of the band `--travel-k` (in `k*d_e`) through the box with their phase and group speeds; the crossing
plot follows one wave packet of the band `--crossing-k` across the whole box. The snapshots must resolve every
frequency the run holds: a wave above `pi/dt_out` folds back into the spectrum.

## Whistler turbulence

`analyze_whistler_turbulence.py` follows the analysis of S. P. Gary, O. Chang and J. Wang, ApJ 755, 142 (2012),
as Section 5 of [whistler turbulence](whistler_turbulence.md) sets it out, in electron units: field energies
and their decay rate, electron temperatures, the perpendicular and parallel magnetic spectra with their slopes,
the `(k_y, k_par)` plane, wavevector anisotropy, gyrotropy, helicity and polarization, the omega-k spectra of `B`
in time windows of the run (with a rectangular window, no window function) against the whistler relation, and
the electron velocity distributions. It divides the node average of the written `B` out of the spectra.
`--noise-dir`, the output of a `WhistlerTurbulenceEpsilon = 0` run of the same deck, sets where the spectra
reach the noise; `--render-3d` adds 3D renderings of the current density and the density fluctuations;
`--only PART ...` redraws some groups of plots alone, reading only the data they need. The script header lists
the plots and the parts.

## Reconnection

`analyze_reconnection.py` measures the reconnection rate on the X line of the Harris-sheet cases `GEM`,
`GEMnoPert`, `GEMHarris`, `DoubleGEM` and `DoubleGEMnoPert` ([cases](cases.md)). It integrates `By` along the
sheet from the X line (`x = Lx/2`, `y = Ly/2`; for the double sheet `(Lx/4, Ly/4)` and `(3Lx/4, 3Ly/4)`, each
analysed as a run of its own) to obtain the reconnected flux, and reports its time derivative in units of
`B0*v_A` against `Omega_ci*t`, as a z average and along z, beside the direct sample `-Ez/(B0*v_A)` over a square
of `--box-cells` cells around the X line. `v_A` uses the drifting sheet ions, or the background ions with
`--alfven-density background`. `--time-average WINDOW` averages both rates over `WINDOW/Omega_ci`;
`--max-cycle CYCLE` ends every run at that cycle, so a run still being written can be compared with a finished
one. The total energy of each run is plotted as `(E - E0)/E0`.

Several runs, given as several `--run-dir` values or as several `--input` decks launched from one directory,
are overlaid with `--labels` and compared with the first over their common time interval: mean and RMS
differences of the flux and the rates, the peak rates and a time lag; the plots and the CSV of a comparison go
to `reconnection_comparison.png` and `.csv` in the current directory unless `--output` and `--csv` name them.
A run directory may instead hold the
output of an external run as one structured-points VTK file per field and cycle with a deck of the same keys,
which the analyzer reads through the same analysis. `--render-cycle` replaces the rate analysis by 3D views of
the ion density and the field lines, `--instabilities` by plots of the lower-hybrid drift waves, the drift kink
and the flux ropes of the sheet, and `--particle-cycle` by the most energetic particles of a particle snapshot.
The script header states the units, the method and every option.

## Gaussian expansion

`analyze_gaussianExpansion.py` compares a `GaussianExpansion` run with the self-similar solution of its deck
along each Gaussian axis: width, centre density, flow, cooling, the ambipolar field and the energy the ions
gain, from the field snapshots, `diagnostics.csv`, `gauss.csv` when present and the particle snapshots
(`--no-particles` skips them). It prints the acceptance checks of
[Gaussian expansion](gaussian_expansion.md) and writes its plots, `gaussian_expansion_snapshots.csv` and
`gaussian_expansion_summary.txt` (the prefix is `--prefix`); `--tables` prints the analytic solution of the deck
and stops, and `--density-3d` adds a 3D view of the ion density.

## Explicit comparison

`tools/explicit_es_pic.py` runs a quasi-1D deck (default `twoStream1D.inp`) again with a momentum-conserving
explicit electrostatic PIC (linear weighting, FFT Poisson solve, leapfrog push), started from the particles of
the TPIC run's snapshot at cycle 0, so the deck needs `ParticlesOutputCycle > 0`. Nothing initialized may depend
on y or z. It writes `explicit_pic_energy.csv` and `explicit_pic_particles.npz` to the run directory.
`analyze_explicit_comparison.py` then plots, for both runs, the energy error with `lambda_D/dx`, the electron
velocity distribution and the electron phase space at `t = 0` and at the last common particle snapshot (or the
one nearest `--end`), after the `--output` prefix (default `RUN_DIR/explicit_comparison`). Times are in
`omega_pe*t`.

## HDF5 to XDMF

`hdf5_fields_to_xdmf.py SOURCE` writes an XDMF file for HDF5 field output, from an output directory (every
published `fields_t*.hdf5.json`) or from one such index; it reads only metadata. `--output` sets the path
(default `fields.xdmf`, or `fields_tNNNNNN.xdmf` beside a single index) and `--overwrite` replaces an existing
file. Opening it in ParaView is described in [output](output.md#viewing-in-paraview).

## Developer tools

`compare_simulation_outputs.py` compares two completed runs ([testing](../developer/testing.md)),
`benchmark_simulation.py` takes paired timing measurements ([performance](../developer/performance.md)), and
`dependency_inventory.py` verifies the bundled LPFx copy ([architecture](../developer/architecture.md)).
