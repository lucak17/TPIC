#!/usr/bin/env python3
"""Growth of the seeded Weibel mode against kinetic theory.

The instability is that of E. S. Weibel, Phys. Rev. Lett. 2, 83 (1959).

Usage from the repository root::

    python3 tools/postprocessing/analyze_weibel.py --run-dir /path/to/run

The run directory is the one the simulation was launched from; it holds the deck, ``weibel2D.inp`` unless
``--input`` names another, and the ``SaveDirName`` of the deck with ``B`` snapshots (``--data-dir`` overrides it).
When the deck also writes the density and the pressure of the electrons (``MomentsOutputTag`` naming ``rho``,
``PXX``, ``PYY`` and ``PZZ`` of the electron species), the theory uses the second moments <v_i^2> of the
electron velocities measured at each snapshot, which include the bulk flow of the growing filament currents;
otherwise it uses the temperatures of the deck. Times are in ``omega_pe*t``.

Method.  The seeded mode is the cos(kx) Fourier mode ``WeibelSeedModeX`` of Bz averaged over y and z.  For k
along the cold direction x and the current along the hot direction y, its kinetic growth rate gamma (in omega_pe)
solves

    (k d_e)^2 + gamma^2 + sum_i omega_pi^2/omega_pe^2 = A - (A + 1) sqrt(pi) eta erfcx(eta),

with eta = gamma / (sqrt(2) k vth_x) and A = T_y/T_x - 1; the electrons are the first species with q/m < 0 and
the ions, every species with q/m > 0, are taken as cold.  A straight fit of ln|Bz(m)| over ``--fit-start`` to
``--fit-end`` is compared with this rate averaged over the same interval.

A magnetic-only seed also excites a standing light wave at the seeded wavenumber, which the scheme does not damp
for th = 0.5.  The cos coefficient of the mode is fitted over ``--light-fit`` with
A exp(gamma t) + B cos(omega t + phi); the light wave is subtracted from it, and the local growth rate is the
slope of the logarithm of the remainder over windows of ``--rate-window``.

Plots, each as a PNG and a PDF beside it:

- the growth plot (``--output``, default RUN_DIR/weibel_growth.png), up to ``--end``, by default the largest
  amplitude of the seeded mode, its saturation.  Left: the amplitude in units of ``WeibelSeedAmplitude`` and the
  growth at the theoretical rate, matched to the fitted growing mode in the middle of ``--light-fit``.  Right: the
  local growth rate and the theory with the measured second moments and with the initial temperatures.
- with the electron pressure, the anisotropy plot (``--anisotropy-plot``, default
  RUN_DIR/weibel_anisotropy.png) over the whole run: <v_y^2>/<v_x^2> and <v_z^2>/<v_x^2>, and the magnetic and
  electric energy of ``diagnostics.csv`` in units of the initial total energy.
- the space-time plots of Bz (``--bz-space-time-plot``, default RUN_DIR/weibel_bz_space_time.png) and, when the
  deck writes E, of Ex (``--ex-space-time-plot``, default RUN_DIR/weibel_ex_space_time.png), each averaged over y and
  z, over the whole box and record, in the layout of the whistler crossing plot.  Both keep the Fourier modes along
  x up to ``--space-time-harmonics`` (4) times ``WeibelSeedModeX``, free of the grid-scale noise.  Left: the field
  along x at seven times spread over the record, each scaled to its own largest value, which stands beside it in
  units of ``WeibelSeedAmplitude``.  Right: the field over x and t, each time divided by its rms over x so that the
  pattern shows from the noise to saturation.  The dashed line is the crest of the seeded Bz mode nearest the middle
  of the box, carried at the speed -(d phase/dt)/k of its phase fitted over the straight fit window; the dots mark
  it at the profiles.  The Weibel mode of a bi-Maxwellian grows without travelling: the line stays on its stripe.
  Ex, the longitudinal field, is thermal noise until the filaments drive its harmonic 2 ``WeibelSeedModeX``.

``FontSizes`` and the line styles below set the look of every plot, that of the two-stream figures.
"""

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # source-checkout imports
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.fields import field_data_exists, open_field_reader, transverse_mean

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.special import erfcx


@dataclass
class FontSizes:
    """Font sizes of every plot, in points."""

    axis_label: float = 22.0
    tick_label: float = 20.0
    legend: float = 18.0


FONT_SIZES = FontSizes()
plt.rcParams.update({
    "axes.labelsize": FONT_SIZES.axis_label,
    "xtick.labelsize": FONT_SIZES.tick_label, "ytick.labelsize": FONT_SIZES.tick_label,
    "legend.fontsize": FONT_SIZES.legend,
})
# TPIC as a thin dark blue line with dots over the theory as a wide light purple one; a third curve dashed.
TPIC_LINE = {"color": "#1f4a9e", "linewidth": 1.3, "marker": "o", "markersize": 4.0, "zorder": 3}
THEORY_LINE = {"color": "#d3a9ee", "linewidth": 4.5, "zorder": 2}
THIRD_LINE = {"color": "C1", "linestyle": "--", "linewidth": 1.6, "zorder": 2.5}
MARKER_EVERY = 4  # a dot on every fourth data point of the TPIC curves


def kinetic_weibel_growth_rate(k_de: float, anisotropy: float, vth_x: float, ions: float) -> float:
    """Return gamma/omega_pe of the Weibel mode k_de, or 0 for a stable one; ions is sum omega_pi^2/omega_pe^2.

    TPIC loads uth as the 1D Gaussian standard deviation sqrt(kT/m), so eta = gamma/(sqrt(2) k vth_x).
    """

    def dispersion(gamma: float) -> float:
        eta = gamma / (math.sqrt(2.0) * k_de * vth_x)
        rhs = anisotropy - (anisotropy + 1.0) * math.sqrt(math.pi) * eta * erfcx(eta)
        return k_de**2 + gamma**2 + ions - rhs

    if anisotropy <= 0.0 or dispersion(1.0e-12) >= 0.0:
        return 0.0
    return brentq(dispersion, 1.0e-12, math.sqrt(anisotropy))


def read_profiles(path: Path, name: str, periodic: tuple[bool, bool, bool]) -> tuple[np.ndarray, np.ndarray]:
    """Return the snapshot times and the field name averaged over y and z, of shape (time, x), at the distinct x
    nodes."""
    times, profiles = [], []
    with open_field_reader(path) as reader:
        for step in range(reader.num_steps()):
            # The last x node repeats the first of the periodic x.
            profiles.append(transverse_mean(reader.read(name, step_selection=[step, 1]), periodic)[:-1])
            times.append(reader.read("time", step_selection=[step, 1])[0])
    return np.asarray(times), np.asarray(profiles)


def read_second_moments(data_dir: Path, species: int, qom: float,
                        periodic: tuple[bool, bool, bool]) -> tuple[np.ndarray, np.ndarray]:
    """Return the snapshot times and the box means of <v_x^2>, <v_y^2>, <v_z^2> of a species, in c^2: walls
    count half, and the last node of the periodic x repeats the first."""
    readers = {name: open_field_reader(data_dir / f"{name}_{species}") for name in ("rho", "PXX", "PYY", "PZZ")}
    count = min(reader.num_steps() for reader in readers.values())
    times, moments = [], []
    for step in range(count):
        mean = lambda name: float(transverse_mean(
            readers[name].read(f"{name}_{species}", step_selection=[step, 1]), periodic)[:-1].mean())
        # rho is 4*pi times the charge density and P_ii the mass density times <v_i^2>.
        mass = abs(mean("rho")) / (4.0 * math.pi * abs(qom))
        times.append(readers["rho"].read("time", step_selection=[step, 1])[0])
        moments.append([mean(name) / mass for name in ("PXX", "PYY", "PZZ")])
    return np.asarray(times), np.asarray(moments)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=200)
    plt.close(fig)


def write_growth_plot(path: Path, t: np.ndarray, amplitude: np.ndarray, theory_t: np.ndarray, theory: np.ndarray,
                      matched_amplitude: float, anchor: float, centres: np.ndarray, rate: np.ndarray,
                      theory_initial: float, mode_x: int, end: float) -> None:
    shown, shown_theory, shown_rate = t <= end, theory_t <= end, centres <= end
    integral = np.concatenate(([0.0], np.cumsum(0.5 * (theory[1:] + theory[:-1]) * np.diff(theory_t))))
    matched = int(np.argmin(np.abs(theory_t - anchor)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(16.0, 5.0), constrained_layout=True)
    left.plot(t[shown], amplitude[shown], label="TPIC", markevery=MARKER_EVERY, **TPIC_LINE)
    left.plot(theory_t[shown_theory], matched_amplitude * np.exp(integral - integral[matched])[shown_theory],
              label="Kinetic theory", **THEORY_LINE)
    left.set_yscale("log")
    left.set_ylabel(rf"$|\hat B_z(m={mode_x})|\,/\,\delta B$")
    left.legend(loc="best")

    right.plot(centres[shown_rate], rate[shown_rate], label="TPIC", markevery=MARKER_EVERY, **TPIC_LINE)
    right.plot(theory_t[shown_theory], theory[shown_theory], label="Kinetic theory", **THEORY_LINE)
    right.axhline(theory_initial, label=r"Kinetic theory, $t=0$", **THIRD_LINE)
    right.set_ylabel(r"$\gamma/\omega_{pe}$")
    right.set_ylim(0.0, 1.4 * theory_initial)
    # Below the plateau, clear of the rise at early times and of the fall at saturation.
    right.legend(loc="lower left", bbox_to_anchor=(0.1, 0.0))
    for axis in (left, right):
        axis.set_xlabel(r"$\omega_{pe} t$")
        axis.set_xlim(0.0, end)
        axis.grid(True, alpha=0.3)
    save(fig, path)


def write_anisotropy_plot(path: Path, moment_t: np.ndarray, moments: np.ndarray, diagnostics: Path,
                          omega_pe: float, end: float) -> None:
    rows = list(csv.reader(line for line in diagnostics.open() if not line.startswith("#")))
    column = {name: index for index, name in enumerate(rows[0])}
    table = np.array([[float(row[column[name]]) for name in ("time", "magnetic_energy", "electric_energy",
                                                              "kinetic_energy")] for row in rows[1:]])
    shown = omega_pe * table[:, 0] <= end
    total = table[0, 1:].sum()
    fig, (left, right) = plt.subplots(1, 2, figsize=(16.0, 8.0), constrained_layout=True)
    left.plot(moment_t, moments[:, 1] / moments[:, 0], color="#2a78d6", linewidth=2.0,
              label=r"$\langle v_y^2\rangle/\langle v_x^2\rangle$")
    left.plot(moment_t, moments[:, 2] / moments[:, 0], color="#1baf7a", linewidth=2.0,
              label=r"$\langle v_z^2\rangle/\langle v_x^2\rangle$")
    left.set_ylabel("electron second-moment ratio")
    left.set_ylim(0.0, 1.125 * max(1.0, float(np.max(moments[:, 1] / moments[:, 0]))))
    left.legend(loc="center left")
    right.semilogy(omega_pe * table[shown, 0], table[shown, 1] / total, color="#2a78d6", linewidth=2.0,
                   label="magnetic")
    right.semilogy(omega_pe * table[shown, 0], table[shown, 2] / total, color="#eb6834", linewidth=2.0,
                   label="electric")
    right.set_ylabel("field energy / initial total energy")
    right.legend(loc="lower right")
    for axis in (left, right):
        axis.set_xlabel(r"$\omega_{pe}t$")
        axis.set_xlim(0.0, end)
        axis.grid(True, alpha=0.3)
    save(fig, path)


# The space-time plot: profiles at times spread over the record, each in the middle of its share.
SPACE_TIME_PROFILES = 7


def write_space_time_plot(path: Path, t: np.ndarray, profiles: np.ndarray, field: str, seed: float,
                          max_mode: int, length_de: float, crest: float, crest_t: float, speed: float) -> None:
    """Write the space-time plot of profiles, the field named field in TeX, with the Bz crest at crest_t."""
    # The grid-scale noise, of Ex above all, would hide the pattern.
    spectrum = np.fft.rfft(profiles, axis=1)
    spectrum[:, max_mode + 1:] = 0.0
    profiles = np.fft.irfft(spectrum, n=profiles.shape[1], axis=1)
    rows = np.round((np.arange(SPACE_TIME_PROFILES) + 0.5) / SPACE_TIME_PROFILES * (t.size - 1)).astype(int)
    # The plots reach x = length, the image of x = 0.
    x = np.linspace(0.0, length_de, profiles.shape[1] + 1)
    closed = np.concatenate((profiles, profiles[:, :1]), axis=1)
    crest_at = lambda at: crest + speed * (at - crest_t)
    ends = t[[0, -1]]

    fig, (stack, space_time) = plt.subplots(1, 2, figsize=(16.0, 8.0), sharey=True, constrained_layout=True)
    gap = t[rows[1]] - t[rows[0]]
    for row in rows:
        largest = np.abs(closed[row]).max()
        stack.axhline(t[row], color="0.8", linewidth=0.8, zorder=0)
        stack.plot(x, t[row] + 0.45 * gap * closed[row] / largest, color="#2a78d6", linewidth=2.0)
        # Each profile fills its own share: its largest value stands beside it.
        value = f"{largest / seed:.0f}" if largest >= 100.0 * seed else f"{largest / seed:.2g}"
        stack.text(1.02, t[row], rf"${value}\,\delta B$", transform=stack.get_yaxis_transform(), va="center",
                   fontsize=FONT_SIZES.legend)
    stack.plot(crest_at(t[rows]), t[rows], "o", color="black", markersize=8)
    stack.set_ylabel(r"$\omega_{pe} t$")

    # A field still zero (E at t = 0) stays zero.
    rms = np.sqrt(np.mean(profiles**2, axis=1, keepdims=True))
    level = np.divide(closed, rms, out=np.zeros_like(closed), where=rms > 0.0)
    # The noise peaks of the early profiles would wash out the stripes: they saturate the colors.
    limit = np.percentile(np.abs(level), 99.0)
    mesh = space_time.pcolormesh(x, t, level, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest",
                                 rasterized=True)
    fig.colorbar(mesh, ax=space_time, label=rf"$\bar {field}\,/\,\mathrm{{rms}}_x\,\bar {field}$")
    for row in rows:
        space_time.axhline(t[row], color="black", linewidth=0.8, linestyle=(0, (2, 4)))
    for axis in (stack, space_time):
        axis.plot(crest_at(ends), ends, color="black", linewidth=2.5, linestyle=(0, (5, 3)))
        axis.set_xlabel(r"$x/d_e$")
        axis.set_xlim(0.0, length_de)
        axis.set_ylim(*ends)
    save(fig, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Growth of the seeded Weibel mode against kinetic theory.")
    parser.add_argument("--run-dir", type=Path, default=Path("."),
                        help="simulation launch/run directory (default: current directory)")
    parser.add_argument("--input", type=Path, default=None, help="input deck (default: RUN_DIR/weibel2D.inp)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="snapshot directory (default: RUN_DIR/SaveDirName of the deck)")
    parser.add_argument("--fit-start", type=float, default=60.0,
                        help="start of the straight fit of ln|Bz(m)|, in omega_pe*t (default: 60)")
    parser.add_argument("--fit-end", type=float, default=160.0,
                        help="end of the straight fit, in omega_pe*t, clipped to the record (default: 160)")
    parser.add_argument("--light-fit", type=float, nargs=2, default=(30.0, 170.0), metavar=("START", "END"),
                        help="omega_pe*t interval of the growing-mode and light-wave fit, clipped to the record "
                             "(default: 30 170)")
    parser.add_argument("--rate-window", type=float, default=10.0,
                        help="width in 1/omega_pe of the windows of the local growth rate (default: 10)")
    parser.add_argument("--end", type=float, default=None,
                        help="last omega_pe*t of the growth plot (default: the largest amplitude of the mode)")
    parser.add_argument("--output", type=Path, default=None,
                        help="growth plot path (default: RUN_DIR/weibel_growth.png); a PDF is written beside it")
    parser.add_argument("--anisotropy-plot", type=Path, default=None,
                        help="anisotropy plot path (default: RUN_DIR/weibel_anisotropy.png); a PDF is written "
                             "beside it")
    parser.add_argument("--bz-space-time-plot", type=Path, default=None,
                        help="Bz space-time plot path (default: RUN_DIR/weibel_bz_space_time.png); a PDF is written "
                             "beside it")
    parser.add_argument("--ex-space-time-plot", type=Path, default=None,
                        help="Ex space-time plot path (default: RUN_DIR/weibel_ex_space_time.png); a PDF is written "
                             "beside it")
    parser.add_argument("--space-time-harmonics", type=int, default=4,
                        help="the space-time plots keep the Fourier modes along x up to this multiple of "
                             "WeibelSeedModeX; 0 keeps every mode (default: 4)")
    return parser


def analyze(args: argparse.Namespace) -> None:
    input_path = args.input or (args.run_dir / "weibel2D.inp")
    params = read_input(input_path)
    periodic = periodic_axes(params)
    if not periodic[0]:
        raise ValueError(f"{input_path}: the seeded mode is a Fourier mode along x: the box must be periodic in x")
    species = range(int(float(params["ns"][0])))
    qom = [float(params["qom"][s]) for s in species]
    rho = [float(params["rhoINIT"][s]) for s in species]
    electron = next((s for s in species if qom[s] < 0.0), None)
    if electron is None:
        raise ValueError(f"{input_path}: no electron species (qom < 0)")
    omega_pe2 = rho[electron] * abs(qom[electron])
    omega_pe = math.sqrt(omega_pe2)
    ions = sum(rho[s] * qom[s] for s in species if qom[s] > 0.0) / omega_pe2
    mode_x = int(float(params["WeibelSeedModeX"][0]))
    k_de = 2.0 * math.pi * mode_x / float(params["Lx"][0]) / omega_pe
    seed = float(params["WeibelSeedAmplitude"][0])
    uth, vth = float(params["uth"][electron]), float(params["vth"][electron])
    theory_initial = kinetic_weibel_growth_rate(k_de, (vth / uth) ** 2 - 1.0, uth, ions)
    data_dir = args.data_dir or args.run_dir / params["SaveDirName"][0]

    time, profiles = read_profiles(data_dir / "B", "Bz", periodic)
    t = omega_pe * time
    coefficient = np.fft.fft(profiles, axis=1)[:, mode_x] / profiles.shape[1]
    amplitude = 2.0 * np.abs(coefficient) / seed
    cosine = 2.0 * coefficient.real
    measured = all(field_data_exists(data_dir, f"{name}_{electron}") for name in ("rho", "PXX", "PYY", "PZZ"))
    if measured:
        moment_time, moments = read_second_moments(data_dir, electron, qom[electron], periodic)
        theory_t = omega_pe * moment_time
        theory = np.array([kinetic_weibel_growth_rate(k_de, vy / vx - 1.0, math.sqrt(vx), ions)
                           for vx, vy, _ in moments])
    else:
        theory_t, theory = t, np.full(t.size, theory_initial)

    fit_end = min(args.fit_end, t[-1])
    window = (t >= args.fit_start) & (t <= fit_end)
    if np.count_nonzero(window) < 3:
        raise ValueError(f"the fit window [{args.fit_start:g}, {fit_end:g}] holds fewer than 3 snapshots")
    gamma_fit = np.polyfit(t[window], np.log(amplitude[window]), 1)[0]
    theory_window = float(np.interp(t[window], theory_t, theory).mean())
    # The mode is 2|c| cos(k x + phase): its crests move at -d(phase)/dt / k, in d_e*omega_pe = c.
    phase_t = 0.5 * (args.fit_start + fit_end)
    phase_line = np.polyfit(t[window] - phase_t, np.unwrap(np.angle(coefficient[window])), 1)
    crest_speed = -phase_line[0] / k_de

    # The growing mode and the light wave; the local rate of the cos coefficient without the light wave.
    light_start, light_end = args.light_fit[0], min(args.light_fit[1], t[-1])
    light = (t >= light_start) & (t <= light_end)
    model = lambda tt, a, g, b, w, phi: a * np.exp(g * tt) + b * np.cos(w * tt + phi)
    p, _ = curve_fit(model, t[light], cosine[light], maxfev=40000,
                     p0=(cosine[0] / 3.0, theory_initial, cosine[0] / 2.0, math.sqrt(1.0 + k_de**2), 0.0))
    growing = cosine - p[2] * np.cos(p[3] * t + p[4])
    half = 0.5 * args.rate_window
    centres = np.arange(t[0] + half, t[-1] - half + 1.0e-9, 1.0)
    rate = np.array([np.polyfit(t[np.abs(t - x) <= half], np.log(np.abs(growing[np.abs(t - x) <= half])), 1)[0]
                     for x in centres])
    anchor = 0.5 * (light_start + light_end)
    end = args.end if args.end is not None else float(t[int(np.argmax(amplitude))])

    growth_path = args.output or (args.run_dir / "weibel_growth.png")
    write_growth_plot(growth_path, t, amplitude, theory_t, theory, p[0] * math.exp(p[1] * anchor) / seed, anchor,
                      centres, rate, theory_initial, mode_x, end)
    anisotropy_path = args.anisotropy_plot or (args.run_dir / "weibel_anisotropy.png")
    if measured:
        write_anisotropy_plot(anisotropy_path, theory_t, moments, data_dir / "diagnostics.csv", omega_pe, t[-1])
    # The crest of the mode nearest the middle of the box at phase_t.
    length_de, wavelength = float(params["Lx"][0]) * omega_pe, 2.0 * math.pi / k_de
    crest = np.mod(-phase_line[1] / k_de, wavelength)
    crest += wavelength * round((0.5 * length_de - crest) / wavelength)
    bz_path = args.bz_space_time_plot or (args.run_dir / "weibel_bz_space_time.png")
    max_mode = args.space_time_harmonics * mode_x if args.space_time_harmonics > 0 else profiles.shape[1] // 2
    write_space_time_plot(bz_path, t, profiles, "B_z", seed, max_mode, length_de, crest, phase_t, crest_speed)
    with_e = field_data_exists(data_dir, "E")
    ex_path = args.ex_space_time_plot or (args.run_dir / "weibel_ex_space_time.png")
    if with_e:
        e_time, e_profiles = read_profiles(data_dir / "E", "Ex", periodic)
        write_space_time_plot(ex_path, omega_pe * e_time, e_profiles, "E_x", seed, max_mode, length_de, crest,
                              phase_t, crest_speed)

    peak = int(np.argmax(amplitude))
    print(f"input deck                     = {input_path}")
    print(f"data directory                 = {data_dir}")
    print(f"seeded mode m, k*d_e           = {mode_x}, {k_de:.6f}")
    print(f"snapshots, omega_pe*t range    = {t.size}, [{t[0]:g}, {t[-1]:g}]")
    print(f"theory anisotropy from         = {'the measured second moments' if measured else 'the deck temperatures'}")
    print(f"theory gamma/omega_pe at t = 0 = {theory_initial:.6f} (cold ions {ions:g} omega_pe^2)")
    print(f"fit of ln|Bz(m)| over omega_pe*t [{args.fit_start:g}, {fit_end:g}] = {gamma_fit:.6f}, theory mean "
          f"{theory_window:.6f} ({100.0 * (gamma_fit / theory_window - 1.0):+.2f} %)")
    print(f"light wave omega/omega_pe, amplitude/seed = {p[3]:.4f}, {abs(p[2]) / seed:.3f} "
          f"(fit over [{light_start:g}, {light_end:g}])")
    print(f"largest amplitude / seed       = {amplitude[peak]:.4g} at omega_pe*t = {t[peak]:g}")
    print(f"crest speed over the fit window = {crest_speed:.3e} c, {crest_speed * (t[-1] - t[0]):+.4f} d_e over "
          f"the record (cell {length_de / profiles.shape[1]:.4f} d_e)")
    print(f"growth plot written to         = {growth_path} and its PDF")
    if measured:
        print(f"anisotropy plot written to     = {anisotropy_path} and its PDF")
    print(f"Bz space-time plot written to  = {bz_path} and its PDF")
    if with_e:
        print(f"Ex space-time plot written to  = {ex_path} and its PDF")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        analyze(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
