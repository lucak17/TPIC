#!/usr/bin/env python3
"""TPIC against the explicit electrostatic PIC of ``tools/explicit_es_pic.py`` on the same run.

Usage from the repository root::

    python3 tools/explicit_es_pic.py --run-dir RUN_DIR
    python3 tools/postprocessing/analyze_explicit_comparison.py --run-dir RUN_DIR

The run directory holds the deck (``RUN_DIR/twoStream1D.inp`` unless ``--input`` names another), the TPIC run
launched from it (its ``SaveDirName``, or ``--data-dir``: ``diagnostics.csv`` and the particle snapshots) and the
explicit run of the same deck (``explicit_pic_energy.csv`` and ``explicit_pic_particles.npz``), which starts from the
TPIC particles of cycle 0.  Times are in ``omega_pe*t``, with omega_pe of all the electrons.

Quantities.  The energy error is |E_tot(t) - E_tot(0)|/E_tot(0) of the total energy: that of ``diagnostics.csv`` for
TPIC, the field plus kinetic energy for the explicit run (drifting ions, when present, carry most of E_tot(0)).  The
electron Debye length lambda_D = v_th/omega_pe follows from the electron thermal energy K_th, the kinetic energy
without the mean x drift of each electron species: v_th^2 = 2 K_th/M with M the electron mass, the spread of the
velocities about their species drift (TPIC's kinetic energy also counts v_y and v_z; in a growing wave it includes the
coherent velocity of the wave).  It is shown in cells, lambda_D/dx.

Plots, each as a PNG and a PDF named after the prefix ``--output`` (default ``RUN_DIR/explicit_comparison``):

- ``_energy``: the energy error (solid, left axis) and lambda_D/dx (dashed, right axis) of both codes; a zero error is
  drawn at ``ENERGY_FLOOR`` and the dotted line is lambda_D = dx.
- ``_distribution``: the electron velocity distribution f(vx) of both codes at t = 0 and at the last particle
  snapshot that both runs hold (``--end`` picks the snapshot nearest to an omega_pe*t).
- ``_phase_space``: the electrons in (x, vx) of both codes at the same two times, overlapped, one dot per particle.

``FontSizes`` and the colors below set the look of every plot.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # source-checkout imports
from tpic_tools.inputs import read_input
from tpic_tools.particles import read_polydata
from tpic_tools.runs import read_csv

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FontSizes:
    """Font sizes of every plot, in points."""

    axis_label: float = 22.0
    tick_label: float = 20.0
    legend: float = 16.0
    title: float = 20.0


FONT_SIZES = FontSizes()
plt.rcParams.update({
    "axes.labelsize": FONT_SIZES.axis_label, "axes.titlesize": FONT_SIZES.title,
    "xtick.labelsize": FONT_SIZES.tick_label, "ytick.labelsize": FONT_SIZES.tick_label,
    "legend.fontsize": FONT_SIZES.legend,
})
# The look of the two-stream figures: TPIC as a thin dark blue line with dots over the explicit PIC as a wide light
# purple one, which stays visible around it.
TPIC_LINE = {"color": "#1f4a9e", "linewidth": 1.3, "marker": "o", "markersize": 6.0, "zorder": 3}
EXPLICIT_LINE = {"color": "#d3a9ee", "linewidth": 4.5, "zorder": 2}
MARKERS_PER_CURVE = 25
ENERGY_FLOOR = 1.0e-17  # where a zero energy error is drawn on the logarithmic axis
# Phase-space dots, in points^2: the explicit run under larger dots, so that both codes show where they coincide.
TPIC_DOT_SIZE, EXPLICIT_DOT_SIZE, DOT_ALPHA = 3.0, 6.0, 0.5


def value(params: dict, key: str, index: int = 0) -> float:
    return float(params[key][index])


def debye_cells(kinetic: list[np.ndarray], momentum: list[np.ndarray], mass: list[np.ndarray],
                omega_pe: float, dx: float) -> np.ndarray:
    """Return lambda_D/dx of the electrons from the kinetic energy, x momentum and mass of each electron species."""
    heat = sum(k - 0.5 * p * p / m for k, p, m in zip(kinetic, momentum, mass))
    # Round-off can make the thermal energy of a cold species slightly negative.
    return np.sqrt(np.maximum(2.0 * heat / sum(mass), 0.0)) / (omega_pe * dx)


def tpic_electrons(data: Path, cycle: int, electrons: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return x and vx of the electrons of every rank piece of the TPIC particle snapshot of a cycle."""
    x, v = [], []
    for piece in sorted(data.glob(f"particles_t{cycle:06d}_r*.vtp")):
        arrays = read_polydata(piece, ("position", "velocity", "species"))
        chosen = np.isin(arrays["species"], electrons)
        x.append(arrays["position"].reshape(-1, 3)[chosen, 0])
        v.append(arrays["velocity"].reshape(-1, 3)[chosen, 0])
    return np.concatenate(x), np.concatenate(v)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for target in (path, path.with_suffix(".pdf")):
        fig.savefig(target, dpi=200)
    plt.close(fig)


def write_energy_plot(path: Path, tpic: dict, explicit: dict, end: float) -> None:
    fig, error = plt.subplots(figsize=(11.0, 6.5), constrained_layout=True)
    length = error.twinx()
    for run, style, label in ((tpic, TPIC_LINE, "TPIC"), (explicit, EXPLICIT_LINE, "explicit PIC")):
        every = max(1, run["t"].size // MARKERS_PER_CURVE)
        error.semilogy(run["t"], np.maximum(run["error"], ENERGY_FLOOR), markevery=every,
                       label=f"{label}, energy error", **style)
        length.semilogy(run["t"], np.where(run["debye"] > 0.0, run["debye"], np.nan), linestyle="--",
                        markevery=every, label=rf"{label}, $\lambda_D/\Delta x$", **style)
    length.axhline(1.0, color="black", linewidth=1.2, linestyle=":", label=r"$\lambda_D=\Delta x$")
    error.set_xlabel(r"$\omega_{pe} t$")
    error.set_ylabel(r"$|E_{\mathrm{tot}}(t)-E_{\mathrm{tot}}(0)|\,/\,E_{\mathrm{tot}}(0)$")
    length.set_ylabel(r"$\lambda_D\,/\,\Delta x$")
    error.set_xlim(0.0, end)
    # A decade below the lowest values, so that the flat lines of the two axes do not lie on one another.
    error.set_ylim(bottom=ENERGY_FLOOR / 10.0)
    smallest = min(np.min(run["debye"][run["debye"] > 0.0], initial=np.inf) for run in (tpic, explicit))
    if np.isfinite(smallest):
        length.set_ylim(bottom=smallest / 10.0)
    error.grid(True, alpha=0.3)
    handles = error.get_legend_handles_labels()[0] + length.get_legend_handles_labels()[0]
    error.legend(handles, [h.get_label() for h in handles], loc="center right", framealpha=0.85)
    save(fig, path)


def write_distribution_plot(path: Path, samples: list, edges: np.ndarray) -> None:
    centres = 0.5 * (edges[1:] + edges[:-1])
    densities = [[np.histogram(v, bins=edges, density=True)[0] for _, v in (tpic, explicit)]
                 for _, tpic, explicit in samples]
    # Empty bins sit below the axis, so that the steps of a single-bin beam still show.
    bottom = 0.5 * min(f[f > 0.0].min() for pair in densities for f in pair)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0), sharey=True, constrained_layout=True)
    for axis, (t, _, _), pair in zip(axes, samples, densities):
        for f, style, label in zip(pair, (TPIC_LINE, EXPLICIT_LINE), ("TPIC", "explicit PIC")):
            filled = np.flatnonzero(f > 0.0)
            axis.plot(centres, np.maximum(f, 0.1 * bottom), drawstyle="steps-mid", label=label,
                      markevery=list(filled[::max(1, filled.size // MARKERS_PER_CURVE)]), **style)
        axis.set_yscale("log")
        axis.set_ylim(bottom=bottom)
        axis.set_title(rf"$\omega_{{pe}} t = {t:.0f}$")
        axis.set_xlabel(r"$v_x/c$")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$f(v_x)$")
    axes[0].legend(loc="upper left")
    save(fig, path)


def write_phase_space_plot(path: Path, samples: list, length_de: float, omega_pe: float, limit: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0), sharey=True, constrained_layout=True)
    for axis, (t, tpic, explicit) in zip(axes, samples):
        for (x, v), color, size, label in ((explicit, EXPLICIT_LINE["color"], EXPLICIT_DOT_SIZE, "explicit PIC"),
                                           (tpic, TPIC_LINE["color"], TPIC_DOT_SIZE, "TPIC")):
            axis.scatter(omega_pe * x, v, s=size, color=color, alpha=DOT_ALPHA, linewidths=0.0, rasterized=True,
                         label=label)
        axis.set_title(rf"$\omega_{{pe}} t = {t:.0f}$")
        axis.set_xlabel(r"$x/d_e$")
        axis.set_xlim(0.0, length_de)
        axis.set_ylim(-limit, limit)
    axes[0].set_ylabel(r"$v_x/c$")
    axes[0].legend(loc="upper right", markerscale=8, framealpha=0.85)
    save(fig, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TPIC against the explicit electrostatic PIC on one run.")
    parser.add_argument("--run-dir", type=Path, default=Path("."),
                        help="simulation launch/run directory (default: current directory)")
    parser.add_argument("--input", type=Path, default=None, help="input deck (default: RUN_DIR/twoStream1D.inp)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="TPIC output directory (default: RUN_DIR/SaveDirName of the deck)")
    parser.add_argument("--end", type=float, default=None,
                        help="omega_pe*t of the last snapshot shown (default: the last one both runs hold)")
    parser.add_argument("--output", type=Path, default=None,
                        help="prefix of the plots (default: RUN_DIR/explicit_comparison); _energy, _distribution "
                             "and _phase_space are added, each written as a PNG and a PDF")
    return parser


def analyze(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    params = read_input(args.input or run_dir / "twoStream1D.inp")
    data = args.data_dir or run_dir / params["SaveDirName"][0]
    electrons = [s for s in range(int(value(params, "ns"))) if value(params, "qom", s) < 0.0]
    omega_pe = math.sqrt(sum(value(params, "rhoINIT", s) * abs(value(params, "qom", s)) for s in electrons))
    dx, dt = value(params, "Lx") / value(params, "nxc"), value(params, "dt")

    rows = read_csv(data)
    column = lambda name: np.array([float(row[name]) for row in rows])
    tpic_mass = [column(f"charge_{s}") / value(params, "qom", s) for s in electrons]
    tpic_total = column("total_energy")
    tpic = {"t": omega_pe * column("time"), "error": np.abs(tpic_total / tpic_total[0] - 1.0),
            "debye": debye_cells([column(f"kinetic_energy_{s}") for s in electrons],
                                 [column(f"momentum_x_{s}") for s in electrons], tpic_mass, omega_pe, dx)}
    with (run_dir / "explicit_pic_energy.csv").open() as stream:
        table = list(csv.DictReader(stream))
    explicit_column = lambda name: np.array([float(row[name]) for row in table])
    explicit_total = explicit_column("field") + explicit_column("kinetic")
    explicit = {"t": explicit_column("omega_pe_t"),
                "error": np.abs(explicit_total / explicit_total[0] - 1.0),
                "debye": debye_cells([explicit_column(f"kinetic_{s}") for s in electrons],
                                     [explicit_column(f"momentum_{s}") for s in electrons],
                                     [explicit_column(f"mass_{s}") for s in electrons], omega_pe, dx)}

    snapshots = np.load(run_dir / "explicit_pic_particles.npz")
    explicit_cycles = {int(key.split("_")[2]) for key in snapshots.files if key.startswith(f"x_{electrons[0]}_")}
    tpic_cycles = {int(re.search(r"particles_t(\d+)_r", p.name).group(1)) for p in data.glob("particles_t*_r*.vtp")}
    common = sorted(explicit_cycles & tpic_cycles)
    if not common or common[0] != 0:
        raise ValueError(f"{data}: TPIC and the explicit run share no particle snapshot of cycle 0")
    last = common[-1] if args.end is None else min(common, key=lambda c: abs(omega_pe * dt * c - args.end))
    samples = []
    for cycle in (0, last):
        ours = (np.concatenate([snapshots[f"x_{s}_{cycle}"] for s in electrons]),
                np.concatenate([snapshots[f"v_{s}_{cycle}"] for s in electrons]))
        samples.append((omega_pe * dt * cycle, tpic_electrons(data, cycle, electrons), ours))
    limit = 1.05 * max(np.abs(v).max() for _, a, b in samples for _, v in (a, b))
    edges = np.linspace(-limit, limit, 82)

    prefix = args.output or run_dir / "explicit_comparison"
    paths = [prefix.with_name(prefix.name + suffix + ".png") for suffix in ("_energy", "_distribution", "_phase_space")]
    write_energy_plot(paths[0], tpic, explicit, max(tpic["t"][-1], explicit["t"][-1]))
    write_distribution_plot(paths[1], samples, edges)
    write_phase_space_plot(paths[2], samples, value(params, "Lx") * omega_pe, omega_pe, limit)

    t_end, (_, (_, tpic_v), (_, explicit_v)) = samples[1][0], samples[1]
    at = lambda run, t: int(np.argmin(np.abs(run["t"] - t)))
    print(f"run directory                  = {run_dir}")
    print(f"electrons TPIC, explicit       = {tpic_v.size}, {explicit_v.size}")
    print(f"omega_pe*t of the snapshots    = 0, {t_end:g}")
    for label, run in (("TPIC", tpic), ("explicit PIC", explicit)):
        i = at(run, t_end)
        print(f"{label + ' energy error':31s}= {run['error'][i]:.3e} of E_tot(0) at omega_pe*t = {run['t'][i]:g} "
              f"(largest {run['error'].max():.3e})")
        print(f"{label + ' lambda_D/dx':31s}= {run['debye'][i]:.3g} at omega_pe*t = {run['t'][i]:g}")
    print(f"rms v_x at the end             = TPIC {np.std(tpic_v):.4f}, explicit PIC {np.std(explicit_v):.4f}")
    for path in paths:
        print(f"plot written to                = {path} and its PDF")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        analyze(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
