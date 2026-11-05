#!/usr/bin/env python3
"""Compare a GaussianExpansion run with the self-similar expansion of a neutral Gaussian plasma.

A neutral, pair-loaded Gaussian plasma released into vacuum stays Gaussian along every axis with a positive
GaussianExpansionSigma* (Dorozhkina & Semenov, JETP Lett. 67, 573 (1998) for cold ions and a non-spherical
bunch, Phys. Rev. Lett. 81, 2691 (1998) for kinetic, warm ions; Kovalev & Bychenkov, Phys. Rev. Lett. 90, 185004
(2003); docs/user/gaussian_expansion.md). Along such an axis a, with M_s the mass of species s
and v_sa its deck thermal speed,

    sigma_a(t)^2 = sigma_a0^2 (1 + t^2/tau_a^2),  tau_a = sigma_a0/c_a,  c_a^2 = sum_s M_s v_sa^2 / sum_s M_s,

the velocity component cools as 1/(1 + t^2/tau_a^2) and gains the flow u_a = x_a t/(t^2 + tau_a^2), and the
ambipolar field is E_a = x_a (c_a^2 - v_ea^2)/(qom_e sigma_a0^2 (1 + t^2/tau_a^2)^2). Components along a uniform
axis keep their temperature. The number D of Gaussian axes enters the centre density, prod_a sigma_a0/sigma_a,
and the share of the energy the ions gain.

Usage from the repository root:

    python3 tools/postprocessing/analyze_gaussianExpansion.py --run-dir /path/to/run
    python3 tools/postprocessing/analyze_gaussianExpansion.py --input input/gaussianExpansion3D.inp --tables

The deck is the single GaussianExpansion deck of --run-dir unless --input names it. The analyzer reads
diagnostics.csv, gauss.csv, the VTK field files rho_<s>, J_<s>, PXX_<s>, PYY_<s>, PZZ_<s>, E, B, and the particle
snapshots when present. TPIC's output moments are rho = 4*pi*sum(q W)/V, J = sum(q v W)/V and the mass-weighted
P_ij = sum((q/qom) v_i v_j W)/V. The widths remove the linear-deposition broadening dx^2/6, and the analytic
curves use the widths measured at t = 0 (the cutoff narrows the deck's). Into --output-dir (default the run
directory) go, each as a PNG and a PDF,

    <prefix>_overview        width, centre density, flow, temperatures and energy share against time
    <prefix>_width_density   the width (left axis) and the centre density (right axis) of the ions against time,
                             as a print figure
    <prefix>_energy_share    the electron and ion kinetic energies of <prefix>_overview alone, as a print figure
    <prefix>_density_maps    with two or more Gaussian axes: the ion density in the Gaussian plane at four times,
                             with the analytic 1 and 2 sigma
    <prefix>_density_3d      with --density-3d (three Gaussian axes, needs pyvista): the ion density in 3D at
                             t/tau = 0, 1.5 and 3, cut open towards the camera, with the analytic 1 and 2 sigma
    <prefix>_profiles        self-similar density profiles, radial flow and ambipolar field
    <prefix>_radial_flow     the radial ion flow of <prefix>_profiles alone, as a print figure
    <prefix>_conservation    energy change, field energies, momentum and the Gauss-law residual
    <prefix>_energy_gauss    (E - E0)/E0 on the left axis and the absolute Gauss-law residual rms on the right
    <prefix>_ions            ion velocity and energy distributions of the particle snapshots
    <prefix>_phase_space     ion x-v_x phase space of the particle snapshots, with the analytic flow

and <prefix>_snapshots.csv (measured and analytic values of every field snapshot) and <prefix>_summary.txt (the
derived parameters and the acceptance checks of the case document, also printed). --tables prints the analytic solution of the
deck alone.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # source-checkout imports
from tpic_tools.fields import field_data_exists, open_field_reader
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.particles import read_polydata
from tpic_tools.runs import read_csv

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import numpy as np

FOUR_PI = 4.0 * math.pi
AXES = "xyz"


@dataclass
class FontSizes:
    """Font sizes of the text of a figure, in points."""

    title: float = 13.0       # figure and panel titles
    axis_label: float = 12.0  # axis and colour-bar labels
    tick_label: float = 10.5  # tick labels, colour bars included
    legend: float = 9.0

    def rc(self) -> dict:
        return {"figure.titlesize": self.title, "axes.titlesize": self.title, "axes.labelsize": self.axis_label,
                "xtick.labelsize": self.tick_label, "ytick.labelsize": self.tick_label, "legend.fontsize": self.legend}


@dataclass
class Density3DFontSizes(FontSizes):
    """Font sizes of the 3D density figure; the box tick labels and axis names are drawn over the rendered images."""

    box_tick_label: float = 8.5
    box_axis_label: float = 10.0


# The overview, density maps, profiles, conservation, ions and phase space figures.
FONT_SIZES = FontSizes()
# The print figures width_density, energy_share, radial_flow and energy_gauss, in the look of the line plots of
# analyze_reconnection.py.
PRINT_FONT_SIZES = FontSizes(title=22.0, axis_label=22.0, tick_label=20.0, legend=18.0)
DENSITY_3D_FONT_SIZES = Density3DFontSizes(title=22.0, axis_label=22.0, tick_label=19.0, box_tick_label=15.0,
                                           box_axis_label=19.0)
# Measured values carry the axis or species color; analytic curves are black.
AXIS_COLORS = ("#1f4a9e", "#c0392b", "#2e8b57")
ELECTRON_COLOR, ION_COLOR = "#1f4a9e", "#c0392b"
ANALYTIC = dict(color="black", linestyle="--", linewidth=1.3)
# The print figures also have a closed axes box and a framed legend.
PRINT_STYLE = {**PRINT_FONT_SIZES.rc(), "axes.spines.top": True, "axes.spines.right": True, "legend.frameon": True}

plt.rcParams.update({
    **FONT_SIZES.rc(),
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "lines.markersize": 5.0, "legend.frameon": False,
    "figure.constrained_layout.use": True, "savefig.dpi": 200,
})


# ---------------------------------------------------------------------------- deck and theory


@dataclass
class Deck:
    path: Path
    save_dir: str
    dt: float
    ncycles: int
    lengths: np.ndarray
    cells: np.ndarray
    periodic: list[bool]
    qom: np.ndarray
    rho_init: np.ndarray
    thermal: np.ndarray  # (species, axis) deck thermal speeds
    center: np.ndarray
    sigma: np.ndarray

    @property
    def spacing(self) -> np.ndarray:
        return self.lengths / self.cells

    @property
    def gaussian(self) -> list[int]:
        return [axis for axis in range(3) if self.sigma[axis] > 0]

    @property
    def electron(self) -> int:
        return int(np.flatnonzero(self.qom < 0)[0])

    @property
    def ion(self) -> int:
        return int(np.flatnonzero(self.qom > 0)[0])


def read_deck(path: Path) -> Deck:
    values = read_input(path)
    if values.get("Case", [""])[0] != "GaussianExpansion":
        raise ValueError(f"{path}: Case is not GaussianExpansion")
    ns = int(values["ns"][0])
    species = lambda key: np.array([float(token) for token in values[key][:ns]])
    lengths = np.array([float(values[f"L{a}"][0]) for a in AXES])
    return Deck(
        path=path, save_dir=values.get("SaveDirName", ["data"])[0], dt=float(values["dt"][0]),
        ncycles=int(values["ncycles"][0]), lengths=lengths,
        cells=np.array([int(values[f"n{a}c"][0]) for a in AXES]),
        periodic=list(periodic_axes(values)),
        qom=species("qom"), rho_init=species("rhoINIT"),
        thermal=np.stack([species("uth"), species("vth"), species("wth")], axis=1),
        center=np.array([float(values.get(f"GaussianExpansionCenter{a.upper()}", [0.5 * lengths[i]])[0])
                         for i, a in enumerate(AXES)]),
        sigma=np.array([float(values.get(f"GaussianExpansionSigma{a.upper()}", ["0"])[0]) for a in AXES]))


def find_input(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    decks = [path for path in sorted(run_dir.glob("*.inp"))
             if read_input(path).get("Case", [""])[0] == "GaussianExpansion"]
    if len(decks) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one GaussianExpansion deck, found {len(decks)}; use --input")
    return decks[0]


class Theory:
    """The self-similar solution per axis; a uniform axis has tau = inf. Masses are relative: every species
    carries the same profile, so its mass is proportional to rhoINIT/|qom|."""

    def __init__(self, deck: Deck, sigma0: np.ndarray):
        self.deck, self.sigma0 = deck, sigma0
        self.mass = deck.rho_init / np.abs(deck.qom)
        self.sound2 = (self.mass[:, None] * deck.thermal**2).sum(axis=0) / self.mass.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tau = np.where(sigma0 > 0, sigma0 / np.sqrt(self.sound2), np.inf)
        self.time_unit = float(self.tau[deck.gaussian[0]])

    def x2(self, t) -> np.ndarray:
        """(t/tau_a)^2 with a trailing axis index."""
        return (np.asarray(t, dtype=float)[..., None] / self.tau) ** 2

    def width(self, t) -> np.ndarray:
        return self.sigma0 * np.sqrt(1 + self.x2(t))

    def cooling(self, t) -> np.ndarray:
        return 1 / (1 + self.x2(t))

    def flow_slope(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)[..., None]
        return t / (t**2 + self.tau**2)

    def centre_density(self, t) -> np.ndarray:
        return np.prod((1 + self.x2(t)[..., self.deck.gaussian]) ** -0.5, axis=-1)

    def kinetic(self, t, species: int) -> np.ndarray:
        """Kinetic energy of a species, thermal plus flow, in the units of self.mass."""
        x2 = self.x2(t)
        return 0.5 * self.mass[species] * ((self.deck.thermal[species] ** 2 + self.sound2 * x2) / (1 + x2)).sum(-1)

    def initial_energy(self) -> float:
        return float(sum(self.kinetic(0.0, s) for s in range(len(self.mass))))

    def ion_spread(self, t) -> np.ndarray:
        """Standard deviation of an ion velocity component."""
        x2 = self.x2(t)
        return np.sqrt((self.sound2 * x2 + self.deck.thermal[self.deck.ion] ** 2) / (1 + x2))

    def field_slope(self, t) -> np.ndarray:
        """E_a/x_a of the ambipolar field."""
        x2 = self.x2(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            return ((self.sound2 - self.deck.thermal[self.deck.electron] ** 2)
                    / (self.deck.qom[self.deck.electron] * self.sigma0**2 * (1 + x2) ** 2))


def derived_parameters(deck: Deck, theory: Theory) -> list[str]:
    e, i, g = deck.electron, deck.ion, deck.gaussian[0]
    wpe = math.sqrt(abs(deck.qom[e]) * deck.rho_init[e])
    debye = deck.thermal[e, g] / wpe
    sigma0, dx, tau = theory.sigma0[g], deck.spacing[g], theory.time_unit
    t_end = deck.dt * deck.ncycles
    walls = [min(deck.center[a], deck.lengths[a] - deck.center[a]) / theory.width(t_end)[a] for a in deck.gaussian
             if not deck.periodic[a]]
    lines = [
        f"Gaussian axes          {''.join(AXES[a] for a in deck.gaussian)} (D = {len(deck.gaussian)})",
        f"m_i/m_e                {abs(deck.qom[e]) / deck.qom[i]:.1f}",
        f"omega_pe (peak)        {wpe:.3f}",
        f"lambda_De0             {debye:.4e}",
        f"Lambda = sigma0/lamD   {sigma0 / debye:.1f}",
        f"sigma0 deck / measured {deck.sigma[g]:.5f} / {sigma0:.5f}",
        f"c_s0                   {math.sqrt(theory.sound2[g]):.5e}",
        f"tau = sigma0/c_s0      {tau:.3f}",
        f"dx                     {dx:.4f} = {dx / debye:.1f} lambda_De0 = sigma0/{sigma0 / dx:.1f}",
        f"omega_pe*dt            {wpe * deck.dt:.3f}",
        f"uth_e*dt/dx            {deck.thermal[e, g] * deck.dt / dx:.3f}",
        f"c*dt/dx                {deck.dt / dx:.3f}",
        f"t_end                  {t_end:.2f} = {t_end / tau:.3f} tau",
        f"sigma(t_end)           {theory.width(t_end)[g]:.4f}",
    ]
    if walls:
        lines.append(f"nearest wall           {min(walls):.2f} sigma(t_end), density there "
                     f"{math.exp(-0.5 * min(walls) ** 2):.1e} of the centre")
    return lines


def solution_table(deck: Deck, theory: Theory) -> list[str]:
    g, e, i = deck.gaussian[0], deck.electron, deck.ion
    energy0 = theory.initial_energy()
    lines = [" t/tau  sigma/s0  Te/Te0  n(0)/n0  tau*u/x  ion v_std/c_s  K_i/E0  K_e/E0"]
    for fraction in (0, 0.5, 1, 2, 3):
        t = fraction * theory.time_unit
        lines.append(f" {fraction:5.2f}  {theory.width(t)[g] / theory.sigma0[g]:8.4f}  {theory.cooling(t)[g]:6.4f}"
                     f"  {float(theory.centre_density(t)):7.4f}  {theory.flow_slope(t)[g] * theory.time_unit:7.4f}"
                     f"  {theory.ion_spread(t)[g] / math.sqrt(theory.sound2[g]):13.4f}"
                     f"  {float(theory.kinetic(t, i)) / energy0:6.4f}  {float(theory.kinetic(t, e)) / energy0:6.4f}")
    return lines


# ---------------------------------------------------------------------------- field snapshots


class Grid:
    """Node displacements from the centre and dual-volume weights of the field arrays (z, y, x); a periodic
    axis drops its repeated end node."""

    def __init__(self, deck: Deck):
        self.keep = tuple(slice(0, -1) if deck.periodic[a] else slice(None) for a in (2, 1, 0))
        self.d, factors = [], []
        for a in range(3):
            nodes = deck.cells[a] + (0 if deck.periodic[a] else 1)
            shape = [1, 1, 1]
            shape[2 - a] = nodes
            self.d.append((np.arange(nodes) * deck.spacing[a] - deck.center[a]).reshape(shape))
            factor = np.ones(nodes)
            if not deck.periodic[a]:
                factor[[0, -1]] = 0.5
            factors.append(factor.reshape(shape))
        self.weight = factors[0] * factors[1] * factors[2]
        self.gaussian = deck.gaussian
        self.radius = np.sqrt(sum(self.d[a] ** 2 for a in self.gaussian))

    def scaled_radius(self, widths: np.ndarray) -> np.ndarray:
        return np.sqrt(sum((self.d[a] / widths[a]) ** 2 for a in self.gaussian))


def shell_average(radius: np.ndarray, values: np.ndarray, weight: np.ndarray, width: float, extent: float):
    """Weighted averages of values over shells of radius; the shell centres and the averages."""
    radius, values, weight = np.broadcast_arrays(radius, values, weight)
    inside = radius < extent
    index = (radius[inside] / width).astype(int)
    total = np.bincount(index, weights=weight[inside])
    mean = np.bincount(index, weights=(weight * values)[inside]) / np.where(total > 0, total, np.nan)
    return (np.arange(total.size) + 0.5) * width, mean


@dataclass
class Snapshot:
    cycle: int
    time: float
    ion_density: np.ndarray  # the output rho, 4 pi times the charge density, (z, y, x)
    width: np.ndarray        # measured sigma per axis (0 on a uniform axis)
    centre: float            # mean ion charge density within 0.2 sigma
    total_charge: float      # ion charge of the domain
    flow_slope: np.ndarray | None
    electron_temperature: np.ndarray | None  # <v_a^2> - u_a^2 per axis within 1 sigma
    ion_temperature: np.ndarray | None
    profile: tuple[np.ndarray, np.ndarray]   # scaled radius, mean ion charge density
    flow: tuple[np.ndarray, np.ndarray] | None
    field: tuple[np.ndarray, np.ndarray] | None
    magnetic: np.ndarray | None  # B_a^2/(8 pi) integrated, per component


def read_snapshots(deck: Deck, data_dir: Path) -> list[Snapshot]:
    grid = Grid(deck)
    e, i, g = deck.electron, deck.ion, deck.gaussian[0]
    names = [f"rho_{i}", f"rho_{e}", f"J_{i}", f"J_{e}", *(f"P{a}{a}_{s}" for s in (e, i) for a in "XYZ"), "E", "B"]
    readers = {name: open_field_reader(data_dir / name) for name in names if field_data_exists(data_dir, name)}
    if f"rho_{i}" not in readers:
        raise FileNotFoundError(f"{data_dir}: no ion density rho_{i}_t*.vti; add rho to MomentsOutputTag")
    steps = readers[f"rho_{i}"].num_steps()
    readers = {name: reader for name, reader in readers.items() if reader.num_steps() == steps}
    cell_volume = float(np.prod(deck.spacing))

    def array(name, step, component=""):
        return readers[name].read(name + component, step_selection=[step, 1])[grid.keep]

    def velocity(s, step, charge):
        """The flow J/charge of species s per axis, 0 where it has no charge."""
        safe = np.where(charge != 0, charge, 1.0)
        return [np.where(charge != 0, array(f"J_{s}", step, a) / safe, 0.0) for a in AXES]

    def radial(vector):
        return sum(vector[a] * grid.d[a] for a in deck.gaussian) / np.where(grid.radius > 0, grid.radius, 1.0)

    snapshots = []
    for step in range(steps):
        time = float(readers[f"rho_{i}"].read("time", step_selection=[step, 1])[0])
        ion = array(f"rho_{i}", step) / FOUR_PI
        mass = ion / deck.qom[i] * grid.weight
        width = np.zeros(3)
        for a in deck.gaussian:
            width[a] = math.sqrt(float((mass * grid.d[a] ** 2).sum() / mass.sum()) - deck.spacing[a] ** 2 / 6)
        scaled = np.broadcast_to(grid.scaled_radius(width), ion.shape)
        centre = float(ion[scaled < 0.2].mean())
        profile = shell_average(scaled, ion, grid.weight, 0.1, 5.0)
        flow_slope = flow = field = None
        temperature = {}
        if f"J_{i}" in readers:
            u = velocity(i, step, ion)
            core = (scaled < 1.5) * mass
            flow_slope = np.array([float((core * u[a] * grid.d[a]).sum() / (core * grid.d[a] ** 2).sum())
                                   if a in deck.gaussian else 0.0 for a in range(3)])
            flow = shell_average(grid.radius, radial(u), mass, deck.spacing[g], 3 * width[g])
        for s in (e, i):
            if not all(name in readers for name in (f"rho_{s}", f"J_{s}", f"PXX_{s}", f"PYY_{s}", f"PZZ_{s}")):
                continue
            charge = array(f"rho_{s}", step) / FOUR_PI
            core = (scaled < 1.0) * grid.weight
            density = charge / deck.qom[s]
            u = velocity(s, step, charge)
            temperature[s] = np.array([float((core * (array(f"P{A}{A}_{s}", step) - density * u[a] ** 2)).sum()
                                             / (core * density).sum()) for a, A in enumerate("XYZ")])
        if "E" in readers:
            field = shell_average(grid.radius, radial([array("E", step, a) for a in AXES]),
                                  np.ones_like(grid.weight), 2 * deck.spacing[g], 3 * width[g])
        magnetic = None
        if "B" in readers:
            magnetic = np.array([float((grid.weight * array("B", step, a) ** 2).sum()) * cell_volume / (2 * FOUR_PI)
                                 for a in AXES])
        snapshots.append(Snapshot(int(round(time / deck.dt)), time, ion * FOUR_PI, width, centre,
                                  float((ion * grid.weight).sum()) * cell_volume, flow_slope,
                                  temperature.get(e), temperature.get(i), profile, flow, field, magnetic))
    for reader in readers.values():
        reader.close()
    return snapshots


# ---------------------------------------------------------------------------- particle snapshots


@dataclass
class ParticleSnapshot:
    time: float
    position: np.ndarray  # (particles, 3)
    velocity: np.ndarray
    charge: np.ndarray


def read_particle_snapshots(data_dir: Path, species: int) -> list[ParticleSnapshot]:
    snapshots = []
    for manifest in sorted(data_dir.glob("particles_t*.pvtp")):
        root = ET.parse(manifest).getroot()
        time = float(next(a.text for a in root.iter("DataArray") if a.get("Name") == "TimeValue"))
        position, velocity, charge = [], [], []
        for piece in root.iter("Piece"):
            arrays = read_polydata(manifest.parent / piece.get("Source"), ("position", "velocity", "charge", "species"))
            keep = arrays["species"] == species
            position.append(arrays["position"].reshape(-1, 3)[keep])
            velocity.append(arrays["velocity"].reshape(-1, 3)[keep])
            charge.append(arrays["charge"][keep])
        snapshots.append(ParticleSnapshot(time, np.concatenate(position), np.concatenate(velocity),
                                          np.concatenate(charge)))
    return snapshots


# ---------------------------------------------------------------------------- figures


def save(figure, output_dir: Path, name: str) -> list[Path]:
    paths = [output_dir / f"{name}.{extension}" for extension in ("png", "pdf")]
    for path in paths:
        figure.savefig(path)
    plt.close(figure)
    return paths


def column(rows, key) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])


def times_label(theory: Theory) -> str:
    return rf"$t/\tau$  ($\tau$ = {theory.time_unit:.1f})"


def plot_overview(deck, theory, snapshots, diagnostics, title):
    tau, g = theory.time_unit, deck.gaussian[0]
    t = np.array([s.time for s in snapshots])
    fine = np.linspace(0, max(t[-1], column(diagnostics, "time")[-1]), 400)
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    figure.suptitle(title)

    ax = axes[0, 0]
    for a in deck.gaussian:
        ax.plot(t / tau, [s.width[a] / theory.sigma0[a] for s in snapshots], "o", color=AXIS_COLORS[a],
                label=rf"$\sigma_{AXES[a]}$")
    ax.plot(fine / tau, theory.width(fine)[:, g] / theory.sigma0[g], **ANALYTIC, label="Analytic")
    ax.set(xlabel=times_label(theory), ylabel=r"$\sigma(t)/\sigma_0$", title="Width of the ion density")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t / tau, [s.centre / snapshots[0].centre for s in snapshots], "o", color=ION_COLOR, label="Ions")
    ax.plot(fine / tau, theory.centre_density(fine), **ANALYTIC, label="Analytic")
    ax.set(xlabel=times_label(theory), ylabel=r"$n(0,t)/n(0,0)$", title="Centre density")
    ax.legend()

    ax = axes[0, 2]
    if snapshots[0].flow_slope is not None:
        for a in deck.gaussian:
            ax.plot(t / tau, [s.flow_slope[a] * tau for s in snapshots], "o", color=AXIS_COLORS[a],
                    label=rf"$u_{AXES[a]}/{AXES[a]}$")
        ax.plot(fine / tau, theory.flow_slope(fine)[:, g] * tau, **ANALYTIC, label="Analytic")
        ax.legend()
    ax.set(xlabel=times_label(theory), ylabel=r"$\tau\,u_a/x_a$", title=r"Ion flow (core fit, $r<1.5\sigma$)")

    ax = axes[1, 0]
    if snapshots[0].electron_temperature is not None:
        thermal = deck.thermal[deck.electron] ** 2
        for a in range(3):
            if thermal[a] == 0:
                continue
            gaussian = a in deck.gaussian
            ax.plot(t / tau, [s.electron_temperature[a] / thermal[a] for s in snapshots], "o" if gaussian else "s",
                    color=AXIS_COLORS[a], label=rf"$T_{{e,{AXES[a]}}}$" + ("" if gaussian else " (uniform axis)"))
        ax.plot(fine / tau, theory.cooling(fine)[:, g], **ANALYTIC, label="Analytic")
        if len(deck.gaussian) < 3:
            ax.axhline(1.0, color="black", linestyle=":", linewidth=1.1)
        ax.legend()
    ax.set(xlabel=times_label(theory), ylabel=r"$T_{e,a}(t)/T_{e,a}(0)$", title=r"Electron temperature ($r<\sigma$)")

    ax = axes[1, 1]
    energy0 = float(diagnostics[0]["total_energy"])
    time = column(diagnostics, "time")
    analytic0 = theory.initial_energy()
    for s, label, color in ((deck.electron, "electrons", ELECTRON_COLOR), (deck.ion, "ions", ION_COLOR)):
        ax.plot(time / tau, column(diagnostics, f"kinetic_energy_{s}") / energy0, color=color, label=f"$K$ {label}")
        ax.plot(fine / tau, theory.kinetic(fine, s) / analytic0, **ANALYTIC)
    ax.plot([], [], **ANALYTIC, label="Analytic")
    ax.set(xlabel=times_label(theory), ylabel=r"$K_s/E_0$", title="Energy share")
    ax.legend()

    ax = axes[1, 2]
    if snapshots[0].ion_temperature is not None:
        for a in deck.gaussian:
            ax.semilogy(t[1:] / tau, [s.ion_temperature[a] / theory.sound2[a] for s in snapshots[1:]], "o-",
                        color=AXIS_COLORS[a], label=rf"$T_{{i,{AXES[a]}}}$")
        thermal = deck.thermal[deck.ion] ** 2
        if thermal[g] > 0:
            ax.semilogy(fine / tau, thermal[g] * theory.cooling(fine)[:, g] / theory.sound2[g], **ANALYTIC,
                        label="Analytic")
        ax.legend()
    ax.set(xlabel=times_label(theory), ylabel=r"$T_{i,a}/(m_i c_s^2)$", title=r"Ion temperature ($r<\sigma$)")
    return figure


def plot_width_density(deck, theory, snapshots):
    """The width of the ion density, sigma_a/sigma_a0 averaged over the Gaussian axes (left axis), and the ion
    density at the centre (right axis) against time: each measured value is a thin line with dots over its analytic
    curve, a wide light line of the same color."""
    tau, gaussian = theory.time_unit, deck.gaussian
    t = np.array([s.time for s in snapshots]) / tau
    fine = np.linspace(0, t[-1] * tau, 400)
    figure, width_axis = plt.subplots(figsize=(8, 5))
    density_axis = width_axis.twinx()
    density_axis.grid(False)
    curves = []
    for ax, color, measured, analytic, label in (
            (width_axis, AXIS_COLORS[0], [np.mean(s.width[gaussian] / theory.sigma0[gaussian]) for s in snapshots],
             np.mean(theory.width(fine)[:, gaussian] / theory.sigma0[gaussian], axis=1), r"Radius $\sigma$"),
            (density_axis, ION_COLOR, [s.centre / snapshots[0].centre for s in snapshots],
             theory.centre_density(fine), "Centre density")):
        ax.plot(fine / tau, analytic, color=color, alpha=0.35, linewidth=8.0)
        curves += ax.plot(t, measured, color=color, linewidth=1.3, marker="o", markersize=8.0, label=label)
    curves += density_axis.plot([], [], color="0.5", alpha=0.35, linewidth=8.0, label="Analytic")
    width_axis.set(xlabel=times_label(theory), ylabel=r"$\sigma(t)/\sigma_0$")
    density_axis.set_ylabel(r"$n_i(0,t)/n_i(0,0)$")
    # The density axis lies on top, so its legend covers both; the curves leave the top centre free.
    density_axis.legend(handles=curves, loc="upper center")
    return figure


def plot_energy_share(deck, theory, diagnostics):
    """The energy share panel of plot_overview alone: the kinetic energy of each species over E_0 against time, a
    thin line with dots over its analytic curve, a wide light line of the same color."""
    tau = theory.time_unit
    time = column(diagnostics, "time")
    fine = np.linspace(0, time[-1], 400)
    energy0, analytic0 = float(diagnostics[0]["total_energy"]), theory.initial_energy()
    figure, ax = plt.subplots(figsize=(8, 5))
    for s, label, color in ((deck.electron, "Electrons", ELECTRON_COLOR), (deck.ion, "Ions", ION_COLOR)):
        ax.plot(fine / tau, theory.kinetic(fine, s) / analytic0, color=color, alpha=0.35, linewidth=8.0)
        ax.plot(time / tau, column(diagnostics, f"kinetic_energy_{s}") / energy0, color=color, linewidth=1.3,
                marker="o", markersize=8.0, markevery=max(1, time.size // 12), label=label)
    ax.plot([], [], color="0.5", alpha=0.35, linewidth=8.0, label="Analytic")
    ax.set(xlabel=times_label(theory), ylabel=r"$K_s/E_0$")
    # The curves cross in the middle and leave the right between them free.
    ax.legend(loc="center right")
    return figure


def plane_axes(deck: Deck) -> tuple[int, int, int]:
    first, second = deck.gaussian[0], (deck.gaussian[1] if len(deck.gaussian) > 1 else (deck.gaussian[0] + 1) % 3)
    return first, second, 3 - first - second


def plot_density_maps(deck, theory, snapshots, title):
    from matplotlib.patches import Ellipse

    first, second, normal = plane_axes(deck)
    grid = Grid(deck)
    picks = sorted(set(np.linspace(0, len(snapshots) - 1, 4).round().astype(int)))
    figure, axes = plt.subplots(1, len(picks), figsize=(4.1 * len(picks) + 1.2, 4.9), squeeze=False)
    figure.suptitle(title)
    peak = FOUR_PI * snapshots[0].centre
    # Up to five analytic widths of the last map, within the box.
    reach = 5 * theory.width(snapshots[picks[-1]].time)
    for ax, index in zip(axes[0], picks):
        snapshot = snapshots[index]
        nodes = grid.d[normal].size
        cut = [slice(None)] * 3
        cut[2 - normal] = min(nodes - 1, int(round(float(deck.center[normal] / deck.spacing[normal]))))
        image = snapshot.ion_density[tuple(cut)]
        if first > second:
            image = image.T
        x, y = grid.d[first].ravel(), grid.d[second].ravel()
        mesh = ax.imshow(np.maximum(image / peak, 1e-6), origin="lower", cmap="magma",
                         norm=LogNorm(vmin=1e-4, vmax=1.0), extent=(x[0], x[-1], y[0], y[-1]))
        widths = theory.width(snapshot.time)
        for multiple, style in ((1, "-"), (2, "--")):
            ax.add_patch(Ellipse((0, 0), 2 * multiple * widths[first], 2 * multiple * widths[second],
                                 fill=False, color="white", linestyle=style, linewidth=1.0))
        ax.set(xlabel=rf"${AXES[first]} - {AXES[first]}_c$", title=rf"$t/\tau$ = {snapshot.time / theory.time_unit:.2f}",
               xlim=(max(x[0], -reach[first]), min(x[-1], reach[first])),
               ylim=(max(y[0], -reach[second]), min(y[-1], reach[second])))
        ax.grid(False)
    axes[0, 0].set_ylabel(rf"${AXES[second]} - {AXES[second]}_c$")
    figure.colorbar(mesh, ax=axes[0].tolist(), label=r"$n_i/n_i(0,0)$; analytic $1\sigma$, $2\sigma$ in white",
                    shrink=0.9)
    return figure


# The 3D density panels: the snapshots nearest these t/tau, rendered in square images of PIXELS_3D from the camera
# direction VIEW_3D, with ticks every TICK_STEP_3D and at the box end, and the volume opacity OPACITY_3D from
# log10(n_i/n_i(0,0)) = FLOOR_3D to 0. The camera looks through the cut-away octant at the centre; its sign per axis
# is CUT_3D: x > 0, y < 0, z > 0 puts x on the bottom left, y on the bottom right and z vertical. The tick labels sit
# LABEL_PAD_3D box units per point of their font size off their edge, the axis names 2.5 times as far (2.8 for the
# vertical z).
TIMES_3D = (0.0, 1.5, 3.0)
PIXELS_3D = 1100
CUT_3D = np.array([1.0, -1.0, 1.0])
VIEW_3D = CUT_3D * [0.82, 1.0, 0.62]
TICK_STEP_3D = 3.0
LABEL_PAD_3D = 0.13
OPACITY_3D = [0.0, 0.04, 0.12, 0.3, 0.55, 0.8]
FLOOR_3D = -4.0


def arc_segments(radius: float, axis: int, dashed: bool, offset: float) -> list[np.ndarray]:
    """The quarter circle of radius on the cut face normal to axis, as polylines (dashes when dashed)."""
    others = [a for a in range(3) if a != axis]
    angles = np.linspace(0, 0.5 * math.pi, 181)
    points = np.zeros((angles.size, 3))
    points[:, others[0]] = radius * np.cos(angles)
    points[:, others[1]] = radius * np.sin(angles)
    points[:, axis] = offset
    points *= CUT_3D
    if not dashed:
        return [points]
    dash = 9  # points per dash and per gap
    return [points[start:start + dash + 1] for start in range(0, angles.size - 1, 2 * dash)]


def box_frame(low: np.ndarray, high: np.ndarray, center: np.ndarray):
    """The box edges away from the camera, the grid on the three back faces, the tick marks and the label anchors
    (3D point, text, font size, rotation). The scene is centred on center; the ticks and labels are in box
    coordinates. The camera looks from the near corner of the CUT_3D octant (x high, y low, z high); ticks sit on
    the bottom edges through the near vertical edge, (y low, z low) for x and (x high, z low) for y, and on the
    left vertical edge (x low, y low) for z."""
    near, far = np.where(CUT_3D > 0, high, low), np.where(CUT_3D > 0, low, high)
    corners = [np.array([high[0] if i & 1 else low[0], high[1] if i & 2 else low[1], high[2] if i & 4 else low[2]])
               for i in range(8)]
    near_corner = int(np.dot(CUT_3D > 0, (1, 2, 4)))
    edge_lines = [np.array([corners[a], corners[b]]) for a in range(8) for b in range(a + 1, 8)
                  if bin(a ^ b).count("1") == 1 and near_corner not in (a, b)]
    # Ticks every TICK_STEP_3D and at the box end, without the multiples less than a step from the end (labels clash).
    ends = high + center
    ticks = [np.append(np.arange(math.ceil((low[a] + center[a]) / TICK_STEP_3D) * TICK_STEP_3D,
                                 ends[a] - TICK_STEP_3D + 1e-9, TICK_STEP_3D), ends[a]) - center[a] for a in range(3)]
    grid_lines = []
    for face in range(3):  # the back face at far[face]
        for along in range(3):
            if along == face:
                continue
            across = 3 - face - along
            for value in ticks[across]:
                start, end = far.copy(), far.copy()
                start[across] = end[across] = value
                end[along] = near[along]
                grid_lines.append(np.array([start, end]))
    # Per axis: the labelled edge as a function of the tick value, and the outward direction.
    edges_of = {0: (lambda v: np.array([v, low[1], low[2]]), np.array([0.0, -1.0, -1.0])),
                1: (lambda v: np.array([high[0], v, low[2]]), np.array([1.0, 0.0, -1.0])),
                2: (lambda v: np.array([low[0], low[1], v]), np.array([-1.0, -1.0, 0.0]))}
    tick_lines, anchors = [], []
    pad = LABEL_PAD_3D * DENSITY_3D_FONT_SIZES.box_tick_label
    for axis, (point, outward) in edges_of.items():
        outward = outward / np.linalg.norm(outward)
        for value in ticks[axis]:
            tick_lines.append(np.array([point(value), point(value) + 0.35 * outward]))
            anchors.append((point(value) + pad * outward, f"{value + center[axis]:g}",
                            DENSITY_3D_FONT_SIZES.box_tick_label, 0))
        middle = 0.5 * (low[axis] + high[axis])
        vertical = axis == 2
        anchors.append((point(middle) + (2.8 if vertical else 2.5) * pad * outward, rf"${AXES[axis]}$ [$d_i$]",
                        DENSITY_3D_FONT_SIZES.box_axis_label, 90 if vertical else 0))
    return edge_lines, grid_lines, tick_lines, anchors


def render_panel(deck, density: np.ndarray, width: float, camera, path: Path):
    """Render one time into path: density is n_i/n_i(0,0) on the nodes, (z, y, x). Returns the label anchors in
    image pixels (column, row, text, font size, rotation)."""
    import pyvista as pv
    from scipy.ndimage import gaussian_filter

    grid = pv.ImageData(dimensions=tuple(int(n) for n in deck.cells + 1), spacing=tuple(deck.spacing),
                        origin=tuple(-deck.center))
    grid.point_data["log_n"] = np.log10(np.maximum(density, 1e-12)).ravel()
    grid.point_data["log_smooth"] = np.log10(np.maximum(gaussian_filter(density, 1.0), 1e-12)).ravel()
    cut = (grid.points * CUT_3D).min(axis=1) > 0  # the cut-away octant
    grid.point_data["volume"] = np.where(cut, FLOOR_3D, np.clip(grid.point_data["log_smooth"], FLOOR_3D, 0.0))

    plotter = pv.Plotter(off_screen=True, window_size=(PIXELS_3D, PIXELS_3D))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("msaa")
    offset = 0.02 * deck.spacing[0]  # lifts the circles off the faces towards the camera
    for axis, normal in enumerate(AXES):
        face = grid.slice(normal=normal, origin=(0.0, 0.0, 0.0))
        for other in range(3):
            if other != axis:  # keeps the CUT_3D side
                face = face.clip(normal=tuple(CUT_3D[other] * np.eye(3)[other]), origin=(0.0, 0.0, 0.0), invert=False)
        face = face.clip_scalar(scalars="log_smooth", value=FLOOR_3D, invert=False)
        plotter.add_mesh(face, scalars="log_n", cmap="magma", clim=(FLOOR_3D, 0.0), lighting=False,
                         show_scalar_bar=False)
        for multiple in (1, 2):
            for segment in arc_segments(multiple * width, axis, multiple == 2, offset):
                plotter.add_mesh(pv.lines_from_points(segment), color="white", line_width=6.0,
                                 render_lines_as_tubes=True, lighting=False)
    plotter.add_volume(grid, scalars="volume", cmap="magma", clim=(FLOOR_3D, 0.0), opacity=OPACITY_3D, shade=False,
                       show_scalar_bar=False)
    edge_lines, grid_lines, tick_lines, anchors = box_frame(np.array(grid.bounds[0::2]),
                                                            np.array(grid.bounds[1::2]), deck.center)
    for lines, color, line_width in ((grid_lines, "#cfcfcf", 1.2), (edge_lines, "#7a7a7a", 1.6),
                                     (tick_lines, "#4a4a4a", 1.6)):
        for line in lines:
            plotter.add_mesh(pv.lines_from_points(line), color=color, line_width=line_width, lighting=False)
    plotter.camera_position = camera
    plotter.screenshot(str(path))
    labels = []
    for point, *label in anchors:
        plotter.renderer.SetWorldPoint(*point, 1.0)
        plotter.renderer.WorldToDisplay()
        column, row, _ = plotter.renderer.GetDisplayPoint()
        labels.append((column, PIXELS_3D - row, *label))
    plotter.close()
    return labels


def plot_density_3d(deck, theory, snapshots):
    """The 3D counterpart of plot_density_maps for three Gaussian axes. At the snapshots nearest TIMES_3D the ion
    density is a sphere with the octant facing the camera cut away through the centre. The three cut faces carry
    n_i/n_i(0,0) on the colour scale of the 2D maps and end where n_i = 1e-4 n_i(0,0); the other seven octants are a
    volume rendering on the same colour scale, more opaque where the density is higher. The face edge and the volume
    follow the density smoothed over one cell (the face colours are not), which removes the particle noise of the
    dilute edge. The analytic 1 sigma (solid) and 2 sigma (dashed) circles lie on the faces in white. The camera is
    the same in every panel; the axes span the simulation box in its own coordinates, with a grid on its three back
    faces and the tick labels placed by projecting the ticks onto the image."""
    import pyvista as pv

    peak = FOUR_PI * snapshots[0].centre
    times = np.array([s.time for s in snapshots]) / theory.time_unit
    picks = [int(np.argmin(np.abs(times - target))) for target in TIMES_3D]

    # One camera for every panel, framed on the simulation box.
    box = pv.Box(bounds=tuple(np.ravel([[-c, l - c] for c, l in zip(deck.center, deck.lengths)])))
    framing = pv.Plotter(off_screen=True, window_size=(PIXELS_3D, PIXELS_3D))
    framing.add_mesh(box.outline())
    framing.camera_position = [tuple(40 * VIEW_3D / np.linalg.norm(VIEW_3D)), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    framing.reset_camera()
    framing.camera.zoom(0.92)
    camera = framing.camera_position
    framing.close()

    figure, axes = plt.subplots(1, len(picks), figsize=(4.1 * len(picks) + 1.2, 4.9), squeeze=False)
    with tempfile.TemporaryDirectory() as scratch:
        for ax, index in zip(axes[0], picks):
            snapshot = snapshots[index]
            path = Path(scratch) / f"panel_{index}.png"
            labels = render_panel(deck, snapshot.ion_density / peak, float(theory.width(snapshot.time)[0]), camera,
                                  path)
            ax.imshow(plt.imread(path))
            for column, row, text, size, rotation in labels:
                ax.text(column, row, text, ha="center", va="center", fontsize=size, rotation=rotation)
            ax.set_title(rf"$t/\tau$ = {snapshot.time / theory.time_unit:.2f}")
            ax.set_axis_off()
    figure.colorbar(ScalarMappable(norm=LogNorm(vmin=10**FLOOR_3D, vmax=1.0), cmap="magma"), ax=axes[0].tolist(),
                    label=r"$n_i/n_i(0,0)$", shrink=0.9)
    return figure


def plot_profiles(deck, theory, snapshots, title):
    tau, g = theory.time_unit, deck.gaussian[0]
    dimension = len(deck.gaussian)
    uniform = float(np.prod([deck.lengths[a] for a in range(3) if a not in deck.gaussian]))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    figure.suptitle(title)
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(snapshots)))

    ax = axes[0]
    for snapshot, color in zip(snapshots, colors):
        peak = snapshot.total_charge / ((2 * math.pi) ** (dimension / 2) * uniform *
                                        np.prod([snapshot.width[a] for a in deck.gaussian]))
        radius, mean = snapshot.profile
        ax.semilogy(radius, mean / peak, color=color, linewidth=1.2, label=rf"$t/\tau$ = {snapshot.time / tau:.2f}")
    radius = np.linspace(0, 5, 200)
    ax.semilogy(radius, np.exp(-radius**2 / 2), **ANALYTIC, label=r"$e^{-\hat r^2/2}$")
    ax.set(xlabel=r"$\hat r = \sqrt{\sum_a (x_a/\sigma_a)^2}$", ylabel=r"$n_i/n_\mathrm{peak}$", ylim=(1e-6, 2),
           title="Self-similar density profile")
    ax.legend(ncol=2, fontsize=FONT_SIZES.legend - 1)

    picks = sorted(set(np.linspace(1, len(snapshots) - 1, 4).round().astype(int)))
    panels = ((axes[1], "flow", theory.flow_slope, math.sqrt(theory.sound2[g]),
               dict(marker="o", linestyle="none", markersize=3.5), r"$u_{i,r}/c_s$", "Radial ion flow (dashed: analytic)"),
              (axes[2], "field", theory.field_slope, abs(theory.field_slope(0.0)[g]) * theory.sigma0[g],
               dict(marker="o", markersize=3.0, linewidth=0.8), r"$E_r/|E_r(\sigma_0, 0)|$",
               "Ambipolar field (shell mean; dashed: analytic)"))
    for ax, measured, slope, scale, style, ylabel, heading in panels:
        if getattr(snapshots[0], measured) is not None:
            for index in picks:
                snapshot, color = snapshots[index], colors[index]
                radius, mean = getattr(snapshot, measured)
                ax.plot(radius / snapshot.width[g], mean / scale, color=color, **style,
                        label=rf"$t/\tau$ = {snapshot.time / tau:.2f}")
                ax.plot(radius / snapshot.width[g], slope(snapshot.time)[g] * radius / scale, color=color,
                        linestyle="--", linewidth=1.1)
            ax.legend()
        ax.set(xlabel=r"$r/\sigma(t)$", ylabel=ylabel, title=heading)
    return figure


def plot_radial_flow(deck, theory, snapshots):
    """The radial ion flow panel of plot_profiles alone: at each time the measured shell means are a thin line with
    dots over the analytic flow, a wide light line of the same color."""
    tau, g = theory.time_unit, deck.gaussian[0]
    sound = math.sqrt(theory.sound2[g])
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(snapshots)))
    figure, ax = plt.subplots(figsize=(8, 5))
    for index in sorted(set(np.linspace(1, len(snapshots) - 1, 4).round().astype(int))):
        snapshot, color = snapshots[index], colors[index]
        radius, mean = snapshot.flow
        scaled = radius / snapshot.width[g]
        ax.plot(scaled, theory.flow_slope(snapshot.time)[g] * radius / sound, color=color, alpha=0.35, linewidth=8.0)
        ax.plot(scaled, mean / sound, color=color, linewidth=1.3, marker="o", markersize=4.0,
                markevery=max(1, scaled.size // 25), label=rf"$t/\tau$ = {snapshot.time / tau:.2f}")
    ax.plot([], [], color="0.5", alpha=0.35, linewidth=8.0, label="Analytic")
    ax.set(xlabel=r"$r/\sigma(t)$", ylabel=r"$u_{i,r}/c_s$")
    ax.legend(loc="upper left")
    return figure


def plot_conservation(deck, theory, snapshots, diagnostics, gauss, gauss_norm, title):
    tau, g = theory.time_unit, deck.gaussian[0]
    time = column(diagnostics, "time") / tau
    energy0 = float(diagnostics[0]["total_energy"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.suptitle(title)

    ax = axes[0, 0]
    ax.semilogy(time[1:], np.abs(column(diagnostics, "total_energy_change")[1:]) / energy0, color="#444444")
    ax.set(xlabel=times_label(theory), ylabel=r"$|E(t) - E_0|/E_0$", title="Total energy change")

    ax = axes[0, 1]
    ax.semilogy(time[1:], column(diagnostics, "electric_energy")[1:] / energy0, color=AXIS_COLORS[0], label="$W_E$")
    ax.semilogy(time[1:], column(diagnostics, "magnetic_energy")[1:] / energy0, color="#555555", label="$W_B$")
    if snapshots[0].magnetic is not None:
        for a, marker in zip(range(3), "o^s"):
            ax.semilogy([s.time / tau for s in snapshots[1:]], [s.magnetic[a] / energy0 for s in snapshots[1:]],
                        marker, color=AXIS_COLORS[a], label=rf"$B_{AXES[a]}$ (snapshots)")
    ax.set(xlabel=times_label(theory), ylabel=r"$W/E_0$", title="Field energies")
    ax.legend()

    ax = axes[1, 0]
    ion_mass = float(diagnostics[0][f"charge_{deck.ion}"]) / deck.qom[deck.ion]
    scale = ion_mass * math.sqrt(theory.sound2[g])
    for a in range(3):
        ax.semilogy(time, np.abs(column(diagnostics, f"momentum_{AXES[a]}")) / scale, color=AXIS_COLORS[a],
                    label=rf"$|P_{AXES[a]}|$")
    ax.set(xlabel=times_label(theory), ylabel=r"$|P|/(M_i c_s)$", title="Total momentum", ylim=(1e-8, None))
    ax.legend()

    ax = axes[1, 1]
    if gauss:
        gauss_time = column(gauss, "time") / tau
        ax.semilogy(gauss_time, column(gauss, "residual_rms") / gauss_norm, "o-", color=AXIS_COLORS[0], markersize=3,
                    label="Residual")
        ax.semilogy(gauss_time, column(gauss, "drift_rms") / gauss_norm, "s-", color=AXIS_COLORS[1], markersize=3,
                    label=r"Drift $r - r_0$")
        ax.legend()
    ax.set(xlabel=times_label(theory), ylabel=r"rms$/$rms$(4\pi\rho_i)$", title="Gauss-law residual (gauss.csv)")
    return figure


def plot_energy_gauss(theory, diagnostics, gauss):
    """The signed relative energy change (left axis) and the absolute Gauss-law residual (right axis) against time:
    a thin dark blue line with dots over a wide light purple one."""
    figure, energy_axis = plt.subplots(figsize=(8, 5))
    gauss_axis = energy_axis.twinx()
    energy_axis.set_zorder(gauss_axis.get_zorder() + 1)
    energy_axis.patch.set_visible(False)
    time = column(diagnostics, "time") / theory.time_unit
    energy = energy_axis.plot(time, column(diagnostics, "total_energy_change") / float(diagnostics[0]["total_energy"]),
                              color="#1f4a9e", linewidth=1.3, marker="o", markersize=4.0,
                              markevery=max(1, time.size // 25), label="Energy")
    residual = gauss_axis.plot(column(gauss, "time") / theory.time_unit, column(gauss, "residual_rms"),
                               color="#d3a9ee", linewidth=4.5, label="Gauss law")
    energy_axis.set(xlabel=times_label(theory), ylabel=r"$(E - E_0)/E_0$")
    gauss_axis.set_ylabel(r"rms$(\nabla\cdot E - 4\pi\rho)$")
    gauss_axis.grid(False)
    for ax in (energy_axis, gauss_axis):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # The energy axis lies on top, so its legend covers both curves; "best" would avoid the energy curve alone.
    energy_axis.legend(handles=energy + residual, loc="lower center")
    return figure


def plot_ions(deck, theory, particles, title):
    from scipy.special import gamma as gamma_function

    tau, gaussian = theory.time_unit, deck.gaussian
    dimension = len(gaussian)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    figure.suptitle(title)
    colors = plt.cm.viridis(np.linspace(0, 0.92, max(len(particles), 1)))
    v = np.linspace(-4.5, 4.5, 300)
    for snapshot, color in zip(particles, colors):
        spread = theory.ion_spread(snapshot.time)
        scaled = np.concatenate([snapshot.velocity[:, a] / spread[a] for a in gaussian])
        weights = np.tile(np.abs(snapshot.charge), dimension)
        measured = math.sqrt(float(np.average(scaled**2, weights=weights)))
        label = rf"$t/\tau$ = {snapshot.time / tau:.2f}, std/analytic = {measured:.3f}"
        axes[0].hist(scaled, bins=120, range=(-4.5, 4.5), weights=weights, density=True, histtype="step",
                     color=color, linewidth=1.3, label=label)
        energy = sum((snapshot.velocity[:, a] / spread[a]) ** 2 for a in gaussian) / 2
        axes[1].hist(energy, bins=120, range=(0, 15), weights=np.abs(snapshot.charge), density=True,
                     histtype="step", color=color, linewidth=1.3, log=True, label=rf"$t/\tau$ = {snapshot.time / tau:.2f}")
    axes[0].plot(v, np.exp(-v**2 / 2) / math.sqrt(2 * math.pi), **ANALYTIC, label="Analytic")
    axes[0].set(xlabel=r"$v_{i,a}/v_\mathrm{std}(t)$ (Gaussian axes pooled)", ylabel="probability density",
                title="Ion velocity distribution")
    axes[0].legend(fontsize=FONT_SIZES.legend - 1)
    energy = np.linspace(1e-3, 15, 400)
    axes[1].plot(energy, energy ** (dimension / 2 - 1) * np.exp(-energy) / gamma_function(dimension / 2), **ANALYTIC,
                 label=rf"$\chi^2$, {dimension} components")
    axes[1].set(xlabel=r"$\mathcal{E}/(m_i v_\mathrm{std}^2)$", ylabel="probability density", ylim=(1e-7, 2),
                title="Ion energy distribution (tail: ion front)")
    axes[1].legend(fontsize=FONT_SIZES.legend - 1)
    return figure


def plot_phase_space(deck, theory, particles, title):
    """Ion x-v_x phase space of the ions within half a width of the centre along the second axis."""
    first, second, _ = plane_axes(deck)
    tau = theory.time_unit
    figure, axes = plt.subplots(1, len(particles), figsize=(4.6 * len(particles) + 0.8, 4.4), squeeze=False)
    figure.suptitle(title)
    for ax, snapshot in zip(axes[0], particles):
        width = theory.width(snapshot.time)
        offset = snapshot.position - deck.center
        band = np.abs(offset[:, second]) < 0.5 * width[second] if second in deck.gaussian else slice(None)
        x = offset[band, first] / width[first]
        v = snapshot.velocity[band, first] / math.sqrt(theory.sound2[first])
        mesh = ax.hist2d(x, v, bins=(160, 160), range=((-4.5, 4.5), (-3.5, 3.5)), weights=np.abs(snapshot.charge[band]),
                         norm=LogNorm(), cmap="viridis")[3]
        line = np.linspace(-4.5, 4.5, 50)
        ax.plot(line, theory.flow_slope(snapshot.time)[first] * width[first] * line / math.sqrt(theory.sound2[first]),
                color="white", linestyle="--", linewidth=1.2)
        ax.set(xlabel=rf"$({AXES[first]} - {AXES[first]}_c)/\sigma(t)$", title=rf"$t/\tau$ = {snapshot.time / tau:.2f}")
        ax.grid(False)
    axes[0, 0].set_ylabel(rf"$v_{{i,{AXES[first]}}}/c_s$")
    figure.colorbar(mesh, ax=axes[0].tolist(), label=r"ion charge per bin; analytic flow dashed", shrink=0.9)
    return figure


# ---------------------------------------------------------------------------- summary


def check(name: str, value: float, target: float) -> str:
    return f"{name:44s} {value:10.3e}   target {target:.0e}   {'ok' if value <= target else 'over'}"


def summarize(deck, theory, snapshots, diagnostics, gauss, gauss_norm, particles) -> list[str]:
    tau, g = theory.time_unit, deck.gaussian[0]
    energy0 = float(diagnostics[0]["total_energy"])
    analytic0 = theory.initial_energy()
    by_cycle = {int(row["cycle"]): row for row in diagnostics}
    later = [s for s in snapshots if s.time >= 0.25 * tau]
    lines = ["Acceptance checks A1-A10 of docs/user/gaussian_expansion.md"]
    lines.append(check("A1 max |E - E0|/E0", float(np.max(np.abs(column(diagnostics, "total_energy_change")))) / energy0,
                       1e-4))
    for a in deck.gaussian:
        error = max(abs(s.width[a] / theory.width(s.time)[a] - 1) for s in snapshots)
        lines.append(check(f"A2 max |sigma_{AXES[a]}/analytic - 1|", error, 1e-2))
    if len(deck.gaussian) > 1:
        error = max(abs(s.width[a] / s.width[b] - 1) for s in snapshots for a in deck.gaussian for b in deck.gaussian)
        lines.append(check("A3 max |sigma_a/sigma_b - 1|", error, 5e-3))
    if later and later[0].flow_slope is not None:
        error = max(abs(s.flow_slope[a] / theory.flow_slope(s.time)[a] - 1) for s in later for a in deck.gaussian)
        lines.append(check("A4 max |flow slope/analytic - 1| (t > tau/4)", error, 3e-2))
    if snapshots[0].electron_temperature is not None:
        thermal = deck.thermal[deck.electron] ** 2
        error = max(abs(s.electron_temperature[a] / thermal[a] / theory.cooling(s.time)[a] - 1)
                    for s in snapshots for a in deck.gaussian)
        lines.append(check("A5 max |core T_e/analytic - 1|", error, 5e-2))
        for a in range(3):
            if a not in deck.gaussian and thermal[a] > 0:
                lines.append(f"   T_e along the uniform axis {AXES[a]} at the end: "
                             f"{snapshots[-1].electron_temperature[a] / thermal[a]:.4f} of its start (analytic 1)")
    for snapshot in particles:
        spread = theory.ion_spread(snapshot.time)
        ratio = max(abs(math.sqrt(float(np.average(snapshot.velocity[:, a] ** 2, weights=np.abs(snapshot.charge))))
                        / spread[a] - 1) for a in deck.gaussian)
        lines.append(check(f"A6 |ion v std/analytic - 1| at t/tau = {snapshot.time / tau:.2f}", ratio, 2e-2))
    for s, name in ((deck.ion, "K_i"), (deck.electron, "K_e")):
        errors = [abs(float(by_cycle[x.cycle][f"kinetic_energy_{s}"]) / energy0
                      / (float(theory.kinetic(x.time, s)) / analytic0) - 1) for x in later if x.cycle in by_cycle]
        if errors:
            lines.append(check(f"A7 max |{name}/analytic - 1| (t > tau/4)", max(errors), 3e-2))
    formed = column(diagnostics, "time") >= 0.25 * tau
    magnetic, electric = column(diagnostics, "magnetic_energy")[formed], column(diagnostics, "electric_energy")[formed]
    lines.append(check("A8 max W_B/W_E (t > tau/4)", float(np.max(magnetic / np.where(electric > 0, electric, np.inf))),
                       1e-2))
    ion_mass = float(diagnostics[0][f"charge_{deck.ion}"]) / deck.qom[deck.ion]
    momentum = float(np.max(np.sqrt(sum(column(diagnostics, f"momentum_{a}") ** 2 for a in AXES))))
    lines.append(check("A9 max |P|/(M_i c_s)", momentum / (ion_mass * math.sqrt(theory.sound2[g])), 1e-3))
    if gauss:
        lines.append(f"A10 Gauss residual rms/rms(4 pi rho_i): first {float(gauss[0]['residual_rms']) / gauss_norm:.3e}, "
                     f"last {float(gauss[-1]['residual_rms']) / gauss_norm:.3e}, "
                     f"last drift {float(gauss[-1]['drift_rms']) / gauss_norm:.3e}")
    return lines


def write_snapshot_csv(path: Path, deck, theory, snapshots, diagnostics):
    by_cycle = {int(row["cycle"]): row for row in diagnostics}
    energy0, analytic0 = float(diagnostics[0]["total_energy"]), theory.initial_energy()
    header = ["cycle", "time", "t_over_tau", "centre_density", "centre_density_analytic"]
    for a in deck.gaussian:
        header += [f"sigma_{AXES[a]}", f"sigma_{AXES[a]}_analytic", f"flow_slope_{AXES[a]}",
                   f"flow_slope_{AXES[a]}_analytic"]
    header += [f"Te_{AXES[a]}_ratio" for a in range(3)] + ["Te_gaussian_ratio_analytic"]
    header += ["Ki_over_E0", "Ki_over_E0_analytic", "Ke_over_E0", "Ke_over_E0_analytic"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for s in snapshots:
            row = [s.cycle, s.time, s.time / theory.time_unit, s.centre / snapshots[0].centre,
                   float(theory.centre_density(s.time))]
            for a in deck.gaussian:
                row += [s.width[a], theory.width(s.time)[a],
                        s.flow_slope[a] if s.flow_slope is not None else "", theory.flow_slope(s.time)[a]]
            thermal = deck.thermal[deck.electron] ** 2
            row += [s.electron_temperature[a] / thermal[a] if s.electron_temperature is not None and thermal[a] > 0
                    else "" for a in range(3)]
            row.append(theory.cooling(s.time)[deck.gaussian[0]])
            diagnostic = by_cycle.get(s.cycle)
            for species in (deck.ion, deck.electron):
                row += [float(diagnostic[f"kinetic_energy_{species}"]) / energy0 if diagnostic else "",
                        float(theory.kinetic(s.time, species)) / analytic0]
            writer.writerow(row)


# ---------------------------------------------------------------------------- driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a GaussianExpansion run with the self-similar expansion.")
    parser.add_argument("--run-dir", type=Path, default=Path("."), help="run directory (default: current directory)")
    parser.add_argument("--input", type=Path, default=None,
                        help="input deck (default: the single GaussianExpansion deck of RUN_DIR)")
    parser.add_argument("--data-dir", type=Path, default=None, help="output directory of the run (default: "
                        "RUN_DIR/SaveDirName)")
    parser.add_argument("--output-dir", type=Path, default=None, help="where the plots go (default: RUN_DIR)")
    parser.add_argument("--prefix", default="gaussian_expansion", help="file-name prefix of the plots and tables")
    parser.add_argument("--tables", action="store_true", help="print the analytic solution of the deck and stop")
    parser.add_argument("--no-particles", action="store_true", help="skip the particle snapshots")
    parser.add_argument("--density-3d", action="store_true",
                        help="also draw <prefix>_density_3d (three Gaussian axes; needs pyvista)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    deck = read_deck(find_input(args.run_dir, args.input))
    if args.density_3d and len(deck.gaussian) != 3:
        raise ValueError(f"{deck.path}: --density-3d needs three Gaussian axes")
    if args.tables:
        theory = Theory(deck, deck.sigma)
        print("\n".join(["Derived parameters (deck widths)", *derived_parameters(deck, theory), "",
                         "Analytic solution", *solution_table(deck, theory)]))
        return
    data_dir = args.data_dir or args.run_dir / deck.save_dir
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = read_snapshots(deck, data_dir)
    if snapshots[0].cycle != 0:
        raise ValueError(f"{data_dir}: the first field snapshot is cycle {snapshots[0].cycle}; the widths at t = 0 "
                         "come from cycle 0")
    theory = Theory(deck, snapshots[0].width)
    diagnostics = read_csv(data_dir)
    gauss = read_csv(data_dir, "gauss.csv") if (data_dir / "gauss.csv").exists() else []
    gauss_norm = float(np.sqrt(np.mean(snapshots[0].ion_density ** 2)))
    # The snapshots with a nonzero analytic ion spread, which the ion plots and A6 divide by.
    particles = [p for p in ([] if args.no_particles else read_particle_snapshots(data_dir, deck.ion))
                 if np.all(theory.ion_spread(p.time)[deck.gaussian] > 0)]
    g = deck.gaussian[0]
    wpe = math.sqrt(abs(deck.qom[deck.electron]) * deck.rho_init[deck.electron])
    title = (rf"GaussianExpansion, {len(deck.gaussian)}D ({deck.path.name}): "
             rf"$m_i/m_e$ = {abs(deck.qom[deck.electron]) / deck.qom[deck.ion]:.0f}, "
             rf"$\sigma_0/\Delta x$ = {theory.sigma0[g] / deck.spacing[g]:.1f}, "
             rf"$\Lambda$ = {theory.sigma0[g] * wpe / deck.thermal[deck.electron, g]:.0f}, "
             rf"$\omega_{{pe}}\Delta t$ = {wpe * deck.dt:.2f}")

    written = save(plot_overview(deck, theory, snapshots, diagnostics, title), output_dir, f"{args.prefix}_overview")
    with plt.rc_context(PRINT_STYLE):
        written += save(plot_width_density(deck, theory, snapshots), output_dir, f"{args.prefix}_width_density")
        written += save(plot_energy_share(deck, theory, diagnostics), output_dir, f"{args.prefix}_energy_share")
    if len(deck.gaussian) > 1:
        written += save(plot_density_maps(deck, theory, snapshots, title), output_dir, f"{args.prefix}_density_maps")
    if args.density_3d:
        with plt.rc_context(DENSITY_3D_FONT_SIZES.rc()):
            written += save(plot_density_3d(deck, theory, snapshots), output_dir, f"{args.prefix}_density_3d")
    written += save(plot_profiles(deck, theory, snapshots, title), output_dir, f"{args.prefix}_profiles")
    if snapshots[0].flow is not None:
        with plt.rc_context(PRINT_STYLE):
            written += save(plot_radial_flow(deck, theory, snapshots), output_dir, f"{args.prefix}_radial_flow")
    written += save(plot_conservation(deck, theory, snapshots, diagnostics, gauss, gauss_norm, title), output_dir,
                    f"{args.prefix}_conservation")
    if gauss:
        with plt.rc_context(PRINT_STYLE):
            written += save(plot_energy_gauss(theory, diagnostics, gauss), output_dir, f"{args.prefix}_energy_gauss")
    if particles:
        written += save(plot_ions(deck, theory, particles, title), output_dir, f"{args.prefix}_ions")
        written += save(plot_phase_space(deck, theory, particles, title), output_dir, f"{args.prefix}_phase_space")
    csv_path = output_dir / f"{args.prefix}_snapshots.csv"
    write_snapshot_csv(csv_path, deck, theory, snapshots, diagnostics)
    lines = ["Derived parameters (measured widths)", *derived_parameters(deck, theory), "",
             "Analytic solution", *solution_table(deck, theory), "",
             *summarize(deck, theory, snapshots, diagnostics, gauss, gauss_norm, particles)]
    summary_path = output_dir / f"{args.prefix}_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nWrote:", *(str(path) for path in [*written, csv_path, summary_path]), sep="\n  ")


if __name__ == "__main__":
    main()
