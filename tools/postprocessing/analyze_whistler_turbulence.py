#!/usr/bin/env python3
"""Decaying whistler turbulence (Case = WhistlerTurbulence) analysed as in Section 5 of
docs/user/whistler_turbulence.md.

[G12] S. P. Gary, O. Chang & J. Wang, ApJ 755, 142 (2012).
[S08] S. Saito, S. P. Gary, H. Li & Y. Narita, Phys. Plasmas 15, 102305 (2008).

Usage from the repository root::

    python3 tools/postprocessing/analyze_whistler_turbulence.py --run-dir /path/to/run
    python3 tools/postprocessing/analyze_whistler_turbulence.py --run-dir /path/to/run --render-3d
    python3 tools/postprocessing/analyze_whistler_turbulence.py --run-dir /path/to/run --only spectra temperatures
    python3 tools/postprocessing/analyze_whistler_turbulence.py --run-dir /path/to/run --only render

The deck is the single WhistlerTurbulence deck of --run-dir unless --input names it, and the data are its
SaveDirName unless --data-dir names them. The analyzer reads diagnostics.csv, the VTK field files B and, when the
deck writes them, rho_<e>, J_<e>, PXX_<e>, PYY_<e>, PZZ_<e> of the first electron species e, and the particle
snapshots. Units are those of the case document: times omega_pe*t, wavenumbers k*d_e, fields over B0 = B0z, energy
densities over B0^2/(8*pi), velocities over the electron uth v_te, with omega_pe^2 = sum over the electron
species of rhoINIT*|qom|, d_e = c/omega_pe and |Omega_e| = |qom_e|*B0/c.

Method.  dB = B - B0 z on the nodes of the periodic box, B_k = rfftn(dB)/N with z (along B0) last, so that
sum(W |B_k|^2) = <|dB|^2> with W = 2 for kz > 0 and 1 at kz = 0 and Nyquist. TPIC writes B at a node as the mean
of its eight cells, the filter cos(kx dx/2) cos(ky dy/2) cos(kz dz/2), which the analysis divides out; the modes
it attenuates in energy by more than 4 are dropped (along an axis, k dx > 2*pi/3).

- Energies (5.1): W_B = <|dB|^2>/B0^2, W_E and the kinetic energies of diagnostics.csv, and the rate
  2*gamma_eff = d ln W_B/dt in |Omega_e|, a Savitzky-Golay derivative over --rate-window.
- Electron temperatures (5.3): the papers' T_par = sum P_zz/sum rho_m and T_perp = sum (P_xx + P_yy)/(2 sum
  rho_m) over the nodes, over T0 = T(0), and the thermal ones, without the bulk flow u = J/(rho/(4*pi)) of each
  node, over T0 = m v_te^2; their ratio and the heating rate d(T_par/T0)/d(|Omega_e| t) over W_B.
- Spectra (5.4): the shell-summed S(k_perp) in shells of the fundamental k0, E(k_perp) = S/k_perp of [G12], the
  reduced S(k_par) and |B_k|^2 in the (k_y, k_par) plane at k_x = 0, at the --times. The slopes (5.5) of E
  below (0.4, 0.9) and above (1.2, k_s) and a continuous broken power law over (0.3, k_s) are fitted to the mean
  spectrum of the snapshots within --fit-window of --fit-time.
- Anisotropy (5.6): tan^2 theta_B = sum k_perp^2 |B_k|^2/sum k_par^2 |B_k|^2 (3D isotropic: 2) over all k <=
  k_s, over the cascaded (unseeded) k <= k_s and in the shells 0.7 <= k <= 3 and 0.3 <= k <= 3. The gyrotropy
  index (5.7, 12 azimuth bins) and the normalized magnetic helicity sigma_m (5.8) are taken in the seeded shell,
  between the smallest and the largest seeded |k|, and in the cascade shell above it, up to twice the largest.
- Polarization (5.8): C_par = |dB_z|^2/|dB|^2 and r_B (along z x k_perp) in (|k|, theta) bins, energy weighted
  over every snapshot, against the helical whistler: C_par = sin^2(theta)/2, r_B = 1/2.
- Frequencies (the omega-k spectra that H. Karimabadi, V. Roytershteyn et al., Phys. Plasmas 21, 062308 (2014) show
  in 2D): B_k(t) of the modes with k_perp <= 4 in the k_par planes of --omega-k-kpar and --omega-k-bins, transformed
  in time over the snapshots of each time window, --omega-k-window long and centred on one of the --omega-k-times,
  without a window; the energy of each k_perp shell per unit omega, omega > 0 travelling towards +k_par (along B0).
  Against the cold electron whistler omega = |Omega_e| k_par k d_e^2/(1 + k^2 d_e^2). The field snapshots, dt apart,
  resolve |omega| up to the Nyquist frequency pi/dt and in steps of 2*pi/T, T the window; a whistler above pi/dt is
  seen at its alias, drawn dotted.
- Electron distributions (5.9) of the particle snapshots: f(v_par) and f(v_x), and delta f(v_x, v_par) = (f(t) -
  f(0))/f(0, 0) at the last snapshot, with the largest whistler Landau speed |Omega_e| d_e/2.

k_s is --k-signal, or with --noise-dir (the data of an epsilon0 = 0 run of the same deck) the smaller of
--k-signal and the first k_perp >= 1 where S(k_perp) falls below 10 times that of the control run at the same
cycle (5.5); the control S(k_perp) is drawn on the spectra.

Into --output-dir (default the run directory) go, each as a PNG and a PDF, named after --prefix:

    <prefix>_energy        W_B and W_E against time, 2*gamma_eff, the energy exchange and the total-energy change
    <prefix>_temperatures  T_par/T0 and T_perp/T0 of both definitions, T_par/T_perp and the heating rate
    <prefix>_electron_temperatures
                           the thermal T_par/T0, T_perp/T0 and T_par/T_perp of the electrons alone, as a print
                           figure (the look of analyze_gaussianExpansion.py), without titles
    <prefix>_temperature_energy
                           a print figure of two panels. Left: the thermal T_par/T0 and T_perp/T0 of the electrons
                           and, on the right axis, T_par/T_perp. Right, on one axis over the initial total energy
                           E0: the magnetic energy of dB (the uniform B0 keeps its own, constant), the electric
                           energy, the electron and ion kinetic energies and the total-energy error |E_tot - E0|
    <prefix>_spectra       the reduced S(k_perp) and S(k_par) at the --times on one scale, as a print figure, with the
                           broken power law fitted to E(k_perp) drawn as S (slopes one more) and the largest seeded
                           k (and the control spectrum)
    <prefix>_spectrum_2d   |B_k|^2 in the (k_y, k_par) plane at k_x = 0 at the --times, as [G12] Fig. 3
    <prefix>_anisotropy    tan^2 theta_B, gyrotropy and helicity against time; tan^2 theta_B against the
                           cascaded energy, as [S08] Fig. 3
    <prefix>_polarization  C_par and r_B against theta in three |k| shells
    <prefix>_omega_k       omega against k_perp in the --omega-k-kpar planes (columns), one row per time window: the
                           energy per unit omega of each k_perp shell over its peak, with the whistler
    <prefix>_omega_power   the energy per unit omega against omega in the --omega-k-bins, one curve per time window
    <prefix>_vdf           electron f(v_par) and f(v_x) at the particle snapshots, delta f at the last one
    <prefix>_render_3d     with --render-3d (needs pyvista): a volume rendering on white of the current density
                           |curl dB| over its rms at the --render-times, the modes above k_s rolled off as
                           exp(-(k/k_s)^4) to keep the particle noise out, B0 vertical (the arrow beside the box),
                           the x, y, z arrows of the view in the lower left corner
    <prefix>_render_density_electrons, <prefix>_render_density_ions
                           with --render-3d: the same renderings of the electron and of the ion density fluctuation
                           dn/n0, both on one colour scale, red where compressed and blue where rarefied; their rms
                           go to the summary. The electrons are rho_<e>; the ions are rho_<i> when the deck writes
                           it, else, unless --no-particles, the particle snapshot nearest each time deposited on the
                           nearest node

and <prefix>_timeseries.csv (every quantity at every field snapshot) and <prefix>_summary.txt (the step-0
checks of Section 3.8, the slopes, the heating and the omega-k peaks, also printed).

Parts.  --only PART ... draws those groups of plots alone and reads only the data they need:
    energy        <prefix>_energy                                  diagnostics.csv
    temperatures  <prefix>_temperatures, _electron_temperatures,  diagnostics.csv and the electron moments of every
                  _temperature_energy                              snapshot
    spectra       <prefix>_spectra, _spectrum_2d                   B at the --times and within --fit-window
    anisotropy    <prefix>_anisotropy                              B at every snapshot
    polarization  <prefix>_polarization                            B at every snapshot
    omega-k       <prefix>_omega_k, _omega_power                   B at every snapshot
    vdf           <prefix>_vdf                                     the electron particles
    render        <prefix>_render_3d, _render_density_<species>    B and rho at the --render-times, the ion particles
Without --only every part but render runs, and --render-3d adds render to either. Only a run of every part but vdf
and render writes the timeseries and the summary, so a partial run leaves those of the last full one in place.
"""

from __future__ import annotations

import argparse
import csv
import functools
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # source-checkout imports
from tpic_tools.fields import field_data_exists, open_field_reader
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.particles import iter_species
from tpic_tools.runs import read_csv

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


@dataclass
class FontSizes:
    """Font sizes of every plot, in points."""

    axis_label: float = 13.0
    tick_label: float = 11.0
    legend: float = 10.0
    title: float = 12.0

    def rc(self) -> dict:
        return {"axes.titlesize": self.title, "axes.labelsize": self.axis_label, "xtick.labelsize": self.tick_label,
                "ytick.labelsize": self.tick_label, "legend.fontsize": self.legend}


FONT_SIZES = FontSizes()
# The print figures, in the look of those of analyze_gaussianExpansion.py and analyze_weibel.py: larger fonts, a
# closed axes box and a framed legend, no titles, the simulation data as thin lines with dots.
PRINT_FONT_SIZES = FontSizes(axis_label=22.0, tick_label=20.0, legend=18.0, title=22.0)
PRINT_STYLE = {**PRINT_FONT_SIZES.rc(), "axes.spines.top": True, "axes.spines.right": True, "legend.frameon": True}
SIMULATION_LINE = dict(linewidth=1.3, marker="o", markersize=6.0)
PAR_COLOR, PERP_COLOR, ANISOTROPY_COLOR = "#c0392b", "#1f4a9e", "#2e8b57"
ENERGY_COLORS = {"magnetic": "#6a3d9a", "electric": "#eb6834", "electrons": "#1f4a9e", "ions": "#c0392b",
                 "error": "0.45"}
SPECTRUM_FLOOR = 1e-12  # spectra values below this fraction of the largest are left out of the y range
SEEDED_LINE = dict(color="0.25", linewidth=2.5, linestyle=":")  # the largest seeded k
plt.rcParams.update({
    "axes.labelsize": FONT_SIZES.axis_label, "axes.titlesize": FONT_SIZES.title,
    "xtick.labelsize": FONT_SIZES.tick_label, "ytick.labelsize": FONT_SIZES.tick_label,
    "legend.fontsize": FONT_SIZES.legend,
})
AXES = "xyz"
FOUR_PI = 4.0 * math.pi
NOISE_FACTOR = 10.0   # a shell is signal above this multiple of the control-run spectrum (Section 5.5)
AZIMUTH_BINS = 12     # of the gyrotropy index (Section 5.7)
THETA_EDGES = np.linspace(0.0, 90.0, 10)
VDF_LIMIT, VDF_BINS = 6.0, 121  # velocity range in v_te and bins of the distributions
PIECE_THREADS = min(16, len(os.sched_getaffinity(0)))  # particle pieces read at once
PIXELS_3D = 900
AXES_3D = 0.32  # side of the lower left corner that holds the x, y, z arrows, in window fractions
# The 3D render: |curl dB|/rms on RENDER_RANGE, transparent below 1.7 rms (opacity at 8 equal steps), dark where
# opaque on the white background.
RENDER_RANGE = (0.0, 4.0)
RENDER_OPACITY = [0.0, 0.0, 0.0, 0.1, 0.45, 0.8, 0.95, 1.0]
RENDER_CMAP = "inferno_r"
# The density renders: dn/n0 within DENSITY_RANGE times the largest rms of the panels, with the opacity of the
# current mirrored about 0 (transparent within 1.7 times that rms).
DENSITY_RANGE = 4.0
DENSITY_OPACITY = RENDER_OPACITY[:0:-1] + RENDER_OPACITY
DENSITY_CMAP = "RdBu_r"
# The omega-k spectra: B_k(t) of the modes with k_perp <= OMEGA_K_PERP, transformed without a window over the
# snapshots of each time window and zero padded OMEGA_K_PAD times (smoother curves, not a finer resolution); a window needs
# OMEGA_K_MIN_SNAPSHOTS evenly spaced snapshots. The maps span OMEGA_K_FLOOR to 1 of the peak of each k_perp shell.
# The parts of --only, each a group of plots; render is drawn only on request. The timeseries and the summary need
# SUMMARY_PARTS.
PARTS = ("energy", "temperatures", "spectra", "anisotropy", "polarization", "omega-k", "vdf", "render")
SUMMARY_PARTS = {"energy", "temperatures", "spectra", "anisotropy", "polarization", "omega-k"}
OMEGA_K_PERP = 4.0
OMEGA_K_PAD = 4
OMEGA_K_MIN_SNAPSHOTS = 8
OMEGA_K_FLOOR = 0.1  # a shell spans about a decade in omega: a deeper floor washes the ridges out


@dataclass
class Deck:
    path: Path
    save_dir: str
    lengths: np.ndarray
    cells: np.ndarray
    b0: float
    epsilon: float
    electron: int
    qom_e: float
    v_te: float
    omega_pe: float
    d_e: float
    omega_ce: float
    ion: int | None = None  # the first species with qom > 0

    @property
    def spacing(self) -> np.ndarray:
        return self.lengths / self.cells


def read_deck(path: Path) -> Deck:
    values = read_input(path)
    if values.get("Case", [""])[0] != "WhistlerTurbulence":
        raise ValueError(f"{path}: Case is not WhistlerTurbulence")
    if not all(periodic_axes(values)):
        raise ValueError(f"{path}: the spectra need a box periodic along every axis")
    b0 = [float(values.get(f"B0{a}", ["0"])[0]) for a in AXES]
    ns = int(values["ns"][0])
    qom = [float(token) for token in values["qom"][:ns]]
    rho = [float(token) for token in values["rhoINIT"][:ns]]
    electrons = [s for s in range(ns) if qom[s] < 0.0]
    ions = [s for s in range(ns) if qom[s] > 0.0]
    c = float(values.get("c", ["1"])[0])
    omega_pe = math.sqrt(sum(rho[s] * -qom[s] for s in electrons))
    e = electrons[0]
    return Deck(path=path, save_dir=values.get("SaveDirName", ["data"])[0],
                lengths=np.array([float(values[f"L{a}"][0]) for a in AXES]),
                cells=np.array([int(values[f"n{a}c"][0]) for a in AXES]), b0=b0[2],
                epsilon=float(values.get("WhistlerTurbulenceEpsilon", ["0"])[0]), electron=e, qom_e=qom[e],
                v_te=float(values["uth"][e]), omega_pe=omega_pe, d_e=c / omega_pe, omega_ce=-qom[e] * b0[2] / c,
                ion=ions[0] if ions else None)


def find_input(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    decks = [path for path in sorted(run_dir.glob("*.inp"))
             if read_input(path).get("Case", [""])[0] == "WhistlerTurbulence"]
    if len(decks) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one WhistlerTurbulence deck, found {len(decks)}; use --input")
    return decks[0]


def seed_indices() -> np.ndarray:
    """The integer wavevectors of the 150 seeded modes, as WhistlerTurbulenceModes::for_each_index lists them."""
    return np.array([(nx, ny, nz) for nz in range(-3, 4) for ny in range(-3, 4) for nx in range(-3, 4)
                     if nz != 0 and (nx == 0 or ny == 0 or nx == ny or nx == -ny)])


class Fourier:
    """The rfftn modes of an (x, y, z) node field and the masks and bins of the analysis, k in 1/d_e."""

    def __init__(self, deck: Deck):
        n, d = deck.cells, deck.spacing
        axes = [2.0 * math.pi * np.fft.fftfreq(n[a], d[a]) for a in (0, 1)] + [2.0 * math.pi * np.fft.rfftfreq(n[2], d[2])]
        transfer = np.prod(np.meshgrid(*[np.cos(0.5 * k * d[a]) for a, k in enumerate(axes)], indexing="ij"), axis=0)
        self.kx, self.ky, self.kz = np.meshgrid(*[k * deck.d_e for k in axes], indexing="ij", sparse=True)
        self.shape, self.cells = transfer.shape, tuple(int(c) for c in n)
        weight = np.full(self.shape[2], 2.0)
        weight[0] = 1.0
        if n[2] % 2 == 0:
            weight[-1] = 1.0
        self.weight = np.broadcast_to(weight, self.shape)
        self.inverse_transfer = np.where(transfer**2 >= 0.25, 1.0 / np.where(transfer > 0, transfer, 1.0), 0.0)
        self.kperp = np.broadcast_to(np.hypot(self.kx, self.ky), self.shape)
        self.kpar = np.broadcast_to(self.kz, self.shape)
        self.k = np.hypot(self.kperp, self.kpar)
        self.dk_perp = 2.0 * math.pi * deck.d_e / max(deck.lengths[0], deck.lengths[1])
        self.dk_par = 2.0 * math.pi * deck.d_e / deck.lengths[2]
        self.perp_bin = np.floor(self.kperp / self.dk_perp + 0.5).astype(np.int64).ravel()
        self.perp_bins = int(self.perp_bin.max()) + 1
        self.k_perp_axis = self.dk_perp * np.arange(self.perp_bins)
        # The seeded modes, each stored once: a mode with kz < 0 is the conjugate of -k.
        self.seeded = np.zeros(self.shape, bool)
        for nx, ny, nz in seed_indices():
            sign = 1 if nz > 0 else -1
            self.seeded[(sign * nx) % n[0], (sign * ny) % n[1], sign * nz] = True
        seeded_k = self.k[self.seeded]
        self.seed_shell = (seeded_k.min(), seeded_k.max())
        self.cascade_shell = (seeded_k.max(), 2.0 * seeded_k.max())
        self.shells = {"seeded": (self.k >= self.seed_shell[0]) & (self.k <= self.seed_shell[1]) & (self.kperp > 0),
                       "cascade": (self.k > self.cascade_shell[0]) & (self.k <= self.cascade_shell[1]) & (self.kperp > 0)}
        self.azimuth_bin = np.minimum((np.mod(np.arctan2(self.ky, self.kx), math.pi) / math.pi * AZIMUTH_BINS)
                                      .astype(np.int64), AZIMUTH_BINS - 1)
        self.azimuth_bin = np.broadcast_to(self.azimuth_bin, self.shape)
        theta = np.degrees(np.arctan2(self.kperp, self.kpar))
        self.polarization_edges = np.array([self.seed_shell[0], self.seed_shell[1], 2 * self.seed_shell[1],
                                            4 * self.seed_shell[1]])
        shell = np.digitize(self.k, self.polarization_edges) - 1
        angle = np.minimum(np.digitize(theta, THETA_EDGES) - 1, THETA_EDGES.size - 2)
        valid = (shell >= 0) & (shell < 3) & (self.kperp > 0)
        self.polarization_bin = np.where(valid, shell * (THETA_EDGES.size - 1) + angle, -1).ravel()

    def modes(self, vector: np.ndarray, b0: float) -> np.ndarray:
        """B_k/B0 of the fluctuation of a (3, nx, ny, nz) node field, the node filter divided out."""
        fluctuation = vector.copy()
        fluctuation[2] -= b0
        return np.stack([np.fft.rfftn(c) for c in fluctuation]) * (self.inverse_transfer / (fluctuation[0].size * b0))

    def energy(self, modes: np.ndarray) -> np.ndarray:
        """W |B_k|^2 per mode, summing to <|dB|^2>/B0^2."""
        return self.weight * np.sum(np.abs(modes) ** 2, axis=0)

    def perp_spectrum(self, energy: np.ndarray) -> np.ndarray:
        """S(k_perp) on k_perp_axis: the energy of each shell over its width."""
        return np.bincount(self.perp_bin, energy.ravel(), self.perp_bins) / self.dk_perp

    def cross(self, modes: np.ndarray) -> np.ndarray:
        """k x B_k; i times it is the transform of curl dB."""
        k = (self.kx, self.ky, self.kz)
        return np.stack([k[(a + 1) % 3] * modes[(a + 2) % 3] - k[(a + 2) % 3] * modes[(a + 1) % 3] for a in range(3)])

    def anisotropy(self, energy: np.ndarray, mask: np.ndarray) -> float:
        return float((energy * self.kperp**2)[mask].sum() / (energy * self.kpar**2)[mask].sum())

    def gyrotropy(self, energy: np.ndarray, shell: np.ndarray) -> float:
        """std/mean over azimuth of the energy per Fourier mode in the shell (Section 5.7)."""
        per_bin = np.bincount(self.azimuth_bin[shell], energy[shell], AZIMUTH_BINS)
        modes = np.bincount(self.azimuth_bin[shell], self.weight[shell], AZIMUTH_BINS)
        profile = per_bin / np.maximum(modes, 1e-30)
        return float(profile.std() / profile.mean())

    def helicity(self, modes: np.ndarray, shell: np.ndarray) -> float:
        """sigma_m = sum k H_k/sum |B_k|^2 with the Coulomb-gauge A_k = i k x B_k/k^2 (Section 5.8)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            helicity = np.real(np.sum(1j * self.cross(modes) * np.conj(modes), axis=0)) / self.k
        return float((self.weight * helicity)[shell].sum() / self.energy(modes)[shell].sum())

    def polarization(self, modes: np.ndarray) -> np.ndarray:
        """Energy-weighted |dB_z|^2, |dB_B|^2 and |dB|^2 of the (|k|, theta) bins, (3, shells*thetas)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            along_b = (-modes[0] * self.ky + modes[1] * self.kx) / self.kperp
        terms = [np.abs(modes[2]) ** 2, np.abs(along_b) ** 2, np.sum(np.abs(modes) ** 2, axis=0)]
        valid = self.polarization_bin >= 0
        size = 3 * (THETA_EDGES.size - 1)
        return np.stack([np.bincount(self.polarization_bin[valid], (self.weight * t).ravel()[valid], size)
                         for t in terms])

    def current(self, modes: np.ndarray, k_max: float) -> np.ndarray:
        """|curl dB| d_e/B0 on the nodes, (x, y, z), of the modes rolled off as exp(-(k/k_max)^4): the curl
        weights the particle noise above k_max by k^2."""
        curl = 1j * math.prod(self.cells) * self.cross(modes * np.exp(-(self.k / k_max) ** 4))
        return np.sqrt(sum(np.fft.irfftn(c, s=self.cells, axes=(0, 1, 2)) ** 2 for c in curl))

    def low_pass(self, field: np.ndarray, k_max: float) -> np.ndarray:
        """An (x, y, z) node field with its modes rolled off as exp(-(k/k_max)^4), as current() rolls off dB."""
        return np.fft.irfftn(np.fft.rfftn(field) * np.exp(-(self.k / k_max) ** 4), s=self.cells, axes=(0, 1, 2))


def whistler_frequency(k_perp, k_par):
    """omega/|Omega_e| of the cold electron whistler, k_par k d_e^2/(1 + k^2 d_e^2) (ions at rest, no displacement
    current), k in 1/d_e."""
    k = np.hypot(k_perp, k_par)
    return k_par * k / (1.0 + k**2)


@dataclass
class OmegaKSpectrum:
    """The magnetic energy of one time window against omega, in |Omega_e|; omega > 0 travels towards +k_par."""

    times: np.ndarray   # omega_pe*t of the snapshots
    omega: np.ndarray   # ascending
    nyquist: float      # pi/dt of the snapshots
    resolution: float   # 2*pi/(snapshots * dt)
    power: np.ndarray   # [plane, shell, omega]: <|dB|^2>/B0^2 of the shell (both signs of k_par) per unit omega


class OmegaK:
    """B_k(t)/B0 of the modes with k_perp <= OMEGA_K_PERP in the k_par planes nearest the requested ones, kept at
    every field snapshot for the transforms in time, and their k_perp shells of width k0, as in S(k_perp)."""

    def __init__(self, fourier: Fourier, k_par):
        kx, ky, kz = (np.ravel(k) for k in (fourier.kx, fourier.ky, fourier.kz))
        shells = int(OMEGA_K_PERP / fourier.dk_perp + 1e-9) + 1
        reach = (shells - 0.5) * fourier.dk_perp
        x, y = (np.flatnonzero(np.abs(k) < reach) for k in (kx, ky))
        self.planes = sorted({int(np.argmin(np.abs(kz - k))) for k in k_par})
        self.index = (slice(None), *np.ix_(x, y, self.planes))
        self.k_par = kz[self.planes]
        self.weight = np.ravel(fourier.weight[0, 0])[self.planes]  # 2 counts the conjugate -k of a plane kz > 0
        shell = np.floor(np.hypot(kx[x, None], ky[None, y]) / fourier.dk_perp + 0.5).astype(np.int64).ravel()
        self.shells = (shell[None, :] == np.arange(shells)[:, None]).astype(float)  # (shell, mode)
        self.k_perp_axis = fourier.dk_perp * np.arange(shells)
        self.times, self.series = [], []

    def add(self, time: float, modes: np.ndarray) -> None:
        self.times.append(time)
        self.series.append(modes[self.index])

    def plane(self, k_par: float) -> int:
        return int(np.argmin(np.abs(self.k_par - k_par)))

    def shell(self, k_perp: float) -> int:
        return min(int(k_perp / self.k_perp_axis[1] + 0.5), self.k_perp_axis.size - 1)

    def spectrum(self, start: float, end: float, omega_ce: float) -> OmegaKSpectrum | None:
        """The spectrum of the snapshots within [start, end] (omega_pe*t), or None without OMEGA_K_MIN_SNAPSHOTS
        evenly spaced ones; omega_ce is |Omega_e|/omega_pe."""
        times = np.array(self.times)
        keep = np.flatnonzero((times >= start - 1e-6) & (times <= end + 1e-6))
        t = times[keep]
        if t.size < OMEGA_K_MIN_SNAPSHOTS or not np.allclose(np.diff(t), t[1] - t[0], rtol=1e-3):
            return None
        dt = (t[-1] - t[0]) / (t.size - 1)
        size = OMEGA_K_PAD * t.size
        series = np.stack([self.series[i] for i in keep])
        # B_k ~ exp(-i omega t) for a wave exp(i (k.x - omega t)): numpy's exp(-2 pi i f t) puts it at f = -omega/2pi.
        power = np.sum(np.abs(np.fft.fft(series, n=size, axis=0)) ** 2, axis=1).reshape(size, -1, self.k_par.size)
        omega = -2.0 * math.pi * np.fft.fftfreq(size, dt) / omega_ce
        order = np.argsort(omega)
        step = 2.0 * math.pi / (size * dt * omega_ce)
        # Parseval: summed over omega, power/(size * snapshots) is the mean of |B_k|^2 over the window.
        density = power[order] * self.weight / (size * t.size * step)
        return OmegaKSpectrum(times=t, omega=omega[order], nyquist=math.pi / (dt * omega_ce),
                              resolution=2.0 * math.pi / (t.size * dt * omega_ce),
                              power=np.einsum("sm,jmp->psj", self.shells, density))


@dataclass
class Snapshot:
    """A field snapshot: the moments give the temperatures and B the rest, each None when not read."""

    cycle: int
    time: float
    magnetic: float | None = None            # <|dB|^2>/B0^2 of the spectrum
    cascaded: float | None = None            # its unseeded part with k <= k_s
    k_signal: float | None = None
    s_perp: np.ndarray | None = None
    s_par: np.ndarray | None = None
    tan2: dict | None = None
    gyrotropy: dict | None = None
    helicity: dict | None = None
    temperatures: dict | None = None  # paper/thermal T_par and T_perp in c^2


def read_vector(reader, name: str, step: int) -> np.ndarray:
    """The (3, nx, ny, nz) components of a node vector without the repeated periodic end nodes."""
    return np.stack([reader.read(name + a, step_selection=[step, 1])[:-1, :-1, :-1].T for a in AXES])


def read_scalar(reader, name: str, step: int) -> np.ndarray:
    return reader.read(name, step_selection=[step, 1])[:-1, :-1, :-1].T


def electron_temperatures(readers: dict, deck: Deck, step: int) -> dict:
    """The papers' and the thermal T_par, T_perp in c^2 from the node moments of the electrons."""
    e = deck.electron
    charge = read_scalar(readers["rho"], f"rho_{e}", step) / FOUR_PI
    current = read_vector(readers["J"], f"J_{e}", step)
    pressure = [read_scalar(readers[name], f"{name}_{e}", step) for name in ("PXX", "PYY", "PZZ")]
    mass = (charge / deck.qom_e).sum()
    safe = np.where(charge != 0.0, charge, np.inf)
    flow = current**2 / (safe * deck.qom_e)  # rho_m u_i^2 of each node
    return {"paper_par": pressure[2].sum() / mass, "paper_perp": (pressure[0] + pressure[1]).sum() / (2 * mass),
            "thermal_par": (pressure[2] - flow[2]).sum() / mass,
            "thermal_perp": (pressure[0] + pressure[1] - flow[0] - flow[1]).sum() / (2 * mass)}


def noise_spectra(noise_dir: Path | None, deck: Deck, fourier: Fourier) -> dict:
    """S(k_perp) of the control run per cycle, empty without one."""
    if noise_dir is None:
        return {}
    spectra = {}
    with open_field_reader(noise_dir / "B") as reader:
        for step, (_, epoch) in enumerate(reader.records):
            modes = fourier.modes(read_vector(reader, "B", step), deck.b0)
            spectra[epoch[0]] = fourier.perp_spectrum(fourier.energy(modes))
    return spectra


def signal_limit(fourier: Fourier, s_perp: np.ndarray, noise: np.ndarray | None, k_signal: float) -> float:
    """k_s: k_signal, or the first k_perp >= 1 where S falls below NOISE_FACTOR times the control S."""
    below = (fourier.k_perp_axis >= 1.0) & (s_perp < NOISE_FACTOR * noise) if noise is not None else []
    return min(k_signal, float(fourier.k_perp_axis[below][0])) if np.any(below) else k_signal


def analyze_fields(deck: Deck, data_dir: Path, fourier: Fourier, noise: dict, k_signal: float, b_steps: set,
                   moments: bool, slice_steps: set, render_steps: set, omega_k: OmegaK | None):
    """The field snapshots: B at b_steps and, with moments, the electron moments at every step. Returns one Snapshot
    per step (None where nothing is read), the polarization sums, the k_x = 0 planes of the slice steps, and at the
    render steps the currents and the (omega_pe*t, dn/n0) of the written electron and ion densities; the modes go to
    omega_k unless it is None."""
    names = ("rho", "J", "PXX", "PYY", "PZZ")
    moments = moments and all(field_data_exists(data_dir, f"{name}_{deck.electron}") for name in names)
    readers = {"B": open_field_reader(data_dir / "B")}
    if moments:
        readers.update({name: open_field_reader(data_dir / f"{name}_{deck.electron}") for name in names})
    density_readers = {kind: open_field_reader(data_dir / f"rho_{s}")
                       for kind, s in (("electrons", deck.electron), ("ions", deck.ion))
                       if render_steps and s is not None and field_data_exists(data_dir, f"rho_{s}")}
    snapshots, planes, currents, densities = [], {}, {}, {}
    polarization = 0.0
    for step, (_, epoch) in enumerate(readers["B"].records):
        cycle, time = epoch[0], deck.omega_pe * epoch[1]
        if step not in b_steps and not moments:
            snapshots.append(None)
            continue
        snapshot = Snapshot(cycle=cycle, time=time,
                            temperatures=electron_temperatures(readers, deck, step) if moments else None)
        snapshots.append(snapshot)
        if step not in b_steps:
            continue
        modes = fourier.modes(read_vector(readers["B"], "B", step), deck.b0)
        energy = fourier.energy(modes)
        s_perp = fourier.perp_spectrum(energy)
        k_s = signal_limit(fourier, s_perp, noise.get(cycle), k_signal)
        signal = fourier.k <= k_s
        cascaded = signal & ~fourier.seeded
        snapshot.magnetic, snapshot.cascaded = float(energy.sum()), float(energy[cascaded].sum())
        snapshot.k_signal, snapshot.s_perp, snapshot.s_par = k_s, s_perp, energy.sum(axis=(0, 1)) / fourier.dk_par
        snapshot.tan2 = {"all": fourier.anisotropy(energy, signal), "cascaded": fourier.anisotropy(energy, cascaded),
                         "0.7-3": fourier.anisotropy(energy, (fourier.k >= 0.7) & (fourier.k <= 3.0)),
                         "0.3-3": fourier.anisotropy(energy, (fourier.k >= 0.3) & (fourier.k <= 3.0))}
        snapshot.gyrotropy = {name: fourier.gyrotropy(energy, shell) for name, shell in fourier.shells.items()}
        snapshot.helicity = {name: fourier.helicity(modes, shell) for name, shell in fourier.shells.items()}
        polarization = polarization + fourier.polarization(modes)
        if omega_k is not None:
            omega_k.add(time, modes)
        if step in slice_steps:
            planes[step] = np.sum(np.abs(modes[:, 0]) ** 2, axis=0)
        if step in render_steps:
            currents[step] = fourier.current(modes, k_s)
            densities[step] = {kind: (time, density_fluctuation(fourier, read_scalar(reader, reader.base.name, step),
                                                                k_s))
                               for kind, reader in density_readers.items()}
    return snapshots, polarization, planes, currents, densities


def density_fluctuation(fourier: Fourier, density: np.ndarray, k_max: float) -> np.ndarray:
    """dn/n0 of an (x, y, z) node density, the modes above k_max rolled off to keep the particle noise out."""
    return fourier.low_pass(density / density.mean() - 1.0, k_max)


def particle_snapshots(data_dir: Path, deck: Deck) -> list[tuple[float, list[Path]]]:
    """omega_pe*t and the piece files of every particle snapshot, in time order."""
    result = []
    for manifest in sorted(data_dir.glob("particles_t*.pvtp")):
        root = ET.parse(manifest).getroot()
        time = deck.omega_pe * float(next(a.text for a in root.iter("DataArray") if a.get("Name") == "TimeValue"))
        result.append((time, [manifest.parent / piece.get("Source") for piece in root.iter("Piece")]))
    return result


def summed(function, pieces):
    """The tuples function(piece) summed element by element over the pieces, PIECE_THREADS pieces at a time: a
    snapshot (1.4e10 particles in 32 pieces at 300^3 cells and 256 per cell) is never held whole."""
    with ThreadPoolExecutor(PIECE_THREADS) as pool:
        return functools.reduce(lambda a, b: tuple(x + y for x, y in zip(a, b)), pool.map(function, pieces))


def piece_distributions(path: Path, deck: Deck):
    """The electron counts of one piece over v_par, over v_x and over (v_x, v_par) in v_te, and their number."""
    edges = np.linspace(-VDF_LIMIT, VDF_LIMIT, VDF_BINS + 1)
    par, perp, plane, count = np.zeros(VDF_BINS), np.zeros(VDF_BINS), np.zeros((VDF_BINS, VDF_BINS)), 0
    for arrays in iter_species(path, ("velocity",), deck.electron):
        v = arrays["velocity"] / deck.v_te
        par += np.histogram(v[:, 2], VDF_BINS, (-VDF_LIMIT, VDF_LIMIT))[0]
        perp += np.histogram(v[:, 0], VDF_BINS, (-VDF_LIMIT, VDF_LIMIT))[0]
        plane += np.histogram2d(v[:, 0], v[:, 2], (edges, edges))[0]
        count += len(v)
    return par, perp, plane, count


def read_distributions(data_dir: Path, deck: Deck):
    """Per particle snapshot: omega_pe*t, f(v_par), f(v_x) (unit area over the electrons within VDF_LIMIT) and the
    2D histogram over (v_x, v_par) over all electrons."""
    edges = np.linspace(-VDF_LIMIT, VDF_LIMIT, VDF_BINS + 1)
    width = np.diff(edges)
    result = []
    for time, pieces in particle_snapshots(data_dir, deck):
        par, perp, plane, count = summed(lambda path: piece_distributions(path, deck), pieces)
        result.append((time, par / width / par.sum(), perp / width / perp.sum(), plane / count))
    return edges, result


def piece_density(path: Path, deck: Deck, species: int):
    """The charge of one species of a piece on the nearest nodes of the periodic box, flat over (x, y, z)."""
    n = deck.cells
    density = np.zeros(math.prod(n))
    for arrays in iter_species(path, ("position", "charge"), species):
        node = np.floor(arrays["position"] / deck.spacing + 0.5).astype(np.int64) % n
        density += np.bincount((node[:, 0] * n[1] + node[:, 1]) * n[2] + node[:, 2], arrays["charge"][:, 0],
                               density.size)
    return (density,)


def deposit_density(data_dir: Path, deck: Deck, fourier: Fourier, species: int, time: float, k_max: float):
    """(omega_pe*t, dn/n0) of a species from the particle snapshot nearest time."""
    snapshot_time, pieces = min(particle_snapshots(data_dir, deck), key=lambda snapshot: abs(snapshot[0] - time))
    density, = summed(lambda path: piece_density(path, deck, species), pieces)
    return snapshot_time, density_fluctuation(fourier, density.reshape(tuple(deck.cells)), k_max)


def powerlaw_fit(k: np.ndarray, spectrum: np.ndarray, low: float, high: float):
    """Slope of log E against log k on [low, high] and its 1-sigma error; NaN with fewer than 3 shells."""
    keep = (k >= low) & (k <= high) & (spectrum > 0)
    if np.count_nonzero(keep) < 3:
        return math.nan, math.nan
    p, cov = np.polyfit(np.log(k[keep]), np.log(spectrum[keep]), 1, cov=True)
    return float(p[0]), float(math.sqrt(cov[0, 0]))


def broken_powerlaw_fit(k: np.ndarray, spectrum: np.ndarray, low: float, high: float):
    """(A, slope below, slope above, k_b) of a continuous broken power law on [low, high], or None."""
    keep = (k >= low) & (k <= high) & (spectrum > 0)
    model = lambda lk, la, a1, a2, lkb: la + np.where(lk < lkb, a1, a2) * (lk - lkb)
    try:
        p, _ = curve_fit(model, np.log(k[keep]), np.log(spectrum[keep]),
                         p0=(np.log(np.interp(1.0, k[keep], spectrum[keep])), -3.0, -5.0, 0.0))
    except (RuntimeError, TypeError, ValueError):
        return None
    return float(np.exp(p[0])), float(p[1]), float(p[2]), float(np.exp(p[3]))


def pick(times: np.ndarray, targets) -> list[int]:
    """The indices of the snapshots nearest each target time, once each, in time order."""
    return sorted({int(np.argmin(np.abs(times - t))) for t in targets if t <= times[-1] + 1e-9})


def column(rows, key) -> np.ndarray:
    return np.array([float(row[key]) for row in rows])


def save(figure, output_dir: Path, name: str) -> list[Path]:
    paths = [output_dir / f"{name}.{extension}" for extension in ("png", "pdf")]
    for path in paths:
        figure.savefig(path, dpi=200)
    plt.close(figure)
    return paths


def time_colors(count: int):
    return plt.cm.viridis(np.linspace(0.0, 0.9, count))


def plot_energy(deck: Deck, diagnostics, rate_window: float):
    unit = deck.b0**2 * np.prod(deck.lengths) / FOUR_PI / 2.0  # B0^2 V/(8 pi)
    t = deck.omega_pe * column(diagnostics, "time")
    magnetic = column(diagnostics, "magnetic_energy") / unit - 1.0
    electric = column(diagnostics, "electric_energy") / unit
    kinetic = [column(diagnostics, key) / unit for key in diagnostics[0] if key.startswith("kinetic_energy_")]
    total = column(diagnostics, "total_energy") / unit
    window = max(5, int(round(rate_window / np.median(np.diff(t)))) | 1)
    rate = savgol_filter(magnetic, window, 2, deriv=1, delta=float(np.median(np.diff(t)))) / \
        savgol_filter(magnetic, window, 2) * deck.omega_pe / deck.omega_ce
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(t, magnetic / magnetic[0], color="#1f4a9e", lw=1.8, label=r"$W_B/W_B(0)$")
    ax.semilogy(t, electric / magnetic[0], color="#eb6834", lw=1.4, label=r"$W_E/W_B(0)$")
    ax.set(ylabel="field energy", title=r"Field energies ([G12] Fig. 1a)")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(t, rate, color="#1f4a9e", lw=1.6)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set(ylabel=r"$(dW_B/dt)/W_B\ [|\Omega_e|]$", title=rf"$2\gamma_\mathrm{{eff}}$ ({rate_window:g}/$\omega_{{pe}}$ window)")
    ax = axes[1, 0]
    ax.plot(t, -(magnetic + electric - magnetic[0] - electric[0]), color="#1f4a9e", lw=1.8,
            label=r"$-\Delta(W_B + W_E)$")
    for s, (values, color) in enumerate(zip(kinetic, ("#d62728", "#2ca02c", "C4", "C5"))):
        name = "electrons" if s == deck.electron else f"species {s}"
        ax.plot(t, values - values[0], color=color, lw=1.6, label=rf"$\Delta K$, {name}")
    ax.set(ylabel=r"energy [$B_0^2/8\pi$]", title="Energy exchange")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(t, (total - total[0]) / magnetic[0], color="#1f4a9e", lw=1.4)
    ax.set(ylabel=r"$(E_\mathrm{tot} - E_\mathrm{tot}(0))/W_B(0)$", title="Total-energy change")
    for ax in axes.ravel():
        ax.set_xlabel(r"$\omega_{pe}t$")
        ax.grid(True, alpha=0.3)
    return figure, t, magnetic, total


def initial_temperatures(deck: Deck, first: dict) -> dict:
    """T0 of T_par and T_perp per definition: the papers' T(0), and m v_te^2 for the thermal one."""
    return {"paper": (first["paper_par"], first["paper_perp"]), "thermal": (deck.v_te**2,) * 2}


def plot_temperatures(deck: Deck, snapshots, energy_t, magnetic):
    t = np.array([s.time for s in snapshots])
    temperature = {key: np.array([s.temperatures[key] for s in snapshots]) for key in snapshots[0].temperatures}
    t0 = initial_temperatures(deck, snapshots[0].temperatures)
    style = {"paper": dict(ls="--", lw=1.5), "thermal": dict(ls="-", lw=1.9)}
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), constrained_layout=True)
    for definition in ("thermal", "paper"):
        par = temperature[f"{definition}_par"] / t0[definition][0]
        perp = temperature[f"{definition}_perp"] / t0[definition][1]
        label = "thermal" if definition == "thermal" else r"papers ($T_0 = T(0)$)"
        axes[0].plot(t, par, color="#d62728", label=rf"$T_\parallel$, {label}", **style[definition])
        axes[0].plot(t, perp, color="#1f77b4", label=rf"$T_\perp$, {label}", **style[definition])
        axes[1].plot(t, temperature[f"{definition}_par"] / temperature[f"{definition}_perp"], color="0.2",
                     label=label, **style[definition])
        window = min(len(t) - (1 - len(t) % 2), 5)
        heating = savgol_filter(par, window, 2, deriv=1, delta=float(np.median(np.diff(t)))) * deck.omega_pe / deck.omega_ce
        axes[2].plot(t, heating / np.interp(t, energy_t, magnetic), color="#d62728", label=label, **style[definition])
    axes[0].set(ylabel=r"$T/T_0$", title=r"Electron temperatures ([G12] Fig. 1b)")
    axes[1].set(ylabel=r"$T_\parallel/T_\perp$", title="Electron anisotropy")
    axes[2].set(ylabel=r"$d(T_\parallel/T_0)/d(|\Omega_e|t)\ /\ W_B$", title=r"Heating rate ([G12] Fig. 1d)")
    for ax in axes:
        ax.set_xlabel(r"$\omega_{pe}t$")
        ax.grid(True, alpha=0.3)
        ax.legend()
    return figure


def plot_electron_temperatures(deck: Deck, snapshots):
    """The thermal T_par/T0 and T_perp/T0 (left) and T_par/T_perp (right) of the electrons against time, as a print
    figure; T0 = m v_te^2."""
    t = np.array([s.time for s in snapshots])
    par, perp = (np.array([s.temperatures[f"thermal_{key}"] for s in snapshots]) for key in ("par", "perp"))
    t0 = initial_temperatures(deck, snapshots[0].temperatures)["thermal"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(16.0, 6.0), constrained_layout=True)
    left.plot(t, par / t0[0], color=PAR_COLOR, label=r"$T_\parallel$", **SIMULATION_LINE)
    left.plot(t, perp / t0[1], color=PERP_COLOR, label=r"$T_\perp$", **SIMULATION_LINE)
    right.axhline(1.0, color="0.6", linewidth=1.0, linestyle=":")  # isotropic
    right.plot(t, par / perp, color=ANISOTROPY_COLOR, **SIMULATION_LINE)
    left.set_ylabel(r"$T_e/T_0$")
    right.set_ylabel(r"$T_\parallel/T_\perp$")
    left.legend(loc="upper left")
    for ax in (left, right):
        ax.set_xlabel(r"$\omega_{pe}t$")
        ax.grid(True, alpha=0.3)
    return figure


def plot_temperature_energy(deck: Deck, snapshots, diagnostics):
    """A print figure. Left: the thermal T_par/T0 and T_perp/T0 of the electrons (left axis) and T_par/T_perp (right
    axis). Right, on one logarithmic axis over the initial total energy E0: the energies of diagnostics.csv, the
    magnetic energy that of dB (the uniform B0 keeps its own, constant), and the total-energy error |E_tot - E0|."""
    t = np.array([s.time for s in snapshots])
    par, perp = (np.array([s.temperatures[f"thermal_{key}"] for s in snapshots]) for key in ("par", "perp"))
    t0 = initial_temperatures(deck, snapshots[0].temperatures)["thermal"]
    figure, (temperature_axis, energy_axis) = plt.subplots(1, 2, figsize=(18.0, 6.5), constrained_layout=True)
    anisotropy_axis = temperature_axis.twinx()
    # The temperature axis lies on top, so its legend covers the anisotropy too.
    temperature_axis.set_zorder(anisotropy_axis.get_zorder() + 1)
    temperature_axis.patch.set_visible(False)
    anisotropy_axis.grid(False)
    curves = temperature_axis.plot(t, par / t0[0], color=PAR_COLOR, label=r"$T_\parallel/T_0$", **SIMULATION_LINE)
    curves += temperature_axis.plot(t, perp / t0[1], color=PERP_COLOR, label=r"$T_\perp/T_0$", **SIMULATION_LINE)
    curves += anisotropy_axis.plot(t, par / perp, color=ANISOTROPY_COLOR, label=r"$T_\parallel/T_\perp$",
                                   **SIMULATION_LINE)
    temperature_axis.set_ylabel(r"$T_e/T_0$")
    anisotropy_axis.set_ylabel(r"$T_\parallel/T_\perp$")
    temperature_axis.legend(handles=curves, loc="upper left")

    time = deck.omega_pe * column(diagnostics, "time")
    total = column(diagnostics, "total_energy")
    background = deck.b0**2 * np.prod(deck.lengths) / (2.0 * FOUR_PI)  # B0^2 V/(8 pi)
    line = {**SIMULATION_LINE, "markersize": 5.0, "markevery": max(1, time.size // 25)}
    energies = [(column(diagnostics, "magnetic_energy") - background, ENERGY_COLORS["magnetic"], r"$U_{\delta B}$"),
                (column(diagnostics, "electric_energy"), ENERGY_COLORS["electric"], r"$U_E$"),
                (column(diagnostics, f"kinetic_energy_{deck.electron}"), ENERGY_COLORS["electrons"], r"$K_e$")]
    if deck.ion is not None:
        energies.append((column(diagnostics, f"kinetic_energy_{deck.ion}"), ENERGY_COLORS["ions"], r"$K_i$"))
    for values, color, label in energies:
        energy_axis.semilogy(time, values / total[0], color=color, label=label, **line)
    # The error is round-off, a few ulp of either sign: its size, without the steps where it is exactly zero.
    error = np.abs(total - total[0]) / total[0]
    energy_axis.semilogy(time, np.where(error > 0.0, error, np.nan), color=ENERGY_COLORS["error"],
                         label=r"$|E_\mathrm{tot} - E_0|$", **{**line, "linewidth": 0.8})
    energy_axis.set_ylabel(r"Energy$/E_0$")
    # The energies lie near 0.1 and the error near 1e-15: the decades between them hold the legend.
    energy_axis.legend(loc="center right")
    for ax in (temperature_axis, energy_axis):
        ax.set_xlabel(r"$\omega_{pe}t$")
        ax.grid(True, alpha=0.3)
    return figure


def plot_spectra(fourier: Fourier, snapshots, picks, noise: dict, fit_time, fits):
    """The reduced spectra S(k_perp) (left) and S(k_par) (right) at the --times, as a print figure. Both are the
    magnetic energy per unit k, each summing over k to <|dB|^2>/B0^2, on one y range. The broken power law fitted to
    E(k_perp) = S(k_perp)/k_perp at fit_time is drawn times k_perp, its slopes one more than those of E; dotted: the
    largest seeded k_perp and k_par; solid: k_perp d_e = 1."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(17.0, 6.5), constrained_layout=True)
    k = fourier.k_perp_axis[1:]
    kz = fourier.dk_par * np.arange(1, snapshots[picks[0]].s_par.size)
    line = dict(linewidth=2.0)  # plain lines: a spectrum is no time series
    shown = []  # the plotted values with k*d_e <= 8, which set the y range of both
    for index, color in zip(picks, time_colors(len(picks))):
        s = snapshots[index]
        label = rf"$\omega_{{pe}}t$ = {s.time:.0f}"
        left.loglog(k, s.s_perp[1:], color=color, label=label, **line)
        right.loglog(kz, s.s_par[1:], color=color, label=label, **line)
        shown += [s.s_perp[1:][k <= 8.0], s.s_par[1:][kz <= 8.0]]
        if s.cycle in noise:
            left.loglog(k, noise[s.cycle][1:], color=color, lw=0.9, ls=":")
    broken = fits.get("broken")
    if broken is not None:
        amplitude, below, above, k_b = broken
        kk = np.geomspace(0.3, fits["k_signal"], 100)
        left.loglog(kk, amplitude * kk * (kk / k_b) ** np.where(kk < k_b, below, above), color="k", ls="--", lw=1.5,
                    label=rf"fit: $k_\perp^{{{below + 1:.2f}}}$, $k_\perp^{{{above + 1:.2f}}}$, $k_b d_e$ = {k_b:.2f}")
    left.axvline(1.0, color="0.5", lw=1.0)
    for ax, seeded in ((left, fourier.kperp), (right, fourier.kpar)):
        ax.axvline(seeded[fourier.seeded].max(), label=r"largest seeded $k$", **SEEDED_LINE)
    left.set(xlabel=r"$k_\perp d_e$", ylabel=r"$S(k_\perp)$  [$B_0^2 d_e$]")
    right.set(xlabel=r"$k_\parallel d_e$", ylabel=r"$S(k_\parallel)$  [$B_0^2 d_e$]")
    values = np.concatenate(shown)
    # A quiet start holds energy in the seeded modes alone: its round-off elsewhere, ~1e-31, leaves the range.
    values = values[values > SPECTRUM_FLOOR * values.max()]
    for ax, dk in ((left, fourier.dk_perp), (right, fourier.dk_par)):
        ax.set_xlim(dk * 0.8, 8.0)
        ax.set_ylim(values.min() / 30.0, values.max() * 3.0)  # room below the lowest values for the left legend
        ax.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 1.0, 2.0, 4.0, 8.0]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.25)
    # The spectra fall from the upper left: the fit and the seeded k go below them on the left, the times in the free
    # upper right corner on the right.
    handles, labels = left.get_legend_handles_labels()
    times = len(picks)
    left.legend(handles[times:], labels[times:], loc="lower left")
    right.legend(*right.get_legend_handles_labels(), loc="upper right")
    return figure


def plot_spectrum_2d(fourier: Fourier, snapshots, planes):
    ny = fourier.shape[1]
    ky = np.fft.fftshift(np.ravel(fourier.ky))
    kz = np.ravel(fourier.kz)
    kz_full = np.concatenate([-kz[1:][::-1], kz])
    peak = max(plane.max() for plane in planes.values())
    norm = LogNorm(vmin=peak * 1e-7, vmax=peak)
    figure, axes = plt.subplots(1, len(planes), figsize=(4.2 * len(planes) + 1.0, 4.4), constrained_layout=True,
                                squeeze=False)
    for ax, (step, plane) in zip(axes[0], sorted(planes.items())):
        # The plane at -kz is the conjugate of the one at -ky, +kz.
        negative = np.fft.fftshift(plane[(-np.arange(ny)) % ny, :], axes=0)[:, 1:][:, ::-1]
        full = np.concatenate([negative, np.fft.fftshift(plane, axes=0)], axis=1)
        ax.pcolormesh(ky, kz_full, np.maximum(full.T, norm.vmin), norm=norm, cmap="magma", shading="nearest")
        ax.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", xlabel=r"$k_y d_e$", ylabel=r"$k_\parallel d_e$",
               title=rf"$\omega_{{pe}}t$ = {snapshots[step].time:.0f}")
    figure.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=axes[0].tolist(), label=r"$|B_k|^2/B_0^2$ at $k_x$ = 0")
    return figure


def plot_anisotropy(fourier: Fourier, snapshots):
    t = np.array([s.time for s in snapshots])
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    for key, color in zip(snapshots[0].tan2, ("#1f4a9e", "#d62728", "#2ca02c", "#9467bd")):
        label = {"all": r"all $k \leq k_s$", "cascaded": r"cascaded $k \leq k_s$"}.get(key, rf"${key}$ shell")
        ax.plot(t, [s.tan2[key] for s in snapshots], color=color, lw=1.6, label=label)
    ax.axhline(2.0, color="0.5", lw=0.8, ls="--", label="3D isotropic")
    ax.set(ylabel=r"$\tan^2\theta_B$", title=r"Wavevector anisotropy ([G12] Fig. 2)")
    ax.legend()
    shell_label = {"seeded": "seeded shell {:.2f}-{:.2f}".format(*fourier.seed_shell),
                   "cascade": "cascade shell {:.2f}-{:.2f}".format(*fourier.cascade_shell)}
    for ax, quantity, ylabel, title in ((axes[0, 1], "gyrotropy", r"$G$ = std/mean over azimuth", "Gyrotropy"),
                                        (axes[1, 0], "helicity", r"$\sigma_m$", "Normalized magnetic helicity")):
        for key, color in zip(fourier.shells, ("#1f4a9e", "#d62728")):
            ax.plot(t, [getattr(s, quantity)[key] for s in snapshots], color=color, lw=1.6, label=shell_label[key])
        ax.set(ylabel=ylabel, title=title)
        ax.legend()
    for ax in axes.ravel()[:3]:
        ax.set_xlabel(r"$\omega_{pe}t$")
    ax = axes[1, 1]
    cascaded = np.array([s.cascaded for s in snapshots])
    scatter = ax.scatter(cascaded, [s.tan2["cascaded"] for s in snapshots], c=t, cmap="viridis", s=14)
    figure.colorbar(scatter, ax=ax, label=r"$\omega_{pe}t$")
    ax.set(xlabel=r"cascaded $\langle|\delta B|^2\rangle/B_0^2$", ylabel=r"cascaded $\tan^2\theta_B$",
           title="Anisotropy against cascaded energy ([S08] Fig. 3)")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    return figure


def plot_polarization(fourier: Fourier, sums: np.ndarray):
    thetas = THETA_EDGES.size - 1
    centre = 0.5 * (THETA_EDGES[1:] + THETA_EDGES[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (sums[:2] / sums[2]).reshape(2, 3, thetas)
    theory = np.linspace(0.0, 90.0, 200)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    edges = fourier.polarization_edges
    for shell, color in zip(range(3), ("#1f4a9e", "#d62728", "#2ca02c")):
        label = rf"${edges[shell]:.2f} \leq k d_e < {edges[shell + 1]:.2f}$"
        for ax, values in zip(axes, ratio):
            ax.plot(centre, values[shell], marker="o", color=color, lw=1.4, label=label)
    axes[0].plot(theory, 0.5 * np.sin(np.radians(theory)) ** 2, color="0.4", lw=3.0, alpha=0.5,
                 label=r"whistler: $\sin^2\theta/2$")
    axes[1].axhline(0.5, color="0.4", lw=3.0, alpha=0.5, label="whistler: 1/2")
    axes[0].set(ylabel=r"$C_\parallel = |\delta B_z|^2/|\delta B|^2$", title="Magnetic compressibility")
    axes[1].set(ylabel=r"$r_B = |\delta B\cdot\hat z\times\hat k_\perp|^2/|\delta B|^2$",
                title=r"Out-of-plane fraction ($> 1/2$: kinetic Alfven)")
    for ax in axes:
        ax.set(xlabel=r"$\theta_{kB}$ [deg]", xlim=(0, 90), ylim=(0, 1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    return figure


def alias(omega, nyquist: float):
    """The frequency in (-nyquist, nyquist] at which snapshots pi/nyquist apart see omega."""
    return nyquist - np.mod(nyquist - omega, 2.0 * nyquist)


def window_label(spectrum: OmegaKSpectrum) -> str:
    return rf"$\omega_{{pe}}t$ = {spectrum.times[0]:.0f}–{spectrum.times[-1]:.0f}"


def plot_omega_k(omega_k: OmegaK, spectra, planes, k_signal: float):
    """omega against k_perp in the k_par planes (columns) of each time window (rows): the energy per unit omega of
    each k_perp shell over its peak, the whistler +-omega dashed and, past the Nyquist frequency, its alias dotted."""
    figure, axes = plt.subplots(len(spectra), len(planes), figsize=(3.7 * len(planes) + 1.2, 3.3 * len(spectra) + 0.3),
                                constrained_layout=True, squeeze=False)
    norm = LogNorm(OMEGA_K_FLOOR, 1.0)
    k_perp = omega_k.k_perp_axis
    curve = np.linspace(0.0, k_perp[-1], 400)
    for row, spectrum in zip(axes, spectra):
        for ax, plane in zip(row, planes):
            k_par = omega_k.k_par[plane]
            peak = spectrum.power[plane].max(axis=1, keepdims=True)
            # A shell without energy (beyond the node-filter cut) is left blank rather than scaled up to its round-off.
            shown = np.where(peak > 1e-12 * peak.max(), spectrum.power[plane] / np.where(peak > 0.0, peak, 1.0), np.nan)
            ax.pcolormesh(k_perp, spectrum.omega, np.maximum(shown.T, OMEGA_K_FLOOR), norm=norm, cmap="magma",
                          shading="nearest", rasterized=True)
            for sign in (1.0, -1.0):
                omega = sign * whistler_frequency(curve, k_par)
                seen = np.abs(omega) <= spectrum.nyquist
                ax.plot(curve, np.where(seen, omega, np.nan), color="cyan", lw=1.6, ls="--")
                if not seen.all():
                    folded = np.where(seen, np.nan, alias(omega, spectrum.nyquist))
                    folded[1:][np.abs(np.diff(folded)) > spectrum.nyquist] = np.nan  # no line across a wrap
                    ax.plot(curve, folded, color="cyan", lw=1.4, ls=":")
            ax.axvline(k_signal, color="0.75", lw=0.8, ls=":")
            ax.set(xlim=(0.0, k_perp[-1]), ylim=(-spectrum.nyquist, spectrum.nyquist),
                   title=rf"{window_label(spectrum)}, $k_\parallel d_e$ = {k_par:.2f}")
        row[0].set_ylabel(r"$\omega/|\Omega_e|$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$k_\perp d_e$")
    figure.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=axes.ravel().tolist(), shrink=0.8,
                    label=r"energy per unit $\omega$ over its peak at each $k_\perp$  (dashed: whistler)")
    return figure


def plot_omega_power(omega_k: OmegaK, spectra, bins):
    """The energy per unit omega against omega in each (k_perp shell, k_par plane) bin, one curve per time window,
    the whistler +-omega dashed (its alias dotted past the Nyquist frequency)."""
    figure, axes = plt.subplots(1, len(bins), figsize=(4.3 * len(bins) + 0.5, 4.3), constrained_layout=True,
                                squeeze=False)
    nyquist = max(spectrum.nyquist for spectrum in spectra)
    for ax, (shell, plane) in zip(axes[0], bins):
        k_perp, k_par = omega_k.k_perp_axis[shell], omega_k.k_par[plane]
        for spectrum, color in zip(spectra, time_colors(len(spectra))):
            ax.semilogy(spectrum.omega, spectrum.power[plane, shell], color=color, lw=1.5, label=window_label(spectrum))
        peak = max(spectrum.power[plane, shell].max() for spectrum in spectra)
        omega = whistler_frequency(k_perp, k_par)
        seen = omega <= nyquist
        for sign in (1.0, -1.0):
            ax.axvline(sign * omega if seen else alias(sign * omega, nyquist), color="0.3", lw=1.0,
                       ls="--" if seen else ":", label="whistler" if sign > 0 else None)
        ax.set(xlim=(-nyquist, nyquist), ylim=(peak * 1e-4, peak * 2.0), xlabel=r"$\omega/|\Omega_e|$",
               title=rf"$k_\perp d_e$ = {k_perp:.2f}, $k_\parallel d_e$ = {k_par:.2f}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_ylabel(r"$\langle|\delta B|^2\rangle/B_0^2$ per unit $\omega/|\Omega_e|$")
    axes[0, 0].legend(loc="lower center")
    return figure


def omega_k_summary(omega_k: OmegaK, spectra, bins) -> list[str]:
    """The resolution of the omega-k spectra and, per bin, the whistler frequency against the peak of each window."""
    windows = ", ".join(f"{s.times[0]:.0f}-{s.times[-1]:.0f}" for s in spectra)
    lines = ["", f"omega-k spectra in |Omega_e|, time windows omega_pe*t {windows}",
             f"  Nyquist pi/dt, resolution 2*pi/T  = {spectra[0].nyquist:.3f}, {spectra[0].resolution:.3f}"]
    for shell, plane in bins:
        k_perp, k_par = omega_k.k_perp_axis[shell], omega_k.k_par[plane]
        peaks = ", ".join(f"{abs(s.omega[np.argmax(s.power[plane, shell])]):.3f}" for s in spectra)
        lines.append(f"  {f'(k_perp, k_par) = ({k_perp:.2f}, {k_par:.2f})':34s}= whistler "
                     f"{whistler_frequency(k_perp, k_par):.3f}, peak |omega| {peaks}")
    return lines


def plot_vdf(deck: Deck, edges, distributions):
    centre = 0.5 * (edges[1:] + edges[:-1])
    landau = deck.omega_ce * deck.d_e / 2.0 / deck.v_te
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.0), constrained_layout=True)
    for (time, f_par, f_x, _), color in zip(distributions, time_colors(len(distributions))):
        axes[0].semilogy(centre, f_par, color=color, lw=1.6, label=rf"$\omega_{{pe}}t$ = {time:.0f}")
        axes[1].semilogy(centre, f_x, color=color, lw=1.6, label=rf"$\omega_{{pe}}t$ = {time:.0f}")
    gaussian = np.exp(-centre**2 / 2) / math.sqrt(2 * math.pi)
    for ax, xlabel, title in ((axes[0], r"$v_\parallel/v_{te}$", r"$f(v_\parallel)$ ([G12] Fig. 5)"),
                              (axes[1], r"$v_x/v_{te}$", r"$f(v_x)$")):
        ax.semilogy(centre, gaussian, color="0.5", ls=":", lw=1.2, label="Maxwellian $v_{te}$")
        ax.set(xlabel=xlabel, ylabel="f (unit area)", ylim=(1e-7, 1.0), title=title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center")
    for sign in (-1, 1):
        axes[0].axvline(sign * landau, color="0.4", lw=0.8, ls="--")
    first, last = distributions[0][3], distributions[-1][3]
    delta = (last - first) / first[VDF_BINS // 2, VDF_BINS // 2]
    limit = np.abs(delta).max()
    mesh = axes[2].pcolormesh(centre, centre, delta.T, cmap="RdBu_r", norm=Normalize(-limit, limit), shading="nearest")
    for sign in (-1, 1):
        axes[2].axhline(sign * landau, color="0.3", lw=0.8, ls="--")
    figure.colorbar(mesh, ax=axes[2], label=r"$[f(t) - f(0)]/f(0, 0)$")
    axes[2].set(xlabel=r"$v_x/v_{te}$", ylabel=r"$v_\parallel/v_{te}$", xlim=(-4, 4), ylim=(-4, 4), aspect="equal",
                title=rf"$\delta f$ at $\omega_{{pe}}t$ = {distributions[-1][0]:.0f} (dashed: $|\Omega_e| d_e/2$)")
    return figure


def render_panel(deck: Deck, values: np.ndarray, path: Path, cmap: str, clim, opacity) -> None:
    """Volume rendering of an (x, y, z) node field in the periodic box on white, B0 (the arrow beside the box)
    vertical, the x, y, z arrows of the view in the lower left corner."""
    import pyvista as pv

    size = deck.lengths / deck.d_e
    grid = pv.ImageData(dimensions=values.shape, spacing=tuple(deck.spacing / deck.d_e))
    grid.point_data["values"] = values.ravel(order="F")
    plotter = pv.Plotter(off_screen=True, window_size=(PIXELS_3D, PIXELS_3D))
    plotter.set_background("white")
    plotter.add_volume(grid, scalars="values", cmap=cmap, clim=clim, opacity=opacity, shade=False,
                       show_scalar_bar=False)
    plotter.add_mesh(grid.outline(), color="#5a5a5a", line_width=1.5)
    plotter.add_arrows(np.array([[1.12 * size[0], -0.12 * size[1], 0.2 * size[2]]]), np.array([[0.0, 0.0, 1.0]]),
                       mag=0.6 * size[2], color="#1f4a9e")
    plotter.add_axes(color="black", line_width=4, label_size=(0.35, 0.14), viewport=(0.0, 0.0, AXES_3D, AXES_3D))
    plotter.camera_position = "iso"
    plotter.camera.azimuth = -20.0
    plotter.reset_camera()
    plotter.screenshot(str(path))
    plotter.close()


def plot_render_3d(deck: Deck, snapshots, currents):
    figure, axes = plt.subplots(1, len(currents), figsize=(4.6 * len(currents) + 1.0, 4.9), constrained_layout=True,
                                squeeze=False)
    with tempfile.TemporaryDirectory() as scratch:
        for ax, (step, current) in zip(axes[0], sorted(currents.items())):
            path = Path(scratch) / f"render_{step}.png"
            render_panel(deck, current / np.sqrt(np.mean(current**2)), path, RENDER_CMAP, RENDER_RANGE, RENDER_OPACITY)
            ax.imshow(plt.imread(path))
            ax.set_title(rf"$\omega_{{pe}}t$ = {snapshots[step].time:.0f}")
            ax.set_axis_off()
    figure.colorbar(ScalarMappable(norm=Normalize(*RENDER_RANGE), cmap=RENDER_CMAP), ax=axes[0].tolist(), shrink=0.8,
                    label=r"$|\nabla\times\delta\mathbf{B}|$ / rms  (arrow: $\mathbf{B}_0$)")
    return figure


def density_rms(densities) -> list[tuple[str, float, float]]:
    """(species, omega_pe*t, rms dn/n0) of every density render."""
    return [(kind, time, float(np.sqrt(np.mean(fluctuation**2))))
            for step in sorted(densities) for kind, (time, fluctuation) in densities[step].items()]


def plot_render_density(deck: Deck, densities, kind: str):
    """The dn/n0 of kind ("electrons" or "ions") at the render steps, on the colour scale shared by both species:
    DENSITY_RANGE times the largest rms of their panels."""
    # A quiet start is uniform: the floor keeps its round-off from being drawn.
    limit = max(DENSITY_RANGE * max(rms for _, _, rms in density_rms(densities)), 1e-6)
    steps = [step for step in sorted(densities) if kind in densities[step]]
    figure, axes = plt.subplots(1, len(steps), figsize=(4.6 * len(steps) + 1.0, 4.9), constrained_layout=True,
                                squeeze=False)
    with tempfile.TemporaryDirectory() as scratch:
        for ax, step in zip(axes[0], steps):
            time, fluctuation = densities[step][kind]
            path = Path(scratch) / f"render_{kind}_{step}.png"
            render_panel(deck, fluctuation, path, DENSITY_CMAP, (-limit, limit), DENSITY_OPACITY)
            ax.imshow(plt.imread(path))
            ax.set_title(rf"$\omega_{{pe}}t$ = {time:.0f}")
            ax.set_axis_off()
    figure.colorbar(ScalarMappable(norm=Normalize(-limit, limit), cmap=DENSITY_CMAP), ax=axes[0].tolist(), shrink=0.8,
                    label=rf"$\delta n_{kind[0]}/n_0$  (arrow: $\mathbf{{B}}_0$)")
    return figure


def write_timeseries(path: Path, snapshots) -> None:
    keys = ["cycle", "time", "magnetic", "cascaded", "k_signal"]
    nested = ["tan2", "gyrotropy", "helicity"] + (["temperatures"] if snapshots[0].temperatures else [])
    header = keys + [f"{group}_{key}" for group in nested for key in getattr(snapshots[0], group)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for s in snapshots:
            writer.writerow([getattr(s, key) for key in keys] +
                            [value for group in nested for value in getattr(s, group).values()])


def summarize(deck: Deck, fourier: Fourier, snapshots, picks, magnetic, total, fit_time, fits) -> list[str]:
    first, last = snapshots[0], snapshots[-1]
    seeds = seed_indices() * (2.0 * math.pi * deck.d_e / deck.lengths)
    expected_tan2 = float((seeds[:, :2] ** 2).sum() / (seeds[:, 2] ** 2).sum())
    lines = [f"deck                                = {deck.path}",
             f"omega_pe, d_e, |Omega_e|/omega_pe   = {deck.omega_pe:g}, {deck.d_e:g}, {deck.omega_ce / deck.omega_pe:.4f}",
             f"k0 d_e (perp, par)                  = {fourier.dk_perp:.4f}, {fourier.dk_par:.4f}",
             f"seeded |k| d_e, cascade shell       = {fourier.seed_shell[0]:.3f}-{fourier.seed_shell[1]:.3f}, "
             f"{fourier.cascade_shell[0]:.3f}-{fourier.cascade_shell[1]:.3f}",
             f"snapshots, omega_pe*t range         = {len(snapshots)}, [{first.time:g}, {last.time:g}]", "",
             "Step-0 checks (Section 3.8)",
             f"  <|dB|^2>/B0^2 spectrum, diagnostics = {first.magnetic:.6f}, {magnetic[0]:.6f} (epsilon0 = {deck.epsilon:g})",
             f"  unseeded fraction of W_B          = {first.cascaded / first.magnetic:.2e}",
             f"  tan^2 theta_B (all k <= k_s)      = {first.tan2['all']:.4f} (equal energy per seeded mode: "
             f"{expected_tan2:.4f})",
             f"  gyrotropy seeded shell            = {first.gyrotropy['seeded']:.3f} (seed in 4 of the "
             f"{AZIMUTH_BINS} azimuth bins: about sqrt(2); gyrotropic: 0)",
             f"  helicity seeded shell             = {first.helicity['seeded']:+.2e} (balanced seed: 0)"]
    if first.temperatures:
        temperature = first.temperatures
        lines += [f"  papers' T_par/T_th, T_perp/T_th   = {temperature['paper_par'] / deck.v_te**2:.4f}, "
                  f"{temperature['paper_perp'] / deck.v_te**2:.4f} (T_th = m v_te^2; the excess is the wave flow)",
                  f"  thermal T_par/T_th, T_perp/T_th   = {temperature['thermal_par'] / deck.v_te**2:.4f}, "
                  f"{temperature['thermal_perp'] / deck.v_te**2:.4f}"]
    lines += ["", "Evolution",
              f"  max |E_tot - E_tot(0)|/W_B(0)     = {np.abs(total - total[0]).max() / magnetic[0]:.2e}",
              f"  W_B(end)/W_B(0)                   = {magnetic[-1] / magnetic[0]:.4f}"]
    for index in picks:
        s = snapshots[index]
        lines.append(f"  omega_pe*t = {s.time:6.0f}: W_B/W_B(0) = {s.magnetic / first.magnetic:.4f}, tan^2 theta_B all "
                     f"{s.tan2['all']:.3f}, cascaded {s.tan2['cascaded']:.3f}, G cascade {s.gyrotropy['cascade']:.3f}, "
                     f"sigma_m cascade {s.helicity['cascade']:+.3f}, k_s {s.k_signal:.2f}")
    if last.temperatures:
        for definition, (par0, perp0) in initial_temperatures(deck, first.temperatures).items():
            lines.append(f"  final {definition:7s} T_par/T0, T_perp/T0, T_par/T_perp = "
                         f"{last.temperatures[definition + '_par'] / par0:.4f}, "
                         f"{last.temperatures[definition + '_perp'] / perp0:.4f}, "
                         f"{last.temperatures[definition + '_par'] / last.temperatures[definition + '_perp']:.4f}")
    slope = lambda fit: "fewer than 3 shells" if math.isnan(fit[0]) else f"{fit[0]:.2f} +- {fit[1]:.2f}"
    broken = fits["broken"]
    lines += ["", f"Slopes of E(k_perp), mean of the snapshots near omega_pe*t = {fit_time:g} (Section 5.5)",
              f"  below the break (0.4, 0.9)        = {slope(fits['below'])}",
              f"  above the break (1.2, {fits['k_signal']:.2f})       = {slope(fits['above'])}",
              "  broken power law (0.3, k_s)       = " +
              ("no fit" if broken is None else f"{broken[1]:.2f} / {broken[2]:.2f}, k_b d_e = {broken[3]:.3f}")]
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decaying whistler turbulence analysed as in "
                                                 "docs/user/whistler_turbulence.md, after [G12] "
                                                 "Gary, Chang & Wang, ApJ 755, 142 (2012) and [S08] Saito et al., "
                                                 "Phys. Plasmas 15, 102305 (2008).")
    parser.add_argument("--run-dir", type=Path, default=Path("."), help="run directory (default: current directory)")
    parser.add_argument("--input", type=Path, default=None,
                        help="input deck (default: the single WhistlerTurbulence deck of RUN_DIR)")
    parser.add_argument("--data-dir", type=Path, default=None, help="output of the run (default: RUN_DIR/SaveDirName)")
    parser.add_argument("--output-dir", type=Path, default=None, help="where the plots go (default: RUN_DIR)")
    parser.add_argument("--prefix", default="whistler_turbulence", help="file-name prefix of the plots and tables")
    parser.add_argument("--times", type=float, nargs="+", default=(0.0, 130.0, 300.0, 1000.0),
                        help="omega_pe*t of the spectra and the (k_y, k_par) planes (default: 0 130 300 1000; "
                             "[G12] Fig. 3 shows 50 130 300 1000)")
    parser.add_argument("--fit-time", type=float, default=300.0, help="omega_pe*t of the slopes (default: 300)")
    parser.add_argument("--fit-window", type=float, default=20.0,
                        help="half width in 1/omega_pe of the snapshots averaged for the slopes (default: 20)")
    parser.add_argument("--k-signal", type=float, default=3.0,
                        help="largest k*d_e of the fits and the anisotropy without a control run (default: 3)")
    parser.add_argument("--noise-dir", type=Path, default=None,
                        help="output directory of an epsilon0 = 0 run of the same deck: its spectrum limits k_s")
    parser.add_argument("--rate-window", type=float, default=20.0,
                        help="Savitzky-Golay window of the field decay rate, in 1/omega_pe (default: 20)")
    parser.add_argument("--omega-k-times", type=float, nargs="+", default=(250.0, 500.0, 750.0),
                        help="omega_pe*t of the centres of the omega-k time windows (default: 250 500 750)")
    parser.add_argument("--omega-k-window", type=float, default=500.0,
                        help="length of each omega-k time window, in 1/omega_pe (default: 500)")
    parser.add_argument("--omega-k-kpar", type=float, nargs="+", default=(0.1, 0.2, 0.3, 0.4),
                        help="k_par*d_e of the omega-k_perp maps, each taken in the nearest plane "
                             "(default: 0.1 0.2 0.3 0.4)")
    parser.add_argument("--omega-k-bins", type=float, nargs="+", default=(0.5, 0.2, 1.0, 0.2, 1.0, 0.4, 2.0, 0.3),
                        metavar="K", help="k_perp*d_e k_par*d_e pairs of the power-against-omega panels "
                                          "(default: 0.5 0.2  1 0.2  1 0.4  2 0.3)")
    parser.add_argument("--only", nargs="+", choices=PARTS, metavar="PART",
                        help=f"draw these parts alone, reading only the data they need: {', '.join(PARTS)} (default: "
                             "all but render, which --render-3d adds); the timeseries and the summary are rewritten "
                             "only when every part but vdf and render is drawn")
    parser.add_argument("--no-particles", action="store_true", help="skip the particle snapshots")
    parser.add_argument("--render-3d", action="store_true",
                        help="also draw <prefix>_render_3d and <prefix>_render_density_<species> (needs pyvista)")
    parser.add_argument("--render-times", type=float, nargs="+", default=(0.0, 300.0, 1000.0),
                        help="omega_pe*t of the 3D panels (default: 0 300 1000)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if len(args.omega_k_bins) % 2:
        parser.error("--omega-k-bins takes k_perp k_par pairs")
    parts = set(args.only or PARTS[:-1]) | ({"render"} if args.render_3d else set())
    complete = SUMMARY_PARTS <= parts
    deck = read_deck(find_input(args.run_dir, args.input))
    data_dir = args.data_dir or args.run_dir / deck.save_dir
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fourier = Fourier(deck)
    with open_field_reader(data_dir / "B") as reader:
        times = deck.omega_pe * np.array([epoch[1] for _, epoch in reader.records])
    picks = pick(times, args.times)
    window = [i for i, t in enumerate(times) if abs(t - args.fit_time) <= args.fit_window]
    renders = pick(times, args.render_times) if "render" in parts else []
    # B at every step for the time histories, else at the spectra and fit steps and the render steps alone.
    b_steps = ((set(range(times.size)) if parts & {"anisotropy", "polarization", "omega-k"} else set())
               | (set(picks) | set(window) if "spectra" in parts else set()) | set(renders))
    noise = noise_spectra(args.noise_dir, deck, fourier) if b_steps else {}
    omega_k = OmegaK(fourier, [*args.omega_k_kpar, *args.omega_k_bins[1::2]]) if "omega-k" in parts else None
    snapshots, polarization, planes, currents, densities = analyze_fields(
        deck, data_dir, fourier, noise, args.k_signal, b_steps, "temperatures" in parts,
        set(picks) if "spectra" in parts else set(), set(renders), omega_k)

    fits, k_s = {}, args.k_signal
    if window and set(window) <= b_steps:
        fit_spectrum = np.mean([snapshots[i].s_perp[1:] for i in window], axis=0) / fourier.k_perp_axis[1:]
        k_s = min(snapshots[i].k_signal for i in window)
        k = fourier.k_perp_axis[1:]
        fits = {"k_signal": k_s, "below": powerlaw_fit(k, fit_spectrum, 0.4, 0.9),
                "above": powerlaw_fit(k, fit_spectrum, 1.2, k_s),
                "broken": broken_powerlaw_fit(k, fit_spectrum, 0.3, k_s)}

    written = []
    if parts & {"energy", "temperatures"}:
        diagnostics = read_csv(data_dir)
        figure, energy_t, magnetic, total = plot_energy(deck, diagnostics, args.rate_window)
        if "energy" in parts:
            written += save(figure, output_dir, f"{args.prefix}_energy")
        else:
            plt.close(figure)
    if "temperatures" in parts:
        if snapshots[0] is not None and snapshots[0].temperatures:
            written += save(plot_temperatures(deck, snapshots, energy_t, magnetic), output_dir,
                            f"{args.prefix}_temperatures")
            with plt.rc_context(PRINT_STYLE):
                written += save(plot_electron_temperatures(deck, snapshots), output_dir,
                                f"{args.prefix}_electron_temperatures")
                written += save(plot_temperature_energy(deck, snapshots, diagnostics), output_dir,
                                f"{args.prefix}_temperature_energy")
        else:
            print(f"No temperatures: the deck writes no rho, J, PXX, PYY and PZZ of species {deck.electron}.")
    if "spectra" in parts:
        with plt.rc_context(PRINT_STYLE):
            written += save(plot_spectra(fourier, snapshots, picks, noise, times[window].mean() if window else math.nan,
                                         fits), output_dir, f"{args.prefix}_spectra")
        written += save(plot_spectrum_2d(fourier, snapshots, planes), output_dir, f"{args.prefix}_spectrum_2d")
    if "anisotropy" in parts:
        written += save(plot_anisotropy(fourier, snapshots), output_dir, f"{args.prefix}_anisotropy")
    if "polarization" in parts:
        written += save(plot_polarization(fourier, polarization), output_dir, f"{args.prefix}_polarization")
    spectra, bins = [], []
    if omega_k is not None:
        half = 0.5 * args.omega_k_window
        spectra = [spectrum for spectrum in (omega_k.spectrum(centre - half, centre + half,
                                                              deck.omega_ce / deck.omega_pe)
                                             for centre in args.omega_k_times) if spectrum is not None]
        bins = [(omega_k.shell(k_perp), omega_k.plane(k_par))
                for k_perp, k_par in zip(args.omega_k_bins[::2], args.omega_k_bins[1::2])]
        if spectra:
            columns = list(dict.fromkeys(omega_k.plane(k_par) for k_par in args.omega_k_kpar))
            written += save(plot_omega_k(omega_k, spectra, columns, k_s), output_dir, f"{args.prefix}_omega_k")
            if bins:
                written += save(plot_omega_power(omega_k, spectra, bins), output_dir, f"{args.prefix}_omega_power")
        else:
            print(f"No omega-k spectra: no time window holds {OMEGA_K_MIN_SNAPSHOTS} evenly spaced field snapshots.")
    particles = not args.no_particles and any(data_dir.glob("particles_t*.pvtp"))
    if "vdf" in parts and particles:
        edges, distributions = read_distributions(data_dir, deck)
        written += save(plot_vdf(deck, edges, distributions), output_dir, f"{args.prefix}_vdf")
    if currents:
        written += save(plot_render_3d(deck, snapshots, currents), output_dir, f"{args.prefix}_render_3d")
        for step, density in densities.items():
            if "ions" not in density and deck.ion is not None and particles:
                density["ions"] = deposit_density(data_dir, deck, fourier, deck.ion, snapshots[step].time,
                                                  snapshots[step].k_signal)
        for kind in ("electrons", "ions"):
            if any(kind in density for density in densities.values()):
                written += save(plot_render_density(deck, densities, kind), output_dir,
                                f"{args.prefix}_render_density_{kind}")
    lines = [f"{f'rms dn_{kind[0]}/n0 at omega_pe t = {time:.0f}':36s}= {rms:.3%}"
             for kind, time, rms in density_rms(densities)]
    if spectra:
        lines += omega_k_summary(omega_k, spectra, bins)
    if complete:
        csv_path = output_dir / f"{args.prefix}_timeseries.csv"
        write_timeseries(csv_path, snapshots)
        lines = summarize(deck, fourier, snapshots, picks, magnetic, total, times[window].mean(), fits) + lines
        summary_path = output_dir / f"{args.prefix}_summary.txt"
        summary_path.write_text("\n".join(lines) + "\n")
        written += [csv_path, summary_path]
    print("\n".join(lines))
    if not complete:
        print(f"\nOnly {', '.join(p for p in PARTS if p in parts)}: the timeseries and the summary are left as they "
              "are.")
    print("\nWrote:", *(str(path) for path in written), sep="\n  ")


if __name__ == "__main__":
    main()
