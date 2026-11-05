#!/usr/bin/env python3
"""Measure the seeded longitudinal mode in a cold two-stream run.

The analyzer reads the TPIC field files ``E_t*.vti``.  A typical invocation
from the repository root is::

    python3 tools/postprocessing/analyze_two_stream.py \
        --run-dir runs/two_stream

The input deck is searched for first in ``--run-dir`` and then in the
repository ``input`` directory.  ``--input`` and ``--data-dir`` override
those locations.  Fit-window arguments use electron time, ``omega_pe*t``.

Plots.  The plot at ``--output`` carries the title, the fit, the growth rates
in the legend and the fit statistics.  Two more go beside it, each as a PNG
and as a PDF: the mode history and the cold theory alone, with ``_figure``
added to the name, and the kinetic, electric and total energy of
``diagnostics.csv`` in units of the initial total energy, with ``_energy``
added.  When the run holds particle snapshots, the electron phase space
(x, vx), one dot per particle and colored by beam, goes beside it too, with
``_phase_space`` added, one panel per time in a row: by default at t = 0, at
half the run and at the last snapshot, or at the ``omega_pe*t`` of
``--phase-space-times``, each at the nearest snapshot.
``FontSizes`` sets the font sizes of every plot, ``FigureStyle`` the labels,
colors, lines and markers of the mode and energy plots, through
``MODE_FIGURE_STYLE`` and ``ENERGY_FIGURE_STYLE``, and ``PhaseSpaceStyle``
the look of the phase space, through ``PHASE_SPACE_STYLE``; edit their values
to change the look.
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
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.fields import open_field_reader, transverse_mean
from tpic_tools.particles import read_polydata
from tpic_tools.runs import read_csv

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TWO_PI = 2.0 * math.pi
_PROJECT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class FontSizes:
    """Font sizes of every plot, in points."""

    title: float = 12.0
    axis_label: float = 22.0
    tick_label: float = 20.0
    legend: float = 18.0
    fit_box: float = 18.0  # the fit statistics of the first plot
    panel_title: float = 20.0  # the omega_pe*t over each phase-space panel


@dataclass
class FigureStyle:
    """Look of a plot of a few curves against omega_pe*t; a tuple holds one entry per curve."""

    y_label: str  # {mode} is the number of the seeded mode and a LaTeX brace is doubled
    labels: tuple[str, ...]
    colors: tuple[str, ...]
    line_styles: tuple[str, ...]
    line_widths: tuple[float, ...]  # points
    markers: tuple[str, ...]  # "" draws none
    figure_size: tuple[float, float] = (8.0, 5.0)  # inches
    marker_size: float = 4.0
    markers_per_curve: int = 25
    legend_location: str = "best"
    x_label: str = r"$\omega_{pe} t$"
    y_scale: str = "log"  # or "linear"


FONT_SIZES = FontSizes()
# The amplitude of the seeded Fourier mode in code units: a thin dark blue line with dots over the cold theory as a
# wide light purple one, which stays visible around it.
MODE_FIGURE_STYLE = FigureStyle(
    y_label=r"$e\,|\hat{{E}}_x(m={mode})|\,/\,(m_i c\,\omega_{{pi}})$",
    labels=("TPIC", "Cold theory"),
    colors=("#1f4a9e", "#d3a9ee"),
    line_styles=("-", "-"),
    line_widths=(1.3, 4.5),
    markers=("o", ""),
)
# The kinetic energy in the same blue over the total energy in the same purple, which it leaves once the electric
# energy, dashed, is comparable with it.
ENERGY_FIGURE_STYLE = FigureStyle(
    y_label=r"$E_{{\mathrm{{tot}}}}(t)/\,E_{{\mathrm{{tot}}}}(0)$",
    labels=("Kinetic", "Electric", "Total"),
    colors=("#1f4a9e", "C1", "#d3a9ee"),
    line_styles=("-", "--", "-"),
    line_widths=(1.3, 1.6, 4.5),
    markers=("o", "", ""),
)


@dataclass
class PhaseSpaceStyle:
    """Look of the electron phase space; a tuple holds one entry per electron beam, in deck order."""

    colors: tuple[str, ...]
    label: str = r"$u_0={drift:+g}\,c$"  # {drift} is the drift of the beam
    title: str = r"$\omega_{{pe}} t = {time:.4g}$"  # {time} is the omega_pe*t of the snapshot
    dot_size: float = 1.0  # points^2
    alpha: float = 0.6
    panel_size: tuple[float, float] = (7.0, 5.0)  # inches
    legend_location: str = "upper right"
    x_label: str = r"$x/d_e$"
    y_label: str = r"$v_x/c$"


# The first beam of the deck (+x in twoStream1D.inp) in purple, the second in green.
PHASE_SPACE_STYLE = PhaseSpaceStyle(colors=("#7b3fb3", "#16a085"))

plt.rcParams.update({
    "figure.titlesize": FONT_SIZES.title, "axes.titlesize": FONT_SIZES.title,
    "axes.labelsize": FONT_SIZES.axis_label,
    "xtick.labelsize": FONT_SIZES.tick_label, "ytick.labelsize": FONT_SIZES.tick_label,
    "legend.fontsize": FONT_SIZES.legend,
})




def _required_tokens(values: dict[str, list[str]], key: str) -> list[str]:
    if key not in values or not values[key]:
        raise ValueError(f"input deck is missing required key '{key}'")
    return values[key]


def _first_float(values: dict[str, list[str]], key: str) -> float:
    token = _required_tokens(values, key)[0]
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(f"input key '{key}' must be numeric, got {token!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"input key '{key}' must be finite, got {token!r}")
    return value


def _first_int(values: dict[str, list[str]], key: str) -> int:
    value = _first_float(values, key)
    integer = int(value)
    if value != integer:
        raise ValueError(f"input key '{key}' must be an integer, got {value}")
    return integer


def _float_list(
    values: dict[str, list[str]],
    key: str,
    required_count: int,
) -> tuple[float, ...]:
    tokens = _required_tokens(values, key)
    if len(tokens) < required_count:
        raise ValueError(
            f"input key '{key}' has {len(tokens)} values; expected at least "
            f"{required_count}"
        )
    try:
        result = tuple(float(token) for token in tokens[:required_count])
    except ValueError as error:
        raise ValueError(f"input key '{key}' contains a non-numeric value") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"input key '{key}' contains a non-finite value")
    return result


def _close(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= 1.0e-10 * scale


@dataclass(frozen=True)
class TwoStreamParams:
    input_path: Path
    save_dir_name: str
    mode_x: int
    lx: float
    dt: float
    nxc: int
    nyc: int
    nzc: int
    periodic: tuple[bool, bool, bool]
    rho_init: tuple[float, ...]
    qom: tuple[float, ...]
    u0: tuple[float, ...]
    electron_indices: tuple[int, int]
    ion_index: int
    omega_pe_sq: float
    drift_speed: float

    @property
    def omega_pe(self) -> float:
        return math.sqrt(self.omega_pe_sq)

    @property
    def k_mode(self) -> float:
        return TWO_PI * self.mode_x / self.lx

    @property
    def mass_ratio(self) -> float:
        electron = self.electron_indices[0]
        return abs(self.qom[electron]) / self.qom[self.ion_index]


def load_params(input_path: Path) -> TwoStreamParams:
    """Load and validate the cold symmetric three-population configuration."""
    values = read_input(input_path)
    case_name = _required_tokens(values, "Case")[0]
    if case_name != "TwoStream":
        raise ValueError(
            f"two-stream analysis requires 'Case = TwoStream', got {case_name!r}"
        )
    ns = _first_int(values, "ns")
    if ns != 3:
        raise ValueError(
            "two-stream analysis requires exactly three physical species "
            f"(two electron beams and one ion population), got ns={ns}"
        )

    rho_init = _float_list(values, "rhoINIT", ns)
    qom = _float_list(values, "qom", ns)
    u0 = _float_list(values, "u0", ns)
    v0 = _float_list(values, "v0", ns)
    w0 = _float_list(values, "w0", ns)
    thermal_speeds = {
        key: _float_list(values, key, ns) for key in ("uth", "vth", "wth")
    }
    electron_indices = tuple(index for index, value in enumerate(qom) if value < 0.0)
    ion_indices = tuple(index for index, value in enumerate(qom) if value > 0.0)
    if len(electron_indices) != 2 or len(ion_indices) != 1:
        raise ValueError(
            "expected exactly two negative-qom electron populations and one "
            f"positive-qom ion population; qom={qom}"
        )

    electron_0, electron_1 = electron_indices
    ion_index = ion_indices[0]
    if any(value <= 0.0 for value in rho_init):
        raise ValueError(f"all rhoINIT values must be positive, got {rho_init}")
    if not _close(rho_init[electron_0], rho_init[electron_1]):
        raise ValueError(
            "electron beams must have equal rhoINIT; got "
            f"{rho_init[electron_0]} and {rho_init[electron_1]}"
        )
    if not _close(abs(qom[electron_0]), abs(qom[electron_1])):
        raise ValueError(
            "electron beams must have equal mass (equal |qom|); got "
            f"{qom[electron_0]} and {qom[electron_1]}"
        )
    drift_0 = u0[electron_0]
    drift_1 = u0[electron_1]
    if _close(drift_0, 0.0) or not _close(drift_0, -drift_1):
        raise ValueError(
            "electron beams must have nonzero, equal-and-opposite x drifts; "
            f"got u0={drift_0} and {drift_1}"
        )
    if not _close(rho_init[ion_index], rho_init[electron_0] + rho_init[electron_1]):
        raise ValueError(
            "charge neutrality requires ion rhoINIT to equal the sum of the "
            "electron-beam rhoINIT values; got "
            f"{rho_init[ion_index]} and "
            f"{rho_init[electron_0] + rho_init[electron_1]}"
        )
    if not _close(u0[ion_index], 0.0):
        raise ValueError(
            f"stationary-ion theory requires ion u0=0, got {u0[ion_index]}"
        )
    if any(not _close(value, 0.0) for value in (*v0, *w0)):
        raise ValueError(
            "two-stream analysis requires x-directed electron drifts and a "
            f"stationary ion: v0={v0}, w0={w0}"
        )
    nonzero_thermal = {
        key: speeds
        for key, speeds in thermal_speeds.items()
        if any(not _close(value, 0.0) for value in speeds)
    }
    if nonzero_thermal:
        details = ", ".join(
            f"{key}={speeds}" for key, speeds in nonzero_thermal.items()
        )
        raise ValueError(
            "the reported cold-beam theory requires zero thermal speeds; " + details
        )

    omega_pe_sq = sum(
        rho_init[index] * abs(qom[index]) for index in electron_indices
    )
    if not math.isfinite(omega_pe_sq) or omega_pe_sq <= 0.0:
        raise ValueError(
            "sum(rhoINIT*abs(qom)) over electron populations must be positive"
        )

    mode_x = _first_int(values, "TwoStreamSeedModeX")
    lx = _first_float(values, "Lx")
    nxc = _first_int(values, "nxc")
    nyc = _first_int(values, "nyc")
    nzc = _first_int(values, "nzc")
    if lx <= 0.0:
        raise ValueError(f"Lx must be positive, got {lx}")
    if min(nxc, nyc, nzc) <= 0:
        raise ValueError(
            f"nxc, nyc, and nzc must be positive, got {(nxc, nyc, nzc)}"
        )
    periodic = periodic_axes(values)
    if not periodic[0]:
        raise ValueError(
            "the seeded mode is a Fourier mode along x: the box must be periodic in x"
        )
    if mode_x <= 0 or 2 * mode_x >= nxc:
        raise ValueError(
            f"TwoStreamSeedModeX={mode_x} must be positive and below the "
            f"x-grid Nyquist mode for nxc={nxc} (require 2*mode < nxc)"
        )

    save_dir_name = _required_tokens(values, "SaveDirName")[0]
    return TwoStreamParams(
        input_path=input_path,
        save_dir_name=save_dir_name,
        mode_x=mode_x,
        lx=lx,
        dt=_first_float(values, "dt"),
        nxc=nxc,
        nyc=nyc,
        nzc=nzc,
        periodic=periodic,
        rho_init=rho_init,
        qom=qom,
        u0=u0,
        electron_indices=(electron_0, electron_1),
        ion_index=ion_index,
        omega_pe_sq=omega_pe_sq,
        drift_speed=0.5 * (abs(drift_0) + abs(drift_1)),
    )


def find_input_path(run_dir: Path, explicit: Path | None) -> Path:
    """Locate twoStream1D.inp in the requested run or repository input dir."""
    if explicit is not None:
        candidates = [explicit]
        if not explicit.is_absolute():
            candidates.append(run_dir / explicit)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"input deck not found; tried: {tried}")

    candidates = [
        run_dir / "twoStream1D.inp",
        _PROJECT_DIR / "input" / "twoStream1D.inp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"could not locate twoStream1D.inp; tried: {tried}. Pass --input explicitly."
    )


def ex_mode_coefficient(ex: np.ndarray, params: TwoStreamParams) -> complex:
    """Return the normalized FFT coefficient of the seeded mode of y,z-mean Ex."""
    nodes = (params.nzc + 1, params.nyc + 1, params.nxc + 1)
    if ex.shape != nodes:
        raise ValueError(f"expected Ex on {nodes} nodes [z,y,x], got shape {ex.shape}")
    # x is periodic: its last node, the image of the first, is dropped.
    profile = transverse_mean(ex, params.periodic)[:-1]
    return complex(np.fft.fft(profile)[params.mode_x] / params.nxc)


def read_mode_history(
    data_dir: Path,
    params: TwoStreamParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Read all Ex snapshots and return ion-time and normalized FFT coefficient."""
    reader = open_field_reader(data_dir / "E")
    times: list[float] = []
    coefficients: list[complex] = []
    try:
        with reader:
            available = reader.available_variables()
            if "Ex" not in available:
                names = ", ".join(sorted(available))
                raise ValueError(
                    f"electric output does not contain 'Ex'; available variables: {names}"
                )
            step_count = int(reader.num_steps())
            if step_count <= 0:
                raise ValueError("electric output contains no time steps")
            for step in range(step_count):
                try:
                    ex = np.asarray(
                        reader.read("Ex", step_selection=[step, 1]), dtype=float
                    )
                    raw_time = np.asarray(
                        reader.read("time", step_selection=[step, 1]), dtype=float
                    ).ravel()
                except Exception as error:
                    raise RuntimeError(
                        f"failed to read electric-field output step {step}: {error}"
                    ) from error
                if raw_time.size != 1 or not math.isfinite(float(raw_time[0])):
                    raise ValueError(
                        f"step {step} has invalid time variable with shape/value {raw_time}"
                    )
                times.append(float(raw_time[0]))
                coefficients.append(ex_mode_coefficient(ex, params))
    except AttributeError as error:
        raise RuntimeError(
            "field reader lacks the expected FileReader API "
            "(available_variables, num_steps, read)"
        ) from error

    time_array = np.asarray(times, dtype=float)
    coefficient_array = np.asarray(coefficients, dtype=complex)
    order = np.argsort(time_array, kind="stable")
    time_array = time_array[order]
    coefficient_array = coefficient_array[order]
    if time_array.size > 1 and np.any(np.diff(time_array) <= 0.0):
        raise ValueError("electric snapshots have duplicate or non-increasing times")
    return time_array, coefficient_array


def cold_symmetric_growth_rate(
    omega_pe: float,
    drift_speed: float,
    wave_number: float,
) -> float:
    """Cold equal-beam growth rate with a stationary ion background.

    The total electron frequency is used in

        1 - (omega_pe^2/2)/(omega-kv0)^2
          - (omega_pe^2/2)/(omega+kv0)^2 = 0.
    """
    ratio = abs(wave_number * drift_speed) / omega_pe
    normalized_gamma_sq = (
        0.5 * math.sqrt(1.0 + 8.0 * ratio * ratio)
        - ratio * ratio
        - 0.5
    )
    return omega_pe * math.sqrt(max(normalized_gamma_sq, 0.0))


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    slope_uncertainty: float
    r_squared: float
    count: int
    mask: np.ndarray


def fit_log_amplitude(
    time_pe: np.ndarray,
    amplitude: np.ndarray,
    fit_start: float,
    fit_end: float,
) -> LinearFit:
    """Fit ``log(amplitude) = intercept + gamma/omega_pe * time_pe``."""
    if not math.isfinite(fit_start) or not math.isfinite(fit_end):
        raise ValueError("fit limits must be finite")
    if fit_end <= fit_start:
        raise ValueError(
            f"fit end must exceed fit start, got [{fit_start}, {fit_end}]"
        )
    # Treat samples microscopically close to a requested boundary as inclusive.
    boundary_tolerance = 1.0e-6 * max(1.0, abs(fit_start), abs(fit_end))
    in_window = (
        (time_pe >= fit_start - boundary_tolerance)
        & (time_pe <= fit_end + boundary_tolerance)
    )
    mask = (
        in_window
        & np.isfinite(time_pe)
        & np.isfinite(amplitude)
        & (amplitude > 0.0)
    )
    count = int(np.count_nonzero(mask))
    if count < 3:
        available = int(
            np.count_nonzero(
                in_window
                & np.isfinite(amplitude)
                & (amplitude > 0.0)
            )
        )
        raise ValueError(
            f"fit window [{fit_start}, {fit_end}] in omega_pe*t contains "
            f"{available} positive finite samples; at least 3 are required"
        )

    x = time_pe[mask]
    y = np.log(amplitude[mask])
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    sxx = float(np.dot(centered_x, centered_x))
    if sxx <= 0.0:
        raise ValueError("fit samples do not span distinct times")
    slope = float(np.dot(centered_x, centered_y) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    sum_squared_error = max(float(np.dot(residual, residual)), 0.0)
    sum_squared_total = float(np.dot(centered_y, centered_y))
    if sum_squared_total > 0.0:
        r_squared = 1.0 - sum_squared_error / sum_squared_total
    else:
        r_squared = 1.0 if sum_squared_error == 0.0 else float("nan")
    variance = sum_squared_error / (count - 2)
    slope_uncertainty = math.sqrt(max(variance / sxx, 0.0))
    return LinearFit(
        slope=slope,
        intercept=intercept,
        slope_uncertainty=slope_uncertainty,
        r_squared=r_squared,
        count=count,
        mask=mask,
    )


def _exp_curve(log_values: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(log_values, -745.0, 709.0))


def comparison_curves(
    time_pe: np.ndarray,
    fit: LinearFit,
    fit_start: float,
    gamma_theory_normalized: float,
) -> tuple[np.ndarray, np.ndarray]:
    numerical = _exp_curve(fit.intercept + fit.slope * time_pe)
    anchor_log = fit.intercept + fit.slope * fit_start
    theory = _exp_curve(
        anchor_log + gamma_theory_normalized * (time_pe - fit_start)
    )
    return numerical, theory


def write_csv(
    path: Path,
    time_ion: np.ndarray,
    time_pe: np.ndarray,
    coefficient: np.ndarray,
    amplitude: np.ndarray,
    numerical_curve: np.ndarray,
    theory_curve: np.ndarray,
    fit_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "time_ion",
                    "omega_pe_t",
                    "Ex_mode_fft_real",
                    "Ex_mode_fft_imag",
                    "Ex_mode_amplitude",
                    "Ex_mode_power",
                    "numerical_fit_amplitude",
                    "cold_theory_amplitude",
                    "in_fit_window",
                ]
            )
            for values in zip(
                time_ion,
                time_pe,
                coefficient.real,
                coefficient.imag,
                amplitude,
                np.abs(coefficient) ** 2,
                numerical_curve,
                theory_curve,
                fit_mask,
            ):
                writer.writerow(
                    [
                        f"{float(values[0]):.17g}",
                        f"{float(values[1]):.17g}",
                        f"{float(values[2]):.17g}",
                        f"{float(values[3]):.17g}",
                        f"{float(values[4]):.17g}",
                        f"{float(values[5]):.17g}",
                        f"{float(values[6]):.17g}",
                        f"{float(values[7]):.17g}",
                        int(bool(values[8])),
                    ]
                )
    except OSError as error:
        raise OSError(f"could not write CSV {path}: {error}") from error


def write_plot(
    path: Path,
    time_pe: np.ndarray,
    amplitude: np.ndarray,
    numerical_curve: np.ndarray,
    theory_curve: np.ndarray,
    fit_start: float,
    fit_end: float,
    fit: LinearFit,
    params: TwoStreamParams,
    gamma_theory_normalized: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positive_amplitude = np.where(amplitude > 0.0, amplitude, np.nan)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.semilogy(
        time_pe,
        positive_amplitude,
        "o-",
        markersize=3.5,
        linewidth=1.2,
        label=rf"PIC $2|\widetilde{{E}}_x(m={params.mode_x})|/N_x$",
    )
    ax.semilogy(
        time_pe,
        numerical_curve,
        "--",
        linewidth=2.0,
        label=rf"log fit: $\gamma/\omega_{{pe}}={fit.slope:.5f}$",
    )
    ax.semilogy(
        time_pe,
        theory_curve,
        ":",
        linewidth=2.4,
        label=rf"cold theory: $\gamma/\omega_{{pe}}={gamma_theory_normalized:.5f}$",
    )
    ax.axvspan(fit_start, fit_end, color="0.88", zorder=0, label="fit window")
    fit_text = (
        rf"$R^2={fit.r_squared:.5f}$" "\n"
        rf"$N_{{fit}}={fit.count}$" "\n"
        rf"$\sigma_\gamma/\omega_{{pe}}={fit.slope_uncertainty:.2e}$"
    )
    ax.text(
        0.025,
        0.975,
        fit_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=FONT_SIZES.fit_box,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ratio = params.k_mode * params.drift_speed / params.omega_pe
    ax.set_title(
        rf"Cold symmetric two-stream mode $m={params.mode_x}$, "
        rf"$kv_0/\omega_{{pe}}={ratio:.4f}$"
    )
    ax.set_xlabel(r"$\omega_{pe}t$")
    ax.set_ylabel(r"physical longitudinal electric-field amplitude")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=200)
    except OSError as error:
        raise OSError(f"could not write plot {path}: {error}") from error
    finally:
        plt.close(fig)


def write_figure(
    path: Path,
    time_pe: np.ndarray,
    curves: tuple[np.ndarray, ...],
    mode_x: int,
    style: FigureStyle,
) -> None:
    """Plot the curves of style against omega_pe*t, as path and as a PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=style.figure_size, constrained_layout=True)
    for index, values in enumerate(curves):
        ax.plot(
            time_pe,
            np.where(values > 0.0, values, np.nan),  # a logarithmic axis holds no zero
            color=style.colors[index],
            linestyle=style.line_styles[index],
            linewidth=style.line_widths[index],
            marker=style.markers[index],
            markersize=style.marker_size,
            markevery=max(1, time_pe.size // style.markers_per_curve),
            zorder=3.0 - index,  # an earlier curve over a later one
            label=style.labels[index],
        )
    ax.set_yscale(style.y_scale)
    ax.set_xlabel(style.x_label)
    ax.set_ylabel(style.y_label.format(mode=mode_x))
    ax.grid(True, alpha=0.3)
    ax.legend(loc=style.legend_location)
    try:
        for figure_path in (path, path.with_suffix(".pdf")):
            fig.savefig(figure_path, dpi=200)
    except OSError as error:
        raise OSError(f"could not write plot {figure_path}: {error}") from error
    finally:
        plt.close(fig)


def read_phase_space(
    data_dir: Path,
    cycle: int,
    electron_indices: tuple[int, ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return x and vx of each electron beam from every rank piece of the particle snapshot of a cycle."""
    arrays = [
        read_polydata(piece, ("position", "velocity", "species"))
        for piece in sorted(data_dir.glob(f"particles_t{cycle:06d}_r*.vtp"))
    ]
    species = np.concatenate([piece["species"] for piece in arrays])
    x = np.concatenate([piece["position"].reshape(-1, 3)[:, 0] for piece in arrays])
    vx = np.concatenate([piece["velocity"].reshape(-1, 3)[:, 0] for piece in arrays])
    return [(x[species == index], vx[species == index]) for index in electron_indices]


def write_phase_space(
    path: Path,
    samples: list[tuple[float, list[tuple[np.ndarray, np.ndarray]]]],
    params: TwoStreamParams,
    style: PhaseSpaceStyle,
) -> None:
    """Plot the electrons in (x/d_e, vx) of each (omega_pe*t, beams) sample, one panel each in a row, as path and as
    a PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(samples), figsize=(style.panel_size[0] * len(samples), style.panel_size[1]),
        sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes[0]
    limit = 1.05 * max(np.abs(vx).max() for _, beams in samples for _, vx in beams)
    for axis, (time_pe, beams) in zip(axes, samples):
        for (x, vx), color, species in zip(beams, style.colors, params.electron_indices):
            axis.scatter(params.omega_pe * x, vx, s=style.dot_size, color=color, alpha=style.alpha,
                         linewidths=0.0, rasterized=True, label=style.label.format(drift=params.u0[species]))
        axis.set_title(style.title.format(time=time_pe), fontsize=FONT_SIZES.panel_title)
        axis.set_xlabel(style.x_label)
        axis.set_xlim(0.0, params.omega_pe * params.lx)
        axis.set_ylim(-limit, limit)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel(style.y_label)
    axes[0].legend(loc=style.legend_location, markerscale=8, framealpha=0.85)
    try:
        for figure_path in (path, path.with_suffix(".pdf")):
            fig.savefig(figure_path, dpi=200)
    except OSError as error:
        raise OSError(f"could not write plot {figure_path}: {error}") from error
    finally:
        plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the seeded Ex Fourier-mode growth in a cold "
            "symmetric two-stream run."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("."),
        help="simulation launch/run directory (default: current directory)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="input deck (default: RUN_DIR/twoStream1D.inp, then repo input)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="field-output directory (default: RUN_DIR/SaveDirName)",
    )
    parser.add_argument(
        "--fit-start",
        type=float,
        default=10.0,
        help="fit start in omega_pe*t (default: 10)",
    )
    parser.add_argument(
        "--fit-end",
        type=float,
        default=25.0,
        help="fit end in omega_pe*t (default: 25)",
    )
    parser.add_argument(
        "--output",
        "--plot",
        dest="plot",
        type=Path,
        default=None,
        help="plot path (default: RUN_DIR/two_stream_mode_growth.png); the plots of the mode history with "
        "the cold theory alone, of the energies and of the electron phase space go beside it, each as a PNG "
        "and as a PDF, with _figure, _energy and _phase_space added to the name",
    )
    parser.add_argument(
        "--phase-space-times",
        type=float,
        nargs="+",
        default=None,
        help="omega_pe*t of the phase-space panels, each taken at the nearest particle snapshot (default: 0, "
        "half the run and the last snapshot)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path (default: RUN_DIR/two_stream_mode_growth.csv)",
    )
    return parser


def analyze(args: argparse.Namespace) -> None:
    run_dir: Path = args.run_dir
    input_path = find_input_path(run_dir, args.input)
    params = load_params(input_path)

    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        configured = Path(params.save_dir_name)
        data_dir = configured if configured.is_absolute() else run_dir / configured

    time_ion, coefficient = read_mode_history(data_dir, params)
    amplitude = 2.0 * np.abs(coefficient)
    time_pe = params.omega_pe * time_ion
    fit = fit_log_amplitude(time_pe, amplitude, args.fit_start, args.fit_end)

    gamma_numerical_normalized = fit.slope
    gamma_numerical_ion = params.omega_pe * gamma_numerical_normalized
    uncertainty_ion = params.omega_pe * fit.slope_uncertainty
    gamma_theory_ion = cold_symmetric_growth_rate(
        params.omega_pe,
        params.drift_speed,
        params.k_mode,
    )
    gamma_theory_normalized = gamma_theory_ion / params.omega_pe
    numerical_curve, theory_curve = comparison_curves(
        time_pe,
        fit,
        args.fit_start,
        gamma_theory_normalized,
    )

    plot_path = args.plot or (run_dir / "two_stream_mode_growth.png")
    csv_path = args.csv or (run_dir / "two_stream_mode_growth.csv")
    write_csv(
        csv_path,
        time_ion,
        time_pe,
        coefficient,
        amplitude,
        numerical_curve,
        theory_curve,
        fit.mask,
    )
    write_plot(
        plot_path,
        time_pe,
        amplitude,
        numerical_curve,
        theory_curve,
        args.fit_start,
        args.fit_end,
        fit,
        params,
        gamma_theory_normalized,
    )
    figure_path = plot_path.with_name(f"{plot_path.stem}_figure.png")
    write_figure(
        figure_path,
        time_pe,
        (amplitude, theory_curve),
        params.mode_x,
        MODE_FIGURE_STYLE,
    )
    energy_path = plot_path.with_name(f"{plot_path.stem}_energy.png")
    rows = read_csv(data_dir) if (data_dir / "diagnostics.csv").exists() else []
    if rows:
        energy = {
            name: np.array([float(row[name]) for row in rows])
            for name in ("time", "kinetic_energy", "electric_energy", "total_energy")
        }
        write_figure(
            energy_path,
            params.omega_pe * energy["time"],
            tuple(
                energy[name] / energy["total_energy"][0]
                for name in ("kinetic_energy", "electric_energy", "total_energy")
            ),
            params.mode_x,
            ENERGY_FIGURE_STYLE,
        )
    phase_space_path = plot_path.with_name(f"{plot_path.stem}_phase_space.png")
    cycles = sorted(
        {int(re.search(r"particles_t(\d+)_r", piece.name).group(1)) for piece in data_dir.glob("particles_t*_r*.vtp")}
    )
    samples = []
    if cycles:
        snapshot_times = params.omega_pe * params.dt * np.asarray(cycles)
        requested = args.phase_space_times or (0.0, 0.5 * snapshot_times[-1], snapshot_times[-1])
        for time_pe in requested:
            nearest = int(np.argmin(np.abs(snapshot_times - time_pe)))
            samples.append(
                (snapshot_times[nearest], read_phase_space(data_dir, cycles[nearest], params.electron_indices))
            )
        write_phase_space(phase_space_path, samples, params, PHASE_SPACE_STYLE)

    kv0_normalized = params.k_mode * params.drift_speed / params.omega_pe
    print(f"input deck                     = {input_path}")
    print(f"seeded mode m                  = {params.mode_x}")
    print(f"grid cells (Nx, Ny, Nz)        = ({params.nxc}, {params.nyc}, {params.nzc})")
    print(f"mass ratio mi/me               = {params.mass_ratio:.9g}")
    print(f"omega_pe^2                     = {params.omega_pe_sq:.9g}")
    print(f"omega_pe                       = {params.omega_pe:.9g}")
    print(f"k [ion-length^-1]              = {params.k_mode:.9g}")
    print(f"beam speed |v0|                = {params.drift_speed:.9g}")
    print(f"k*v0/omega_pe                  = {kv0_normalized:.9g}")
    print(
        "fit window omega_pe*t          = "
        f"[{args.fit_start:.9g}, {args.fit_end:.9g}]"
    )
    print(f"fit samples                    = {fit.count}")
    print(f"fit R^2                        = {fit.r_squared:.9g}")
    print(
        "numerical gamma [ion units]    = "
        f"{gamma_numerical_ion:.9g} +/- {uncertainty_ion:.3g}"
    )
    print(
        "numerical gamma/omega_pe       = "
        f"{gamma_numerical_normalized:.9g} +/- {fit.slope_uncertainty:.3g}"
    )
    print(f"cold theory gamma [ion units]  = {gamma_theory_ion:.9g}")
    print(f"cold theory gamma/omega_pe     = {gamma_theory_normalized:.9g}")
    print(f"CSV written to                 = {csv_path}")
    print(f"plot written to                = {plot_path}")
    print(f"figure written to              = {figure_path}")
    print(f"figure PDF written to          = {figure_path.with_suffix('.pdf')}")
    if rows:
        print(f"energy plot written to         = {energy_path}")
        print(f"energy plot PDF written to     = {energy_path.with_suffix('.pdf')}")
    else:
        print(f"energy plot                    = no diagnostics.csv records in {data_dir}")
    for time_pe, beams in samples:
        spread = ", ".join(f"{np.std(vx):.3g}" for _, vx in beams)
        largest = max(np.abs(vx).max() for _, vx in beams)
        print(f"phase space omega_pe*t = {time_pe:<7.4g}: vx standard deviation per beam = {spread}; "
              f"largest |vx| = {largest:.4g}")
    if samples:
        print(f"phase-space plot written to    = {phase_space_path}")
        print(f"phase-space PDF written to     = {phase_space_path.with_suffix('.pdf')}")
    else:
        print(f"phase-space plot               = no particle snapshots in {data_dir}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        analyze(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
