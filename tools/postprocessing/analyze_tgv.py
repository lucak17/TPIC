#!/usr/bin/env python3
"""Analyze Taylor-Green vortex TPIC output.

The script performs layered checks for a kinetic Taylor-Green vortex (TGV)
run:

* input-derived resolution and time-scale checks,
* analytic t=0 field and species bulk-flow checks,
* div(B), Gauss-law and Ampere-consistency residuals,
* total-energy conservation from diagnostics.csv,
* magnetic/current shell spectra and the magnetic energy in the TGV modes,
* electron pressure-anisotropy diagnostics from the raw momentum-flux output,
* an empirical Weibel-like growth fit from the fastest-growing magnetic shell,
* a local kinetic Weibel growth-rate estimate using pressure-derived
  anisotropy and thermal speed.

TPIC writes ``rho`` multiplied by 4π relative to the internal deposited
charge density.  It writes ``J`` unscaled.  The ``PXX``...``PZZ`` output is a
mass-weighted raw second moment, not a peculiar-pressure tensor.  This script
therefore reconstructs the species bulk velocity from J and rho, then subtracts
the bulk-flow contribution before diagonalizing the electron pressure tensor.

Example from the repository root:

    python3 tools/postprocessing/analyze_tgv.py --run-dir /path/to/run

Useful options:

    python3 tools/postprocessing/analyze_tgv.py \
        --run-dir /path/to/run \
        --weibel-fit-start 150 \
        --weibel-fit-end 450 \
        --weibel-min-kde 0.25 \
        --weibel-max-kde 2.5

    python3 tools/postprocessing/analyze_tgv.py \
        --run-dir /path/to/run \
        --from-csv /path/to/run/tgv_analysis.csv \
        --weibel-fit-start 450 \
        --weibel-fit-end 1350
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # source-checkout imports
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.fields import field_data_exists, open_field_reader


TWO_PI = 2.0 * math.pi
FOUR_PI = 4.0 * math.pi
EPS = 1.0e-30




def first_float(values: dict[str, list[str]], key: str, default: float | None = None) -> float:
    if key not in values:
        if default is None:
            raise KeyError(f"missing required input key {key}")
        return default
    return float(values[key][0])


def first_int(values: dict[str, list[str]], key: str, default: int | None = None) -> int:
    return int(first_float(values, key, None if default is None else float(default)))


def first_bool(values: dict[str, list[str]], key: str, default: bool = False) -> bool:
    if key not in values:
        return default
    # TPIC spells a boolean 0, 1, true or false.
    return values[key][0].strip() in {"1", "true"}


def first_string(values: dict[str, list[str]], key: str, default: str) -> str:
    if key not in values:
        return default
    return values[key][0].strip()


def floats(values: dict[str, list[str]], key: str, default: Iterable[float]) -> list[float]:
    if key not in values:
        return list(default)
    return [float(token) for token in values[key]]


def qom_mass_ratio(qoms: list[float]) -> float:
    electron = next((abs(qom) for qom in qoms if qom < 0.0), None)
    ion = next((qom for qom in qoms if qom > 0.0), None)
    if electron is None or ion is None:
        return float("nan")
    return electron / ion


def electron_species_index(qoms: tuple[float, ...]) -> int | None:
    for index, qom in enumerate(qoms):
        if qom < 0.0:
            return index
    return None


def species_charge_sign(qom: float) -> float:
    return -1.0 if qom < 0.0 else 1.0


@dataclass(frozen=True)
class TgvParams:
    input_path: Path
    run_dir: Path
    data_dir: Path
    mode_x: int
    mode_y: int
    mode_z: int
    lx: float
    ly: float
    lz: float
    nxc: int
    nyc: int
    nzc: int
    u0: float
    b1: float
    b0x: float
    b0y: float
    b0z: float
    seed_motional_e: bool
    dt: float
    smooth: float
    smooth_passes: int
    smooth_operator: str
    ncycles: int
    field_output_cycle: int
    qoms: tuple[float, ...]
    rho_init: tuple[float, ...]
    ns: int
    c: float

    @property
    def lengths(self) -> tuple[float, float, float]:
        return (self.lx, self.ly, self.lz)

    @property
    def cells(self) -> tuple[int, int, int]:
        return (self.nxc, self.nyc, self.nzc)

    @property
    def modes(self) -> tuple[int, int, int]:
        return (self.mode_x, self.mode_y, self.mode_z)

    @property
    def kx(self) -> float:
        return TWO_PI * self.mode_x / self.lx if self.lx > 0.0 else float("nan")

    @property
    def ky(self) -> float:
        return TWO_PI * self.mode_y / self.ly if self.ly > 0.0 else float("nan")

    @property
    def kz(self) -> float:
        return TWO_PI * self.mode_z / self.lz if self.lz > 0.0 else float("nan")

    @property
    def mu(self) -> float:
        return qom_mass_ratio(list(self.qoms))

    @property
    def omega_pe(self) -> float:
        return math.sqrt(self.mu) if self.mu > 0.0 else float("nan")

    @property
    def de(self) -> float:
        return 1.0 / self.omega_pe if self.omega_pe > 0.0 else float("nan")

    @property
    def b0_magnitude(self) -> float:
        return math.sqrt(self.b0x * self.b0x + self.b0y * self.b0y + self.b0z * self.b0z)

    @property
    def tau_tg(self) -> float:
        k_ref = max(abs(self.kx), abs(self.ky), abs(self.kz))
        if k_ref == 0.0 or self.u0 == 0.0:
            return float("inf")
        return 1.0 / (k_ref * abs(self.u0))


def load_params(input_path: Path, run_dir: Path, data_dir_override: Path | None) -> TgvParams:
    raw = read_input(input_path)
    # The spectra and the analytic fields cover the distinct nodes of a box periodic along every axis.
    if not all(periodic_axes(raw)):
        raise ValueError(f"{input_path}: the analysis needs a box periodic in x, y and z")
    mode = first_int(raw, "TaylorGreenVortexMode", 1)
    save_dir = first_string(raw, "SaveDirName", "data")
    qoms = tuple(floats(raw, "qom", [-1.0, 1.0]))
    rho_init = tuple(floats(raw, "rhoINIT", [1.0] * len(qoms)))
    ns = first_int(raw, "ns", len(qoms))
    smooth = first_float(raw, "Smooth", 1.0)
    smoothing = first_string(raw, "ECSimSmoothing", "none") == "energy_conserving" and smooth < 1.0
    data_dir = data_dir_override or (run_dir / save_dir)
    return TgvParams(
        input_path=input_path,
        run_dir=run_dir,
        data_dir=data_dir,
        mode_x=first_int(raw, "TaylorGreenVortexModeX", mode),
        mode_y=first_int(raw, "TaylorGreenVortexModeY", mode),
        mode_z=first_int(raw, "TaylorGreenVortexModeZ", mode),
        lx=first_float(raw, "Lx"),
        ly=first_float(raw, "Ly"),
        lz=first_float(raw, "Lz"),
        nxc=first_int(raw, "nxc"),
        nyc=first_int(raw, "nyc"),
        nzc=first_int(raw, "nzc"),
        u0=first_float(raw, "TaylorGreenVortexVelocityAmplitude", 0.0),
        b1=first_float(raw, "TaylorGreenVortexMagneticAmplitude", 0.0),
        b0x=first_float(raw, "B0x", 0.0),
        b0y=first_float(raw, "B0y", 0.0),
        b0z=first_float(raw, "B0z", 0.0),
        seed_motional_e=first_bool(raw, "TaylorGreenVortexSeedMotionalE", False),
        dt=first_float(raw, "dt", float("nan")),
        smooth=smooth,
        smooth_passes=first_int(raw, "SmoothNiter", 6) if smoothing else 0,
        smooth_operator=first_string(raw, "ECSimSmoothingOperator", "face7"),
        ncycles=first_int(raw, "ncycles", 0),
        field_output_cycle=first_int(raw, "FieldOutputCycle", 1),
        qoms=qoms,
        rho_init=rho_init,
        ns=ns,
        c=first_float(raw, "c", 1.0),
    )


def find_input_path(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidate = run_dir / "TaylorGreenVortex3D.inp"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find a TaylorGreenVortex input deck in {run_dir}. "
        "Pass --input explicitly."
    )


def find_output_log(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        candidates = [explicit]
        if not explicit.is_absolute():
            candidates.append(run_dir / explicit)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"output log not found: {explicit}")

    candidates: list[Path] = []
    for pattern in ("tgv_*_ecsim.out", "*ecsim*.out", "*.out"):
        candidates.extend(sorted(run_dir.glob(pattern)))
    unique = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    return unique[0] if unique else None


def read_scalar(reader, name: str, step: int) -> float:
    return float(np.asarray(reader.read(name, step_selection=[step, 1])).ravel()[0])


def read_array(reader, name: str, step: int) -> np.ndarray:
    return np.asarray(reader.read(name, step_selection=[step, 1]), dtype=float)


def read_vector(reader, base_name: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        read_array(reader, base_name + "x", step),
        read_array(reader, base_name + "y", step),
        read_array(reader, base_name + "z", step),
    )


def trim_periodic_nodes(arr: np.ndarray, cells: tuple[int, int, int]) -> np.ndarray:
    nxc, nyc, nzc = cells
    if arr.shape == (nzc + 1, nyc + 1, nxc + 1):
        return arr[:-1, :-1, :-1]
    return arr


def trim_vector(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    cells: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(trim_periodic_nodes(component, cells) for component in vec)  # type: ignore[return-value]


def node_mesh(params: TgvParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nxc, nyc, nzc = params.cells
    x = np.arange(nxc, dtype=float) * params.lx / nxc
    y = np.arange(nyc, dtype=float) * params.ly / nyc
    z = np.arange(nzc, dtype=float) * params.lz / nzc
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz


def trig_factors(params: TgvParams) -> tuple[np.ndarray, ...]:
    x, y, z = node_mesh(params)
    sin_x = np.sin(params.kx * x)
    cos_x = np.cos(params.kx * x)
    sin_y = np.sin(params.ky * y)
    cos_y = np.cos(params.ky * y)
    sin_z = np.sin(params.kz * z)
    cos_z = np.cos(params.kz * z)
    return sin_x, cos_x, sin_y, cos_y, sin_z, cos_z


def analytic_velocity(params: TgvParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_x, cos_x, sin_y, cos_y, _, cos_z = trig_factors(params)
    ux = params.u0 * sin_x * cos_y * cos_z
    uy = -params.u0 * cos_x * sin_y * cos_z
    uz = np.zeros_like(ux)
    return ux, uy, uz


def analytic_b(params: TgvParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_x, cos_x, sin_y, cos_y, _, cos_z = trig_factors(params)
    bx = np.full_like(sin_x, params.b0x) + params.b1 * sin_x * cos_y * cos_z
    by = np.full_like(sin_x, params.b0y) - params.b1 * cos_x * sin_y * cos_z
    bz = np.full_like(sin_x, params.b0z)
    return bx, by, bz


def analytic_curl_b(params: TgvParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_x, cos_x, sin_y, cos_y, sin_z, cos_z = trig_factors(params)
    jx = -params.b1 * params.kz * cos_x * sin_y * sin_z
    jy = -params.b1 * params.kz * sin_x * cos_y * sin_z
    jz = params.b1 * (params.kx + params.ky) * sin_x * sin_y * cos_z
    return jx, jy, jz


def analytic_e(params: TgvParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ux, uy, uz = analytic_velocity(params)
    bx, by, bz = analytic_b(params)
    if not params.seed_motional_e or params.u0 == 0.0:
        zeros = np.zeros_like(ux)
        return zeros, zeros.copy(), zeros.copy()
    ex = -(uy * bz - uz * by)
    ey = -(uz * bx - ux * bz)
    ez = -(ux * by - uy * bx)
    return ex, ey, ez


def analytic_induction_dt(params: TgvParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_x, cos_x, sin_y, cos_y, sin_z, cos_z = trig_factors(params)
    dux_dx = params.u0 * params.kx * cos_x * cos_y * cos_z
    dux_dy = -params.u0 * params.ky * sin_x * sin_y * cos_z
    dux_dz = -params.u0 * params.kz * sin_x * cos_y * sin_z
    duy_dx = params.u0 * params.kx * sin_x * sin_y * cos_z
    duy_dy = -params.u0 * params.ky * cos_x * cos_y * cos_z
    duy_dz = params.u0 * params.kz * cos_x * sin_y * sin_z

    dbx_dt = params.b0x * dux_dx + params.b0y * dux_dy + params.b0z * dux_dz
    dby_dt = params.b0x * duy_dx + params.b0y * duy_dy + params.b0z * duy_dz
    dbz_dt = np.zeros_like(dbx_dt)
    return dbx_dt, dby_dt, dbz_dt


def derivative(arr: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * spacing)


def grid_spacing(params: TgvParams, shape: tuple[int, int, int]) -> tuple[float, float, float]:
    nz, ny, nx = shape
    return params.lx / nx, params.ly / ny, params.lz / nz


def node_to_cell(arr: np.ndarray, axes: Iterable[int] = (0, 1, 2)) -> np.ndarray:
    # Cell (k, j, i) lies between nodes k..k+1, j..j+1 and i..i+1.
    for axis in axes:
        arr = 0.5 * (arr + np.roll(arr, -1, axis=axis))
    return arr


def divergence(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    params: TgvParams,
) -> np.ndarray:
    # The solver's node-to-cell divergence: per axis, the node difference
    # averaged over the four cell edges along it (NodeCellSamples::derivative in
    # ecsim/fields/node_cell_curl.hpp). It annihilates the solver's cell-to-node
    # curl of B, so div(E) changes only through J. On the nodal output B, the
    # eight-cell average, it gives the cell average of the solver's nodal div(B).
    return sum(
        node_to_cell((np.roll(c, -1, axis=a) - c) / h, [other for other in (0, 1, 2) if other != a])
        for c, a, h in zip(vec, (2, 1, 0), grid_spacing(params, vec[0].shape))
    )


def smooth_nodes(arr: np.ndarray, params: TgvParams) -> np.ndarray:
    # The solver's energy-conserving smoothing S (ecsim/fields/node_smoothing.hpp),
    # the identity when it is off.
    for _ in range(params.smooth_passes):
        if params.smooth_operator == "binomial27":
            for axis in range(3):
                arr = 0.25 * (np.roll(arr, 1, axis=axis) + 2.0 * arr + np.roll(arr, -1, axis=axis))
        else:
            faces = sum(np.roll(arr, shift, axis=axis) for axis in range(3) for shift in (-1, 1))
            arr = params.smooth * arr + (1.0 - params.smooth) / 6.0 * faces
    return arr


def gauss_residual(
    e_vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho: np.ndarray,
    j_vec: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    params: TgvParams,
) -> float:
    """Relative cell-centred Gauss residual div(E) - 4*pi*rho.

    rho is the output 4*pi*rho and J the output current of the same snapshot.
    The solve only sees S(J), so the charge consistent with div(E) is the
    eight-node average of S(rho). Snapshot particles sit half a step after E:
    their density is moved back by dt/2*div(J), the linearised midpoint of the
    positions at n-1/2 and n+1/2 (skipped without J).
    """
    div_e = divergence(e_vec, params)
    charge = node_to_cell(smooth_nodes(rho, params))
    if j_vec is not None:
        smoothed = tuple(smooth_nodes(component, params) for component in j_vec)
        charge = charge + 0.5 * params.dt * FOUR_PI * divergence(smoothed, params)  # type: ignore[arg-type]
    return relative_scalar_residual(div_e - charge, div_e, charge)


def curl(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    params: TgvParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vx, vy, vz = vec
    dx, dy, dz = grid_spacing(params, vx.shape)
    cx = derivative(vz, axis=1, spacing=dy) - derivative(vy, axis=0, spacing=dz)
    cy = derivative(vx, axis=0, spacing=dz) - derivative(vz, axis=2, spacing=dx)
    cz = derivative(vy, axis=2, spacing=dx) - derivative(vx, axis=1, spacing=dy)
    return cx, cy, cz


def scalar_rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(arr) ** 2)))


def vector_rms(vec: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    return float(np.sqrt(np.mean(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)))


def vector_max(vec: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    return float(np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2).max())


def vector_diff(
    lhs: tuple[np.ndarray, np.ndarray, np.ndarray],
    rhs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]


def relative_vector_error(
    measured: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    floor: float = EPS,
) -> float:
    return vector_rms(vector_diff(measured, expected)) / max(vector_rms(expected), floor)


def relative_scalar_residual(residual: np.ndarray, *references: np.ndarray) -> float:
    denom = sum(scalar_rms(ref) for ref in references)
    return scalar_rms(residual) / max(denom, EPS)


def relative_vector_residual(
    residual: tuple[np.ndarray, np.ndarray, np.ndarray],
    *references: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    denom = sum(vector_rms(ref) for ref in references)
    return vector_rms(residual) / max(denom, EPS)


@dataclass
class Diagnostics:
    cycle: np.ndarray
    time: np.ndarray
    e_energy: np.ndarray
    b_energy: np.ndarray
    field_energy: np.ndarray
    kinetic_energy: np.ndarray
    total_energy: np.ndarray


def read_diagnostics(path: Path) -> Diagnostics | None:
    if not path.exists():
        return None
    with path.open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    if not rows:
        return None
    columns = {key: np.asarray([float(row[key]) for row in rows]) for key in
               ("cycle", "time", "electric_energy", "magnetic_energy", "kinetic_energy", "total_energy")}
    return Diagnostics(columns["cycle"], columns["time"], columns["electric_energy"],
                       columns["magnetic_energy"], columns["electric_energy"] + columns["magnetic_energy"],
                       columns["kinetic_energy"], columns["total_energy"])


@dataclass
class OutputScanHistory:
    cycle: np.ndarray
    time: np.ndarray
    electric_energy: np.ndarray
    magnetic_energy: np.ndarray
    kinetic_energy: np.ndarray
    system_energy: np.ndarray
    solver_iterations: np.ndarray
    solver_residual: np.ndarray


def scan_output_log(path: Path) -> OutputScanHistory | None:
    """One row per cycle header and for the final state of the rank-0 console log,
    with the cycle's GMRES line and, on energy cycles, its field and total energy lines."""
    if not path.exists():
        return None

    number = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    cycle_re = re.compile(
        rf"ECSIM (?:Cycle\s+(\d+)\s*\||Final State\s*\|\s*cycle\s*=\s*(\d+),)\s*t\s*=\s*{number}"
    )
    line_patterns = (
        (re.compile(rf"Maxwell GMRES\s*:\s*converged in\s+(\d+)\s+iterations;\s*residual\s*=\s*{number}"),
         ("solver_iterations", "solver_residual")),
        (re.compile(rf"Field energy\s*:\s*electric\s*=\s*{number}\s+magnetic\s*=\s*{number}"),
         ("electric_energy", "magnetic_energy")),
        (re.compile(rf"Total energy\s*:\s*system\s*=\s*{number}\s+kinetic\s*=\s*{number}"),
         ("system_energy", "kinetic_energy")),
    )

    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        cycle_match = cycle_re.search(line)
        if cycle_match is not None:
            cycle = cycle_match.group(1) or cycle_match.group(2)
            rows.append({"cycle": float(cycle), "time": float(cycle_match.group(3))})
            continue
        if not rows:
            continue
        for pattern, names in line_patterns:
            match = pattern.search(line)
            if match is not None:
                rows[-1].update(zip(names, (float(value) for value in match.groups())))
                break
    if not rows:
        return None

    def column(name: str) -> np.ndarray:
        return np.asarray([row.get(name, float("nan")) for row in rows], dtype=float)

    return OutputScanHistory(
        cycle=column("cycle").astype(int),
        time=column("time"),
        electric_energy=column("electric_energy"),
        magnetic_energy=column("magnetic_energy"),
        kinetic_energy=column("kinetic_energy"),
        system_energy=column("system_energy"),
        solver_iterations=column("solver_iterations"),
        solver_residual=column("solver_residual"),
    )


@dataclass(frozen=True)
class SpectralGrid:
    k_de: np.ndarray
    mode_shell: np.ndarray
    tgv_mask: np.ndarray
    nonzero_mask: np.ndarray


def spectral_grid(shape: tuple[int, int, int], params: TgvParams) -> SpectralGrid:
    nz, ny, nx = shape
    mx = np.fft.fftfreq(nx, d=1.0 / nx)
    my = np.fft.fftfreq(ny, d=1.0 / ny)
    mz = np.fft.fftfreq(nz, d=1.0 / nz)
    zz, yy, xx = np.meshgrid(mz, my, mx, indexing="ij")
    k_di = np.sqrt(
        (TWO_PI * xx / params.lx) ** 2
        + (TWO_PI * yy / params.ly) ** 2
        + (TWO_PI * zz / params.lz) ** 2
    )
    k_de = k_di / params.omega_pe if params.omega_pe > 0.0 else np.full_like(k_di, np.nan)
    mode_shell = np.rint(np.sqrt(xx * xx + yy * yy + zz * zz)).astype(int)
    tgv_mask = (
        (np.abs(xx).astype(int) == params.mode_x)
        & (np.abs(yy).astype(int) == params.mode_y)
        & (np.abs(zz).astype(int) == params.mode_z)
    )
    nonzero_mask = (xx != 0.0) | (yy != 0.0) | (zz != 0.0)
    return SpectralGrid(k_de=k_de, mode_shell=mode_shell, tgv_mask=tgv_mask, nonzero_mask=nonzero_mask)


def fft_components(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    subtract_mean: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = vec[0].size
    components = []
    for component in vec:
        data = component - component.mean() if subtract_mean else component
        components.append(np.fft.fftn(data) / n)
    return components[0], components[1], components[2]


def spectral_energy_density(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    subtract_mean: bool = True,
) -> np.ndarray:
    fx, fy, fz = fft_components(vec, subtract_mean=subtract_mean)
    return 0.5 * (np.abs(fx) ** 2 + np.abs(fy) ** 2 + np.abs(fz) ** 2)


def band_energy(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: SpectralGrid,
    kde_min: float,
    kde_max: float | None,
) -> float:
    return band_energy_and_k_stats(vec, grid, kde_min, kde_max)[0]


def band_energy_and_k_stats(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: SpectralGrid,
    kde_min: float,
    kde_max: float | None,
) -> tuple[float, float, float]:
    energy = spectral_energy_density(vec, subtract_mean=True)
    return band_energy_and_k_stats_from_density(energy, grid, kde_min, kde_max)


def band_energy_and_k_stats_from_density(
    energy: np.ndarray,
    grid: SpectralGrid,
    kde_min: float,
    kde_max: float | None,
) -> tuple[float, float, float]:
    mask = grid.nonzero_mask & (grid.k_de >= kde_min)
    if kde_max is not None:
        mask &= grid.k_de <= kde_max
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")

    band = energy[mask]
    kde = grid.k_de[mask]
    total = float(band.sum())
    if total <= 0.0:
        return total, float("nan"), float("nan")
    peak_kde = float(kde[np.argmax(band)])
    weighted_kde = float(np.sum(kde * band) / total)
    return total, peak_kde, weighted_kde


def tgv_mode_energy(vec: tuple[np.ndarray, np.ndarray, np.ndarray], grid: SpectralGrid) -> float:
    energy = spectral_energy_density(vec, subtract_mean=True)
    return tgv_mode_energy_from_density(energy, grid)


def tgv_mode_energy_from_density(energy: np.ndarray, grid: SpectralGrid) -> float:
    return float(energy[grid.tgv_mask].sum())


def fluctuation_energy(vec: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    return 0.5 * (
        scalar_rms(vec[0] - vec[0].mean()) ** 2
        + scalar_rms(vec[1] - vec[1].mean()) ** 2
        + scalar_rms(vec[2] - vec[2].mean()) ** 2
    )


def shell_spectrum(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: SpectralGrid,
    max_shell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = spectral_energy_density(vec, subtract_mean=True)
    return shell_spectrum_from_density(energy, grid, max_shell)


def shell_spectrum_from_density(
    energy: np.ndarray,
    grid: SpectralGrid,
    max_shell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shells = np.arange(max_shell + 1)
    spectrum = np.zeros_like(shells, dtype=float)
    kde_center = np.zeros_like(shells, dtype=float)
    for shell in shells:
        mask = grid.mode_shell == shell
        if np.any(mask):
            spectrum[shell] = float(energy[mask].sum())
            kde_center[shell] = float(np.mean(grid.k_de[mask]))
        else:
            kde_center[shell] = float("nan")
    return shells, kde_center, spectrum


def fit_exponential_growth(
    time_pe: np.ndarray,
    energy: np.ndarray,
    fit_start: float,
    fit_end: float,
) -> tuple[float, float, float, int]:
    keep = (
        np.isfinite(time_pe)
        & np.isfinite(energy)
        & (energy > 0.0)
        & (time_pe >= fit_start)
        & (time_pe <= fit_end)
    )
    n = int(np.count_nonzero(keep))
    if n < 3:
        raise ValueError(
            f"fit window [{fit_start}, {fit_end}] contains fewer than 3 positive samples"
        )
    x = time_pe[keep]
    y = np.log(energy[keep])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    gamma_over_omega_pe = 0.5 * slope
    return float(gamma_over_omega_pe), float(intercept), r2, n


def automatic_fit_window(time_pe: np.ndarray, energy: np.ndarray) -> tuple[float, float]:
    valid = np.flatnonzero(np.isfinite(time_pe) & np.isfinite(energy) & (energy > 0.0))
    if valid.size < 3:
        return float(time_pe[0]), float(time_pe[-1])

    start = valid[1] if valid.size > 3 else valid[0]
    end = valid[-1]
    peak = valid[np.argmax(energy[valid])]
    if peak - start >= 3:
        end = peak

    span = end - start + 1
    if span >= 6:
        start = start + span // 5
        end = start + max(2, (3 * span) // 5)
        end = min(end, valid[-1])

    if end <= start:
        start, end = valid[0], valid[-1]
    return float(time_pe[start]), float(time_pe[end])


def erfcx_positive(x: float) -> float:
    """Scaled complementary error function for x >= 0 without requiring SciPy."""
    if x < 0.0:
        raise ValueError("erfcx_positive expects x >= 0")
    if x < 26.0:
        return math.exp(x * x) * math.erfc(x)
    inv = 1.0 / x
    inv2 = inv * inv
    return inv / math.sqrt(math.pi) * (1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2)


def kinetic_weibel_growth_rate(k_de: float, v_perp_c: float, anisotropy: float) -> float:
    """Return gamma/omega_pe for a bi-Maxwellian electron Weibel mode.

    The velocity-space width is the Gaussian standard deviation
    sqrt(P_perp / rho_mass), matching TPIC's thermal-speed convention (uth = sqrt(kT/m)).
    """
    if not (np.isfinite(k_de) and np.isfinite(v_perp_c) and np.isfinite(anisotropy)):
        return float("nan")
    if k_de <= 0.0 or v_perp_c <= 0.0 or anisotropy <= 0.0:
        return float("nan")

    sqrt_pi = math.sqrt(math.pi)

    def dispersion(gamma: float) -> float:
        eta = gamma / (math.sqrt(2.0) * k_de * v_perp_c)
        rhs = anisotropy - (anisotropy + 1.0) * sqrt_pi * eta * erfcx_positive(eta)
        return k_de * k_de + gamma * gamma - rhs

    f_lo = dispersion(0.0)
    if f_lo >= 0.0:
        return float("nan")

    lo = 0.0
    hi = max(math.sqrt(anisotropy), math.sqrt(max(anisotropy - k_de * k_de, 0.0)), 1.0e-8)
    f_hi = dispersion(hi)
    for _ in range(64):
        if f_hi > 0.0:
            break
        hi *= 2.0
        f_hi = dispersion(hi)
    else:
        return float("nan")

    for _ in range(96):
        mid = 0.5 * (lo + hi)
        f_mid = dispersion(mid)
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass
class InitChecks:
    b_relative_error: float = float("nan")
    e_relative_error: float = float("nan")
    divb_relative: float = float("nan")
    gauss_relative: float = float("nan")
    charge_mean: float = float("nan")
    charge_rms: float = float("nan")
    current_rms: float = float("nan")
    ampere_relative: float = float("nan")
    induction_scale: float = float("nan")
    induction_correlation: float = float("nan")
    induction_relative_residual: float = float("nan")
    velocity_error_by_species: dict[int, float] | None = None
    velocity_tgv_fraction_by_species: dict[int, float] | None = None


def velocity_from_moments(
    j_vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.abs(rho_output) > 1.0e-14
    out = []
    for component in j_vec:
        velocity = np.full_like(component, np.nan)
        velocity[mask] = FOUR_PI * component[mask] / rho_output[mask]
        out.append(velocity)
    return out[0], out[1], out[2]


def pressure_output_available(params: TgvParams, species_index: int) -> bool:
    components = ("PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ")
    return (
        field_data_exists(params.data_dir, f"rho_{species_index}")
        and field_data_exists(params.data_dir, f"J_{species_index}")
        and all(field_data_exists(params.data_dir, f"{component}_{species_index}") for component in components)
    )


def pressure_tensor_from_raw_moments(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho_output: np.ndarray,
    qom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert TPIC raw second moments to a peculiar pressure tensor.

    Disk ``rho_s`` is 4π times the internal charge density.  Disk ``J_s`` is
    unscaled, so ``u = 4π J_s / rho_s``.  Disk ``Pij_s`` is already divided by
    qom and is therefore a mass-weighted raw second moment in internal-density
    units.  The mass density in the same units is ``(rho_s / 4π) / qom``.
    """
    ux, uy, uz = velocity
    mass_density = rho_output / (FOUR_PI * qom)

    pxx = raw[0] - mass_density * ux * ux
    pxy = raw[1] - mass_density * ux * uy
    pxz = raw[2] - mass_density * ux * uz
    pyy = raw[3] - mass_density * uy * uy
    pyz = raw[4] - mass_density * uy * uz
    pzz = raw[5] - mass_density * uz * uz

    tensor = np.empty(pxx.shape + (3, 3), dtype=float)
    tensor[..., 0, 0] = pxx
    tensor[..., 0, 1] = pxy
    tensor[..., 0, 2] = pxz
    tensor[..., 1, 0] = pxy
    tensor[..., 1, 1] = pyy
    tensor[..., 1, 2] = pyz
    tensor[..., 2, 0] = pxz
    tensor[..., 2, 1] = pyz
    tensor[..., 2, 2] = pzz
    return tensor, mass_density


def pressure_anisotropy_summary(
    pressure_tensor: np.ndarray,
    mass_density: np.ndarray,
    top_fraction: float,
) -> dict[str, float]:
    evals = np.linalg.eigvalsh(0.5 * (pressure_tensor + np.swapaxes(pressure_tensor, -1, -2)))
    evals = evals[..., ::-1]
    p1 = evals[..., 0]
    p2 = evals[..., 1]
    p3 = evals[..., 2]
    pperp = 0.5 * (p2 + p3)

    valid = (
        np.isfinite(p1)
        & np.isfinite(p2)
        & np.isfinite(p3)
        & np.isfinite(mass_density)
        & (mass_density > EPS)
        & (p3 > EPS)
        & (pperp > EPS)
    )
    if not np.any(valid):
        return {
            "valid_fraction": 0.0,
            "ae_rms": float("nan"),
            "ae_max": float("nan"),
            "ae_top_median": float("nan"),
            "vperp_top_median": float("nan"),
            "p2_over_p3_top_median": float("nan"),
        }

    ae = 2.0 * p1[valid] / (p2[valid] + p3[valid]) - 1.0
    vperp = np.sqrt(pperp[valid] / mass_density[valid])
    p2_over_p3 = p2[valid] / p3[valid]

    finite = np.isfinite(ae) & np.isfinite(vperp) & np.isfinite(p2_over_p3)
    if not np.any(finite):
        return {
            "valid_fraction": float(np.count_nonzero(valid) / valid.size),
            "ae_rms": float("nan"),
            "ae_max": float("nan"),
            "ae_top_median": float("nan"),
            "vperp_top_median": float("nan"),
            "p2_over_p3_top_median": float("nan"),
        }

    ae = ae[finite]
    vperp = vperp[finite]
    p2_over_p3 = p2_over_p3[finite]

    fraction = min(max(top_fraction, 0.0), 1.0)
    if fraction <= 0.0:
        top = np.ones_like(ae, dtype=bool)
    else:
        threshold = float(np.quantile(ae, max(0.0, 1.0 - fraction)))
        top = ae >= threshold
        if not np.any(top):
            top = np.ones_like(ae, dtype=bool)

    return {
        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
        "ae_rms": float(np.sqrt(np.mean(ae * ae))),
        "ae_max": float(np.max(ae)),
        "ae_top_median": float(np.median(ae[top])),
        "vperp_top_median": float(np.median(vperp[top])),
        "p2_over_p3_top_median": float(np.median(p2_over_p3[top])),
    }


def current_split_for_species(params: TgvParams, species_index: int) -> float:
    if params.mu <= 0.0 or species_index >= len(params.qoms):
        return 0.0
    if params.qoms[species_index] < 0.0:
        return -params.mu / (params.mu + 1.0)
    return 1.0 / (params.mu + 1.0)


def expected_species_velocity(
    params: TgvParams,
    species_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ux, uy, uz = analytic_velocity(params)
    if params.b1 == 0.0:
        return ux, uy, uz
    density = params.rho_init[species_index] if species_index < len(params.rho_init) else 1.0
    split = current_split_for_species(params, species_index) / max(density, EPS)
    jx, jy, jz = analytic_curl_b(params)
    return ux + split * jx, uy + split * jy, uz + split * jz


def vector_mode_fraction(
    vec: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: SpectralGrid,
) -> float:
    total = float(spectral_energy_density(vec, subtract_mean=True)[grid.nonzero_mask].sum())
    if total <= 0.0:
        return float("nan")
    return tgv_mode_energy(vec, grid) / total


def compute_initial_checks(params: TgvParams) -> InitChecks:
    checks = InitChecks(velocity_error_by_species={}, velocity_tgv_fraction_by_species={})
    data_dir = params.data_dir

    b0_vec = None
    j0_vec = None
    grid = spectral_grid((params.nzc, params.nyc, params.nxc), params)
    if field_data_exists(data_dir, "J"):
        with open_field_reader(data_dir / "J") as reader:
            j0_vec = trim_vector(read_vector(reader, "J", 0), params.cells)
    if field_data_exists(data_dir, "B"):
        with open_field_reader(data_dir / "B") as reader:
            b0_vec = trim_vector(read_vector(reader, "B", 0), params.cells)
        b_expected = analytic_b(params)
        checks.b_relative_error = relative_vector_error(b0_vec, b_expected)
        div_b = divergence(b0_vec, params)
        b_ref = max(params.b0_magnitude, abs(params.b1), vector_rms(b0_vec), EPS)
        k_ref = max(abs(params.kx), abs(params.ky), abs(params.kz), 1.0 / max(params.lx, params.ly, params.lz))
        checks.divb_relative = scalar_rms(div_b) / max(k_ref * b_ref, EPS)

    if field_data_exists(data_dir, "E"):
        with open_field_reader(data_dir / "E") as reader:
            e0_vec = trim_vector(read_vector(reader, "E", 0), params.cells)
        e_expected = analytic_e(params)
        e_scale = max(
            vector_rms(e_expected),
            params.b0_magnitude * abs(params.u0),
            abs(params.b1) * abs(params.u0),
            EPS,
        )
        checks.e_relative_error = vector_rms(vector_diff(e0_vec, e_expected)) / e_scale
        if field_data_exists(data_dir, "rho"):
            with open_field_reader(data_dir / "rho") as reader:
                rho = trim_periodic_nodes(read_array(reader, "rho", 0), params.cells)
            checks.gauss_relative = gauss_residual(e0_vec, rho, j0_vec, params)
            checks.charge_mean = float(rho.mean())
            checks.charge_rms = scalar_rms(rho)

    if j0_vec is not None:
        checks.current_rms = vector_rms((FOUR_PI * j0_vec[0], FOUR_PI * j0_vec[1], FOUR_PI * j0_vec[2]))
        if b0_vec is not None:
            curl_b = curl(b0_vec, params)
            ampere_residual = (
                curl_b[0] / FOUR_PI - j0_vec[0],
                curl_b[1] / FOUR_PI - j0_vec[1],
                curl_b[2] / FOUR_PI - j0_vec[2],
            )
            checks.ampere_relative = relative_vector_residual(
                ampere_residual,
                (curl_b[0] / FOUR_PI, curl_b[1] / FOUR_PI, curl_b[2] / FOUR_PI),
                j0_vec,
            )

    if b0_vec is not None and field_data_exists(data_dir, "B"):
        with open_field_reader(data_dir / "B") as reader:
            if reader.num_steps() >= 2:
                b1_vec = trim_vector(read_vector(reader, "B", 1), params.cells)
                t0 = read_scalar(reader, "time", 0)
                t1 = read_scalar(reader, "time", 1)
                dt = t1 - t0
                if dt > 0.0:
                    measured = (
                        (b1_vec[0] - b0_vec[0]) / dt,
                        (b1_vec[1] - b0_vec[1]) / dt,
                        (b1_vec[2] - b0_vec[2]) / dt,
                    )
                    expected = analytic_induction_dt(params)
                    numerator = float(
                        np.mean(
                            measured[0] * expected[0]
                            + measured[1] * expected[1]
                            + measured[2] * expected[2]
                        )
                    )
                    expected_power = max(vector_rms(expected) ** 2, EPS)
                    measured_power = max(vector_rms(measured) ** 2, EPS)
                    checks.induction_scale = numerator / expected_power
                    checks.induction_correlation = numerator / math.sqrt(expected_power * measured_power)
                    scaled_expected = (
                        checks.induction_scale * expected[0],
                        checks.induction_scale * expected[1],
                        checks.induction_scale * expected[2],
                    )
                    checks.induction_relative_residual = relative_vector_error(measured, scaled_expected)

    for species_index in range(params.ns):
        j_name = f"J_{species_index}"
        rho_name = f"rho_{species_index}"
        if not field_data_exists(data_dir, j_name) or not field_data_exists(data_dir, rho_name):
            continue
        with open_field_reader(data_dir / j_name) as j_reader, open_field_reader(data_dir / rho_name) as rho_reader:
            j_vec = trim_vector(read_vector(j_reader, j_name, 0), params.cells)
            rho = trim_periodic_nodes(read_array(rho_reader, rho_name, 0), params.cells)
        velocity = velocity_from_moments(j_vec, rho)
        expected = expected_species_velocity(params, species_index)
        valid = np.isfinite(velocity[0]) & np.isfinite(velocity[1]) & np.isfinite(velocity[2])
        if np.any(valid):
            measured_valid = (velocity[0][valid], velocity[1][valid], velocity[2][valid])
            expected_valid = (expected[0][valid], expected[1][valid], expected[2][valid])
            checks.velocity_error_by_species[species_index] = relative_vector_error(
                measured_valid,
                expected_valid,
                floor=max(abs(params.u0), abs(params.b1) * max(abs(params.kx), abs(params.ky), abs(params.kz)), EPS),
            )
            checks.velocity_tgv_fraction_by_species[species_index] = vector_mode_fraction(
                velocity,
                grid,
            )

    return checks


@dataclass
class FieldHistory:
    step_index: np.ndarray
    time: np.ndarray
    b_fluct_energy: np.ndarray
    b_tgv_energy: np.ndarray
    b_weibel_band_energy: np.ndarray
    b_weibel_peak_kde: np.ndarray
    b_weibel_weighted_kde: np.ndarray
    b_shell_kde: np.ndarray
    b_shell_energy: np.ndarray
    divb_relative: np.ndarray
    gauss_relative: np.ndarray
    charge_mean: np.ndarray
    charge_rms: np.ndarray
    current_rms: np.ndarray
    current_max: np.ndarray


@dataclass
class PressureHistory:
    species_index: int
    step_index: np.ndarray
    time: np.ndarray
    valid_fraction: np.ndarray
    ae_rms: np.ndarray
    ae_max: np.ndarray
    ae_top_median: np.ndarray
    vperp_top_median: np.ndarray
    p2_over_p3_top_median: np.ndarray


@dataclass
class Spectra:
    b_first: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    b_last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    j_first: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    j_last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def selected_steps(n_steps: int, stride: int, max_steps: int | None) -> list[int]:
    steps = list(range(0, n_steps, max(stride, 1)))
    if max_steps is not None:
        steps = steps[:max_steps]
    return steps


def compute_field_history(
    params: TgvParams,
    field_stride: int,
    max_field_steps: int | None,
    weibel_min_kde: float,
    weibel_max_kde: float | None,
    max_spectrum_shell: int,
) -> tuple[FieldHistory | None, Spectra]:
    data_dir = params.data_dir
    available = {name: field_data_exists(data_dir, name) for name in ("B", "E", "rho", "J")}
    if not any(available.values()):
        return None, Spectra()

    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(open_field_reader(data_dir / name))
            for name, exists in available.items()
            if exists
        }
        primary = readers.get("B") or next(iter(readers.values()))
        steps = selected_steps(primary.num_steps(), field_stride, max_field_steps)
        if not steps:
            return None, Spectra()

        step_values: list[int] = []
        times: list[float] = []
        b_fluct_energy_values: list[float] = []
        b_tgv_energy_values: list[float] = []
        b_weibel_values: list[float] = []
        b_weibel_peak_kde_values: list[float] = []
        b_weibel_weighted_kde_values: list[float] = []
        b_shell_kde_values = np.full(max_spectrum_shell + 1, float("nan"), dtype=float)
        b_shell_energy_values: list[np.ndarray] = []
        divb_values: list[float] = []
        gauss_values: list[float] = []
        charge_mean_values: list[float] = []
        charge_rms_values: list[float] = []
        current_rms_values: list[float] = []
        current_max_values: list[float] = []

        grid = spectral_grid((params.nzc, params.nyc, params.nxc), params)
        spectra = Spectra()

        for output_index, step in enumerate(steps):
            step_values.append(step)
            times.append(read_scalar(primary, "time", step))

            b_vec = None
            if "B" in readers:
                b_vec = trim_vector(read_vector(readers["B"], "B", step), params.cells)
                spectral_energy = spectral_energy_density(b_vec, subtract_mean=True)
                b_fluct_energy_values.append(fluctuation_energy(b_vec))
                b_tgv_energy_values.append(tgv_mode_energy_from_density(spectral_energy, grid))
                band_total, peak_kde, weighted_kde = band_energy_and_k_stats_from_density(
                    spectral_energy,
                    grid,
                    weibel_min_kde,
                    weibel_max_kde,
                )
                b_weibel_values.append(band_total)
                b_weibel_peak_kde_values.append(peak_kde)
                b_weibel_weighted_kde_values.append(weighted_kde)
                shell_data = shell_spectrum_from_density(spectral_energy, grid, max_spectrum_shell)
                b_shell_kde_values = shell_data[1]
                b_shell_energy_values.append(shell_data[2])

                div_b = divergence(b_vec, params)
                b_ref = max(params.b0_magnitude, abs(params.b1), vector_rms(b_vec), EPS)
                k_ref = max(abs(params.kx), abs(params.ky), abs(params.kz), 1.0 / max(params.lx, params.ly, params.lz))
                divb_values.append(scalar_rms(div_b) / max(k_ref * b_ref, EPS))

                if output_index == 0:
                    spectra.b_first = shell_data
                if output_index == len(steps) - 1:
                    spectra.b_last = shell_data
            else:
                b_fluct_energy_values.append(float("nan"))
                b_tgv_energy_values.append(float("nan"))
                b_weibel_values.append(float("nan"))
                b_weibel_peak_kde_values.append(float("nan"))
                b_weibel_weighted_kde_values.append(float("nan"))
                b_shell_energy_values.append(np.full(max_spectrum_shell + 1, float("nan"), dtype=float))
                divb_values.append(float("nan"))

            j_vec_internal = (
                trim_vector(read_vector(readers["J"], "J", step), params.cells) if "J" in readers else None
            )

            if "E" in readers and "rho" in readers:
                e_vec = trim_vector(read_vector(readers["E"], "E", step), params.cells)
                rho = trim_periodic_nodes(read_array(readers["rho"], "rho", step), params.cells)
                gauss_values.append(gauss_residual(e_vec, rho, j_vec_internal, params))
                charge_mean_values.append(float(rho.mean()))
                charge_rms_values.append(scalar_rms(rho))
            else:
                gauss_values.append(float("nan"))
                charge_mean_values.append(float("nan"))
                charge_rms_values.append(float("nan"))

            if j_vec_internal is not None:
                j_vec_physical = (
                    FOUR_PI * j_vec_internal[0],
                    FOUR_PI * j_vec_internal[1],
                    FOUR_PI * j_vec_internal[2],
                )
                current_rms_values.append(vector_rms(j_vec_physical))
                current_max_values.append(vector_max(j_vec_physical))
                if output_index == 0:
                    spectra.j_first = shell_spectrum(j_vec_physical, grid, max_spectrum_shell)
                if output_index == len(steps) - 1:
                    spectra.j_last = shell_spectrum(j_vec_physical, grid, max_spectrum_shell)
            else:
                current_rms_values.append(float("nan"))
                current_max_values.append(float("nan"))

    history = FieldHistory(
        step_index=np.asarray(step_values, dtype=int),
        time=np.asarray(times, dtype=float),
        b_fluct_energy=np.asarray(b_fluct_energy_values, dtype=float),
        b_tgv_energy=np.asarray(b_tgv_energy_values, dtype=float),
        b_weibel_band_energy=np.asarray(b_weibel_values, dtype=float),
        b_weibel_peak_kde=np.asarray(b_weibel_peak_kde_values, dtype=float),
        b_weibel_weighted_kde=np.asarray(b_weibel_weighted_kde_values, dtype=float),
        b_shell_kde=np.asarray(b_shell_kde_values, dtype=float),
        b_shell_energy=np.asarray(b_shell_energy_values, dtype=float),
        divb_relative=np.asarray(divb_values, dtype=float),
        gauss_relative=np.asarray(gauss_values, dtype=float),
        charge_mean=np.asarray(charge_mean_values, dtype=float),
        charge_rms=np.asarray(charge_rms_values, dtype=float),
        current_rms=np.asarray(current_rms_values, dtype=float),
        current_max=np.asarray(current_max_values, dtype=float),
    )
    return history, spectra


def compute_pressure_history(
    params: TgvParams,
    field_stride: int,
    max_field_steps: int | None,
    pressure_top_fraction: float,
) -> PressureHistory | None:
    species_index = electron_species_index(params.qoms)
    if species_index is None or species_index >= params.ns:
        return None
    if not pressure_output_available(params, species_index):
        return None

    qom = params.qoms[species_index]
    data_dir = params.data_dir
    component_names = ("PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ")

    with ExitStack() as stack:
        rho_reader = stack.enter_context(open_field_reader(data_dir / f"rho_{species_index}"))
        j_reader = stack.enter_context(open_field_reader(data_dir / f"J_{species_index}"))
        p_readers = {
            component: stack.enter_context(open_field_reader(data_dir / f"{component}_{species_index}"))
            for component in component_names
        }
        steps = selected_steps(rho_reader.num_steps(), field_stride, max_field_steps)
        if not steps:
            return None

        step_values: list[int] = []
        times: list[float] = []
        valid_fraction_values: list[float] = []
        ae_rms_values: list[float] = []
        ae_max_values: list[float] = []
        ae_top_values: list[float] = []
        vperp_top_values: list[float] = []
        p2p3_top_values: list[float] = []

        for step in steps:
            step_values.append(step)
            times.append(read_scalar(rho_reader, "time", step))

            rho = trim_periodic_nodes(
                read_array(rho_reader, f"rho_{species_index}", step),
                params.cells,
            )
            j_vec = trim_vector(read_vector(j_reader, f"J_{species_index}", step), params.cells)
            velocity = velocity_from_moments(j_vec, rho)
            raw = tuple(
                trim_periodic_nodes(
                    read_array(p_readers[component], f"{component}_{species_index}", step),
                    params.cells,
                )
                for component in component_names
            )
            pressure_tensor, mass_density = pressure_tensor_from_raw_moments(raw, velocity, rho, qom)
            summary = pressure_anisotropy_summary(
                pressure_tensor,
                mass_density,
                pressure_top_fraction,
            )

            valid_fraction_values.append(summary["valid_fraction"])
            ae_rms_values.append(summary["ae_rms"])
            ae_max_values.append(summary["ae_max"])
            ae_top_values.append(summary["ae_top_median"])
            vperp_top_values.append(summary["vperp_top_median"])
            p2p3_top_values.append(summary["p2_over_p3_top_median"])

    return PressureHistory(
        species_index=species_index,
        step_index=np.asarray(step_values, dtype=int),
        time=np.asarray(times, dtype=float),
        valid_fraction=np.asarray(valid_fraction_values, dtype=float),
        ae_rms=np.asarray(ae_rms_values, dtype=float),
        ae_max=np.asarray(ae_max_values, dtype=float),
        ae_top_median=np.asarray(ae_top_values, dtype=float),
        vperp_top_median=np.asarray(vperp_top_values, dtype=float),
        p2_over_p3_top_median=np.asarray(p2p3_top_values, dtype=float),
    )


@dataclass
class WeibelFit:
    source: str
    shell_index: int
    k_de: float
    fit_start_pe: float
    fit_end_pe: float
    gamma_over_omega_pe: float
    gamma_over_omega_pi: float
    intercept: float
    r2: float
    sample_count: int


@dataclass
class WeibelTheoryComparison:
    k_de: float
    anisotropy: float
    vperp_over_c: float
    gamma_over_omega_pe: float
    gamma_over_omega_pi: float
    pic_to_theory_ratio: float
    pressure_sample_count: int


def compute_weibel_fit(
    history: FieldHistory | None,
    params: TgvParams,
    fit_start: float | None,
    fit_end: float | None,
    weibel_min_kde: float,
    weibel_max_kde: float | None,
    fit_source: str,
) -> WeibelFit | None:
    if history is None or history.time.size < 3 or not np.isfinite(params.omega_pe):
        return None
    time_pe = params.omega_pe * history.time
    start, end = automatic_fit_window(time_pe, history.b_weibel_band_energy)
    if fit_start is not None:
        start = fit_start
    if fit_end is not None:
        end = fit_end

    if fit_source == "fastest-shell" and history.b_shell_energy.ndim == 2:
        best: WeibelFit | None = None
        shell_count = min(history.b_shell_energy.shape[1], history.b_shell_kde.size)
        for shell_index in range(shell_count):
            k_de = float(history.b_shell_kde[shell_index])
            if not np.isfinite(k_de) or k_de <= 0.0 or k_de < weibel_min_kde:
                continue
            if weibel_max_kde is not None and k_de > weibel_max_kde:
                continue
            try:
                gamma_pe, intercept, r2, sample_count = fit_exponential_growth(
                    time_pe,
                    history.b_shell_energy[:, shell_index],
                    start,
                    end,
                )
            except ValueError:
                continue
            if not np.isfinite(gamma_pe):
                continue
            candidate = WeibelFit(
                source="fastest_shell",
                shell_index=shell_index,
                k_de=k_de,
                fit_start_pe=start,
                fit_end_pe=end,
                gamma_over_omega_pe=gamma_pe,
                gamma_over_omega_pi=gamma_pe * params.omega_pe,
                intercept=intercept,
                r2=r2,
                sample_count=sample_count,
            )
            if best is None or candidate.gamma_over_omega_pe > best.gamma_over_omega_pe:
                best = candidate
        if best is not None:
            return best

    gamma_pe, intercept, r2, sample_count = fit_exponential_growth(
        time_pe,
        history.b_weibel_band_energy,
        start,
        end,
    )
    return WeibelFit(
        source="band",
        shell_index=-1,
        k_de=representative_band_k_de(history, time_pe, start, end),
        fit_start_pe=start,
        fit_end_pe=end,
        gamma_over_omega_pe=gamma_pe,
        gamma_over_omega_pi=gamma_pe * params.omega_pe,
        intercept=intercept,
        r2=r2,
        sample_count=sample_count,
    )


def representative_band_k_de(
    history: FieldHistory | None,
    time_pe: np.ndarray,
    fit_start_pe: float,
    fit_end_pe: float,
) -> float:
    if history is None:
        return float("nan")
    keep = (
        np.isfinite(time_pe)
        & (time_pe >= fit_start_pe)
        & (time_pe <= fit_end_pe)
        & np.isfinite(history.b_weibel_weighted_kde)
        & (history.b_weibel_weighted_kde > 0.0)
    )
    if np.any(keep):
        return float(np.median(history.b_weibel_weighted_kde[keep]))
    keep = np.isfinite(history.b_weibel_peak_kde) & (history.b_weibel_peak_kde > 0.0)
    if np.any(keep):
        return float(np.median(history.b_weibel_peak_kde[keep]))
    return float("nan")


def representative_k_de(
    history: FieldHistory | None,
    params: TgvParams,
    weibel: WeibelFit | None,
    override_kde: float | None,
) -> float:
    if override_kde is not None:
        return override_kde
    if weibel is not None and np.isfinite(weibel.k_de) and weibel.k_de > 0.0:
        return weibel.k_de
    if history is None or weibel is None:
        return float("nan")
    time_pe = params.omega_pe * history.time
    return representative_band_k_de(history, time_pe, weibel.fit_start_pe, weibel.fit_end_pe)


def compute_weibel_theory_comparison(
    params: TgvParams,
    history: FieldHistory | None,
    pressure: PressureHistory | None,
    weibel: WeibelFit | None,
    theory_kde: float | None,
) -> WeibelTheoryComparison | None:
    if pressure is None or weibel is None:
        return None
    k_de = representative_k_de(history, params, weibel, theory_kde)
    if not np.isfinite(k_de) or k_de <= 0.0:
        return None

    time_pe = params.omega_pe * pressure.time
    keep = (
        np.isfinite(time_pe)
        & (time_pe >= weibel.fit_start_pe)
        & (time_pe <= weibel.fit_end_pe)
        & np.isfinite(pressure.ae_top_median)
        & np.isfinite(pressure.vperp_top_median)
        & (pressure.ae_top_median > 0.0)
        & (pressure.vperp_top_median > 0.0)
    )
    if np.count_nonzero(keep) == 0:
        return None

    anisotropy = float(np.median(pressure.ae_top_median[keep]))
    vperp = float(np.median(pressure.vperp_top_median[keep]))
    gamma_pe = kinetic_weibel_growth_rate(k_de, vperp, anisotropy)
    gamma_pi = gamma_pe * params.omega_pe if np.isfinite(gamma_pe) else float("nan")
    if np.isfinite(gamma_pe) and gamma_pe != 0.0:
        ratio = weibel.gamma_over_omega_pe / gamma_pe
    else:
        ratio = float("nan")
    return WeibelTheoryComparison(
        k_de=k_de,
        anisotropy=anisotropy,
        vperp_over_c=vperp,
        gamma_over_omega_pe=gamma_pe,
        gamma_over_omega_pi=gamma_pi,
        pic_to_theory_ratio=ratio,
        pressure_sample_count=int(np.count_nonzero(keep)),
    )


def write_timeseries_csv(path: Path, history: FieldHistory, pressure: PressureHistory | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pressure_by_step: dict[int, tuple[float, float, float, float, float, float]] = {}
    shell_count = (
        min(history.b_shell_energy.shape[1], history.b_shell_kde.size)
        if history.b_shell_energy.ndim == 2 and history.b_shell_energy.shape[0] == history.time.size
        else 0
    )
    if pressure is not None:
        for row in zip(
            pressure.step_index,
            pressure.valid_fraction,
            pressure.ae_rms,
            pressure.ae_max,
            pressure.ae_top_median,
            pressure.vperp_top_median,
            pressure.p2_over_p3_top_median,
        ):
            pressure_by_step[int(row[0])] = tuple(float(value) for value in row[1:])

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "field_step_index",
                "time_omega_pi",
                "B_fluct_energy",
                "B_tgv_mode_energy",
                "B_weibel_band_energy",
                "B_weibel_peak_kde",
                "B_weibel_weighted_kde",
                *[f"B_shell_{index:03d}_kde" for index in range(shell_count)],
                *[f"B_shell_{index:03d}_energy" for index in range(shell_count)],
                "divB_relative",
                "gauss_relative",
                "charge_mean",
                "charge_rms",
                "current_rms",
                "current_max",
                "electron_pressure_valid_fraction",
                "electron_Ae_rms",
                "electron_Ae_max",
                "electron_Ae_top_median",
                "electron_vperp_top_median",
                "electron_P2_over_P3_top_median",
            ]
        )
        for output_index, row in enumerate(
            zip(
                history.step_index,
                history.time,
                history.b_fluct_energy,
                history.b_tgv_energy,
                history.b_weibel_band_energy,
                history.b_weibel_peak_kde,
                history.b_weibel_weighted_kde,
                history.divb_relative,
                history.gauss_relative,
                history.charge_mean,
                history.charge_rms,
                history.current_rms,
                history.current_max,
            )
        ):
            p_row = pressure_by_step.get(
                int(row[0]),
                (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")),
            )
            shell_kde_row = tuple(float(value) for value in history.b_shell_kde[:shell_count])
            shell_energy_row = tuple(float(value) for value in history.b_shell_energy[output_index, :shell_count])
            writer.writerow(tuple(row[:7]) + shell_kde_row + shell_energy_row + tuple(row[7:]) + p_row)


def write_output_scan_csv(path: Path, scan: OutputScanHistory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cycle",
                "time_omega_pi",
                "electric_energy",
                "magnetic_energy",
                "kinetic_energy",
                "system_energy",
                "solver_iterations",
                "solver_residual",
            ]
        )
        writer.writerows(
            zip(
                scan.cycle,
                scan.time,
                scan.electric_energy,
                scan.magnetic_energy,
                scan.kinetic_energy,
                scan.system_energy,
                scan.solver_iterations,
                scan.solver_residual,
            )
        )


def csv_float(text: str | None) -> float:
    if text is None:
        return float("nan")
    stripped = text.strip()
    if not stripped:
        return float("nan")
    return float(stripped)


def read_timeseries_csv(path: Path) -> tuple[FieldHistory, PressureHistory | None]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"timeseries CSV is empty: {path}")

    columns = set(rows[0])

    def required(name: str) -> np.ndarray:
        if name not in columns:
            raise ValueError(f"timeseries CSV {path} is missing required column {name}")
        return np.asarray([csv_float(row.get(name)) for row in rows], dtype=float)

    def optional(name: str) -> np.ndarray:
        if name not in columns:
            return np.full(len(rows), float("nan"), dtype=float)
        return np.asarray([csv_float(row.get(name)) for row in rows], dtype=float)

    step_index = required("field_step_index").astype(int)
    time = required("time_omega_pi")
    shell_indices = sorted(
        int(match.group(1))
        for name in columns
        if (match := re.fullmatch(r"B_shell_(\d+)_energy", name)) is not None
    )
    if shell_indices:
        b_shell_kde = np.asarray(
            [
                finite_median(optional(f"B_shell_{index:03d}_kde"))
                for index in shell_indices
            ],
            dtype=float,
        )
        b_shell_energy = np.column_stack(
            [optional(f"B_shell_{index:03d}_energy") for index in shell_indices]
        )
    else:
        b_shell_kde = np.empty(0, dtype=float)
        b_shell_energy = np.empty((len(rows), 0), dtype=float)

    history = FieldHistory(
        step_index=step_index,
        time=time,
        b_fluct_energy=optional("B_fluct_energy"),
        b_tgv_energy=optional("B_tgv_mode_energy"),
        b_weibel_band_energy=required("B_weibel_band_energy"),
        b_weibel_peak_kde=optional("B_weibel_peak_kde"),
        b_weibel_weighted_kde=optional("B_weibel_weighted_kde"),
        b_shell_kde=b_shell_kde,
        b_shell_energy=b_shell_energy,
        divb_relative=optional("divB_relative"),
        gauss_relative=optional("gauss_relative"),
        charge_mean=optional("charge_mean"),
        charge_rms=optional("charge_rms"),
        current_rms=optional("current_rms"),
        current_max=optional("current_max"),
    )

    pressure_columns = [
        "electron_pressure_valid_fraction",
        "electron_Ae_rms",
        "electron_Ae_max",
        "electron_Ae_top_median",
        "electron_vperp_top_median",
        "electron_P2_over_P3_top_median",
    ]
    if not any(name in columns for name in pressure_columns):
        return history, None

    pressure = PressureHistory(
        species_index=-1,
        step_index=step_index,
        time=time,
        valid_fraction=optional("electron_pressure_valid_fraction"),
        ae_rms=optional("electron_Ae_rms"),
        ae_max=optional("electron_Ae_max"),
        ae_top_median=optional("electron_Ae_top_median"),
        vperp_top_median=optional("electron_vperp_top_median"),
        p2_over_p3_top_median=optional("electron_P2_over_P3_top_median"),
    )
    return history, pressure


def semilogy_positive(ax, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> None:
    keep = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.any(keep):
        ax.semilogy(x[keep], y[keep], *args, **kwargs)


def weibel_fit_energy(history: FieldHistory, weibel: WeibelFit | None) -> np.ndarray:
    if (
        weibel is not None
        and weibel.source == "fastest_shell"
        and 0 <= weibel.shell_index < history.b_shell_energy.shape[1]
    ):
        return history.b_shell_energy[:, weibel.shell_index]
    return history.b_weibel_band_energy


def weibel_fit_label(weibel: WeibelFit | None, weibel_min_kde: float, weibel_max_kde: float | None) -> str:
    if weibel is not None and weibel.source == "fastest_shell":
        return rf"fastest shell $k d_e={weibel.k_de:.3g}$"
    return rf"band $k d_e \in [{weibel_min_kde:g}, {weibel_max_kde if weibel_max_kde is not None else 'inf'}]$"


def plot_results(
    output_path: Path,
    params: TgvParams,
    diagnostics: Diagnostics | None,
    init: InitChecks,
    history: FieldHistory | None,
    pressure: PressureHistory | None,
    spectra: Spectra,
    weibel: WeibelFit | None,
    theory: WeibelTheoryComparison | None,
    weibel_min_kde: float,
    weibel_max_kde: float | None,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 12.0))

    ax = axes[0, 0]
    if diagnostics is not None:
        rel_total = (diagnostics.total_energy - diagnostics.total_energy[0]) / diagnostics.total_energy[0]
        ax.plot(diagnostics.time, rel_total, label="total energy rel. error")
        ax.set_xlabel(r"$\omega_{pi} t$")
        ax.set_ylabel("relative total-energy error")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax2 = ax.twinx()
        ax2.plot(diagnostics.time, diagnostics.field_energy, color="tab:orange", alpha=0.45, label="field")
        ax2.plot(diagnostics.time, diagnostics.kinetic_energy, color="tab:green", alpha=0.45, label="kinetic")
        ax2.set_ylabel("energy")
    else:
        ax.text(0.5, 0.5, "no diagnostics file", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[0, 1]
    if history is not None:
        semilogy_positive(ax, history.time, history.b_fluct_energy, "o-", label=r"$E_{B'}$")
        semilogy_positive(ax, history.time, history.b_tgv_energy, "s-", label="TGV Fourier modes")
        semilogy_positive(ax, history.time, history.b_weibel_band_energy, "^-", label="Weibel band")
        ax.set_xlabel(r"$\omega_{pi} t$")
        ax.set_ylabel("magnetic fluctuation energy density")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no field history", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[0, 2]
    if history is not None and weibel is not None:
        time_pe = params.omega_pe * history.time
        selected_energy = weibel_fit_energy(history, weibel)
        if weibel.source == "fastest_shell":
            semilogy_positive(
                ax,
                time_pe,
                history.b_weibel_band_energy,
                "o-",
                color="0.72",
                label="full Weibel band",
            )
        semilogy_positive(
            ax,
            time_pe,
            selected_energy,
            "s-",
            label=weibel_fit_label(weibel, weibel_min_kde, weibel_max_kde),
        )
        fit_time = np.linspace(time_pe.min(), time_pe.max(), 256)
        fit_energy = np.exp(weibel.intercept + 2.0 * weibel.gamma_over_omega_pe * fit_time)
        ax.semilogy(
            fit_time,
            fit_energy,
            "--",
            label=rf"fit $\gamma/\omega_{{pe}}={weibel.gamma_over_omega_pe:.3e}$",
        )
        if theory is not None and np.isfinite(theory.gamma_over_omega_pe):
            keep = (
                np.isfinite(time_pe)
                & np.isfinite(selected_energy)
                & (selected_energy > 0.0)
            )
            if np.any(keep):
                anchor_time = weibel.fit_start_pe
                anchor_log_energy = float(
                    np.interp(anchor_time, time_pe[keep], np.log(selected_energy[keep]))
                )
                theory_energy = np.exp(
                    anchor_log_energy + 2.0 * theory.gamma_over_omega_pe * (fit_time - anchor_time)
                )
                ax.semilogy(
                    fit_time,
                    theory_energy,
                    ":",
                    linewidth=2.0,
                    label=rf"theory $\gamma/\omega_{{pe}}={theory.gamma_over_omega_pe:.3e}$",
                )
        ax.axvspan(weibel.fit_start_pe, weibel.fit_end_pe, color="0.92", zorder=0)
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel("spectral magnetic energy")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no Weibel-band fit", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 0]
    if history is not None:
        semilogy_positive(ax, history.time, history.divb_relative, "o-", label=r"$\nabla\cdot B$")
        semilogy_positive(ax, history.time, history.gauss_relative, "s-", label=r"$\nabla\cdot E-\rho$")
        ax.axhline(max(init.divb_relative, EPS), color="tab:blue", linestyle=":", alpha=0.5)
        if np.isfinite(init.gauss_relative):
            ax.axhline(max(init.gauss_relative, EPS), color="tab:orange", linestyle=":", alpha=0.5)
        ax.set_xlabel(r"$\omega_{pi} t$")
        ax.set_ylabel("relative residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no residual history", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 1]
    if history is not None:
        semilogy_positive(ax, history.time, history.charge_rms, "o-", label="charge rms")
        semilogy_positive(ax, history.time, history.current_rms, "s-", label="current rms")
        semilogy_positive(ax, history.time, history.current_max, "^-", label="current max")
        ax.set_xlabel(r"$\omega_{pi} t$")
        ax.set_ylabel("moment magnitude")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no moment history", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 2]
    plotted = False
    for spectrum, style, label in (
        (spectra.b_first, "o-", "B first"),
        (spectra.b_last, "s-", "B last"),
        (spectra.j_first, "^-", "J first"),
        (spectra.j_last, "v-", "J last"),
    ):
        if spectrum is None:
            continue
        _, kde, energy = spectrum
        keep = np.isfinite(kde) & (kde > 0.0) & np.isfinite(energy) & (energy > 0.0)
        if np.any(keep):
            ax.loglog(kde[keep], energy[keep], style, label=label)
            plotted = True
    if plotted:
        ax.axvspan(weibel_min_kde, weibel_max_kde or ax.get_xlim()[1], color="0.93", zorder=0)
        if weibel is not None and np.isfinite(weibel.k_de) and weibel.k_de > 0.0:
            ax.axvline(weibel.k_de, color="tab:red", linestyle="--", alpha=0.7, label="fit shell")
        ax.set_xlabel(r"$k d_e$")
        ax.set_ylabel("shell energy")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no spectra", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2, 0]
    if pressure is not None:
        time_pe = params.omega_pe * pressure.time
        semilogy_positive(ax, time_pe, pressure.ae_rms, "o-", label=r"$A_{e,\rm rms}$")
        semilogy_positive(ax, time_pe, pressure.ae_top_median, "s-", label="top-region median")
        semilogy_positive(ax, time_pe, pressure.ae_max, "^-", label="max")
        if weibel is not None:
            ax.axvspan(weibel.fit_start_pe, weibel.fit_end_pe, color="0.92", zorder=0)
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel("electron pressure anisotropy")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no electron pressure", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2, 1]
    if pressure is not None:
        time_pe = params.omega_pe * pressure.time
        ax.plot(time_pe, pressure.vperp_top_median, "o-", label=r"top-region $v_\perp/c$")
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$v_\perp/c$")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(
            time_pe,
            pressure.p2_over_p3_top_median,
            "s-",
            color="tab:orange",
            alpha=0.65,
            label=r"top-region $P_2/P_3$",
        )
        ax2.set_ylabel(r"$P_2/P_3$")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.text(0.5, 0.5, "no pressure thermal speed", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2, 2]
    ax.set_axis_off()
    if theory is not None and weibel is not None:
        text = "\n".join(
            [
                "Weibel comparison",
                rf"fit source = {weibel.source.replace('_', ' ')}",
                rf"shell = {weibel.shell_index if weibel.shell_index >= 0 else 'n/a'}",
                rf"$k d_e = {theory.k_de:.4g}$",
                rf"$A_e = {theory.anisotropy:.4g}$",
                rf"$v_\perp/c = {theory.vperp_over_c:.4g}$",
                rf"$\gamma_{{PIC}}/\omega_{{pe}} = {weibel.gamma_over_omega_pe:.4g}$",
                rf"$\gamma_{{th}}/\omega_{{pe}} = {theory.gamma_over_omega_pe:.4g}$",
                rf"ratio = {theory.pic_to_theory_ratio:.4g}",
            ]
        )
    elif pressure is not None:
        text = "Weibel theory unavailable\n(no unstable pressure/k combination in fit window)"
    else:
        text = "Weibel theory unavailable\n(no reconstructed electron pressure)"
    ax.text(0.02, 0.98, text, ha="left", va="top", transform=ax.transAxes)

    fig.suptitle("Taylor-Green vortex validation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_light_csv_analysis(
    output_path: Path,
    params: TgvParams,
    history: FieldHistory,
    pressure: PressureHistory | None,
    weibel: WeibelFit | None,
    theory: WeibelTheoryComparison | None,
    weibel_min_kde: float,
    weibel_max_kde: float | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    time_pe = params.omega_pe * history.time

    ax = axes[0, 0]
    selected_energy = weibel_fit_energy(history, weibel)
    if weibel is not None and weibel.source == "fastest_shell":
        semilogy_positive(ax, time_pe, history.b_weibel_band_energy, "o-", color="0.72", label="full Weibel band")
    semilogy_positive(
        ax,
        time_pe,
        selected_energy,
        "s-",
        color="0.45",
        label=weibel_fit_label(weibel, weibel_min_kde, weibel_max_kde),
    )
    if weibel is not None:
        keep = (
            np.isfinite(time_pe)
            & np.isfinite(selected_energy)
            & (selected_energy > 0.0)
            & (time_pe >= weibel.fit_start_pe)
            & (time_pe <= weibel.fit_end_pe)
        )
        semilogy_positive(
            ax,
            time_pe[keep],
            selected_energy[keep],
            "o-",
            color="tab:blue",
            label="fit interval",
        )
        fit_time = np.linspace(weibel.fit_start_pe, weibel.fit_end_pe, 128)
        fit_energy = np.exp(weibel.intercept + 2.0 * weibel.gamma_over_omega_pe * fit_time)
        ax.semilogy(
            fit_time,
            fit_energy,
            "--",
            color="tab:red",
            label=rf"fit $\gamma/\omega_{{pe}}={weibel.gamma_over_omega_pe:.3e}$",
        )
        if theory is not None and np.isfinite(theory.gamma_over_omega_pe) and np.any(keep):
            anchor_time = weibel.fit_start_pe
            anchor_log_energy = float(
                np.interp(anchor_time, time_pe[keep], np.log(selected_energy[keep]))
            )
            theory_energy = np.exp(
                anchor_log_energy + 2.0 * theory.gamma_over_omega_pe * (fit_time - anchor_time)
            )
            ax.semilogy(
                fit_time,
                theory_energy,
                ":",
                linewidth=2.0,
                color="tab:purple",
                label=rf"theory $\gamma/\omega_{{pe}}={theory.gamma_over_omega_pe:.3e}$",
            )
        ax.axvspan(weibel.fit_start_pe, weibel.fit_end_pe, color="0.92", zorder=0)
    ax.set_xlabel(r"$\omega_{pe} t$")
    ax.set_ylabel("spectral magnetic energy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    if pressure is not None:
        pressure_time_pe = params.omega_pe * pressure.time
        semilogy_positive(ax, pressure_time_pe, pressure.ae_rms, "o-", label=r"$A_{e,\rm rms}$")
        semilogy_positive(ax, pressure_time_pe, pressure.ae_top_median, "s-", label="top-region median")
        if weibel is not None:
            ax.axvspan(weibel.fit_start_pe, weibel.fit_end_pe, color="0.92", zorder=0)
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel("electron pressure anisotropy")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no pressure columns in CSV", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 0]
    plotted = False
    keep_peak = np.isfinite(history.b_weibel_peak_kde) & (history.b_weibel_peak_kde > 0.0)
    keep_weighted = np.isfinite(history.b_weibel_weighted_kde) & (history.b_weibel_weighted_kde > 0.0)
    if np.any(keep_peak):
        ax.plot(time_pe[keep_peak], history.b_weibel_peak_kde[keep_peak], "o-", label="peak")
        plotted = True
    if np.any(keep_weighted):
        ax.plot(time_pe[keep_weighted], history.b_weibel_weighted_kde[keep_weighted], "s-", label="energy-weighted")
        plotted = True
    if plotted:
        if weibel is not None:
            ax.axvspan(weibel.fit_start_pe, weibel.fit_end_pe, color="0.92", zorder=0)
            if np.isfinite(weibel.k_de) and weibel.k_de > 0.0:
                ax.axhline(weibel.k_de, color="tab:red", linestyle="--", alpha=0.65, label="fit shell")
        ax.set_xlabel(r"$\omega_{pe} t$")
        ax.set_ylabel(r"$k d_e$")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no k columns in CSV", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 1]
    ax.set_axis_off()
    lines = [
        "CSV replay fit",
        rf"candidate band $k d_e$ = [{weibel_min_kde:g}, {weibel_max_kde if weibel_max_kde is not None else 'inf'}]",
    ]
    if weibel is not None:
        lines.extend(
            [
                f"fit source = {weibel.source}",
                f"shell index = {weibel.shell_index}" if weibel.shell_index >= 0 else "shell index = n/a",
                rf"fit $k d_e$ = {weibel.k_de:.4g}",
                rf"fit window $\omega_{{pe}} t$ = [{weibel.fit_start_pe:.4g}, {weibel.fit_end_pe:.4g}]",
                rf"samples = {weibel.sample_count}",
                rf"$\gamma/\omega_{{pe}}$ = {weibel.gamma_over_omega_pe:.4g}",
                rf"$\gamma/\omega_{{pi}}$ = {weibel.gamma_over_omega_pi:.4g}",
                rf"$R^2$ = {weibel.r2:.4g}",
            ]
        )
    else:
        lines.append("fit unavailable")
    if theory is not None:
        lines.extend(
            [
                "",
                "theory comparison",
                rf"$k d_e$ = {theory.k_de:.4g}",
                rf"$A_e$ = {theory.anisotropy:.4g}",
                rf"$v_\perp/c$ = {theory.vperp_over_c:.4g}",
                rf"$\gamma_{{th}}/\omega_{{pe}}$ = {theory.gamma_over_omega_pe:.4g}",
                rf"PIC/theory = {theory.pic_to_theory_ratio:.4g}",
            ]
        )
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", transform=ax.transAxes)

    fig.suptitle("Taylor-Green vortex CSV replay")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def finite_max_abs(arr: np.ndarray) -> float:
    keep = np.isfinite(arr)
    if not np.any(keep):
        return float("nan")
    return float(np.max(np.abs(arr[keep])))


def finite_median(arr: np.ndarray) -> float:
    keep = np.isfinite(arr)
    if not np.any(keep):
        return float("nan")
    return float(np.median(arr[keep]))


def summary_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g}"
    return str(value)


def analysis_summary_rows(
    params: TgvParams,
    diagnostics: Diagnostics | None,
    init: InitChecks | None,
    history: FieldHistory | None,
    pressure: PressureHistory | None,
    weibel: WeibelFit | None,
    theory: WeibelTheoryComparison | None,
    weibel_min_kde: float,
    weibel_max_kde: float | None,
    output_path: Path | None,
    timeseries_csv_path: Path | None,
    output_scan_csv_path: Path | None,
    output_log_path: Path | None,
    output_scan: OutputScanHistory | None,
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    def add(category: str, key: str, value) -> None:
        rows.append((category, key, summary_value(value)))

    dx = params.lx / params.nxc
    dy = params.ly / params.nyc
    dz = params.lz / params.nzc
    k_di = math.sqrt(params.kx * params.kx + params.ky * params.ky + params.kz * params.kz)
    k_de = k_di / params.omega_pe if params.omega_pe > 0.0 else float("nan")
    omega_ci_dt = params.b0_magnitude * params.dt

    add("files", "input_file", params.input_path)
    add("files", "data_directory", params.data_dir)
    add("files", "plot_file", output_path)
    add("files", "timeseries_csv", timeseries_csv_path)
    add("files", "summary_csv", "")
    add("files", "output_scan_csv", output_scan_csv_path)
    add("files", "output_log", output_log_path)

    add("parameters", "nxc", params.nxc)
    add("parameters", "nyc", params.nyc)
    add("parameters", "nzc", params.nzc)
    add("parameters", "lx", params.lx)
    add("parameters", "ly", params.ly)
    add("parameters", "lz", params.lz)
    add("parameters", "mode_x", params.mode_x)
    add("parameters", "mode_y", params.mode_y)
    add("parameters", "mode_z", params.mode_z)
    add("parameters", "k_TGV_di", k_di)
    add("parameters", "k_TGV_de", k_de)
    add("parameters", "U0", params.u0)
    add("parameters", "B0x", params.b0x)
    add("parameters", "B0y", params.b0y)
    add("parameters", "B0z", params.b0z)
    add("parameters", "B0_magnitude", params.b0_magnitude)
    add("parameters", "B1_TGV_amplitude", params.b1)
    add("parameters", "seed_motional_E", params.seed_motional_e)
    add("parameters", "mass_ratio_mu", params.mu)
    add("parameters", "omega_pe_over_omega_pi", params.omega_pe)
    add("parameters", "electron_skin_depth_de", params.de)
    add("parameters", "dx", dx)
    add("parameters", "dy", dy)
    add("parameters", "dz", dz)
    add("parameters", "max_cell_over_de", max(dx, dy, dz) / params.de if params.de > 0.0 else float("nan"))
    add("parameters", "dt", params.dt)
    add("parameters", "tau_TG", params.tau_tg)
    add("parameters", "dt_over_tau_TG", params.dt / params.tau_tg if math.isfinite(params.tau_tg) else float("nan"))
    add("parameters", "Omega_ci_dt_abs", omega_ci_dt)

    if diagnostics is not None and diagnostics.total_energy.size:
        rel_energy = (diagnostics.total_energy - diagnostics.total_energy[0]) / diagnostics.total_energy[0]
        add("diagnostics", "sample_count", diagnostics.time.size)
        add("diagnostics", "final_total_energy_rel_error", rel_energy[-1])
        add("diagnostics", "max_total_energy_rel_error", finite_max_abs(rel_energy))
    else:
        add("diagnostics", "sample_count", 0)

    if init is not None:
        add("initial_checks", "B_relative_error", init.b_relative_error)
        add("initial_checks", "E_relative_error", init.e_relative_error)
        add("initial_checks", "divB_relative", init.divb_relative)
        add("initial_checks", "Gauss_relative", init.gauss_relative)
        add("initial_checks", "charge_mean", init.charge_mean)
        add("initial_checks", "charge_rms", init.charge_rms)
        add("initial_checks", "physical_J_rms", init.current_rms)
        add("initial_checks", "Ampere_relative", init.ampere_relative)
        add("initial_checks", "early_induction_scale", init.induction_scale)
        add("initial_checks", "early_induction_correlation", init.induction_correlation)
        add("initial_checks", "early_induction_relative_residual", init.induction_relative_residual)
        if init.velocity_error_by_species:
            for species_index, error in init.velocity_error_by_species.items():
                add("initial_checks", f"species_{species_index}_bulk_u_relative_error", error)
                fraction = init.velocity_tgv_fraction_by_species.get(species_index, float("nan"))
                add("initial_checks", f"species_{species_index}_TGV_mode_fraction", fraction)

    if history is not None and history.time.size:
        add("field_history", "sample_count", history.time.size)
        add("field_history", "magnetic_shell_count", history.b_shell_kde.size)
        add("field_history", "final_B_fluct_energy", history.b_fluct_energy[-1])
        add("field_history", "final_B_TGV_mode_energy", history.b_tgv_energy[-1])
        add("field_history", "final_B_Weibel_band_energy", history.b_weibel_band_energy[-1])
        add("field_history", "median_Weibel_weighted_k_de", finite_median(history.b_weibel_weighted_kde))
        add("field_history", "max_divB_relative", finite_max_abs(history.divb_relative))
        add("field_history", "max_Gauss_relative", finite_max_abs(history.gauss_relative))
        add("field_history", "max_charge_rms", finite_max_abs(history.charge_rms))
        add("field_history", "max_physical_J_rms", finite_max_abs(history.current_rms))

    if pressure is not None and pressure.time.size:
        add("pressure", "electron_species_index", pressure.species_index)
        add("pressure", "sample_count", pressure.time.size)
        add("pressure", "final_Ae_rms", pressure.ae_rms[-1])
        add("pressure", "max_Ae_rms", finite_max_abs(pressure.ae_rms))
        add("pressure", "max_Ae", finite_max_abs(pressure.ae_max))
        add("pressure", "median_top_Ae", finite_median(pressure.ae_top_median))
        add("pressure", "median_top_vperp_over_c", finite_median(pressure.vperp_top_median))
        add("pressure", "median_top_P2_over_P3", finite_median(pressure.p2_over_p3_top_median))

    add("weibel_fit", "candidate_k_de_min", weibel_min_kde)
    add("weibel_fit", "candidate_k_de_max", weibel_max_kde if weibel_max_kde is not None else "inf")
    if weibel is not None:
        add("weibel_fit", "source", weibel.source)
        add("weibel_fit", "shell_index", weibel.shell_index)
        add("weibel_fit", "k_de", weibel.k_de)
        add("weibel_fit", "fit_start_omega_pe_t", weibel.fit_start_pe)
        add("weibel_fit", "fit_end_omega_pe_t", weibel.fit_end_pe)
        add("weibel_fit", "sample_count", weibel.sample_count)
        add("weibel_fit", "gamma_over_omega_pe", weibel.gamma_over_omega_pe)
        add("weibel_fit", "gamma_over_omega_pi", weibel.gamma_over_omega_pi)
        add("weibel_fit", "intercept_log_energy", weibel.intercept)
        add("weibel_fit", "r2", weibel.r2)

    if theory is not None:
        add("weibel_theory", "k_de", theory.k_de)
        add("weibel_theory", "A_e", theory.anisotropy)
        add("weibel_theory", "vperp_over_c", theory.vperp_over_c)
        add("weibel_theory", "sample_count", theory.pressure_sample_count)
        add("weibel_theory", "gamma_over_omega_pe", theory.gamma_over_omega_pe)
        add("weibel_theory", "gamma_over_omega_pi", theory.gamma_over_omega_pi)
        add("weibel_theory", "PIC_to_theory_gamma_ratio", theory.pic_to_theory_ratio)

    if output_scan is not None and output_scan.cycle.size:
        add("output_scan", "sample_count", output_scan.cycle.size)
        add("output_scan", "first_cycle", int(output_scan.cycle[0]))
        add("output_scan", "last_cycle", int(output_scan.cycle[-1]))
        add("output_scan", "final_system_energy", output_scan.system_energy[-1])
        add("output_scan", "max_solver_iterations", finite_max_abs(output_scan.solver_iterations))
        add("output_scan", "max_solver_residual", finite_max_abs(output_scan.solver_residual))

    return rows


def write_summary_csv(
    path: Path,
    params: TgvParams,
    diagnostics: Diagnostics | None,
    init: InitChecks | None,
    history: FieldHistory | None,
    pressure: PressureHistory | None,
    weibel: WeibelFit | None,
    theory: WeibelTheoryComparison | None,
    weibel_min_kde: float,
    weibel_max_kde: float | None,
    output_path: Path | None,
    timeseries_csv_path: Path | None,
    output_scan_csv_path: Path | None,
    output_log_path: Path | None,
    output_scan: OutputScanHistory | None,
) -> None:
    rows = analysis_summary_rows(
        params=params,
        diagnostics=diagnostics,
        init=init,
        history=history,
        pressure=pressure,
        weibel=weibel,
        theory=theory,
        weibel_min_kde=weibel_min_kde,
        weibel_max_kde=weibel_max_kde,
        output_path=output_path,
        timeseries_csv_path=timeseries_csv_path,
        output_scan_csv_path=output_scan_csv_path,
        output_log_path=output_log_path,
        output_scan=output_scan,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "key", "value"])
        for category, key, value in rows:
            if category == "files" and key == "summary_csv":
                value = str(path)
            writer.writerow([category, key, value])


def print_summary(
    params: TgvParams,
    diagnostics: Diagnostics | None,
    init: InitChecks,
    history: FieldHistory | None,
    pressure: PressureHistory | None,
    weibel: WeibelFit | None,
    theory: WeibelTheoryComparison | None,
    weibel_min_kde: float,
    weibel_max_kde: float | None,
    output_path: Path,
    csv_path: Path | None,
    summary_csv_path: Path | None,
    output_scan_csv_path: Path | None,
    output_log_path: Path | None,
    output_scan_attempted: bool,
) -> None:
    dx = params.lx / params.nxc
    dy = params.ly / params.nyc
    dz = params.lz / params.nzc
    k_di = math.sqrt(params.kx * params.kx + params.ky * params.ky + params.kz * params.kz)
    k_de = k_di / params.omega_pe if params.omega_pe > 0.0 else float("nan")
    omega_ci_dt = params.b0_magnitude * params.dt

    print(f"input file                  = {params.input_path}")
    print(f"data directory              = {params.data_dir}")
    print(f"grid cells                  = {params.nxc} x {params.nyc} x {params.nzc}")
    print(f"domain                      = ({params.lx:.6g}, {params.ly:.6g}, {params.lz:.6g})")
    print(f"TGV modes                   = ({params.mode_x}, {params.mode_y}, {params.mode_z})")
    print(f"k_TGV d_i                   = {k_di:.9g}")
    print(f"k_TGV d_e                   = {k_de:.9g}")
    print(f"U0                          = {params.u0:.9g}")
    print(f"B0 vector                   = ({params.b0x:.9g}, {params.b0y:.9g}, {params.b0z:.9g})")
    print(f"B1 magnetic TGV amplitude   = {params.b1:.9g}")
    print(f"seed motional E             = {params.seed_motional_e}")
    print(f"mass ratio mu               = {params.mu:.9g}")
    print(f"electron skin depth d_e     = {params.de:.9g}")
    print(f"cell sizes                  = ({dx:.9g}, {dy:.9g}, {dz:.9g})")
    if params.de > 0.0:
        print(f"max cell / d_e              = {max(dx, dy, dz) / params.de:.9g}")
    print(f"dt                          = {params.dt:.9g}")
    print(f"eddy time tau_TG            = {params.tau_tg:.9g}")
    if math.isfinite(params.tau_tg):
        print(f"dt / tau_TG                 = {params.dt / params.tau_tg:.9g}")
    print(f"|Omega_ci| dt               = {omega_ci_dt:.9g}")

    if diagnostics is not None:
        rel_energy = (diagnostics.total_energy - diagnostics.total_energy[0]) / diagnostics.total_energy[0]
        print(f"diagnostic samples          = {diagnostics.time.size}")
        print(f"final total-energy rel err  = {rel_energy[-1]:.9e}")
        print(f"max total-energy rel err    = {finite_max_abs(rel_energy):.9e}")
    else:
        print("diagnostic samples          = unavailable")

    print(f"initial B relative error    = {init.b_relative_error:.9e}")
    print(f"initial E relative error    = {init.e_relative_error:.9e}")
    print(f"initial divB relative       = {init.divb_relative:.9e}")
    print(f"initial Gauss relative      = {init.gauss_relative:.9e}")
    print(f"initial charge mean         = {init.charge_mean:.9e}")
    print(f"initial charge rms          = {init.charge_rms:.9e}")
    print(f"initial physical J rms      = {init.current_rms:.9e}")
    print(f"initial Ampere relative     = {init.ampere_relative:.9e}")
    print(f"early induction scale       = {init.induction_scale:.9e}")
    print(f"early induction correlation = {init.induction_correlation:.9e}")
    print(f"early induction rel resid   = {init.induction_relative_residual:.9e}")

    if init.velocity_error_by_species:
        for species_index, error in init.velocity_error_by_species.items():
            fraction = init.velocity_tgv_fraction_by_species.get(species_index, float("nan"))
            print(f"species {species_index} bulk-u rel err = {error:.9e}")
            print(f"species {species_index} TGV mode frac  = {fraction:.9e}")
    else:
        print("species bulk-u checks       = unavailable")

    if history is not None:
        print(f"field output samples used   = {history.time.size}")
        print(f"magnetic shell samples      = {history.b_shell_kde.size}")
        print(f"final B fluct energy        = {history.b_fluct_energy[-1]:.9e}")
        print(f"final B TGV-mode energy     = {history.b_tgv_energy[-1]:.9e}")
        print(f"final B Weibel-band energy  = {history.b_weibel_band_energy[-1]:.9e}")
        print(f"median Weibel weighted k d_e= {finite_median(history.b_weibel_weighted_kde):.9e}")
        print(f"max divB relative           = {finite_max_abs(history.divb_relative):.9e}")
        print(f"max Gauss relative          = {finite_max_abs(history.gauss_relative):.9e}")
        print(f"max charge rms              = {finite_max_abs(history.charge_rms):.9e}")
        print(f"max physical J rms          = {finite_max_abs(history.current_rms):.9e}")

    if pressure is not None:
        print(f"electron pressure species   = {pressure.species_index}")
        print(f"pressure samples used       = {pressure.time.size}")
        print(f"final electron A_e,rms      = {pressure.ae_rms[-1]:.9e}")
        print(f"max electron A_e,rms        = {finite_max_abs(pressure.ae_rms):.9e}")
        print(f"max electron A_e            = {finite_max_abs(pressure.ae_max):.9e}")
        print(f"median top electron A_e     = {finite_median(pressure.ae_top_median):.9e}")
        print(f"median top v_perp/c         = {finite_median(pressure.vperp_top_median):.9e}")
        print(f"median top P2/P3            = {finite_median(pressure.p2_over_p3_top_median):.9e}")
    else:
        print("electron pressure analysis  = unavailable")

    print(f"Weibel candidate k d_e min  = {weibel_min_kde:.9g}")
    print(f"Weibel candidate k d_e max  = {weibel_max_kde if weibel_max_kde is not None else 'inf'}")
    if weibel is not None:
        print(f"Weibel fit source           = {weibel.source}")
        print(f"Weibel fit shell index      = {weibel.shell_index if weibel.shell_index >= 0 else 'n/a'}")
        print(f"Weibel fit k d_e            = {weibel.k_de:.9e}")
        print(f"Weibel fit window omega_pe t= [{weibel.fit_start_pe:.9g}, {weibel.fit_end_pe:.9g}]")
        print(f"Weibel fit samples          = {weibel.sample_count}")
        print(f"Weibel gamma / omega_pe     = {weibel.gamma_over_omega_pe:.9e}")
        print(f"Weibel gamma / omega_pi     = {weibel.gamma_over_omega_pi:.9e}")
        print(f"Weibel fit R^2              = {weibel.r2:.9e}")
        if weibel.source == "band":
            print("warning                     = fastest-shell data unavailable; using broad-band growth fit")
        if weibel.r2 < 0.9:
            print("warning                     = selected Weibel spectral energy is not very exponential in this window")
        if weibel.gamma_over_omega_pe <= 0.0:
            print("warning                     = fitted Weibel growth rate is non-positive")
    else:
        print("Weibel fit                  = unavailable")

    if theory is not None:
        print(f"Weibel theory k d_e         = {theory.k_de:.9e}")
        print(f"Weibel theory A_e           = {theory.anisotropy:.9e}")
        print(f"Weibel theory v_perp/c      = {theory.vperp_over_c:.9e}")
        print(f"Weibel theory samples       = {theory.pressure_sample_count}")
        print(f"Weibel theory gamma/omega_pe= {theory.gamma_over_omega_pe:.9e}")
        print(f"Weibel theory gamma/omega_pi= {theory.gamma_over_omega_pi:.9e}")
        print(f"PIC/theory gamma ratio      = {theory.pic_to_theory_ratio:.9e}")
        print(f"omega_pe / omega_pi         = {params.omega_pe:.9e}")
    elif pressure is None:
        print("Weibel theory               = unavailable; need rho_s, J_s, and PXX/PXY/PXZ/PYY/PYZ/PZZ for electrons")
    else:
        print("Weibel theory               = unavailable; no unstable pressure/k estimate in the fit window")

    if params.b0_magnitude > 0.0:
        print("warning                     = nonzero guide field; Weibel theory comparison is only a formal local check")

    print(f"plot written to             = {output_path}")
    if csv_path is not None:
        print(f"timeseries csv written to   = {csv_path}")
    if summary_csv_path is not None:
        print(f"summary csv written to      = {summary_csv_path}")
    if output_scan_csv_path is not None:
        print(f"output scan csv written to  = {output_scan_csv_path}")
    elif output_scan_attempted and output_log_path is None:
        print("output scan csv written to  = unavailable; no output log found")
    elif output_scan_attempted:
        print("output scan csv written to  = unavailable; no parseable output-log cycles")


def print_csv_replay_summary(
    params: TgvParams,
    csv_input_path: Path,
    history: FieldHistory,
    pressure: PressureHistory | None,
    weibel: WeibelFit | None,
    theory: WeibelTheoryComparison | None,
    output_path: Path,
    summary_csv_path: Path | None,
) -> None:
    print(f"input file                  = {params.input_path}")
    print(f"precomputed timeseries csv  = {csv_input_path}")
    print(f"field samples loaded        = {history.time.size}")
    print(f"magnetic shells loaded      = {history.b_shell_kde.size}")
    if pressure is not None:
        print(f"pressure samples loaded     = {pressure.time.size}")
    else:
        print("pressure samples loaded     = unavailable")
    if weibel is not None:
        print(f"Weibel fit source           = {weibel.source}")
        print(f"Weibel fit shell index      = {weibel.shell_index if weibel.shell_index >= 0 else 'n/a'}")
        print(f"Weibel fit k d_e            = {weibel.k_de:.9e}")
        print(f"Weibel fit window omega_pe t= [{weibel.fit_start_pe:.9g}, {weibel.fit_end_pe:.9g}]")
        print(f"Weibel fit samples          = {weibel.sample_count}")
        print(f"Weibel gamma / omega_pe     = {weibel.gamma_over_omega_pe:.9e}")
        print(f"Weibel gamma / omega_pi     = {weibel.gamma_over_omega_pi:.9e}")
        print(f"Weibel fit R^2              = {weibel.r2:.9e}")
        if weibel.source == "band":
            print("warning                     = fastest-shell data unavailable; using broad-band growth fit")
        if weibel.r2 < 0.9:
            print("warning                     = selected Weibel spectral energy is not very exponential in this window")
    else:
        print("Weibel fit                  = unavailable")
    if theory is not None:
        print(f"Weibel theory k d_e         = {theory.k_de:.9e}")
        print(f"Weibel theory A_e           = {theory.anisotropy:.9e}")
        print(f"Weibel theory v_perp/c      = {theory.vperp_over_c:.9e}")
        print(f"Weibel theory gamma/omega_pe= {theory.gamma_over_omega_pe:.9e}")
        print(f"PIC/theory gamma ratio      = {theory.pic_to_theory_ratio:.9e}")
    print(f"plot written to             = {output_path}")
    if summary_csv_path is not None:
        print(f"summary csv written to      = {summary_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="reload a precomputed timeseries CSV and only redo the fit/plot",
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--no-summary-csv", action="store_true")
    parser.add_argument("--out-log", type=Path, default=None, help="stdout/stderr log to scan")
    parser.add_argument("--output-scan-csv", type=Path, default=None)
    parser.add_argument("--no-output-scan", action="store_true")
    parser.add_argument("--field-stride", type=int, default=1)
    parser.add_argument("--max-field-steps", type=int, default=None)
    parser.add_argument("--max-spectrum-shell", type=int, default=48)
    parser.add_argument("--weibel-min-kde", type=float, default=0.25)
    parser.add_argument("--weibel-max-kde", type=float, default=None)
    parser.add_argument("--weibel-fit-start", type=float, default=None, help="fit start in omega_pe t")
    parser.add_argument("--weibel-fit-end", type=float, default=None, help="fit end in omega_pe t")
    parser.add_argument(
        "--weibel-fit-source",
        choices=("fastest-shell", "band"),
        default="fastest-shell",
        help="fit the fastest-growing magnetic shell, or fall back to the broad band diagnostic",
    )
    parser.add_argument("--weibel-theory-kde", type=float, default=None, help="override k d_e for theory comparison")
    parser.add_argument(
        "--pressure-top-fraction",
        type=float,
        default=0.05,
        help="fraction of highest-A_e cells used for representative pressure/theory values",
    )
    args = parser.parse_args()

    input_path = find_input_path(args.run_dir, args.input)
    params = load_params(input_path, args.run_dir, args.data_dir)

    if args.from_csv is not None:
        history, pressure = read_timeseries_csv(args.from_csv)
        try:
            weibel = compute_weibel_fit(
                history,
                params,
                fit_start=args.weibel_fit_start,
                fit_end=args.weibel_fit_end,
                weibel_min_kde=args.weibel_min_kde,
                weibel_max_kde=args.weibel_max_kde,
                fit_source=args.weibel_fit_source,
            )
        except ValueError as exc:
            print(f"warning                     = {exc}")
            weibel = None
        theory = compute_weibel_theory_comparison(
            params=params,
            history=history,
            pressure=pressure,
            weibel=weibel,
            theory_kde=args.weibel_theory_kde,
        )
        output_path = args.output or args.from_csv.with_name(args.from_csv.stem + "_fit.png")
        summary_csv_path = None
        if not args.no_summary_csv:
            summary_csv_path = args.summary_csv or output_path.with_name(output_path.stem + "_summary.csv")
            write_summary_csv(
                path=summary_csv_path,
                params=params,
                diagnostics=None,
                init=None,
                history=history,
                pressure=pressure,
                weibel=weibel,
                theory=theory,
                weibel_min_kde=args.weibel_min_kde,
                weibel_max_kde=args.weibel_max_kde,
                output_path=output_path,
                timeseries_csv_path=args.from_csv,
                output_scan_csv_path=None,
                output_log_path=None,
                output_scan=None,
            )
        plot_light_csv_analysis(
            output_path=output_path,
            params=params,
            history=history,
            pressure=pressure,
            weibel=weibel,
            theory=theory,
            weibel_min_kde=args.weibel_min_kde,
            weibel_max_kde=args.weibel_max_kde,
        )
        print_csv_replay_summary(
            params=params,
            csv_input_path=args.from_csv,
            history=history,
            pressure=pressure,
            weibel=weibel,
            theory=theory,
            output_path=output_path,
            summary_csv_path=summary_csv_path,
        )
        return

    diagnostics = read_diagnostics(params.data_dir / "diagnostics.csv")
    init_checks = compute_initial_checks(params)
    history, spectra = compute_field_history(
        params=params,
        field_stride=args.field_stride,
        max_field_steps=args.max_field_steps,
        weibel_min_kde=args.weibel_min_kde,
        weibel_max_kde=args.weibel_max_kde,
        max_spectrum_shell=args.max_spectrum_shell,
    )
    pressure = compute_pressure_history(
        params=params,
        field_stride=args.field_stride,
        max_field_steps=args.max_field_steps,
        pressure_top_fraction=args.pressure_top_fraction,
    )
    try:
        weibel = compute_weibel_fit(
            history,
            params,
            fit_start=args.weibel_fit_start,
            fit_end=args.weibel_fit_end,
            weibel_min_kde=args.weibel_min_kde,
            weibel_max_kde=args.weibel_max_kde,
            fit_source=args.weibel_fit_source,
        )
    except ValueError as exc:
        print(f"warning                     = {exc}")
        weibel = None
    theory = compute_weibel_theory_comparison(
        params=params,
        history=history,
        pressure=pressure,
        weibel=weibel,
        theory_kde=args.weibel_theory_kde,
    )

    output_path = args.output or (params.run_dir / "tgv_analysis.png")
    csv_path = None if args.no_csv else (args.csv or output_path.with_suffix(".csv"))
    if history is not None and csv_path is not None:
        write_timeseries_csv(csv_path, history, pressure)

    output_log_path = None
    output_scan = None
    output_scan_csv_path = None
    if not args.no_output_scan:
        output_log_path = find_output_log(params.run_dir, args.out_log)
        if output_log_path is not None:
            output_scan = scan_output_log(output_log_path)
            if output_scan is not None:
                output_scan_csv_path = args.output_scan_csv or output_path.with_name(
                    output_path.stem + "_output_scan.csv"
                )
                write_output_scan_csv(output_scan_csv_path, output_scan)

    summary_csv_path = None
    if not args.no_summary_csv:
        summary_csv_path = args.summary_csv or output_path.with_name(output_path.stem + "_summary.csv")
        write_summary_csv(
            path=summary_csv_path,
            params=params,
            diagnostics=diagnostics,
            init=init_checks,
            history=history,
            pressure=pressure,
            weibel=weibel,
            theory=theory,
            weibel_min_kde=args.weibel_min_kde,
            weibel_max_kde=args.weibel_max_kde,
            output_path=output_path,
            timeseries_csv_path=csv_path if history is not None else None,
            output_scan_csv_path=output_scan_csv_path,
            output_log_path=output_log_path,
            output_scan=output_scan,
        )

    plot_results(
        output_path=output_path,
        params=params,
        diagnostics=diagnostics,
        init=init_checks,
        history=history,
        pressure=pressure,
        spectra=spectra,
        weibel=weibel,
        theory=theory,
        weibel_min_kde=args.weibel_min_kde,
        weibel_max_kde=args.weibel_max_kde,
    )
    print_summary(
        params=params,
        diagnostics=diagnostics,
        init=init_checks,
        history=history,
        pressure=pressure,
        weibel=weibel,
        theory=theory,
        weibel_min_kde=args.weibel_min_kde,
        weibel_max_kde=args.weibel_max_kde,
        output_path=output_path,
        csv_path=csv_path if history is not None else None,
        summary_csv_path=summary_csv_path,
        output_scan_csv_path=output_scan_csv_path,
        output_log_path=output_log_path,
        output_scan_attempted=not args.no_output_scan,
    )


if __name__ == "__main__":
    main()
