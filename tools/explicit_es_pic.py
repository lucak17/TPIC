#!/usr/bin/env python3
"""Explicit electrostatic PIC run of a quasi-1D TPIC deck, started from the particles of the TPIC run.

The same particles that TPIC loaded move under the standard momentum-conserving explicit electrostatic scheme
instead of ECSIM; ``tools/postprocessing/analyze_explicit_comparison.py`` compares the two runs.  Usage from the
repository root::

    python3 tools/explicit_es_pic.py --run-dir RUN_DIR [--input DECK] [--cycles N]

The run directory holds the deck (``RUN_DIR/twoStream1D.inp`` unless ``--input`` names another) and the TPIC run
launched from it: its ``SaveDirName`` must hold the particle snapshot of cycle 0 (``ParticlesOutputCycle`` > 0).  The
deck sets the box, the cells along x, the time step, the number of cycles (``ncycles`` unless ``--cycles``) and the
charge-to-mass ratios.  Nothing initialized may depend on y or z, so that a 1D code moves the same particles.

Particles.  The x, vx and charge of every particle of the snapshot.  TPIC charges hold 1/(4*pi) and the cell volume;
times 4*pi/(Ly*Lz) they are charges per unit area with rho = sum q/dx including 4*pi, so that the energies are those
of TPIC up to one constant factor.  The masses are q/(q/m).

Scheme (Birdsall and Langdon, ES1): linear (cloud-in-cell) weighting for the charge and the field, the three-point
Poisson equation -d2(phi)/dx2 = rho solved with FFTs, the centred field E_j = -(phi_{j+1} - phi_{j-1})/(2 dx), and the
leapfrog push with v at half steps, started half a step back from the Poisson field of the loaded charge.  The energy
of step n is the field energy (1/2) sum_j rho_j phi_j dx plus the kinetic energy with the mean of the squared
velocities before and after the step.

Writes ``RUN_DIR/explicit_pic_energy.csv`` (every step: omega_pe*t, the field and kinetic energy, and per species
the kinetic energy, x momentum and mass) and ``RUN_DIR/explicit_pic_particles.npz`` (``x_<species>_<cycle>`` and
``v_<species>_<cycle>`` at every ``ParticlesOutputCycle``, the velocity at the step, the mean of its two half steps).
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from tpic_tools.inputs import read_input
from tpic_tools.particles import read_polydata


def value(params: dict, key: str, index: int = 0) -> float:
    return float(params[key][index])


def initial_particles(params: dict, data: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return x, vx and the charge per unit area of every species at the TPIC snapshot of cycle 0."""
    pieces = sorted(data.glob("particles_t000000_r*.vtp"))
    if not pieces:
        raise FileNotFoundError(f"{data}: no particle snapshot of cycle 0 (set ParticlesOutputCycle > 0)")
    arrays = [read_polydata(piece, ("position", "velocity", "charge", "species")) for piece in pieces]
    to_area = 4.0 * math.pi / (value(params, "Ly") * value(params, "Lz"))
    result = []
    for s in range(int(value(params, "ns"))):
        chosen = [a["species"] == s for a in arrays]
        result.append((np.concatenate([a["position"].reshape(-1, 3)[c, 0] for a, c in zip(arrays, chosen)]),
                       np.concatenate([a["velocity"].reshape(-1, 3)[c, 0] for a, c in zip(arrays, chosen)]),
                       to_area * np.concatenate([a["charge"][c] for a, c in zip(arrays, chosen)])))
    return result


def simulate(params: dict, cycles: int, initial: list) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the energy history and the particle snapshots of the explicit run."""
    nx, length, dt = int(value(params, "nxc")), value(params, "Lx"), value(params, "dt")
    dx = length / nx
    qom = [value(params, "qom", s) for s in range(len(initial))]
    species = [{"x": x.copy(), "v": v.copy(), "q": q, "m": q / qom[s], "qom": qom[s]}
               for s, (x, v, q) in enumerate(initial)]
    wavenumbers = 2.0 * math.pi * np.fft.fftfreq(nx, dx)
    operator = (2.0 / dx * np.sin(0.5 * wavenumbers * dx)) ** 2
    operator[0] = 1.0

    def weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = x / dx
        j = np.floor(g).astype(np.int64)
        return j % nx, (j + 1) % nx, g - j

    def field() -> tuple[np.ndarray, float]:
        rho = np.zeros(nx)
        for sp in species:
            left, right, w = weights(sp["x"])
            rho += (np.bincount(left, sp["q"] * (1.0 - w), nx) + np.bincount(right, sp["q"] * w, nx)) / dx
        spectrum = np.fft.fft(rho) / operator
        spectrum[0] = 0.0
        phi = np.fft.ifft(spectrum).real
        return -(np.roll(phi, -1) - np.roll(phi, 1)) / (2.0 * dx), 0.5 * float(np.dot(rho, phi)) * dx

    def gather(e: np.ndarray, x: np.ndarray) -> np.ndarray:
        left, right, w = weights(x)
        return e[left] * (1.0 - w) + e[right] * w

    e, _ = field()
    for sp in species:
        sp["v"] = sp["v"] - 0.5 * dt * sp["qom"] * gather(e, sp["x"])
    names = ["field", "kinetic"] + [f"{name}_{s}" for s in range(len(species))
                                    for name in ("kinetic", "momentum", "mass")]
    history = {name: np.empty(cycles + 1) for name in names}
    cadence, snapshots = int(value(params, "ParticlesOutputCycle")), {}
    for n in range(cycles + 1):
        e, history["field"][n] = field()
        history["kinetic"][n] = 0.0
        for s, sp in enumerate(species):
            before = sp["v"]
            after = before + dt * sp["qom"] * gather(e, sp["x"])
            history[f"kinetic_{s}"][n] = 0.25 * float(np.dot(sp["m"], before**2 + after**2))
            history[f"momentum_{s}"][n] = 0.5 * float(np.dot(sp["m"], before + after))
            history[f"mass_{s}"][n] = float(sp["m"].sum())
            history["kinetic"][n] += history[f"kinetic_{s}"][n]
            if cadence > 0 and n % cadence == 0:
                snapshots[f"x_{s}_{n}"], snapshots[f"v_{s}_{n}"] = sp["x"].copy(), 0.5 * (before + after)
            sp["v"] = after
            sp["x"] = np.mod(sp["x"] + dt * after, length)
    history["time"] = dt * np.arange(cycles + 1)
    return history, snapshots


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit electrostatic PIC run from the particles of a TPIC run.")
    parser.add_argument("--run-dir", type=Path, default=Path("."),
                        help="simulation launch/run directory (default: current directory)")
    parser.add_argument("--input", type=Path, default=None, help="input deck (default: RUN_DIR/twoStream1D.inp)")
    parser.add_argument("--cycles", type=int, default=None, help="cycles to run (default: ncycles of the deck)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        params = read_input(args.input or args.run_dir / "twoStream1D.inp")
        cycles = args.cycles if args.cycles is not None else int(value(params, "ncycles"))
        omega_pe = math.sqrt(sum(value(params, "rhoINIT", s) * abs(value(params, "qom", s))
                                 for s in range(int(value(params, "ns"))) if value(params, "qom", s) < 0.0))
        initial = initial_particles(params, args.run_dir / params["SaveDirName"][0])
        history, snapshots = simulate(params, cycles, initial)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    energy_path, particles_path = args.run_dir / "explicit_pic_energy.csv", args.run_dir / "explicit_pic_particles.npz"
    columns = [name for name in history if name != "time"]
    with energy_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["omega_pe_t"] + columns)
        for n in range(cycles + 1):
            writer.writerow([omega_pe * history["time"][n]] + [history[name][n] for name in columns])
    np.savez_compressed(particles_path, **snapshots)
    total = history["field"] + history["kinetic"]
    print(f"cycles, omega_pe*dt            = {cycles}, {omega_pe * value(params, 'dt'):g}")
    print(f"particles per species          = {', '.join(str(x.size) for x, _, _ in initial)}")
    print(f"total energy change at the end = {total[-1] / total[0] - 1.0:+.3e} of the initial total")
    print(f"energy history written to      = {energy_path}")
    print(f"particle snapshots written to  = {particles_path}")


if __name__ == "__main__":
    main()
