#!/usr/bin/env python3
"""Frequency-wavenumber spectrum of the parallel whistlers of a uniform magnetized run.

The run of ``input/whistlerAnisotropy2D.inp`` seeds no wave: the particle
noise of a uniform plasma, whose electrons are hotter across the background
field than along it, excites the waves along the field.  This script
transforms the transverse magnetic field in x and in time and compares the
spectrum with the cold electron whistler relation

    c^2 k^2 / omega^2 = 1 - omega_pe^2 / (omega * (omega - Omega_ce)),

the test of D. Gonzalez-Herrero, E. Boella and G. Lapenta, Computer Physics
Communications 229, 162 (2018), section 5.

Usage from the repository root::

    python3 tools/postprocessing/analyze_whistler_spectrum.py --run-dir /path/to/run

The run directory is the one the simulation was launched from; it holds the
deck, ``whistlerAnisotropy2D.inp`` unless ``--input`` names another, and the
``SaveDirName`` of the deck with ``B`` snapshots at a fixed cadence.  The
background field must lie along x and the box must be periodic in x.

Method.  By and Bz are averaged over y and z, which keeps the parallel
wavenumbers, and transformed over x and t; no window is applied in time.
The power is ``|By(k, omega)|^2 + |Bz(k, omega)|^2``; a wave with k > 0 and
omega > 0 travels towards +x.  For every parallel mode in ``--ridge-k`` the
frequency of the largest power within 30 % of the whistler relation, refined
with a parabola through its neighbours, is compared with that relation: this
is the ridge of the printed summary and of the CSV.  A ridge found on the edge
of that interval is no ridge, and the summary counts those modes.  The second
panel is the transverse magnetic energy of the whole box, which shows whether
the anisotropy drives an instability, and where it grows and saturates.

The waves plot shows the power of ``By - i*s*Bz``, with s the sign of B0x.
There a positive frequency is a field that turns about the background field in
the sense of the electrons, right-handed, and a negative frequency one that
turns in the sense of the ions, left-handed.  Its first panel covers every
wavenumber of the grid and the frequencies up to the Nyquist frequency of the
snapshots, pi/dt_out, and frames the range of its second panel,
``--waves-max-k-de`` by ``--waves-max-omega``.  Both panels carry the
four waves of the cold relation along the field,

    c^2 k^2 = omega^2 - omega_pe^2 omega / (omega - Omega_ce)
                      - omega_pi^2 omega / (omega + Omega_ci),

the right-handed light wave and the whistler at positive frequency, the
ion-cyclotron wave and the left-handed light wave at negative frequency; the
ions are the first species of positive charge.

The travelling plot follows the whistlers of one band of wavenumbers,
``--travel-k`` in k*d_e, through the box: the transform of ``By - i*s*Bz``
keeps the band and the frequencies within 30 % of the whistler relation, and
the real part of the inverse is the By of those waves alone, free of the
light waves, of the other wavenumbers and of the waves travelling the other
way; a negative band selects the waves travelling towards -x.  Its first panel
stacks that By along x at seven times a sixth of a wave period apart, from
``--travel-start`` in Omega_ce*t (default: the middle of the record), each
drawn about its own time, so a crest moves along the dashed line of the phase
speed.  The second panel is the same field over x and t for twenty periods:
the crests run along the phase speed, the wave packets along the group speed.
Both speeds come from the ridge of the band, the frequency at the band centre
over its wavenumber and the slope of a line fitted through the ridge, and the
summary prints them beside those of the cold relation.

The crossing plot follows one wave packet of the band ``--crossing-k``
(default 0.6 to 0.8, whose packets cross the box of the shipped deck in about
1000/Omega_ce) across the whole periodic box.  Its first panel stacks By of
the band and its envelope over the whole box at seven times spread over the
record; the packet is the one whose path at the group speed holds the most
envelope, or with ``--crossing-packet N`` the N-th packet to its right (to its
left for N < 0), and its envelope is drawn in orange.  The packets are the
paths whose envelope is a local maximum among the paths beside them.  The peak
of the packet is followed from its highest point at the first profile,
snapshot by snapshot; the dots and the dashed lines lie on the straight line
fitted through it, so they move at the packet's own speed, printed in the
summary, and sit on its peak on average.  The orange covers the packet at each
profile, from the envelope minimum on its left to the one on its right.  The
second panel is the envelope over the whole box and record, where every packet
is a stripe along the group speed; the dashed line is the path of the packet.

The snapshots must resolve every frequency the run holds: a wave above
pi/dt_out folds back into the spectrum.  With th = 1 the scheme damps the
light waves, which lie there; with th = 0.5 write B every cycle.

Results: the plot ``RUN_DIR/whistler_spectrum.png``, the waves plot
``RUN_DIR/whistler_waves.png``, the travelling plot
``RUN_DIR/whistler_travelling.png`` and the crossing plot
``RUN_DIR/whistler_crossing.png``, each of the last three with a PDF of the
same name beside it, and the ridge table ``RUN_DIR/whistler_spectrum.csv``, or
the paths of ``--output``, ``--waves-plot``, ``--travel-plot``,
``--crossing-plot`` and ``--csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # source-checkout imports
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.fields import open_field_reader, transverse_mean

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


@dataclass
class FontSizes:
    """Font sizes of every plot, in points."""

    axis_label: float = 24.0  # the axes and the color bars
    tick_label: float = 22.0
    legend: float = 18.0


FONT_SIZES = FontSizes()

plt.rcParams.update({
    "axes.labelsize": FONT_SIZES.axis_label,
    "xtick.labelsize": FONT_SIZES.tick_label, "ytick.labelsize": FONT_SIZES.tick_label,
    "legend.fontsize": FONT_SIZES.legend, "legend.title_fontsize": FONT_SIZES.legend,
})


def first_float(values: dict[str, list[str]], key: str) -> float:
    return float(values[key][0])


# Roots of the cold relation along B0, in ascending order, with the name and the color of each wave in the plots.
LIGHT_L, ION_CYCLOTRON, WHISTLER, LIGHT_R = range(4)
WAVE_NAMES = ("Light wave (L)", "Ion-cyclotron", "Whistler", "Light wave (R)")
WAVE_COLORS = ("#e87ba4", "#008300", "#eda100", "#b71c1c")


def parallel_waves(k: np.ndarray, omega_pe: float, omega_ce: float, omega_pi: float, omega_ci: float,
                   c: float) -> np.ndarray:
    """Return the four roots of the cold relation along B0 for every wavenumber k, ascending, shape (k, 4).

    c^2 k^2 = omega^2 - omega_pe^2 omega / (omega - Omega_ce) - omega_pi^2 omega / (omega + Omega_ci).  A positive
    root turns in the sense of the electrons.  Without ions (omega_pi = Omega_ci = 0) the ion-cyclotron root is
    zero and the three others are those of the electron relation.
    """
    electron, ion = [1.0, -omega_ce], [1.0, omega_ci]
    species = omega_pe**2 * np.polymul([1.0, 0.0], ion) + omega_pi**2 * np.polymul([1.0, 0.0], electron)
    roots = np.empty((np.size(k), 4))
    for index, wavenumber in enumerate(np.abs(k)):
        # (omega^2 - c^2 k^2) (omega - Omega_ce) (omega + Omega_ci), minus the two species terms.
        vacuum = np.polymul([1.0, 0.0, -(c * wavenumber) ** 2], np.polymul(electron, ion))
        roots[index] = np.sort(np.roots(np.polysub(vacuum, species)).real)
    return roots


def read_transverse_field(path: Path,
                          periodic: tuple[bool, bool, bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the snapshot times, By and Bz averaged over y and z, shape (time, x), and the box mean of By^2 + Bz^2.

    x is periodic: its last node, the image of the first, is dropped.
    """
    times, by, bz, energy = [], [], [], []
    with open_field_reader(path) as reader:
        for step in range(reader.num_steps()):
            times.append(reader.read("time", step_selection=[step, 1])[0])
            components = [reader.read(name, step_selection=[step, 1]) for name in ("By", "Bz")]
            by.append(transverse_mean(components[0], periodic)[:-1])
            bz.append(transverse_mean(components[1], periodic)[:-1])
            energy.append(float(transverse_mean(components[0] ** 2 + components[1] ** 2, periodic)[:-1].mean()))
    return np.asarray(times), np.asarray(by), np.asarray(bz), np.asarray(energy)


def power_spectrum(field: np.ndarray) -> np.ndarray:
    """Return |FFT|^2 over (omega, k), both ascending, of a field of shape (time, x); no window is applied."""
    # exp(i*(k*x - omega*t)) lands on the negative frequencies of the transform: reverse that axis.
    return np.fft.fftshift(np.abs(np.fft.fft2(field - field.mean())) ** 2, axes=(0, 1))[::-1]


# The ridge of a mode is searched within this fraction of the whistler relation.
RIDGE_WINDOW = 0.3


def ridge(power: np.ndarray, omega: np.ndarray, k: np.ndarray, theory: np.ndarray, k_range: tuple[float, float],
          d_e: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the modes of k_range (in k*d_e, both signs), their ridge frequency and whether it lies on the edge."""
    modes = np.nonzero((np.abs(k) * d_e >= k_range[0]) & (np.abs(k) * d_e <= k_range[1]))[0]
    peaks, on_edge = np.empty(modes.size), np.zeros(modes.size, dtype=bool)
    for index, mode in enumerate(modes):
        band = np.nonzero(np.abs(omega / theory[mode] - 1.0) <= RIDGE_WINDOW)[0]
        if band.size < 3:
            raise ValueError(f"the frequency resolution leaves {band.size} samples around the whistler relation at "
                             f"k*d_e = {abs(k[mode]) * d_e:g}; raise the lower bound of --ridge-k")
        column = np.log(power[:, mode])
        top = band[np.argmax(column[band])]
        on_edge[index] = top in (band[0], band[-1])
        below, here, above = column[top - 1], column[top], column[top + 1]
        offset = 0.0 if on_edge[index] else 0.5 * (below - above) / (below - 2.0 * here + above)
        peaks[index] = omega[top] + offset * (omega[1] - omega[0])
    return modes, peaks, on_edge


def write_plot(path: Path, power: np.ndarray, omega: np.ndarray, k: np.ndarray, theory: np.ndarray,
               time: np.ndarray, energy: np.ndarray, omega_ce: float, d_e: float, b0: float,
               max_k_de: float, max_omega: float) -> None:
    fig, (map_axis, energy_axis) = plt.subplots(1, 2, figsize=(16, 8), width_ratios=(1.7, 1.0),
                                                constrained_layout=True)
    shown_k = np.abs(k) * d_e <= max_k_de
    shown_omega = np.abs(omega) <= max_omega * omega_ce
    level = np.log10(power[np.ix_(shown_omega, shown_k)])
    mesh = map_axis.pcolormesh(k[shown_k] * d_e, omega[shown_omega] / omega_ce, level, cmap="Blues",
                               vmin=np.percentile(level, 5.0), vmax=level.max(), shading="nearest", rasterized=True)
    fig.colorbar(mesh, ax=map_axis, label=r"$\log_{10}$ power of $B_y$ and $B_z$")
    for sign in (1.0, -1.0):
        map_axis.plot(k[shown_k] * d_e, sign * theory[shown_k] / omega_ce, color="#eb6834", linewidth=1.2,
                      linestyle=(0, (5, 3)), label="cold electron whistler" if sign > 0.0 else None)
    map_axis.set_xlim(-max_k_de, max_k_de)
    map_axis.set_ylim(-max_omega, max_omega)
    map_axis.set_xlabel(r"$k_\parallel d_e$")
    map_axis.set_ylabel(r"$\omega/\Omega_{ce}$")
    map_axis.legend(loc="upper center", frameon=False)

    energy_axis.semilogy(omega_ce * time, energy / b0**2, color="#2a78d6", linewidth=2.0)
    energy_axis.set_xlabel(r"$\Omega_{ce} t$")
    energy_axis.set_ylabel(r"$\langle B_y^2 + B_z^2 \rangle / B_0^2$")
    energy_axis.grid(True, which="both", alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


WAVE_LINE = {"linewidth": 2, "linestyle": (0, (5, 3))}


def draw_waves(axis, k_de: np.ndarray, waves: np.ndarray, limit: float) -> None:
    """Draw the four waves of the cold relation, in units of Omega_ce, up to +-limit."""
    for wave in range(4):
        axis.plot(k_de, np.where(np.abs(waves[:, wave]) <= limit, waves[:, wave], np.nan), color=WAVE_COLORS[wave],
                  **WAVE_LINE)


def write_waves_plot(path: Path, handed: np.ndarray, omega: np.ndarray, k: np.ndarray, waves: np.ndarray,
                     omega_ce: float, d_e: float, field_sign: float, max_k_de: float, max_omega: float) -> None:
    fig, (whole, near) = plt.subplots(1, 2, figsize=(16.0, 8), constrained_layout=True)
    k_de, frequency, waves = k * d_e, omega / omega_ce, waves / omega_ce
    positive = k_de > 0.0
    # The Nyquist frequency of the snapshots, pi/dt_out, in units of Omega_ce: the limit of the first panel.
    span = 0.5 * frequency.size * (frequency[1] - frequency[0])
    max_k_de, max_omega = min(max_k_de, k_de[-1]), min(max_omega, span)
    shown_k = np.abs(k_de) <= max_k_de
    shown_omega = np.abs(frequency) <= max_omega
    side = positive & shown_k

    level = np.log10(handed)
    near_level = level[np.ix_(shown_omega, shown_k)]
    # One color scale for both panels, set by the second one, where the waves lie; the first panel leaves white
    # what is weaker.
    look = {"cmap": "Blues", "vmin": np.percentile(near_level, 5.0), "vmax": np.percentile(near_level, 99.97),
            "shading": "nearest", "rasterized": True}
    mesh = whole.pcolormesh(k_de, frequency, level, **look)
    near.pcolormesh(k_de[shown_k], frequency[shown_omega], near_level, **look)
    fig.colorbar(mesh, ax=(whole, near),
                 label=rf"$\log_{{10}}|\hat B_y {'-' if field_sign > 0.0 else '+'} i\hat B_z|^2$")

    draw_waves(whole, k_de[positive], waves[positive], span)
    whole.add_patch(Rectangle((-max_k_de, -max_omega), 2.0 * max_k_de, 2.0 * max_omega, fill=False,
                              edgecolor="black", linewidth=0.9))
    # Listed from the highest frequency to the lowest, as the waves lie on the map.
    whole.legend([Line2D([], [], color=color, **WAVE_LINE) for color in reversed(WAVE_COLORS)],
                 list(reversed(WAVE_NAMES)), title="Cold relation", loc="upper left")
    whole.set_xlim(k_de[0], k_de[-1])
    whole.set_ylim(-span, span)
    whole.set_xlabel(r"$k_\parallel d_e$")
    whole.set_ylabel(r"$\omega/\Omega_{ce}$")

    draw_waves(near, k_de[side], waves[side], max_omega)
    near.set_xlim(-max_k_de, max_k_de)
    near.set_ylim(-max_omega, max_omega)
    near.set_xlabel(r"$k_\parallel d_e$")
    near.set_ylabel(r"$\omega/\Omega_{ce}$")

    path.parent.mkdir(parents=True, exist_ok=True)
    # In the PDF the maps are images at this resolution; the lines and the text are drawn.
    for target in {path, path.with_suffix(".pdf")}:
        fig.savefig(target, dpi=200)
    plt.close(fig)


def travelling_whistlers(right_handed: np.ndarray, output_dt: float, k: np.ndarray, theory: np.ndarray,
                         k_range: tuple[float, float], d_e: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the modes of k_range (in k*d_e, signed) and the amplitudes c(t, mode) of their whistlers.

    sum over the modes of c(t, mode) exp(i k x) is the right-handed field of the whistlers of the band: its real
    part is their By, its modulus their envelope.  Only the frequencies within RIDGE_WINDOW of the whistler relation
    are kept.
    """
    modes = np.nonzero((k * d_e >= k_range[0]) & (k * d_e <= k_range[1]))[0]
    if 0 in k[modes]:
        raise ValueError(f"the k*d_e band {k_range[0]:g} {k_range[1]:g} must hold modes of one sign, without k = 0")
    amplitudes = np.fft.fftshift(np.fft.fft(right_handed, axis=1), axes=1)[:, modes] / k.size
    # numpy transforms with exp(-i*w*t): the wave exp(i*(k*x - omega*t)) lies at w = -omega.
    omega = -2.0 * math.pi * np.fft.fftfreq(right_handed.shape[0], output_dt)
    kept = np.abs(omega[:, None] / theory[modes] - 1.0) <= RIDGE_WINDOW
    return modes, np.fft.ifft(np.fft.fft(amplitudes, axis=0) * kept, axis=0)


def band_field(amplitudes: np.ndarray, k_band: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the right-handed field of a band at the positions x: a sum over its few modes, smooth on any grid."""
    return amplitudes @ np.exp(1j * np.outer(k_band, x))


# The travelling plot: profiles a sixth of a period apart, over a window of wavelengths centred on the strongest
# packet; the map over more periods and wavelengths, centred on the path of that packet.
TRAVEL_PROFILES, TRAVEL_PROFILE_WAVELENGTHS = 7, 4.0
TRAVEL_MAP_PERIODS, TRAVEL_MAP_WAVELENGTHS = 20.0, 25.0


def band_speeds(k_band: np.ndarray, frequency: np.ndarray) -> tuple[float, float]:
    """Return the phase speed at the centre of a band and the group speed of the line fitted through omega(k)."""
    slope, intercept = np.polyfit(k_band, frequency, 1)
    return slope + intercept / k_band.mean(), slope


def write_travel_plot(path: Path, time: np.ndarray, k_band: np.ndarray, amplitudes: np.ndarray, start: float,
                      phase_speed: float, group_speed: float, omega_ce: float, d_e: float, b0: float,
                      length: float) -> None:
    wavelength, period = 2.0 * math.pi / abs(k_band.mean()), 2.0 * math.pi / (phase_speed * k_band.mean())
    output_dt = time[1] - time[0]
    first = int(np.argmin(np.abs(time - start)))
    step = max(1, round(period / 6.0 / output_dt))
    rows = first + step * np.arange(TRAVEL_PROFILES)
    last = min(time.size - 1, first + round(TRAVEL_MAP_PERIODS * period / output_dt))
    if rows[-1] > last:
        raise ValueError(f"--travel-start {omega_ce * start:g} leaves less than a wave period of the record")
    grid = np.linspace(0.0, length, 4096, endpoint=False)
    packet = grid[np.argmax(np.abs(band_field(amplitudes[first], k_band, grid)))]

    profile_x = packet + wavelength * np.linspace(-0.5, 0.5, 401) * TRAVEL_PROFILE_WAVELENGTHS
    profiles = band_field(amplitudes[rows], k_band, profile_x).real
    centre = packet + 0.5 * group_speed * (time[last] - time[first])
    map_x = centre + wavelength * np.linspace(-0.5, 0.5, 1201) * TRAVEL_MAP_WAVELENGTHS
    span = slice(first, last + 1)
    level = band_field(amplitudes[span], k_band, map_x).real / b0
    limit = np.abs(level).max()
    exponent = math.floor(math.log10(limit))
    scale = 10.0 ** exponent

    fig, (stack, space_time) = plt.subplots(1, 2, figsize=(16.0, 8), width_ratios=(1.0, 1.25),
                                            constrained_layout=True)
    tau, gap = omega_ce * time, omega_ce * step * output_dt
    for row, profile in zip(rows, profiles):
        stack.axhline(tau[row], color="0.8", linewidth=0.8, zorder=0)
        stack.plot(profile_x / d_e, tau[row] + 0.45 * gap * profile / np.abs(profiles).max(), color="#2a78d6",
                   linewidth=2.0)
    # The crest nearest the centre of the first profile, carried at the phase speed.
    crests = np.nonzero((profiles[0, 1:-1] > profiles[0, :-2]) & (profiles[0, 1:-1] >= profiles[0, 2:]))[0] + 1
    crest = profile_x[crests[np.argmin(np.abs(crests - profile_x.size // 2))]]
    stack.plot((crest + phase_speed * (time[rows[[0, -1]]] - time[first])) / d_e, tau[rows[[0, -1]]], color="black",
               **WAVE_LINE)
    stack.set_xlim(profile_x[0] / d_e, profile_x[-1] / d_e)
    stack.set_ylim(tau[rows[0]] - gap, tau[rows[-1]] + gap)
    stack.set_xlabel(r"$x/d_e$")
    stack.set_ylabel(r"$\Omega_{ce} t$")

    mesh = space_time.pcolormesh(map_x / d_e, tau[span], level / scale, cmap="RdBu_r", vmin=-limit / scale,
                                 vmax=limit / scale, shading="nearest", rasterized=True)
    fig.colorbar(mesh, ax=space_time, label=rf"$B_y / (10^{{{exponent}}} B_0)$")
    lines, unit = time[[first, last]] - time[first], d_e * omega_ce
    space_time.plot((crest + phase_speed * lines) / d_e, tau[[first, last]], color="black",
                    **WAVE_LINE, label=rf"phase speed ${phase_speed / unit:.3f}\,d_e\Omega_{{ce}}$")
    space_time.plot((packet + group_speed * lines) / d_e, tau[[first, last]], color="black",
                    linewidth=2.5, linestyle=":", label=rf"group speed ${group_speed / unit:.3f}\,d_e\Omega_{{ce}}$")
    space_time.add_patch(Rectangle((profile_x[0] / d_e, tau[rows[0]]), (profile_x[-1] - profile_x[0]) / d_e,
                                   tau[rows[-1]] - tau[rows[0]], fill=False, edgecolor="black", linewidth=1.5))
    space_time.set_xlim(map_x[0] / d_e, map_x[-1] / d_e)
    space_time.set_ylim(tau[first], tau[last])
    space_time.set_xlabel(r"$x/d_e$")
    space_time.set_ylabel(r"$\Omega_{ce} t$")
    # The lines run towards the upper corner on the side the waves travel to: the legend takes the other one.
    space_time.legend(loc="upper left" if phase_speed > 0.0 else "upper right", framealpha=0.85)

    path.parent.mkdir(parents=True, exist_ok=True)
    for target in {path, path.with_suffix(".pdf")}:
        fig.savefig(target, dpi=200)
    plt.close(fig)


# The crossing plot: profiles of the whole box at times spread over the record, each in the middle of its share.
CROSSING_PROFILES = 7


def write_crossing_plot(path: Path, time: np.ndarray, k_band: np.ndarray, amplitudes: np.ndarray,
                        group_speed: float, omega_ce: float, d_e: float, b0: float, length: float,
                        packet_offset: int) -> float:
    rows = np.round((np.arange(CROSSING_PROFILES) + 0.5) / CROSSING_PROFILES * (time.size - 1)).astype(int)
    # 600 to 1200 snapshots of the envelope (all of a shorter record), on a grid finer than a packet. The grids of
    # the whole box end at x = length, the image of x = 0, so the plots reach the right edge.
    sample = slice(None, None, max(1, time.size // 600))
    grid = np.linspace(0.0, length, 1025)
    envelope = np.abs(band_field(amplitudes[sample], k_band, grid)) / b0
    # The packets: the starts, at the first profile, of the paths at the group speed whose envelope sum is a local
    # maximum over the distinct points of the periodic box. The packet followed is packet_offset places to the right
    # of the one with the largest sum.
    moving = amplitudes[sample] * np.exp(1j * np.outer(group_speed * (time[sample] - time[rows[0]]), k_band))
    total = np.abs(band_field(moving, k_band, grid[:-1])).sum(axis=0)
    starts = np.nonzero((total > np.roll(total, 1)) & (total >= np.roll(total, -1)))[0]
    stripe = grid[starts[(np.argmax(total[starts]) + packet_offset) % starts.size]]
    x = np.linspace(0.0, length, 4097)
    profiles = band_field(amplitudes[rows], k_band, x)
    signed = lambda at, centre: np.mod(at - centre + 0.5 * length, length) - 0.5 * length

    def nearest_maximum(values: np.ndarray, points: np.ndarray, guess: float) -> int:
        """Return the index of the local maximum of the periodic values nearest to the point guess."""
        tops = np.nonzero((values > np.roll(values, 1)) & (values >= np.roll(values, -1)))[0]
        return tops[np.argmin(np.abs(signed(points[tops], guess)))]

    # The peak of the packet: at the first profile the highest point of its envelope within half a packet length,
    # pi over the band width, of its stripe; at every other sampled snapshot the envelope maximum nearest to where
    # the group speed carries the peak of the snapshot beside it.
    near = np.abs(signed(x, stripe)) <= math.pi / (k_band.max() - k_band.min())
    times = time[sample]
    first = int(np.argmin(np.abs(times - time[rows[0]])))
    peaks = np.empty(times.size)
    peaks[first] = x[near][np.argmax(np.abs(profiles[0, near]))]
    for i in (*range(first + 1, times.size), *range(first - 1, -1, -1)):
        j = i - 1 if i > first else i + 1
        peaks[i] = grid[nearest_maximum(envelope[i, :-1], grid[:-1], peaks[j] + group_speed * (times[i] - times[j]))]
    # The dots move along the straight line fitted through the peak over the record, unwrapped across the periodic
    # boundary: at the speed of the packet, and on its peak on average.
    unwrapped = peaks[0] + np.concatenate(([0.0], np.cumsum(signed(peaks[1:], peaks[:-1]))))
    speed, start = np.polyfit(times - time[rows[0]], unwrapped, 1)

    position = lambda at: np.mod(start + speed * (at - time[rows[0]]), length)

    def track(at: np.ndarray) -> np.ndarray:
        """Return the packet position at the dense times at, nan where it leaves one edge for the other."""
        x = position(at)
        x[1:][np.abs(np.diff(x)) > 0.5 * length] = np.nan
        return x

    fig, (stack, space_time) = plt.subplots(1, 2, figsize=(16.0, 8), sharey=True, constrained_layout=True)
    tau, gap = omega_ce * time, omega_ce * (time[rows[1]] - time[rows[0]])
    height = 0.45 * gap / np.abs(profiles).max()
    for row, profile in zip(rows, profiles):
        stack.axhline(tau[row], color="0.8", linewidth=0.8, zorder=0)
        stack.plot(x / d_e, tau[row] + height * profile.real, color="#9ec5f4", linewidth=0.6)
        # The packet at this profile: from its peak down to the nearest envelope minimum on either side.
        values = np.abs(profile[:-1])
        left = right = nearest_maximum(values, x[:-1], peaks[np.argmin(np.abs(times - time[row]))])
        while values[(left - 1) % values.size] < values[left % values.size]:
            left -= 1
        while values[(right + 1) % values.size] < values[right % values.size]:
            right += 1
        inside = np.zeros(x.size, dtype=bool)
        inside[np.arange(left, right + 1) % values.size] = True
        inside[-1] = inside[0]
        for sign in (1.0, -1.0):
            stack.plot(x / d_e, tau[row] + sign * height * np.abs(profile), color="#184f95", linewidth=1.5)
            stack.plot(x / d_e, np.where(inside, tau[row] + sign * height * np.abs(profile), np.nan),
                       color="#eb6834", linewidth=3.0)
    line = np.linspace(time[rows[0]], time[rows[-1]], 2001)
    stack.plot(track(line) / d_e, omega_ce * line, color="black", linewidth=2.5, linestyle=(0, (5, 3)))
    stack.plot(position(time[rows]) / d_e, tau[rows], "o", color="black", markersize=8)
    stack.set_xlim(0.0, length / d_e)
    stack.set_ylim(tau[0], tau[-1])
    stack.set_xlabel(r"$x/d_e$")
    stack.set_ylabel(r"$\Omega_{ce} t$")

    exponent = math.floor(math.log10(envelope.max()))
    scale = 10.0 ** exponent
    mesh = space_time.pcolormesh(grid / d_e, tau[sample], envelope / scale, cmap="Blues", vmin=0.0,
                                 shading="nearest", rasterized=True)
    fig.colorbar(mesh, ax=space_time, label=rf"Envelope of $B_y$ / $(10^{{{exponent}}} B_0)$")
    line = np.linspace(time[0], time[-1], 4001)
    space_time.plot(track(line) / d_e, omega_ce * line, color="black", linewidth=2.5, linestyle=(0, (5, 3)),
                    label=rf"Packet speed ${speed / (d_e * omega_ce):.3f}\,d_e\Omega_{{ce}}$")
    for row in rows:
        space_time.axhline(tau[row], color="black", linewidth=0.8, linestyle=(0, (2, 4)))
    space_time.set_xlim(0.0, length / d_e)
    space_time.set_xlabel(r"$x/d_e$")
    space_time.legend(loc="upper left" if group_speed < 0.0 else "upper right", framealpha=0.85)

    path.parent.mkdir(parents=True, exist_ok=True)
    for target in {path, path.with_suffix(".pdf")}:
        fig.savefig(target, dpi=200)
    plt.close(fig)
    return speed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frequency-wavenumber spectrum of the parallel whistlers of a uniform magnetized run."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("."),
                        help="simulation launch/run directory (default: current directory)")
    parser.add_argument("--input", type=Path, default=None,
                        help="input deck (default: RUN_DIR/whistlerAnisotropy2D.inp)")
    parser.add_argument("--max-k-de", type=float, default=2.0,
                        help="largest |k*d_e| of the plot (default: 2)")
    parser.add_argument("--max-omega", type=float, default=1.0,
                        help="largest |omega|/Omega_ce of the plot (default: 1)")
    parser.add_argument("--ridge-k", type=float, nargs=2, default=(0.5, 1.5), metavar=("MIN", "MAX"),
                        help="|k*d_e| interval whose ridge is compared with the whistler relation (default: 0.5 1.5)")
    parser.add_argument("--output", "--plot", dest="plot", type=Path, default=None,
                        help="plot path (default: RUN_DIR/whistler_spectrum.png)")
    parser.add_argument("--waves-plot", type=Path, default=None,
                        help="waves plot path (default: RUN_DIR/whistler_waves.png); a PDF of the same name is "
                             "written beside it")
    parser.add_argument("--waves-max-k-de", type=float, default=2.5,
                        help="largest |k*d_e| of the second panel of the waves plot (default: 2.5)")
    parser.add_argument("--waves-max-omega", type=float, default=3.5,
                        help="largest |omega|/Omega_ce of the second panel of the waves plot (default: 3.5)")
    parser.add_argument("--travel-plot", type=Path, default=None,
                        help="travelling plot path (default: RUN_DIR/whistler_travelling.png); a PDF of the same name "
                             "is written beside it")
    parser.add_argument("--travel-k", type=float, nargs=2, default=(1.4, 1.6), metavar=("MIN", "MAX"),
                        help="k*d_e band of the travelling plot; negative values select the waves travelling towards "
                             "-x (default: 1.4 1.6)")
    parser.add_argument("--travel-start", type=float, default=None,
                        help="Omega_ce*t of the first profile of the travelling plot (default: the middle of the "
                             "record)")
    parser.add_argument("--crossing-plot", type=Path, default=None,
                        help="crossing plot path (default: RUN_DIR/whistler_crossing.png); a PDF of the same name is "
                             "written beside it")
    parser.add_argument("--crossing-k", type=float, nargs=2, default=(0.6, 0.8), metavar=("MIN", "MAX"),
                        help="k*d_e band of the crossing plot, signed like --travel-k (default: 0.6 0.8, whose packets "
                             "cross the box of the shipped deck in about 1000/Omega_ce)")
    parser.add_argument("--crossing-packet", type=int, default=0, metavar="N",
                        help="packet the crossing plot follows: 0 the one whose path at the group speed holds the most "
                             "envelope, -1 the next packet to its left, +1 the next to its right, and so on "
                             "(default: 0)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="ridge table path (default: RUN_DIR/whistler_spectrum.csv)")
    return parser


def analyze(args: argparse.Namespace) -> None:
    input_path = args.input or (args.run_dir / "whistlerAnisotropy2D.inp")
    params = read_input(input_path)
    if first_float(params, "B0y") != 0.0 or first_float(params, "B0z") != 0.0 or first_float(params, "B0x") == 0.0:
        raise ValueError(f"{input_path}: the background field must lie along x (B0x nonzero, B0y = B0z = 0)")
    periodic = periodic_axes(params)
    if not periodic[0]:
        raise ValueError(f"{input_path}: the spectrum needs a box periodic in x")
    species = range(int(first_float(params, "ns")))
    electron = next((index for index in species if float(params["qom"][index]) < 0.0), None)
    if electron is None:
        raise ValueError(f"{input_path}: no electron species (qom < 0)")
    ion = next((index for index in species if float(params["qom"][index]) > 0.0), None)
    c = float(params.get("c", ["1"])[0])
    b0 = abs(first_float(params, "B0x"))
    mass_ratio = abs(float(params["qom"][electron]))
    omega_pe = math.sqrt(float(params["rhoINIT"][electron]) * mass_ratio)
    omega_ce = mass_ratio * b0
    omega_pi = 0.0 if ion is None else math.sqrt(float(params["rhoINIT"][ion]) * float(params["qom"][ion]))
    omega_ci = 0.0 if ion is None else float(params["qom"][ion]) * b0
    d_e = c / omega_pe
    cells, length = int(first_float(params, "nxc")), first_float(params, "Lx")

    data_dir = args.run_dir / params["SaveDirName"][0]
    time, by, bz, energy = read_transverse_field(data_dir / "B", periodic)
    if time.size < 8:
        raise ValueError(f"{data_dir}: a spectrum needs at least 8 snapshots, found {time.size}")
    output_dt = (time[-1] - time[0]) / (time.size - 1)
    if not np.allclose(np.diff(time), output_dt, rtol=1.0e-6):
        raise ValueError(f"{data_dir}: the snapshots are not evenly spaced in time")
    if by.shape[1] != cells:
        raise ValueError(f"{data_dir}: {by.shape[1]} cells along x, the deck has {cells}")

    total = power_spectrum(by) + power_spectrum(bz)
    field_sign = math.copysign(1.0, first_float(params, "B0x"))
    right_handed = by - 1j * field_sign * bz
    handed = power_spectrum(right_handed)
    k = 2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(cells, length / cells))
    omega = -2.0 * math.pi * np.fft.fftshift(np.fft.fftfreq(time.size, output_dt))[::-1]
    # The ridge is compared with the relation of the paper, which has no ion term.
    theory = parallel_waves(k, omega_pe, omega_ce, 0.0, 0.0, c)[:, WHISTLER]
    modes, peaks, on_edge = ridge(total, omega, k, theory, tuple(args.ridge_k), d_e)
    if on_edge.all():
        raise ValueError(f"no parallel mode in --ridge-k {args.ridge_k[0]:g} {args.ridge_k[1]:g} has a ridge within "
                         f"{RIDGE_WINDOW:g} of the whistler relation")
    deviation = (peaks / theory[modes] - 1.0)[~on_edge]

    def band(k_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
        """Return the modes of a k*d_e band, the amplitudes of their whistlers and the phase and group speeds of the
        band, from its ridge in the right-handed spectrum and from the whistler relation."""
        band_modes, amplitudes = travelling_whistlers(right_handed, output_dt, k, theory, k_range, d_e)
        ridge_modes, ridge_peaks, ridge_edge = ridge(handed, omega, k, theory, tuple(np.sort(np.abs(k_range))), d_e)
        found = np.isin(ridge_modes, band_modes) & ~ridge_edge
        if np.count_nonzero(found) < 2:
            raise ValueError(f"the k*d_e band {k_range[0]:g} {k_range[1]:g} has fewer than two modes with a ridge")
        return (band_modes, amplitudes, band_speeds(k[ridge_modes[found]], ridge_peaks[found]),
                band_speeds(k[ridge_modes[found]], theory[ridge_modes[found]]))

    travel_modes, amplitudes, (phase_speed, group_speed), (cold_phase_speed, cold_group_speed) = band(
        tuple(args.travel_k))
    travel_start = time[time.size // 2] if args.travel_start is None else args.travel_start / omega_ce
    # By Parseval the power of every k != 0 is the variance along x.
    band_share = (np.abs(amplitudes) ** 2).sum() / right_handed.var(axis=1).sum()
    crossing_modes, crossing_amplitudes, crossing_speeds, cold_crossing_speeds = band(tuple(args.crossing_k))

    plot_path = args.plot or (args.run_dir / "whistler_spectrum.png")
    waves_path = args.waves_plot or (args.run_dir / "whistler_waves.png")
    travel_path = args.travel_plot or (args.run_dir / "whistler_travelling.png")
    crossing_path = args.crossing_plot or (args.run_dir / "whistler_crossing.png")
    csv_path = args.csv or (args.run_dir / "whistler_spectrum.csv")
    write_plot(plot_path, total, omega, k, theory, time, energy, omega_ce, d_e, b0, args.max_k_de, args.max_omega)
    write_waves_plot(waves_path, handed, omega, k, parallel_waves(k, omega_pe, omega_ce, omega_pi, omega_ci, c),
                     omega_ce, d_e, field_sign, args.waves_max_k_de, args.waves_max_omega)
    write_travel_plot(travel_path, time, k[travel_modes], amplitudes, travel_start, phase_speed, group_speed,
                      omega_ce, d_e, b0, length)
    packet_speed = write_crossing_plot(crossing_path, time, k[crossing_modes], crossing_amplitudes,
                                       crossing_speeds[1], omega_ce, d_e, b0, length, args.crossing_packet)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["k_de", "omega_over_omega_ce", "whistler_omega_over_omega_ce", "on_edge"])
        for mode, peak, edge in zip(modes, peaks, on_edge):
            writer.writerow([f"{k[mode] * d_e:.9g}", f"{peak / omega_ce:.9g}", f"{theory[mode] / omega_ce:.9g}",
                             int(edge)])

    growth = int(np.argmax(energy))
    print(f"input deck                     = {input_path}")
    print(f"data directory                 = {data_dir}")
    print(f"mass ratio mi/me               = {mass_ratio:g}")
    print(f"omega_pe, Omega_ce [omega_pi]  = {omega_pe:.9g}, {omega_ce:.9g}")
    print(f"snapshots, output dt           = {time.size}, {output_dt:.9g}")
    print(f"Omega_ce*t range               = [{omega_ce * time[0]:.9g}, {omega_ce * time[-1]:.9g}]")
    print(f"largest omega/Omega_ce, k*d_e  = {math.pi / output_dt / omega_ce:.9g},"
          f" {math.pi * cells / length * d_e:.9g}")
    print(f"resolution in omega/Omega_ce   = {2.0 * math.pi / (time.size * output_dt) / omega_ce:.9g}")
    print(f"resolution in k*d_e            = {2.0 * math.pi / length * d_e:.9g}")
    print(f"transverse B^2/B0^2 first, max = {energy[0] / b0**2:.9g}, {energy[growth] / b0**2:.9g}"
          f" at Omega_ce*t = {omega_ce * time[growth]:.9g}")
    print(f"ridge |k*d_e| interval         = [{args.ridge_k[0]:g}, {args.ridge_k[1]:g}] ({modes.size} modes,"
          f" {int(on_edge.sum())} without a ridge)")
    print(f"ridge / whistler relation - 1  = mean {deviation.mean():+.4f}, RMS {math.sqrt((deviation**2).mean()):.4f},"
          f" largest {np.abs(deviation).max():.4f}")
    speed_unit = d_e * omega_ce
    print(f"travelling k*d_e band          = [{args.travel_k[0]:g}, {args.travel_k[1]:g}] ({travel_modes.size} modes,"
          f" {band_share:.4f} of the transverse B energy at k_y = 0, k != 0)")
    print(f"phase speed ridge, cold        = {phase_speed / speed_unit:+.4f}, {cold_phase_speed / speed_unit:+.4f}"
          f" d_e*Omega_ce")
    print(f"group speed ridge, cold        = {group_speed / speed_unit:+.4f}, {cold_group_speed / speed_unit:+.4f}"
          f" d_e*Omega_ce")
    print(f"crossing k*d_e band            = [{args.crossing_k[0]:g}, {args.crossing_k[1]:g}] ({crossing_modes.size}"
          f" modes)")
    print(f"group speed ridge, cold        = {crossing_speeds[1] / speed_unit:+.4f},"
          f" {cold_crossing_speeds[1] / speed_unit:+.4f} d_e*Omega_ce: the box in"
          f" Omega_ce*t = {omega_ce * length / abs(crossing_speeds[1]):.0f}")
    print(f"crossing packet {args.crossing_packet:+d} speed       = {packet_speed / speed_unit:+.4f} d_e*Omega_ce")
    print(f"CSV written to                 = {csv_path}")
    print(f"plot written to                = {plot_path}")
    print(f"waves plot written to          = {waves_path}")
    print(f"waves plot as PDF              = {waves_path.with_suffix('.pdf')}")
    print(f"travelling plot written to     = {travel_path}")
    print(f"travelling plot as PDF         = {travel_path.with_suffix('.pdf')}")
    print(f"crossing plot written to       = {crossing_path}")
    print(f"crossing plot as PDF           = {crossing_path.with_suffix('.pdf')}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        analyze(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
