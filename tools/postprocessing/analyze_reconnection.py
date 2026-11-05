#!/usr/bin/env python3
"""Measure and compare the reconnection rate on the X line of Harris-sheet runs.

Running.  The script needs NumPy and Matplotlib, and SciPy for
``--instabilities`` and ``--render-view ropes`` (``python3 -m pip install -e
'.[analysis]'`` in the TPIC root installs them) and runs from any directory;
``--help`` lists every option.  One run is analyzed alone; several runs are
overlaid on the same normalized axes and compared with the first one over
their common time interval::

    # one run: the only deck of the directory it was launched from
    python3 tools/postprocessing/analyze_reconnection.py --run-dir runs/gem3d

    # two runs, each launched from its own directory
    python3 tools/postprocessing/analyze_reconnection.py \
        --run-dir runs/a runs/b --labels "run a" "run b"

    # two decks launched from one directory, the rates averaged over
    # 1/Omega_ci, the plots and the CSV named after the comparison
    python3 tools/postprocessing/analyze_reconnection.py \
        --run-dir runs/gem3d --input runs/gem3d/a.inp runs/gem3d/b.inp \
        --labels "code a" "code b" --time-average 1 \
        --output runs/gem3d/a_vs_b.png --csv runs/gem3d/a_vs_b.csv

    # instead of the rates, the 3D view of the ion density and the field
    # lines at the X line at cycles 3000 and 7000
    python3 tools/postprocessing/analyze_reconnection.py --run-dir runs/gem3d \
        --render-cycle 3000 7000

Every argument is optional:

* ``--run-dir DIR ...``: the directory each run was launched from, to which
  the ``SaveDirName`` of its deck is relative (default: the current
  directory).  One directory serves every ``--input`` deck.
* ``--input DECK ...``: the deck of each run, relative to the current
  directory (default: the only ``*.inp`` of each run directory).
* ``--labels NAME ...``: the legend label of each run.
* ``--max-cycle CYCLE``: analyze only the snapshots and the energy records up
  to cycle CYCLE, in every run (default: all of them).
* ``--time-average WINDOW``: average the rates over WINDOW, in 1/Omega_ci
  (default: 0, no average).
* ``--box-cells N``: the side of the square around the X line that averages
  Ez (default: 8).
* ``--alfven-density sheet|background``: the ions whose density sets v_A.
* ``--print-data``: print the values of every snapshot.
* ``--output PLOT`` and ``--csv CSV``: where the results go.
* ``--render-cycle CYCLE ...``: render the 3D view of each run at each CYCLE
  instead of analyzing the rates (see "3D view" below).
* ``--render-view VIEW`` and ``--render-region X0 X1 Y0 Y1 Z0 Z1``: what the
  3D view shows, and where.
* ``--render-style FIELD=VALUE ...``: change the look of the 3D view, field by
  field of ``RENDER_STYLE``.
* ``--render-inset X0 X1 Y0 Y1 Z0 Z1`` and ``--render-inset-view VIEW``: add a
  zoom on that box to the 3D view, showing VIEW (default: the main view).
* ``--render-particles P ...``: draw the energetic particles over the 3D view
  (see "Energetic particles" below).
* ``--density-species S ...``: the species whose density the 3D view shows.
* ``--instabilities`` and ``--workers N``: plot the sheet instabilities instead
  of the rates (see "Sheet instabilities" below).
* ``--particle-cycle CYCLE ...``, ``--particle-species S ...``,
  ``--particle-percentile P`` and ``--particle-points N``: plot the most
  energetic particles of a particle snapshot instead of the rates (see
  "Energetic particles" below).

The results replace files of the same names; every plot written as a PNG is
also written as a PDF of the same name beside it, in which a 3D view embeds
its image.  They are:

* the plot of all panels, at ``--output`` (default:
  ``RUN_DIR/reconnection_rate.png`` for one deck and
  ``reconnection_comparison.png`` in the current directory for several);
* the plot of the flux rate alone, at the path of the first plot with
  ``_flux_rate`` added to the name;
* the plot of the total energy, likewise with ``_energy`` added, when a run
  has an energy log;
* the CSV, at ``--csv`` (default: the default path of the first plot with the
  extension ``.csv``, whatever ``--output`` is);
* on the terminal the normalization, the summary and the total energy of
  each run, the comparison with the first run and the paths written.

A run is an input deck and the directory it was launched from; one run
directory serves several decks, which then label the runs.  Under the
``SaveDirName`` of its deck a run holds nodal B and E snapshots in one of two
layouts, recognized from the files:

* TPIC field files ``B_t*.vti`` and ``E_t*.vti``, one per field and cycle
  (``FieldOutputTag = B+E``).  The sheet amplitude is the deck key ``HarrisB0x``.
* One binary structured-points VTK file per field and cycle,
  ``<SimulationName>_B_<cycle>.vtk`` and ``<SimulationName>_E_<cycle>.vtk``.
  The pair of a cycle is written between the two field updates of that cycle:
  B is at the time ``cycle*dt`` and E one step later, at ``(cycle + 1)*dt``.
  The sheet amplitude is the deck key ``B0x``.

A run therefore carries two time axes, one for the flux and its rate and one
for the direct sample of Ez; they coincide for TPIC snapshots.

Both layouts are read with the same deck keys, units and geometry, so every
run goes through one analysis.

``--max-cycle CYCLE`` ends every run at cycle CYCLE: the snapshots and the
energy records of a later cycle are left out and their field files are not
read.  The cycle of a TPIC snapshot is the one its file holds, that of a
structured-points file the one in its name.  A run that is still being written
is thereby compared with a finished one over the cycles that both hold.

Geometry.  ``Case = GEM``, ``GEMnoPert`` or ``GEMHarris``: the sheet is the plane
y = Ly/2 and the X line is x = Lx/2, y = Ly/2 for every z.  ``Case = DoubleGEM`` or
``DoubleGEMnoPert``: two sheets of opposite field in a fully periodic box, the
lower with its X line at x = Lx/4, y = Ly/4 and the upper at x = 3Lx/4,
y = 3Ly/4.  Each X line is analyzed as a run of its own, labelled with its
sheet, so the two sheets of one deck are compared like two runs.  With periodic
x the O line of a sheet is half a period in x from its X line.

Units.  The runs have c = 1 and advance

    dB/dt = -curl E,   dE/dt = curl B - 4*pi*J,   dv/dt = qom*(E + v x B),

where a macroparticle charge carries the factor 1/(4*pi): 4*pi times the
charge density of a species is its rhoINIT profile, and 4*pi times its mass
density is rhoINIT/qom.  E and B share one unit, lengths are c/omega_pi and
times 1/omega_pi of a species with rhoINIT = qom = 1, and velocities are in
units of c.  With B0 the magnitude of the sheet amplitude,

    v_A = B0/sqrt(sum(rhoINIT/qom)),   Omega_ci = qom*B0,   d_i = v_A/Omega_ci.

The sum runs over the sheet ions, the ion species that drift: this is the
normalization of the GEM challenge.  ``--alfven-density background`` takes the
ions that do not drift instead, the inflow density.

Method.  The flux reconnected through the X line, per unit length of z, is

    psi(z, t) = 1/2 * [int_X^{X+Lx/2} By dx - int_{X-Lx/2}^X By dx]   on the sheet,

with x periodic and X the x of the X line, multiplied by the sign of the sheet
field so that reconnection makes it grow.  Faraday's law gives, for the z average, d psi/dt = Ez(O line) - Ez(X
line), so

    rate = (d psi/dt) / (B0*v_A)

is the reconnection electric field on the X line in units of B0*v_A, averaged
over the time between the snapshots that the time derivative spans.

The second estimate samples that field directly: -Ez/(B0*v_A), with the same
sign convention, averaged over z and over a square of ``--box-cells`` cells in
x and y centred on the X line (default 8; 0 samples the X line alone).  It is an
instantaneous value, so it keeps every field oscillation that the snapshot
holds; a light wave has E = dB in these units, which is 1/(v_A/c) times the
reconnection field of the same dB.

``--time-average WINDOW`` replaces both rates by their mean over the WINDOW
(in 1/Omega_ci) centred on each snapshot, which removes the oscillations
shorter than the window and makes the direct sample comparable with the flux
rate and between runs.  The mean is that of the rate interpolated linearly
between the snapshots, so uneven snapshot times are weighted by time.  A
snapshot closer than half a window to either end of its run has no rate: the
CSV holds nan there and the plot and the printed numbers leave it out.  The
rates before that average stay in the ``unaveraged`` columns of the CSV,
whatever the window.

``--print-data`` prints the z averages of every snapshot of every run: the
flux, the unaveraged rates and, with a window, their average.

Energy.  The total energy of a run comes from the scalar log of its output
directory: ``diagnostics.csv``, whose columns ``cycle``, ``time`` and
``total_energy`` follow its comment lines, or the table ``ConservedQuantities.txt``, whose
columns ``Cycle`` and ``Total_Energy`` give the energy at ``Cycle*dt`` to
within one time step.  It is plotted as its change since the first record,
``(E - E0)/E0``, so that runs loaded with slightly different energies start
together; the terminal states the first and the last value.  A run without a
log is left out of the plot.

Plots.  The runs take their colors from ``RUN_COLORS`` below, their markers
from ``RUN_MARKERS``, their line styles and widths from ``RUN_LINE_STYLES``
and ``RUN_LINE_WIDTHS`` and their place in the legends in the order given on
the command line, in every plot; an earlier run is drawn over a later one.
``FontSizes`` sets the font sizes of every plot.
``LinePlotStyle`` sets the colors, line styles, legend, axis labels and title
of the plot of the flux rate alone and of the plot of the energy, each through
its own instance, ``FLUX_RATE_PLOT_STYLE`` and ``ENERGY_PLOT_STYLE``; edit
their values to change the look.

Comparison.  Every other run is compared with the first one on the snapshot
times of the first that both cover: the mean and RMS difference of the flux
and of the two rates, the peak flux rate of both, and a time lag.  The lag is
the difference of the times at which the flux gained since its minimum reaches
half of the smaller of the two total gains.  The onset of reconnection depends
on the noise that seeds it, so two runs that reconnect at the same rate differ
by a lag, which is also what the rate differences then show.

3D view.  ``--render-cycle CYCLE ...`` replaces the rate analysis by one image
per run and cycle, rendered off screen with PyVista (``python3 -m pip install
pyvista``): the volume of the ion density with the field lines of B through
the X lines drawn as tubes, in a box with axes in d_i.  The density is the sum
of the ``--density-species`` (default: the sheet ions, those that drift), read
from ``rho_<s>_t<cycle>.vti`` or ``<SimulationName>_rho<s>_<cycle>.vtk``.  The
field lines start in a few planes of z, above and below each X line on field
lines still to reconnect and on the neutral sheet either side of it on
reconnected ones, and are traced both ways.  ``RENDER_STYLE`` below sets the colormap, opacity,
seeds, line length, camera and fonts; edit its values to change the look.  An
image goes to ``RUN_DIR/<deck name>_3d_t<cycle>.png``, or with ``--output
PLOT`` to the path of PLOT with ``_t<cycle>`` added to the name, and with the
label of the run before it when there are several runs.

``--render-view`` zooms instead on one instability of the sheet (see "Sheet
instabilities"), into ``<deck name>_3d_<view>_t<cycle>``.  ``kink`` keeps the
density volume alone, in which the current sheet bends along z.  The other two
thin the volume and add an overlay with a second colour bar: ``lower-hybrid``
the isosurfaces of the lower-hybrid part of the Ez fluctuation, or of Ez itself with
``lh_quantity = "Ez"``, at plus and minus ``lh_iso_fraction`` of its 99.5th
percentile, and ``ropes`` the field lines
that start either side of the secondary O points, coloured by Bz: a flux rope
winds them about a core field along z, a two-dimensional island closes them in
x-y planes.  The zoom
spans ``--render-region`` (X0 X1 Y0 Y1 Z0 Z1 in d_i) or, by default, the
half-sizes ``kink_zoom``, ``lower_hybrid_zoom`` and ``ropes_zoom`` of
``RENDER_STYLE`` about the kink station on the sheet or the O point of most
flux.  ``--render-region`` also zooms the ``x-line`` view.

Sheet instabilities.  ``--instabilities`` replaces the rate analysis by three
plots per sheet that follow the current sheet through every snapshot up to
``--max-cycle``, read by ``--workers`` processes at once; the box must be
periodic in x and z.  z is the direction of the sheet current, y_s the y of
the sheet and delta its half-thickness (deck key ``delta``).

* ``_lhdi``: the lower-hybrid drift waves, short waves of Ez along z on the
  sheet edges.  The fluctuation of Ez about its z mean is split by kz; the part
  with kz*sqrt(rho_e*rho_i) >= ``LH_SPLIT`` (rho = uth/(|qom|*B0) of the first
  sheet electron and ion species) is the lower-hybrid part.  Panels: its rms
  over x and z against y and time; the power spectrum in kz over the sheet
  edges, ``LH_EDGE_BAND`` delta from the sheet, against time; the rms on the
  edges and at the centre with an exponential fit; and its y-z slices, with the
  sheet outlined by half the peak density, at the station x ``KINK_STATION``*Lx
  before the X line.
* ``_kink``: the drift kink, a bend of the neutral sheet along z.  The neutral
  sheet is the y where Bx changes sign closest to the sheet, within
  ``SHEET_REACH``*Ly; its displacement xi is its y minus the z mean.  Panels:
  xi at the station against z and time; the power of xi in kz, mean over x,
  against time; the rms of xi with an exponential fit; and slices of the
  ``--density-species`` density at the station with the neutral sheet.
* ``_flux_ropes``: islands of the in-plane flux function psi (dpsi/dy = Bx,
  dpsi/dx = -By in each x-y plane).  Along the neutral sheet an O point is a
  minimum of psi times the sign of the sheet field; those that hold at least
  ``ROPE_MIN_FLUX`` B0*d_i, apart from the deepest one, the main island, are
  secondary islands, the cross-sections of flux ropes.  Panels: their number
  per x-y plane and the largest flux against time; Bz on the neutral sheet with
  the O points, in the snapshot of the largest flux; and the x-y plane through
  that O point with Bz and the contours of psi.

Each fit is the e-folding rate from the amplitude first reaching 10% to 50% of
its largest value.  The measures also go to ``_instabilities.npz``.

Energetic particles.  ``--particle-cycle CYCLE ...`` replaces the rate analysis
by the particles of the ``--particle-species`` (default: the electrons) above
the ``--particle-percentile`` P of their kinetic energy (default 80, the most
energetic 20%), read from the TPIC particle snapshot ``particles_t<CYCLE>.pvtp``
by ``--workers`` processes at once.  The energy is that of one physical
particle, v^2/(2|qom|), and a macroparticle counts for its |q|, the physical
particles it carries: the sheet species are loaded with charges that follow the
Harris profile.  A first pass over the snapshot bins the energies, which places
the percentile; a second draws about ``--particle-points`` N (default 200000) of
the particles above it, each with a probability proportional to its |q|, and
reads their positions.  The terminal states the energy of the percentile and
the speed |v| = sqrt(2*E*|qom|) it means for each mass, in c, uth of the first
``--particle-species`` and v_A.  The plots, with energies in T0 = m*uth^2 of
that species:

* ``_particles_3d_p<P>_t<CYCLE>``: the particles drawn as points in the 3D view
  of the box, coloured by energy, with the camera and fonts of ``RENDER_STYLE``;
* ``_particles_p<P>_t<CYCLE>``: their projections on the x-y, z-y and x-z
  planes and the energy distribution of each species, with the threshold.

The sample goes to ``_particles_t<CYCLE>.npz`` with the binned energies of every
particle, and serves again without reading the snapshot while the snapshot and
the species are the same, P is not lower and N of its particles lie above P:
those drawn above a higher percentile are a fair sample of the particles above
it, and a random subset of N of them is too.  A sample drawn at a low P with a
large N thereby serves every higher P.

``--render-cycle CYCLE ... --render-particles P ...`` draws instead
``particle_overlay_points`` of the particles above each P over the 3D view, as
spheres coloured by energy in the greys of ``particle_overlay_colors``, apart
from the colours of the density and of the field lines, into
``<image name>_particles_p<P>_t<CYCLE>``; a thinner volume
(``opacity_unit_distance`` of 2 or more) lets those inside the sheet show.  The sample is
``<deck name>_particles_t<CYCLE>.npz`` beside the image, drawn first when it
does not serve.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Iterator

import _bootstrap  # source-checkout imports
from tpic_tools.inputs import periodic_axes, read_input
from tpic_tools.fields import field_data_exists, open_field_reader

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np


@dataclass
class FontSizes:
    """Font sizes of every plot, in points."""

    title: float = 12.0
    axis_label: float = 22.0
    tick_label: float = 20.0
    legend: float = 18.0


# Colors, markers, line styles and line widths (in points) of the runs in every plot, in the order of the runs, which
# is also their order in the legends; a longer list of runs repeats them.  An earlier run is drawn over a later one:
# the first is a thin dark blue line with dots and the second a wide light purple one, which stays visible around
# the first where the two agree.  The other colors are Matplotlib's.
RUN_COLORS = ("#1f4a9e", "#d3a9ee", "C1", "C2", "C3", "C5", "C6", "C7", "C8", "C9")
RUN_MARKERS = ("o", "")  # "" draws none
RUN_LINE_STYLES = ("-", "-", "--", "-.")
RUN_LINE_WIDTHS = (1.3, 4.5)
# A curve carries about this many markers, of this size in points.
MARKERS_PER_CURVE = 25
MARKER_SIZE = 4.0


@dataclass
class LinePlotStyle:
    """Look of a plot of one quantity of every run; a tuple holds one entry per run and repeats when it is shorter."""

    y_label: str
    figure_size: tuple[float, float] = (8.0, 5.0)  # inches
    colors: tuple[str, ...] = RUN_COLORS
    line_styles: tuple[str, ...] = RUN_LINE_STYLES
    line_widths: tuple[float, ...] = RUN_LINE_WIDTHS
    markers: tuple[str, ...] = RUN_MARKERS  # ("",) draws none
    marker_size: float = MARKER_SIZE
    markers_per_curve: int = MARKERS_PER_CURVE
    legend: bool = True
    legend_location: str = "best"
    x_label: str = r"$\Omega_{ci} t$"  # "" leaves the axis unlabelled
    title: bool = True  # the time-average title of a run with --time-average


FONT_SIZES = FontSizes()
# A plot takes another look from further fields here, as in LinePlotStyle(y_label=..., markers=("",)).
FLUX_RATE_PLOT_STYLE = LinePlotStyle(y_label=r"$\dot\psi/(B_0 v_A)$")
ENERGY_PLOT_STYLE = LinePlotStyle(y_label=r"$(E - E_0)/E_0$")

# ParaView's "Rainbow Uniform" colormap: (position in [0, 1], RGB in [0, 255]).
RAINBOW_UNIFORM = (
    (0.0, (5, 97, 255)), (0.091, (5, 137, 225)), (0.203, (5, 178, 175)), (0.287, (5, 212, 126)),
    (0.357, (4, 236, 78)), (0.385, (68, 243, 40)), (0.427, (150, 248, 18)), (0.503, (255, 254, 5)),
    (0.573, (255, 222, 53)), (0.706, (255, 154, 55)), (0.797, (255, 99, 55)), (0.860, (253, 40, 53)),
    (0.923, (224, 15, 80)), (1.0, (177, 13, 104)),
)


@dataclass
class RenderStyle:
    """Look of the 3D view of --render-cycle; lengths are in d_i and angles in degrees."""

    window_size: tuple[int, int] = (1980, 1050)  # pixels
    margin: int | None = 20  # pixels of background kept around the drawing; None keeps the whole window
    background: str = "white"
    text_color: str = "black"
    colormap: tuple[tuple[float, tuple[int, int, int]], ...] = RAINBOW_UNIFORM
    density_label: str = "rho_i"
    density_range: tuple[float, float] | None = None  # None: from 0 to the maximum of the snapshot
    # A PyVista opacity name ("linear", "sigmoid", ...) or the opacities at evenly spaced densities of the range.
    opacity: str | tuple[float, ...] = "linear"
    opacity_unit_distance: float = 0.5  # the depth over which a sample takes the opacity of its density
    # The field lines of the x-line view start in seed_planes planes of z, at inflow_seeds above and below each
    # X line, on field lines still to reconnect, and on the neutral sheet at outflow_seeds either side of it, on
    # reconnected field lines; distances in d_i from the X line.
    seed_planes: int = 3
    inflow_seeds: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    outflow_seeds: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)
    line_length: float = 10.0  # traced each way from a seed
    line_radius: float = 0.09
    line_color: str | None = "#10202c"  # one colour for all the field lines; None colours them by Bz
    # The zoomed views of --render-view kink, lower-hybrid and ropes.  Without --render-region a zoom spans these
    # half-sizes, in fractions of Lx, Ly and Lz, about the kink station on the sheet (kink, lower-hybrid) or the O
    # point of most flux (ropes).
    kink_zoom: tuple[float, float, float] = (1 / 6, 1 / 8, 1 / 2)
    lower_hybrid_zoom: tuple[float, float, float] = (1 / 12, 1 / 10, 1 / 6)
    ropes_zoom: tuple[float, float, float] = (1 / 6, 1 / 8, 1 / 4)
    zoom_camera: tuple[float, float] = (40.0, 20.0)  # azimuth and elevation, as camera_azimuth and camera_elevation
    # The lower left corner, in window fractions, of the x, y, z arrows of a zoom: below the near end of the z axis,
    # left of the colour bars, where the box leaves the window empty.
    zoom_orientation_axes_corner: tuple[float, float] = (0.5, 0.0)
    zoom_opacity_unit_distance: float = 4.0  # a thinner volume, for the overlay inside the sheet to show
    lh_iso_fraction: float = 0.5  # lower-hybrid isosurfaces at this fraction of the 99.5th percentile of |dEz|
    lh_quantity: str = "dEz"  # "dEz", the lower-hybrid part of the Ez fluctuation, or "Ez" itself
    # Rope field lines start rope_seed_offset either side in y of the secondary O points of planes rope_seed_spacing
    # apart in z and are traced rope_line_length each way.
    rope_seed_spacing: float = 1.0
    rope_seed_offset: float = 0.3
    rope_line_length: float = 40.0
    rope_line_radius: float = 0.06
    rope_line_color: str | None = None  # one colour for all the rope field lines; None colours them by Bz
    # --render-inset: the zoom, without axes, spans inset_width of the width of the main view over its upper right
    # corner, rising inset_raise of its height above it; both views share colour bars inset_bar_spacing of that
    # width apart, and the box of the zoom is outlined in the main view.
    inset_width: float = 0.45
    inset_raise: float = 1.0
    inset_align: str = "right"  # the zoom at the "left" or "right" end of the main view, or centred above its "box"
    inset_shift: float = 0.0  # then moves the zoom right by this fraction of the width of the main view
    inset_camera: tuple[float, float] | None = None  # azimuth and elevation of the zoom; None: those of zoom_camera
    inset_bar_spacing: float = 0.17
    inset_outline_color: str = "black"
    colorbar_font: str = "DejaVu Sans"
    inset_outline_width: float = 3.0  # the frame of the zoom
    inset_box_line_width: float = 1.5  # the box of the zoom outlined in the main view, and the lines joining them
    camera_azimuth: float = 40.0  # about y, from +z toward +x
    camera_elevation: float = 11.0  # above the x-z plane
    camera_distance: float = 1.7  # from the box centre, in box diagonals
    # VTK camera window centre: a positive x moves the box left and a positive y down.
    window_center: tuple[float, float] = (0.25, -0.04)
    axis_font_size: int = 30
    title_font_size: int = 14
    colorbar_position: tuple[float, float] = (0.745, 0.1)  # lower left corner, in fractions of the window
    colorbar_spacing: float = 0.12  # between the lower left corners of successive colour bars, in window fractions
    overlay_colorbar_shift: float = 0.03  # moves both colour bars of a view with an overlay right, clear of the box
    overlay_colorbar: bool = True  # the colour bar of the overlay, Bz or the lower-hybrid dEz
    colorbar_title_size: int = 40
    colorbar_label_size: int = 34
    orientation_axes_size: float = 0.25  # side of the corner that holds the x, y, z arrows, in window fractions
    title: bool = False  # the run, the cycle and Omega_ci*t at the top left
    # The energetic particles of --particle-cycle: the 3D view draws them as spheres of particle_point_size pixels,
    # the projections as dots of particle_dot_size square points, both coloured by energy over particle_energy_range
    # (E/T0); None spans the selection threshold to the 99.9th percentile of the energies drawn.
    particle_point_size: float = 3.0
    particle_dot_size: float = 0.5
    particle_energy_range: tuple[float, float] | None = None
    # --render-particles: particle_overlay_points of them over a 3D view, as spheres of particle_overlay_size pixels
    # and opacity particle_overlay_opacity, coloured by energy along particle_overlay_colors, greys apart from the
    # colours of the density and of the field lines.  VTK draws translucent points after the volume, which then hides
    # them where it is dense; opaque points stay sorted in depth with it, and a thinner volume (opacity_unit_distance
    # of 2 or more) lets those inside the sheet show.
    particle_overlay_points: int = 20000
    particle_overlay_size: float = 4.0
    particle_overlay_opacity: float = 1.0
    particle_overlay_colors: tuple[str, ...] = ("#a0a0a0", "#000000")


RENDER_STYLE = RenderStyle()

plt.rcParams.update({
    "figure.titlesize": FONT_SIZES.title, "axes.titlesize": FONT_SIZES.title,
    "axes.labelsize": FONT_SIZES.axis_label,
    "xtick.labelsize": FONT_SIZES.tick_label, "ytick.labelsize": FONT_SIZES.tick_label,
    "legend.fontsize": FONT_SIZES.legend,
})


@dataclass
class Run:
    """One analyzed run; flux and rates have shape (snapshot, z node)."""

    label: str
    input_path: Path
    case: str
    layout: str
    field_unit: float  # B0*v_A
    alfven_speed: float
    omega_ci: float
    ion_length: float
    x_line: tuple[float, float]
    time_ci: np.ndarray  # Omega_ci*t
    z: np.ndarray  # z/d_i
    flux: np.ndarray  # psi/(B0*d_i)
    flux_rate: np.ndarray  # (d psi/dt)/(B0*v_A)
    field_time_ci: np.ndarray  # Omega_ci*t of field_rate
    field_rate: np.ndarray  # -Ez/(B0*v_A), box average
    unaveraged_flux_rate: np.ndarray  # flux_rate before the time average
    unaveraged_field_rate: np.ndarray  # field_rate before the time average

    def time_of(self, member: str) -> np.ndarray:
        """Return Omega_ci*t of the samples of flux or of a rate."""
        return self.field_time_ci if member.endswith("field_rate") else self.time_ci


@dataclass
class EnergyHistory:
    """Total energy of the run of one deck, from the scalar log of its output directory."""

    label: str
    time_ci: np.ndarray  # Omega_ci*t
    total: np.ndarray  # in the units of the log


# The X lines of a case: the name of the sheet, x/Lx, y/Ly and the sign of the
# sheet field relative to the sheet amplitude.
SINGLE_SHEET = (("", 0.5, 0.5, 1.0),)
DOUBLE_SHEET = (("lower sheet", 0.25, 0.25, 1.0), ("upper sheet", 0.75, 0.75, -1.0))
X_LINES = {"GEM": SINGLE_SHEET, "GEMnoPert": SINGLE_SHEET, "GEMHarris": SINGLE_SHEET,
           "DoubleGEM": DOUBLE_SHEET, "DoubleGEMnoPert": DOUBLE_SHEET}


def find_input_path(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    decks = sorted(run_dir.glob("*.inp"))
    if len(decks) != 1:
        raise FileNotFoundError(
            f"{run_dir} holds {len(decks)} input decks; pass --input explicitly"
        )
    return decks[0]


def first_float(values: dict[str, list[str]], key: str) -> float:
    return float(values[key][0])


def ion_species(params: dict[str, list[str]], density: str, charge: float = 1.0) -> list[int]:
    """Return the indices of the sheet or the background ion species, or electron species with a negative charge."""
    count = int(first_float(params, "ns"))
    drifts = [[float(token) for token in params[key][:count]] for key in ("u0", "v0", "w0")]
    # A species carries the sheet density when it drifts (the rule of the GEM case).
    return [index for index, qom in enumerate(params["qom"][:count])
            if float(qom) * charge > 0.0
            and any(abs(drift[index]) > 1.0e-10 for drift in drifts) == (density == "sheet")]


def sheet_normalization(
    params: dict[str, list[str]], amplitude: float, density: str
) -> tuple[float, float, float]:
    """Return (v_A, Omega_ci, d_i) for the sheet or the background ions."""
    b0 = abs(amplitude)
    if b0 == 0.0:
        raise ValueError("the sheet amplitude is zero: there is no reconnecting field")
    ions = [(float(params["rhoINIT"][index]), float(params["qom"][index]))
            for index in ion_species(params, density)]
    if not ions:
        raise ValueError(f"no {density} ion species: the Alfven speed is undefined")
    alfven_speed = b0 / math.sqrt(sum(rho / qom for rho, qom in ions))
    omega_ci = ions[0][1] * b0
    return alfven_speed, omega_ci, alfven_speed / omega_ci


def at_node_position(values: np.ndarray, position: float, axis: int) -> np.ndarray:
    """Interpolate nodal values linearly at a fractional node index along axis."""
    lower = min(int(position), values.shape[axis] - 2)
    weight = position - lower
    return (1.0 - weight) * np.take(values, lower, axis) + weight * np.take(values, lower + 1, axis)


def box_average(values: np.ndarray, x_node: float, y_node: float, cells: int) -> np.ndarray:
    """Average nodal values of shape (z, y, x) over a square of cells x cells centred on the X line."""
    offsets = np.arange(cells + 1) - 0.5 * cells
    weights = np.ones(cells + 1)
    if cells:
        weights[[0, -1]] = 0.5  # trapezoidal rule: an edge node bounds half a cell
    weights /= weights.sum()
    rows = sum(w * at_node_position(values, y_node + o, axis=1) for w, o in zip(weights, offsets))
    return sum(w * at_node_position(rows, x_node + o, axis=1) for w, o in zip(weights, offsets))


def reconnected_flux(by: np.ndarray, dx: float, x_node: float, y_node: float) -> np.ndarray:
    """Return psi(z) of the module docstring from nodal By of shape (z, y, x)."""
    by_sheet = at_node_position(by, y_node, axis=1)
    # Flux function F(x) = int_0^x By dx' at the nodes, by the trapezoidal rule.
    steps = 0.5 * dx * (by_sheet[:, 1:] + by_sheet[:, :-1])
    flux_function = np.concatenate([np.zeros_like(steps[:, :1]), np.cumsum(steps, axis=1)], axis=1)
    # The O line is half a period from the X line; F gains F(Lx) per period.
    cells = by_sheet.shape[1] - 1
    o_node = (x_node + 0.5 * cells) % cells
    return (at_node_position(flux_function, o_node, axis=1) - at_node_position(flux_function, x_node, axis=1)
            + math.copysign(0.5, x_node - o_node) * flux_function[:, -1])


def read_structured_points(path: Path, component: int | slice) -> np.ndarray:
    """Return one component, shape (z, y, x), of the vector or scalar field of a binary structured-points VTK file.

    A slice of components returns shape (z, y, x, component).
    """
    raw = path.read_bytes()
    header = raw[:1024]
    dimensions = re.search(rb"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", header)
    field = re.search(rb"(VECTORS|SCALARS)\s+\S+\s+(float|double)[^\n]*\n(LOOKUP_TABLE[^\n]*\n)?", header)
    if b"BINARY" not in header or b"STRUCTURED_POINTS" not in header or not dimensions or not field:
        raise ValueError(f"{path}: expected a binary STRUCTURED_POINTS file with one vector or scalar field")
    nx, ny, nz = map(int, dimensions.groups())
    components = 3 if field.group(1) == b"VECTORS" else 1
    # The format stores big-endian values, x fastest, components interleaved.
    values = np.frombuffer(raw, dtype=">f4" if field.group(2) == b"float" else ">f8",
                           count=components * nx * ny * nz, offset=field.end())
    return values.reshape(nz, ny, nx, components)[..., component].astype(float)


# Time of By, time of Ez, then By and Ez of shape (z, y, x).
Snapshots = Iterator[tuple[float, float, np.ndarray, np.ndarray]]


def tpic_snapshots(data_dir: Path, max_cycle: float) -> Snapshots:
    with open_field_reader(data_dir / "B") as b_reader, open_field_reader(data_dir / "E") as e_reader:
        # The records of a reader are sorted by cycle, the first entry of their epoch.
        if [r[1] for r in b_reader.records] != [r[1] for r in e_reader.records]:
            raise ValueError(f"{data_dir}: the B and E files hold different snapshots")
        for step, (_, epoch) in enumerate(b_reader.records):
            if epoch[0] > max_cycle:
                break
            time = b_reader.read("time", step_selection=[step, 1])[0]
            yield (time, time,
                   b_reader.read("By", step_selection=[step, 1]),
                   e_reader.read("Ez", step_selection=[step, 1]))


def structured_points_snapshots(data_dir: Path, name: str, dt: float, max_cycle: float) -> Snapshots:
    cycles = sorted(int(path.stem.rsplit("_", 1)[1]) for path in data_dir.glob(f"{name}_B_*.vtk"))
    for cycle in cycles:
        if cycle > max_cycle:
            break
        yield (cycle * dt, (cycle + 1) * dt,
               read_structured_points(data_dir / f"{name}_B_{cycle}.vtk", component=1),
               read_structured_points(data_dir / f"{name}_E_{cycle}.vtk", component=2))


def read_energy_log(data_dir: Path, dt: float, max_cycle: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the times and the total energy of the scalar log of data_dir up to max_cycle, or None without a log."""
    # File, column separator (None: blanks), cycle column, time column and its unit, energy column.
    for name, separator, cycle_column, time_column, time_unit, energy_column in (
            ("diagnostics.csv", ",", "cycle", "time", 1.0, "total_energy"),
            ("ConservedQuantities.txt", None, "Cycle", "Cycle", dt, "Total_Energy")):
        if (data_dir / name).exists():
            lines = [line for line in (data_dir / name).read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.startswith("#")]
            header = lines[0].split(separator)
            rows = np.array([line.split(separator) for line in lines[1:]])
            rows = rows[rows[:, header.index(cycle_column)].astype(float) <= max_cycle]
            return (time_unit * rows[:, header.index(time_column)].astype(float),
                    rows[:, header.index(energy_column)].astype(float))
    return None


def moving_average(values: np.ndarray, time: np.ndarray, window: float) -> np.ndarray:
    """Average values of shape (snapshot, z node) over the window centred on each snapshot time.

    The mean is that of the linear interpolant; it is nan where the window leaves the run.
    """
    averaged = np.full_like(values, np.nan)
    for index, centre in enumerate(time):
        start, end = centre - 0.5 * window, centre + 0.5 * window
        # The tolerance keeps the snapshots that lie half a window from an end.
        if start < time[0] - 1.0e-9 * window or end > time[-1] + 1.0e-9 * window:
            continue
        nodes = np.concatenate([[start], time[(time > start) & (time < end)], [end]])
        samples = np.stack([np.interp(nodes, time, column) for column in values.T], axis=1)
        averaged[index] = (0.5 * (samples[1:] + samples[:-1]) * np.diff(nodes)[:, None]).sum(axis=0) / window
    return averaged


def read_history(
    snapshots: Snapshots, dx: float, x_lines: list[tuple[float, float, float]], box_cells: int, periodic_z: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the times of By and of Ez and, per snapshot, X line and z node, psi and the box average of -Ez.

    An X line is its x node, its y node and the sign of its sheet field.
    """
    # A periodic z repeats its first node at the end; count it once.
    unique_z = slice(None, -1) if periodic_z else slice(None)
    times, field_times, flux, field = [], [], [], []
    for time, field_time, by, ez in snapshots:
        times.append(time)
        field_times.append(field_time)
        flux.append([sign * reconnected_flux(by, dx, x_node, y_node)[unique_z] for x_node, y_node, sign in x_lines])
        field.append([-sign * box_average(ez, x_node, y_node, box_cells)[unique_z]
                      for x_node, y_node, sign in x_lines])
    return np.asarray(times), np.asarray(field_times), np.asarray(flux), np.asarray(field)


# The snapshot layouts and the deck key of the sheet amplitude in each.
TPIC_LAYOUT, STRUCTURED_POINTS_LAYOUT = "TPIC snapshots", "structured-points VTK"
AMPLITUDE_KEYS = {TPIC_LAYOUT: "HarrisB0x", STRUCTURED_POINTS_LAYOUT: "B0x"}


def open_deck(run_dir: Path, input_path: Path) -> tuple[dict[str, list[str]], Path, str]:
    """Return the parameters of a Harris-sheet deck, its output directory and the layout of its snapshots."""
    params = read_input(input_path)
    case = params["Case"][0]
    if case not in X_LINES:
        raise ValueError(f"{input_path}: Case = {case}; only the Harris-sheet cases {', '.join(X_LINES)} are analyzed")
    if float(params.get("c", ["1"])[0]) != 1.0:
        raise ValueError(f"{input_path}: the normalization assumes c = 1")

    configured = Path(params["SaveDirName"][0])
    data_dir = configured if configured.is_absolute() else run_dir / configured
    name = params.get("SimulationName", [""])[0]
    if field_data_exists(data_dir, "B"):
        return params, data_dir, TPIC_LAYOUT
    if name and any(data_dir.glob(f"{name}_B_*.vtk")):
        return params, data_dir, STRUCTURED_POINTS_LAYOUT
    raise FileNotFoundError(f"{data_dir}: no B_t*.vti and no {name or '<SimulationName>'}_B_*.vtk")


def analyze_deck(
    run_dir: Path, input_path: Path, label: str, box_cells: int, density: str, window: float, max_cycle: float
) -> tuple[list[Run], EnergyHistory | None]:
    """Return one run per X line of the case of the deck, and the energy history of the deck when it has a log.

    The run ends at max_cycle; math.inf keeps all of it.
    """
    params, data_dir, layout = open_deck(run_dir, input_path)
    case = params["Case"][0]
    if layout == TPIC_LAYOUT:
        snapshots = tpic_snapshots(data_dir, max_cycle)
    else:
        snapshots = structured_points_snapshots(data_dir, params["SimulationName"][0], first_float(params, "dt"),
                                                max_cycle)

    amplitude = first_float(params, AMPLITUDE_KEYS[layout])
    alfven_speed, omega_ci, ion_length = sheet_normalization(params, amplitude, density)
    field_unit = abs(amplitude) * alfven_speed
    lx, ly = first_float(params, "Lx"), first_float(params, "Ly")
    nxc, nyc = int(first_float(params, "nxc")), int(first_float(params, "nyc"))
    if not 0 <= box_cells <= min(nxc, nyc):
        raise ValueError(f"{input_path}: --box-cells must lie in [0, {min(nxc, nyc)}]")
    periodic_x, _, periodic_z = periodic_axes(params)
    if not periodic_x:
        raise ValueError(f"{input_path}: the reconnected flux takes the O line half a period in x from the X line, "
                         "so the box must be periodic in x")

    x_lines = X_LINES[case]
    time, field_time, all_flux, all_field = read_history(
        snapshots, lx / nxc,
        [(x * nxc, y * nyc, sign * math.copysign(1.0, amplitude)) for _, x, y, sign in x_lines], box_cells, periodic_z)
    if time.size < 2:
        raise ValueError(f"{data_dir}: a rate needs at least two field snapshots"
                         + (f" up to cycle {max_cycle}" if math.isfinite(max_cycle) else ""))
    cell_z = first_float(params, "Lz") / int(first_float(params, "nzc"))
    runs = []
    for index, (sheet, x, y, _) in enumerate(x_lines):
        flux, field = all_flux[:, index], all_field[:, index]
        unaveraged = [np.gradient(flux, time, axis=0) / field_unit, field / field_unit]
        rates = unaveraged
        if window > 0.0:
            rates = [moving_average(rate, omega_ci * axis, window)
                     for rate, axis in zip(unaveraged, (time, field_time))]
            if np.isnan(rates[0]).all():
                raise ValueError(f"{data_dir}: --time-average {window:g} exceeds the Omega_ci*t span "
                                 f"{omega_ci * (time[-1] - time[0]):g} of the run")
        runs.append(Run(
            label=f"{label}, {sheet}" if sheet else label, input_path=input_path, case=case, layout=layout,
            field_unit=field_unit, alfven_speed=alfven_speed, omega_ci=omega_ci, ion_length=ion_length,
            x_line=(x * lx, y * ly),
            time_ci=omega_ci * time,
            z=cell_z * np.arange(flux.shape[1]) / ion_length,
            flux=flux / (field_unit / omega_ci),  # B0*d_i = B0*v_A/Omega_ci
            flux_rate=rates[0],
            field_time_ci=omega_ci * field_time,
            field_rate=rates[1],
            unaveraged_flux_rate=unaveraged[0],
            unaveraged_field_rate=unaveraged[1],
        ))
    energy_log = read_energy_log(data_dir, first_float(params, "dt"), max_cycle)
    return runs, EnergyHistory(label, omega_ci * energy_log[0], energy_log[1]) if energy_log else None


def difference_to_first(first: Run, run: Run, member: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the times of member in first that run covers and, there, the z average of member of run minus first."""
    time, run_time = first.time_of(member), run.time_of(member)
    inside = (time >= run_time[0]) & (time <= run_time[-1])
    time = time[inside]
    return time, (np.interp(time, run_time, getattr(run, member).mean(axis=1))
                  - getattr(first, member).mean(axis=1)[inside])


def flux_gain(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """Return, from the minimum of the z-averaged flux on, Omega_ci*t and the flux gained since."""
    flux = run.flux.mean(axis=1)
    start = int(np.argmin(flux))
    return run.time_ci[start:], flux[start:] - flux[start]


def curve_look(index: int, points: int, colors: tuple[str, ...] = RUN_COLORS,
               markers: tuple[str, ...] = RUN_MARKERS, line_styles: tuple[str, ...] = RUN_LINE_STYLES,
               line_widths: tuple[float, ...] = RUN_LINE_WIDTHS, size: float = MARKER_SIZE,
               per_curve: int = MARKERS_PER_CURVE) -> dict:
    """Return the Matplotlib color, marker, line style, line width and drawing order of curve index of a plot."""
    return {"color": colors[index % len(colors)], "marker": markers[index % len(markers)], "markersize": size,
            "markevery": max(1, points // per_curve), "linestyle": line_styles[index % len(line_styles)],
            "linewidth": line_widths[index % len(line_widths)], "zorder": 3.0 - 0.01 * index}


def with_pdf(path: Path) -> str:
    """Return the path for the terminal, with the PDF that save_figure and save_image write beside a PNG."""
    return f"{path} and {path.with_suffix('.pdf').name}" if path.suffix.lower() == ".png" else str(path)


def save_figure(fig, path: Path, **options) -> None:
    """Save fig to path and, when path is a PNG, to a PDF of the same name beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **options)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), **options)


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an RGB(A) image to path and, when path is a PNG, to a PDF of the same name that embeds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ascontiguousarray(image)
    plt.imsave(path, image)
    if path.suffix.lower() == ".png":
        plt.imsave(path.with_suffix(".pdf"), image)


def write_csv(path: Path, runs: list[Run], window: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["run", "omega_ci_t", "flux", "flux_z_std", "flux_rate", "flux_rate_z_std",
                         "field_omega_ci_t", "field_rate", "field_rate_z_std",
                         "unaveraged_flux_rate", "unaveraged_flux_rate_z_std",
                         "unaveraged_field_rate", "unaveraged_field_rate_z_std", "rate_average_window"])
        for run in runs:
            columns = []
            for member in ("flux", "flux_rate", "field_rate", "unaveraged_flux_rate", "unaveraged_field_rate"):
                values = getattr(run, member)
                # The flux rates share the time column of the flux, the unaveraged Ez rate that of the Ez rate.
                columns += ([run.time_of(member)] * (member in ("flux", "field_rate"))
                            + [values.mean(axis=1), values.std(axis=1)])
            for row in zip(*columns):
                writer.writerow([run.label, *(f"{value:.17g}" for value in row), f"{window:g}"])


def write_plot(path: Path, runs: list[Run], box_cells: int, window: float) -> None:
    single = len(runs) == 1
    # The fourth panel is the map along z of one run with more than one z node, or the comparison of several.
    with_map = single and runs[0].z.size > 1
    fourth = with_map or not single
    fig, axes = plt.subplots(4 if fourth else 3, 1, figsize=(8, 12.5 if fourth else 10),
                             sharex=True, constrained_layout=True)
    panels = (
        ("flux", r"reconnected flux $\psi/(B_0 d_i)$"),
        ("flux_rate", r"reconnection rate $\dot\psi/(B_0 v_A)$"),
        ("field_rate", rf"$-E_z/(B_0 v_A)$, {box_cells}x{box_cells}-cell box"),
    )
    for axis, (member, ylabel) in zip(axes, panels):
        for index, run in enumerate(runs):
            values = getattr(run, member)
            mean, spread = values.mean(axis=1), values.std(axis=1)
            look = curve_look(index, mean.size)
            # The z spread of the box average is its noise along z, not the uncertainty of the mean.
            if member != "field_rate":
                axis.fill_between(run.time_ci, mean - spread, mean + spread, color=look["color"], alpha=0.25)
            axis.plot(run.time_of(member), mean, **look,
                      label="z average" if single else run.label)
        if member != "flux":
            axis.axhline(0.0, color="0.4", linewidth=0.8)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    if single:
        run = runs[0]
        axes[2].plot(run.time_ci, run.flux_rate.mean(axis=1), color="C3", label=r"$\dot\psi/(B_0 v_A)$, z average")
    if with_map:
        run = runs[0]
        # The relaxation of the initial perturbation would otherwise set the colour range.
        limit = np.nanpercentile(np.abs(run.flux_rate), 90.0) or 1.0
        mesh = axes[3].pcolormesh(run.time_ci, run.z, run.flux_rate.T, cmap="RdBu_r",
                                  vmin=-limit, vmax=limit, shading="nearest")
        fig.colorbar(mesh, ax=axes[3], label=r"$\dot\psi/(B_0 v_A)$ along the X line")
        axes[3].set_ylabel(r"$z/d_i$")
    for index, run in enumerate(runs[1:], start=1):
        # Here the line style tells the two rates apart, whatever that of the run.
        for member, style, name in (("flux_rate", "-", r"$\dot\psi$"), ("field_rate", "--", r"$-E_z$")):
            time, difference = difference_to_first(runs[0], run, member)
            axes[3].plot(time, difference, **{**curve_look(index, time.size), "linestyle": style},
                         label=f"{run.label}, {name}")
    if not single:
        axes[3].axhline(0.0, color="0.4", linewidth=0.8)
        axes[3].set_ylabel(rf"rate $/(B_0 v_A)$ minus that of {runs[0].label}")
        axes[3].grid(True, alpha=0.3)
    for axis in axes[:3] if with_map else axes:
        axis.legend()
    axes[-1].set_xlabel(r"$\Omega_{ci} t$")
    if window > 0.0:
        fig.suptitle(rf"rates: mean over $\Omega_{{ci}} \Delta t = {window:g}$ centred on each snapshot")

    save_figure(fig, path, dpi=200)
    plt.close(fig)


def write_line_plot(path: Path, curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]],
                    title: str, style: LinePlotStyle) -> None:
    """Plot one quantity of every run; a curve is a label, Omega_ci*t, the values and their spread or None."""
    fig, axis = plt.subplots(figsize=style.figure_size, constrained_layout=True)
    for index, (label, time, values, spread) in enumerate(curves):
        look = curve_look(index, time.size, style.colors, style.markers, style.line_styles, style.line_widths,
                          style.marker_size, style.markers_per_curve)
        if spread is not None:
            # Without an edge a band of no spread draws nothing under a dashed line.
            axis.fill_between(time, values - spread, values + spread, color=look["color"], alpha=0.25,
                              linewidth=0.0)
        axis.plot(time, values, **look, label=label)
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.set_xlabel(style.x_label)
    axis.set_ylabel(style.y_label)
    axis.grid(True, alpha=0.3)
    if style.legend:
        axis.legend(loc=style.legend_location)
    if style.title and title:
        axis.set_title(title)

    save_figure(fig, path, dpi=200)
    plt.close(fig)


def read_snapshot(
    data_dir: Path, layout: str, params: dict[str, list[str]], cycle: int, species: list[int], electric: bool = False
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return the time of cycle, the summed density of species, shape (z, y, x), B, shape (z, y, x, 3), and E alike.

    E is None unless electric; a structured-points E is one step later than B.
    """
    if layout == STRUCTURED_POINTS_LAYOUT:
        name = params["SimulationName"][0]
        paths = [data_dir / f"{name}_B_{cycle}.vtk", data_dir / f"{name}_E_{cycle}.vtk",
                 *(data_dir / f"{name}_rho{s}_{cycle}.vtk" for s in species)]
        for path in paths[:1] + paths[2 - electric:]:
            if not path.exists():
                raise FileNotFoundError(f"{data_dir}: no {path.name}")
        return (cycle * first_float(params, "dt"), sum(read_structured_points(path, 0) for path in paths[2:]),
                read_structured_points(paths[0], slice(None)),
                read_structured_points(paths[1], slice(None)) if electric else None)

    def read(field: str, names: tuple[str, ...]) -> tuple[float, list[np.ndarray]]:
        with open_field_reader(data_dir / field) as reader:
            cycles = [epoch[0] for _, epoch in reader.records]
            if cycle not in cycles:
                raise FileNotFoundError(f"{data_dir}: no {field}_t*.vti of cycle {cycle}")
            step = cycles.index(cycle)
            return reader.read("time", [step, 1])[0], [reader.read(name, [step, 1]) for name in names]

    time, b = read("B", ("Bx", "By", "Bz"))
    e = np.stack(read("E", ("Ex", "Ey", "Ez"))[1], axis=-1) if electric else None
    density = sum(read(f"rho_{s}", (f"rho_{s}",))[1][0] for s in species)
    return time, density, np.stack(b, axis=-1), e


def grid_axes(bounds: np.ndarray, view: np.ndarray, color: tuple[float, float, float], font_size: int):
    """Return the axes grid of ParaView: the edges of the three faces of a box away from view, labelled in d_i.

    bounds holds the low and high x, y and z of the box, shape (3, 2); view points from its centre toward the camera.
    """
    import vtkmodules.vtkRenderingContextOpenGL2  # noqa: F401, draws the labels
    from vtkmodules.vtkRenderingCore import vtkProperty
    from vtkmodules.vtkRenderingGridAxes import vtkGridAxesActor3D

    axes = vtkGridAxesActor3D()
    axes.SetGridBounds(*bounds.ravel())
    # The faces at the minimum x, y and z are bits 1, 2 and 4 of the mask and those at the maximum 8, 16 and 32.
    axes.SetFaceMask(sum((1 if view[axis] > 0.0 else 8) << axis for axis in range(3)))
    axes.SetGenerateGrid(False)
    for axis, (name, (low, high)) in enumerate(zip("xyz", bounds)):
        # The step of 1, 2 or 5 times a power of ten just above a sixth of the side.
        rough = (high - low) / 6.0
        power = 10.0 ** math.floor(math.log10(rough))
        step = next(factor * power for factor in (1.0, 2.0, 5.0, 10.0) if factor * power >= rough)
        ticks = step * np.arange(math.ceil(low / step - 1.0e-9), math.floor(high / step + 1.0e-9) + 1)
        # A title sits on the middle tick; the blanks keep its label clear on whichever side it is justified.
        axes.SetTitle(axis, f"\n    {name} (d_i)    \n")
        axes.SetNumberOfLabels(axis, ticks.size)
        for index, tick in enumerate(ticks):
            axes.SetLabel(axis, index, tick)
        axes.SetUseCustomLabels(axis, True)
        for text in (axes.GetTitleTextProperty(axis), axes.GetLabelTextProperty(axis)):
            text.SetColor(color)
            text.SetFontSize(font_size)
    # A new property reaches the edges of every face; the default one colors only the first.
    edges = vtkProperty()
    edges.SetColor(color)
    axes.SetProperty(edges)
    return axes


def lower_hybrid_length(params: dict[str, list[str]], amplitude: float, ion_length: float) -> float:
    """Return sqrt(rho_e*rho_i) in d_i, rho = uth/(|qom|*B0) of the first electron and ion species of the sheet, or
    of the background without one."""
    electrons = ion_species(params, "sheet", -1.0) or ion_species(params, "background", -1.0)
    ions = ion_species(params, "sheet") or ion_species(params, "background")
    if not electrons:
        raise ValueError("no electron species: the lower-hybrid scale is undefined")
    gyroradius = [float(params["uth"][s]) / abs(float(params["qom"][s]) * amplitude) for s in (electrons[0], ions[0])]
    return math.sqrt(gyroradius[0] * gyroradius[1]) / ion_length


# The views of --render-view: the field lines through the X lines, or one instability of the sheet.
RENDER_VIEWS = ("x-line", "kink", "lower-hybrid", "ropes")


def zoom_region(centre: np.ndarray, half: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return the box, shape (3, 2), of half-sizes half about centre, moved inside [0, lengths] and cut to it."""
    half = np.minimum(half, 0.5 * lengths)
    low = np.clip(centre - half, 0.0, lengths - 2.0 * half)
    return np.stack([low, low + 2.0 * half], axis=1)


def render_snapshot(path: Path, run_dir: Path, input_path: Path, label: str, cycle: int, species: list[int] | None,
                    density_ions: str, style: RenderStyle, view: str = "x-line", region: np.ndarray | None = None,
                    inset: np.ndarray | None = None, inset_view: str | None = None,
                    particles: dict | None = None) -> tuple[float, float]:
    """Render the view of render_image into path, with a zoom on the box inset (d_i, shape (3, 2)) when given.

    The zoom shows inset_view, by default view, with the colour scales and the overlay level of the main view, which
    outlines its box, and the two share one set of colour bars.  particles, from energetic_particles with its colour
    range added as "range", go over both.  Return Omega_ci*t and the largest density shown.
    """
    if inset is None:
        image, scales = render_image(run_dir, input_path, label, cycle, species, density_ions, style, view, region,
                                     particles=particles)
        save_image(path, image)
        return scales["time"], scales["shown"]
    image, scales = render_image(run_dir, input_path, label, cycle, species, density_ions, style, view, region,
                                 colorbars=False, outline=inset, particles=particles)
    # The zoom shows the data alone: the outlined box in the main view places it.
    detail_style = style if style.inset_camera is None else replace(style, zoom_camera=style.inset_camera)
    detail, _ = render_image(run_dir, input_path, label, cycle, species, density_ions, detail_style, inset_view or view,
                             inset, colorbars=False, density_range=scales["density_range"],
                             overlay_limit=scales["overlay"] and scales["overlay"][1], orientation_axes=False,
                             axes_grid=False, particles=particles)
    write_inset_figure(path, image, detail, scales, style)
    return scales["time"], scales["shown"]


def write_inset_figure(path: Path, image: np.ndarray, detail: np.ndarray, scales: dict, style: RenderStyle) -> None:
    """Save the main view, the zoom detail above its right end joined to the outlined box, and the shared colour
    bars at the right."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    height, width = image.shape[:2]
    inset_width = style.inset_width * width
    inset_height = inset_width * detail.shape[0] / detail.shape[1]
    # The zoom rises inset_raise of its height above the main view, whose upper right corner is mostly empty.
    top = height + style.inset_raise * inset_height
    bars = [(style.density_label, LinearSegmentedColormap.from_list(
        "density", [(p, np.array(rgb) / 255.0) for p, rgb in style.colormap]), Normalize(*scales["density_range"]))]
    if scales["overlay"] and style.overlay_colorbar:
        title, limit = scales["overlay"]
        bars.append((title, plt.get_cmap("coolwarm"), Normalize(-limit, limit)))
    if scales["particles"]:
        bars.append(("E/T0", LinearSegmentedColormap.from_list("particles", style.particle_overlay_colors),
                     LogNorm(*scales["particles"])))
    slot = style.inset_bar_spacing * width
    right = width + slot * len(bars)
    fig = plt.figure(figsize=(right / 100.0, top / 100.0), dpi=100)
    main = fig.add_axes((0.0, 0.0, width / right, height / top))
    main.imshow(image)
    main.set_axis_off()
    left = width - inset_width
    if style.inset_align == "left":
        left = 0.0
    elif style.inset_align == "box" and scales["outline"] is not None:
        upper = scales["outline"][scales["outline"][:, 2] == scales["outline"][:, 2].max()]
        left = float(np.clip(0.5 * (upper[:, 0].min() + upper[:, 0].max()) - 0.5 * inset_width, 0.0, left))
    elif style.inset_align != "right":
        raise ValueError(f"inset_align = {style.inset_align!r}: expected 'left', 'right' or 'box'")
    left = float(np.clip(left + style.inset_shift * width, 0.0, width - inset_width))
    box = (left / right, (top - inset_height) / top, inset_width / right, inset_height / top)
    zoom = fig.add_axes(box)
    zoom.imshow(detail)
    zoom.set_xticks([])
    zoom.set_yticks([])
    for spine in zoom.spines.values():
        spine.set_color(style.inset_outline_color)
        spine.set_linewidth(style.inset_outline_width)
    if scales["outline"] is not None:
        # Join the lower corners of the zoom to the leftmost and rightmost upper corners of the outlined box.
        upper = scales["outline"][scales["outline"][:, 2] == scales["outline"][:, 2].max()]
        for corner, x in ((upper[np.argmin(upper[:, 0])], box[0]), (upper[np.argmax(upper[:, 0])], box[0] + box[2])):
            fig.add_artist(Line2D([x, corner[0] / right], [box[1], (height - corner[1]) / top],
                                  color=style.inset_outline_color, linewidth=style.inset_box_line_width))
    # Colour bars in points of a 100 dpi figure, the size of those of the 3D view in pixels.
    points = 72.0 / 100.0
    for index, (title, cmap, norm) in enumerate(bars):
        axis = fig.add_axes(((width + slot * (index + 0.1)) / right, 0.1 * height / top, 0.25 * slot / right,
                             0.75 * height / top))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=axis, format="%.2g")
        if index and not isinstance(norm, LogNorm):
            bar.set_ticks([norm.vmin, 0.0, norm.vmax])
        bar.ax.tick_params(labelsize=style.colorbar_label_size * points, colors=style.text_color)
        axis.set_title(title, fontsize=style.colorbar_title_size * points, color=style.text_color, pad=20,
                       loc="left", fontfamily=style.colorbar_font)
        for label in bar.ax.get_yticklabels():
            label.set_fontfamily(style.colorbar_font)
    save_figure(fig, path, dpi=100, facecolor=style.background)
    plt.close(fig)


def render_image(run_dir: Path, input_path: Path, label: str, cycle: int, species: list[int] | None, density_ions: str,
                 style: RenderStyle, view: str = "x-line", region: np.ndarray | None = None, *, colorbars: bool = True,
                 outline: np.ndarray | None = None, density_range: tuple[float, float] | None = None,
                 overlay_limit: float | None = None, orientation_axes: bool = True,
                 axes_grid: bool = True, particles: dict | None = None) -> tuple[np.ndarray, dict]:
    """Render the density of species with the overlay of view at cycle, over the box region or a zoom.

    species None takes the sheet ions; density_ions sets d_i as in the rate analysis.  view "x-line" draws the field
    lines through the X lines, "kink" the bent current sheet in the density alone, "lower-hybrid" the
    isosurfaces of the lower-hybrid part of the Ez fluctuation and "ropes" the field lines around the secondary O
    points coloured by Bz.  region, in d_i and of shape (3, 2), defaults to the whole box for "x-line" and to the
    zoom of style for the others.  outline draws the edges of another box; axes_grid False leaves out the labelled
    axes of the box and orientation_axes False the x, y, z arrows; density_range and overlay_limit replace
    the scales taken from the data.  particles, from energetic_particles with the energy range "range", are drawn
    over the view.  Return the image and its scales: Omega_ci*t ("time"), the largest density shown ("shown"), the
    density colour range, the overlay title and limit, None without a coloured overlay, and the particle energy range
    ("particles"), None without particles.
    """
    try:
        import pyvista as pv
        from vtkmodules.vtkCommonCore import vtkLogger
    except ImportError as error:
        raise RuntimeError("the 3D view needs PyVista: python3 -m pip install pyvista") from error

    params, data_dir, layout = open_deck(run_dir, input_path)
    count = int(first_float(params, "ns"))
    species = ion_species(params, "sheet") if species is None else species
    if not species or not all(0 <= s < count for s in species):
        raise ValueError(f"{input_path}: --density-species must name species in [0, {count - 1}]")
    if view != "x-line" and not all(periodic_axes(params)[::2]):
        raise ValueError(f"{input_path}: the {view} view needs a box periodic in x and z")
    amplitude = first_float(params, AMPLITUDE_KEYS[layout])
    alfven_speed, omega_ci, ion_length = sheet_normalization(params, amplitude, density_ions)
    time, density, b, e = read_snapshot(data_dir, layout, params, cycle, species, electric=view == "lower-hybrid")
    density = np.abs(density)  # an electron species has a negative charge density

    # Nodal arrays of shape (z, y, x) on a grid in d_i.
    lengths = np.array([first_float(params, key) for key in ("Lx", "Ly", "Lz")]) / ion_length
    nodes = np.array(density.shape[::-1])
    spacing = lengths / (nodes - 1)
    grid = pv.ImageData(dimensions=nodes, spacing=spacing)
    grid["density"] = density.ravel()
    grid["B"] = b.reshape(-1, 3)
    grid["Bz"] = b[..., 2].ravel() / abs(amplitude)
    x_lines = [(x * lengths[0], y * lengths[1], sign * math.copysign(1.0, amplitude))
               for _, x, y, sign in X_LINES[params["Case"][0]]]
    # A zoom without --render-region centres on the first sheet: at the kink station, or at its O point of most flux.
    sheet = np.array([(x_lines[0][0] - KINK_STATION * lengths[0]) % lengths[0], x_lines[0][1], 0.5 * lengths[2]])

    seeds, line_length = None, style.line_length
    if view == "ropes":
        # Seeds either side in y of the secondary O points, of planes rope_seed_spacing apart in z.
        step = max(1, int(round(style.rope_seed_spacing / spacing[2])))
        seeds, strongest = [], None
        for x, y, sign in x_lines:
            node, _, o_points = secondary_o_points(b[:-1, :, :-1] / abs(amplitude), x / spacing[0], y / spacing[1],
                                                   sign, spacing)
            for z, x0, flux, _ in o_points:
                iz, ix = int(round(z / spacing[2])), int(round(x0 / spacing[0]))
                if strongest is None or flux > strongest[0]:
                    strongest = (flux, np.array([x0, node[iz, ix] * spacing[1], z]))
                if iz % step == 0:
                    seeds += [(x0, node[iz, ix] * spacing[1] + offset, z)
                              for offset in (-style.rope_seed_offset, style.rope_seed_offset)]
        if not seeds:
            raise ValueError(f"{data_dir}: cycle {cycle} has no secondary O point holding {ROPE_MIN_FLUX:g} B0*d_i "
                             "or more, so no flux rope to seed")
        seeds, line_length, sheet = np.array(seeds), style.rope_line_length, strongest[1]
    elif view == "x-line":
        # Seeds about each X line, measured from the neutral sheet, which a kink displaces in y.
        seeds = []
        for x, y, _ in x_lines:
            node = neutral_sheet(b[..., 0], y / spacing[1], int(SHEET_REACH * nodes[1]))

            def sheet_y(x0: float, z0: float) -> float:
                found = node[int(round(z0 / spacing[2])), int(round(x0 / spacing[0])) % (nodes[0] - 1)]
                return y if np.isnan(found) else found * spacing[1]

            for z in lengths[2] * (np.arange(style.seed_planes) + 0.5) / style.seed_planes:
                seeds += [(x, sheet_y(x, z) + side * offset, z) for offset in style.inflow_seeds for side in (-1, 1)]
                seeds += [((x + side * offset) % lengths[0], sheet_y(x + side * offset, z), z)
                          for offset in style.outflow_seeds for side in (-1, 1)]
        seeds = np.array(seeds)
    elif view == "lower-hybrid" and style.lh_quantity == "Ez":
        grid["lower_hybrid"] = e[..., 2].ravel() / (abs(amplitude) * alfven_speed)
    elif view == "lower-hybrid":
        if style.lh_quantity != "dEz":
            raise ValueError(f"lh_quantity = {style.lh_quantity!r}: expected 'dEz' or 'Ez'")
        # The lower-hybrid part of the fluctuation of Ez about its z mean, in B0*v_A.
        ez = e[:-1, :, :, 2]
        spectrum = np.fft.rfft(ez - ez.mean(axis=0), axis=0)
        cut = LH_SPLIT / lower_hybrid_length(params, amplitude, ion_length)
        spectrum[2.0 * np.pi * np.fft.rfftfreq(ez.shape[0], spacing[2]) < cut] = 0.0
        wave = np.fft.irfft(spectrum, n=ez.shape[0], axis=0) / (abs(amplitude) * alfven_speed)
        grid["lower_hybrid"] = np.concatenate([wave, wave[:1]]).ravel()

    if region is None and view != "x-line":
        region = zoom_region(sheet, np.array(getattr(style, view.replace("-", "_") + "_zoom")) * lengths, lengths)
    if region is not None:
        low = np.clip(np.floor(region[:, 0] / spacing + 1.0e-9).astype(int), 0, nodes - 1)
        high = np.clip(np.ceil(region[:, 1] / spacing - 1.0e-9).astype(int), 0, nodes - 1)
        if np.any(high - low < 2):
            raise ValueError("--render-region must span two cells or more along every axis inside the box")
        grid = grid.extract_subset((low[0], high[0], low[1], high[1], low[2], high[2]))
    bounds = np.array(grid.bounds).reshape(3, 2)

    vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)  # off screen, VTK warns that there is no X display
    plotter = pv.Plotter(off_screen=True, window_size=list(style.window_size))
    plotter.set_background(style.background)
    opacity = style.opacity if isinstance(style.opacity, str) else np.asarray(style.opacity)
    bar = {"vertical": True, "position_y": style.colorbar_position[1], "height": 0.75, "width": 0.05,
           "title_font_size": style.colorbar_title_size, "label_font_size": style.colorbar_label_size,
           "color": style.text_color, "fmt": "%.2g"}
    shown = float(grid["density"].max())
    bars_x = style.colorbar_position[0] + style.overlay_colorbar_shift * (view in ("lower-hybrid", "ropes"))
    plotter.add_volume(
        grid, scalars="density", opacity=opacity,
        # The kink shows in the volume of the current sheet itself; an overlay needs a thinner volume.
        opacity_unit_distance=(style.opacity_unit_distance if view in ("x-line", "kink")
                               else style.zoom_opacity_unit_distance),
        cmap=LinearSegmentedColormap.from_list("density", [(p, np.array(rgb) / 255.0) for p, rgb in style.colormap]),
        clim=density_range or style.density_range or (0.0, shown), show_scalar_bar=colorbars,
        scalar_bar_args={**bar, "title": style.density_label, "position_x": bars_x,
                         "n_labels": 7, "use_opacity": False})
    # The overlay takes a second colour bar, right of that of the density.
    overlay_bar = {**bar, "position_x": bars_x + style.colorbar_spacing, "n_labels": 5}
    overlay, overlay_shown = None, colorbars and style.overlay_colorbar
    if seeds is not None:
        inside = np.all((seeds >= bounds[:, 0]) & (seeds <= bounds[:, 1]), axis=1)
        # Steps are in cell lengths.
        lines = grid.streamlines_from_source(pv.PolyData(seeds[inside]), vectors="B", max_length=line_length,
                                             initial_step_length=0.2, max_step_length=0.5, max_steps=100000,
                                             compute_vorticity=False) if inside.any() else pv.PolyData()
        radius, color = ((style.rope_line_radius, style.rope_line_color) if view == "ropes"
                         else (style.line_radius, style.line_color))
        if lines.n_points and color:
            plotter.add_mesh(lines.tube(radius=radius, n_sides=12), color=color, smooth_shading=True,
                             specular=0.4, specular_power=20.0)
        elif lines.n_points:
            # Bz colours the field lines: the core field of a flux rope, the Hall field about an X line.
            limit = overlay_limit or float(np.abs(lines["Bz"]).max()) or 1.0
            overlay = ("Bz/B0", limit)
            plotter.add_mesh(lines.tube(radius=radius, n_sides=12), scalars="Bz", cmap="coolwarm",
                             clim=(-limit, limit), smooth_shading=True, specular=0.4, specular_power=20.0,
                             show_scalar_bar=overlay_shown, scalar_bar_args={**overlay_bar, "title": "Bz/B0"})
    elif view == "lower-hybrid":
        level = overlay_limit or style.lh_iso_fraction * float(np.percentile(np.abs(grid["lower_hybrid"]), 99.5))
        overlay = (f"{style.lh_quantity}/(B0 vA)", level)
        fronts = grid.contour([-level, level], scalars="lower_hybrid")
        if fronts.n_points:
            plotter.add_mesh(fronts, scalars="lower_hybrid", cmap="coolwarm", clim=(-level, level),
                             smooth_shading=True, specular=0.3, show_scalar_bar=overlay_shown,
                             scalar_bar_args={**overlay_bar, "title": f"{style.lh_quantity}/(B0 vA)", "n_labels": 3})
    if particles is not None:
        inside = np.all((particles["position"] >= bounds[:, 0]) & (particles["position"] <= bounds[:, 1]), axis=1)
        order = np.flatnonzero(inside)[np.argsort(particles["energy"][inside])]  # the most energetic drawn last
        cloud = pv.PolyData(np.ascontiguousarray(particles["position"][order], dtype=np.float32))
        cloud["energy"] = particles["energy"][order]
        # Next to the colour bar of the overlay, when it has one.
        slot = 1 + bool(overlay and overlay_shown)
        plotter.add_mesh(cloud, scalars="energy", log_scale=True, clim=particles["range"],
                         cmap=LinearSegmentedColormap.from_list("particles", style.particle_overlay_colors),
                         render_points_as_spheres=True, point_size=style.particle_overlay_size,
                         opacity=style.particle_overlay_opacity, show_scalar_bar=colorbars,
                         scalar_bar_args={**bar, "title": "E/T0", "position_x": bars_x + style.colorbar_spacing * slot,
                                                            "n_labels": 4})
    if outline is not None:
        plotter.add_mesh(pv.Box(bounds=outline.ravel()).outline(), color=style.inset_outline_color,
                         line_width=style.inset_box_line_width)
    # The arrows sit in the lower left corner, or in a corner of style for a zoom, whose box reaches lower.
    corner = np.array((0.0, 0.0) if region is None else style.zoom_orientation_axes_corner)
    if orientation_axes:
        plotter.add_axes(color=style.text_color, line_width=4, label_size=(0.35, 0.14),
                         viewport=(*corner, *(corner + style.orientation_axes_size)))
    if style.title:
        plotter.add_text(f"{label}, cycle {cycle}, Omega_ci t = {omega_ci * time:.4g}", position="upper_left",
                         font_size=style.title_font_size, color=style.text_color)

    aim_camera(plotter, bounds, (style.camera_azimuth, style.camera_elevation) if region is None else style.zoom_camera,
               style, axes_grid)

    image = plotter.screenshot(return_img=True)
    corners = None
    if outline is not None:
        # The corners of the outlined box in image pixels (column, row).
        from vtkmodules.vtkRenderingCore import vtkCoordinate
        coordinate = vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        corners = []
        for corner in itertools.product(*outline):
            coordinate.SetValue(*corner)
            column, row = coordinate.GetComputedDoubleDisplayValue(plotter.renderer)
            corners.append((column, image.shape[0] - row, corner[1]))
        corners = np.array(corners)
    plotter.close()
    image, cut = trim_margin(image, style)
    if corners is not None:
        corners[:, :2] -= cut
    return image, {"time": omega_ci * time, "shown": shown, "overlay": overlay, "outline": corners,
                   "density_range": density_range or style.density_range or (0.0, shown),
                   "particles": particles and particles["range"]}


def aim_camera(plotter, bounds: np.ndarray, angles: tuple[float, float], style: RenderStyle,
               axes_grid: bool = True) -> None:
    """Point the camera of plotter at the box bounds (shape (3, 2)) from the azimuth and elevation angles, with the
    labelled axes of the box unless axes_grid is False."""
    import pyvista as pv

    centre = bounds.mean(axis=1)
    azimuth, elevation = np.radians(angles)
    direction = np.array([np.sin(azimuth) * np.cos(elevation), np.sin(elevation),
                          np.cos(azimuth) * np.cos(elevation)])
    camera = plotter.camera
    camera.focal_point = centre
    camera.position = centre + style.camera_distance * np.linalg.norm(np.ptp(bounds, axis=1)) * direction
    camera.up = (0.0, 1.0, 0.0)
    camera.view_angle = 30.0
    camera.SetWindowCenter(*style.window_center)
    if axes_grid:
        plotter.renderer.AddViewProp(grid_axes(bounds, direction, pv.Color(style.text_color).float_rgb,
                                               style.axis_font_size))


def trim_margin(image: np.ndarray, style: RenderStyle) -> tuple[np.ndarray, tuple[int, int]]:
    """Cut the background beyond style.margin pixels around the drawing; return the image and the pixels cut at
    the left and the top."""
    import pyvista as pv

    if style.margin is None:
        return image, (0, 0)
    drawn = np.any(image[..., :3] != pv.Color(style.background).int_rgb, axis=2)
    rows, columns = np.flatnonzero(drawn.any(axis=1)), np.flatnonzero(drawn.any(axis=0))
    if not rows.size:
        return image, (0, 0)
    top, left = max(rows[0] - style.margin, 0), max(columns[0] - style.margin, 0)
    return image[top:rows[-1] + style.margin + 1, left:columns[-1] + style.margin + 1], (left, top)


# Measures of --instabilities.  Distances from a sheet are in its half-thickness, the deck key delta.
LH_EDGE_BAND = (0.5, 3.0)  # |y - y_sheet| over which the lower-hybrid drift waves are measured
LH_SPLIT = 0.3  # kz*sqrt(rho_e*rho_i) above which a fluctuation of Ez along z counts as a lower-hybrid wave
KINK_STATION = 0.25  # the y-z slices cut x this fraction of Lx before the X line, half way to the O line
SHEET_REACH = 0.2  # the neutral sheet is sought this fraction of Ly on either side of the sheet
ROPE_MIN_FLUX = 0.1  # B0*d_i: the smallest flux that an island of the in-plane flux function must hold


@dataclass(frozen=True)
class SheetSetup:
    """The deck values that the measures of one snapshot need; lengths are in code units."""

    data_dir: Path
    layout: str
    params: dict
    species: tuple[int, ...]
    x_lines: tuple[tuple[float, float, float], ...]  # x node, y node and sign of the sheet field of each X line
    b0: float  # magnitude of the sheet amplitude
    field_unit: float  # B0*v_A
    ion_length: float  # d_i
    cell: tuple[float, float, float]  # dx, dy, dz
    half_thickness: float  # delta
    lh_split: float  # kz*d_i between the kink and the lower-hybrid waves


@dataclass
class SheetState:
    """The measures of one sheet in one snapshot; z, y and x run over unique nodes and lengths are in d_i."""

    fluctuation: np.ndarray  # (y,) rms over x and z of the lower-hybrid part of the Ez fluctuation, in B0*v_A
    edge_power: np.ndarray  # (kz,) power of the Ez fluctuation per kz bin over the sheet edges, in (B0*v_A)^2
    lh_slice: np.ndarray  # (z, y) lower-hybrid part of the Ez fluctuation at the kink station, in B0*v_A
    density_slice: np.ndarray  # (z, y) density at the kink station
    sheet_y: np.ndarray  # (z, x) y of the neutral sheet, nan where Bx keeps its sign
    sheet_bz: np.ndarray  # (z, x) Bz on the neutral sheet, in B0
    o_points: np.ndarray  # (point, 4) z, x, flux in B0*d_i and Bz in B0 of the secondary O points
    rope_plane: tuple[float, np.ndarray, np.ndarray] | None  # z, psi and Bz (y, x) through the largest O point


def neutral_sheet(bx: np.ndarray, centre: float, reach: int) -> np.ndarray:
    """Return the fractional y node, shape (z, x), where Bx of shape (z, y, x) changes sign closest to node centre.

    The search spans reach nodes on either side of centre; nan marks a column where Bx keeps its sign there.
    """
    low, high = max(int(centre) - reach, 0), min(int(centre) + reach + 1, bx.shape[1] - 1)
    window = bx[:, low:high + 1]
    change = window[:, :-1] * window[:, 1:] <= 0.0
    distance = np.where(change, np.abs(low + 0.5 + np.arange(change.shape[1])[None, :, None] - centre), np.inf)
    lower = distance.argmin(axis=1)
    below = np.take_along_axis(window, lower[:, None], 1)[:, 0]
    above = np.take_along_axis(window, lower[:, None] + 1, 1)[:, 0]
    fraction = np.where(below != above, below / np.where(below != above, below - above, 1.0), 0.5)
    return np.where(np.isfinite(distance.min(axis=1)), low + lower + fraction, np.nan)


def on_sheet(values: np.ndarray, node: np.ndarray) -> np.ndarray:
    """Interpolate values of shape (z, y, x) linearly in y at the fractional y node of shape (z, x)."""
    known = np.nan_to_num(node)
    lower = np.clip(known.astype(int), 0, values.shape[1] - 2)
    weight = known - lower
    below = np.take_along_axis(values, lower[:, None], 1)[:, 0]
    above = np.take_along_axis(values, lower[:, None] + 1, 1)[:, 0]
    return np.where(np.isnan(node), np.nan, (1.0 - weight) * below + weight * above)


def snapshot_cycles(data_dir: Path, layout: str, params: dict[str, list[str]], max_cycle: float) -> list[int]:
    """Return the cycles of the field snapshots of a run up to max_cycle."""
    if layout == TPIC_LAYOUT:
        with open_field_reader(data_dir / "B") as reader:
            cycles = [epoch[0] for _, epoch in reader.records]
    else:
        name = params["SimulationName"][0]
        cycles = sorted(int(path.stem.rsplit("_", 1)[1]) for path in data_dir.glob(f"{name}_B_*.vtk"))
    return [cycle for cycle in cycles if cycle <= max_cycle]


def measure_snapshot(cycle: int, setup: SheetSetup) -> tuple[float, list[SheetState]]:
    """Return the time of cycle and the measures of each sheet (module docstring, "Sheet instabilities")."""
    time, density, b, e = read_snapshot(setup.data_dir, setup.layout, setup.params, cycle, list(setup.species),
                                        electric=True)
    # x and z are periodic: their last node repeats the first.
    b, ez, density = b[:-1, :, :-1] / setup.b0, e[:-1, :, :-1, 2] / setup.field_unit, density[:-1, :, :-1]
    nz, ny, nx = ez.shape
    dx, dy, dz = (length / setup.ion_length for length in setup.cell)
    delta = setup.half_thickness / setup.ion_length
    # The fluctuation of Ez along z, the direction of the sheet current, as the one-sided power per kz bin.
    spectrum = np.fft.rfft(ez - ez.mean(axis=0), axis=0) / nz
    power = np.abs(spectrum) ** 2
    power[1:(nz + 1) // 2] *= 2.0
    lower_hybrid = 2.0 * np.pi * np.fft.rfftfreq(nz, dz) >= setup.lh_split
    fluctuation = np.sqrt(power[lower_hybrid].sum(axis=0).mean(axis=1))

    states = []
    for x_node, y_node, sign in setup.x_lines:
        distance = np.abs(np.arange(ny) - y_node) * dy / delta
        edges = (distance >= LH_EDGE_BAND[0]) & (distance <= LH_EDGE_BAND[1])
        station = int(round(x_node - KINK_STATION * nx)) % nx
        lh_slice = np.fft.irfft(np.where(lower_hybrid[:, None], spectrum[:, :, station], 0.0) * nz, n=nz, axis=0)

        sheet_node, psi, o_points = secondary_o_points(b, x_node, y_node, sign, (dx, dy, dz))
        rope_plane = None
        if len(o_points):
            iz = int(round(o_points[np.argmax(o_points[:, 2]), 0] / dz))
            rope_plane = (iz * dz, psi[iz], b[iz, :, :, 2])
        states.append(SheetState(
            fluctuation=fluctuation, edge_power=power[:, edges].mean(axis=(1, 2)), lh_slice=lh_slice,
            density_slice=density[:, :, station], sheet_y=sheet_node * dy, sheet_bz=on_sheet(b[..., 2], sheet_node),
            o_points=o_points, rope_plane=rope_plane))
    return time, states


def secondary_o_points(b: np.ndarray, x_node: float, y_node: float, sign: float,
                       spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the neutral sheet, the in-plane flux function and the secondary O points of one sheet.

    b is B/B0 of shape (z, y, x, 3) on the unique nodes of a box periodic in x and z, spacing dx, dy and dz in d_i.
    The neutral sheet is a fractional y node of shape (z, x), psi has the shape (z, y, x) and the O points the
    shape (point, 4): z, x, flux and Bz on the sheet.  psi jumps only at the O line of the sheet, half a period in
    x from its X line.
    """
    from scipy.signal import find_peaks

    nz, ny, nx, _ = b.shape
    dx, dy, dz = spacing
    # Rolled in x so that the X line sits in the middle and the O line of its sheet at both ends.
    shift = int(round(0.5 * nx - x_node))
    bx, by = np.roll(b[..., 0], shift, axis=2), np.roll(b[..., 1], shift, axis=2)
    node = neutral_sheet(bx, y_node, int(SHEET_REACH * ny))
    sheet_bz = on_sheet(b[..., 2], np.roll(node, -shift, axis=1))
    # The in-plane flux function of every x-y plane: dpsi/dy = Bx, dpsi/dx = -By, zero at the first node.
    rise = np.concatenate([np.zeros((nz, 1)), np.cumsum(0.5 * dy * (bx[:, 1:, 0] + bx[:, :-1, 0]), axis=1)], axis=1)
    run = np.concatenate([np.zeros((nz, ny, 1)), np.cumsum(0.5 * dx * (by[..., 1:] + by[..., :-1]), axis=2)], axis=2)
    psi = rise[:, :, None] - run
    # Along the neutral sheet the X line is a maximum of sign*psi and an O point a minimum; the island of the O
    # line, the deepest minimum, is not secondary.
    o_points = []
    for iz, row in enumerate(sign * on_sheet(psi, node)):
        known = np.isfinite(row)
        if known.sum() < 2:
            continue
        row = np.interp(np.arange(nx), np.flatnonzero(known), row[known])
        peaks, properties = find_peaks(-row, prominence=ROPE_MIN_FLUX)
        for peak, flux in zip(peaks, properties["prominences"]):
            if peak != np.argmin(row):
                ix = (peak - shift) % nx
                o_points.append((iz * dz, ix * dx, flux, sheet_bz[iz, ix]))
    return np.roll(node, -shift, axis=1), np.roll(psi, -shift, axis=2), np.array(o_points).reshape(-1, 4)


def growth_fit(time: np.ndarray, amplitude: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Return the e-folding rate of amplitude from it first reaching 10% to 50% of its largest value, and the fitted
    curve on those snapshots; None when fewer than three snapshots span them."""
    peak = np.nanmax(amplitude)
    start, end = int(np.argmax(amplitude >= 0.1 * peak)), int(np.argmax(amplitude >= 0.5 * peak))
    if not peak > 0.0 or end - start < 2:
        return None
    span = slice(start, end + 1)
    slope, offset = np.polyfit(time[span], np.log(amplitude[span]), 1)
    return slope, time[span], np.exp(offset + slope * time[span])


def plot_growth(axis, time: np.ndarray, curves: list[tuple[str, np.ndarray]], ylabel: str) -> None:
    """Draw amplitudes against Omega_ci*t on a log scale, the first with its exponential fit."""
    for index, (label, amplitude) in enumerate(curves):
        positive = amplitude > 0.0
        axis.semilogy(time[positive], amplitude[positive], **{**curve_look(index, time.size), "label": label})
    fit = growth_fit(time, curves[0][1])
    if fit:
        axis.semilogy(fit[1], fit[2], color="0.2", linestyle="--", linewidth=2.0,
                      label=rf"$\gamma = {fit[0]:.2f}\,\Omega_{{ci}}$")
    axis.set_xlabel(r"$\Omega_{ci} t$")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()


def slice_panels(fig, axes, z: np.ndarray, y: np.ndarray, time: np.ndarray, picks: list[int], slices: np.ndarray,
                 cmap: str, symmetric: bool, label: str, overlay) -> None:
    """Draw (z, y) slices of the picked snapshots on a shared colour scale; overlay(axis, index) adds to each."""
    values = slices[picks]
    limit = np.nanpercentile(np.abs(values), 99.5) or 1.0
    scale = {"vmin": -limit, "vmax": limit} if symmetric else {"vmin": 0.0, "vmax": np.nanmax(values)}
    for axis, index in zip(axes, picks):
        mesh = axis.pcolormesh(z, y, slices[index].T, cmap=cmap, shading="nearest", **scale)
        overlay(axis, index)
        axis.set_xlabel(r"$z/d_i$")
        axis.set_title(rf"$\Omega_{{ci}} t = {time[index]:.3g}$", fontsize=FONT_SIZES.legend)
    axes[0].set_ylabel(r"$(y - y_s)/d_i$")
    fig.colorbar(mesh, ax=list(axes), label=label)


def write_instability_plots(base: Path, suffix: str, title: str, time: np.ndarray, states: list[SheetState],
                            lengths: tuple[float, float, float], sheet: tuple[float, float], delta: float,
                            lh_length: float) -> list[Path]:
    """Plot the lower-hybrid drift waves, the drift kink and the flux ropes of one sheet; return the paths written.

    lengths are Lx, Ly and Lz, sheet the x of the X line and the y of the sheet and delta its half-thickness, all in
    d_i; lh_length is sqrt(rho_e*rho_i) in d_i.
    """
    nz, ny = states[0].lh_slice.shape
    nx = states[0].sheet_y.shape[1]
    x, z = lengths[0] * np.arange(nx) / nx, lengths[2] * np.arange(nz) / nz
    y = lengths[1] * np.arange(ny) / (ny - 1) - sheet[1]
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, lengths[2] / nz)
    station = (sheet[0] - KINK_STATION * lengths[0]) % lengths[0]
    near = np.abs(y) <= min(6.0 * delta, 0.5 * lengths[1])
    paths = []

    # Lower-hybrid drift waves: short waves of Ez along the current on the sheet edges.
    fluctuation = np.array([state.fluctuation for state in states])
    edge_rows = (np.abs(y) >= LH_EDGE_BAND[0] * delta) & (np.abs(y) <= LH_EDGE_BAND[1] * delta)
    edge = fluctuation[:, edge_rows].max(axis=1)
    centre = fluctuation[:, np.argmin(np.abs(y))]
    fig, axes = plt.subplots(2, 3, figsize=(27, 14), constrained_layout=True)
    mesh = axes[0, 0].pcolormesh(time, y[near], fluctuation[:, near].T, cmap="inferno", shading="nearest")
    fig.colorbar(mesh, ax=axes[0, 0], label=r"lower-hybrid $\delta E_z$ rms $/(B_0 v_A)$")
    for side in (-1.0, 1.0):
        axes[0, 0].axhline(side * delta, color="w", linestyle=":", linewidth=1.5)
    axes[0, 0].set_xlabel(r"$\Omega_{ci} t$")
    axes[0, 0].set_ylabel(r"$(y - y_s)/d_i$")
    power = np.array([state.edge_power for state in states])
    shown = (kz > 0.0) & (kz <= 3.0 / lh_length)
    floor = np.nanmax(power[:, shown]) * 1.0e-5
    mesh = axes[0, 1].pcolormesh(time, kz[shown], np.maximum(power[:, shown], floor).T, cmap="viridis",
                                 shading="nearest", norm=LogNorm())
    fig.colorbar(mesh, ax=axes[0, 1], label=r"edge $|\delta E_z(k_z)|^2/(B_0 v_A)^2$")
    axes[0, 1].axhline(1.0 / lh_length, color="w", linestyle="--", linewidth=2.0)
    axes[0, 1].axhline(LH_SPLIT / lh_length, color="w", linestyle=":", linewidth=1.5)
    axes[0, 1].text(time[0], 1.0 / lh_length, r" $k_z\sqrt{\rho_e\rho_i} = 1$", color="w", va="bottom",
                    fontsize=FONT_SIZES.legend)
    axes[0, 1].set_xlabel(r"$\Omega_{ci} t$")
    axes[0, 1].set_ylabel(r"$k_z d_i$")
    plot_growth(axes[0, 2], time, [("sheet edges", edge), ("sheet centre", centre)],
                r"lower-hybrid $\delta E_z$ rms $/(B_0 v_A)$")
    peak = int(np.argmax(edge))
    picks = sorted({int(np.argmax(edge >= 0.5 * edge[peak])), peak, len(states) - 1})
    picks = picks if len(picks) == 3 else [round(i) for i in np.linspace(0, len(states) - 1, 3)]

    def sheet_outline(axis, index):
        density = states[index].density_slice[:, near]
        axis.contour(z, y[near], density.T, levels=[0.5 * density.max()], colors="k", linewidths=1.5)

    slice_panels(fig, axes[1], z, y[near], time, picks, np.array([state.lh_slice[:, near] for state in states]),
                 "RdBu_r", True, rf"lower-hybrid $\delta E_z/(B_0 v_A)$ at $x = {station:.3g}\,d_i$", sheet_outline)
    fig.suptitle(f"{title}: lower-hybrid drift waves, the part of the Ez fluctuation along z with "
                 rf"$k_z\sqrt{{\rho_e\rho_i}} \geq {LH_SPLIT:g}$", fontsize=FONT_SIZES.axis_label)
    paths.append(base.with_name(f"{base.name}_lhdi{suffix}"))
    save_figure(fig, paths[-1], dpi=120)
    plt.close(fig)

    # Drift kink: the neutral sheet bends along the current with a wavelength of several delta.
    sheet_y = np.array([state.sheet_y for state in states]) - sheet[1]
    displacement = sheet_y - np.nanmean(sheet_y, axis=1, keepdims=True)
    rms = np.sqrt(np.nanmean(displacement ** 2, axis=1))  # (snapshot, x)
    ix_station, ix_line = int(round(station / lengths[0] * nx)) % nx, int(round(sheet[0] / lengths[0] * nx)) % nx
    fig, axes = plt.subplots(2, 3, figsize=(27, 14), constrained_layout=True)
    at_station = displacement[:, :, ix_station]
    limit = np.nanmax(np.abs(at_station)) or 1.0
    mesh = axes[0, 0].pcolormesh(time, z, at_station.T, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
    fig.colorbar(mesh, ax=axes[0, 0], label=rf"sheet displacement $\xi/d_i$ at $x = {station:.3g}\,d_i$")
    axes[0, 0].set_xlabel(r"$\Omega_{ci} t$")
    axes[0, 0].set_ylabel(r"$z/d_i$")
    kink_power = (np.abs(np.fft.rfft(np.nan_to_num(displacement), axis=1) / nz) ** 2).mean(axis=2)
    shown = (kz > 0.0) & (kz <= LH_SPLIT / lh_length)
    floor = np.nanmax(kink_power[:, shown]) * 1.0e-5
    mesh = axes[0, 1].pcolormesh(time, kz[shown] * delta, np.maximum(kink_power[:, shown], floor).T, cmap="viridis",
                                 shading="nearest", norm=LogNorm())
    fig.colorbar(mesh, ax=axes[0, 1], label=r"$|\xi(k_z)|^2/d_i^2$, mean over x")
    axes[0, 1].set_xlabel(r"$\Omega_{ci} t$")
    axes[0, 1].set_ylabel(r"$k_z \delta$")
    amplitude = np.nanmean(rms, axis=1)
    plot_growth(axes[0, 2], time, [("mean over x", amplitude), (rf"$x = {station:.3g}\,d_i$", rms[:, ix_station]),
                                   (rf"X line, $x = {sheet[0]:.3g}\,d_i$", rms[:, ix_line])],
                r"rms sheet displacement $/d_i$")
    picks = sorted({int(np.argmax(amplitude >= 0.1 * np.nanmax(amplitude))),
                    int(np.argmax(amplitude >= 0.5 * np.nanmax(amplitude))), len(states) - 1})
    picks = picks if len(picks) == 3 else [round(i) for i in np.linspace(0, len(states) - 1, 3)]

    def neutral_line(axis, index):
        axis.plot(z, sheet_y[index, :, ix_station], color="w", linewidth=2.0)

    slice_panels(fig, axes[1], z, y[near], time, picks, np.array([state.density_slice[:, near] for state in states]),
                 "viridis", False, rf"density at $x = {station:.3g}\,d_i$; white: neutral sheet", neutral_line)
    dominant = kz[1:][np.argmax(kink_power[-1, 1:])]
    fig.suptitle(rf"{title}: drift kink, the neutral sheet $B_x = 0$ displaced in y along z; at the end "
                 rf"$k_z\delta = {dominant * delta:.2f}$, $\lambda_z = {2.0 * np.pi / dominant:.3g}\,d_i$",
                 fontsize=FONT_SIZES.axis_label)
    paths.append(base.with_name(f"{base.name}_kink{suffix}"))
    save_figure(fig, paths[-1], dpi=120)
    plt.close(fig)

    # Flux ropes: O points of the in-plane flux function on the neutral sheet besides that of the main island.
    counts = np.array([len(state.o_points) / nz for state in states])
    largest = np.array([state.o_points[:, 2].max() if len(state.o_points) else 0.0 for state in states])
    pick = int(np.argmax(largest)) if largest.any() else len(states) - 1
    fig, axes = plt.subplots(1, 3, figsize=(30, 8.5), constrained_layout=True)
    axes[0].plot(time, counts, **curve_look(0, time.size))
    axes[0].set_xlabel(r"$\Omega_{ci} t$")
    axes[0].set_ylabel("secondary O points per x-y plane")
    axes[0].grid(True, alpha=0.3)
    twin = axes[0].twinx()
    twin.plot(time, largest, **{**curve_look(2, time.size), "color": "C3"})
    twin.set_ylabel(r"largest island flux $/(B_0 d_i)$", color="C3")
    bz = states[pick].sheet_bz
    limit = np.nanpercentile(np.abs(bz), 99.5) or 1.0
    mesh = axes[1].pcolormesh(x, z, bz, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
    fig.colorbar(mesh, ax=axes[1], label=r"$B_z/B_0$ on the neutral sheet")
    points = states[pick].o_points
    if len(points):
        axes[1].scatter(points[:, 1], points[:, 0], s=400.0 * points[:, 2] / points[:, 2].max(), facecolors="none",
                        edgecolors="k", linewidths=1.5, label=r"O points, area $\propto$ flux")
        axes[1].legend(loc="upper right")
    axes[1].set_xlabel(r"$x/d_i$")
    axes[1].set_ylabel(r"$z/d_i$")
    axes[1].set_title(rf"$\Omega_{{ci}} t = {time[pick]:.3g}$", fontsize=FONT_SIZES.legend)
    plane = states[pick].rope_plane
    if plane is None:
        axes[2].text(0.5, 0.5, f"no O point besides the main island\nholds {ROPE_MIN_FLUX:g} B0 d_i or more",
                     ha="center", va="center", transform=axes[2].transAxes, fontsize=FONT_SIZES.legend)
        axes[2].set_axis_off()
    else:
        rope_z, psi, rope_bz = plane
        limit = np.nanpercentile(np.abs(rope_bz[near]), 99.5) or 1.0
        mesh = axes[2].pcolormesh(x, y[near], rope_bz[near], cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
        axes[2].contour(x, y[near], psi[near], 40, colors="k", linewidths=0.8)
        fig.colorbar(mesh, ax=axes[2], label=r"$B_z/B_0$; lines: in-plane flux function")
        axes[2].set_xlabel(r"$x/d_i$")
        axes[2].set_ylabel(r"$(y - y_s)/d_i$")
        axes[2].set_title(rf"plane $z = {rope_z:.3g}\,d_i$ of the largest O point", fontsize=FONT_SIZES.legend)
    fig.suptitle(f"{title}: flux ropes, islands of the in-plane flux function holding at least "
                 rf"${ROPE_MIN_FLUX:g}\,B_0 d_i$", fontsize=FONT_SIZES.axis_label)
    paths.append(base.with_name(f"{base.name}_flux_ropes{suffix}"))
    save_figure(fig, paths[-1], dpi=120)
    plt.close(fig)

    paths.append(base.with_name(f"{base.name}_instabilities.npz"))
    np.savez_compressed(
        paths[-1], omega_ci_t=time, x=x, y=y, z=z, kz=kz, fluctuation=fluctuation, edge_power=power,
        lh_slice=np.array([state.lh_slice for state in states]),
        density_slice=np.array([state.density_slice for state in states]), sheet_y=sheet_y + sheet[1],
        sheet_bz=np.array([state.sheet_bz for state in states]),
        o_points=np.concatenate([np.column_stack([np.full(len(state.o_points), t), state.o_points])
                                 for t, state in zip(time, states)]))
    return paths


def analyze_instabilities(base: Path, suffix: str, run_dir: Path, input_path: Path, label: str,
                          species: list[int] | None, density_ions: str, max_cycle: float, workers: int) -> list[Path]:
    """Measure the sheet instabilities of a deck in every snapshot up to max_cycle and plot them; return the paths."""
    params, data_dir, layout = open_deck(run_dir, input_path)
    case = params["Case"][0]
    periodic_x, _, periodic_z = periodic_axes(params)
    if not (periodic_x and periodic_z) or int(first_float(params, "nzc")) < 8:
        raise ValueError(f"{input_path}: the sheet instabilities need a box periodic in x and z with 8 z cells or more")
    count = int(first_float(params, "ns"))
    species = ion_species(params, "sheet") if species is None else species
    if not species or not all(0 <= s < count for s in species):
        raise ValueError(f"{input_path}: --density-species must name species in [0, {count - 1}]")
    amplitude = first_float(params, AMPLITUDE_KEYS[layout])
    alfven_speed, omega_ci, ion_length = sheet_normalization(params, amplitude, density_ions)
    try:
        lh_length = lower_hybrid_length(params, amplitude, ion_length)
    except ValueError as error:
        raise ValueError(f"{input_path}: {error}") from None
    lengths = tuple(first_float(params, key) for key in ("Lx", "Ly", "Lz"))
    cells = tuple(int(first_float(params, key)) for key in ("nxc", "nyc", "nzc"))
    setup = SheetSetup(
        data_dir=data_dir, layout=layout, params=params, species=tuple(species),
        x_lines=tuple((x * cells[0], y * cells[1], sign * math.copysign(1.0, amplitude))
                      for _, x, y, sign in X_LINES[case]),
        b0=abs(amplitude), field_unit=abs(amplitude) * alfven_speed, ion_length=ion_length,
        cell=tuple(length / n for length, n in zip(lengths, cells)),
        half_thickness=first_float(params, "delta"), lh_split=LH_SPLIT / lh_length)
    cycles = snapshot_cycles(data_dir, layout, params, max_cycle)
    if len(cycles) < 2:
        raise ValueError(f"{data_dir}: the sheet instabilities need at least two field snapshots")

    measure = partial(measure_snapshot, setup=setup)
    results = []
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        for cycle, result in zip(cycles, pool.map(measure, cycles) if pool else map(measure, cycles)):
            results.append(result)
            print(f"measured cycle                 = {cycle} ({len(results)}/{len(cycles)})", flush=True)
    time = omega_ci * np.array([result[0] for result in results])
    paths = []
    for index, (sheet, x, y, _) in enumerate(X_LINES[case]):
        sheet_base = base.with_name(base.name + ("_" + sheet.replace(" ", "_") if sheet else ""))
        sheet_base.parent.mkdir(parents=True, exist_ok=True)
        paths += write_instability_plots(
            sheet_base, suffix, f"{label}, {sheet}" if sheet else label, time, [states[index] for _, states in results],
            tuple(length / ion_length for length in lengths), (x * lengths[0] / ion_length, y * lengths[1] / ion_length),
            setup.half_thickness / ion_length, lh_length)
    return paths


# Energetic particles of --particle-cycle.  Energies are kinetic, per physical particle, in m c^2 of a particle with
# qom = 1: a particle of species s has the mass 1/|qom_s| (c = 1, the velocities stay well below c).
PARTICLE_CHUNK = 1 << 24  # particles that a worker reads at once: 512 MB of velocity and charge
ENERGY_BINS = 1 << 16  # bins of ln E over ENERGY_RANGE, 0.06% wide, that place the percentile
ENERGY_RANGE = (1.0e-16, 1.0)
PARTICLE_SEED = 20261030  # the downsampling draws the same particles every time


@dataclass(frozen=True)
class ParticleChunk:
    """The particles start to stop of one species in one raw-appended piece of a TPIC particle snapshot."""

    path: Path
    species: int
    mass: float  # 1/|qom|
    start: int
    stop: int
    arrays: tuple[tuple[str, int, str], ...]  # name, byte offset of the first value, NumPy dtype

    def read(self, name: str, components: int, indices: np.ndarray | None = None) -> np.ndarray:
        """Return the array name of the particles, shape (particles, components), or of those at indices only."""
        offset, dtype = next((offset, np.dtype(dtype)) for array, offset, dtype in self.arrays if array == name)
        offset += components * dtype.itemsize * self.start
        if indices is None:
            values = np.fromfile(self.path, dtype=dtype, count=components * (self.stop - self.start), offset=offset)
        else:
            values = np.memmap(self.path, dtype=dtype, mode="r", offset=offset,
                               shape=(components * (self.stop - self.start),)).reshape(-1, components)[indices]
        return values.reshape(-1, components)


def particle_chunks(data_dir: Path, cycle: int, species: list[int],
                    params: dict[str, list[str]]) -> tuple[float, Path, list[ParticleChunk]]:
    """Return the time of the particle snapshot of cycle, its manifest and its particles of species in chunks.

    TPIC writes the particles of a rank into one piece species after species, so a species is one run of the
    species array of the piece, which bisection finds.
    """
    manifest = data_dir / f"particles_t{cycle:06d}.pvtp"
    if not manifest.exists():
        cycles = sorted(int(path.stem[11:]) for path in data_dir.glob("particles_t*.pvtp"))
        raise FileNotFoundError(f"{data_dir}: no {manifest.name}; the particle snapshots are those of cycles "
                                f"{', '.join(map(str, cycles)) or 'none'}")
    root = ET.parse(manifest).getroot()
    time = float(next(array.text for array in root.iter("DataArray") if array.get("Name") == "TimeValue"))
    types = {"Float64": "f8", "Float32": "f4", "UInt64": "u8", "Int64": "i8", "UInt32": "u4", "Int32": "i4"}
    chunks = []
    for piece in root.iter("Piece"):
        path = data_dir / piece.get("Source")
        with path.open("rb") as stream:
            head = stream.read(1 << 20)
        marker = re.search(rb'<AppendedData\b[^>]*encoding="raw"[^>]*>\s*_', head)
        if marker is None:
            raise ValueError(f"{path}: no raw appended data")
        header = ET.fromstring(head[:marker.start()] + b"</VTKFile>")
        endian = "<" if header.get("byte_order") == "LittleEndian" else ">"
        size = 8 if header.get("header_type") == "UInt64" else 4  # the byte count before each array
        arrays = tuple((array.get("Name"), marker.end() + int(array.get("offset")) + size,
                        endian + types[array.get("type")])
                       for array in header.iter("DataArray") if array.get("format") == "appended")
        count = int(header.find(".//Piece").get("NumberOfPoints"))
        offset, dtype = next((offset, np.dtype(dtype)) for name, offset, dtype in arrays if name == "species")
        with path.open("rb", buffering=0) as stream:
            # One value at a time: NumPy would copy the whole array of an unaligned memmap to bisect it.
            def kind(index: int) -> int:
                return int(np.frombuffer(os.pread(stream.fileno(), dtype.itemsize, offset + index * dtype.itemsize),
                                         dtype)[0])

            if np.any(np.diff([kind(int(index)) for index in np.linspace(0, count - 1, min(count, 257))]) < 0):
                raise ValueError(f"{path}: the particles are not grouped by species in increasing order")
            for s in species:
                low, high = bisect_left(range(count), s, key=kind), bisect_right(range(count), s, key=kind)
                chunks += [ParticleChunk(path, s, 1.0 / abs(float(params["qom"][s])), start,
                                         min(start + PARTICLE_CHUNK, high), arrays)
                           for start in range(low, high, PARTICLE_CHUNK)]
    return time, manifest, chunks


def chunk_energy(chunk: ParticleChunk) -> tuple[np.ndarray, np.ndarray]:
    """Return the kinetic energy and the weight |q| of the particles of chunk."""
    velocity = chunk.read("velocity", 3)
    return 0.5 * chunk.mass * np.einsum("ij,ij->i", velocity, velocity), np.abs(chunk.read("charge", 1)[:, 0])


def energy_bins(energy: np.ndarray) -> np.ndarray:
    """Return the bins of ln E of the energies."""
    scale = ENERGY_BINS / math.log(ENERGY_RANGE[1] / ENERGY_RANGE[0])
    bins = np.log(np.maximum(energy, ENERGY_RANGE[0]) / ENERGY_RANGE[0]) * scale
    return np.clip(bins.astype(np.int64), 0, ENERGY_BINS - 1)


def bin_edge(index: int | np.ndarray) -> float | np.ndarray:
    """Return the lowest energy of bin index."""
    return ENERGY_RANGE[0] * (ENERGY_RANGE[1] / ENERGY_RANGE[0]) ** (np.asarray(index) / ENERGY_BINS)


def chunk_histogram(chunk: ParticleChunk) -> np.ndarray:
    """Return the weight of the particles of chunk in each energy bin."""
    energy, weight = chunk_energy(chunk)
    return np.bincount(energy_bins(energy), weights=weight, minlength=ENERGY_BINS)


def chunk_sample(chunk: ParticleChunk, index: int, first_bin: int, rate: float) -> dict[str, np.ndarray]:
    """Draw each particle of chunk in bin first_bin or above with the probability rate*|q|; return the position,
    energy, weight and species of those drawn.  index seeds the draw."""
    energy, weight = chunk_energy(chunk)
    above = np.flatnonzero(energy_bins(energy) >= first_bin)
    drawn = above[np.random.default_rng([PARTICLE_SEED, index]).random(above.size) < rate * weight[above]]
    return {"position": chunk.read("position", 3, drawn), "energy": energy[drawn], "weight": weight[drawn],
            "species": np.full(drawn.size, chunk.species, dtype=np.int16)}


def percentile_bin(histogram: np.ndarray, percentile: float) -> int:
    """Return the energy bin of the percentile of the weight in histogram, shape (species, bin): the bins from it on
    hold 1 - percentile/100 of the weight, to one bin."""
    cumulative = np.cumsum(histogram.sum(axis=0))
    return int(np.searchsorted(cumulative, percentile / 100.0 * cumulative[-1]))


def sample_particles(data_dir: Path, cycle: int, species: list[int], params: dict[str, list[str]], percentile: float,
                     points: int, workers: int) -> dict[str, np.ndarray]:
    """Select the particles of species above the percentile of energy and draw about points of them.

    The percentile weighs a macroparticle by its |q|, the physical particles it carries, and so does the draw:
    the drawn particles are a fair sample of the physical particles above it.  A first pass bins the energies of
    every particle, which places the percentile; a second draws from above it and reads the positions of those
    drawn alone.
    """
    time, manifest, chunks = particle_chunks(data_dir, cycle, species, params)
    if not chunks:
        raise ValueError(f"{manifest}: no particle of species {', '.join(map(str, species))}")
    total = sum(chunk.stop - chunk.start for chunk in chunks)
    histogram = np.zeros((len(species), ENERGY_BINS))
    sample = []
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        for name in ("binned", "drawn"):
            if name == "binned":
                work = pool.map(chunk_histogram, chunks) if pool else map(chunk_histogram, chunks)
            else:
                first_bin = percentile_bin(histogram, percentile)
                draw = partial(chunk_sample, first_bin=first_bin, rate=points / histogram[:, first_bin:].sum())
                work = pool.map(draw, chunks, range(len(chunks))) if pool else map(draw, chunks, range(len(chunks)))
            started, done, reported = perf_counter(), 0, 0
            for chunk, result in zip(chunks, work):
                if name == "binned":
                    histogram[species.index(chunk.species)] += result
                else:
                    sample.append(result)
                done += chunk.stop - chunk.start
                if 10 * done >= (reported + 1) * total or done == total:
                    reported = 10 * done // total
                    elapsed = perf_counter() - started
                    print(f"particles {name:<20} = {done}/{total} in {elapsed:.0f} s "
                          f"({32.0e-9 * done / max(elapsed, 1.0e-9):.2g} GB/s of velocity and charge)", flush=True)
    sample = {key: np.concatenate([part[key] for part in sample]) for key in sample[0]}
    return {**sample, "time": np.float64(time), "histogram": histogram, "particles": np.int64(total), "manifest_mtime": np.int64(manifest.stat().st_mtime_ns)}


def species_name(params: dict[str, list[str]], s: int) -> str:
    """Return "sheet electrons", "background ions" or the like."""
    kind = "electrons" if float(params["qom"][s]) < 0.0 else "ions"
    sheet = s in ion_species(params, "sheet", -1.0 if kind == "electrons" else 1.0)
    return f"{'sheet' if sheet else 'background'} {kind}"


def energetic_particles(cache: Path, run_dir: Path, input_path: Path, cycle: int, species: list[int] | None,
                        percentile: float, points: int, density_ions: str, workers: int) -> dict:
    """Return at most points of the particles of species above the percentile of energy at cycle, from the sample
    in cache or drawn into it: positions in d_i, energies in T0 = m*uth^2 of the first species, and their setting.

    The sample serves again while the snapshot and the species are the same, its percentile is not higher and it
    holds at least points particles above the percentile: those drawn above a higher percentile are a fair sample
    of the particles above it, and a uniform subset of a fair sample is a fair sample.
    """
    params, data_dir, layout = open_deck(run_dir, input_path)
    if layout != TPIC_LAYOUT:
        raise ValueError(f"{input_path}: the particles are read from TPIC particles_t*.pvtp snapshots only")
    count = int(first_float(params, "ns"))
    species = [s for s in range(count) if float(params["qom"][s]) < 0.0] if species is None else species
    if not species or not all(0 <= s < count for s in species) or len(set(species)) < len(species):
        raise ValueError(f"{input_path}: --particle-species must name distinct species in [0, {count - 1}]")
    amplitude = first_float(params, AMPLITUDE_KEYS[layout])
    alfven_speed, omega_ci, ion_length = sheet_normalization(params, amplitude, density_ions)
    thermal_speed = float(params["uth"][species[0]])
    temperature = thermal_speed ** 2 / abs(float(params["qom"][species[0]]))

    sample = None
    if cache.exists():
        with np.load(cache) as stored:
            sample = {key: stored[key] for key in stored.files}
        manifest = data_dir / f"particles_t{cycle:06d}.pvtp"
        if (not manifest.exists() or sample["manifest_mtime"] != manifest.stat().st_mtime_ns
                or list(sample["selected_species"]) != species or sample["percentile"] > percentile
                or np.count_nonzero(energy_bins(sample["energy"])
                                    >= percentile_bin(sample["histogram"], percentile)) < points):
            sample = None
        else:
            print(f"particle sample read from      = {cache}")
    if sample is None:
        sample = sample_particles(data_dir, cycle, species, params, percentile, points, workers)
        sample.update(selected_species=np.array(species), percentile=np.float64(percentile))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **sample)
        print(f"particle sample written to     = {cache}")
    histogram = sample["histogram"]
    first_bin = percentile_bin(histogram, percentile)
    threshold = float(bin_edge(first_bin))
    keep = np.flatnonzero(energy_bins(sample["energy"]) >= first_bin)
    if keep.size > points:
        keep = np.sort(np.random.default_rng(PARTICLE_SEED).choice(keep, points, replace=False))
    print(f"particle snapshot              = cycle {cycle}, Omega_ci*t = {omega_ci * float(sample['time']):.6g}, "
          f"{int(sample['particles'])} macroparticles of species {', '.join(map(str, species))}")
    print(f"energy threshold               = {threshold / temperature:.6g} T0 ({percentile:g}th percentile by |q|; "
          f"{100.0 * histogram[:, first_bin:].sum() / histogram.sum():.4g}% of the weight above it; "
          f"T0 = m*uth^2 of species {species[0]})")
    # The speed of that energy, for each mass among the species.
    masses = {}
    for s in species:
        masses.setdefault(abs(float(params["qom"][s])), []).append(s)
    for qom, members in masses.items():
        speed = math.sqrt(2.0 * threshold * qom)
        print(f"speed threshold                = |v| >= {speed:.6g} c = {speed / thermal_speed:.6g} uth = "
              f"{speed / alfven_speed:.6g} v_A (species {', '.join(map(str, members))}; uth = {thermal_speed:g} c of "
              f"species {species[0]}, v_A = {alfven_speed:.6g} c)")
    for row, s in enumerate(species):
        print(f"{'  species ' + str(s):<31}= {species_name(params, s)}: "
              f"{100.0 * histogram[row, first_bin:].sum() / histogram[row].sum():.4g}% of its weight above, "
              f"{np.count_nonzero(sample['species'][keep] == s)} drawn")
    return {"position": sample["position"][keep] / ion_length, "energy": sample["energy"][keep] / temperature,
            "species": sample["species"][keep], "threshold": threshold / temperature, "histogram": histogram,
            "temperature": temperature,
            "names": [f"{species_name(params, s)} (species {s})" for s in species],
            "lengths": np.array([first_float(params, key) for key in ("Lx", "Ly", "Lz")]) / ion_length}


def particle_energy_range(sample: dict, style: RenderStyle) -> tuple[float, float]:
    """Return the energy range, in T0, that colours the particles of sample: that of style, or by default from the
    threshold to the 99.9th percentile of their energies."""
    energy = sample["energy"]
    return style.particle_energy_range or (sample["threshold"],
                                           float(np.percentile(energy, 99.9)) if energy.size else 1.0)


def analyze_particles(base: Path, suffix: str, run_dir: Path, input_path: Path, cycles: list[int],
                      species: list[int] | None, percentile: float, points: int, density_ions: str, workers: int,
                      style: RenderStyle) -> list[Path]:
    """Plot the particles of species above the percentile of energy at each of cycles; return the paths.

    The sample drawn goes to <base>_particles_t<cycle>.npz and serves again as energetic_particles states.
    """
    paths = []
    for cycle in cycles:
        cache = base.with_name(f"{base.name}_particles_t{cycle:06d}.npz")
        sample = energetic_particles(cache, run_dir, input_path, cycle, species, percentile, points, density_ions,
                                     workers)
        energy_range = particle_energy_range(sample, style)
        tag = f"p{percentile:g}_t{cycle:06d}{suffix}"
        view = base.with_name(f"{base.name}_particles_3d_{tag}")
        save_image(view, render_particles(sample["position"], sample["energy"], sample["lengths"], energy_range,
                                          style))
        projections = base.with_name(f"{base.name}_particles_{tag}")
        write_particle_projections(projections, sample["position"], sample["energy"], sample["species"],
                                   sample["lengths"], energy_range, sample["histogram"], sample["temperature"],
                                   sample["threshold"], percentile, sample["names"], style)
        paths += [view, projections, cache]
    return paths


def energy_colormap(style: RenderStyle) -> LinearSegmentedColormap:
    """Return the colormap of style as a Matplotlib colormap."""
    return LinearSegmentedColormap.from_list("energy", [(p, np.array(rgb) / 255.0) for p, rgb in style.colormap])


def render_particles(position: np.ndarray, energy: np.ndarray, lengths: np.ndarray,
                     energy_range: tuple[float, float], style: RenderStyle) -> np.ndarray:
    """Return the 3D view of the particles at position (d_i) coloured by energy, in the box of the run."""
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkLogger

    vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)
    plotter = pv.Plotter(off_screen=True, window_size=list(style.window_size))
    plotter.set_background(style.background)
    order = np.argsort(energy)  # the most energetic drawn last
    cloud = pv.PolyData(np.ascontiguousarray(position[order], dtype=np.float32))
    cloud["energy"] = energy[order]
    plotter.add_mesh(
        cloud, scalars="energy", cmap=energy_colormap(style), clim=energy_range, log_scale=True,
        render_points_as_spheres=True, point_size=style.particle_point_size,
        scalar_bar_args={"vertical": True, "position_x": style.colorbar_position[0],
                         "position_y": style.colorbar_position[1], "height": 0.75, "width": 0.05,
                         "title_font_size": style.colorbar_title_size, "label_font_size": style.colorbar_label_size,
                         "color": style.text_color, "fmt": "%.2g", "n_labels": 5, "title": "E/T0"})
    plotter.add_axes(color=style.text_color, line_width=4, label_size=(0.35, 0.14),
                     viewport=(0.0, 0.0, style.orientation_axes_size, style.orientation_axes_size))
    aim_camera(plotter, np.stack([np.zeros(3), lengths], axis=1), (style.camera_azimuth, style.camera_elevation),
               style)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return trim_margin(image, style)[0]


def write_particle_projections(path: Path, position: np.ndarray, energy: np.ndarray, species: np.ndarray,
                               lengths: np.ndarray, energy_range: tuple[float, float], histogram: np.ndarray,
                               temperature: float, threshold: float, percentile: float, names: list[str],
                               style: RenderStyle) -> None:
    """Plot the particles projected on the x-y, z-y and x-z planes, coloured by energy/T0, with the energy
    distribution of each species beside them: histogram, its weight in each energy bin, and the threshold."""
    order = np.argsort(energy)
    position, energy = position[order], energy[order]
    fig = plt.figure(figsize=(16.0, 13.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(lengths[0], lengths[2]), height_ratios=(lengths[1], lengths[2]))
    norm = LogNorm(*energy_range)
    panels = (((0, 0), 0, 1), ((0, 1), 2, 1), ((1, 0), 0, 2))
    for (row, column), across, up in panels:
        axis = fig.add_subplot(grid[row, column])
        dots = axis.scatter(position[:, across], position[:, up], c=energy, s=style.particle_dot_size,
                            cmap=energy_colormap(style), norm=norm, linewidths=0.0, rasterized=True)
        axis.set_xlim(0.0, lengths[across])
        axis.set_ylim(0.0, lengths[up])
        axis.set_aspect("equal")
        axis.set_xlabel(f"{'xyz'[across]} $(d_i)$")
        axis.set_ylabel(f"{'xyz'[up]} $(d_i)$")
    bar = fig.colorbar(dots, ax=fig.axes, location="top", shrink=0.6, aspect=40, pad=0.02, label=r"$E/T_0$")
    # Ticks at the lower end and at 1, 2 and 5 times the powers of ten above it, labelled plainly.
    powers = 10.0 ** np.arange(math.floor(math.log10(energy_range[0])), math.ceil(math.log10(energy_range[1])) + 1)
    ticks = np.outer(powers, (1.0, 2.0, 5.0)).ravel()
    bar.set_ticks([energy_range[0], *ticks[(ticks >= 1.25 * energy_range[0]) & (ticks <= energy_range[1])]])
    bar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3g"))
    bar.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axis = fig.add_subplot(grid[1, 1])
    # The fraction of the physical particles of each species per unit of ln E, on 1024 bins.
    group = ENERGY_BINS // 1024
    coarse = histogram.reshape(len(names), -1, group).sum(axis=2)
    edges = bin_edge(np.arange(0, ENERGY_BINS + 1, group)) / temperature
    centres = np.sqrt(edges[1:] * edges[:-1])
    shares = coarse / coarse.sum(axis=1, keepdims=True) / math.log(edges[1] / edges[0])
    for row, name in enumerate(names):
        shown = shares[row] > 0.0
        axis.loglog(centres[shown], shares[row, shown], label=name, color=RUN_COLORS[row % len(RUN_COLORS)],
                    linewidth=2.5)
    axis.axvspan(threshold, edges[-1], color="0.85", zorder=0)
    axis.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                 label=f"{percentile:g}th percentile")
    peak = shares.max()
    axis.set_ylim(1.0e-6 * peak, 2.0 * peak)
    # From a thousandth of the peak on the cold side, which few particles sample, to a millionth on the hot side.
    axis.set_xlim(edges[np.flatnonzero((shares >= 1.0e-3 * peak).any(axis=0))[0]],
                  edges[np.flatnonzero((shares >= 1.0e-6 * peak).any(axis=0))[-1] + 1])
    axis.set_xlabel(r"$E/T_0$")
    axis.set_ylabel(r"$dN/N\,d\ln E$")
    axis.legend(fontsize=0.7 * FONT_SIZES.legend, loc="lower left")
    save_figure(fig, path, dpi=200)
    plt.close(fig)


def print_data(run: Run, window: float) -> None:
    """Print the z averages of every snapshot: the flux, the unaveraged rates and, with a window, their average."""
    columns = [("Omega_ci*t", run.time_ci), ("flux/(B0*d_i)", run.flux.mean(axis=1)),
               ("flux rate", run.unaveraged_flux_rate.mean(axis=1)),
               ("Ez Omega_ci*t", run.field_time_ci), ("Ez rate", run.unaveraged_field_rate.mean(axis=1))]
    if window > 0.0:
        columns += [("flux rate, window", run.flux_rate.mean(axis=1)),
                    ("Ez rate, window", run.field_rate.mean(axis=1))]
    print(f"data of {run.label}, rates in B0*v_A")
    print(" ".join(f"{name:>17}" for name, _ in columns))
    for row in zip(*(values for _, values in columns)):
        print(" ".join(f"{value:17.9g}" for value in row))


def print_run(run: Run) -> None:
    mean_rate = run.flux_rate.mean(axis=1)
    peak = int(np.nanargmax(mean_rate))
    print(f"run                            = {run.label}")
    print(f"  input deck                   = {run.input_path}")
    print(f"  case / layout                = {run.case} / {run.layout}")
    print(f"  B0*v_A (field unit)          = {run.field_unit:.9g}")
    print(f"  v_A/c                        = {run.alfven_speed:.9g}")
    print(f"  Omega_ci/omega_pi            = {run.omega_ci:.9g}")
    print(f"  d_i [code length]            = {run.ion_length:.9g}")
    print(f"  X line (x, y)                = ({run.x_line[0]:.9g}, {run.x_line[1]:.9g})")
    print(f"  snapshots                    = {run.time_ci.size}")
    print(f"  Omega_ci*t range             = [{run.time_ci[0]:.9g}, {run.time_ci[-1]:.9g}]")
    print(f"  flux/(B0*d_i) first, last    = {run.flux[0].mean():.9g}, {run.flux[-1].mean():.9g}")
    print(f"  peak flux rate (z average)   = {mean_rate[peak]:.9g} at Omega_ci*t = {run.time_ci[peak]:.9g}")


def print_energy(energy: EnergyHistory) -> None:
    change = energy.total / energy.total[0] - 1.0
    largest = int(np.argmax(np.abs(change)))
    print(f"total energy                   = {energy.label}")
    print(f"  records                      = {energy.total.size}")
    print(f"  first, last                  = {energy.total[0]:.9g}, {energy.total[-1]:.9g}")
    print(f"  largest (E - E0)/E0          = {change[largest]:.3g} at Omega_ci*t = {energy.time_ci[largest]:.9g}")


def print_interval(runs: list[Run]) -> None:
    """Compare the runs over the Omega_ci*t interval that all of them cover."""
    start = max(run.time_ci[0] for run in runs)
    end = min(run.time_ci[-1] for run in runs)
    if len(runs) == 1:
        # One run: its second half, after the start-up of the sheet.
        start = runs[0].time_ci[runs[0].time_ci.size // 2]
    if not start < end:
        print("the runs share no Omega_ci*t interval")
        return
    print(f"interval Omega_ci*t            = [{start:.9g}, {end:.9g}]")
    for run in runs:
        flux = run.flux.mean(axis=1)
        gain = np.interp(end, run.time_ci, flux) - np.interp(start, run.time_ci, flux)
        field_rate = run.field_rate.mean(axis=1)
        field_rate = field_rate[(run.field_time_ci >= start) & (run.field_time_ci <= end) & np.isfinite(field_rate)]
        print(f"  {run.label}")
        print(f"    flux gain / interval       = {gain / (end - start):.9g}")
        print(f"    Ez rate, mean +/- std      = {field_rate.mean():.9g} +/- {field_rate.std():.3g}"
              f" ({field_rate.size} snapshots)")


def print_comparison(runs: list[Run]) -> None:
    """Compare every other run with the first on the times of the first that both cover."""
    first = runs[0]
    for run in runs[1:]:
        print(f"{run.label} minus {first.label}")
        for member, name in (("flux", "flux"), ("flux_rate", "flux rate"), ("field_rate", "Ez rate")):
            difference = difference_to_first(first, run, member)[1]
            difference = difference[np.isfinite(difference)]
            if difference.size == 0:
                print(f"  {name + ', mean, RMS':<29}= none: no common snapshot")
                continue
            print(f"  {name + ', mean, RMS':<29}= {difference.mean():.9g}, {math.sqrt((difference**2).mean()):.9g}"
                  f" ({difference.size} snapshots)")
        start, end = max(first.time_ci[0], run.time_ci[0]), min(first.time_ci[-1], run.time_ci[-1])
        for each in (first, run):
            rate = each.flux_rate.mean(axis=1)
            inside = np.isfinite(rate) & (each.time_ci >= start) & (each.time_ci <= end)
            if inside.any():
                peak = int(np.argmax(np.where(inside, rate, -np.inf)))
                print(f"  {'peak flux rate, ' + each.label:<28} = {rate[peak]:.9g}"
                      f" at Omega_ci*t = {each.time_ci[peak]:.9g}")
        gains = [flux_gain(each) for each in (first, run)]
        level = 0.5 * min(gain[-1] for _, gain in gains)
        if level <= 0.0:
            print("  time lag at half flux gain   = none: the flux of a run does not grow")
            continue
        # The first snapshot at or above the level and the one before it bracket the crossing.
        crossings = []
        for time, gain in gains:
            index = int(np.argmax(gain >= level))
            crossings.append(np.interp(level, gain[index - 1:index + 1], time[index - 1:index + 1]))
        print(f"  time lag at half flux gain   = {crossings[1] - crossings[0]:.9g}"
              f" (gain {level:.9g} at Omega_ci*t = {crossings[0]:.9g} and {crossings[1]:.9g})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconnection rate on the X line of GEM Harris-sheet runs; several runs are compared.",
        epilog="The header of this script shows example commands and states the results written, the method "
        "and the units.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        default=[Path(".")],
        help="simulation launch/run directories; one directory serves every --input deck (default: current directory)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="input deck of each run, relative to the working directory (default: the only *.inp in each RUN_DIR)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="legend label of each run (default: the last two parts of its RUN_DIR; "
        "the deck name when runs share a RUN_DIR)",
    )
    parser.add_argument(
        "--max-cycle",
        type=int,
        default=None,
        metavar="CYCLE",
        help="analyze only the field snapshots and the energy records up to cycle CYCLE, in every run; "
        "the field files of a later cycle are not read (default: all cycles)",
    )
    parser.add_argument(
        "--box-cells",
        type=int,
        default=8,
        help="side in cells of the x-y square around the X line that averages Ez (default: 8)",
    )
    parser.add_argument(
        "--time-average",
        type=float,
        default=0.0,
        metavar="WINDOW",
        help="average both rates over WINDOW/Omega_ci centred on each snapshot; a snapshot closer than "
        "WINDOW/2 to an end of its run gets no rate (default: 0, no average)",
    )
    parser.add_argument(
        "--print-data",
        action="store_true",
        help="print the z averages of every snapshot of every run: the flux, the rates before the time average "
        "and, with --time-average, the averaged rates",
    )
    parser.add_argument(
        "--alfven-density",
        choices=("sheet", "background"),
        default="sheet",
        help="ions whose density sets v_A: the drifting sheet ions or the background (default: sheet)",
    )
    parser.add_argument(
        "--output",
        "--plot",
        dest="plot",
        type=Path,
        default=None,
        help="plot path (default: RUN_DIR/reconnection_rate.png; reconnection_comparison.png here for several decks); "
        "the plots of the flux rate alone and of the total energy go beside it, with _flux_rate and _energy "
        "added to the name",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path (default: beside the plot default, with the extension .csv)",
    )
    parser.add_argument(
        "--render-cycle",
        type=int,
        nargs="+",
        default=None,
        metavar="CYCLE",
        help="instead of the rate analysis, render the 3D view of the ion density and the field lines through the "
        "X lines of each run at each CYCLE, with PyVista; an image goes to RUN_DIR/<deck name>_3d_t<CYCLE>.png, "
        "or to the --output path with _t<CYCLE> added",
    )
    parser.add_argument(
        "--render-view",
        choices=RENDER_VIEWS,
        default="x-line",
        help="what the 3D view of --render-cycle shows on the density: the field lines through the X lines "
        "(x-line, the whole box), or a zoom on the bent current sheet in the density alone (kink), on the "
        "isosurfaces of the lower-hybrid Ez fluctuation (lower-hybrid) or on the field lines around the flux ropes "
        "coloured by Bz (ropes); the images go to RUN_DIR/<deck name>_3d[_VIEW]_t<CYCLE>.png (default: x-line)",
    )
    parser.add_argument(
        "--render-region",
        type=float,
        nargs=6,
        default=None,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        help="the box, in d_i, that the 3D view shows (default: the whole box for x-line, a zoom chosen from the "
        "data for the other views)",
    )
    parser.add_argument(
        "--render-inset",
        type=float,
        nargs=6,
        default=None,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        help="add to the 3D view a zoom on this box, in d_i, with the same colour scales and overlay level; its box is "
        "outlined in the main view and the colour bars are shared",
    )
    parser.add_argument(
        "--render-inset-view",
        choices=RENDER_VIEWS,
        default=None,
        help="what the zoom of --render-inset shows, as --render-view (default: the view of the main plot)",
    )
    parser.add_argument(
        "--render-style",
        nargs="+",
        default=[],
        metavar="FIELD=VALUE",
        help="override fields of RENDER_STYLE in the script for every run of this call, each value a Python "
        "literal or else a string, as in zoom_opacity_unit_distance=1.5 overlay_colorbar=False",
    )
    parser.add_argument(
        "--render-particles",
        type=float,
        nargs="+",
        default=None,
        metavar="P",
        help="draw over the 3D view the --particle-species above the P-th percentile of energy, one image per P, "
        "<image name>_particles_p<P>_t<CYCLE>.png, from the sample <deck name>_particles_t<CYCLE>.npz of "
        "--particle-cycle beside the image, drawn first when it does not serve",
    )
    parser.add_argument(
        "--density-species",
        "--render-species",
        dest="density_species",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="species, by their index in the deck, whose summed density the 3D view and the drift-kink slices show "
        "(default: the sheet ions, those that drift)",
    )
    parser.add_argument(
        "--instabilities",
        action="store_true",
        help="instead of the rate analysis, follow the lower-hybrid drift waves, the drift kink and the flux ropes "
        "of the current sheet through every snapshot up to --max-cycle; the plots go to "
        "RUN_DIR/<deck name>_lhdi.png, _kink.png and _flux_ropes.png, the measures to _instabilities.npz",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes that read and measure snapshots at once for --instabilities, each holding about ten copies "
        "of one field component, or particle chunks for --particle-cycle, each about 1.5 GB (default: 1)",
    )
    parser.add_argument(
        "--particle-cycle",
        type=int,
        nargs="+",
        default=None,
        metavar="CYCLE",
        help="instead of the rate analysis, plot the most energetic particles of the TPIC particle snapshot of each "
        "CYCLE: a 3D view, RUN_DIR/<deck name>_particles_3d_p<P>_t<CYCLE>.png, and their projections with the "
        "energy distributions, _particles_p<P>_t<CYCLE>.png; the sample drawn goes to _particles_t<CYCLE>.npz and "
        "serves again for the same or a higher --particle-percentile P",
    )
    parser.add_argument(
        "--particle-species",
        type=int,
        nargs="+",
        default=None,
        metavar="S",
        help="species, by their index in the deck, of --particle-cycle (default: the electrons, qom < 0)",
    )
    parser.add_argument(
        "--particle-percentile",
        type=float,
        default=80.0,
        metavar="P",
        help="keep the particles above the P-th percentile of the kinetic energy of the --particle-species, each "
        "macroparticle weighed by its |q| (default: 80, the most energetic 20%%)",
    )
    parser.add_argument(
        "--particle-points",
        type=int,
        default=200000,
        metavar="N",
        help="draw about N of the particles kept, each with a probability proportional to its |q|, for the plots "
        "(default: 200000)",
    )
    return parser


def parse_render_style(items: list[str]) -> RenderStyle:
    """Return RENDER_STYLE with the --render-style changes FIELD=VALUE of items."""
    changes = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in {field.name for field in fields(RenderStyle)} or not value:
            raise ValueError(f"--render-style {item}: expected FIELD=VALUE with a field of RenderStyle")
        try:
            changes[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            changes[name] = value  # a bare string, such as a colour name
    return replace(RENDER_STYLE, **changes)


def analyze(args: argparse.Namespace) -> None:
    run_dirs: list[Path] = args.run_dir
    if args.input is not None and len(run_dirs) == 1:
        run_dirs = run_dirs * len(args.input)
    if args.input is not None and len(args.input) != len(run_dirs):
        raise ValueError("--input needs one deck per run directory, or one run directory for all decks")
    if args.labels is not None and len(args.labels) != len(run_dirs):
        raise ValueError("--labels needs one value per run")
    if args.time_average < 0.0:
        raise ValueError("--time-average must not be negative")
    if args.max_cycle is not None and args.max_cycle < 0:
        raise ValueError("--max-cycle must not be negative")
    max_cycle = math.inf if args.max_cycle is None else args.max_cycle
    inputs = [find_input_path(run_dir, explicit)
              for run_dir, explicit in zip(run_dirs, args.input or [None] * len(run_dirs))]
    # Runs that share a directory differ by their deck.
    shared = len(set(run_dirs)) < len(run_dirs)
    labels = args.labels or [input_path.stem if shared else "/".join(run_dir.resolve().parts[-2:])
                             for run_dir, input_path in zip(run_dirs, inputs)]

    def run_base(run_dir: Path, input_path: Path, label: str, name: str) -> tuple[Path, str]:
        """Return the path stem and suffix of the results of one run: RUN_DIR/<deck name><name> and .png, or the
        --output stem, with the label of the run when there are several runs, and the --output suffix."""
        if args.plot is None:
            return run_dir / f"{input_path.stem}{name}", ".png"
        tag = "_" + re.sub(r"[^A-Za-z0-9.+-]+", "_", label) if len(inputs) > 1 else ""
        return args.plot.with_name(args.plot.stem + tag), args.plot.suffix or ".png"

    if args.render_cycle is not None:
        if min(args.render_cycle) < 0:
            raise ValueError("--render-cycle must not be negative")
        style = parse_render_style(args.render_style)
        for run_dir, input_path, label in zip(run_dirs, inputs, labels):
            view = "" if args.render_view == "x-line" else "_" + args.render_view.replace("-", "_")
            base, suffix = run_base(run_dir, input_path, label, "_3d" + view)
            region = None if args.render_region is None else np.reshape(args.render_region, (3, 2))
            inset = None if args.render_inset is None else np.reshape(args.render_inset, (3, 2))
            for cycle, percentile in itertools.product(args.render_cycle, args.render_particles or [None]):
                particles, name = None, f"{base.name}_t{cycle:06d}{suffix}"
                if percentile is not None:
                    # The sample of --particle-cycle for this deck, beside the image.
                    particles = energetic_particles(
                        base.with_name(f"{input_path.stem}_particles_t{cycle:06d}.npz"), run_dir, input_path, cycle,
                        args.particle_species, percentile, style.particle_overlay_points, args.alfven_density,
                        args.workers)
                    particles["range"] = particle_energy_range(particles, style)
                    name = f"{base.name}_particles_p{percentile:g}_t{cycle:06d}{suffix}"
                path = base.with_name(name)
                time, peak = render_snapshot(path, run_dir, input_path, label, cycle, args.density_species,
                                             args.alfven_density, style, args.render_view, region, inset,
                                             args.render_inset_view, particles)
                print(f"3D view written to             = {with_pdf(path)} "
                      f"(Omega_ci*t = {time:.9g}, largest density {peak:.6g})")
        return
    if args.instabilities:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        for run_dir, input_path, label in zip(run_dirs, inputs, labels):
            for path in analyze_instabilities(*run_base(run_dir, input_path, label, ""), run_dir, input_path, label,
                                              args.density_species, args.alfven_density, max_cycle, args.workers):
                print(f"instabilities written to       = {with_pdf(path)}")
        return
    if args.particle_cycle is not None:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        if not 0.0 <= args.particle_percentile < 100.0:
            raise ValueError("--particle-percentile must lie in [0, 100)")
        if args.particle_points < 1:
            raise ValueError("--particle-points must be positive")
        style = parse_render_style(args.render_style)
        for run_dir, input_path, label in zip(run_dirs, inputs, labels):
            for path in analyze_particles(*run_base(run_dir, input_path, label, ""), run_dir, input_path,
                                          args.particle_cycle, args.particle_species, args.particle_percentile,
                                          args.particle_points, args.alfven_density, args.workers, style):
                print(f"particles written to           = {with_pdf(path)}")
        return

    runs, energies = [], []
    for run_dir, input_path, label in zip(run_dirs, inputs, labels):
        deck_runs, energy = analyze_deck(run_dir, input_path, label, args.box_cells, args.alfven_density,
                                         args.time_average, max_cycle)
        runs += deck_runs
        energies.append(energy)

    default = run_dirs[0] / "reconnection_rate" if len(inputs) == 1 else Path("reconnection_comparison")
    plot_path = args.plot or default.with_suffix(".png")
    csv_path = args.csv or default.with_suffix(".csv")
    write_csv(csv_path, runs, args.time_average)
    write_plot(plot_path, runs, args.box_cells, args.time_average)
    flux_rate_plot_path = plot_path.with_name(f"{plot_path.stem}_flux_rate{plot_path.suffix}")
    title = rf"mean over $\Omega_{{ci}} \Delta t = {args.time_average:g}$ centred on each snapshot"
    write_line_plot(
        flux_rate_plot_path,
        [(run.label, run.time_ci, run.flux_rate.mean(axis=1), run.flux_rate.std(axis=1)) for run in runs],
        title * (args.time_average > 0.0), FLUX_RATE_PLOT_STYLE)
    energy_plot_path = plot_path.with_name(f"{plot_path.stem}_energy{plot_path.suffix}")
    logged = [energy for energy in energies if energy]
    if logged:
        write_line_plot(
            energy_plot_path,
            [(energy.label, energy.time_ci, energy.total / energy.total[0] - 1.0, None) for energy in logged],
            "", ENERGY_PLOT_STYLE)

    if args.print_data:
        for run in runs:
            print_data(run, args.time_average)
    print(f"Alfven-speed density           = {args.alfven_density} ions")
    print(f"Ez box [cells]                 = {args.box_cells} x {args.box_cells}")
    print(f"rate average [1/Omega_ci]      = {args.time_average:g}")
    print(f"last cycle analyzed            = {'that of each run' if args.max_cycle is None else args.max_cycle}")
    for run in runs:
        print_run(run)
    for label, energy in zip(labels, energies):
        if energy:
            print_energy(energy)
        else:
            print(f"total energy                   = no log in the output directory of {label}")
    print_interval(runs)
    print_comparison(runs)
    print(f"CSV written to                 = {csv_path}")
    print(f"plot written to                = {with_pdf(plot_path)}")
    print(f"flux rate plot written to      = {with_pdf(flux_rate_plot_path)}")
    if logged:
        print(f"energy plot written to         = {with_pdf(energy_plot_path)}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        analyze(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
