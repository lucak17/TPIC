"""Checks of tools/postprocessing/analyze_whistler_turbulence.py on synthetic data, without a run: the particle
snapshots read piece by piece and chunk by chunk against a direct computation, and the omega-k spectra of plane
whistlers of known frequency, direction and energy."""
from pathlib import Path
import math
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "postprocessing")]
try:
    import numpy as np
    import analyze_whistler_turbulence as turbulence
    from tpic_tools import particles
except ImportError:  # NumPy, SciPy and Matplotlib are optional analysis extras
    turbulence = None


def write_piece(path: Path, arrays: list) -> None:
    """A raw-appended PolyData piece as TPIC writes it: (name, VTK type, components, values) per array, the last
    one the points."""
    offset, descriptors, payload = 0, [], b""
    for name, kind, components, values in arrays:
        raw = values.astype("<u8" if kind == "UInt64" else "<f8").tobytes()
        descriptors.append(f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{components}" '
                           f'format="appended" offset="{offset}"/>')
        block = np.array([len(raw)], "<u8").tobytes() + raw
        payload += block
        offset += len(block)
    head = ('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
            f'header_type="UInt64">\n<PolyData><Piece NumberOfPoints="{len(arrays[0][3])}"><PointData>'
            + "".join(descriptors[:-1]) + "</PointData><Points>" + descriptors[-1]
            + '</Points></Piece></PolyData>\n<AppendedData encoding="raw">\n_')
    path.write_bytes(head.encode() + payload + b"\n</AppendedData>\n</VTKFile>\n")


def write_manifest(path: Path, time: float, sources: list) -> None:
    path.write_text('<?xml version="1.0"?>\n<VTKFile type="PPolyData"><PPolyData><FieldData>'
                    f'<DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">{time}</DataArray>'
                    "</FieldData>" + "".join(f'<Piece Source="{s}"/>' for s in sources) + "</PPolyData></VTKFile>\n")


@unittest.skipIf(turbulence is None, "NumPy, SciPy or Matplotlib unavailable")
class WhistlerTurbulenceTests(unittest.TestCase):
    def test_streamed_particle_snapshots(self):
        # Two snapshots of two pieces, electrons first with a few ions among them, read in chunks of 997 particles
        # so that chunks hold one species, the other or both: the distributions and the nearest-node ion density
        # must be those of the whole snapshot at once.
        rng = np.random.default_rng(2)
        cells, length = 8, 1.6
        deck = turbulence.Deck(path=Path("deck.inp"), save_dir="data", lengths=np.full(3, length),
                               cells=np.full(3, cells), b0=1.0, epsilon=0.0, electron=0, qom_e=-1.0, v_te=0.1,
                               omega_pe=1.0, d_e=1.0, omega_ce=1.0, ion=1)
        edges = np.linspace(-turbulence.VDF_LIMIT, turbulence.VDF_LIMIT, turbulence.VDF_BINS + 1)
        expected, ions = {}, np.zeros((cells,) * 3)
        defaults = particles.iter_species.__defaults__
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            for time, cycle in ((0.0, 0), (2.5, 10)):
                sources, velocities = [], []
                for rank in range(2):
                    n = 30011 + rank
                    species = np.sort(rng.integers(0, 2, n)).astype("<u8")
                    species[n // 3: n // 3 + 7] = 1
                    velocity = rng.normal(0.0, 0.12, (n, 3))
                    velocity[::611] *= 9.0  # some beyond VDF_LIMIT
                    charge = np.where(species == 0, -1.0, 1.0) * rng.uniform(0.5, 1.5, n)
                    position = rng.uniform(0.0, length, (n, 3))
                    position[:5] = length - 1e-9  # nearest to node 0 across the periodic end
                    source = f"particles_t{cycle:06d}_r{rank:06d}.vtp"
                    write_piece(data / source, [("velocity", "Float64", 3, velocity), ("charge", "Float64", 1, charge),
                                                ("species", "UInt64", 1, species),
                                                ("position", "Float64", 3, position)])
                    sources.append(source)
                    velocities.append(velocity[species == 0] / deck.v_te)
                    if cycle == 10:
                        node = np.floor(position / (length / cells) + 0.5).astype(int) % cells
                        np.add.at(ions, tuple(node[species == 1].T), charge[species == 1])
                write_manifest(data / f"particles_t{cycle:06d}.pvtp", time, sources)
                v = np.concatenate(velocities)
                par, perp = (np.histogram(v[:, a], edges)[0] for a in (2, 0))
                expected[time] = (par / np.diff(edges) / par.sum(), perp / np.diff(edges) / perp.sum(),
                                  np.histogram2d(v[:, 0], v[:, 2], (edges, edges))[0] / len(v))
            try:
                particles.iter_species.__defaults__ = (997,)
                result_edges, distributions = turbulence.read_distributions(data, deck)
                time, density = turbulence.deposit_density(data, deck, turbulence.Fourier(deck), 1, 2.4, 1e9)
            finally:
                particles.iter_species.__defaults__ = defaults
        np.testing.assert_array_equal(result_edges, edges)
        self.assertEqual([d[0] for d in distributions], [0.0, 2.5])
        for snapshot_time, *values in distributions:
            for value, whole in zip(values, expected[snapshot_time]):
                np.testing.assert_allclose(value, whole, rtol=1e-12, atol=0.0)
        self.assertEqual(time, 2.5)  # the snapshot nearest 2.4; k_max = 1e9 rolls nothing off
        np.testing.assert_allclose(density, ions / ions.mean() - 1.0, rtol=0.0, atol=1e-12)

    def test_omega_k_of_plane_whistlers(self):
        # Three plane whistlers of the cold relation, two along +k and one along -k, sampled as
        # whistlerTurbulence3D.inp writes B (every 25/omega_pe): each must peak at its frequency, with its sign, within
        # one step of the padded frequency grid, and carry its energy amplitude^2/2 (Parseval) in its (k_perp shell,
        # k_par plane) bin.
        cells, length, omega_ce = 40, 60.0, 1.0 / math.sqrt(5.0)
        deck = turbulence.Deck(path=Path("deck.inp"), save_dir="data", lengths=np.full(3, length),
                               cells=np.full(3, cells), b0=1.0, epsilon=0.0, electron=0, qom_e=-1.0, v_te=0.1,
                               omega_pe=1.0, d_e=1.0, omega_ce=omega_ce)
        fourier = turbulence.Fourier(deck)
        omega_k = turbulence.OmegaK(fourier, [0.2, 0.3, 0.4])
        k0, dx = 2.0 * math.pi / length, length / cells
        x = np.meshgrid(*[np.arange(cells) * dx] * 3, indexing="ij")
        waves = [((5, 0, 2), +1, 0.02), ((10, 0, 3), -1, 0.01), ((0, 7, 4), +1, 0.005)]  # index, sign, amplitude
        for step in range(41):
            t = 25.0 * step
            field = np.zeros((3, cells, cells, cells))
            field[2] += deck.b0
            for index, sign, amplitude in waves:
                k = k0 * np.array(index, float)
                omega = sign * turbulence.whistler_frequency(math.hypot(k[0], k[1]), k[2]) * omega_ce
                polarization = np.cross(k, [1.0, 0.0, 0.0] if index[0] == 0 else [0.0, 1.0, 0.0])
                polarization /= np.linalg.norm(polarization)
                field += amplitude * polarization[:, None, None, None] * np.cos(
                    k[0] * x[0] + k[1] * x[1] + k[2] * x[2] - omega * t)
            omega_k.add(t, fourier.modes(field, deck.b0))
        spectrum = omega_k.spectrum(0.0, 500.0, omega_ce)
        step = spectrum.omega[1] - spectrum.omega[0]
        self.assertAlmostEqual(spectrum.nyquist, math.pi / (25.0 * omega_ce))
        for index, sign, amplitude in waves:
            k_perp, k_par = k0 * math.hypot(index[0], index[1]), k0 * index[2]
            shell, plane = omega_k.shell(k_perp), omega_k.plane(k_par)
            power = spectrum.power[plane, shell]
            peak = spectrum.omega[np.argmax(power)]
            self.assertLessEqual(abs(peak - sign * turbulence.whistler_frequency(k_perp, k_par)), step)
            transfer = np.prod(np.cos(0.5 * k0 * np.array(index) * dx))  # the node filter that modes() divides out
            self.assertAlmostEqual(power.sum() * step / (0.5 * amplitude**2 / transfer**2), 1.0, places=9)
        # The plots draw without error.
        bins = [(omega_k.shell(k0 * 5), omega_k.plane(k0 * 2))]
        for figure in (turbulence.plot_omega_k(omega_k, [spectrum], [omega_k.plane(0.2)], 3.0),
                       turbulence.plot_omega_power(omega_k, [spectrum], bins)):
            turbulence.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
