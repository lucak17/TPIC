"""Independent format checks and production-writer round trips."""
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from tpic_tools.vtk import image_header, metadata, read_piece

FIXTURE = sys.argv[1:]
del sys.argv[1:]
try:
    import numpy as np
except ImportError:
    np = None

COMPONENTS = {"E": 3, "B": 3, "rho": 1}


class FieldReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="tpic-field-reader-")
        cls.root = Path(cls.temporary.name)
        subprocess.run([*FIXTURE, str(cls.root)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def folder(self, name, files):
        """Return a fresh directory holding copies of the given fixture files."""
        directory = self.root / name
        directory.mkdir()
        for source, target in files:
            shutil.copy(self.root / source, directory / target)
        return directory

    def test_production_writer(self):
        for folder, final_cycle in (("r1", 2), ("zero", 0), ("selected", 1)):
            names = ["E", "rho"] if folder == "selected" else ["E", "B", "rho"]
            expected = sorted(f"{name}_t{cycle:06d}.vti" for name in names for cycle in range(final_cycle + 1))
            self.assertEqual(sorted(path.name for path in (self.root / folder).iterdir()), expected)
            for name in names:
                for cycle in range(final_cycle + 1):
                    path = self.root / folder / f"{name}_t{cycle:06d}.vti"
                    _, image, box, arrays, _ = image_header(path)
                    self.assertEqual(box, (0, 7, 0, 2, 0, 2))
                    self.assertEqual(arrays, {name: COMPONENTS[name]})
                    self.assertEqual(metadata(image), (cycle, cycle * 0.125, cycle == final_cycle, "double/double"))
                    values = read_piece(path).values[name]
                    self.assertEqual(values[0], cycle * 1000)
                    self.assertEqual(values[-1], cycle * 1000 + 227 + 0.25 * (COMPONENTS[name] - 1))

    def test_independent_big_endian_fixture(self):
        # Deliberately authored independently of the VTK field engine and its metadata.
        data = (b'<VTKFile type="ImageData" byte_order="BigEndian" header_type="UInt64">'
                b'<ImageData WholeExtent="0 1 0 0 0 0"><Piece Extent="0 1 0 0 0 0">'
                b'<PointData><DataArray type="Float64" Name="rho" NumberOfComponents="1" '
                b'format="appended" offset="0"/></PointData></Piece></ImageData>'
                b'<AppendedData encoding="raw">_')
        path = self.root / "golden.vti"
        path.write_bytes(data + struct.pack(">Qdd", 16, -1.5, 2.25) + b'</AppendedData>\n</VTKFile>')
        self.assertEqual(list(read_piece(path).values["rho"]), [-1.5, 2.25])
        path.write_bytes(path.read_bytes()[:-20])
        with self.assertRaises(ValueError):
            read_piece(path)

    def test_cell_array(self):
        # A cell array of a 2 x 1 x 1 grid: CellData with one value per cell.
        data = (b'<VTKFile type="ImageData" byte_order="LittleEndian" header_type="UInt64">'
                b'<ImageData WholeExtent="0 2 0 1 0 1"><Piece Extent="0 2 0 1 0 1"><PointData/>'
                b'<CellData><DataArray type="Float64" Name="Gauss" NumberOfComponents="1" '
                b'format="appended" offset="0"/></CellData></Piece></ImageData>'
                b'<AppendedData encoding="raw">_')
        path = self.root / "cells.vti"
        path.write_bytes(data + struct.pack("<Qdd", 16, -1.5, 2.25) + b'</AppendedData>\n</VTKFile>')
        self.assertEqual(list(read_piece(path).values["Gauss"]), [-1.5, 2.25])

    def test_reject_corrupt_file(self):
        source = self.root / "r1/E_t000000.vti"
        for old, new in ((b'offset="0"', b'offset="1"'),
                         (b'type="Float64" Name="E"', b'type="Float32" Name="E"'),
                         (b'header_type="UInt64"', b'header_type="UInt32"'),
                         (b'<Piece Extent="0 7', b'<Piece Extent="0 6')):
            path = self.root / "corrupt.vti"
            path.write_bytes(source.read_bytes().replace(old, new, 1))
            with self.assertRaises(ValueError):
                read_piece(path)

    @unittest.skipIf(np is None, "NumPy analysis extra unavailable")
    def test_field_adapter(self):
        from tpic_tools.fields import FieldReader, field_data_exists
        for folder in ("r1", "zero", "selected"):
            with FieldReader(self.root / folder / "E") as reader:
                self.assertEqual(set(reader.available_variables()), {"Ex", "Ey", "Ez", "time"})
                cycle = reader.num_steps() - 1
                actual = reader.read("Ey", [cycle, 1])
                z, y, x = np.indices((3, 3, 8))
                np.testing.assert_array_equal(actual, cycle * 1000 + z * 100 + y * 10 + x + 0.25)
                self.assertEqual(reader.read("time", [cycle, 1])[0], cycle * 0.125)
        self.assertFalse(field_data_exists(self.root / "selected", "B"))
        # A species array beside its total is a different field.
        directory = self.folder("species", [(f"r1/rho_t{c:06d}.vti", f"rho_t{c:06d}.vti") for c in range(3)]
                                + [("r1/rho_t000000.vti", "rho_0_t000000.vti")])
        self.assertEqual(FieldReader(directory / "rho").num_steps(), 3)
        with self.assertRaisesRegex(ValueError, "single array rho_0"):
            FieldReader(directory / "rho_0")

    @unittest.skipIf(np is None, "NumPy analysis extra unavailable")
    def test_reject_inconsistent_records(self):
        from tpic_tools.fields import FieldReader
        # The final state of cycle 0 followed by later cycles.
        directory = self.folder("final_first", [("zero/E_t000000.vti", "E_t000000.vti"),
                                                ("r1/E_t000001.vti", "E_t000001.vti")])
        with self.assertRaisesRegex(ValueError, "before a final state"):
            FieldReader(directory / "E")
        directory = self.folder("truncated", [(f"r1/E_t{c:06d}.vti", f"E_t{c:06d}.vti") for c in range(3)])
        path = directory / "E_t000001.vti"
        path.write_bytes(path.read_bytes()[:-100])
        reader = FieldReader(directory / "E")
        reader.read("Ex", [0, 1])
        with self.assertRaisesRegex(ValueError, "truncated"):
            reader.read("Ex", [1, 1])
        with self.assertRaises(FileNotFoundError):
            FieldReader(directory / "B")

    @unittest.skipIf(np is None, "NumPy analysis extra unavailable")
    def test_analyzers_read_field_files(self):
        # Run all four analyzers' real data-loading paths on production VTK.
        # These are I/O smoke checks, not physical-growth qualification.
        try:
            import matplotlib
            import scipy
        except ImportError:
            self.skipTest("optional analysis dependencies unavailable")
        from types import SimpleNamespace
        sys.path.insert(0, str(TOOLS / "postprocessing"))
        import analyze_two_stream as two_stream
        import analyze_weibel as weibel
        import analyze_whistler as whistler
        import analyze_tgv as tgv
        from tpic_tools.fields import open_field_reader
        expected = sum(x * complex(np.cos(-2*np.pi*x/7), np.sin(-2*np.pi*x/7))
                       for x in range(7)) / 7
        directory = self.root / "r1"
        periodic = (True, True, True)
        times, coeff = two_stream.read_mode_history(
            directory, SimpleNamespace(nxc=7, nyc=2, nzc=2, mode_x=1, periodic=periodic))
        np.testing.assert_allclose(times, [0, .125, .25])
        np.testing.assert_allclose(coeff, [expected] * 3, atol=1e-12)
        times, profiles = weibel.read_profiles(directory / "B", "Bz", periodic)
        coeff = np.fft.fft(profiles, axis=1)[:, 1] / profiles.shape[1]
        np.testing.assert_allclose(coeff, [expected] * 3, atol=1e-12)
        times, modes = whistler.read_component_modes(directory / "B", ("By", "Bz"), 1, periodic)
        np.testing.assert_allclose(modes["By"], [expected] * 3, atol=1e-12)
        with open_field_reader(directory / "E") as reader:
            vector = tgv.read_vector(reader, "E", 2)
            self.assertEqual(vector[0].shape, (3, 3, 8))
            self.assertEqual(vector[2][-1, -1, -1], 2227.5)
        csv_path = self.root / "diagnostics.csv"
        csv_path.write_text("# precision_profile = double/double\n"
            "cycle,time,electric_energy,magnetic_energy,kinetic_energy,total_energy\n"
            "0,0,1,2,3,6\n1,0.125,2,3,4,9\n")
        diagnostics = tgv.read_diagnostics(csv_path)
        np.testing.assert_array_equal(diagnostics.field_energy, [3, 5])
        np.testing.assert_array_equal(diagnostics.total_energy, [6, 9])

    @unittest.skipIf(np is None, "NumPy analysis extra unavailable")
    def test_whistler_turbulence_node_filter(self):
        # A cell field written as TPIC writes B, the mean of the eight cells around each node: the spectrum
        # divides that mean out and returns the energy of the cell field.
        try:
            import matplotlib
            import scipy
        except ImportError:
            self.skipTest("optional analysis dependencies unavailable")
        sys.path.insert(0, str(TOOLS / "postprocessing"))
        import analyze_whistler_turbulence as turbulence
        n, dx = 8, 0.2
        deck = turbulence.Deck(path=Path("deck.inp"), save_dir="data", lengths=np.full(3, n * dx),
                               cells=np.full(3, n), b0=1.0, epsilon=0.0, electron=0, qom_e=-1.0, v_te=0.1,
                               omega_pe=1.0, d_e=1.0, omega_ce=1.0)
        fourier = turbulence.Fourier(deck)
        # The 150 seeded modes are 75 pairs k, -k, each one stored rfft coefficient.
        self.assertEqual(np.count_nonzero(fourier.seeded), 75)
        k = 2.0 * np.pi / (n * dx)
        x, y, z = np.meshgrid(*[(np.arange(n) + 0.5) * dx] * 3, indexing="ij")
        cell = np.stack([0.3 * np.cos(k * (x + 2.0 * z)), 0.2 * np.sin(k * (y - z)), np.zeros_like(x)])
        node = sum(np.roll(cell, (a, b, c), axis=(1, 2, 3)) for a in (0, 1) for b in (0, 1) for c in (0, 1)) / 8.0
        node[2] += deck.b0
        energy = (fourier.weight * np.sum(np.abs(fourier.modes(node, deck.b0)) ** 2, axis=0)).sum()
        self.assertAlmostEqual(energy, np.mean(np.sum(cell**2, axis=0)), places=12)


if __name__ == "__main__":
    unittest.main()
