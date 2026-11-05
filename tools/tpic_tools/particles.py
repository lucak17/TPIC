"""NumPy reader of TPIC's raw-appended PolyData particle snapshot pieces."""

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

TYPES = {"Float64": "f8", "Float32": "f4", "UInt64": "u8", "Int64": "i8", "Int32": "i4"}


def polydata_layout(path: Path) -> dict[str, tuple[np.dtype, int, int, int]]:
    """Per appended array of a raw-appended PolyData piece written by TPIC: dtype, components, byte offset of the
    first value and number of values."""
    with path.open("rb") as stream:
        head = stream.read(1 << 20)
    marker = re.search(rb'<AppendedData\b[^>]*encoding="raw"[^>]*>\s*_', head)
    if marker is None:
        raise ValueError(f"{path}: no raw appended data")
    root = ET.fromstring(head[:marker.start()] + b"</VTKFile>")
    endian = "<" if root.get("byte_order") == "LittleEndian" else ">"
    layout = {}
    for descriptor in root.iter("DataArray"):
        if descriptor.get("offset") is None:
            continue
        start = marker.end() + int(descriptor.get("offset"))
        size = int(np.fromfile(path, dtype=endian + "u8", count=1, offset=start)[0])
        dtype = np.dtype(endian + TYPES[descriptor.get("type")])
        layout[descriptor.get("Name")] = (dtype, int(descriptor.get("NumberOfComponents", "1")), start + 8,
                                          size // dtype.itemsize)
    return layout


def read_polydata(path: Path, names) -> dict[str, np.ndarray]:
    """The named arrays of a raw-appended PolyData piece written by TPIC."""
    return {name: np.fromfile(path, dtype=dtype, count=count, offset=offset)
            for name, (dtype, _, offset, count) in polydata_layout(path).items() if name in names}


def iter_species(path: Path, names, species: int, chunk: int = 1 << 24):
    """The named arrays, (n, components), of the particles of one species in a piece, chunk particles at a time.

    For pieces too large to hold: a chunk without that species is read no further than its species values (TPIC
    writes each piece species by species)."""
    layout = polydata_layout(path)
    dtype, _, offset, count = layout["species"]
    for begin in range(0, count, chunk):
        n = min(chunk, count - begin)
        keep = np.fromfile(path, dtype=dtype, count=n, offset=offset + dtype.itemsize * begin) == species
        if not keep.any():
            continue
        arrays = {}
        for name in names:
            values_dtype, components, values_offset, _ = layout[name]
            values = np.fromfile(path, dtype=values_dtype, count=components * n,
                                 offset=values_offset + values_dtype.itemsize * components * begin)
            values = values.reshape(n, components)
            arrays[name] = values if keep.all() else values[keep]
        yield arrays
