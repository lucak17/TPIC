"""NumPy adapter exposing TPIC field files to the scientific analyzers."""

from pathlib import Path
import numpy as np

from .vtk import cell_arrays, image_header, metadata, read_piece, require


def field_files(data_dir, field_name):
    # TPIC names each file <name>_t<cycle>.vti; [0-9] matches the first cycle digit.
    return list(Path(data_dir).glob(f"{field_name}_t[0-9]*.vti"))


class FieldReader:
    """Read a selected scalar/vector field across its <name>_t<cycle>.vti files.

    Arrays have shape (z, y, x) over every global node, or every global cell for
    a cell array (Gauss). Vector names follow the analyzers' existing convention:
    Ex, By, J_0z, etc.
    """

    def __init__(self, base_path):
        self.base = Path(base_path)
        name = self.base.name
        files = field_files(self.base.parent, name)
        if not files:
            raise FileNotFoundError(f"{self.base.parent}: no TPIC field files {name}_t*.vti")
        self.records, layouts = [], set()
        for path in files:
            _, image, box, arrays, _ = image_header(path)
            require(list(arrays) == [name], f"{path}: expected the single array {name}")
            epoch = metadata(image)
            self.records.append((path, epoch))
            cell = name in cell_arrays(image.find("Piece"))
            layouts.add((box, arrays[name], cell, epoch[3], image.get("Origin"), image.get("Spacing")))
        require(len(layouts) == 1, f"{self.base}: snapshot geometry, components or precision changed")
        (self.box, self.components, self.cell, *_), = layouts
        self.records.sort(key=lambda r: r[1][0])
        epochs = [r[1] for r in self.records]
        require(all(a[0] < b[0] and a[1] < b[1] and not a[2] for a, b in zip(epochs, epochs[1:])),
                f"{self.base}: snapshot cycles/times must increase before a final state")
        self._step = None
        self._values = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._step, self._values = None, {}

    def num_steps(self):
        return len(self.records)

    def available_variables(self):
        name = self.base.name
        names = [name + axis for axis in "xyz"] if self.components == 3 else [name]
        return {key: {} for key in (*names, "time")}

    def read(self, name, step_selection):
        require(len(step_selection) == 2 and step_selection[1] == 1, f"{self.base}: read one step at a time")
        step = step_selection[0]
        if not 0 <= step < self.num_steps():
            raise IndexError(f"{self.base}: snapshot step out of range")
        if self._step != step:
            self._load(step)
        if name not in self._values:
            raise KeyError(f"{self.base}: {name} is not one of {list(self._values)}")
        return self._values[name]

    def _load(self, step):
        path, epoch = self.records[step]
        shape = tuple(self.box[i + 1] - self.box[i] + (not self.cell) for i in (4, 2, 0))
        result = np.asarray(read_piece(path).values[self.base.name]).reshape(*shape, self.components)
        self._values = {"time": np.asarray([epoch[1]])}
        if self.components == 3:
            self._values.update({self.base.name + axis: result[..., i] for i, axis in enumerate("xyz")})
        else:
            self._values[self.base.name] = result[..., 0]
        self._step = step


def transverse_mean(field, periodic):
    """Return the mean over z and y of a nodal field of shape (z, y, x), at every x node.

    periodic holds the x, y and z flags of inputs.periodic_axes. A periodic axis
    counts its end node, the image of its first, once; an axis with walls counts
    its two wall nodes as half a cell each.
    """
    for closed in (periodic[2], periodic[1]):
        if closed:
            field = field[:-1].mean(axis=0)
        else:
            weights = np.ones(field.shape[0])
            weights[[0, -1]] = 0.5
            field = np.average(field, axis=0, weights=weights)
    return field


def field_data_exists(data_dir, field_name):
    return bool(field_files(data_dir, field_name))


def open_field_reader(base_path):
    return FieldReader(base_path)
