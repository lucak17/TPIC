"""Verify ADIOS2 vector fields through the ParaView reader used by the GUI."""
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def paraview_modules():
    try:
        simple = importlib.import_module("paraview.simple")
    except ModuleNotFoundError as error:
        if error.name not in ("paraview", "paraview.simple"):
            raise
        return None
    if not hasattr(simple, "FidesReader"):
        return None
    return simple, importlib.import_module("paraview.servermanager")


def check_piece(simple, servermanager, path, cycle, rank, names):
    from vtkmodules.vtkCommonCore import VTK_STRING, vtkOutputWindow
    from vtkmodules.util.misc import calldata_type

    errors = []
    warnings = []

    @calldata_type(VTK_STRING)
    def on_error(_object, _event, message):
        errors.append(message)

    @calldata_type(VTK_STRING)
    def on_warning(_object, _event, message):
        warnings.append(message)

    # The simple constructor already requests metadata. Observe VTK's output
    # window before constructing it, then observe the reader and executive too.
    window = vtkOutputWindow.GetInstance()
    observer = window.AddObserver("ErrorEvent", on_error)
    warning_observer = window.AddObserver("WarningEvent", on_warning)
    reader = None
    view = None
    try:
        reader = simple.FidesReader(FileName=str(path))
        require(not errors, f"{path}: reader initialization failed: {errors}")
        client = reader.GetClientSideObject()
        client.AddObserver("ErrorEvent", on_error)
        client.GetExecutive().AddObserver("ErrorEvent", on_error)
        client.AddObserver("WarningEvent", on_warning)
        client.GetExecutive().AddObserver("WarningEvent", on_warning)
        reader.UpdatePipelineInformation()
        require(not errors, f"{path}: information request failed: {errors}")
        require(tuple(reader.TimestepValues) == (cycle * 0.125,),
                f"{path}: incorrect or missing physical time")
        reader.UpdatePipeline(time=cycle * 0.125)
        require(not errors, f"{path}: data request failed: {errors}")
        output = servermanager.Fetch(reader)
        require(not errors, f"{path}: data fetch failed: {errors}")
        check_image(output, path, cycle, rank, names, on_warning)
        require(not warnings, f"{path}: reader or array access generated warnings: {warnings}")

        if "E" in names:
            information = reader.GetDataInformation().GetPointDataInformation()
            electric = information.GetArrayInformation("E")
            require(electric is not None and electric.GetNumberOfComponents() == 3,
                    f"{path}: ParaView does not recognize E as a vector")
            points = (4 if rank == 0 else 3) * 3 * 3
            magnitudes = [math.sqrt(sum(field_value(point * 3 + component, cycle, rank) ** 2
                                       for component in range(3))) for point in range(points)]
            require(all(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                        for actual, expected in zip(electric.GetComponentRange(-1),
                                                    (min(magnitudes), max(magnitudes)))),
                    f"{path}: incorrect automatic vector magnitude range")

            # Exercise the GUI's vector coloring setup without rendering: the
            # regression must run on hosts without a graphics device or display.
            view = simple.CreateView("RenderView")
            display = simple.Show(reader, view)
            simple.ColorBy(display, ("POINTS", "E", "Magnitude"))
            require(simple.GetColorTransferFunction("E").VectorMode == "Magnitude",
                    f"{path}: ParaView cannot color E by magnitude")
        require(not errors, f"{path}: coloring or array access failed: {errors}")
        # Creating a view can report an unavailable X display before selecting
        # the offscreen backend. Still reject any vector-conversion warning.
        conversion_warnings = [message for message in warnings if "GetVoidPointer" in message]
        require(not conversion_warnings,
                f"{path}: ParaView copied vector components: {conversion_warnings}")
    finally:
        if view is not None:
            simple.Delete(view)
        if reader is not None:
            simple.Delete(reader)
        window.RemoveObserver(observer)
        window.RemoveObserver(warning_observer)


def field_value(index, cycle, rank):
    return index * 0.25 - 1 + cycle + rank * 100


def check_image(output, path, cycle, rank, names, on_warning):
    require(output is not None and output.IsA("vtkPartitionedDataSetCollection"),
            f"{path}: reader did not produce a partitioned dataset collection")
    iterator = output.NewIterator()
    iterator.InitTraversal()
    pieces = []
    while not iterator.IsDoneWithTraversal():
        piece = iterator.GetCurrentDataObject()
        if piece is not None:
            require(piece.IsA("vtkImageData"), f"{path}: unexpected piece type")
            pieces.append(piece)
        iterator.GoToNextItem()
    require(len(pieces) == 1, f"{path}: expected exactly one image piece")
    image = pieces[0]
    cells = (3 if rank == 0 else 2, 2, 2)
    origin = (0.0 if rank == 0 else 6.0, 0.0, 0.0)
    spacing = (2.0, 2.0, 3.0)
    dimensions = tuple(count + 1 for count in cells)
    points = dimensions[0] * dimensions[1] * dimensions[2]
    require(image.GetDimensions() == dimensions, f"{path}: incorrect dimensions")
    require(image.GetExtent() == (0, cells[0], 0, cells[1], 0, cells[2]),
            f"{path}: incorrect local extent")
    require(image.GetOrigin() == origin, f"{path}: incorrect physical origin")
    require(image.GetSpacing() == spacing, f"{path}: incorrect physical spacing")
    require(image.GetNumberOfPoints() == points, f"{path}: incorrect point count")
    require(image.GetNumberOfCells() == cells[0] * cells[1] * cells[2],
            f"{path}: incorrect cell count")
    bounds = tuple(value for axis in range(3)
                   for value in (origin[axis], origin[axis] + cells[axis] * spacing[axis]))
    require(image.GetBounds() == bounds, f"{path}: incorrect physical bounds")

    data = image.GetPointData()
    require({data.GetArrayName(i) for i in range(data.GetNumberOfArrays())} == set(names),
            f"{path}: missing or unexpected field arrays")
    for name in names:
        array = data.GetArray(name)
        components = 3 if name == "E" else 1
        require(array.GetNumberOfComponents() == components and array.GetNumberOfTuples() == points,
                f"{path}: incorrect array shape for {name}")
        require(array.HasStandardMemoryLayout(),
                f"{path}: {name} requires a copy to expose contiguous VTK tuples")
        # Rendering and filters request this pointer. Separate component buffers
        # would require an interleaving copy and emit the expensive-access warning.
        warning_observer = array.AddObserver("WarningEvent", on_warning)
        try:
            require(array.GetVoidPointer(0) is not None,
                    f"{path}: {name} cannot expose its contiguous values")
        finally:
            array.RemoveObserver(warning_observer)
        for point in range(points):
            expected = tuple(field_value(point * components + component, cycle, rank)
                             for component in range(components))
            require(array.GetTuple(point) == expected,
                    f"{path}: incorrect {name} value at point {point}")
    point = 0
    for z in range(dimensions[2]):
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                coordinate = tuple(origin[axis] + index * spacing[axis]
                                   for axis, index in enumerate((x, y, z)))
                require(image.GetPoint(point) == coordinate,
                        f"{path}: incorrect coordinate at point {point}")
                point += 1


def main():
    require(len(sys.argv) == 2, "usage: test_adios_paraview_reader.py dataset_engine_tests")
    os.environ.pop("VTK_SILENCE_GET_VOID_POINTER_WARNINGS", None)
    modules = paraview_modules()
    if modules is None:
        print("SKIP: this ParaView installation has no FidesReader")
        return 77
    simple, servermanager = modules
    writer = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="tpic-adios-paraview-") as temporary:
        fixtures = Path(temporary) / "fixtures"
        subprocess.run([str(writer), str(fixtures)], check=True, timeout=120)
        for cycle in (0, 2, 4):
            for rank in (0, 1):
                path = fixtures / "adios2-adios2" / f"fields_t{cycle:06d}_r{rank:06d}.bp"
                require(path.is_dir(), f"{path}: fixture is missing")
                check_piece(simple, servermanager, path, cycle, rank, ("E", "rho"))
        for selection, names in (("vector", ("E",)), ("scalar", ("rho",))):
            for rank in (0, 1):
                path = fixtures / f"adios2-{selection}" / f"fields_t000000_r{rank:06d}.bp"
                require(path.is_dir(), f"{path}: fixture is missing")
                check_piece(simple, servermanager, path, 0, rank, names)
    print("PASS: ParaView read ten mixed, vector-only and scalar-only ADIOS2 field pieces "
          "with correct geometry, time, values, magnitude coloring and contiguous tuples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
