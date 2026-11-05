"""Selected-array scientific checks shared by integration suites."""
import math
from . import integration_checks as common
from tpic_tools.vtk import read_piece as decode_piece

def component_count(name):
    return 3 if name in ("E", "B", "J") or name.startswith("J_") else 1


def check_snapshots(case, directory):
    names = case["expected_arrays"]
    epochs = (list(range(0, case["cycles"], case["field_cycle"])) + [case["cycles"]]
              if names and case["field_cycle"] > 0 else [])
    expected = [f"{name}_t{cycle:06}.vti" for cycle in epochs for name in names]
    common.require(sorted(path.name for path in directory.glob("*.vti")) == sorted(expected),
                   "wrong snapshot filenames/count/cadence")
    decoded = {}
    total_values = 0
    for cycle in epochs:
        arrays = {}
        for name in names:
            filename = f"{name}_t{cycle:06}.vti"
            piece = decode_piece(directory / filename)
            common.metadata(piece.image, cycle, cycle == case["cycles"], case.get("precision"))
            common.require(piece.extent == (0, 7, 0, 3, 0, case["nz"]), f"{filename}: incorrect global VTK extent")
            # The reader admits Float64 arrays only.
            common.require(piece.components == {name: component_count(name)},
                           f"{filename}: incorrect array or components")
            decoded[filename] = piece.values[name]
            arrays[name] = piece.values[name]
        for base in common.MOMENTS:
            species_names = [f"{base}_{species}" for species in range(case["species"])]
            if base in arrays and all(name in arrays for name in species_names):
                for index, value in enumerate(arrays[base]):
                    common.close(value, sum(arrays[name][index] for name in species_names),
                                 f"cycle {cycle}: species total {base}[{index}]")
                    total_values += 1
    return dict(epochs=epochs, arrays=names, files=len(expected),
                compared_species_total_values=total_values), decoded


def compare(baseline, current, rtol=common.STRICT_RTOL, atol=common.STRICT_ATOL, drift_atol=1e-12):
    _, old_snapshots, old_csv = baseline
    _, new_snapshots, new_csv = current
    common.require(old_snapshots.keys() == new_snapshots.keys(), "builds produced different file names")
    statistics = dict(vtk_values=0, csv_values=0, maximum_absolute=0.0, maximum_relative=0.0)

    def value(actual, expected, context, kind):
        common.require(math.isfinite(actual) and math.isfinite(expected) and
                       math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
                       f"{context}: {actual!r} != {expected!r} (rtol={rtol}, atol={atol})")
        statistics[kind + "_values"] += 1
        absolute = abs(actual - expected)
        scale = max(abs(actual), abs(expected))
        statistics["maximum_absolute"] = max(statistics["maximum_absolute"], absolute)
        statistics["maximum_relative"] = max(statistics["maximum_relative"], absolute / scale if scale else 0.)

    for filename, values in new_snapshots.items():
        common.require(len(values) == len(old_snapshots[filename]), "build payload lengths differ")
        for index, (actual, expected) in enumerate(zip(values, old_snapshots[filename])):
            value(actual, expected, f"{filename}[{index}] changed", "vtk")
    common.require(len(old_csv) == len(new_csv), "build CSV row counts differ")
    for index, (old, new) in enumerate(zip(old_csv, new_csv)):
        common.require(list(old) == list(new), "build CSV columns differ")
        for name in old:
            if name == "state":
                common.require(old[name] == new[name], "build CSV epoch labels differ")
            else:
                value(float(new[name]), float(old[name]), f"CSV row {index} {name} changed", "csv")
    old_initial, new_initial = float(old_csv[0]["total_energy"]), float(new_csv[0]["total_energy"])
    common.require(old_initial > 0 and new_initial > 0, "energy drift fixture has nonpositive initial energy")
    drift_error = max(abs(float(new["total_energy_change"]) / new_initial -
                          float(old["total_energy_change"]) / old_initial)
                      for old, new in zip(old_csv, new_csv))
    common.require(drift_error <= drift_atol,
                   f"normalized energy drift changed by {drift_error:g}, limit {drift_atol:g}")
    statistics["maximum_normalized_energy_drift_error"] = drift_error
    statistics["normalized_energy_drift_tolerance"] = drift_atol
    return statistics

