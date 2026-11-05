#!/usr/bin/env python3
"""Compare runtime precision output against its own build-selected baseline.

Use --precision <storage>/<compute> with the corresponding --baseline executable; only the
current executable receives runtime precision selection. Requires real
CUDA/HIP+MPI executables and no third-party Python packages.
Eight bounded cases independently verify selected descriptors, output epochs,
global extents, scalar metadata and numerical values before comparing builds.
This is an output/lifetime regression, not a performance measurement.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import time

import _bootstrap
from support import integration_checks as common
from support.snapshot_checks import check_snapshots, compare
from tpic_tools.runs import digest, read_csv, run_command


def cases():
    all_moments = "+".join(token for base in common.MOMENTS for token in (base, base + "_tot"))
    all_arrays = ["E", "B"] + [f"{base}_{species}" for species in range(3)
                               for base in common.MOMENTS] + list(common.MOMENTS)
    selections = [
        ("all-selected", 2, "full", False, 3, 1, "E+B", all_moments, all_arrays),
        ("totals-only-smoothed", 2, "partial", True, 3, 1, "",
         "+".join(base + "_tot" for base in common.MOMENTS), list(common.MOMENTS)),
        ("mixed-selection-smoothed", 2, "full", True, 3, 1, "E",
         "rho0+rho2+rho_tot+J1+J_tot+PXX0+PXX_tot+PXY2+PZZ1+PZZ_tot",
         ["E", "rho_0", "PXX_0", "J_1", "PZZ_1", "rho_2", "PXY_2", "rho", "J", "PXX", "PZZ"]),
        ("electric-only-cadence2", 1, "unsorted", False, 3, 2, "E", "", ["E"]),
        ("magnetic-only", 2, "partial", False, 2, 1, "B", "", ["B"]),
        ("output-disabled", 1, "full", False, 3, 0, "E+B", all_moments, []),
        ("zero-cycle-totals-final", 2, "full", False, 0, 2, "",
         "rho_tot+J_tot+PZZ_tot", ["rho", "J", "PZZ"]),
        ("mixed-cadence2-smoothed", 1, "full", True, 3, 2, "E+B",
         "rho0+rho_tot+J2+PYY1+PYY_tot", ["E", "B", "rho_0", "PYY_1", "J_2", "rho", "PYY"]),
    ]
    for name, ranks, order, smoothing, cycles, cadence, fields, moments, arrays in selections:
        yield dict(name=name, ranks=ranks, order=order, smoothing=smoothing,
                   cycles=cycles, field_cycle=cadence, fields=fields, moments=moments,
                   expected_arrays=arrays, boundary="periodic", nz=2, species=3)


def run_build(case, executable, directory, args, runtime=False):
    directory.mkdir(parents=True)
    text = common.make_input(case, directory)
    for key, value in (("FieldOutputCycle", case["field_cycle"]),
                       ("FieldOutputTag", case["fields"]), ("MomentsOutputTag", case["moments"])):
        text, changed = re.subn(rf"^{key} = .*?$", f"{key} = {value}", text, flags=re.MULTILINE)
        common.require(changed == 1, f"generated input lacks unique {key}")
    input_path = directory / "input.inp"
    input_path.write_text(text)
    command = shlex.split(args.launcher) + ["-n", str(case["ranks"]), str(executable),
               "--run", str(input_path), "--order", case["order"], "--mass-matrix", "scalar"]
    if runtime:
        command += common.precision_arguments(args.precision)
    environment = dict(os.environ, TPIC_PROFILE_STAGES="0")
    started = time.monotonic()
    process, _, _ = run_command(command, directory, args.timeout, environment)
    if process["timed_out"]:
        raise RuntimeError(f"job exceeded {args.timeout:g} seconds")
    returncode = process["returncode"]
    common.require(returncode == 0, f"simulation failed ({returncode}); inspect {directory}")
    output = directory / "output"
    snapshot_result, snapshots = check_snapshots(case, output)
    report = dict(command=command, input_sha256=digest(input_path), returncode=returncode,
                  elapsed_seconds=time.monotonic() - started,
                  counts=common.check_counts(case, (directory / "stdout.log").read_text()),
                  csv=common.check_csv(case, output), vtk=snapshot_result)
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, snapshots, read_csv(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="build-selected baseline for the selected --precision profile")
    parser.add_argument("--precision", choices=common.PRECISIONS, default="double/double",
                        help="runtime profile; baseline metadata must report the same profile")
    parser.add_argument("--comparison-rtol", type=float,
                        help="baseline equivalence tolerance (default: documented 3e-5 with single field storage, 3e-10 otherwise)")
    parser.add_argument("--comparison-atol", type=float,
                        help="baseline equivalence tolerance (default: documented 3e-7 with single field storage, 3e-12 otherwise)")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, help="also retain the summary at this path")
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--timeout", type=float, default=90.)
    args = parser.parse_args()
    args.baseline, args.executable = args.baseline.resolve(strict=True), args.executable.resolve(strict=True)
    mixed = common.single_field_storage(args.precision)
    rtol = args.comparison_rtol if args.comparison_rtol is not None else 3e-5 if mixed else common.STRICT_RTOL
    atol = args.comparison_atol if args.comparison_atol is not None else 3e-7 if mixed else common.STRICT_ATOL
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        parser.error("comparison tolerances must be finite and nonnegative")
    args.work_dir = args.work_dir.resolve()
    if args.work_dir.exists():
        parser.error("--work-dir must be fresh/nonexistent")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    args.work_dir.mkdir(parents=True)
    summary = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                   baseline=dict(path=str(args.baseline), sha256=digest(args.baseline)),
                   executable=dict(path=str(args.executable), sha256=digest(args.executable)),
                   precision=args.precision,
                   script_sha256=digest(Path(__file__)), launcher=args.launcher,
                   tolerance=dict(rtol=rtol, atol=atol),
                   tolerance_policy="docs/developer/testing.md",
                   qualification="Own-profile runtime/build-selected output and lifetime regression; no performance claim.",
                   work_dir=str(args.work_dir), runs=[])
    for case in cases():
        if case["ranks"] not in args.ranks:
            continue
        case = dict(case, precision=args.precision)
        result = dict(case, status="failed")
        try:
            baseline = run_build(case, args.baseline, args.work_dir / case["name"] / "baseline", args)
            result["baseline"] = baseline[0]
            current = run_build(case, args.executable, args.work_dir / case["name"] / "current", args, runtime=True)
            result["current"] = current[0]
            result["comparison"] = compare(baseline, current, rtol, atol, 1e-6 if mixed else 1e-12)
            result["status"] = "passed"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        summary["runs"].append(result)
        summary["passed"] = sum(run["status"] == "passed" for run in summary["runs"])
        summary["failed"] = sum(run["status"] == "failed" for run in summary["runs"])
        serialized = json.dumps(summary, indent=2) + "\n"
        (args.work_dir / "results.json").write_text(serialized)
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(serialized)
        print(f"{case['name']}: {result['status']}" +
              (": " + result["error"] if "error" in result else ""), flush=True)
    print(f"{summary['passed']} passed, {summary['failed']} failed; {args.work_dir / 'results.json'}")
    return int(summary["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
