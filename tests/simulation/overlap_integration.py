#!/usr/bin/env python3
"""Run bounded CUDA/HIP+MPI overlap/output smoke checks with no third-party Python packages.

Example (use a fresh directory; requires the executable's real GPU transport):
  python3 tests/simulation/overlap_integration.py --executable build/tpic \
      --work-dir /tmp/tpic-overlap-live --ranks 1 2

The default matrix runs every storage level with double compute in this one executable.
This checks integrated execution and output invariants, not numerical
equivalence or performance.
Every input, command, log and result is retained in the work dir.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import time

import _bootstrap
from support.integration_checks import (MOMENTS, PRECISIONS,
    require, precision_arguments, make_input, check_csv, check_counts)
from support.snapshot_checks import check_snapshots
from tpic_tools.runs import run_command


def cases(ranks, precisions=PRECISIONS):
    for profile in dict.fromkeys(precisions):
        for case in physical_cases(ranks):
            yield dict(case, name=f"{profile.replace('/', '-')}-{case['name']}",
                       precision=profile, requested_precision=profile, field_cycle=1,
                       expected_arrays=["E", "B"] + [f"{name}_{species}" for species in range(case["species"])
                                                     for name in MOMENTS] + list(MOMENTS))


def physical_cases(ranks):
    """Eight runs per rank count plus the one-/ten-species one-rank endpoints."""
    for rank_count in ranks:
        configurations = [(order, "periodic", False, 3, 2)
                          for order in ("full", "partial", "unsorted")]
        configurations += [("full", "periodic", True, 3, 2),
                           ("partial", "pec", True, 3, 2),
                           ("unsorted", "pmc", True, 3, 2),
                           ("full", "periodic", False, 0, 2),
                           ("full", "periodic", False, 3, 1)]
        for order, boundary, smoothing, cycles, nz in configurations:
            name = (f"r{rank_count}-{order}-{boundary}-smooth{int(smoothing)}"
                    f"-steps{cycles}-nz{nz}")
            yield dict(name=name, ranks=rank_count, order=order, boundary=boundary,
                       smoothing=smoothing, cycles=cycles, nz=nz, species=3)
        if rank_count == 1:
            for species in (1, 10):
                yield dict(name=f"r1-full-periodic-species{species}", ranks=1, order="full",
                           boundary="periodic", smoothing=False, cycles=3, nz=2, species=species)


def run_case(case, args):
    directory = args.work_dir / case["name"]
    directory.mkdir()
    input_path = directory / "input.inp"
    input_path.write_text(make_input(case, directory))
    command = shlex.split(args.launcher) + ["-n", str(case["ranks"]), str(args.executable),
               "--run", str(input_path), "--order", case["order"], "--mass-matrix", args.mass_matrix,
               *precision_arguments(case["requested_precision"])]
    result = dict(case, command=command, status="failed")
    started = time.monotonic()
    try:
        process, _, _ = run_command(command, directory, args.timeout)
        result["returncode"] = process["returncode"]
        if process["timed_out"]:
            raise RuntimeError(f"job exceeded {args.timeout:g} seconds")
        require(result["returncode"] == 0, "simulation failed; inspect stdout.log/stderr.log")
        result["counts"] = check_counts(case, (directory / "stdout.log").read_text())
        result["csv"] = check_csv(case, directory / "output")
        result["vtk"] = check_snapshots(case, directory / "output")[0]
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = time.monotonic() - started
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--launcher", default="mpiexec", help="launcher command, optionally with arguments")
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--timeout", type=float, default=90, help="seconds per MPI job")
    parser.add_argument("--precision", nargs="+", choices=PRECISIONS, default=PRECISIONS,
                        help="precision pairs to test in the same executable (default: every storage level with double compute)")
    parser.add_argument("--mass-matrix", choices=("scalar", "matrix-core"),
                        default="scalar")
    args = parser.parse_args()
    args.executable = args.executable.resolve(strict=True)
    args.work_dir = args.work_dir.resolve()
    if args.work_dir.exists():
        parser.error("--work-dir must be fresh/nonexistent")
    if not (math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("--timeout must be positive and finite")
    args.work_dir.mkdir(parents=True)
    executable_digest = hashlib.sha256()
    with args.executable.open("rb") as executable:
        for block in iter(lambda: executable.read(1024 * 1024), b""):
            executable_digest.update(block)
    summary = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                   executable=str(args.executable),
                   executable_sha256=executable_digest.hexdigest(),
                   launcher=args.launcher, mass_matrix=args.mass_matrix, precisions=args.precision,
                   profiled=os.environ.get("TPIC_PROFILE_STAGES", "0"),
                   qualification="Integrated smoke/output invariants; no numerical-equivalence or speedup claim.",
                   empty_species="Not included: the Uniform input planner requires positive particles per cell.",
                   runs=[])
    for case in cases(dict.fromkeys(args.ranks), args.precision):
        if args.mass_matrix != "scalar" and case["order"] == "unsorted":
            result = dict(case, status="skipped",
                          reason="matrix-core mass deposition requires full or partial ordering")
        else:
            result = run_case(case, args)
        summary["runs"].append(result)
        summary["passed"] = sum(run["status"] == "passed" for run in summary["runs"])
        summary["failed"] = sum(run["status"] == "failed" for run in summary["runs"])
        summary["skipped"] = sum(run["status"] == "skipped" for run in summary["runs"])
        (args.work_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{case['name']}: {result['status']}" + (": " + result["error"] if "error" in result else ""),
              flush=True)
    print(f"{summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped; "
          f"{args.work_dir / 'results.json'}")
    return int(summary["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
