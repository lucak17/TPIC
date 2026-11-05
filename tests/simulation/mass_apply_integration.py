#!/usr/bin/env python3
"""Compare compiled full/split Maxwell schedules through the runtime flag.

The default suite has six pairs per rank count (12 pairs for --ranks 1 2),
plus two bounded two-rank startup rejection jobs. Existing integration helpers
decode E/B/J output and diagnostics. Thermal particle loading produces spatially
varying current; the suite checks field/current evolution and changing values
next to communicating interfaces. No third-party Python packages are required.

An optional --baseline adds a baseline-executable comparison for the unsmoothed
double/scalar/periodic pair at each rank count. This is numerical/lifetime
qualification, not a performance measurement.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys

import _bootstrap
from support import integration_checks as common, snapshot_checks as snapshots
from tpic_tools.runs import digest, read_csv, run_command as launch_command


GMRES_TOLERANCE = 1e-10
RESIDUAL_RTOL = 5e-3
RESIDUAL_ATOL = 1e-12


class MatrixCoreUnavailable(RuntimeError):
    pass


def cases(ranks):
    # Seven x cells give unequal two-rank partitions. The PMC fixture uses
    # the minimum admitted two z cells; low-level device tests cover empty cores.
    configurations = (
        ("periodic-full", "double/double", "scalar", "full", "periodic", None, 3),
        ("pec-partial", "single-position/double", "scalar", "partial", "pec", None, 2),
        ("pmc-thin-unsorted", "single-particles-moments-fields/double", "scalar", "unsorted", "pmc", None, 2),
        ("periodic-face7", "double/double", "scalar", "partial", "periodic", "face7", 2),
        ("pec-binomial27", "single-particles-moments-fields/double", "scalar", "full", "pec", "binomial27", 3),
        ("periodic-matrix-core", "single-position/double", "matrix-core", "partial", "periodic", None, 2),
    )
    for rank_count in ranks:
        for name, precision, mass, order, boundary, smoothing, nz in configurations:
            yield dict(name=f"r{rank_count}-{name}", ranks=rank_count, precision=precision,
                       mass_matrix=mass, order=order, boundary=boundary,
                       smoothing=smoothing is not None, smoothing_operator=smoothing,
                       cycles=3, nz=nz, species=3, field_cycle=1,
                       expected_arrays=["E", "B", "J"],
                       baseline_case=name == "periodic-full")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), Path(common.__file__).resolve(), Path(snapshots.__file__).resolve()]
    paths += [root / name for name in (
        "include/tpic/config/numerics_options.hpp",
        "include/tpic/ecsim/maxwell/maxwell_apply_policy.hpp", "include/tpic/ecsim/fields/stencil_region.hpp",
        "include/tpic/ecsim/mass_matrix/mass_matrix_apply_device.hpp", "include/tpic/ecsim/maxwell/maxwell_device.hpp",
        "include/tpic/ecsim/fields/field_vector_device.hpp", "include/tpic/ecsim/fields/node_cell_curl_device.hpp",
        "include/tpic/communication/fields/field_exchange.hpp",
        "include/tpic/simulation/actions/field_solve_actions_device.hpp",
        "include/tpic/simulation/driver/pipeline.hpp",
        "include/tpic/simulation/driver/simulation_device.hpp",
        "src/app/main.cpp", "src/simulation/run_distributed.cpp",
    )]
    paths += list((root / "tools/tpic_tools").glob("*.py"))
    return {str(path.relative_to(root)): digest(path) for path in paths if path.is_file()}


def make_input(case, directory):
    text = common.make_input(case, directory)
    text, changed = re.subn(r"^MomentsOutputTag = .*?$", "MomentsOutputTag = J_tot",
                           text, flags=re.MULTILINE)
    common.require(changed == 1, "generated input lacks unique moment selection")
    text += f"ECSimSmoothingOperator = {case['smoothing_operator'] or 'face7'}\n"
    return text


def run_command(command, directory, timeout):
    # Detailed stage instrumentation must not change the qualification schedule.
    result = launch_command(command, directory, timeout, dict(os.environ, TPIC_PROFILE_STAGES="0"))
    write_json(directory / "process.json", result[0])
    return result


def read_gmres(log, cycles):
    matches = re.findall(r"Maxwell GMRES\s*:\s*converged in (\d+) iterations;\s*residual = (\S+)", log)
    common.require(len(matches) == cycles, "missing/duplicate GMRES iteration and residual reports")
    results = [dict(cycle=cycle, iterations=int(iterations), residual=float(residual))
               for cycle, (iterations, residual) in enumerate(matches)]
    for result in results:
        common.require(result["iterations"] >= 0 and math.isfinite(result["residual"]) and
                       0 <= result["residual"] <= GMRES_TOLERANCE * (1 + 1e-6),
                       f"GMRES report violates convergence tolerance: {result}")
    return results


def evolution(case, decoded):
    def array(name, cycle):
        return decoded[f"{name}_t{cycle:06}.vti"]

    maximum_change = {name: max(abs(a - b) for a, b in zip(array(name, 0), array(name, case["cycles"])))
                      for name in case["expected_arrays"]}
    maximum_spread = {name: max(max(array(name, case["cycles"])[c::3]) - min(array(name, case["cycles"])[c::3])
                                for c in range(3)) for name in case["expected_arrays"]}
    interface_field_change = 0.0
    if case["ranks"] == 2:
        # These owned planes supply the opposite rank's x ghosts: one node inside
        # rank 0's upper face and rank 1's lower face. LPFx gives the first rank
        # the remainder cell of the seven.
        nx, ny, nz, offset = 8, 4, case["nz"] + 1, 4
        for plane in (offset - 1, offset + 1):
            for name in ("E", "B"):
                initial, final = array(name, 0), array(name, case["cycles"])
                for k in range(nz):
                    for j in range(ny):
                        first = 3 * (plane + nx * (j + ny * k))
                        interface_field_change = max(interface_field_change,
                            max(abs(initial[first + c] - final[first + c]) for c in range(3)))
    common.require(max(maximum_change["E"], maximum_change["B"]) > 0 and maximum_change["J"] > 0,
                   "fixture fields/current did not evolve across output epochs")
    common.require(max(maximum_spread["E"], maximum_spread["B"]) > 0 and maximum_spread["J"] > 0,
                   "fixture lacks spatially varying fields/current")
    if case["ranks"] == 2:
        common.require(interface_field_change > 0, "fixture did not change remote-halo source planes")
    return dict(maximum_absolute_change=maximum_change, maximum_component_spread=maximum_spread,
                remote_source_field_change=interface_field_change if case["ranks"] == 2 else None)


def run_policy(case, mode, directory, args, executable=None, allow_unavailable=False):
    directory.mkdir(parents=True)
    input_path = directory / "input.inp"
    input_path.write_text(make_input(case, directory))
    executable = executable or args.executable
    command = shlex.split(args.launcher) + ["-n", str(case["ranks"]), str(executable),
               "--run", str(input_path), *common.precision_arguments(case["precision"]),
               "--order", case["order"], "--mass-matrix", case["mass_matrix"]]
    if mode is not None:
        command += ["--mass-apply", mode]
    report, stdout, stderr = run_command(command, directory, args.job_timeout)
    report.update(input_sha256=digest(input_path), executable_sha256=digest(executable),
                  requested_mass_apply=mode, effective_mass_apply=
                      ("smoothed full-domain" if case["smoothing"] else mode) if mode else "default mass apply")
    try:
        common.require(not report["timed_out"], f"job timed out; inspect {directory}")
        if report["returncode"] != 0 and allow_unavailable:
            unavailable = (
                "Matrix-core mass deposition requires a TPIC CUDA or HIP device build",
                "Matrix-core mass deposition requires a supported FP64 matrix device and compiled kernel image",
            )
            reason = next((message for message in unavailable if message in stdout + stderr), None)
            if reason is not None:
                raise MatrixCoreUnavailable(reason)
        common.require(report["returncode"] == 0,
                       f"simulation failed ({report['returncode']}); inspect {directory}")
        if mode is not None:
            expected = rf"Mass apply\s*:\s*{mode}; effective: {re.escape(report['effective_mass_apply'])}(?:\n|\r|$)"
            common.require(re.search(expected, stdout) is not None, "missing/incorrect requested/effective mass report")
        report["counts"] = common.check_counts(case, stdout)
        report["gmres"] = read_gmres(stdout, case["cycles"])
        report["csv"] = common.check_csv(case, directory / "output")
        report["vtk"], decoded = snapshots.check_snapshots(case, directory / "output")
        report["evolution"] = evolution(case, decoded)
        report["status"] = "passed"
        return report, decoded, read_csv(directory / "output")
    except MatrixCoreUnavailable as error:
        report.update(status="skipped", reason=str(error))
        raise
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(directory / "result.json", report)


def compare_gmres(full, split):
    common.require(len(full) == len(split), "full/split GMRES report counts differ")
    comparisons = []
    for before, after in zip(full, split):
        common.require(before["cycle"] == after["cycle"] and before["iterations"] == after["iterations"],
                       f"full/split GMRES iteration count changed: {before} -> {after}")
        common.require(math.isclose(before["residual"], after["residual"],
                                    rel_tol=RESIDUAL_RTOL, abs_tol=RESIDUAL_ATOL),
                       f"full/split reported residual changed: {before} -> {after}")
        comparisons.append(dict(cycle=before["cycle"], iterations=before["iterations"],
                                full_residual=before["residual"], split_residual=after["residual"],
                                absolute_error=abs(before["residual"] - after["residual"])))
    return comparisons


def run_pair(case, args):
    result = dict(case, status="failed")
    directory = args.work_dir / case["name"]
    mixed = common.single_field_storage(case["precision"])
    rtol, atol, drift = (3e-5, 3e-7, 1e-6) if mixed else (common.STRICT_RTOL, common.STRICT_ATOL, 1e-12)
    result["tolerance"] = dict(rtol=rtol, atol=atol, normalized_energy_drift=drift)
    try:
        full = run_policy(case, "full", directory / "full", args,
                          allow_unavailable=case["mass_matrix"] == "matrix-core")
        result["full"] = full[0]
        split = run_policy(case, "split", directory / "split", args)
        result["split"] = split[0]
        result["comparison"] = snapshots.compare(full, split, rtol, atol, drift)
        result["gmres_comparison"] = compare_gmres(full[0]["gmres"], split[0]["gmres"])
        if args.baseline is not None and case["baseline_case"]:
            baseline = run_policy(case, None, directory / "baseline", args, executable=args.baseline)
            result["baseline"] = baseline[0]
            result["baseline_comparison"] = snapshots.compare(baseline, full, rtol, atol, drift)
            result["baseline_gmres_comparison"] = compare_gmres(baseline[0]["gmres"], full[0]["gmres"])
        result["status"] = "passed"
    except MatrixCoreUnavailable as error:
        result.update(status="skipped", reason=str(error))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(directory / "result.json", result)
    return result


def mismatched_rank():
    """Internal MPI-launched wrapper; use launcher-provided rank, never a guess."""
    common.require(len(sys.argv) == 4, "invalid mismatch wrapper arguments")
    rank = next((os.environ[key] for key in
                 ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID") if key in os.environ), None)
    common.require(rank in ("0", "1"), "MPI launcher did not provide a supported two-rank environment")
    executable, input_path = sys.argv[2:]
    command = [executable, "--initialize", input_path, "--order", "full", "--storage-precision", "double", "--compute-precision", "double",
               "--mass-matrix", "scalar", "--mass-apply", "full" if rank == "0" else "split"]
    os.execv(executable, command)


def mismatch_case(args, smoothing):
    name = "r2-mismatched-mass-apply" + ("-smoothed" if smoothing else "-unsmoothed")
    directory = args.work_dir / name
    directory.mkdir()
    case = next(cases([2]))
    case.update(cycles=0, smoothing=smoothing, smoothing_operator="face7" if smoothing else None)
    input_path = directory / "input.inp"
    input_path.write_text(make_input(case, directory))
    command = shlex.split(args.launcher) + ["-n", "2", sys.executable, str(Path(__file__).resolve()),
               "--_mismatched-rank", str(args.executable), str(input_path)]
    result = dict(name=name, smoothing=smoothing, status="failed", requested_mass_apply_by_rank=["full", "split"],
                  effective_mass_apply_by_rank=["smoothed full-domain"] * 2 if smoothing else ["full", "split"],
                  input_sha256=digest(input_path),
                  executable_sha256=digest(args.executable),
                  rank_commands=[[str(args.executable), "--initialize", str(input_path), "--order", "full",
                                  "--storage-precision", "double", "--compute-precision", "double",
                                  "--mass-matrix", "scalar", "--mass-apply", mode]
                                 for mode in ("full", "split")])
    try:
        report, stdout, stderr = run_command(command, directory, args.job_timeout)
        result["process"] = report
        common.require(not report["timed_out"], "mismatched flags hung instead of rejecting collectively")
        common.require(report["returncode"] != 0 and
                       "TPIC startup: --mass-apply differs across ranks" in stdout + stderr,
                       "mismatched flags did not report the expected collective rejection")
        common.require("Maxwell GMRES" not in stdout + stderr,
                       "mismatched flags reached numerical execution")
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--job-timeout", "--timeout", dest="job_timeout", type=float, default=90.)
    parser.add_argument("--baseline", type=Path,
                        help="optional baseline executable for the double/scalar/periodic subset")
    args = parser.parse_args()
    args.executable = args.executable.resolve(strict=True)
    if args.baseline is not None:
        args.baseline = args.baseline.resolve(strict=True)
    args.work_dir = args.work_dir.resolve()
    if args.work_dir.exists():
        parser.error("--work-dir must be fresh/nonexistent")
    if not math.isfinite(args.job_timeout) or args.job_timeout <= 0:
        parser.error("--job-timeout must be finite and positive")
    if not shlex.split(args.launcher):
        parser.error("--launcher must be nonempty")
    args.ranks = list(dict.fromkeys(args.ranks))
    args.work_dir.mkdir(parents=True)
    summary = dict(started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                   executable=dict(path=str(args.executable), sha256=digest(args.executable)),
                   baseline=(dict(path=str(args.baseline), sha256=digest(args.baseline))
                             if args.baseline is not None else None),
                   sources=source_hashes(),
                   launcher=args.launcher, ranks=args.ranks, job_timeout_seconds=args.job_timeout,
                   residual_tolerance=dict(rtol=RESIDUAL_RTOL, atol=RESIDUAL_ATOL),
                   residual_note="Console true residuals are rounded to six decimal places in scientific notation.",
                   output_tolerance_policy="Existing own-profile snapshot comparison tolerances; docs/developer/testing.md.",
                   qualification="Full/split numerical and lifetime regression, no performance claim.",
                   pairs=[], rejection_checks=[])

    def save():
        results = summary["pairs"] + summary["rejection_checks"]
        for status in ("passed", "failed", "skipped"):
            summary[status] = sum(result["status"] == status for result in results)
        write_json(args.work_dir / "results.json", summary)

    for case in cases(args.ranks):
        result = run_pair(case, args)
        summary["pairs"].append(result)
        save()
        print(f"{case['name']}: {result['status']}" +
              (": " + result.get("error", result.get("reason", "")) if result["status"] != "passed" else ""),
              flush=True)
    if 2 in args.ranks:
        for smoothing in (False, True):
            result = mismatch_case(args, smoothing)
            summary["rejection_checks"].append(result)
            save()
            print(f"{result['name']}: {result['status']}" +
                  (": " + result["error"] if "error" in result else ""), flush=True)
    print(f"{summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped; "
          f"{args.work_dir / 'results.json'}")
    return int(summary["failed"] != 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_mismatched-rank":
        mismatched_rank()
    else:
        raise SystemExit(main())
