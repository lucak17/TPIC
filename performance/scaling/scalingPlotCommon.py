"""Shared parsing and plotting for TPIC strong/weak scaling campaigns."""

import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "The TPIC scaling plot tools require Python 3.10 or newer "
        f"(found {sys.version_info.major}.{sys.version_info.minor})."
    )

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class FontSizes:
    """Font sizes in points for all scaling plots; edit the defaults here."""

    text: float = 20
    title: float = 20
    axis_label: float = 24
    legend: float = 20
    legend_title: float = 20
    x_tick: float = 20
    y_tick: float = 20
    annotation: float = 16
    figure_title: float = 20
    figure_label: float = 20


FONT_SIZES = FontSizes()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator, ScalarFormatter
except ImportError as error:
    raise SystemExit(
        "The TPIC scaling plot tools require Matplotlib. "
        "Install it in the active Python environment and retry."
    ) from error


matplotlib.rcParams.update(
    {
        "font.size": FONT_SIZES.text,
        "axes.titlesize": FONT_SIZES.title,
        "axes.labelsize": FONT_SIZES.axis_label,
        "legend.fontsize": FONT_SIZES.legend,
        "legend.title_fontsize": FONT_SIZES.legend_title,
        "xtick.labelsize": FONT_SIZES.x_tick,
        "ytick.labelsize": FONT_SIZES.y_tick,
        "figure.titlesize": FONT_SIZES.figure_title,
        "figure.labelsize": FONT_SIZES.figure_label,
    }
)


ScalingKind = Literal["strong", "weak"]

STDOUT_PATTERN = re.compile(r"^stdout_N(?P<nodes>\d+)(?:_|$)")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

TOTAL_CYCLES_PATTERN = re.compile(
    r"^\s*Total cycles:\s*(?P<cycles>\d+)\s*$", re.MULTILINE
)
ECSIM_TOTAL_PATTERN = re.compile(
    rf"^\s*Total time:\s*(?P<seconds>{NUMBER})\s+s\s+\(maximum rank\)\s*$",
    re.MULTILINE,
)
STEADY_CYCLES_PATTERN = re.compile(
    r"^\s*Steady-state cycles:\s*(?P<cycles>\d+)\s*$",
    re.MULTILINE,
)
STEADY_TIME_PATTERN = re.compile(
    rf"^\s*Steady-state time:\s*(?P<seconds>{NUMBER})\s+s\s+"
    r"\(maximum rank; first cycle excluded\)\s*$",
    re.MULTILINE,
)
ATTEMPT_PATTERN = re.compile(
    r"^TPIC_SCALING_ATTEMPT\s+solver=(?P<solver>ecsim)\s+"
    r"restart_count=(?P<restart_count>\d+)\s*$",
    re.MULTILINE,
)
ATTEMPT_SENTINEL_PATTERN = re.compile(
    r"^.*TPIC_SCALING_ATTEMPT.*$", re.MULTILINE
)
SCALING_COMPLETION_PATTERN = re.compile(
    r"^TPIC_SCALING_COMPLETE\s+solver=(?P<solver>ecsim)\s*$",
    re.MULTILINE,
)

COMPLETION_PATTERNS = {
    "ecsim": re.compile(r"^ECSIM Main Loop Complete\s*$", re.MULTILINE),
}
SOLVER_NONCONVERGENCE_PATTERNS = {
    "ecsim": re.compile(r"GMRES did not converge"),
}
INVALID_DIAGNOSTIC_PATTERNS = (
    (
        "TPIC runtime initialization failure",
        re.compile(r"TPIC startup: runtime initialization: lpfx="),
    ),
    ("TPIC execution failure", re.compile(r"TPIC execution(?: rank \d+)?: ")),
    ("fatal error", re.compile(r"\bFatal error\b", re.IGNORECASE)),
    (
        "MPI abort",
        re.compile(r"\bMPI_Abort\b|\baborting\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class Timing:
    seconds: float
    reported_cycles: int
    measured_cycles: int
    solver: str
    source: str


@dataclass(frozen=True)
class Measurement:
    nodes: int
    timing: Timing
    stdout_path: Path
    tasks_per_node: int
    gpu_label: str

    @property
    def gpus(self) -> int:
        return self.nodes * self.tasks_per_node


@dataclass(frozen=True)
class Attempt:
    solver: str
    restart_count: int
    start: int


@dataclass(frozen=True)
class CampaignManifest:
    planned_nodes: frozenset[int]
    solver: str
    kind: ScalingKind
    tasks_per_node: int
    gpu_label: str


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _attempts(output: str, source_label: str) -> list[Attempt]:
    marker_lines = list(ATTEMPT_SENTINEL_PATTERN.finditer(output))
    matches = list(ATTEMPT_PATTERN.finditer(output))
    if len(marker_lines) != len(matches):
        raise ValueError(f"Malformed scaling-attempt marker in {source_label}")

    attempts = [
        Attempt(
            match.group("solver"),
            int(match.group("restart_count")),
            match.start(),
        )
        for match in matches
    ]
    if not attempts:
        return attempts

    solvers = {attempt.solver for attempt in attempts}
    if len(solvers) != 1:
        raise ValueError(f"Scaling attempts mix solvers in {source_label}")

    restart_counts = [attempt.restart_count for attempt in attempts]
    if len(set(restart_counts)) != len(restart_counts):
        raise ValueError(f"Duplicate scaling-attempt marker in {source_label}")
    if any(
        current <= previous
        for previous, current in zip(restart_counts, restart_counts[1:])
    ):
        raise ValueError(
            f"Scaling-attempt restart counts are not increasing in {source_label}"
        )
    return attempts


def _diagnostics_for_attempt(
    diagnostics: str,
    attempt: Attempt,
    source_label: str,
) -> str:
    if not diagnostics:
        return diagnostics

    diagnostic_attempts = _attempts(diagnostics, source_label)
    if not diagnostic_attempts:
        return diagnostics

    latest = diagnostic_attempts[-1]
    if (
        latest.solver != attempt.solver
        or latest.restart_count != attempt.restart_count
    ):
        raise ValueError(
            f"Latest stdout/stderr scaling attempts do not match in {source_label}"
        )
    return diagnostics[latest.start:]


def _warn_about_diagnostics(
    output: str,
    solver: str,
    source_label: str,
) -> None:
    if SOLVER_NONCONVERGENCE_PATTERNS[solver].search(output):
        _warn(f"{solver.upper()} solver non-convergence found in {source_label}")
    for description, pattern in INVALID_DIAGNOSTIC_PATTERNS:
        if pattern.search(output):
            _warn(f"{description} found in {source_label}")


def _only_match(
    pattern: re.Pattern[str], output: str, description: str, source_label: str
) -> re.Match[str]:
    matches = list(pattern.finditer(output))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {description} in {source_label}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _single_manifest_value(
    entries: dict[str, list[str]], key: str, manifest_path: Path
) -> str:
    values = entries.get(key, [])
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one {key}= entry in {manifest_path}; "
            f"found {len(values)}"
        )
    return values[0]


def _load_campaign_manifest(results_folder: Path) -> CampaignManifest:
    manifest_path = results_folder / "manifest.txt"
    if not manifest_path.exists():
        raise ValueError(f"Scaling campaign requires manifest.txt: {results_folder}")
    if not manifest_path.is_file():
        raise ValueError(f"Campaign manifest is not a file: {manifest_path}")

    entries: dict[str, list[str]] = {}
    for line in manifest_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        entries.setdefault(key.strip(), []).append(value.strip())

    manifest_format = _single_manifest_value(entries, "format", manifest_path)
    if manifest_format != "TPIC-scaling-v1":
        raise ValueError(
            f"Unsupported campaign manifest format in {manifest_path}: "
            f"{manifest_format}"
        )

    kind = _single_manifest_value(entries, "kind", manifest_path)
    if kind not in {"strong", "weak"}:
        raise ValueError(f"Invalid scaling kind in {manifest_path}: {kind}")

    solver = _single_manifest_value(entries, "solver", manifest_path)
    if solver != "ecsim":
        raise ValueError(f"Invalid solver entry in {manifest_path}: {solver}")

    tasks_per_node = _single_manifest_value(
        entries, "tasks_per_node", manifest_path
    )
    if re.fullmatch(r"[1-9]\d*", tasks_per_node) is None:
        raise ValueError(f"Malformed tasks_per_node in {manifest_path}")
    gpu_label = _single_manifest_value(entries, "gpu_label", manifest_path)

    submission_complete = _single_manifest_value(
        entries, "submission_complete", manifest_path
    )
    if submission_complete != "true":
        _warn(
            f"Campaign submission is not complete according to {manifest_path}; "
            "plotting the runs that finished"
        )

    planned_value = _single_manifest_value(
        entries, "planned_nodes", manifest_path
    )
    if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", planned_value) is None:
        raise ValueError(f"Malformed planned_nodes entry in {manifest_path}")
    planned_list = [int(value) for value in planned_value.split(",")]
    if any(nodes < 1 for nodes in planned_list):
        raise ValueError(
            f"planned_nodes must contain positive counts in {manifest_path}"
        )
    if len(set(planned_list)) != len(planned_list):
        raise ValueError(f"Duplicate node count in planned_nodes in {manifest_path}")

    return CampaignManifest(
        frozenset(planned_list), solver, kind, int(tasks_per_node), gpu_label
    )


def parse_timing(
    output: str,
    source: Union[Path, str] = "stdout",
    diagnostic_output: Optional[str] = None,
) -> Timing:
    """Parse a completed TPIC ECSIM timing summary.

    TPIC output uses the synchronized steady-state summary that skips the
    first cycle. The scaling inputs run 101 cycles, so 100 are measured.
    Output is accepted only when it has a worker-attempt marker, both
    completion markers and a consistent steady-state summary. Failure lines in
    the output or in the diagnostics are reported as warnings.
    """

    source_label = str(source)
    attempts = _attempts(output, source_label)
    if not attempts:
        raise ValueError(f"No scaling-attempt marker found in {source_label}")

    attempt = attempts[-1]
    solver = attempt.solver
    run_output = output[attempt.start:]

    _warn_about_diagnostics(run_output, solver, source_label)
    if diagnostic_output is not None:
        diagnostic_label = f"diagnostics for {source_label}"
        try:
            diagnostic_output = _diagnostics_for_attempt(
                diagnostic_output, attempt, diagnostic_label
            )
        except ValueError as error:
            _warn(str(error))
        _warn_about_diagnostics(diagnostic_output, solver, diagnostic_label)

    application_completion = _only_match(
        COMPLETION_PATTERNS[solver],
        run_output,
        f"{solver} application-completion marker",
        source_label,
    )
    scaling_completion = _only_match(
        SCALING_COMPLETION_PATTERN,
        run_output,
        "scaling-completion marker",
        source_label,
    )
    if scaling_completion.group("solver") != solver:
        raise ValueError(
            f"Scaling-completion solver does not match attempt in {source_label}"
        )
    if application_completion.start() >= scaling_completion.start():
        raise ValueError(
            f"Scaling completion precedes application completion in {source_label}"
        )

    cycles_match = _only_match(
        TOTAL_CYCLES_PATTERN, run_output, "total-cycle summary", source_label
    )
    total_match = _only_match(
        ECSIM_TOTAL_PATTERN, run_output, "ECSIM total-time summary", source_label
    )

    reported_cycles = int(cycles_match.group("cycles"))
    total_seconds = float(total_match.group("seconds"))
    if (
        reported_cycles < 2
        or not math.isfinite(total_seconds)
        or total_seconds <= 0.0
    ):
        raise ValueError(f"Invalid {solver} loop timing in {source_label}")

    steady_cycles_match = _only_match(
        STEADY_CYCLES_PATTERN,
        run_output,
        "steady-state cycle summary",
        source_label,
    )
    steady_time_match = _only_match(
        STEADY_TIME_PATTERN,
        run_output,
        "steady-state time summary",
        source_label,
    )
    measured_cycles = int(steady_cycles_match.group("cycles"))
    steady_state_seconds = float(steady_time_match.group("seconds"))
    if measured_cycles != reported_cycles - 1 or measured_cycles < 1:
        raise ValueError(
            f"Steady-state cycle count is inconsistent in {source_label}: "
            f"reported {reported_cycles}, measured {measured_cycles}"
        )
    if not math.isfinite(steady_state_seconds) or steady_state_seconds <= 0.0:
        raise ValueError(f"Non-positive steady-state timing in {source_label}")
    return Timing(
        steady_state_seconds,
        reported_cycles,
        measured_cycles,
        solver,
        "steady-state total (first cycle excluded)",
    )


def _node_count_list(value: str) -> list[int]:
    try:
        node_counts = [int(item) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated node counts, got {value!r}"
        ) from None
    if not node_counts:
        raise argparse.ArgumentTypeError("expected at least one node count")
    if any(nodes < 1 for nodes in node_counts):
        raise argparse.ArgumentTypeError("node counts must be positive")
    return node_counts


def parse_arguments(kind: ScalingKind) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Plot {kind}-scaling efficiency and execution time "
            "from TPIC stdout files."
        )
    )
    parser.add_argument(
        "results_folder",
        type=Path,
        help="campaign folder containing stdout_N<NODES>_<JOB_ID> files",
    )
    parser.add_argument(
        "--add-campaign",
        dest="additional_results_folders",
        action="append",
        default=[],
        type=Path,
        metavar="RESULTS_FOLDER",
        help=(
            "merge results from another campaign folder; repeat for more "
            "campaigns"
        ),
    )
    parser.add_argument("--min-nodes", type=int, help="minimum nodes (inclusive)")
    parser.add_argument("--max-nodes", type=int, help="maximum nodes (inclusive)")
    parser.add_argument(
        "--skip-nodes",
        action="extend",
        default=[],
        type=_node_count_list,
        metavar="NODES",
        help=(
            "node counts to leave out, comma-separated (e.g. 128,256); "
            "repeat to add more"
        ),
    )
    args = parser.parse_args()
    if args.min_nodes is not None and args.min_nodes < 1:
        parser.error("--min-nodes must be positive")
    if args.max_nodes is not None and args.max_nodes < 1:
        parser.error("--max-nodes must be positive")
    if (
        args.min_nodes is not None
        and args.max_nodes is not None
        and args.min_nodes > args.max_nodes
    ):
        parser.error("--min-nodes cannot exceed --max-nodes")
    return args


def load_measurements(
    results_folder: Path,
    min_nodes: Optional[int] = None,
    max_nodes: Optional[int] = None,
    skip_nodes: frozenset[int] = frozenset(),
    expected_kind: Optional[ScalingKind] = None,
    ignore_unplanned: bool = False,
) -> list[Measurement]:
    if not results_folder.is_dir():
        raise ValueError(f"Results folder does not exist: {results_folder}")

    manifest = _load_campaign_manifest(results_folder)
    if expected_kind is not None and manifest.kind != expected_kind:
        raise ValueError(
            f"Requested {expected_kind} scaling, but "
            f"{results_folder / 'manifest.txt'} declares {manifest.kind}"
        )
    expected_nodes = frozenset(
        nodes
        for nodes in manifest.planned_nodes
        if (min_nodes is None or nodes >= min_nodes)
        and (max_nodes is None or nodes <= max_nodes)
        and nodes not in skip_nodes
    )

    measurements: dict[int, Measurement] = {}
    for stdout_path in sorted(results_folder.glob("stdout_N*")):
        filename_match = STDOUT_PATTERN.match(stdout_path.name)
        if filename_match is None or not stdout_path.is_file():
            continue
        nodes = int(filename_match.group("nodes"))
        if min_nodes is not None and nodes < min_nodes:
            continue
        if max_nodes is not None and nodes > max_nodes:
            continue
        if nodes in skip_nodes:
            continue
        if nodes not in expected_nodes:
            if ignore_unplanned:
                continue
            _warn(f"Plotting unplanned {nodes}-node run: {stdout_path}")

        stderr_path = stdout_path.with_name(
            "stderr_" + stdout_path.name.removeprefix("stdout_")
        )
        diagnostic_output: Optional[str] = None
        if stderr_path.is_file():
            diagnostic_output = stderr_path.read_text(errors="replace")
            if ATTEMPT_SENTINEL_PATTERN.search(diagnostic_output) is None:
                _warn(f"Scaling-attempt marker is missing from {stderr_path}")
        else:
            _warn(f"Missing stderr: {stderr_path}")

        try:
            timing = parse_timing(
                stdout_path.read_text(errors="replace"),
                stdout_path,
                diagnostic_output,
            )
        except ValueError as error:
            _warn(f"Skipping the {nodes}-node run: {error}")
            continue
        if nodes in measurements:
            raise ValueError(f"Multiple completed runs found for {nodes} nodes")
        measurements[nodes] = Measurement(
            nodes, timing, stdout_path, manifest.tasks_per_node, manifest.gpu_label
        )

    if not measurements:
        raise ValueError(f"No completed run to plot in {results_folder}")
    missing_nodes = expected_nodes.difference(measurements)
    if missing_nodes:
        missing = ", ".join(str(nodes) for nodes in sorted(missing_nodes))
        _warn(f"No completed run for planned node counts: {missing}")

    ordered = [measurements[nodes] for nodes in sorted(measurements)]
    first_timing = ordered[0].timing
    for measurement in ordered[1:]:
        timing = measurement.timing
        if timing.solver != first_timing.solver:
            raise ValueError("Campaign mixes output from different solvers")
        if timing.reported_cycles != first_timing.reported_cycles:
            raise ValueError("Campaign runs report different cycle counts")
        if timing.measured_cycles != first_timing.measured_cycles:
            raise ValueError("Campaign runs use different measured cycle counts")
        if timing.source != first_timing.source:
            raise ValueError("Campaign runs use different timing sources")
    return ordered


def efficiencies_for(
    kind: ScalingKind, measurements: list[Measurement]
) -> list[float]:
    baseline = measurements[0]
    if kind == "strong":
        return [
            100.0
            * baseline.timing.seconds
            * baseline.gpus
            / (measurement.timing.seconds * measurement.gpus)
            for measurement in measurements
        ]
    return [
        100.0 * baseline.timing.seconds / measurement.timing.seconds
        for measurement in measurements
    ]


def create_plot(
    kind: ScalingKind,
    results_folder: Path,
    measurements: list[Measurement],
    efficiencies: list[float],
) -> Path:
    gpu_counts = [measurement.gpus for measurement in measurements]
    timing = measurements[0].timing
    kind_title = kind.capitalize()
    solver_title = timing.solver.upper()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        gpu_counts,
        efficiencies,
        color="#176b87",
        marker="o",
        markersize=7,
        linewidth=2,
        label="Measured efficiency",
    )
    ax.axhline(
        100.0,
        color="#c23b22",
        linestyle="--",
        linewidth=1.5,
        label="Ideal efficiency",
    )
    for gpu_count, efficiency in zip(gpu_counts, efficiencies):
        ax.annotate(
            f"{efficiency:.1f}%",
            (gpu_count, efficiency),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=FONT_SIZES.annotation,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(gpu_counts)
    ax.set_xticklabels(gpu_counts)
    ax.set_ylim(
        max(0.0, min(efficiencies) - 3.0),
        max(102.0, max(efficiencies) + 3.0),
    )
    ax.set_xlabel(f"Number of {measurements[0].gpu_label}", color="black")
    ax.set_ylabel("Parallel Efficiency (%)", color="black")
    ax.tick_params(axis="both", which="both", colors="black")
    ax.set_title(
        f"TPIC {solver_title} {kind_title} Scaling "
        f"({timing.measured_cycles} measured cycles)"
    )
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    ax.legend()

    output_path = results_folder / f"{kind}_scaling_efficiency.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_time_efficiency_plot(
    kind: ScalingKind,
    results_folder: Path,
    measurements: list[Measurement],
    efficiencies: list[float],
) -> tuple[Path, Path]:
    """Save a title-free plot of parsed execution time and scaling efficiency."""
    gpu_counts = [measurement.gpus for measurement in measurements]
    seconds = [measurement.timing.seconds for measurement in measurements]
    time_color = "#176b87"
    efficiency_color = "#c23b22"

    fig, time_ax = plt.subplots(figsize=(10, 6))
    efficiency_ax = time_ax.twinx()
    time_line, = time_ax.plot(
        gpu_counts,
        seconds,
        color=time_color,
        marker="o",
        linestyle="-",
        markersize=8,
        linewidth=2,
        label="Execution time",
    )
    efficiency_line, = efficiency_ax.plot(
        gpu_counts,
        efficiencies,
        color=efficiency_color,
        marker="^",
        linestyle="--",
        markersize=8,
        linewidth=2,
        label="Scaling efficiency",
    )
    for gpu_count, efficiency in zip(gpu_counts, efficiencies):
        efficiency_ax.annotate(
            f"{efficiency:.1f}%",
            (gpu_count, efficiency),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            color=efficiency_color,
            fontsize=FONT_SIZES.annotation,
            fontweight="bold",
        )

    time_ax.set_xscale("log", base=2)
    time_ax.set_xticks(gpu_counts)
    time_ax.set_xticklabels(gpu_counts)
    time_ax.set_xlabel(f"Number of {measurements[0].gpu_label}", color="black")
    time_ax.set_ylabel("Execution time (s)", color="black")
    time_ax.tick_params(axis="both", which="both", colors="black")
    if kind == "strong":
        time_ax.set_yscale("log")
        time_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        time_ax.yaxis.set_minor_locator(NullLocator())
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        time_ax.yaxis.set_major_formatter(formatter)
        time_ax.set_ylim(0.95 * min(seconds), 1.05 * max(seconds))
    else:
        time_ax.set_ylim(0.0, 1.05 * max(seconds))
    efficiency_ax.set_ylabel("Parallel Efficiency (%)", color="black")
    efficiency_ax.tick_params(axis="both", which="both", colors="black")
    efficiency_ax.set_ylim(0.0, max(110.0, 1.1 * max(efficiencies)))
    time_ax.grid(True, which="major", linestyle=":", alpha=0.6)
    efficiency_ax.legend(handles=[time_line, efficiency_line])

    output_paths = (
        results_folder / f"{kind}_scaling_time_efficiency.png",
        results_folder / f"{kind}_scaling_time_efficiency.pdf",
    )
    fig.tight_layout()
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_paths


def main(kind: ScalingKind) -> None:
    args = parse_arguments(kind)
    results_folder = args.results_folder.expanduser().resolve()
    additional_results_folders = [
        folder.expanduser().resolve()
        for folder in args.additional_results_folders
    ]
    campaign_folders = [results_folder, *additional_results_folders]
    mixed_campaigns = bool(additional_results_folders)
    measurements_by_nodes: dict[int, Measurement] = {}
    for campaign_folder in campaign_folders:
        campaign_measurements = load_measurements(
            campaign_folder,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            skip_nodes=frozenset(args.skip_nodes),
            expected_kind=kind,
            ignore_unplanned=mixed_campaigns,
        )
        for measurement in campaign_measurements:
            previous = measurements_by_nodes.get(measurement.nodes)
            if previous is not None:
                raise ValueError(
                    f"Multiple campaigns provide {measurement.nodes}-node runs: "
                    f"{previous.stdout_path} and {measurement.stdout_path}"
                )
            measurements_by_nodes[measurement.nodes] = measurement
    measurements = [
        measurements_by_nodes[nodes]
        for nodes in sorted(measurements_by_nodes)
    ]

    first_timing = measurements[0].timing
    for measurement in measurements[1:]:
        timing = measurement.timing
        if timing.solver != first_timing.solver:
            raise ValueError("Campaigns mix output from different solvers")
        if timing.reported_cycles != first_timing.reported_cycles:
            raise ValueError("Campaigns report different cycle counts")
        if timing.measured_cycles != first_timing.measured_cycles:
            raise ValueError("Campaigns use different measured cycle counts")
        if timing.source != first_timing.source:
            raise ValueError("Campaigns use different timing sources")
        if measurement.tasks_per_node != measurements[0].tasks_per_node:
            raise ValueError("Campaigns use different MPI ranks per node")
    efficiencies = efficiencies_for(kind, measurements)
    timing = measurements[0].timing

    print(
        f"Solver: {timing.solver}; timing: {timing.source}; "
        f"reported/measured cycles: "
        f"{timing.reported_cycles}/{timing.measured_cycles}"
    )
    for measurement, efficiency in zip(measurements, efficiencies):
        print(
            f"{measurement.nodes:3d} nodes "
            f"({measurement.gpus:4d} {measurement.gpu_label}): "
            f"{measurement.timing.seconds:.6f} s, "
            f"efficiency = {efficiency:.2f}%"
        )

    output_path = create_plot(kind, results_folder, measurements, efficiencies)
    print(f"Saved plot to {output_path}")
    for output_path in create_time_efficiency_plot(
        kind, results_folder, measurements, efficiencies
    ):
        print(f"Saved plot to {output_path}")
