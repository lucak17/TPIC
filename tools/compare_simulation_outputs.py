#!/usr/bin/env python3
"""Compare two completed TPIC run directories without changing either run.

Each directory must contain simulation.out and output/diagnostics.csv. Matching
VTK field files are compared when present. This uses only the Python standard
library and reads binary arrays in bounded chunks, including large DoubleGEM
outputs. Thresholds are fixed for the invocation; they are never relaxed.

Example:
  python3 tools/compare_simulation_outputs.py before after --report comparison.json

Default numeric thresholds follow the FP64 integration checks. Other precision
profiles need explicitly chosen thresholds; the script never infers them from
the profile. Values pass when abs(candidate-baseline) <= atol + rtol times the
larger absolute value. --energy-drift-atol additionally bounds the difference between the
two normalized energy-drift histories, independently of diagnostic magnitudes.
"""

import argparse
from array import array
import csv
import json
import math
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tpic_tools.vtk import cell_arrays, image_header, require, sample_count


class Errors:
    def __init__(self, rtol, atol):
        self.rtol, self.atol = rtol, atol
        self.values = self.mismatches = 0
        self.maximum_absolute = self.maximum_relative = self.maximum_tolerance_ratio = 0.0
        self.maximum_absolute_location = None
        self.examples = []

    def add(self, baseline, candidate, location):
        self.values += 1
        finite = math.isfinite(baseline) and math.isfinite(candidate)
        absolute = abs(candidate - baseline) if finite else math.inf
        scale = max(abs(baseline), abs(candidate)) if finite else 0.0
        relative = absolute / scale if scale else absolute
        threshold = self.atol + self.rtol * scale
        ratio = absolute / threshold if threshold else (0.0 if absolute == 0 else math.inf)
        if absolute > self.maximum_absolute:
            self.maximum_absolute = absolute
            self.maximum_absolute_location = location
        self.maximum_relative = max(self.maximum_relative, relative)
        self.maximum_tolerance_ratio = max(self.maximum_tolerance_ratio, ratio)
        if not finite or absolute > threshold:
            self.mismatches += 1
            if len(self.examples) < 5:
                self.examples.append(dict(location=location, baseline=baseline,
                                          candidate=candidate, threshold=threshold))

    def report(self):
        return dict(vars(self), passed=self.mismatches == 0)


def read_log(path):
    source = path.read_text()
    blocks = re.split(r'======= ECSIM Cycle (\d+)[^\n]*', source)
    iterations = []
    for index in range(1, len(blocks), 2):
        match = re.search(r'Maxwell GMRES\s*:\s*converged in (\d+) iterations', blocks[index + 1])
        require(match is not None, f'{path}: incomplete/nonconverged cycle {blocks[index]}')
        iterations.append((int(blocks[index]), int(match[1])))
    summary_start = source.rfind('\nSimulation complete')
    require(summary_start >= 0, f'{path}: missing completed simulation summary')
    summary = source[summary_start:]
    completed = re.search(r'Timesteps completed\s*:\s*(\d+)', summary)
    require(completed is not None, f'{path}: missing completed timestep report')
    cycles = int(completed[1])
    require([cycle for cycle, _ in iterations] == list(range(cycles)),
            f'{path}: incomplete or duplicate cycle sequence')
    total = re.search(r'Species / final particles\s*:\s*(\d+)\s*/\s*(\d+)', summary)
    require(total is not None, f'{path}: missing final particle report')
    species = int(total[1])
    counts = {int(index): int(count) for index, count in
              re.findall(r'Species (\d+)\s*:\s*(\d+) particles\b', summary)}
    require(set(counts) == set(range(species)), f'{path}: incomplete species counts')
    require(sum(counts.values()) == int(total[2]), f'{path}: inconsistent final particle count')
    precision = re.search(r'^Precision\s*:\s*(\S+)', source, re.MULTILINE)
    require(precision is not None, f'{path}: missing precision profile')
    diagnostic_rows = re.search(r'Diagnostics rows\s*:\s*(\d+)', summary)
    snapshots = re.search(r'Field snapshots\s*:\s*(\d+)', summary)
    return dict(completed_cycles=cycles, gmres_iterations=iterations,
                species_particles=counts, final_particles=int(total[2]), precision=precision[1],
                diagnostic_rows=int(diagnostic_rows[1]) if diagnostic_rows else 0,
                snapshots=int(snapshots[1]) if snapshots else 0)


def read_csv(path):
    with path.open() as stream:
        comments, lines = [], []
        for line in stream:
            (comments if line.startswith('#') else lines).append(line)
    reader = csv.DictReader(lines)
    rows = list(reader)
    require(rows and reader.fieldnames, f'{path}: empty diagnostics')
    require(len(set(reader.fieldnames)) == len(reader.fieldnames), f'{path}: duplicate CSV columns')
    precision = re.search(r'^# precision_profile\s*=\s*(\S+)', ''.join(comments), re.MULTILINE)
    require(precision is not None, f'{path}: missing precision profile')
    for row in rows:
        require(None not in row and all(value is not None for value in row.values()),
                f'{path}: malformed CSV record')
    return reader.fieldnames, rows, precision[1]


def compare_diagnostics(baseline, candidate, logs, args):
    columns, before, precision = read_csv(baseline / 'output/diagnostics.csv')
    after_columns, after, after_precision = read_csv(candidate / 'output/diagnostics.csv')
    require(columns == after_columns, 'diagnostics: column names/order differ')
    require(precision == after_precision == logs[0]['precision'] == logs[1]['precision'],
            'diagnostics/logs: precision profiles differ')
    identities = lambda rows: [(int(row['cycle']), float(row['time']), row['state']) for row in rows]
    epochs = identities(before)
    require(epochs == identities(after), 'diagnostics: cycles/times/states differ')
    require(len(epochs) == logs[0]['diagnostic_rows'] == logs[1]['diagnostic_rows'],
            'diagnostics: row count disagrees with completed run summary')
    require(epochs[0][0] == 0 and epochs[0][1] == 0.0, 'diagnostics: missing initial state')
    require(all(math.isfinite(time) for _, time, _ in epochs), 'diagnostics: nonfinite time')
    require(all(a[0] < b[0] and a[1] < b[1] for a, b in zip(epochs, epochs[1:])),
            'diagnostics: cycles/times are not strictly increasing')
    require(epochs[-1][0] == logs[0]['completed_cycles'] and epochs[-1][2] == 'final',
            'diagnostics: missing completed final state')
    require(all(state == 'pre_solve' for _, _, state in epochs[:-1]),
            'diagnostics: unexpected intermediate state')
    errors = {column: Errors(args.diagnostic_rtol, args.diagnostic_atol)
              for column in columns if column not in ('cycle', 'time', 'state')}
    for epoch, left, right in zip(epochs, before, after):
        for column, values in errors.items():
            values.add(float(left[column]), float(right[column]), dict(cycle=epoch[0], state=epoch[2]))
    energy = []
    for rows in (before, after):
        values = [float(row['total_energy']) for row in rows]
        require(all(math.isfinite(value) for value in values), 'diagnostics: nonfinite total energy')
        initial = values[0]
        energy.append(None if initial == 0 else [(value - initial) / abs(initial) for value in values])
    drift = dict(baseline_maximum_absolute=None, candidate_maximum_absolute=None,
                 maximum_difference=None, absolute_tolerance=args.energy_drift_atol, passed=None)
    if all(history is not None for history in energy):
        drift.update(baseline_maximum_absolute=max(map(abs, energy[0])),
                     candidate_maximum_absolute=max(map(abs, energy[1])),
                     maximum_difference=max(abs(a - b) for a, b in zip(*energy)))
        if args.energy_drift_atol is not None:
            drift['passed'] = drift['maximum_difference'] <= args.energy_drift_atol
    elif args.energy_drift_atol is not None:
        require(False, 'diagnostics: normalized energy drift is undefined for zero initial energy')
    return dict(rows=len(epochs), precision=precision, columns={k: v.report() for k, v in errors.items()},
                energy_drift=drift, passed=all(value.mismatches == 0 for value in errors.values())
                and drift['passed'] is not False)


def xml_identity(element):
    return (element.tag, sorted(element.attrib.items()), (element.text or '').split(),
            [xml_identity(child) for child in element])


def read_chunk(stream, count, endian):
    values = array('d')
    values.fromfile(stream, count)
    if (endian == '<') != (sys.byteorder == 'little'):
        values.byteswap()
    return values


def field_cycles(directory):
    return {path.name: int(match[1]) for path in directory.glob('*.vti')
            if (match := re.fullmatch(r'.+_t(\d+)\.vti', path.name))}


def compare_snapshots(baseline, candidate, args, logs):
    left, right = field_cycles(baseline / 'output'), field_cycles(candidate / 'output')
    require(left.keys() == right.keys(), 'snapshots: field filenames differ')
    files, cycles = sorted(left), set(left.values())
    if logs is not None:
        require(len(cycles) == logs[0]['snapshots'] == logs[1]['snapshots'],
                'snapshots: field cycle count disagrees with completed run summary')
    reports = {}
    for name in files:
        paths = baseline / 'output' / name, candidate / 'output' / name
        headers = [image_header(path) for path in paths]
        require(xml_identity(headers[0][0]) == xml_identity(headers[1][0]), f'{name}: field metadata differ')
        root, image, box, arrays, _ = headers[0]
        endian = '<' if root.get('byte_order') == 'LittleEndian' else '>'
        with paths[0].open('rb') as first, paths[1].open('rb') as second:
            piece = image.find('Piece')
            cells = cell_arrays(piece)
            for descriptor in piece.iter('DataArray'):
                array_name = descriptor.attrib['Name']
                require(descriptor.attrib['format'] == 'appended', f'{name}: non-appended array')
                components = arrays[array_name]
                count = sample_count(box, array_name in cells) * components
                for stream, (*_, origin) in zip((first, second), headers):
                    stream.seek(origin + int(descriptor.attrib['offset']))
                    length = struct.unpack(endian + 'Q', stream.read(8))[0]
                    require(length == count * 8, f'{name}: incorrect {array_name} payload size')
                errors = reports.setdefault(array_name, Errors(args.field_rtol, args.field_atol))
                for start in range(0, count, 65536):
                    width = min(count - start, 65536)
                    a = read_chunk(first, width, endian)
                    b = read_chunk(second, width, endian)
                    # Common fixed-state/self comparisons avoid creating a
                    # location for every identical finite value. Nonfinite
                    # matching data still must be rejected.
                    if a == b and all(map(math.isfinite, a)):
                        errors.values += width
                        continue
                    for index, (before, after) in enumerate(zip(a, b), start):
                        errors.add(before, after, dict(file=name, value_index=index,
                                                      node_index=index // components,
                                                      component=index % components))
    return dict(files=len(files), cycles=len(cycles),
                arrays={name: values.report() for name, values in reports.items()},
                passed=all(values.mismatches == 0 for values in reports.values()))


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--report', type=Path, required=True, help='JSON comparison report')
    parser.add_argument('--field-rtol', type=float, default=3e-10)
    parser.add_argument('--field-atol', type=float, default=3e-12)
    parser.add_argument('--diagnostic-rtol', type=float, default=3e-10)
    parser.add_argument('--diagnostic-atol', type=float, default=3e-12)
    parser.add_argument('--energy-drift-atol', type=float,
                        help='optional absolute tolerance on normalized energy-drift differences')
    args = parser.parse_args()
    for name in ('field_rtol', 'field_atol', 'diagnostic_rtol', 'diagnostic_atol', 'energy_drift_atol'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    args.baseline, args.candidate = args.baseline.resolve(), args.candidate.resolve()
    report_path = args.report.resolve()
    if any(report_path.is_relative_to(directory) for directory in (args.baseline, args.candidate)):
        parser.error('--report must be outside both input run directories')
    report = dict(baseline=str(args.baseline), candidate=str(args.candidate),
                  comparison_rule='abs(candidate-baseline) <= atol + rtol * max(abs(baseline), abs(candidate))',
                  thresholds={name: getattr(args, name) for name in vars(args)
                              if name.endswith(('_rtol', '_atol'))}, errors=[])
    logs = None
    try:
        logs = [read_log(directory / 'simulation.out') for directory in (args.baseline, args.candidate)]
        report['logs'] = dict(baseline=logs[0], candidate=logs[1], passed=logs[0] == logs[1])
    except (OSError, ValueError, KeyError) as error:
        report['errors'].append(f'logs: {error}')
    if logs is not None:
        try:
            report['diagnostics'] = compare_diagnostics(args.baseline, args.candidate, logs, args)
        except (OSError, ValueError, KeyError) as error:
            report['errors'].append(f'diagnostics: {error}')
    try:
        report['snapshots'] = compare_snapshots(args.baseline, args.candidate, args, logs)
    except (OSError, ValueError, KeyError, TypeError, EOFError, ET.ParseError, struct.error) as error:
        report['errors'].append(f'snapshots: {error}')
    report['passed'] = not report['errors'] and all(report.get(part, {}).get('passed', False)
                                                    for part in ('logs', 'diagnostics', 'snapshots'))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')
    print(f'{"PASS" if report["passed"] else "FAIL"}: {args.report}')
    for error in report['errors']:
        print(error, file=sys.stderr)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
