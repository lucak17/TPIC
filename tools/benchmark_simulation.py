#!/usr/bin/env python3
"""Compare existing TPIC binaries on frozen copies of a one-rank input.

Runs are sequential, use separate output directories, and never overwrite the
source input. Timings use matched cycle indices; repeated-run means, rather than
individual timesteps, are the samples for the reported paired comparison.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import shutil
import subprocess
import time


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gpu_state():
    if not shutil.which('nvidia-smi'):
        return None
    queries = dict(device='--query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,memory.used,utilization.gpu',
                   processes='--query-compute-apps=pid,process_name,used_memory')
    result = {}
    for label, query in queries.items():
        state = subprocess.run(['nvidia-smi', query, '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=10)
        result[label] = dict(returncode=state.returncode, output=state.stdout.strip(),
                             error=state.stderr.strip())
    return result


def replace_input(source, values):
    for name, value in values.items():
        pattern = rf'(?m)^\s*{re.escape(name)}\s*=.*$'
        line = f'{name} = {value}'
        source, count = re.subn(pattern, lambda _: line, source)
        if not count:
            source += '\n' + line + '\n'
    return source


def parse_log(path, cycles, first, field_cycle):
    blocks = re.split(r'======= ECSIM Cycle (\d+)[^\n]*', path.read_text())
    rows = []
    for index in range(1, len(blocks), 2):
        cycle, body = int(blocks[index]), blocks[index + 1]
        timing = re.search(r'Cycle time\s*:\s*([\d.]+) ms', body)
        iterations = re.search(r'converged in (\d+) iterations', body)
        if timing is None or iterations is None:
            raise RuntimeError(f'Incomplete cycle {cycle}: {path}')
        rows.append(dict(cycle=cycle, ms=float(timing[1]), iterations=int(iterations[1])))
    if [row['cycle'] for row in rows] != list(range(cycles)):
        raise RuntimeError(f'Expected {cycles} complete cycles: {path}')
    selected = [row for row in rows if row['cycle'] >= first and
                (field_cycle == 0 or row['cycle'] % field_cycle)]
    if not selected:
        raise RuntimeError('No ordinary cycles remain in the requested window')
    return dict(cycles=rows, selected_cycles=[row['cycle'] for row in selected],
                ordinary_mean_ms=statistics.mean(row['ms'] for row in selected),
                ordinary_median_ms=statistics.median(row['ms'] for row in selected),
                iteration_counts=sorted(set(row['iterations'] for row in rows)))


def diagnostics(path):
    if not path.exists():
        return None
    with path.open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith('#')))
    if not rows:
        raise RuntimeError(f'Empty diagnostics: {path}')
    energies = [float(row['total_energy']) for row in rows]
    if not all(math.isfinite(value) for value in energies):
        raise RuntimeError(f'Nonfinite energy: {path}')
    return dict(rows=len(rows), initial_energy=energies[0], final_energy=energies[-1],
                max_absolute_relative_energy_drift=max(
                    abs(value - energies[0]) / abs(energies[0]) for value in energies)
                    if energies[0] else None,
                final_relative_energy_drift=(energies[-1] - energies[0]) / abs(energies[0])
                    if energies[0] else None)


def comparisons(runs):
    result = {}
    paired = []
    for pair in sorted({run['pair'] for run in runs}):
        members = {run['name']: run for run in runs if run['pair'] == pair}
        if 'candidate' in members and 'baseline' in members:
            candidate, baseline = members['candidate'], members['baseline']
            if candidate['selected_cycles'] != baseline['selected_cycles']:
                raise RuntimeError('Mismatched timing windows')
            paired.append(candidate['ordinary_mean_ms'] - baseline['ordinary_mean_ms'])
    if paired:
        item = dict(pairs=len(paired), candidate_minus_other_ms=paired,
                    mean_difference_ms=statistics.mean(paired))
        if len(paired) >= 2:
            rng = random.Random(9216)
            bootstrap = sorted(statistics.mean(rng.choices(paired, k=len(paired)))
                               for _ in range(10000))
            item['bootstrap_run_pair_95_percent_interval_ms'] = [bootstrap[249], bootstrap[9749]]
        result['baseline'] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=200)
    parser.add_argument('--first-cycle', type=int, default=20)
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--mass-matrix', choices=('scalar', 'matrix-core'), default='scalar')
    parser.add_argument('--snapshots', action='store_true', help='retain original field-output cadence')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if not (1 <= args.first_cycle < args.cycles and args.pairs > 0 and args.threads > 0):
        parser.error('Require 1 <= first-cycle < cycles, positive pairs and threads')
    source = args.input.read_text()
    topology = [re.search(rf'(?m)^\s*{axis}LEN\s*=\s*(\d+)', source) for axis in 'XYZ']
    if not all(topology) or math.prod(int(match[1]) for match in topology) != 1:
        parser.error('This benchmark expects a one-rank input')
    field_match = re.search(r'(?m)^\s*FieldOutputCycle\s*=\s*(\d+)', source)
    field_cycle = int(field_match[1]) if args.snapshots and field_match else 0
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / 'original.inp').write_text(source)
    executables = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    hashes = {name: sha256(path) for name, path in executables.items()}
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), TPIC_PROFILE_STAGES='0')
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  input_sha256=sha256(args.input), executable_sha256=hashes,
                  options={key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()}, runs=[])
    for pair in range(args.pairs):
        order = list(executables)
        if pair % 2:
            order.reverse()
        for name in order:
            directory = work / f'{pair:02}-{name}'
            directory.mkdir()
            output = directory / 'output'
            output.mkdir()
            inp = directory / 'simulation.inp'
            inp.write_text(replace_input(source, dict(ncycles=args.cycles, SaveDirName=output,
                RestartDirName=output, FieldOutputCycle=field_cycle)))
            executable = executables[name]
            if sha256(executable) != hashes[name]:
                raise RuntimeError(f'Executable changed during benchmark: {executable}')
            command = [str(executable), '--run', str(inp), '--order', 'full',
                       '--mass-matrix', args.mass_matrix]
            log = directory / 'simulation.out'
            started = time.monotonic()
            run = dict(pair=pair, name=name, command=command,
                       started_utc=datetime.now(timezone.utc).isoformat(),
                       input_sha256=sha256(inp), log=str(log))
            try:
                run['gpu_state_before'] = gpu_state()
                with log.open('w') as stream:
                    application_started = time.monotonic()
                    try:
                        process = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT, timeout=args.timeout)
                    finally:
                        run['application_elapsed_seconds'] = time.monotonic() - application_started
                run['returncode'] = process.returncode
                if process.returncode:
                    raise RuntimeError(f'Benchmark failed: {log}')
                run.update(parse_log(log, args.cycles, args.first_cycle, field_cycle))
                run['diagnostics'] = diagnostics(output / 'diagnostics.csv')
            except Exception as error:
                run.update(elapsed_seconds=time.monotonic()-started, error=str(error))
                report['runs'].append(run)
                (work / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
                raise
            run['elapsed_seconds'] = time.monotonic()-started
            report['runs'].append(run)
            report['comparisons'] = comparisons(report['runs'])
            (work / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({key: run[key] for key in
                  ('pair', 'name', 'ordinary_mean_ms', 'ordinary_median_ms', 'iteration_counts', 'diagnostics')}), flush=True)
    print(json.dumps(report['comparisons'], indent=2), flush=True)


if __name__ == '__main__':
    main()
