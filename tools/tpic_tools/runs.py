"""Bounded subprocess jobs and run metadata, with standard-library dependencies."""
import csv
import hashlib
import os
import signal
import subprocess
import time


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_csv(directory, name="diagnostics.csv"):
    with (directory / name).open() as source:
        return list(csv.DictReader(line for line in source if not line.startswith("#")))


def run_command(command, directory, timeout, environment=None):
    started = time.monotonic()
    report = dict(command=command, timeout_seconds=timeout, timed_out=False)
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env=environment, start_new_session=True)
        try:
            report["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            report["timed_out"] = True
            # Retire the entire MPI job, not only the launcher parent.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            report["returncode"] = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    report["elapsed_seconds"] = time.monotonic() - started
    return report, (directory / "stdout.log").read_text(), (directory / "stderr.log").read_text()
