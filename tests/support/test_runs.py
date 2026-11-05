"""Job lifetime and scientific-check import smoke tests, without MPI or a GPU."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tests')]
from tpic_tools.runs import run_command
from support import snapshot_checks


class RunTests(unittest.TestCase):
    def test_exit_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, out, err = run_command(
                [sys.executable, '-c', 'import sys; print("out"); print("err", file=sys.stderr); sys.exit(7)'],
                Path(tmp), 5)
            self.assertEqual(report['returncode'], 7)
            self.assertFalse(report['timed_out'])
            self.assertEqual((out, err), ('out\n', 'err\n'))

    @unittest.skipUnless(os.name == 'posix', 'MPI process-group launcher uses POSIX')
    def test_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _, _ = run_command([sys.executable, '-c', 'import time; time.sleep(60)'], Path(tmp), .1)
            self.assertTrue(report['timed_out'])
            self.assertNotEqual(report['returncode'], 0)
            self.assertLess(report['elapsed_seconds'], 5)

    def test_case_helpers_keep_policy(self):
        # A baseline discrepancy must still fail independently of VTK decoding.
        baseline = ({}, {'a': [1.]}, [])
        current = ({}, {'a': [2.]}, [])
        with self.assertRaisesRegex(AssertionError, 'changed'):
            snapshot_checks.compare(baseline, current)


if __name__ == '__main__':
    unittest.main()
