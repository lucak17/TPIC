"""Regression checks for the CTest suite runner, without MPI or a GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'tests/support/run_suite.cmake'
CMAKE = 'cmake'
SKIPPED_MARKER = 'TPIC_SUITE_SKIPPED: all cases skipped'


def cmake_argument(value):
    """Quote one CMake argument without expanding paths, escapes or lists."""
    value = str(value)
    delimiter = '='
    while f']{delimiter}]' in value:
        delimiter += '='
    return f'[{delimiter}[{value}]{delimiter}]'


class SuiteRunnerTests(unittest.TestCase):
    def run_suite(self, cases):
        with tempfile.TemporaryDirectory(prefix='tpic suite runner ') as directory:
            directory = Path(directory)
            trace = directory / 'child trace.jsonl'
            helper = directory / 'child helper.py'
            helper.write_text('''import json
import sys
import time
from pathlib import Path

trace, name, mode, *arguments = sys.argv[1:]
with Path(trace).open('a') as output:
    output.write(json.dumps({'name': name, 'arguments': arguments}) + '\\n')
print('child ' + name + ' started', flush=True)
if mode == 'sleep':
    time.sleep(float(arguments[0]))
elif mode == 'exit':
    sys.exit(int(arguments[0]))
''')
            lines = [f'include({cmake_argument(RUNNER)})']
            for case in cases:
                name = case['name']
                if case.get('missing'):
                    command = [directory / 'executable that does not exist']
                else:
                    command = [sys.executable, helper, trace, name,
                               case.get('mode', 'exit'),
                               *case.get('arguments', ['0'])]
                arguments = ['NAME', cmake_argument(name), 'TIMEOUT',
                             str(case.get('timeout', 5))]
                if case.get('skip'):
                    arguments.extend(['SKIP_RETURN_CODE', '77'])
                arguments.extend(['COMMAND', *map(cmake_argument, command)])
                lines.append('tpic_run_case(' + ' '.join(arguments) + ')')
            lines.append('tpic_finish_suite()')
            script = directory / 'suite.cmake'
            script.write_text('\n'.join(lines) + '\n')
            result = subprocess.run([CMAKE, '-P', str(script)],
                                    capture_output=True, text=True, timeout=30)
            records = ([json.loads(line) for line in trace.read_text().splitlines()]
                       if trace.exists() else [])
            return result.returncode, result.stdout + result.stderr, records

    def test_arguments_preserve_boundaries(self):
        # execute_process itself reserves COMMAND/TIMEOUT; exercise runner-only
        # option names as literal data along with CMake quoting edge cases.
        arguments = ['two words', 'a;b;c', r'a\;b', '', ']==]', '${literal}',
                     'value with "quotes"', 'line one\nline two',
                     'NAME', 'SKIP_RETURN_CODE']
        code, output, records = self.run_suite([
            {'name': 'arguments', 'mode': 'arguments', 'arguments': arguments}])
        self.assertEqual(code, 0, output)
        self.assertEqual(records, [{'name': 'arguments', 'arguments': arguments}])
        self.assertNotIn(SKIPPED_MARKER, output)

    def test_all_skipped(self):
        code, output, records = self.run_suite([
            {'name': name, 'arguments': ['77'], 'skip': True}
            for name in ('first', 'second')])
        self.assertEqual(code, 0, output)
        self.assertIn(SKIPPED_MARKER, output)
        self.assertEqual([record['name'] for record in records], ['first', 'second'])

    def test_mixed_pass_and_skip(self):
        code, output, records = self.run_suite([
            {'name': 'skipped', 'arguments': ['77'], 'skip': True},
            {'name': 'passed'}])
        self.assertEqual(code, 0, output)
        self.assertNotIn(SKIPPED_MARKER, output)
        self.assertEqual([record['name'] for record in records], ['skipped', 'passed'])
        self.assertRegex(output, r'1 passed')
        self.assertRegex(output, r'1 skipped')

    def test_failure_after_skip_does_not_stop_remaining_cases(self):
        code, output, records = self.run_suite([
            {'name': 'skipped_first', 'arguments': ['77'], 'skip': True},
            {'name': 'failed', 'arguments': ['9']},
            {'name': 'passed_after_failure'},
            {'name': 'skipped_last', 'arguments': ['77'], 'skip': True}])
        self.assertNotEqual(code, 0, output)
        self.assertNotIn(SKIPPED_MARKER, output)
        self.assertEqual([record['name'] for record in records],
                         ['skipped_first', 'failed', 'passed_after_failure',
                          'skipped_last'])
        self.assertIn('failed', output)

    def test_exit_77_requires_explicit_skip_policy(self):
        code, output, records = self.run_suite([
            {'name': 'not_a_skip', 'arguments': ['77']}])
        self.assertNotEqual(code, 0, output)
        self.assertNotIn(SKIPPED_MARKER, output)
        self.assertEqual(len(records), 1)

    def test_timeout_fails_and_continues(self):
        code, output, records = self.run_suite([
            {'name': 'slow', 'mode': 'sleep', 'arguments': ['10'], 'timeout': 0.5},
            {'name': 'passed_after_timeout'}])
        self.assertNotEqual(code, 0, output)
        self.assertNotIn(SKIPPED_MARKER, output)
        self.assertEqual([record['name'] for record in records],
                         ['slow', 'passed_after_timeout'])

    def test_missing_executable_fails_and_continues(self):
        code, output, records = self.run_suite([
            {'name': 'missing', 'missing': True},
            {'name': 'passed_after_missing'}])
        self.assertNotEqual(code, 0, output)
        self.assertNotIn(SKIPPED_MARKER, output)
        self.assertEqual([record['name'] for record in records],
                         ['passed_after_missing'])
        self.assertIn('missing', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cmake', default='cmake')
    options, remaining = parser.parse_known_args()
    CMAKE = options.cmake
    unittest.main(argv=[sys.argv[0], *remaining])
