#!/usr/bin/env python3
"""Reject text that names another code base or justifies a rule by one.

Every tracked (or new, unignored) file is scanned except LICENSE, this script
and the LPFx copy in include/lpfx/ and cmake/lpfx/. An allow list names, per
file pattern, the only contexts in which denied words may appear; those
contexts are blanked out of a line before the deny patterns are applied. The
licence and attribution notice that the comparison deck carries is kept
verbatim and is allowed only at its top. Every run first checks that each
pattern still fires.
"""
from fnmatch import fnmatch
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SELF = 'tests/compile/foreign_references.py'
EXCLUDED_FILES = {'LICENSE', SELF}
EXCLUDED_DIRECTORIES = ('include/lpfx/', 'cmake/lpfx/')

DENY = [
    ('code-base name', re.compile(r'(?i)ipic3d|ecsimspace')),
    ('code-base name', re.compile(r'(?i)\btcmm\b')),
    ('legacy', re.compile(r'(?i)legacy')),
    ('provenance', re.compile(r"(?i)\b(?:ported|porting|port of)\b|upstream|\bthe source's\b|"
                              r'historical manifest|frozen oracle')),
    ('provenance', re.compile(r"(?i)\b(?:the|as in the|matching the|match(?:es)? the|preserv\w* the|"
                              r"retain\w* the) reference(?:'s)?\b")),
    ('reference name', re.compile(
        r'(?i)\breference[-_ ](?:implementation|code|parser|factory|startup|initiali[sz]er|backend|'
        r'output|inputs?|corpus|rng|rounding|grid|ecsim|algebra|loading|smoother|diagnostics|fields?|'
        r'executable|parity|inventory|manifest|source|compatible|comparable|gate|apply|smooth|current|'
        r'mass|values|slots|validation)\b')),
    ('reference name', re.compile(r'TPIC_(?:TEST_)?REFERENCE|tpic_reference|tpic_test_reference_|'
                                  r'reference_(?:fields|inputs|output|parity|inventory|magnetic_startup)')),
    ('reference', re.compile(r'(?i)\breference\b')),
    ('parity', re.compile(r'(?i)\bparity\b')),
    ('compatibility', re.compile(
        r'(?i)compatibility (?:adapter|overload|entry point|boundary|spelling|metadata|placeholder|setup|'
        r'path|fixture|oracle)|source-compatibility|reference-compatible|parsing compatibility|'
        r'for compatibility')),
    ('licence holder', re.compile(r'KTH Royal Institute')),
    ('age', re.compile(r'(?i)\bold (?:kernel|implementation|code|parser)|original (?:kernel|implementation|'
                       r'parser)|unchanged (?:reference|legacy|source|ECSIM|method|oracle|parser)')),
]

# (file pattern, context) pairs: the context is blanked before the deny patterns run.
ALLOWED = [
    ('*', re.compile(r'(?i)numerical[-_ ]reference(?:\.md)?|input key reference|\[key reference\]')),
    ('*', re.compile(r'\S*(?:/reference/|api-reference)\S*')),
    ('*', re.compile(r'(?i)references, pointers and views|\ba reference to\b')),
    ('tools/postprocessing/analyze_whistler.py', re.compile(r'(?i)omega theory reference')),
    ('*', re.compile(r'(?i)\b(?:magnetic|electric|ghost) parity\b')),
    ('*', re.compile(r'\b(?:cuda|hip)StreamLegacy\b')),
    # The GEM comparison deck keeps its title, file name and output directory; the golden deck tables list the
    # latter two.
    ('input/reconnection_comparison/GEM3D_ipic3d_gpu.inp', re.compile(r'iPIC3D-GPU|data_ipic3d_gpu_gemharris')),
    ('tests/data/golden/*', re.compile(r'GEM3D_ipic3d_gpu\.inp|data_ipic3d_gpu_gemharris')),
]

# The licence and attribution notice carried by the comparison deck stays verbatim.
NOTICE_FILES = ('input/reconnection_comparison/GEM3D_ipic3d_gpu.inp',)
NOTICE_LINES = {
    '# Publications that use results from iPIC3D need to properly cite',
    "# 'S. Markidis, G. Lapenta, and Rizwan-uddin. \"Multi-scale simulations of",
    "# plasma with iPIC3D.\" Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'",
    '# Copyright 2015 KTH Royal Institute of Technology',
}
NOTICE_LENGTH = 16


def scanned_files():
    try:
        listing = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--cached', '--others',
                                  '--exclude-standard', '-z'], check=True, capture_output=True).stdout
        paths = sorted({name for name in listing.decode().split('\0') if name})
    except (OSError, subprocess.CalledProcessError):
        skipped = {'.git', 'build', '__pycache__'}
        paths = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
                       if p.is_file() and not skipped.intersection(p.relative_to(ROOT).parts))
    return [p for p in paths if p not in EXCLUDED_FILES and not p.startswith(EXCLUDED_DIRECTORIES)
            and (ROOT / p).is_file()]


def is_notice(path, number, line):
    return (number <= NOTICE_LENGTH and any(fnmatch(path, pattern) for pattern in NOTICE_FILES)
            and ' '.join(line.split()) in NOTICE_LINES)


def violations(path, text):
    contexts = [context for pattern, context in ALLOWED if fnmatch(path, pattern)]
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if is_notice(path, number, line):
            continue
        for context in contexts:
            line = context.sub(lambda match: ' ' * len(match.group(0)), line)
        for label, pattern in DENY:
            match = pattern.search(line)
            if match:
                found.append(f'{path}:{number}: {label}: {match.group(0)!r}')
    return found


def self_test():
    seeded = ['iPIC3D-new', 'ecsimspace', 'TCMM', 'legacy_input', 'Ported from', "the source's",
              'frozen oracle', "the reference's order", 'reference_current', 'TPIC_TEST_REFERENCE',
              'a reference value', 'parser parity', 'compatibility spelling',
              'KTH Royal Institute', 'the old kernel']
    accepted = ['remove_reference_t', 'dereference', 'see numerical_reference.md', 'magnetic parity',
                'returns a reference to the slot', 'CUDA-compatible', 'mass halo port']
    failures = [f'seeded violation not caught: {text!r}' for text, pattern in zip(seeded, [
        0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11]) if not DENY[pattern][1].search(text)]
    failures += [f'allowed text flagged: {text!r}' for text in accepted if violations('x.md', text)]
    notice = '# Publications that use results from iPIC3D need to properly cite'
    if (violations('input/reconnection_comparison/GEM3D_ipic3d_gpu.inp', notice)
            or not violations('performance/scaling/mi250x/weak/GEMweakN4.inp', notice)):
        failures.append('licence notice allow-list is not limited to the comparison deck')
    return failures


def main():
    failures = self_test()
    files = scanned_files()
    for path in files:
        try:
            text = (ROOT / path).read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        failures += violations(path, text)
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'No foreign references in {len(files)} files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
