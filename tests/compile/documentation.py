#!/usr/bin/env python3
"""Check the maintained documentation.

Every local link of the maintained documents must resolve, and the input-key
guide must list each key the input parser reads exactly once, and no other.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
files = [ROOT / 'README.md', ROOT / 'AGENTS.md', ROOT / 'docs/README.md', ROOT / 'performance/scaling/README.md']
for folder in ('docs/user', 'docs/developer'):
    files.extend(sorted((ROOT / folder).glob('*.md')))
failures = []
for path in files:
    text = re.sub(r'```.*?```|`[^`\n]*`', '', path.read_text(), flags=re.DOTALL)
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if '://' in target or target.startswith('#'):
            continue
        destination = path.parent / target.split('#')[0]
        if not destination.exists():
            failures.append(f'{path.relative_to(ROOT)}: missing {target}')

source = (ROOT / 'src/config/simulation_parameters.cpp').read_text()
keys = set(re.findall(r'cfg\.read_(?:required|optional|list|into|output_method)(?:<[^>]+>)?\s*\(\s*"([^"]+)"', source))
guide = ROOT / 'docs/user/input_keys.md'
documented = re.findall(r'^\| `([^`]+)` \|', guide.read_text(), re.MULTILINE)
for label, difference in (('missing', keys - set(documented)), ('unknown', set(documented) - keys)):
    if difference:
        failures.append(f"{guide.relative_to(ROOT)}: {label} keys {', '.join(sorted(difference))}")
if len(documented) != len(set(documented)):
    failures.append(f'{guide.relative_to(ROOT)}: a key is listed twice')

if failures:
    print('\n'.join(failures))
    sys.exit(1)
print(f'Checked local links in {len(files)} documents and {len(keys)} input keys.')
