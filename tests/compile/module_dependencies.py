#!/usr/bin/env python3
"""Check the dependency rules of TPIC on the transitive include graph of
include/tpic/ and src/: the ordered layer table, the phase order inside
simulation/, the runtime isolation, the Simulation name and the environment policy."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)

# The ordered layer table: (layer, directory, what it may include besides its
# own tree). A directory entry names an earlier row; a file entry is a single
# header of the first layer or of a lower layer. src/app sits on
# top and includes only headers.
LAYERS = (
    (1, 'tpic/device/', ()),
    (1, 'tpic/data/', ('tpic/device/', 'tpic/config/precision.hpp', 'tpic/config/limits.hpp',
                       'tpic/config/domain.hpp')),
    (1, 'tpic/config/', ()),
    (2, 'tpic/communication/', ('tpic/config/', 'tpic/device/', 'tpic/data/')),
    (3, 'tpic/ecsim/', ('tpic/config/', 'tpic/device/', 'tpic/data/', 'tpic/communication/')),
    (4, 'tpic/initialization/', ('tpic/config/', 'tpic/device/', 'tpic/data/')),
    (4, 'tpic/diagnostics/', ('tpic/config/', 'tpic/device/', 'tpic/data/', 'tpic/ecsim/particles/cic.hpp',
                              'tpic/ecsim/fields/cell_to_node.hpp', 'tpic/ecsim/fields/field_boundary.hpp',
                              'tpic/ecsim/fields/field_boundary_device.hpp', 'tpic/ecsim/fields/field_checks.hpp',
                              'tpic/ecsim/fields/node_cell_curl.hpp', 'tpic/ecsim/fields/node_smoothing.hpp',
                              'tpic/ecsim/fields/node_smoothing_device.hpp', 'tpic/ecsim/fields/stencil_region.hpp',
                              'tpic/ecsim/fields/wall_faces.hpp', 'tpic/ecsim/deposition/moment_field_device.hpp',
                              'tpic/communication/fields/field_exchange.hpp')),
    (4, 'tpic/io/', ('tpic/config/', 'tpic/data/')),
    (5, 'tpic/simulation/', 'every lower layer'),
    (6, 'app', ('tpic/simulation/', 'tpic/config/')),
)
# Inside the first layer, config/ reaches data/ only from these two headers.
FIRST_LAYER_FILES = {
    'tpic/config/output.hpp': ('tpic/data/output/output_moment_quantities.hpp',
                               'tpic/data/output/output_snapshot.hpp', 'tpic/data/output/checkpoint.hpp'),
    'tpic/config/output_selection.hpp': ('tpic/data/output/output_moment_quantities.hpp',
                                         'tpic/data/output/output_snapshot.hpp',
                                         'tpic/data/output/checkpoint.hpp'),
}

# Inside simulation/ a directory includes only itself and lower directories.
PHASES = ('planning', 'execution', 'timing', 'state', 'transport', 'actions', 'reporting', 'driver')

# The runtime stays host-only: walking from it reaches no device header, no MPI
# and no io header.
PIPELINE = 'tpic/simulation/driver/pipeline.hpp'
DRIVER = 'tpic/simulation/driver/'
DRIVER_HEADER = 'tpic/simulation/driver/simulation_device.hpp'
NAMES_SIMULATION = re.compile(r'(class|typename)\s+Simulation(?:\s|[,>])|\bSimulation\s*[&*]')
ENVIRONMENT_FREE = ('tpic/data/', 'tpic/ecsim/')

def layer_rows():
    rows = {}
    for layer, directory, allowed in LAYERS:
        if allowed == 'every lower layer':
            allowed = tuple(d for l, d, _ in LAYERS if l < layer and d.startswith('tpic/'))
        rows[directory] = (layer, allowed)
    return rows


def layer_of(name, rows):
    return next((d for d in rows if d.startswith('tpic/') and name.startswith(d)), None)


def phase_of(name):
    parts = name.split('/')
    return PHASES.index(parts[2]) if len(parts) > 3 and parts[2] in PHASES else None


def walk(graph, start):
    """Yield each reachable header with its first include route."""
    stack, seen = [(start, [start])], set()
    while stack:
        node, route = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        yield node, route
        stack.extend((child, route + [child]) for child in graph.get(node, []))


def check_table(rows, headers, failures):
    """The table is ordered and every rule names something that exists."""
    order = list(rows)
    for directory, (layer, allowed) in rows.items():
        if directory.startswith('tpic/') and not any(h.startswith(directory) for h in headers):
            failures.append(f'Stale rule: no header in {directory}')
        for entry in allowed:
            if entry.endswith('/'):
                if entry not in rows or order.index(entry) >= order.index(directory):
                    failures.append(f'Layer table out of order: {directory} may include {entry}')
            elif entry not in headers:
                failures.append(f'Stale rule: {entry} does not exist')
    for path, entries in FIRST_LAYER_FILES.items():
        for entry in (path,) + entries:
            if entry not in headers:
                failures.append(f'Stale rule: {entry} does not exist')
    for phase in PHASES:
        if not any(h.startswith(f'tpic/simulation/{phase}/') for h in headers):
            failures.append(f'Stale rule: no header in tpic/simulation/{phase}/')
    for path in (DRIVER_HEADER, PIPELINE):
        if path not in headers:
            failures.append(f'Stale rule: {path} does not exist')


def check_repository(root):
    rows = layer_rows()
    include_root = root / 'include'
    headers = {str(p.relative_to(include_root)): p for p in (include_root / 'tpic').rglob('*.hpp')}
    contents = {name: path.read_text() for name, path in headers.items()}
    graph = {name: [h for h in INCLUDE.findall(text) if h.startswith('tpic/')]
             for name, text in contents.items()}
    failures = []
    check_table(rows, headers, failures)
    for name in headers:
        if layer_of(name, rows) is None:
            failures.append(f'{name}: header outside the layer table')
        if name.startswith('tpic/simulation/') and phase_of(name) is None:
            failures.append(f'{name}: simulation header outside the phase directories')
    for name, dependencies in graph.items():
        for dependency in dependencies:
            if dependency not in headers:
                failures.append(f'{name}: unresolved include {dependency}')
    if failures:
        return failures, len(headers)

    for start in graph:
        layer = layer_of(start, rows)
        allowed = (layer,) + rows[layer][1] + FIRST_LAYER_FILES.get(start, ())
        for node, route in walk(graph, start):
            if not node.startswith(allowed):
                failures.append('Layer violation: ' + ' -> '.join(route))
                break
        if start.startswith('tpic/simulation/'):
            for node, route in walk(graph, start):
                phase = phase_of(node)
                if phase is not None and phase > phase_of(start):
                    failures.append('Phase order violation: ' + ' -> '.join(route))
                    break

    for node, route in walk(graph, PIPELINE):
        if node.endswith('_device.hpp') or node.startswith('tpic/io/') or \
                re.search(r'^\s*#\s*include\s*<mpi\.h>', contents[node], re.M):
            failures.append('The runtime reaches device, MPI or io code: ' + ' -> '.join(route))
            break
    for name, text in contents.items():
        if name.startswith(ENVIRONMENT_FREE) and re.search(r'getenv\s*\(', text):
            failures.append(f'State/numerical code owns environment policy: {name}')
        if not name.startswith(DRIVER) and (DRIVER_HEADER in graph[name] or NAMES_SIMULATION.search(text)):
            failures.append(f'Header outside driver/ depends on the enclosing simulation: {name}')

    for path in sorted((root / 'src').rglob('*.cpp')):
        module = path.relative_to(root / 'src').parts[0]
        row = rows.get('app' if module == 'app' else f'tpic/{module}/')
        for dependency in INCLUDE.findall(path.read_text()):
            if not dependency.startswith('tpic/'):
                continue
            if dependency not in headers:
                failures.append(f'{path.relative_to(root)}: unresolved include {dependency}')
            elif row is None or not (dependency.startswith(f'tpic/{module}/') or
                                     dependency.startswith(row[1])):
                failures.append(f'Module boundary violation: {path.relative_to(root)} includes {dependency}')

    return failures, len(headers)


def main():
    failures, count = check_repository(ROOT)
    if failures:
        print('\n'.join(failures))
        return 1
    print(f'Checked {count} headers against the layer table, the simulation phase order and the '
          'ownership rules.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
