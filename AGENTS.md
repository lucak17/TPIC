# Agent instructions

- Follow [standards](docs/developer/standards.md) for every change and [architecture](docs/developer/architecture.md)
  for layers and ownership.
- Run the checks before handing a change over: `ctest --preset host`, and the device preset
  (`ctest --preset cuda` or `hip`) when device code changes.
- Never edit the LPFx copy (`include/lpfx`, `cmake/lpfx` except the three build files `cmake/dependencies.json`
  names as adapted, and `tests/support/device/mock_backend.hpp`) by hand: copy an LPFx revision in and record it with
  `python3 tools/dependency_inventory.py --refresh --lpfx <checkout>`.
