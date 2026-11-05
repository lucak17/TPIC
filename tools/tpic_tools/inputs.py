"""Read key = value input decks for analysis without interpreting physics."""

from pathlib import Path


def read_input(path: Path) -> dict[str, list[str]]:
    values = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and value.split():
            values[key.strip()] = value.split()
    return values


def periodic_axes(values: dict[str, list[str]]) -> tuple[bool, bool, bool]:
    """Return whether the fields are periodic along x, y and z.

    TPIC reads PERIODICX, PERIODICY and PERIODICZ as 1 or true (periodic),
    0 or false (walls), and as periodic when absent.
    """
    return tuple(values.get(f"PERIODIC{axis}", ["true"])[0] in ("1", "true")
                 for axis in "XYZ")
