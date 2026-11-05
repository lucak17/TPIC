#!/usr/bin/env python3
"""Verify the LPFx copy against the LPFx revision and file hashes recorded in cmake/dependencies.json."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "cmake/dependencies.json"
# The build support adapted to this tree; every other file of copied() is an unchanged LPFx file.
ADAPTED = ("cmake/lpfx/CMakeLists.txt", "cmake/lpfx/LPFxOptions.cmake", "cmake/lpfx/LPFxTargets.cmake")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def revision(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True, stderr=subprocess.PIPE).strip()


def copied():
    """Each unchanged LPFx file of this tree with its path in LPFx."""
    paths = {p.relative_to(ROOT).as_posix(): p.relative_to(ROOT).as_posix()
             for p in (ROOT / "include/lpfx").rglob("*") if p.is_file()}
    for p in (ROOT / "cmake/lpfx").iterdir():
        if p.relative_to(ROOT).as_posix() not in ADAPTED:
            paths[p.relative_to(ROOT).as_posix()] = "cmake/" + p.name
    paths["tests/support/device/mock_backend.hpp"] = "tests/support/device/mock_backend.hpp"
    return dict(sorted(paths.items()))


def committed(root, commit):
    data = subprocess.run(["git", "-C", str(root), "archive", commit], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {member.name: tar.extractfile(member).read() for member in tar.getmembers() if member.isfile()}


def check():
    data = json.loads(CONTRACT.read_text())
    recorded = {entry["path"]: entry["sha256"] for entry in data["files"]}
    present = copied()
    failures = [f"Changed since LPFx {data['revision']}: {path}" for path, sha256 in recorded.items()
                if path in present and digest((ROOT / path).read_bytes()) != sha256]
    failures += [f"Missing: {path}" for path in recorded if path not in present]
    failures += [f"Not recorded: {path}" for path in present if path not in recorded]
    if not failures:
        print(f"The {len(recorded)} copied files match LPFx {data['revision']}.")
    return failures


def refresh(lpfx):
    commit = revision(lpfx)
    files = committed(lpfx, commit)
    present = copied()
    failures = [f"Differs from LPFx {commit}: {path}" for path, source in present.items()
                if files.get(source) != (ROOT / path).read_bytes()]
    failures += [f"Not copied: {source}" for source in files
                 if source.startswith("include/lpfx/") and source not in present.values()]
    if failures:
        return failures
    data = {"repository": "LPFx", "revision": commit,
            "scope": "Files copied unchanged from LPFx at this revision; " + ", ".join(ADAPTED) +
                     " are adapted to this tree",
            "files": [{"path": path, "sha256": digest((ROOT / path).read_bytes())} for path in present]}
    CONTRACT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Recorded {len(present)} files of LPFx {commit}.")
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Verify the LPFx copy against the record")
    mode.add_argument("--refresh", action="store_true",
                      help="Record the revision of the --lpfx checkout once its files are copied into this tree")
    parser.add_argument("--lpfx", type=Path, help="The LPFx checkout that --refresh records")
    args = parser.parse_args()
    if args.refresh and args.lpfx is None:
        parser.error("--refresh needs --lpfx <LPFx checkout>")
    try:
        failures = refresh(args.lpfx) if args.refresh else check()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Inventory unavailable: {error}")
        return 1
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
