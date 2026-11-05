#!/usr/bin/env python3
"""Create a ParaView XDMF companion for published TPIC HDF5 field output.

Pass an output directory to include its published fields_t*.hdf5.json steps,
or pass one such manifest. Only metadata is read; HDF5 payloads are untouched.
Open the resulting .xdmf with ParaView's Xdmf3 Reader S.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def text_attribute(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def geometry_attribute(attributes, name: str, *, positive: bool = False) -> list[float]:
    values = [float(value) for value in attributes[name]]
    if len(values) != 3 or any(not math.isfinite(value) or (positive and value <= 0)
                               for value in values):
        raise ValueError(f"{name} must contain three {'positive ' if positive else ''}finite values")
    return values


def append_piece(parent, source: Path, piece: dict, manifest: dict, output: Path, h5py) -> None:
    """Read only attributes, dataset shapes and dtypes; add one structured rank."""
    with h5py.File(source, "r") as handle:
        attributes = handle.attrs
        expected = {"schema": "tpic-output-v1", "channel": "fields", "field_order": "z,y,x,component"}
        for name, value in expected.items():
            if text_attribute(attributes[name]) != value:
                raise ValueError(f"{source}: expected {name}={value}")
        for name, value in (("cycle", manifest["cycle"]), ("rank", piece["rank"]),
                            ("ranks", len(manifest["pieces"]))):
            if int(attributes[name]) != value:
                raise ValueError(f"{source}: {name} disagrees with its manifest")
        if float(attributes["time"]) != manifest["time"]:
            raise ValueError(f"{source}: time disagrees with its manifest")
        if not attributes["fields_present"] or attributes["particles_present"]:
            raise ValueError(f"{source}: expected a fields-only piece")
        cells = [int(value) for value in attributes["local_cells"]]
        if len(cells) != 3 or any(value < 0 for value in cells):
            raise ValueError(f"{source}: invalid local_cells")
        dimensions = tuple(value + 1 for value in reversed(cells))
        origin = geometry_attribute(attributes, "local_origin")
        spacing = geometry_attribute(attributes, "spacing", positive=True)
        fields = handle["fields"]
        if not isinstance(fields, h5py.Group):
            raise ValueError(f"{source}: no fields found")
        grid = ET.SubElement(parent, "Grid", Name=f"rank_{piece['rank']}", GridType="Uniform")
        ET.SubElement(grid, "Topology", TopologyType="3DCoRectMesh",
                      Dimensions=" ".join(map(str, dimensions)))
        geometry = ET.SubElement(grid, "Geometry", GeometryType="ORIGIN_DXDYDZ")
        # Xdmf3 Reader S uses z,y,x geometry ordering for a 3DCoRectMesh.
        for values in (origin, spacing):
            item = ET.SubElement(geometry, "DataItem", Dimensions="3", NumberType="Float",
                                 Precision="8", Format="XML")
            item.text = " ".join(format(value, ".17g") for value in reversed(values))
        relative = os.path.relpath(source, output.parent)
        if ":" in relative:
            raise ValueError(f"{source}: ':' in a filename is not supported by XDMF HDF references")
        for name in sorted(fields):
            dataset = fields[name]
            if (not isinstance(dataset, h5py.Dataset) or dataset.ndim != 4
                    or dataset.shape[:3] not in (dimensions, tuple(reversed(cells)))
                    or dataset.shape[3] not in (1, 3) or dataset.dtype.kind != "f" or dataset.dtype.itemsize != 8):
                raise ValueError(f"{source}:/fields/{name}: expected float64 [z,y,x,1 or 3]")
            cell = dataset.shape[:3] != dimensions
            attribute = ET.SubElement(grid, "Attribute", Name=name, Center="Cell" if cell else "Node",
                                      AttributeType="Vector" if dataset.shape[3] == 3 else "Scalar")
            item = ET.SubElement(attribute, "DataItem", Dimensions=" ".join(map(str, dataset.shape)),
                                 NumberType="Float", Precision="8", Format="HDF")
            item.text = f"{relative}:/fields/{name}"


def build_document(manifests: list[Path], output: Path, h5py) -> ET.ElementTree:
    steps = []
    for path in manifests:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        for name, value in (("schema", "tpic-output-v1"), ("backend", "hdf5"), ("channel", "fields")):
            if manifest[name] != value:
                raise ValueError(f"{path}: expected {name}={value}")
        if (not manifest["fields_present"] or manifest["particles_present"]
                or not isinstance(manifest["cycle"], int) or not math.isfinite(manifest["time"])):
            raise ValueError(f"{path}: invalid fields manifest")
        pieces = manifest["pieces"]
        if not pieces or sorted(piece["rank"] for piece in pieces) != list(range(len(pieces))):
            raise ValueError(f"{path}: expected one piece for each consecutive rank")
        steps.append((manifest["cycle"], path, manifest))
    steps.sort(key=lambda step: step[0])
    if len({cycle for cycle, _, _ in steps}) != len(steps):
        raise ValueError("duplicate output cycles")
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    series = ET.SubElement(domain, "Grid", Name="TPIC fields", GridType="Collection",
                           CollectionType="Temporal")
    for cycle, path, manifest in steps:
        step = ET.SubElement(series, "Grid", Name=f"cycle_{cycle}", GridType="Collection",
                             CollectionType="Spatial")
        ET.SubElement(step, "Time", Value=format(manifest["time"], ".17g"))
        for piece in sorted(manifest["pieces"], key=lambda piece: piece["rank"]):
            source = (path.parent / piece["file"]).resolve(strict=True)
            if output == source:
                raise ValueError("the XDMF companion must not replace a source HDF5 file")
            append_piece(step, source, piece, manifest, output, h5py)
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="output directory or fields_t*.hdf5.json manifest")
    parser.add_argument("--output", type=Path, help="XDMF path (default: fields.xdmf or manifest sibling)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing XDMF companion")
    args = parser.parse_args(argv)
    try:
        import h5py
    except ImportError:
        parser.exit(1, "error: h5py is required; use a Python environment with h5py installed\n")
    temporary = None
    try:
        source = args.source.resolve(strict=True)
        if source.is_dir():
            manifests = sorted(source.glob("fields_t*.hdf5.json"))
            default_output = source / "fields.xdmf"
        elif source.name.endswith(".hdf5.json"):
            manifests = [source]
            default_output = source.with_name(source.name.removesuffix(".hdf5.json") + ".xdmf")
        else:
            raise ValueError("source must be an output directory or a fields .hdf5.json manifest")
        if not manifests:
            raise ValueError(f"{source}: no published fields_t*.hdf5.json manifests found")
        output = (args.output or default_output).resolve()
        if output.suffix not in (".xdmf", ".xmf"):
            raise ValueError("output must have an .xdmf or .xmf extension")
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} already exists; use --overwrite to replace it")
        document = build_document(manifests, output, h5py)
        with tempfile.NamedTemporaryFile(mode="wb", dir=output.parent, prefix=".xdmf-",
                                         suffix=".xml", delete=False) as stream:
            temporary = Path(stream.name)
            document.write(stream, encoding="utf-8", xml_declaration=True)
            stream.write(b"\n")
        if args.overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
        print(f"Created {output} ({len(manifests)} steps); open it with ParaView's Xdmf3 Reader S.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
