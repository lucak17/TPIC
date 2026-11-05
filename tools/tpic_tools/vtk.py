"""Decode TPIC's uncompressed, raw-appended Float64 VTK ImageData field files.

A file holds one piece whose extent is the whole image. No NumPy or plotting
dependencies. The header reader is bounded and can be used
by streaming comparison tools without loading the numerical payload.
"""

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
import sys
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extent(text):
    result = tuple(map(int, text.split()))
    require(len(result) == 6 and all(result[i] <= result[i + 1] for i in (0, 2, 4)),
            f"invalid VTK extent: {text}")
    return result


def sample_count(box, cell=False):
    return math.prod(box[i + 1] - box[i] + (not cell) for i in (0, 2, 4))


def metadata(image):
    fields = image.find("FieldData")
    require(fields is not None, "missing VTK FieldData")
    values = {a.get("Name"): a.text or "" for a in fields}
    try:
        cycle, time = int(values["cycle"]), float(values["TimeValue"])
        final = int(values["final_state"])
        state = bytes(map(int, values["state"].split())).rstrip(b"\0").decode()
        precision = bytes(map(int, values["precision_profile"].split())).rstrip(b"\0").decode()
    except (KeyError, UnicodeError, ValueError) as error:
        raise ValueError("invalid TPIC snapshot metadata") from error
    require(cycle >= 0 and math.isfinite(time) and time >= 0 and final in (0, 1),
            "invalid TPIC snapshot epoch")
    require(state == ("final" if final else "pre_solve"), "inconsistent TPIC snapshot state")
    levels = ("double", "single-position", "single-position-velocity",
              "single-particles-moments-fields-double-mass", "single-particles-moments-fields")
    # The profile name is the pair "<storage>/<compute>".
    require(precision.count("/") == 1 and all(level in levels for level in precision.split("/")),
            "unknown TPIC snapshot precision")
    return cycle, time, bool(final), precision


def descriptors(piece):
    """Components of every point and cell array of a piece, in payload order."""
    result = {}
    for a in piece.iter("DataArray"):
        name = a.get("Name")
        components = int(a.get("NumberOfComponents", "1"))
        require(name and name not in result and a.get("type") == "Float64" and components in (1, 3),
                "invalid or duplicate VTK array descriptor")
        result[name] = components
    require(result, "empty VTK point and cell data")
    return result


def cell_arrays(piece):
    return {a.get("Name") for a in piece.findall("CellData/DataArray")}


@dataclass
class Piece:
    image: ET.Element
    extent: tuple
    components: dict
    values: dict


def image_header(path):
    """Return the root, image, extent, array components and payload start of a one-piece file."""
    path = Path(path)
    header = bytearray()
    with path.open("rb") as stream:
        while not (marker := re.search(rb'<AppendedData\b[^>]*encoding="raw"[^>]*>\s*_', header)):
            block = stream.read(4096)
            require(block and len(header) < 1024 * 1024, f"{path}: missing or oversized VTK header")
            header.extend(block)
    root = ET.fromstring(header[:marker.start()] + b"</VTKFile>")
    require(root.get("type") == "ImageData" and root.get("compressor") is None,
            f"{path}: only uncompressed ImageData is supported")
    require(root.get("header_type") == "UInt64", f"{path}: expected UInt64 headers")
    require(root.get("byte_order") in ("LittleEndian", "BigEndian"), f"{path}: unsupported byte order")
    image = root.find("ImageData")
    require(image is not None and len(image.findall("Piece")) == 1, f"{path}: expected one VTK piece")
    piece = image.find("Piece")
    require(piece.get("Extent") and image.get("WholeExtent"), f"{path}: missing VTK extent")
    box = extent(piece.get("Extent"))
    require(box == extent(image.get("WholeExtent")), f"{path}: piece/image extent mismatch")
    return root, image, box, descriptors(piece), marker.end()


def read_piece(path):
    """Read one piece; preserve x-fastest, interleaved component ordering."""
    path = Path(path)
    root, image, box, arrays, payload = image_header(path)
    endian = "<" if root.get("byte_order") == "LittleEndian" else ">"
    values, next_offset = {}, 0
    piece = image.find("Piece")
    cells = cell_arrays(piece)
    with path.open("rb") as stream:
        for descriptor in piece.iter("DataArray"):
            name = descriptor.attrib["Name"]
            offset = int(descriptor.attrib["offset"])
            require(descriptor.get("format") == "appended" and offset == next_offset,
                    f"{path}: noncontiguous or unsupported array payload")
            stream.seek(payload + offset)
            size = stream.read(8)
            require(len(size) == 8, f"{path}: truncated array header")
            length = struct.unpack(endian + "Q", size)[0]
            require(length == sample_count(box, name in cells) * arrays[name] * 8,
                    f"{path}: incorrect array length")
            require(length <= path.stat().st_size - stream.tell(), f"{path}: truncated array")
            data = array("d")
            data.frombytes(stream.read(length))
            if (sys.byteorder == "little") != (endian == "<"):
                data.byteswap()
            require(all(map(math.isfinite, data)), f"{path}: nonfinite array {name}")
            values[name] = data
            next_offset += 8 + length
        require(stream.read().strip() == b"</AppendedData>\n</VTKFile>",
                f"{path}: truncated or trailing VTK payload")
    return Piece(image, box, arrays, values)
