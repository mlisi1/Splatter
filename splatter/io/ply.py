"""PLY export for COLMAP point clouds."""

import struct
from pathlib import Path


def write_ply(points: dict, path: Path) -> None:
    """
    Write a COLMAP points3D dict to a binary little-endian PLY file with XYZ + RGB.

    Compatible with CloudCompare, MeshLab, Open3D, and any other PLY viewer.
    Points with no colour (all zeros) are written as-is.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode()

    with open(path, "wb") as f:
        f.write(header)
        for pt in points.values():
            f.write(struct.pack("<fff", *pt.xyz))
            f.write(bytes(int(c) for c in pt.rgb))
