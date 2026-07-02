"""COLMAP binary/text format: data structures and readers/writers."""

import struct
from pathlib import Path

import numpy as np


# ---- Camera models --------------------------------------------------------

CAMERA_MODEL_IDS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


# ---- Data structures -------------------------------------------------------

class Camera:
    __slots__ = ("id", "model", "width", "height", "params")

    def __init__(self, id: int, model: str, width: int, height: int, params: np.ndarray):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params


class Image:
    __slots__ = ("id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids")

    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = np.asarray(qvec, dtype=np.float64)
        self.tvec = np.asarray(tvec, dtype=np.float64)
        self.camera_id = camera_id
        self.name = name
        self.xys = np.asarray(xys, dtype=np.float64).reshape(-1, 2)
        self.point3D_ids = np.asarray(point3D_ids, dtype=np.int64)


class Point3D:
    __slots__ = ("id", "xyz", "rgb", "error", "image_ids", "point2D_idxs")

    def __init__(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.id = id
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.rgb = np.asarray(rgb, dtype=np.uint8)
        self.error = float(error)
        self.image_ids = np.asarray(image_ids, dtype=np.uint32)
        self.point2D_idxs = np.asarray(point2D_idxs, dtype=np.uint32)


ColmapModel = tuple[dict, dict, dict]  # cameras, images, points3D


# ---- Readers ---------------------------------------------------------------

def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model_name, n_params = CAMERA_MODEL_IDS[model_id]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = Camera(cam_id, model_name, width, height, np.array(params))
    return cameras


def read_images_binary(path: Path) -> dict[int, Image]:
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            xys_flat = struct.unpack(f"<{2 * n_pts2d}d", f.read(16 * n_pts2d))
            pid_flat = struct.unpack(f"<{n_pts2d}q", f.read(8 * n_pts2d))
            images[img_id] = Image(img_id, qvec, tvec, camera_id, name.decode(),
                                   np.array(xys_flat).reshape(-1, 2), np.array(pid_flat))
    return images


def read_points3d_binary(path: Path) -> dict[int, Point3D]:
    points3d = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track = struct.unpack(f"<{2 * track_len}I", f.read(8 * track_len))
            points3d[pid] = Point3D(pid, xyz, rgb, error,
                                    np.array(track[0::2], dtype=np.uint32),
                                    np.array(track[1::2], dtype=np.uint32))
    return points3d


def read_model(sparse_dir: Path) -> ColmapModel:
    """Read cameras, images, and points3D from a COLMAP sparse directory."""
    return (
        read_cameras_binary(sparse_dir / "cameras.bin"),
        read_images_binary(sparse_dir / "images.bin"),
        read_points3d_binary(sparse_dir / "points3D.bin"),
    )


# ---- Writers ---------------------------------------------------------------

def write_points3d_binary(points3d: dict[int, Point3D], path: Path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points3d)))
        for p in points3d.values():
            f.write(struct.pack("<Q", p.id))
            f.write(struct.pack("<3d", *p.xyz))
            f.write(struct.pack("<3B", *p.rgb))
            f.write(struct.pack("<d", p.error))
            tl = len(p.image_ids)
            f.write(struct.pack("<Q", tl))
            for img_id, pt2d in zip(p.image_ids, p.point2D_idxs):
                f.write(struct.pack("<II", int(img_id), int(pt2d)))


def write_points3d_text(points3d: dict[int, Point3D], path: Path):
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3d)}, mean track length: 0\n")
        for p in points3d.values():
            track = " ".join(f"{i} {j}" for i, j in zip(p.image_ids, p.point2D_idxs))
            f.write(f"{p.id} {p.xyz[0]:.6f} {p.xyz[1]:.6f} {p.xyz[2]:.6f} "
                    f"{p.rgb[0]} {p.rgb[1]} {p.rgb[2]} {p.error:.6f} {track}\n")
