"""Undistortion — both the known-calibration (Stage 1.5) and
self-calibration (Stage 3.5) paths."""

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from splatter.io.colmap import read_cameras_binary

logger = logging.getLogger("gs_init")

_UNDISTORTED_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}
_KNOWN_CALIBRATION_MODELS = {"OPENCV": 8, "FULL_OPENCV": 12}


# ---------------------------------------------------------------------------
# Stage 1.5 — known calibration, applied right after extraction
# ---------------------------------------------------------------------------

def read_calibration_file(path: Path) -> tuple[str, int, int, np.ndarray]:
    """
    Parse a known-calibration file:
        line 1: COLMAP camera model name — OPENCV or FULL_OPENCV
        line 2: "<width> <height>" the calibration was measured at
        line 3: space-separated params in COLMAP order for that model
                (fx fy cx cy k1 k2 p1 p2[ k3 k4 k5 k6])

    COLMAP's OPENCV/FULL_OPENCV param order matches OpenCV's own
    cameraMatrix/distCoeffs convention exactly, so no reordering is needed.
    """
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"Calibration file {path} needs 3 lines: model, 'W H', params")

    model = lines[0].strip().upper()
    if model not in _KNOWN_CALIBRATION_MODELS:
        raise ValueError(
            f"Unsupported calibration model '{model}' — supported: "
            f"{sorted(_KNOWN_CALIBRATION_MODELS)}"
        )

    calib_w, calib_h = (int(x) for x in lines[1].split())

    params = np.array([float(x) for x in lines[2].split()], dtype=np.float64)
    expected = _KNOWN_CALIBRATION_MODELS[model]
    if len(params) != expected:
        raise ValueError(f"{model} expects {expected} params, got {len(params)}")

    return model, calib_w, calib_h, params


def undistort_known_calibration(images_dir: Path, calibration_path: Path, output_dir: Path) -> None:
    """
    Undistort every extracted frame in place using a user-supplied
    calibration, before feature extraction (Stage 2) ever sees them.

    Unlike Stage 3.5 (which undoes distortion hloc/COLMAP had to
    self-calibrate from scratch, and so can only run after SfM converges),
    this runs when the true calibration is already known — so
    distortion-free pixels are available from the very first stage that
    reads pixel content, giving feature matching and SfM themselves more
    accurate input rather than only cleaning things up afterward.

    Stage 3.5 still runs later regardless: COLMAP will self-estimate a
    model with near-zero residual distortion on these already-undistorted
    frames, and image_undistorter's pass over that is a cheap near-no-op.

    Idempotent via a marker file — the frames on disk are already
    undistorted on a second invocation, and re-applying the map would
    double-warp them.
    """
    marker = output_dir / "_calibration_undistorted"
    if marker.exists():
        logger.info("Known-calibration undistortion already applied — skipping")
        return

    model, calib_w, calib_h, params = read_calibration_file(calibration_path)

    frames = sorted(images_dir.glob("*.png"))
    if not frames:
        logger.warning("No frames found for known-calibration undistortion — skipping")
        return

    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]
    sx, sy = w / calib_w, h / calib_h
    if abs(sx - sy) > 1e-3:
        logger.warning(
            f"Frame size ({w}x{h}) aspect ratio doesn't match calibration "
            f"({calib_w}x{calib_h}) — scaling fx/cx and fy/cy independently, "
            "but this usually means the wrong calibration file was supplied"
        )

    # fx, fy, cx, cy scale linearly with resolution; k1, k2, p1, p2[, k3..]
    # are defined in normalized coordinates and are resolution-invariant.
    fx, fy, cx, cy = params[0] * sx, params[1] * sy, params[2] * sx, params[3] * sy
    dist = params[4:]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # alpha=0: crop to the valid pixel region (no black borders) rather than
    # keep every source pixel — matches Stage 3.5's image_undistorter, and
    # avoids feeding meaningless black-border pixels to feature matching.
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)

    logger.info(f"Undistorting {len(frames)} frames with known {model} calibration")
    for f in tqdm(frames, desc="Undistorting (known calibration)"):
        img = cv2.imread(str(f))
        cv2.imwrite(str(f), cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR))

    marker.write_text(
        f"model=PINHOLE fx={new_K[0, 0]:.4f} fy={new_K[1, 1]:.4f} "
        f"cx={new_K[0, 2]:.4f} cy={new_K[1, 2]:.4f}\n"
    )
    logger.info("Known-calibration undistortion complete")


# ---------------------------------------------------------------------------
# Stage 3.5 — self-calibrated undistortion, applied right after SfM
# ---------------------------------------------------------------------------


def undistort_images(output_dir: Path, sparse_0: Path) -> None:
    """
    Replace distorted frames + camera model with COLMAP's undistorted
    equivalent (PINHOLE, zero distortion coefficients), in place.

    hloc/COLMAP recover an OPENCV/RADIAL/SIMPLE_RADIAL model with nonzero
    distortion coefficients, but:
      - Gaussian Splatting rasterizers (3DGS/2DGS/gsplat) assume a
        distortion-free pinhole camera and either reject non-PINHOLE models
        or silently ignore the distortion terms, baking lens distortion into
        the splats (blur/doubling that worsens toward the frame edges).
      - Depth unprojection (splatter/utils/geometry.py get_intrinsics) only
        ever reads fx, fy, cx, cy — it has no distortion-aware ray model —
        so densifying with a distorted camera introduces systematic geometric
        error in the merged point cloud.

    Running this before Stage 4 means densification, the PLY export, and
    gs_dataset/ all operate on already-undistorted images and a PINHOLE
    camera, with no changes needed downstream.

    No-op if the model is already PINHOLE/SIMPLE_PINHOLE (undistortion
    already ran on a previous invocation against this output directory).
    Runs (cheaply, as a near-identity remap) even when
    undistort_known_calibration() ran earlier this run, since COLMAP still
    self-estimates its own model name/params (e.g. SIMPLE_RADIAL with k1≈0)
    on top of already-undistorted pixels — this normalises that to a clean
    PINHOLE model rather than special-casing the calibrated path.
    """
    cameras = read_cameras_binary(sparse_0 / "cameras.bin")
    if all(c.model in _UNDISTORTED_MODELS for c in cameras.values()):
        logger.info("Camera model already undistorted (PINHOLE) — skipping")
        return

    models = sorted({c.model for c in cameras.values()})
    logger.info(f"Undistorting images and camera model (COLMAP image_undistorter); source model(s): {models}")

    images_dir = output_dir / "images"
    tmp_dir = output_dir / "_undistort_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    subprocess.run(
        ["colmap", "image_undistorter",
         "--image_path", str(images_dir),
         "--input_path", str(sparse_0),
         "--output_path", str(tmp_dir),
         "--output_type", "COLMAP"],
        check=True, capture_output=True,
    )

    tmp_sparse = tmp_dir / "sparse"
    tmp_images = tmp_dir / "images"
    if not (tmp_sparse / "cameras.bin").exists() or not tmp_images.exists():
        raise RuntimeError("colmap image_undistorter did not produce the expected output")

    # Keep the distorted originals rather than deleting them. Leading
    # underscore on the sparse backup avoids the sparse/[0-9]*/ glob that
    # sfm.retry_with_incremental() uses to clear numbered sub-models.
    dist_images_backup = output_dir / "images_distorted"
    dist_sparse_backup = sparse_0.parent / "_distorted_backup"

    if dist_images_backup.exists():
        shutil.rmtree(dist_images_backup)
    images_dir.rename(dist_images_backup)
    tmp_images.rename(images_dir)

    if dist_sparse_backup.exists():
        shutil.rmtree(dist_sparse_backup)
    sparse_0.rename(dist_sparse_backup)
    sparse_0.mkdir(parents=True, exist_ok=True)
    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        (tmp_sparse / fname).rename(sparse_0 / fname)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(
        "Undistortion complete — distorted originals kept at "
        f"{dist_images_backup} and {dist_sparse_backup}"
    )
