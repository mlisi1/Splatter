"""Stage 3.7 — Up-axis alignment via COLMAP's Manhattan-world orientation aligner."""

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("gs_init")


def align_up_axis(output_dir: Path, sparse_0: Path) -> bool:
    """
    Rotate sparse_0 so the scene's true vertical direction — estimated from
    vanishing-point/line detection on the undistorted frames — aligns with
    COLMAP's own coordinate convention (world +Y = image-down, so world -Y
    ends up "up"). Monocular SfM has no constraint that forces this: the
    reconstruction's rotation is whatever the bundle adjustment happened to
    converge to. It lands close to -Y-up only by coincidence for a roughly
    level handheld/vehicle/drone capture, with the residual "skew" left
    uncorrected — there's nothing else in this pipeline that fixes world
    orientation unless telemetry georegistration (Stage 3.6) succeeded.

    Pure rotation, no rescaling — verified empirically against COLMAP 3.9.1
    on a synthetic model: pairwise point/camera distances are preserved to
    float64 precision after alignment. That makes it safe to run even when
    --lidar is supplied, unlike telemetry's Sim3 georegistration: it doesn't
    touch T_cam_to_lidar's calibrated scale, since a uniform rotation of the
    whole world frame doesn't change any camera's *relative* transform to
    the LiDAR rig (T_cam_to_lidar is a fixed physical measurement in
    camera-local space).

    Skipped by the caller whenever Stage 3.6 telemetry already succeeded —
    ENU georegistration is grounded in real GPS, so a heuristic Manhattan-
    world re-alignment on top of it could only make a correct orientation
    worse, not better.

    Requires --method MANHATTAN-WORLD's line detection to find enough
    orthogonal structure in the frames (building edges, floor/wall
    boundaries, etc.) — best-effort like telemetry: any failure (COLMAP
    error, no usable vanishing points) logs a warning and leaves sparse_0
    untouched rather than aborting the pipeline. Scenes with little
    man-made structure (open fields, forests, water) may not have enough
    signal for this to succeed.
    """
    images_dir = output_dir / "images"
    tmp_dir = output_dir / "_orientation_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    r = subprocess.run(
        ["colmap", "model_orientation_aligner",
         "--image_path", str(images_dir),
         "--input_path", str(sparse_0),
         "--output_path", str(tmp_dir),
         "--method", "MANHATTAN-WORLD"],
        capture_output=True,
    )
    if r.returncode != 0 or not (tmp_dir / "cameras.bin").exists():
        stderr = r.stderr.decode(errors="replace").strip()
        logger.warning(f"colmap model_orientation_aligner failed — skipping up-axis alignment. {stderr[:500]}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    backup_dir = sparse_0.parent / "_pre_orientation_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(sparse_0, backup_dir)

    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        (tmp_dir / fname).replace(sparse_0 / fname)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "Up-axis alignment applied — world -Y should now match true up "
        f"(pre-alignment backup at {backup_dir.name})"
    )
    return True
