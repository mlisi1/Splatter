"""Stage 3 — Structure-from-Motion via COLMAP (driven by hloc)."""

import logging
import subprocess
from pathlib import Path

import numpy as np

from splatter.io.colmap import read_model

logger = logging.getLogger("gs_init")

_FALLBACK_ORDER = {
    "global": ["global", "hierarchical", "incremental"],
    "hierarchical": ["hierarchical", "incremental"],
    "incremental": ["incremental"],
}


def run_sfm(
    output_dir: Path,
    sfm_type: str,
    feature_path: Path,
    match_path: Path,
    pairs_path: Path,
) -> None:
    """
    Import hloc features into COLMAP, run the mapper, and export a text model.

    Falls back through global → hierarchical → incremental if an earlier
    variant fails.
    """
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    sparse_0 = sparse_dir / "0"
    images_dir = output_dir / "images"
    db_path = output_dir / "database.db"

    hloc_ok = False
    try:
        import pycolmap
        from hloc import reconstruction as hloc_recon
        logger.info("Running hloc reconstruction (feature import + COLMAP mapper)")
        hloc_recon.main(
            sfm_dir=sparse_dir,
            image_dir=images_dir,
            pairs=pairs_path,
            features=feature_path,
            matches=match_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
        )
        # hloc writes the largest model directly into sfm_dir (not sfm_dir/0/).
        # Detect this layout and normalise to sparse/0/.
        if (sparse_dir / "cameras.bin").exists() and not sparse_0.exists():
            logger.info("Moving hloc model from sparse/ to sparse/0/")
            sparse_0.mkdir(parents=True, exist_ok=True)
            for f in ("cameras.bin", "images.bin", "points3D.bin"):
                src = sparse_dir / f
                if src.exists():
                    src.rename(sparse_0 / f)
        hloc_ok = (sparse_0 / "cameras.bin").exists()
        # hloc writes its database inside sparse/; copy it to the canonical
        # output location so the incremental-retry fallback can find it.
        hloc_db = sparse_dir / "database.db"
        if hloc_db.exists():
            import shutil as _sh
            _sh.copy2(hloc_db, db_path)
            logger.info(f"Copied hloc database → {db_path}")
    except Exception as e:
        logger.warning(f"hloc reconstruction raised: {e} — falling back to manual COLMAP")

    if not hloc_ok:
        logger.warning("sparse/0 missing after hloc — running COLMAP directly")
        _run_mapper_with_fallback(sfm_type, db_path, images_dir, sparse_dir)


def check_quality(sparse_0: Path, total_images: int) -> dict:
    """
    Read the reconstructed model and log quality metrics.

    Returns a dict with keys: n_registered, n_points, reg_rate, mean_reproj.
    """
    cameras, images, points3d = read_model(sparse_0)
    n_reg = len(images)
    n_pts = len(points3d)
    reg_rate = n_reg / max(total_images, 1)
    errors = [p.error for p in points3d.values() if p.error > 0]
    mean_reproj = float(np.mean(errors)) if errors else 0.0

    logger.info(f"Registered: {n_reg}/{total_images} ({reg_rate:.1%})")
    logger.info(f"Sparse points: {n_pts:,}")
    logger.info(f"Mean reprojection error: {mean_reproj:.2f} px")

    if reg_rate < 0.9:
        logger.warning(f"Low registration rate: {reg_rate:.1%} < 90 %")
    if mean_reproj > 1.5:
        logger.warning(f"High reprojection error: {mean_reproj:.2f} px > 1.5 px")
    if n_pts < 1000:
        logger.warning(f"Only {n_pts} sparse points")

    return dict(n_registered=n_reg, n_points=n_pts,
                reg_rate=reg_rate, mean_reproj=mean_reproj)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_mapper_with_fallback(sfm_type: str, db_path: Path,
                               images_dir: Path, sparse_dir: Path) -> None:
    for mtype in _FALLBACK_ORDER[sfm_type]:
        logger.info(f"Running COLMAP {mtype} mapper")
        if _try_mapper(mtype, db_path, images_dir, sparse_dir):
            logger.info(f"COLMAP {mtype} mapper succeeded")
            return
        logger.warning(f"COLMAP {mtype} mapper failed")
    raise RuntimeError("All COLMAP mapper variants failed")


def _try_mapper(mtype: str, db_path: Path, images_dir: Path, sparse_dir: Path) -> bool:
    if mtype == "hierarchical":
        cmd = ["colmap", "hierarchical_mapper",
               "--database_path", str(db_path),
               "--image_path", str(images_dir),
               "--output_path", str(sparse_dir)]
    else:
        extra = ["--Mapper.mapper_type", "GLOBAL"] if mtype == "global" else []
        cmd = ["colmap", "mapper",
               "--database_path", str(db_path),
               "--image_path", str(images_dir),
               "--output_path", str(sparse_dir)] + extra
    r = subprocess.run(cmd, capture_output=True, timeout=7200)
    if r.returncode != 0:
        stderr = r.stderr.decode(errors="replace").strip()
        if stderr:
            logger.warning(f"COLMAP {mtype} stderr: {stderr[:800]}")
    return r.returncode == 0


def retry_with_incremental(output_dir: Path) -> None:
    """
    Re-run the mapper as incremental on the existing COLMAP database.
    Called automatically when global mapper achieves < 80 % registration.
    Backs up the previous sparse/0/ so it can be restored if the retry fails.
    """
    import shutil

    sparse_dir = output_dir / "sparse"
    sparse_0 = sparse_dir / "0"
    sparse_0_backup = sparse_dir / "_global_backup"
    db_path = output_dir / "database.db"
    images_dir = output_dir / "images"

    # Backup previous (global) result before touching anything
    if sparse_0.exists():
        if sparse_0_backup.exists():
            shutil.rmtree(sparse_0_backup)
        shutil.copytree(sparse_0, sparse_0_backup)
        shutil.rmtree(sparse_0)

    # Remove any other numbered sub-models left by the previous run so
    # COLMAP can write a clean set starting from 0/
    for d in sparse_dir.glob("[0-9]*/"):
        shutil.rmtree(d, ignore_errors=True)

    # Prefer the canonical database; fall back to the one hloc left in sparse/
    actual_db = db_path
    if not actual_db.exists() or actual_db.stat().st_size < 1_000_000:
        hloc_db = sparse_dir / "database.db"
        if hloc_db.exists():
            actual_db = hloc_db
            logger.info(f"Using hloc database at {actual_db} for retry")

    logger.info("Retrying with COLMAP incremental mapper (existing feature database reused)")
    ok = _try_mapper("incremental", actual_db, images_dir, sparse_dir)

    if not ok:
        logger.warning("Incremental mapper retry failed — restoring global result")
        if sparse_0_backup.exists():
            sparse_0_backup.rename(sparse_0)
        return

    # Normalise: COLMAP writes sparse/0/, 1/, … — pick the largest
    sub_models = sorted(sparse_dir.glob("[0-9]*/cameras.bin"))
    if not sub_models:
        logger.warning("Incremental mapper produced no models — restoring global result")
        if sparse_0_backup.exists():
            sparse_0_backup.rename(sparse_0)
        return

    best = max(sub_models, key=lambda p: p.stat().st_size)
    best_dir = best.parent
    if best_dir != sparse_0:
        if sparse_0.exists():
            shutil.rmtree(sparse_0)
        best_dir.rename(sparse_0)

    # Clean up remaining sub-models and backup
    for d in sparse_dir.glob("[0-9]*/"):
        if d != sparse_0:
            shutil.rmtree(d, ignore_errors=True)
    if sparse_0_backup.exists():
        shutil.rmtree(sparse_0_backup)


def export_text_model(output_dir: Path, sparse_0: Path) -> None:
    txt_dir = output_dir / "sparse_txt" / "0"
    txt_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["colmap", "model_converter",
         "--input_path", str(sparse_0),
         "--output_path", str(txt_dir),
         "--output_type", "TXT"],
        check=True, capture_output=True,
    )
