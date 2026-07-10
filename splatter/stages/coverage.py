"""Post-hoc diagnostic: is the registered image set dense enough for good GS training?

Frame count alone doesn't answer this — a scene can have 500 registered images and still
have gaps where consecutive kept frames are too far apart (in translation or rotation)
relative to how close the camera actually is to the scene, leaving that region of the
scene thinly observed. This module measures that directly from the SfM reconstruction:

- Baseline-to-depth ratio between temporally consecutive registered cameras — how far the
  camera moved between two kept frames, relative to the scene depth it's looking at there.
  A large ratio (rule-of-thumb threshold below) means a real coverage hole, whether from a
  too-low --fps, an aggressive --iqa-threshold/--dedup-threshold cut, or SfM simply failing
  to register frames in between.
- Rotation gap between consecutive cameras — catches fast pans/whip-rotations that a pure
  translation check misses.
- Mean SfM track length and the fraction of weakly-triangulated points — a global multi-view
  redundancy signal independent of trajectory order.

All thresholds here are rules of thumb, not values derived from a labeled dataset — treat
the verdict as a diagnostic prompt to go look at report.html's camera trajectory plot and
worst-gap list, not an automated pass/fail gate.
"""

from pathlib import Path

import numpy as np

from splatter.utils.geometry import qvec2rotmat

# Baseline (camera translation) exceeding this fraction of local scene depth flags a gap:
# roughly the point past which consecutive views stop sharing enough surface overlap for
# solid multi-view constraints, for a typical ~60-90 deg FOV.
_TRANSLATION_RATIO_FLAG = 0.4
# Rotation between consecutive kept frames exceeding this many degrees flags a gap
# independently of translation (catches fast pans/whip-rotations).
_ROTATION_DEG_FLAG = 25.0

_MIN_IMAGES = 20
_MIN_POINTS = 500


def load_sfm_points3d(sparse_0: Path, fallback: dict) -> dict:
    """
    Return the original SfM-triangulated points (real per-image tracks), not whatever is
    currently in points3D.bin. After Stage 4 densification, points3D.bin holds the
    depth-unprojected cloud instead — every entry has an empty track (image_ids/
    point2D_idxs), so track-length/observation stats computed from it directly would be
    meaningless. densification.py already preserves the pre-densification cloud at
    points3D_sfm.bin for exactly this reason; read it back if present, else *fallback* is
    already the SfM cloud (--no-densify, or densification hasn't run yet).
    """
    sfm_path = sparse_0 / "points3D_sfm.bin"
    if sfm_path.exists():
        from splatter.io.colmap import read_points3d_binary
        return read_points3d_binary(sfm_path)
    return fallback


def _visible_sfm_points(img, points3d: dict) -> np.ndarray:
    pids = img.point3D_ids[img.point3D_ids >= 0]
    xyz = [points3d[p].xyz for p in pids if p in points3d]
    return np.array(xyz) if xyz else np.empty((0, 3))


def analyze_coverage(images: dict, sfm_points3d: dict) -> dict:
    """
    Returns {} if there are too few registered images (< 20) or SfM points (< 500) to make
    a reliable estimate. Otherwise returns per-gap diagnostics, summary stats, a verdict,
    and a human-readable reasoning string.
    """
    if len(images) < _MIN_IMAGES or len(sfm_points3d) < _MIN_POINTS:
        return {}

    imgs_sorted = sorted(images.values(), key=lambda im: im.name)

    centers, depths = [], []
    for img in imgs_sorted:
        R = qvec2rotmat(img.qvec)
        centers.append(-R.T @ img.tvec)
        pts = _visible_sfm_points(img, sfm_points3d)
        cam_xyz = centers[-1]
        depths.append(float(np.median(np.linalg.norm(pts - cam_xyz, axis=1))) if len(pts) >= 5 else np.nan)
    centers = np.array(centers)
    depths = np.array(depths)

    gaps_m, ratios, rot_degs = [], [], []
    for i in range(len(imgs_sorted) - 1):
        gap = float(np.linalg.norm(centers[i + 1] - centers[i]))
        gaps_m.append(gap)

        local_depth = np.nanmean([depths[i], depths[i + 1]])
        ratios.append(gap / local_depth if local_depth > 1e-6 and not np.isnan(local_depth) else np.nan)

        R_i = qvec2rotmat(imgs_sorted[i].qvec)
        R_j = qvec2rotmat(imgs_sorted[i + 1].qvec)
        R_rel = R_j @ R_i.T
        cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        rot_degs.append(float(np.degrees(np.arccos(cos_angle))))

    gaps_m, ratios, rot_degs = np.array(gaps_m), np.array(ratios), np.array(rot_degs)
    valid_ratio = ~np.isnan(ratios)
    n_valid = int(valid_ratio.sum())

    worst_translation = sorted(
        (
            {"image_a": imgs_sorted[i].name, "image_b": imgs_sorted[i + 1].name,
             "ratio": float(ratios[i]), "gap_m": float(gaps_m[i])}
            for i in range(len(ratios)) if valid_ratio[i]
        ),
        key=lambda d: -d["ratio"],
    )[:5]
    worst_rotation = sorted(
        (
            {"image_a": imgs_sorted[i].name, "image_b": imgs_sorted[i + 1].name,
             "rotation_deg": float(rot_degs[i])}
            for i in range(len(rot_degs))
        ),
        key=lambda d: -d["rotation_deg"],
    )[:5]

    n_translation_flagged = int((ratios[valid_ratio] > _TRANSLATION_RATIO_FLAG).sum())
    n_rotation_flagged = int((rot_degs > _ROTATION_DEG_FLAG).sum())
    frac_translation_flagged = n_translation_flagged / n_valid if n_valid else 0.0
    frac_rotation_flagged = n_rotation_flagged / len(rot_degs) if len(rot_degs) else 0.0

    track_lengths = np.array([len(p.point2D_idxs) for p in sfm_points3d.values() if len(p.point2D_idxs) > 0])
    mean_track_length = float(track_lengths.mean()) if len(track_lengths) else None
    frac_weak_points = float((track_lengths < 3).mean()) if len(track_lengths) else None

    obs_counts = np.array([int((im.point3D_ids >= 0).sum()) for im in imgs_sorted])
    mean_obs = float(obs_counts.mean())
    min_obs = int(obs_counts.min())

    verdict, reasons = _make_verdict(
        frac_translation_flagged, frac_rotation_flagged, mean_track_length,
    )

    return {
        "n_images": len(imgs_sorted),
        "n_gaps": len(ratios),
        "gap_translation_m": gaps_m,
        "baseline_depth_ratio": ratios,
        "gap_rotation_deg": rot_degs,
        "n_translation_flagged": n_translation_flagged,
        "frac_translation_flagged": frac_translation_flagged,
        "translation_ratio_threshold": _TRANSLATION_RATIO_FLAG,
        "worst_translation_gaps": worst_translation,
        "n_rotation_flagged": n_rotation_flagged,
        "frac_rotation_flagged": frac_rotation_flagged,
        "rotation_deg_threshold": _ROTATION_DEG_FLAG,
        "worst_rotation_gaps": worst_rotation,
        "mean_track_length": mean_track_length,
        "frac_weak_points": frac_weak_points,
        "mean_obs_per_image": mean_obs,
        "min_obs_per_image": min_obs,
        "verdict": verdict,
        "reasoning": reasons,
    }


def _make_verdict(frac_t: float, frac_r: float, mean_track_length: float | None) -> tuple[str, str]:
    weak_tracks = mean_track_length is not None and mean_track_length < 2.5
    marginal_tracks = mean_track_length is not None and mean_track_length < 3.5

    reasons = []
    if frac_t > 0:
        reasons.append(f"{frac_t:.0%} of inter-frame gaps exceed the baseline/depth "
                        f"threshold ({_TRANSLATION_RATIO_FLAG})")
    if frac_r > 0:
        reasons.append(f"{frac_r:.0%} of inter-frame gaps exceed the rotation threshold "
                        f"({_ROTATION_DEG_FLAG:.0f}°)")
    if mean_track_length is not None:
        reasons.append(f"mean SfM track length {mean_track_length:.1f} images/point")

    if frac_t > 0.15 or frac_r > 0.15 or weak_tracks:
        verdict = "Insufficient — likely needs more frames or better coverage"
    elif frac_t > 0.05 or frac_r > 0.05 or marginal_tracks:
        verdict = "Marginal — some coverage gaps present"
    else:
        verdict = "Good — coverage looks adequate"

    return verdict, "; ".join(reasons) if reasons else "no gaps detected"


def write_coverage_file(output_dir: Path, coverage: dict) -> None:
    """Write <output_dir>/coverage_report.txt, or skip silently if coverage is empty."""
    if not coverage:
        return
    path = output_dir / "coverage_report.txt"

    lines = [
        "Image coverage diagnostic for GS training",
        "=============================================================",
        "Heuristic only — a diagnostic prompt, not an automated pass/fail gate.",
        "",
        f"Verdict: {coverage['verdict']}",
        f"Reasoning: {coverage['reasoning']}",
        "",
        f"Registered images       : {coverage['n_images']}",
        f"Inter-frame gaps checked : {coverage['n_gaps']}",
        f"Mean SfM track length   : "
        f"{coverage['mean_track_length']:.2f}" if coverage['mean_track_length'] is not None else "N/A",
        f"Weakly-triangulated points (track < 3): "
        f"{coverage['frac_weak_points']:.1%}" if coverage['frac_weak_points'] is not None else "N/A",
        f"Mean observations/image : {coverage['mean_obs_per_image']:.0f}",
        f"Min observations/image  : {coverage['min_obs_per_image']}",
        "",
        f"Baseline/depth gaps flagged (> {coverage['translation_ratio_threshold']}): "
        f"{coverage['n_translation_flagged']} / {coverage['n_gaps']}",
    ]
    for g in coverage["worst_translation_gaps"]:
        lines.append(f"  {g['image_a']} -> {g['image_b']}: ratio {g['ratio']:.2f}  (gap {g['gap_m']:.2f} m)")

    lines.append("")
    lines.append(f"Rotation gaps flagged (> {coverage['rotation_deg_threshold']:.0f} deg): "
                 f"{coverage['n_rotation_flagged']} / {coverage['n_gaps']}")
    for g in coverage["worst_rotation_gaps"]:
        lines.append(f"  {g['image_a']} -> {g['image_b']}: {g['rotation_deg']:.1f} deg")

    path.write_text("\n".join(lines) + "\n")
