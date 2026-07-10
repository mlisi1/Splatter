"""Heuristic 2DGS training-parameter suggestions from the reconstructed scene geometry.

2DGS (Huang et al., SIGGRAPH 2024, github.com/hbb1/2d-gaussian-splatting) exposes
--depth_ratio and --lambda_dist as the two knobs its own docs and eval scripts treat as
scene-dependent: the README recommends depth_ratio=0 (mean depth) for unbounded/large
scenes to reduce "disk-aliasing" artifacts, while the project's own DTU eval script
(scripts/dtu_eval.py) trains those bounded, object-centric, masked-background captures
with `--depth_ratio 1.0 --lambda_dist 1000`. This module estimates which regime a
Splatter reconstruction falls into by intersecting camera viewing rays: cameras orbiting
a central subject converge their principal axes near a common point relative to the
scene's own scale; cameras moving through a scene along a path do not.

This is a heuristic starting point for manual tuning, not a substitute for it — always
sanity-check against the camera trajectory plot in report.html.
"""

import numpy as np

from splatter.utils.geometry import qvec2rotmat


def _camera_poses(images: dict) -> tuple[np.ndarray, np.ndarray]:
    """Camera centers and viewing directions, in temporal (filename) order."""
    centers, dirs = [], []
    for img in sorted(images.values(), key=lambda im: im.name):
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        centers.append(-R.T @ t)
        dirs.append(R.T @ np.array([0.0, 0.0, 1.0]))
    return np.array(centers), np.array(dirs)


def _loop_closure_ratio(centers: np.ndarray) -> float:
    """
    Ratio of the distance between the first and last camera positions (in temporal
    order) to the trajectory's own bounding-box diagonal.

    Small (near 0): the camera returned close to where it started — an orbit/loop.
    Large (~1 or more): start and end are far apart relative to the trajectory's own
    extent — a one-way path (walkthrough, flyover, drive-by).

    This is a necessary complement to ray convergence: a camera translating sideways
    while facing a fixed direction (a "crab-walk" or wide establishing pan) can produce
    a deceptively low ray-convergence ratio despite never orbiting anything, because its
    residual and the visible scene extent happen to be on the same order over a bounded
    path segment. Loop closure catches this because a one-way path's endpoints are, by
    definition, far apart, regardless of how much its rays happen to converge.
    """
    diag = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    if diag < 1e-9:
        return 0.0
    return float(np.linalg.norm(centers[-1] - centers[0]) / diag)


def _rays_intersection(centers: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Least-squares closest point to a bundle of 3D lines (camera principal axes).

    Each line contributes the normal equation (I - d d^T)(X - o) = 0; summing over all
    cameras and solving gives the point X minimizing the sum of squared perpendicular
    distances to every ray. Returns (X, rms perpendicular distance from X to each ray).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(centers, dirs):
        d = d / np.linalg.norm(d)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    X = np.linalg.lstsq(A, b, rcond=None)[0]

    residuals = [np.linalg.norm((np.eye(3) - np.outer(d / np.linalg.norm(d), d / np.linalg.norm(d))) @ (X - o))
                 for o, d in zip(centers, dirs)]
    return X, float(np.sqrt(np.mean(np.square(residuals))))


def suggest_2dgs_params(images: dict, points3d: dict) -> dict:
    """
    Classify the reconstruction as an object-centric orbit vs. an unbounded path capture,
    and return suggested 2DGS flags.

    Returns {} if there are too few registered images (< 20) or 3D points (< 500) to make
    a reliable geometric estimate. Otherwise returns a dict with scene_type, depth_ratio,
    lambda_dist, the diagnostic numbers behind the call, a human-readable reasoning
    string, and an example training command.
    """
    if len(images) < 20 or len(points3d) < 500:
        return {}

    centers, dirs = _camera_poses(images)
    focus, ray_rms = _rays_intersection(centers, dirs)
    loop_ratio = _loop_closure_ratio(centers)

    pts = np.array([p.xyz for p in points3d.values()])
    lo, hi = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
    scene_diag = float(np.linalg.norm(hi - lo))
    if scene_diag < 1e-6:
        return {}

    # Low ratio: rays converge tightly relative to scene scale -> orbiting a subject.
    # High ratio: rays are closer to parallel / diverging -> moving along a path.
    convergence_ratio = ray_rms / scene_diag

    # Ray convergence alone is fooled by a camera that translates sideways while facing
    # a fixed direction: over a bounded path segment its residual and the visible scene
    # extent end up on the same order, giving a deceptively low ratio. Loop closure is
    # required too — an orbit returns near its start; a one-way path does not.
    if convergence_ratio < 0.35 and loop_ratio < 0.5:
        scene_type = "object-centric / orbit"
        depth_ratio, lambda_dist = 1.0, 1000
        reasoning = (
            f"Camera viewing rays converge tightly (perpendicular-distance RMS {ray_rms:.2f} m "
            f"vs. scene diagonal {scene_diag:.2f} m, ratio {convergence_ratio:.2f}) and the "
            f"trajectory closes back near its start (loop ratio {loop_ratio:.2f}) — consistent "
            "with an orbit around a central subject, like 2DGS's DTU benchmark, which trains with "
            "these same flags."
        )
    else:
        scene_type = "unbounded / path"
        depth_ratio, lambda_dist = 0.0, 0
        reasoning = (
            f"Camera viewing rays converge poorly relative to scene scale (RMS {ray_rms:.2f} m "
            f"vs. scene diagonal {scene_diag:.2f} m, ratio {convergence_ratio:.2f}) and/or the "
            f"trajectory ends far from where it started (loop ratio {loop_ratio:.2f}) — consistent "
            "with the camera moving through the scene along a path rather than orbiting a subject. "
            "The 2DGS README recommends depth_ratio=0 (mean depth) here to reduce \"disk-aliasing\" "
            "artifacts; lambda_dist is left at its default (off)."
        )

    return {
        "scene_type": scene_type,
        "depth_ratio": depth_ratio,
        "lambda_dist": lambda_dist,
        "convergence_ratio": convergence_ratio,
        "ray_rms": ray_rms,
        "scene_diag": scene_diag,
        "loop_ratio": loop_ratio,
        "reasoning": reasoning,
        "command": f"python train.py -s <dataset_dir> --depth_ratio {depth_ratio} --lambda_dist {lambda_dist}",
    }


def write_suggestion_file(output_dir, suggestion: dict) -> None:
    """Write <output_dir>/2dgs_suggested_params.txt, or skip silently if suggestion is empty."""
    if not suggestion:
        return
    path = output_dir / "2dgs_suggested_params.txt"
    path.write_text(
        "2DGS (2D Gaussian Splatting) suggested training parameters\n"
        "=============================================================\n"
        "Heuristic only — a starting point for manual tuning, not a guarantee.\n"
        "Based on: github.com/hbb1/2d-gaussian-splatting (Huang et al., SIGGRAPH 2024)\n\n"
        f"Detected scene type : {suggestion['scene_type']}\n"
        f"Convergence ratio   : {suggestion['convergence_ratio']:.3f} "
        f"(ray RMS {suggestion['ray_rms']:.2f} m / scene diagonal {suggestion['scene_diag']:.2f} m)\n"
        f"Loop closure ratio  : {suggestion['loop_ratio']:.3f} "
        f"(first/last camera distance / trajectory diagonal)\n\n"
        f"Reasoning:\n{suggestion['reasoning']}\n\n"
        f"Suggested command:\n  {suggestion['command']}\n"
    )
