"""Camera geometry helpers: rotation, intrinsics, unprojection, RANSAC scale fitting."""

import numpy as np

from splatter.io.colmap import Camera


def qvec2rotmat(qvec) -> np.ndarray:
    """Convert a COLMAP quaternion (w, x, y, z) to a 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def get_intrinsics(camera: Camera) -> np.ndarray:
    """Return a 3×3 K matrix from a COLMAP Camera object."""
    p = camera.params
    if camera.model in ("OPENCV", "FULL_OPENCV", "PINHOLE"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif camera.model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    elif camera.model in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx = p[1]; cy = p[2]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    img_bgr: np.ndarray,
    max_pts: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject a metric depth map to world-space points with RGB colours.

    Uses the COLMAP convention: X_world = R^T @ (X_cam - t).

    Returns:
        pts   : (N, 3) float64 world-space coordinates
        colors: (N, 3) uint8 RGB
    """
    import cv2

    valid = (depth > 0.1) & (depth < 300.0)
    ys, xs = np.where(valid)
    if len(ys) > max_pts:
        sel = np.random.choice(len(ys), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    depths = depth[ys, xs].astype(np.float64)

    K_inv = np.linalg.inv(K)
    hom = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    X_cam = K_inv @ hom * depths[np.newaxis, :]
    X_world = R.T @ (X_cam - t[:, np.newaxis])

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    colors = img_rgb[ys, xs]

    return X_world.T, colors


def fit_scale_shift_ransac(
    metric_d: np.ndarray,
    raw_d: np.ndarray,
    inlier_thr: float,
    min_inliers: int,
) -> tuple[float | None, float | None, int]:
    """
    Fit D_metric = s * D_raw + b via LO-RANSAC.

    Returns (s, b, n_inliers), or (None, None, 0) if fitting fails.
    """
    n = len(metric_d)
    if n < 2:
        return None, None, 0

    rng = np.random.default_rng(0)
    n_iters = min(2000, max(100, n * (n - 1) // 2))
    best_s, best_b, best_n = None, None, 0

    for _ in range(n_iters):
        idx = rng.choice(n, 2, replace=False)
        dr, dm = raw_d[idx], metric_d[idx]
        if abs(dr[1] - dr[0]) < 1e-9:
            continue
        s = (dm[1] - dm[0]) / (dr[1] - dr[0])
        b = dm[0] - s * dr[0]
        if s <= 0:
            continue
        mask = np.abs(metric_d - (s * raw_d + b)) < inlier_thr
        n_in = int(mask.sum())
        if n_in > best_n:
            A = np.stack([raw_d[mask], np.ones(n_in)], axis=1)
            sol, *_ = np.linalg.lstsq(A, metric_d[mask], rcond=None)
            s_lo, b_lo = sol
            if s_lo > 0:
                mask2 = np.abs(metric_d - (s_lo * raw_d + b_lo)) < inlier_thr
                n_in2 = int(mask2.sum())
                if n_in2 > best_n:
                    best_s, best_b, best_n = s_lo, b_lo, n_in2

    if best_n < min_inliers:
        return None, None, best_n
    return best_s, best_b, best_n
