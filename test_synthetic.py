#!/usr/bin/env python3
"""
Synthetic densification test.

Creates a known 3-D scene (tilted ground plane + a box), places 8 cameras
around it, renders exact depth maps, runs our unprojection code, and checks
whether the recovered point cloud matches the ground truth.

Also tests the scale/shift RANSAC path to verify DAv2 output interpretation.

Run inside the Docker container:
    python test_synthetic.py
"""

import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, "/home/elechim/Splatter")

from splatter.utils.geometry import unproject_depth, fit_scale_shift_ransac


# ---------------------------------------------------------------------------
# Scene definition  (world coords: x=right, y=forward, z=up)
# ---------------------------------------------------------------------------

def make_scene_points(n_per_dim: int = 30) -> np.ndarray:
    """Tilted ground plane z = 0.05*x, plus a 1×1×1 box at (0,0,0.5)."""
    xs = np.linspace(-3, 3, n_per_dim)
    ys = np.linspace(-3, 3, n_per_dim)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = 0.05 * XX  # slight tilt
    ground = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # box faces (top/bottom/sides sampled)
    box = []
    for s in np.linspace(-0.5, 0.5, 10):
        for t in np.linspace(-0.5, 0.5, 10):
            box += [[s, t,  0.5]]   # top
            box += [[s, t,  0.0]]   # bottom
            box += [[0.5, s,  t + 0.25]]  # side
            box += [[-0.5, s, t + 0.25]]  # side
    box = np.array(box)

    return np.vstack([ground, box])


# ---------------------------------------------------------------------------
# Camera helpers  (COLMAP convention: X_cam = R @ X_world + t)
# ---------------------------------------------------------------------------

def look_at(cam_pos: np.ndarray, target: np.ndarray, world_up=np.array([0., 0., 1.])):
    """Return (R, t) for a camera at cam_pos looking at target."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, world_up)
    if np.linalg.norm(x) < 1e-6:
        world_up = np.array([0., 1., 0.])
        x = np.cross(z, world_up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)   # already normalised
    # rows of R = camera axes expressed in world coords
    R = np.stack([x, y, z], axis=0)        # world→cam rotation
    t = -R @ cam_pos                        # COLMAP translation
    return R.astype(np.float64), t.astype(np.float64)


def make_cameras(n: int = 8, radius: float = 5.0, height: float = 3.0):
    """Circular orbit around origin."""
    cameras = []
    for i in range(n):
        angle = i * 2 * np.pi / n
        pos = np.array([radius * np.cos(angle), radius * np.sin(angle), height])
        R, t = look_at(pos, np.array([0., 0., 0.]))
        cameras.append((R, t, pos))
    return cameras


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def project(X_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    """Project world points; return (u, v, z_cam), filtering z>0."""
    X_cam = (R @ X_world.T + t[:, None]).T  # (N, 3)
    mask = X_cam[:, 2] > 0.1
    X_c = X_cam[mask]
    uv = (K @ X_c.T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    return u, v, X_c[:, 2], mask


def render_depth_map(world_pts: np.ndarray, R, t, K, W: int, H: int) -> np.ndarray:
    """
    Render an exact per-pixel depth map from world_pts.
    Each pixel gets the nearest (smallest z_cam) point projected onto it.
    """
    depth = np.zeros((H, W), dtype=np.float32)
    u, v, z, _ = project(world_pts, R, t, K)
    ui = u.astype(int)
    vi = v.astype(int)
    valid = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[valid], vi[valid], z[valid]

    # nearest-z per pixel
    order = np.argsort(-z)  # descending → later writes = smaller z (nearer)
    for idx in order:
        if depth[vi[idx], ui[idx]] == 0 or z[idx] < depth[vi[idx], ui[idx]]:
            depth[vi[idx], ui[idx]] = z[idx]
    return depth


def render_color_image(world_pts: np.ndarray, R, t, K, W: int, H: int) -> np.ndarray:
    """Render a colour image: colour each pixel by the world-Z of the nearest point."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    u, v, z, _ = project(world_pts, R, t, K)
    ui, vi = u.astype(int), v.astype(int)
    valid = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    for idx in np.where(valid)[0]:
        # colour: blue→red gradient over world-z range
        wz = world_pts[idx, 2] if idx < len(world_pts) else z[idx]
        r = int(np.clip(wz * 60 + 128, 0, 255))
        img[vi[idx], ui[idx]] = [r, 100, 255 - r]
    return img


# ---------------------------------------------------------------------------
# Test 1 — perfect depth → unproject → compare to ground truth
# ---------------------------------------------------------------------------

def test_perfect_unproject():
    print("\n" + "="*60)
    print("TEST 1 — Perfect depth maps → unproject → ground truth check")
    print("="*60)

    W, H = 320, 240
    fx = fy = 250.0
    cx, cy = W / 2, H / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    world_pts = make_scene_points(n_per_dim=20)
    cameras = make_cameras(n=8)

    all_recovered = []
    for cam_idx, (R, t, pos) in enumerate(cameras):
        depth = render_depth_map(world_pts, R, t, K, W, H)
        img_bgr = render_color_image(world_pts, R, t, K, W, H)

        # number of non-zero pixels (coverage check)
        n_pix = (depth > 0).sum()
        if n_pix < 10:
            print(f"  Cam {cam_idx}: too few projected pixels ({n_pix}), skipping")
            continue

        pts_w, _ = unproject_depth(depth, K, R, t, img_bgr, max_pts=200_000)
        all_recovered.append(pts_w)

        # Check: every recovered point should be close to some world point
        # (nearest-neighbour distance should be < 0.05 m)
        from scipy.spatial import cKDTree
        tree = cKDTree(world_pts)
        dists, _ = tree.query(pts_w, k=1)
        bad = (dists > 0.1).sum()
        print(f"  Cam {cam_idx}: pos={pos.round(2)}, pixels={n_pix:,}, "
              f"recovered={len(pts_w):,}, bad (>0.1m)={bad} ({100*bad/max(len(pts_w),1):.1f}%)")

    if all_recovered:
        all_pts = np.vstack(all_recovered)
        print(f"\n  Total recovered: {len(all_pts):,}")
        print(f"  World X range: {world_pts[:,0].min():.2f}..{world_pts[:,0].max():.2f}")
        print(f"  Recovered X range: {all_pts[:,0].min():.2f}..{all_pts[:,0].max():.2f}")
        print(f"  World Z range: {world_pts[:,2].min():.2f}..{world_pts[:,2].max():.2f}")
        print(f"  Recovered Z range: {all_pts[:,2].min():.2f}..{all_pts[:,2].max():.2f}")
        print("\n  PASS if recovered ranges ~match world ranges and bad% < 5%")


# ---------------------------------------------------------------------------
# Test 2 — simulate DAv2 output, check RANSAC in depth vs disparity space
# ---------------------------------------------------------------------------

def simulate_dav2_output(depth_gt: np.ndarray, mode: str, alpha: float, beta: float):
    """
    Simulate DAv2 relative output.

    mode='depth':     d_raw = alpha * z + beta  (larger=farther)
    mode='disparity': d_raw = alpha * (1/z) + beta  (larger=closer)
    """
    eps = 1e-6
    if mode == 'depth':
        d_raw = alpha * depth_gt + beta
    else:
        d_raw = alpha / np.maximum(depth_gt, eps) + beta
    # add tiny noise
    d_raw += np.random.default_rng(7).normal(0, 0.002, d_raw.shape)
    return d_raw.astype(np.float32)


def test_ransac_interpretation():
    print("\n" + "="*60)
    print("TEST 2 — RANSAC interpretation: depth vs disparity")
    print("="*60)

    # Ground truth depths for 200 SfM anchor points in [1, 10] m
    rng = np.random.default_rng(0)
    z_gt = rng.uniform(1.0, 10.0, 200)

    for alpha, beta in [(0.8, 0.3), (2.0, -0.5), (0.5, 1.0)]:
        for mode in ('depth', 'disparity'):
            d_raw = simulate_dav2_output(z_gt, mode, alpha, beta)

            # What the pipeline does: fit 1/z = s * d_raw + b  (disparity assumption)
            s_d, b_d, n_d = fit_scale_shift_ransac(1.0 / z_gt, d_raw, 0.05, 20)
            # Alternative: fit z = s * d_raw + b  (depth assumption)
            s_z, b_z, n_z = fit_scale_shift_ransac(z_gt, d_raw, 0.3, 20)

            # Check reconstruction accuracy
            if s_d is not None:
                z_rec_d = 1.0 / np.clip(s_d * d_raw + b_d, 1e-6, None)
                err_d = np.abs(z_rec_d - z_gt).mean()
            else:
                err_d = float('inf')

            if s_z is not None:
                z_rec_z = s_z * d_raw + b_z
                err_z = np.abs(z_rec_z - z_gt).mean()
            else:
                err_z = float('inf')

            winner = "DISPARITY-fit" if err_d < err_z else "DEPTH-fit"
            print(f"\n  mode={mode:10s}  alpha={alpha:.1f}  beta={beta:.1f}")
            print(f"    Disparity-fit (1/z=s*d+b):  inliers={n_d:3d}  mean_err={err_d:.4f} m")
            print(f"    Depth-fit     (z=s*d+b):     inliers={n_z:3d}  mean_err={err_z:.4f} m")
            print(f"    => Better: {winner}")


# ---------------------------------------------------------------------------
# Test 3 — single-camera round-trip with simulated DAv2 scale
# ---------------------------------------------------------------------------

def test_roundtrip_with_scale():
    print("\n" + "="*60)
    print("TEST 3 — Round-trip: render scene → add DAv2-style scale → RANSAC → unproject → check")
    print("="*60)

    W, H = 320, 240
    fx = fy = 250.0
    K = np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=np.float64)

    world_pts = make_scene_points(n_per_dim=25)
    R, t, cam_pos = make_cameras(n=1)[0]

    # Render ground-truth depth
    depth_gt = render_depth_map(world_pts, R, t, K, W, H)
    img_bgr = render_color_image(world_pts, R, t, K, W, H)

    # Get the SfM "anchor" pixels: project world_pts and sample depth_gt
    u, v, z_sfm, _ = project(world_pts, R, t, K)
    ui, vi = u.astype(int), v.astype(int)
    in_frame = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z_sfm > 0)
    ui, vi, z_sfm = ui[in_frame], vi[in_frame], z_sfm[in_frame]

    for mode in ('depth', 'disparity'):
        # Simulate DAv2 output (alpha=0.7, beta=0.4)
        d_raw_full = simulate_dav2_output(depth_gt, mode, alpha=0.7, beta=0.4)
        d_raw_full[depth_gt == 0] = 0   # background = 0

        # Sample d_raw at the SfM projected positions
        d_anchor = d_raw_full[vi, ui]
        anchor_valid = (z_sfm > 0.1) & (d_anchor > 1e-6)

        # Fit: disparity space (current pipeline)
        inv_z = 1.0 / np.clip(z_sfm[anchor_valid], 1e-6, None)
        s, b, n_in = fit_scale_shift_ransac(inv_z, d_anchor[anchor_valid], 0.05, 20)

        if s is not None:
            inv_metric = np.clip(s * d_raw_full + b, 1.0/300, 1.0/0.1)
            d_metric = (1.0 / inv_metric).astype(np.float32)
            pts_w, _ = unproject_depth(d_metric, K, R, t, img_bgr, max_pts=500_000)
            # Check accuracy: fraction of recovered points within 0.2 m of ground truth
            from scipy.spatial import cKDTree
            tree = cKDTree(world_pts)
            dists, _ = tree.query(pts_w, k=1)
            good = (dists < 0.2).sum() / max(len(pts_w), 1)
            print(f"\n  mode={mode:10s} → disparity RANSAC:  inliers={n_in}, "
                  f"recovered={len(pts_w):,}, within 0.2m={100*good:.1f}%")
        else:
            print(f"\n  mode={mode:10s} → disparity RANSAC FAILED (n_in={n_in})")

        # Fit: depth space (alternative)
        s2, b2, n_in2 = fit_scale_shift_ransac(z_sfm[anchor_valid], d_anchor[anchor_valid], 0.3, 20)
        if s2 is not None:
            d_metric2 = np.clip(s2 * d_raw_full + b2, 0.1, 300.0).astype(np.float32)
            d_metric2[depth_gt == 0] = 0
            pts_w2, _ = unproject_depth(d_metric2, K, R, t, img_bgr, max_pts=500_000)
            tree = cKDTree(world_pts)
            dists2, _ = tree.query(pts_w2, k=1)
            good2 = (dists2 < 0.2).sum() / max(len(pts_w2), 1)
            print(f"  mode={mode:10s} → depth RANSAC:       inliers={n_in2}, "
                  f"recovered={len(pts_w2):,}, within 0.2m={100*good2:.1f}%")
        else:
            print(f"  mode={mode:10s} → depth RANSAC FAILED (n_in2={n_in2})")


# ---------------------------------------------------------------------------
# Test 4 — camera coordinate sanity check
# ---------------------------------------------------------------------------

def test_camera_sanity():
    print("\n" + "="*60)
    print("TEST 4 — Camera coordinate sanity")
    print("="*60)

    # Camera at (0, -5, 3) looking at origin
    cam_pos = np.array([0., -5., 3.])
    target = np.array([0., 0., 0.])
    R, t = look_at(cam_pos, target)

    # The origin in world should project to image center
    K = np.array([[300., 0., 160.], [0., 300., 120.], [0., 0., 1.]])
    origin_cam = R @ target + t
    print(f"  Origin in cam frame: {origin_cam.round(3)}")
    print(f"  Expected z > 0 (in front): {'PASS' if origin_cam[2] > 0 else 'FAIL'}")

    u = K[0, 0] * origin_cam[0] / origin_cam[2] + K[0, 2]
    v = K[1, 1] * origin_cam[1] / origin_cam[2] + K[1, 2]
    print(f"  Origin projects to ({u:.1f}, {v:.1f}), image center is (160, 120)")
    print(f"  Close to center: {'PASS' if abs(u-160)<5 and abs(v-120)<5 else 'FAIL'}")

    # Camera center recovery: C_world = -R^T @ t
    C = -R.T @ t
    dist = np.linalg.norm(C - cam_pos)
    print(f"  Recovered cam pos: {C.round(3)}, expected: {cam_pos}")
    print(f"  Error: {dist:.6f} m  {'PASS' if dist < 1e-10 else 'FAIL'}")

    # Unproject a pixel at image center with depth = ||target - cam_pos||
    depth_to_target = np.linalg.norm(target - cam_pos)
    # The ray from image center has direction = R^T @ [0,0,1] in world
    # But depth_map depth = z_cam (not ray distance!)
    # z_cam of the origin:
    z_cam_origin = origin_cam[2]
    depth_map = np.zeros((240, 320), dtype=np.float32)
    depth_map[120, 160] = float(z_cam_origin)
    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    pts, _ = unproject_depth(depth_map, K, R, t, dummy_img, max_pts=10)
    if len(pts) > 0:
        err = np.linalg.norm(pts[0] - target)
        print(f"  Unproject image center at z={z_cam_origin:.3f} → {pts[0].round(3)}, "
              f"expected {target}, error={err:.6f}  {'PASS' if err < 1e-6 else 'FAIL'}")
    else:
        print("  Unproject returned no points — FAIL")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError:
        print("scipy not found, installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scipy", "-q"])

    test_camera_sanity()
    test_perfect_unproject()
    test_ransac_interpretation()
    test_roundtrip_with_scale()

    print("\n" + "="*60)
    print("All tests complete.")
    print("="*60)
