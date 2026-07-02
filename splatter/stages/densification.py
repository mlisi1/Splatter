"""Stage 4 — Point cloud densification via Depth Anything v2 (±LiDAR)."""

import gc
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from splatter.io.colmap import Point3D, read_model, write_points3d_binary, write_points3d_text
from splatter.utils.geometry import (
    fit_scale_shift_ransac,
    get_intrinsics,
    qvec2rotmat,
    unproject_depth,
)

logger = logging.getLogger("gs_init")

_ENCODER_MAP = {"dav2_vitl": "vitl", "dav2_vitb": "vitb", "dav2_vits": "vits"}
_MODEL_CFGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
_WEIGHT_URLS = {
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
}


# ---------------------------------------------------------------------------
# Depth model
# ---------------------------------------------------------------------------

def load_depth_model(model_name: str, device: str):
    """Load a Depth Anything V2 model, downloading weights on first use."""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise ImportError(
            "Install: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
        )
    import torch

    encoder = _ENCODER_MAP[model_name]
    cache_dir = Path.home() / ".cache" / "depth_anything_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weight_file = cache_dir / f"depth_anything_v2_{encoder}.pth"

    if not weight_file.exists():
        logger.info(f"Downloading Depth Anything V2 {encoder} weights…")
        urllib.request.urlretrieve(_WEIGHT_URLS[encoder], weight_file)

    model = DepthAnythingV2(**_MODEL_CFGS[encoder])
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    return model.to(device).eval()


def infer_depth(model, img_path: Path) -> np.ndarray:
    """Run Depth Anything V2 on a single image; returns float32 H×W array."""
    import torch

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        depth = model.infer_image(img_rgb)
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Sky segmentation (SegFormer-B0 on ADE20K, class 2 = sky)
# ---------------------------------------------------------------------------

_SKY_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
_ADE20K_SKY_CLASS = 2


def load_sky_model(device: str):
    """Load SegFormer-B0 sky segmentation model (downloads ~14 MB on first use)."""
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError:
        raise ImportError("pip install transformers>=4.37 for --sky-mask support")
    import torch

    logger.info("Loading sky segmentation model (SegFormer-B0 / ADE20K)…")
    processor = SegformerImageProcessor.from_pretrained(_SKY_MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(_SKY_MODEL_NAME)
    return processor, model.to(device).eval()


def compute_sky_mask(img_bgr: np.ndarray, sky_model, device: str) -> np.ndarray:
    """
    Returns a boolean mask (True = sky) at the same resolution as img_bgr.
    Uses SegFormer-B0 pretrained on ADE20K; sky is class index 2.
    """
    from PIL import Image
    import torch

    processor, model = sky_model
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=Image.fromarray(img_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, 150, H/4, W/4)
    seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    seg_full = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
    return seg_full == _ADE20K_SKY_CLASS


# ---------------------------------------------------------------------------
# Re-process cached depth maps (skip DAv2 inference)
# ---------------------------------------------------------------------------

def reprocess_depth_maps(
    output_dir: Path,
    sparse_0: Path,
    voxel_size: float = 0.02,
    max_points: int = 5_000_000,
    outlier_nb: int = 20,
    outlier_std: float = 3.0,
) -> dict:
    """
    Load pre-computed depth maps from depth/*.npy and re-run unproject +
    merge/downsample. Skips DAv2 inference entirely.

    Use this when only the point-cloud post-processing flags have changed
    (--voxel-size, --max-points, --outlier-std, --outlier-nb).
    """
    cameras, images, _ = read_model(sparse_0)
    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"

    all_pts, all_cols = [], []

    for img_data in tqdm(list(images.values()), desc="Unprojecting cached depth maps"):
        img_path = images_dir / img_data.name
        depth_path = depth_dir / (img_path.stem + ".npy")
        if not depth_path.exists() or not img_path.exists():
            continue

        cam = cameras[img_data.camera_id]
        K = get_intrinsics(cam)
        R = qvec2rotmat(img_data.qvec)
        t = img_data.tvec

        d_metric = np.load(str(depth_path))
        img_bgr = cv2.imread(str(img_path))
        pts_w, cols = unproject_depth(d_metric, K, R, t, img_bgr)
        all_pts.append(pts_w)
        all_cols.append(cols)

    return _build_point3d_dict(_merge_and_downsample(
        all_pts, all_cols,
        voxel_size=voxel_size, max_points=max_points,
        outlier_nb=outlier_nb, outlier_std=outlier_std,
    ))


# ---------------------------------------------------------------------------
# Branch A — images only
# ---------------------------------------------------------------------------

def run_images_only(
    output_dir: Path, sparse_0: Path, depth_model, device: str,
    min_depth: float = 0.0,
    sky_model=None,
    mask_bottom: float = 0.0,
    voxel_size: float = 0.02,
    max_points: int = 5_000_000,
    outlier_nb: int = 20,
    outlier_std: float = 3.0,
) -> dict:
    """
    Densify using Depth Anything V2 depth maps anchored to SfM points.

    For each registered image:
      - Infer raw depth with Depth Anything V2
      - Align to metric scale via LO-RANSAC on visible SfM points
      - Unproject to world space

    min_depth: discard pixels closer than this (camera Z, metres).
               Useful to exclude a robot body / camera rig always in view.

    Returns a dict of Point3D objects (empty-track, for GS initialisation).
    """
    cameras, images, points3d = read_model(sparse_0)
    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(exist_ok=True)

    # Preserve the original SfM sparse cloud so re-runs always anchor to
    # true triangulated points rather than the densified cloud.
    sfm_pts_path = sparse_0 / "points3D_sfm.bin"
    if not sfm_pts_path.exists():
        import shutil
        shutil.copy(sparse_0 / "points3D.bin", sfm_pts_path)
        sfm_points3d = points3d
    else:
        from splatter.io.colmap import read_points3d_binary
        sfm_points3d = read_points3d_binary(sfm_pts_path)
        logger.info(f"Using SfM backup ({len(sfm_points3d):,} pts) for depth alignment")

    all_pts, all_cols = [], []

    for batch in _batches(list(images.values()), 50):
        for img_data in tqdm(batch, desc="Depth estimation"):
            img_path = images_dir / img_data.name
            if not img_path.exists():
                continue

            cam = cameras[img_data.camera_id]
            K = get_intrinsics(cam)
            R = qvec2rotmat(img_data.qvec)
            t = img_data.tvec
            W, H = cam.width, cam.height

            sfm_xyz = _visible_sfm_points(img_data, sfm_points3d)
            if len(sfm_xyz) < 20:
                continue

            X_cam = R @ sfm_xyz.T + t[:, np.newaxis]
            sfm_z = X_cam[2]
            uv = K @ X_cam
            us = uv[0] / uv[2]
            vs = uv[1] / uv[2]

            img_bgr = cv2.imread(str(img_path))
            d_raw = infer_depth(depth_model, img_path)
            dh, dw = d_raw.shape
            valid = (sfm_z > 0) & (us >= 0) & (us < dw) & (vs >= 0) & (vs < dh)
            if valid.sum() < 20:
                continue

            us_v = us[valid].astype(int)
            vs_v = vs[valid].astype(int)
            # DAv2 relative output is disparity (larger = closer), not depth.
            # Fit in disparity space: 1/z = s * d_raw + b  (both sides positive,
            # positively correlated), then invert to get metric depth.
            sfm_inv_z = 1.0 / np.clip(sfm_z[valid], 1e-6, None)
            s, b, _ = fit_scale_shift_ransac(sfm_inv_z, d_raw[vs_v, us_v], 0.05, 20)
            if s is None:
                continue

            # inv_raw ≤ 0 → invalid disparity → always mask out.
            # Use 20× the farthest SfM point as a loose safety cap; when
            # --sky-mask is active the semantic mask handles sky precisely.
            z_far_sfm = float(sfm_z[valid].max())
            max_depth = z_far_sfm * 20.0
            near_cut = max(min_depth, 0.05)  # never below 5 cm
            inv_raw = s * d_raw + b
            d_raw_m = np.where(inv_raw > 1e-6, 1.0 / np.maximum(inv_raw, 1e-9), 0.0)
            d_metric = np.where(
                (d_raw_m > near_cut) & (d_raw_m < max_depth),
                d_raw_m,
                0.0,
            ).astype(np.float32)
            if sky_model is not None:
                sky_px = compute_sky_mask(img_bgr, sky_model, device)
                d_metric[sky_px] = 0.0
            if mask_bottom > 0.0:
                cut = int(d_metric.shape[0] * mask_bottom)
                if cut > 0:
                    d_metric[-cut:, :] = 0.0
            np.save(depth_dir / (img_path.stem + ".npy"), d_metric)

            pts_w, cols = unproject_depth(d_metric, K, R, t, img_bgr)
            all_pts.append(pts_w)
            all_cols.append(cols)

        gc.collect()

    return _build_point3d_dict(_merge_and_downsample(
        all_pts, all_cols,
        voxel_size=voxel_size, max_points=max_points,
        outlier_nb=outlier_nb, outlier_std=outlier_std,
    ))


# ---------------------------------------------------------------------------
# Branch B — LiDAR + Depth Anything v2
# ---------------------------------------------------------------------------

def run_lidar(
    output_dir: Path,
    sparse_0: Path,
    lidar_path: Path,
    T_cam_to_lidar: np.ndarray,
    depth_model,
    device: str,
    min_depth: float = 0.0,
    sky_model=None,
    mask_bottom: float = 0.0,
    voxel_size: float = 0.02,
    max_points: int = 5_000_000,
    outlier_nb: int = 20,
    outlier_std: float = 3.0,
) -> dict:
    """
    Densify using LiDAR for metric scale and Depth Anything V2 for dense coverage.

    For each registered image:
      - Project LiDAR into image space (with HPR)
      - Align Depth Anything V2 raw depth to LiDAR via LO-RANSAC
      - Fuse: use LiDAR where available, scaled depth elsewhere
      - Fall back to SfM-anchored alignment if LiDAR coverage is too sparse

    Returns a dict of Point3D objects (empty-track).
    """
    import open3d as o3d

    cameras, images, points3d = read_model(sparse_0)
    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(exist_ok=True)

    sfm_pts_path = sparse_0 / "points3D_sfm.bin"
    if not sfm_pts_path.exists():
        import shutil
        shutil.copy(sparse_0 / "points3D.bin", sfm_pts_path)
        sfm_points3d = points3d
    else:
        from splatter.io.colmap import read_points3d_binary
        sfm_points3d = read_points3d_binary(sfm_pts_path)
        logger.info(f"Using SfM backup ({len(sfm_points3d):,} pts) for depth alignment")

    pts_lidar = _load_lidar(lidar_path)
    T_lidar_to_cam = np.linalg.inv(T_cam_to_lidar)
    pts_hom = np.hstack([pts_lidar, np.ones((len(pts_lidar), 1))])

    all_pts, all_cols = [], []

    for batch in _batches(list(images.values()), 50):
        for img_data in tqdm(batch, desc="LiDAR+Depth estimation"):
            img_path = images_dir / img_data.name
            if not img_path.exists():
                continue

            cam = cameras[img_data.camera_id]
            K = get_intrinsics(cam)
            R = qvec2rotmat(img_data.qvec)
            t = img_data.tvec
            W, H = cam.width, cam.height

            pts_cam = (T_lidar_to_cam @ pts_hom.T).T[:, :3]
            pts_cam = pts_cam[pts_cam[:, 2] > 0]
            if len(pts_cam) == 0:
                continue

            pts_cam = _apply_hpr(pts_cam)
            L = _build_lidar_depth_image(pts_cam, K, W, H)

            img_bgr = cv2.imread(str(img_path))
            d_raw = infer_depth(depth_model, img_path)
            d_raw_r = cv2.resize(d_raw, (W, H))

            lidar_mask = ~np.isnan(L)
            s, b = _align_to_lidar(L, d_raw_r, lidar_mask)

            if s is None:
                s, b = _align_to_sfm(img_data, sfm_points3d, R, t, K, d_raw_r, W, H)
                use_lidar_fill = False
            else:
                use_lidar_fill = True

            if s is None:
                continue

            lidar_z_vals = L[lidar_mask] if lidar_mask.any() else np.array([])
            sfm_z_l = _visible_sfm_cam_depths(img_data, sfm_points3d, R, t)
            ref_z = np.concatenate([lidar_z_vals, sfm_z_l])
            z_far_ref = float(ref_z.max()) if len(ref_z) else 10.0
            max_depth_l = z_far_ref * 20.0
            near_cut_l = max(min_depth, 0.05)
            inv_raw_l = s * d_raw_r + b
            d_raw_ml = np.where(inv_raw_l > 1e-6, 1.0 / np.maximum(inv_raw_l, 1e-9), 0.0)
            d_fused = np.where(
                (d_raw_ml > near_cut_l) & (d_raw_ml < max_depth_l),
                d_raw_ml,
                0.0,
            ).astype(np.float32)
            if use_lidar_fill:
                d_fused[lidar_mask] = L[lidar_mask]
            if sky_model is not None:
                sky_px = compute_sky_mask(img_bgr, sky_model, device)
                d_fused[sky_px] = 0.0
            if mask_bottom > 0.0:
                cut = int(d_fused.shape[0] * mask_bottom)
                if cut > 0:
                    d_fused[-cut:, :] = 0.0

            np.save(depth_dir / (img_path.stem + ".npy"), d_fused)

            pts_w, cols = unproject_depth(d_fused, K, R, t, img_bgr)
            all_pts.append(pts_w)
            all_cols.append(cols)

        gc.collect()

    return _build_point3d_dict(_merge_and_downsample(
        all_pts, all_cols,
        voxel_size=voxel_size, max_points=max_points,
        outlier_nb=outlier_nb, outlier_std=outlier_std,
    ))


# ---------------------------------------------------------------------------
# Shared: write densified cloud back to sparse/0
# ---------------------------------------------------------------------------

def read_existing_points(sparse_0: Path) -> dict:
    """Return the current points3D dict from sparse/0/ (for cached-run stats)."""
    from splatter.io.colmap import read_points3d_binary
    p = sparse_0 / "points3D.bin"
    return read_points3d_binary(p) if p.exists() else {}


def replace_points3d(output_dir: Path, sparse_0: Path, new_points: dict) -> None:
    """Overwrite points3D.bin and points3D.txt with the densified cloud."""
    logger.info(f"Writing {len(new_points):,} densified points to sparse/0/")
    write_points3d_binary(new_points, sparse_0 / "points3D.bin")
    txt_dir = output_dir / "sparse_txt" / "0"
    txt_dir.mkdir(parents=True, exist_ok=True)
    write_points3d_text(new_points, txt_dir / "points3D.txt")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _visible_sfm_points(img_data, points3d: dict) -> np.ndarray:
    pids = img_data.point3D_ids[img_data.point3D_ids >= 0]
    xyz = [points3d[p].xyz for p in pids if p in points3d]
    return np.array(xyz) if xyz else np.empty((0, 3))


def _visible_sfm_cam_depths(img_data, points3d: dict, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return camera-space Z values for SfM points visible in this image."""
    xyz = _visible_sfm_points(img_data, points3d)
    if len(xyz) == 0:
        return np.empty(0)
    z = (R @ xyz.T + t[:, np.newaxis])[2]
    return z[z > 0]


def _load_lidar(lidar_path: Path) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(lidar_path))
    pts = np.asarray(pcd.points)
    rng = np.linalg.norm(pts, axis=1)
    keep = (pts[:, 2] >= 0) & (rng <= 200.0)
    pts = pts[keep]
    logger.info(f"LiDAR points after filter: {len(pts):,}")
    return pts


def _apply_hpr(pts_cam: np.ndarray) -> np.ndarray:
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        _, pt_map = pcd.hidden_point_removal([0.0, 0.0, 0.0], 1000.0)
        return pts_cam[pt_map]
    except Exception:
        return pts_cam


def _build_lidar_depth_image(pts_cam: np.ndarray, K: np.ndarray, W: int, H: int) -> np.ndarray:
    uv = K @ pts_cam.T
    us = (uv[0] / uv[2]).astype(int)
    vs = (uv[1] / uv[2]).astype(int)
    zs = pts_cam[:, 2]
    ib = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    L = np.full((H, W), np.nan, dtype=np.float32)
    for u, v, z in zip(us[ib], vs[ib], zs[ib]):
        if np.isnan(L[v, u]) or z < L[v, u]:
            L[v, u] = z
    return L


def _align_to_lidar(L: np.ndarray, d_raw: np.ndarray,
                    mask: np.ndarray) -> tuple:
    if mask.sum() < 10:
        return None, None
    lidar_inv_z = 1.0 / np.clip(L[mask], 1e-6, None)
    s, b, _ = fit_scale_shift_ransac(lidar_inv_z, d_raw[mask], 0.05, 10)
    return s, b


def _align_to_sfm(img_data, points3d: dict, R: np.ndarray, t: np.ndarray,
                  K: np.ndarray, d_raw: np.ndarray, W: int, H: int) -> tuple:
    sfm_xyz = _visible_sfm_points(img_data, points3d)
    if len(sfm_xyz) < 20:
        return None, None
    X_cam = R @ sfm_xyz.T + t[:, np.newaxis]
    sfm_z = X_cam[2]
    uv = K @ X_cam
    us = (uv[0] / uv[2])
    vs = (uv[1] / uv[2])
    valid = (sfm_z > 0) & (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    if valid.sum() < 20:
        return None, None
    us_i = us[valid].astype(int)
    vs_i = vs[valid].astype(int)
    sfm_inv_z = 1.0 / np.clip(sfm_z[valid], 1e-6, None)
    s, b, _ = fit_scale_shift_ransac(sfm_inv_z, d_raw[vs_i, us_i], 0.05, 20)
    return s, b


def _merge_and_downsample(
    all_pts: list,
    all_cols: list,
    voxel_size: float = 0.02,
    max_points: int = 5_000_000,
    outlier_nb: int = 20,
    outlier_std: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    logger.info(f"Total raw points: {len(pts):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

    # Initial voxel downsample — increase size until under 10M points
    for v in (voxel_size, voxel_size * 1.5, voxel_size * 2.5):
        pcd = pcd.voxel_down_sample(v)
        logger.info(f"After voxel down (voxel={v:.3f}m): {len(pcd.points):,}")
        if len(pcd.points) <= 10_000_000:
            break

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=outlier_nb, std_ratio=outlier_std)
    logger.info(f"After outlier removal: {len(pcd.points):,}")

    # Secondary downsample to hit the max_points target
    if len(pcd.points) > max_points:
        v2 = voxel_size * 1.5
        pcd = pcd.voxel_down_sample(v2)
        logger.info(f"Re-downsampled to {len(pcd.points):,} (voxel={v2:.3f}m)")

    pts_out = np.asarray(pcd.points)
    cols_out = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
    return pts_out, cols_out


def _build_point3d_dict(
    pts_cols: tuple[np.ndarray, np.ndarray],
) -> dict[int, Point3D]:
    pts, cols = pts_cols
    return {
        i + 1: Point3D(i + 1, xyz, rgb, 0.0,
                       np.array([], dtype=np.uint32),
                       np.array([], dtype=np.uint32))
        for i, (xyz, rgb) in enumerate(zip(pts, cols))
    }
