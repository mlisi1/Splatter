#!/usr/bin/env python3
"""GS Initialization Pipeline — CLI entry point."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from splatter.io.colmap import read_model
from splatter.io.ply import write_ply
from splatter.stages import densification, extraction, features, sfm
from splatter.utils.logging import setup_logging
from splatter import report as report_gen


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GS Initialization Pipeline: video → COLMAP sparse format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path, help="Input video file")
    p.add_argument("--lidar", type=Path, default=None, help="LiDAR PLY file")
    p.add_argument("--extrinsics", type=Path, default=None,
                   help="T_cam_to_lidar 4×4 plain-text matrix")
    p.add_argument("--output", type=Path, default=Path("./output"), help="Output directory")
    p.add_argument("--fps", type=float, default=None,
                   help="Frame extraction rate (default: 5)")
    p.add_argument("--auto-fps", action="store_true",
                   help="Probe 5/7/10 fps on a 30s clip to find the minimum rate that achieves full registration")
    p.add_argument("--max-frames", type=int, default=2000)
    p.add_argument("--iqa-threshold", type=float, default=0.5,
                   help="Blur rejection threshold 0–1 (0 = keep all)")
    p.add_argument("--matcher", choices=["disk", "aliked"], default="disk")
    p.add_argument("--sfm", choices=["global", "hierarchical", "incremental"], default="global")
    p.add_argument("--depth-model", choices=["dav2_vitl", "dav2_vitb", "dav2_vits"],
                   default="dav2_vitl")
    p.add_argument("--resize", type=int, default=0,
                   help="Resize frames so the longer edge is this many pixels (0 = full res)")
    p.add_argument("--min-depth", type=float, default=0.0,
                   help="Discard depth-map pixels closer than this (metres). "
                        "Use to exclude a robot body or camera rig always in view.")
    p.add_argument("--sky-mask", action="store_true",
                   help="Run SegFormer-B0 semantic sky segmentation and zero sky pixels "
                        "before unprojection. Downloads ~14 MB on first use.")
    p.add_argument("--mask-bottom", type=float, default=0.0,
                   help="Fraction of image height to zero out at the bottom of every depth map "
                        "(0.0–1.0). Use to exclude a camera rig or robot body that occupies "
                        "a fixed bottom region of the frame.")
    p.add_argument("--voxel-size", type=float, default=0.02,
                   help="Initial voxel size (m) for point cloud downsampling. "
                        "Larger = fewer points, faster. Tries 1.5× and 2.5× if "
                        "result exceeds 10 M points.")
    p.add_argument("--max-points", type=int, default=5_000_000,
                   help="Target maximum number of points in the densified cloud. "
                        "A secondary voxel pass runs if outlier removal leaves more than this.")
    p.add_argument("--outlier-std", type=float, default=3.0,
                   help="std_ratio for statistical outlier removal (higher = keep more points).")
    p.add_argument("--outlier-nb", type=int, default=20,
                   help="Number of neighbours used in statistical outlier removal.")
    p.add_argument("--no-densify", action="store_true",
                   help="Skip densification; use raw SfM sparse cloud")
    p.add_argument("--skip-sfm", action="store_true",
                   help="Skip SfM if sparse/0/ already exists")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(output_dir: Path, logger) -> bool:
    sparse_0 = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    ok = True

    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        p = sparse_0 / fname
        if not p.exists() or p.stat().st_size == 0:
            logger.error(f"Missing or empty: {p}")
            ok = False

    if not ok:
        return False

    _, images, points3d = read_model(sparse_0)

    if len(images) < 20:
        logger.error(f"Only {len(images)} registered images (need ≥ 20)")
        ok = False
    if len(points3d) < 1000:
        logger.error(f"Only {len(points3d)} 3D points (need ≥ 1000)")
        ok = False

    missing = [img.name for img in images.values()
               if not (images_dir / img.name).exists()]
    if missing:
        logger.warning(f"{len(missing)} images in images.bin missing from images/")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(stats: dict, output_dir: Path, elapsed: float, logger) -> None:
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    n_reg = stats["frames_registered"]
    n_iqa = stats["frames_after_iqa"]
    reg_pct = n_reg / max(n_iqa, 1) * 100

    msg = (
        "\n========================================"
        "\n GS Initialization Pipeline — Summary"
        "\n========================================"
        f"\n Frames extracted       : {stats['frames_extracted']:,}"
        f"\n Frames after IQA filter: {n_iqa:,}"
        f"\n Frames registered      : {n_reg:,}  ({reg_pct:.1f}%)"
        f"\n Sparse points (SfM)    : {stats['sparse_points']:,}"
        f"\n Densification branch   : {stats['densification_branch']}"
        f"\n Densified points       : {stats['densified_points']:,}"
        f"\n Mean reprojection error: {stats['mean_reproj']:.2f} px"
        f"\n Output directory       : {output_dir}"
        f"\n Total runtime          : {mins}m {secs}s"
        "\n========================================"
    )
    print(msg)
    logger.info(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _create_gs_dataset(output_dir: Path, logger) -> None:
    """
    Create <output>/gs_dataset/ — a minimal COLMAP dataset ready for any GS trainer.

    Uses the human-readable text format (cameras.txt / images.txt / points3D.txt)
    which is accepted by all major trainers alongside the binary format.

    Layout:
        gs_dataset/
        ├── images/       → symlink to ../images  (avoids duplicating large frame set)
        └── sparse/
            └── 0/
                ├── cameras.txt
                ├── images.txt
                └── points3D.txt
    """
    import shutil

    src_txt = output_dir / "sparse_txt" / "0"
    gs_dir = output_dir / "gs_dataset"
    gs_sparse = gs_dir / "sparse" / "0"
    gs_sparse.mkdir(parents=True, exist_ok=True)

    # Symlink images — relative path keeps the folder self-contained
    img_link = gs_dir / "images"
    if img_link.exists() or img_link.is_symlink():
        img_link.unlink()
    img_link.symlink_to(Path("../images"))

    # Copy the three COLMAP text files
    for fname in ("cameras.txt", "images.txt", "points3D.txt"):
        src = src_txt / fname
        if src.exists():
            shutil.copy2(src, gs_sparse / fname)

    logger.info(f"GS-ready dataset: {gs_dir}")
    logger.info("  Train with:")
    logger.info(f"    2DGS / 3DGS  : python train.py -s {gs_dir}")
    logger.info(f"    gsplat       : ns-train splatfacto --data {gs_dir}")
    logger.info(f"    OpenSplat    : opensplat {gs_dir} -n 30000")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    t0 = time.time()

    logger.info("GS Initialization Pipeline starting")
    logger.info(f"  video={args.video}  output={output_dir}  device={args.device}")
    logger.info(f"  matcher={args.matcher}  sfm={args.sfm}  depth={args.depth_model}")

    stats = dict(
        frames_extracted=0, frames_after_iqa=0, frames_registered=0,
        sparse_points=0, densified_points=0, mean_reproj=0.0,
        densification_branch="None",
    )

    # ---- Stage 1: Frame Extraction ----------------------------------------
    logger.info("\n=== Stage 1: Frame Extraction ===")
    if args.fps is not None:
        fps = args.fps
    elif args.auto_fps:
        fps = extraction.auto_select_fps(args.video, output_dir)
    else:
        fps = 5.0
        logger.info("Using default 5 fps (pass --auto-fps to probe optimal rate)")
    frames, extraction_skipped = extraction.extract_frames(args.video, output_dir, fps, args.max_frames, args.resize)
    stats["frames_extracted"] = len(frames)
    if extraction_skipped:
        logger.info("Skipping IQA (frames already filtered by prior run)")
        stats["frames_after_iqa"] = len(frames)
    else:
        frames = extraction.filter_blurry_frames(frames, args.iqa_threshold)
        stats["frames_after_iqa"] = len(frames)

    # ---- Stage 2: Feature Extraction and Matching --------------------------
    logger.info("\n=== Stage 2: Feature Extraction and Matching ===")
    feature_path, match_path, pairs_path = features.run_hloc(output_dir, args.matcher)

    # ---- Stage 3: SfM ------------------------------------------------------
    sparse_0 = output_dir / "sparse" / "0"
    if args.skip_sfm and sparse_0.exists() and (sparse_0 / "cameras.bin").exists():
        logger.info("Skipping SfM (--skip-sfm and sparse/0 exists)")
    else:
        logger.info("\n=== Stage 3: SfM (COLMAP) ===")
        sfm.run_sfm(output_dir, args.sfm, feature_path, match_path, pairs_path)

    sfm_stats = sfm.check_quality(sparse_0, stats["frames_after_iqa"])

    # If global mapper got < 80% registration, retry with incremental which is
    # more robust for sequential video with poorly-connected sub-trajectories.
    if (
        not args.skip_sfm
        and args.sfm == "global"
        and sfm_stats["reg_rate"] < 0.80
    ):
        logger.warning(
            f"Global mapper registration too low ({sfm_stats['reg_rate']:.1%}) "
            "— retrying with incremental mapper"
        )
        sfm.retry_with_incremental(output_dir)
        sfm_stats = sfm.check_quality(sparse_0, stats["frames_after_iqa"])

    stats.update(
        frames_registered=sfm_stats["n_registered"],
        sparse_points=sfm_stats["n_points"],
        mean_reproj=sfm_stats["mean_reproj"],
    )

    # ---- Stage 4: Densification --------------------------------------------
    depth_dir = output_dir / "depth"
    densification_done = depth_dir.exists() and any(depth_dir.glob("*.npy"))

    if args.no_densify:
        logger.info("Skipping densification (--no-densify)")
        stats["densification_branch"] = "None (skipped)"
    elif sfm_stats["n_points"] < 1000:
        logger.warning("Too few SfM points — skipping densification")
        stats["densification_branch"] = "None (insufficient SfM points)"
    elif densification_done:
        logger.info("Depth maps found — skipping DAv2 inference, re-running merge/downsample")
        new_pts = densification.reprocess_depth_maps(
            output_dir, sparse_0,
            voxel_size=args.voxel_size, max_points=args.max_points,
            outlier_nb=args.outlier_nb, outlier_std=args.outlier_std,
        )
        if new_pts:
            stats["densified_points"] = len(new_pts)
            densification.replace_points3d(output_dir, sparse_0, new_pts)
        stats["densification_branch"] = "Depth Anything v2 (cached depth maps)"
    else:
        logger.info("\n=== Stage 4: Point Cloud Densification ===")
        depth_model = densification.load_depth_model(args.depth_model, args.device)
        sky_model = densification.load_sky_model(args.device) if args.sky_mask else None

        if args.lidar is not None:
            if args.extrinsics is not None:
                T_cam_to_lidar = np.loadtxt(str(args.extrinsics))
            else:
                logger.warning("--lidar provided without --extrinsics; assuming identity")
                T_cam_to_lidar = np.eye(4)
            stats["densification_branch"] = "LiDAR + Depth Anything v2"
            new_pts = densification.run_lidar(
                output_dir, sparse_0, args.lidar, T_cam_to_lidar, depth_model, args.device,
                min_depth=args.min_depth, sky_model=sky_model, mask_bottom=args.mask_bottom,
                voxel_size=args.voxel_size, max_points=args.max_points,
                outlier_nb=args.outlier_nb, outlier_std=args.outlier_std,
            )
        else:
            stats["densification_branch"] = "Depth Anything v2 (images only)"
            new_pts = densification.run_images_only(
                output_dir, sparse_0, depth_model, args.device,
                min_depth=args.min_depth, sky_model=sky_model, mask_bottom=args.mask_bottom,
                voxel_size=args.voxel_size, max_points=args.max_points,
                outlier_nb=args.outlier_nb, outlier_std=args.outlier_std,
            )

        # Free GPU memory held by inference models before downstream stages
        import torch
        del depth_model
        if sky_model is not None:
            del sky_model
        torch.cuda.empty_cache()

        if new_pts:
            stats["densified_points"] = len(new_pts)
            densification.replace_points3d(output_dir, sparse_0, new_pts)

    # ---- Text export (cameras + images; points3D already written by densification) ----
    sfm.export_text_model(output_dir, sparse_0)

    # ---- PLY export --------------------------------------------------------
    logger.info("\n=== Exporting point cloud ===")
    cameras, images, points3d = read_model(sparse_0)
    ply_path = output_dir / "points3D.ply"
    write_ply(points3d, ply_path)
    logger.info(f"PLY written: {ply_path}  ({len(points3d):,} points)")

    # ---- Validation --------------------------------------------------------
    logger.info("\n=== Validation ===")
    valid = validate_output(output_dir, logger)
    print_summary(stats, output_dir, time.time() - t0, logger)

    # ---- Report ------------------------------------------------------------
    logger.info("\n=== Generating report ===")
    report_path = report_gen.generate(output_dir, cameras, images, points3d, stats)
    logger.info(f"Open in a browser: {report_path}")

    if not valid:
        logger.error("Output validation FAILED")
        sys.exit(1)

    # ---- GS-ready dataset folder ----------------------------------------
    _create_gs_dataset(output_dir, logger)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
