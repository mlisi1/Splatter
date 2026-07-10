#!/usr/bin/env python3
"""GS Initialization Pipeline — CLI entry point."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from splatter.io.colmap import read_cameras_binary, read_images_binary, read_points3d_binary
from splatter.io.ply import write_ply
from splatter.stages import coverage, densification, extraction, features, gs_params, orientation, sfm, telemetry, undistort
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
    p.add_argument("--calibration", type=Path, default=None,
                   help="Known camera calibration file (model, WxH, params — see CLAUDE.md). "
                        "If supplied, frames are undistorted right after extraction using "
                        "this calibration instead of relying on COLMAP self-calibration.")
    p.add_argument("--telemetry", choices=["auto", "dji", "embedded", "off"], default="auto",
                   help="Georegister the reconstruction to real-world metric scale using GPS "
                        "telemetry embedded in the video: a DJI SRT sidecar, or an embedded GPS "
                        "track decoded via exiftool (GoPro GPMF, DJI's newer protobuf djmd "
                        "track used by drones/Osmo Action, Garmin VIRB, Insta360, etc.). "
                        "'auto' detects whichever is present; 'off' disables this stage. "
                        "Ignored (skipped) when --lidar is supplied — LiDAR already provides "
                        "metric scale.")
    p.add_argument("--align-up", choices=["auto", "off"], default="auto",
                   help="Rotate the reconstruction so its true vertical direction (estimated "
                        "from vanishing-point detection on the frames) matches world -Y. "
                        "Monocular SfM has no constraint forcing this — the arbitrary bundle-"
                        "adjustment rotation only lands close to -Y-up by coincidence for a "
                        "roughly level capture, leaving a residual skew. Pure rotation, no "
                        "rescaling, so it runs regardless of --lidar. 'auto' (default) skips "
                        "itself when Stage 3.6 telemetry georegistration already succeeded "
                        "(real GPS-derived orientation beats this heuristic); 'off' disables it.")
    p.add_argument("--output", type=Path, default=Path("./output"), help="Output directory")
    p.add_argument("--fps", type=float, default=None,
                   help="Frame extraction rate (default: 5)")
    p.add_argument("--auto-fps", action="store_true",
                   help="Probe 5/7/10 fps on a 30s clip to find the minimum rate that achieves full registration")
    p.add_argument("--max-frames", type=int, default=10000,
                   help="Hard cap on extracted frames. If ffmpeg extraction produces more "
                        "than this, the temporal tail is dropped (frames are numbered in "
                        "capture order, so this keeps only the first N seconds of footage, "
                        "not an evenly spaced subsample) — lower --fps or use --auto-fps "
                        "instead of relying on this cap to thin a long video.")
    p.add_argument("--iqa-threshold", type=float, default=0.5,
                   help="Blur rejection threshold 0–1 (0 = keep all)")
    p.add_argument("--dedup-threshold", type=float, default=0.0,
                   help="Sequential near-duplicate rejection, ~0-1 (0 = off, default). "
                        "Collapses a run of near-identical frames (e.g. camera/robot "
                        "sitting still) to a handful by comparing each frame only to "
                        "the last frame kept -- a later revisit of an earlier viewpoint "
                        "is unaffected. No universally safe non-zero default: start "
                        "around 0.02 and increase if static holds still aren't "
                        "collapsing, decrease (or leave at 0) if a low-texture scene "
                        "(long corridor, tiled floor) starts losing distinct viewpoints.")
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
    p.add_argument("--cleanup", action="store_true",
                   help="After the pipeline finishes, delete intermediate artifacts "
                        "(sparse/, sparse_txt/, database.db, feature/match caches, "
                        "pairs files, depth/, images_distorted/) leaving only "
                        "<name>_dataset/, config.json, report.html, and points3D.ply. "
                        "Irreversible: --skip-sfm and cached-depth reprocessing can no "
                        "longer resume this output directory afterward.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(output_dir: Path, logger, images: dict, n_points: int) -> bool:
    """
    images/n_points are passed in from the caller's already-loaded model
    rather than re-read from sparse_0 here — points3D.bin can hold millions
    of entries for a densified cloud, and the caller already has both loaded
    for report generation, so a second full read (a third copy of the point
    cloud alongside `new_pts` and the caller's own `points3d`) was pure
    memory pressure for a check that only needs a count.
    """
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

    if len(images) < 20:
        logger.error(f"Only {len(images)} registered images (need ≥ 20)")
        ok = False
    if n_points < 1000:
        logger.error(f"Only {n_points} 3D points (need ≥ 1000)")
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
    n_dedup = stats["frames_after_dedup"]
    reg_pct = n_reg / max(n_dedup, 1) * 100

    msg = (
        "\n========================================"
        "\n GS Initialization Pipeline — Summary"
        "\n========================================"
        f"\n Frames extracted       : {stats['frames_extracted']:,}"
        f"\n Frames after IQA filter: {n_iqa:,}"
        f"\n Frames after dedup     : {n_dedup:,}"
        f"\n Frames registered      : {n_reg:,}  ({reg_pct:.1f}%)"
        f"\n Sparse points (SfM)    : {stats['sparse_points']:,}"
        f"\n Georegistration        : {stats['georegistration']}"
        f"\n Up-axis aligned        : {stats['up_axis_aligned']}"
        f"\n Densification branch   : {stats['densification_branch']}"
        f"\n Densified points       : {stats['densified_points']:,}"
        f"\n Mean reprojection error: {stats['mean_reproj']:.2f} px"
        f"\n Output directory       : {output_dir}"
        f"\n Total runtime          : {mins}m {secs}s"
        "\n========================================"
    )
    print(msg)
    logger.info(msg)


def _write_pipeline_config(output_dir: Path, args: argparse.Namespace, stats: dict) -> None:
    """
    Persist the CLI args used for this invocation, plus the true Stage 1
    frame-count stats, to <output>/config.json — lets a user check exactly
    what parameters produced a given reconstruction.

    The frame counts are the reason this is written on *every* invocation,
    not just the first: on a resumed run (extraction skipped because
    images/ already exists) they're read back via _read_prior_extraction_stats
    rather than re-derived from images/'s current contents, which by then
    already reflect post-IQA/dedup/undistortion filtering.
    """
    config_path = output_dir / "config.json"
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    data = {
        "args": args_dict,
        "frames_extracted": stats["frames_extracted"],
        "frames_after_iqa": stats["frames_after_iqa"],
        "frames_after_dedup": stats["frames_after_dedup"],
    }
    config_path.write_text(json.dumps(data, indent=2))


def _read_prior_extraction_stats(output_dir: Path) -> dict | None:
    """Read back frames_extracted/frames_after_iqa/frames_after_dedup from a
    prior run's config.json, or None if it doesn't exist / is unreadable."""
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text())
        return {
            "frames_extracted": data["frames_extracted"],
            "frames_after_iqa": data["frames_after_iqa"],
            "frames_after_dedup": data["frames_after_dedup"],
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _create_dataset_folder(output_dir: Path, logger) -> Path:
    """
    Create <output>/<output_dir.name>_dataset/ — a self-contained COLMAP
    dataset ready for any GS trainer, with real (non-symlinked) image files
    so the whole folder can be copied elsewhere (e.g. to a Jetson for
    training) without leaving behind a symlink that breaks once separated
    from its target.

    images/ is *moved* (not copied) into the dataset folder — still only one
    copy of the frame set on disk — and a relative symlink is left behind at
    <output>/images pointing into the dataset folder, so every other part of
    the pipeline that expects <output>/images to exist (extract_frames'
    resume check, undistort_images' idempotency gate, report.html thumbnails
    on a later --skip-sfm run) keeps working unmodified; the symlink itself
    lives outside the dataset folder, so copying the dataset folder alone
    never carries a symlink with it. Idempotent: if <output>/images is
    already a symlink (the move already happened on a prior run against this
    output dir), the move step is skipped and only the sparse/*.txt files
    (which may have changed, e.g. after a densification re-tune) are refreshed.

    Layout:
        <name>_dataset/
        ├── images/       real directory (moved from <output>/images/)
        └── sparse/
            └── 0/
                ├── cameras.txt
                ├── images.txt
                └── points3D.txt
    """
    import shutil

    src_txt = output_dir / "sparse_txt" / "0"
    dataset_dir = output_dir / f"{output_dir.name}_dataset"
    dataset_sparse = dataset_dir / "sparse" / "0"
    dataset_sparse.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    dataset_images = dataset_dir / "images"

    if images_dir.is_symlink():
        pass  # already moved by a prior run against this output dir
    elif images_dir.is_dir():
        if dataset_images.exists():
            shutil.rmtree(dataset_images)
        images_dir.rename(dataset_images)
        images_dir.symlink_to(Path(f"{dataset_dir.name}/images"))

    # Copy the three COLMAP text files
    for fname in ("cameras.txt", "images.txt", "points3D.txt"):
        src = src_txt / fname
        if src.exists():
            shutil.copy2(src, dataset_sparse / fname)

    logger.info(f"GS-ready dataset: {dataset_dir}")
    logger.info("  Train with:")
    logger.info(f"    2DGS / 3DGS  : python train.py -s {dataset_dir}")
    logger.info(f"    gsplat       : ns-train splatfacto --data {dataset_dir}")
    logger.info(f"    OpenSplat    : opensplat {dataset_dir} -n 30000")
    return dataset_dir


def _cleanup_intermediate_files(output_dir: Path, logger) -> None:
    """
    Delete everything in output_dir except the GS-ready dataset folder,
    config.json, report.html, and points3D.ply.

    Irreversible: removes the SfM sparse model, feature/match caches, and
    cached depth maps that --skip-sfm and depth-map reprocessing rely on to
    resume this output directory, so run it only once you're done tuning.
    """
    import shutil

    keep = {
        f"{output_dir.name}_dataset",
        "config.json", "report.html", "points3D.ply", "pipeline.log",
        "telemetry_ref_images.txt", "telemetry_transform.txt",
        "2dgs_suggested_params.txt", "coverage_report.txt",
    }

    freed = 0
    for entry in output_dir.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir() and not entry.is_symlink():
            freed += sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            shutil.rmtree(entry)
        else:
            freed += entry.stat().st_size if entry.exists() else 0
            entry.unlink()

    logger.info(f"--cleanup: removed intermediate artifacts ({freed / 1e6:.1f} MB freed)")


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
        frames_extracted=0, frames_after_iqa=0, frames_after_dedup=0, frames_registered=0,
        sparse_points=0, densified_points=0, mean_reproj=0.0,
        densification_branch="None", georegistration="None", up_axis_aligned="No",
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
        prior = _read_prior_extraction_stats(output_dir)
        if prior is not None:
            logger.info(
                "Skipping IQA + dedup (frames already filtered by prior run) — "
                f"restoring true frame counts from config.json: "
                f"extracted={prior['frames_extracted']}, after_iqa={prior['frames_after_iqa']}, "
                f"after_dedup={prior['frames_after_dedup']}"
            )
            stats["frames_extracted"] = prior["frames_extracted"]
            stats["frames_after_iqa"] = prior["frames_after_iqa"]
            stats["frames_after_dedup"] = prior["frames_after_dedup"]
        else:
            logger.warning(
                "Skipping IQA + dedup (frames already filtered by prior run) — no "
                "config.json from a prior run found, so frame-count stats will reflect "
                "the current images/ folder, not the true original extraction/IQA counts"
            )
            stats["frames_after_iqa"] = len(frames)
            stats["frames_after_dedup"] = len(frames)
    else:
        frames = extraction.filter_blurry_frames(frames, args.iqa_threshold)
        stats["frames_after_iqa"] = len(frames)
        frames = extraction.filter_duplicate_frames(frames, args.dedup_threshold)
        stats["frames_after_dedup"] = len(frames)

    _write_pipeline_config(output_dir, args, stats)

    # ---- Stage 1.5: Known-calibration undistortion (optional) --------------
    if args.calibration is not None:
        logger.info("\n=== Stage 1.5: Known-calibration undistortion ===")
        undistort.undistort_known_calibration(output_dir / "images", args.calibration, output_dir)

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

    sfm_stats = sfm.check_quality(sparse_0, stats["frames_after_dedup"])

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
        sfm_stats = sfm.check_quality(sparse_0, stats["frames_after_dedup"])

    stats.update(
        frames_registered=sfm_stats["n_registered"],
        sparse_points=sfm_stats["n_points"],
        mean_reproj=sfm_stats["mean_reproj"],
    )

    # ---- Stage 3.5: Undistortion --------------------------------------------
    logger.info("\n=== Stage 3.5: Undistortion ===")
    undistort.undistort_images(output_dir, sparse_0)

    # ---- Stage 3.6: Georegistration from telemetry (optional) --------------
    telemetry_source = None
    if args.lidar is not None:
        if args.telemetry != "off":
            logger.info(
                "\n=== Stage 3.6: Georegistration (telemetry) ===\n"
                "Skipping — --lidar already provides metric scale"
            )
    else:
        logger.info("\n=== Stage 3.6: Georegistration (telemetry) ===")
        telemetry_source = telemetry.run(output_dir, sparse_0, args.video, fps, mode=args.telemetry)
        if telemetry_source:
            stats["georegistration"] = telemetry_source

    # ---- Stage 3.7: Up-axis alignment (optional) ----------------------------
    if args.align_up == "off":
        logger.info("\n=== Stage 3.7: Up-axis alignment ===\nSkipping (--align-up off)")
    elif telemetry_source:
        logger.info(
            "\n=== Stage 3.7: Up-axis alignment ===\n"
            "Skipping — telemetry georegistration already grounded orientation to real GPS"
        )
    else:
        logger.info("\n=== Stage 3.7: Up-axis alignment ===")
        if orientation.align_up_axis(output_dir, sparse_0):
            stats["up_axis_aligned"] = "Yes"

    # ---- Stage 4: Densification --------------------------------------------
    depth_dir = output_dir / "depth"
    densification_done = depth_dir.exists() and any(depth_dir.glob("*.npy"))
    ply_path = output_dir / "points3D.ply"
    ply_already_exists = ply_path.exists()

    new_pts = None  # only set when densification actually ran and produced points
    if ply_already_exists:
        logger.info(
            f"Skipping densification — {ply_path} already exists from a prior "
            "completed run (delete it, or the whole output dir, to force re-densification)"
        )
        stats["densification_branch"] = "Skipped (points3D.ply already exists)"
    elif args.no_densify:
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
    cameras = read_cameras_binary(sparse_0 / "cameras.bin")
    images = read_images_binary(sparse_0 / "images.bin")
    if new_pts:
        # Reuse the in-memory densified cloud instead of re-reading points3D.bin
        # (which replace_points3d() just wrote from this same dict) — holding both
        # the original and a freshly re-parsed copy at once is what pushed RAM to
        # 96%+ around this stage on a large densified cloud.
        points3d = new_pts
        new_pts = None
    else:
        points3d = read_points3d_binary(sparse_0 / "points3D.bin")
    if ply_already_exists:
        stats["densified_points"] = len(points3d)
        logger.info(f"PLY already exists: {ply_path}  ({len(points3d):,} points) — not rewriting")
    else:
        write_ply(points3d, ply_path)
        logger.info(f"PLY written: {ply_path}  ({len(points3d):,} points)")

    # ---- 2DGS suggested training parameters (heuristic) --------------------
    gs_suggestion = gs_params.suggest_2dgs_params(images, points3d)
    if gs_suggestion:
        logger.info(
            f"2DGS suggestion: {gs_suggestion['scene_type']} — {gs_suggestion['command']}"
        )
        gs_params.write_suggestion_file(output_dir, gs_suggestion)

    # ---- Image coverage diagnostic (heuristic) ------------------------------
    sfm_points3d = coverage.load_sfm_points3d(sparse_0, points3d)
    coverage_stats = coverage.analyze_coverage(images, sfm_points3d)
    if coverage_stats:
        logger.info(f"Coverage diagnostic: {coverage_stats['verdict']} ({coverage_stats['reasoning']})")
        if coverage_stats["verdict"] != "Good — coverage looks adequate":
            logger.warning(
                "Image coverage may be too sparse for good GS training — see coverage_report.txt"
            )
        coverage.write_coverage_file(output_dir, coverage_stats)

    # ---- Validation --------------------------------------------------------
    logger.info("\n=== Validation ===")
    valid = validate_output(output_dir, logger, images, len(points3d))
    print_summary(stats, output_dir, time.time() - t0, logger)

    # ---- Report ------------------------------------------------------------
    logger.info("\n=== Generating report ===")
    report_path = report_gen.generate(output_dir, cameras, images, points3d, stats, gs_suggestion,
                                       sfm_points3d, coverage_stats)
    logger.info(f"Open in a browser: {report_path}")

    if not valid:
        logger.error("Output validation FAILED")
        sys.exit(1)

    # ---- GS-ready dataset folder ----------------------------------------
    _create_dataset_folder(output_dir, logger)

    # ---- Cleanup (optional) ------------------------------------------------
    if args.cleanup:
        logger.info("\n=== Cleanup ===")
        _cleanup_intermediate_files(output_dir, logger)

    total_elapsed = time.time() - t0
    mins, secs = int(total_elapsed // 60), int(total_elapsed % 60)
    logger.info(f"Pipeline complete in {mins}m {secs}s.")


if __name__ == "__main__":
    main()
