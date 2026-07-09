"""Stage 1 — Frame extraction and blur filtering."""

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from splatter.io.colmap import read_images_binary

logger = logging.getLogger("gs_init")


def auto_select_fps(video_path: Path, output_dir: Path) -> float:
    """
    Test FPS values [5, 7, 10] on a 30-second clip using COLMAP feature
    extraction + sequential matching. Return the first FPS that achieves
    ≥ 99 % registration, defaulting to 5 if none reach that threshold.
    """
    sample_dir = output_dir / "_fps_test"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for fps in (5, 7, 10):
        test_dir = sample_dir / f"imgs_{fps}"
        test_dir.mkdir(exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-t", "30",
             "-vsync", "vfr", "-q:v", "2", "-vf", f"fps={fps}",
             str(test_dir / "%06d.png"), "-y"],
            capture_output=True,
        )
        if r.returncode != 0:
            continue
        frames = list(test_dir.glob("*.png"))
        if not frames:
            continue

        test_db = sample_dir / f"test_{fps}.db"
        test_sparse = sample_dir / f"sparse_{fps}"
        test_sparse.mkdir(exist_ok=True)
        try:
            subprocess.run(
                ["colmap", "feature_extractor",
                 "--database_path", str(test_db),
                 "--image_path", str(test_dir),
                 "--ImageReader.camera_model", "OPENCV"],
                capture_output=True, timeout=120,
            )
            subprocess.run(
                ["colmap", "sequential_matcher",
                 "--database_path", str(test_db)],
                capture_output=True, timeout=120,
            )
            subprocess.run(
                ["colmap", "mapper",
                 "--database_path", str(test_db),
                 "--image_path", str(test_dir),
                 "--output_path", str(test_sparse)],
                capture_output=True, timeout=300,
            )
            sparse_0 = test_sparse / "0"
            if sparse_0.exists() and (sparse_0 / "images.bin").exists():
                imgs = read_images_binary(sparse_0 / "images.bin")
                rate = len(imgs) / max(len(frames), 1)
                logger.info(f"FPS {fps}: registration {rate:.1%}")
                if rate >= 0.99:
                    shutil.rmtree(sample_dir, ignore_errors=True)
                    logger.info(f"Auto-selected FPS: {fps}")
                    return float(fps)
        except Exception as e:
            logger.debug(f"FPS {fps} test error: {e}")

    shutil.rmtree(sample_dir, ignore_errors=True)
    logger.info("Auto FPS: defaulting to 5")
    return 5.0


def extract_frames(
    video_path: Path, output_dir: Path, fps: float, max_frames: int,
    resize: int = 0,
) -> tuple[list[Path], bool]:
    """
    Extract frames from *video_path* at *fps* into <output_dir>/images/.

    If *resize* > 0, each frame is resized so its longer edge equals *resize*
    pixels (aspect ratio preserved) before being saved.

    Returns (frames, already_done). When already_done is True the images
    directory was populated by a prior run and IQA should be skipped.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(images_dir.glob("*.png"))
    if existing:
        logger.info(f"Found {len(existing)} existing frames — skipping extraction")
        return existing[:max_frames], True

    # Build FFmpeg vf filter: fps + optional scale
    vf = f"fps={fps}"
    if resize > 0:
        # scale so the longer edge = resize, keep aspect ratio, round to even
        vf += f",scale='if(gt(iw,ih),{resize},-2)':'if(gt(iw,ih),-2,{resize})'"
        logger.info(f"Extracting frames at {fps} fps (resized to max {resize}px)")
    else:
        logger.info(f"Extracting frames at {fps} fps (full resolution)")

    subprocess.run(
        ["ffmpeg", "-i", str(video_path),
         "-vsync", "vfr", "-q:v", "1", "-vf", vf,
         str(images_dir / "%06d.png"), "-y"],
        check=True, capture_output=True,
    )

    frames = sorted(images_dir.glob("*.png"))
    logger.info(f"Extracted {len(frames)} frames")

    if len(frames) > max_frames:
        for f in frames[max_frames:]:
            f.unlink()
        frames = frames[:max_frames]
        logger.info(f"Capped to {max_frames} frames")

    return frames, False


def filter_blurry_frames(frames: list[Path], threshold: float) -> list[Path]:
    """
    Score each frame with Laplacian variance, normalise to [0, 1], and
    delete frames whose normalised score is below *threshold*.

    threshold=0 keeps everything; threshold=1 keeps only the sharpest frame.
    """
    if threshold <= 0:
        return frames

    scores = []
    for f in tqdm(frames, desc="Blur scoring"):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        scores.append(cv2.Laplacian(img, cv2.CV_64F).var() if img is not None else 0.0)

    scores = np.array(scores, dtype=np.float64)
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-12)

    keep, n_discard = [], 0
    for f, ns in zip(frames, norm):
        if ns >= threshold:
            keep.append(f)
        else:
            f.unlink()
            n_discard += 1

    logger.info(f"IQA: discarded {n_discard} frames, kept {len(keep)}")
    return keep


def filter_duplicate_frames(frames: list[Path], threshold: float, max_gap: int = 15) -> list[Path]:
    """
    Greedily walk frames in temporal order and drop a frame if it is
    near-identical to the most recently *kept* frame, collapsing static
    holds (e.g. a robot/camera sitting still) down to a handful of frames.

    Each frame is downsampled to 64x64 grayscale and mean-centered (its own
    per-frame average subtracted) before differencing, so the comparison is
    on local structure rather than absolute brightness -- this keeps slow
    auto-exposure/white-balance drift or flicker during a static hold from
    being misread as a genuinely new viewpoint.

    threshold<=0 keeps everything (default/off). Unlike filter_blurry_frames,
    there is no batch-wide min-max renormalisation -- the score is always
    relative to the single last-kept frame, so its meaning doesn't depend
    on what else is in the batch.

    Guards against slow continuous pans/dollies -- where every consecutive
    raw frame is individually near-identical to its neighbour but drift
    accumulates real parallax over many frames -- by always keeping at
    least one frame out of every `max_gap` consecutive raw frames.
    """
    if threshold <= 0 or len(frames) <= 1:
        return frames

    def _descriptor(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64)
        return small - small.mean()

    keep = [frames[0]]
    last_desc = _descriptor(frames[0])
    since_kept, n_discard = 0, 0

    for f in tqdm(frames[1:], desc="Dedup scoring"):
        desc = _descriptor(f)
        score = np.abs(desc - last_desc).mean() / 255.0
        since_kept += 1
        if score >= threshold or since_kept >= max_gap:
            keep.append(f)
            last_desc = desc
            since_kept = 0
        else:
            f.unlink()
            n_discard += 1

    logger.info(f"Dedup: discarded {n_discard} frames, kept {len(keep)}")
    return keep
