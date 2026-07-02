"""Pipeline HTML report generator."""

import base64
import io
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splatter.utils.geometry import qvec2rotmat

logger = logging.getLogger("gs_init")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate(output_dir: Path, cameras: dict, images: dict,
             points3d: dict, stats: dict) -> Path:
    """
    Build a self-contained HTML report and write it to <output_dir>/report.html.
    Returns the path to the report.
    """
    sections = []

    sections.append(_section_summary(stats))
    sections.append(_section_cameras(images))
    sections.append(_section_reprojection(points3d))
    sections.append(_section_tracks(points3d))
    sections.append(_section_observations(images))
    sections.append(_section_sample_frames(output_dir))
    sections.append(_section_depth_maps(output_dir))

    html = _page("\n".join(sections))
    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _section_summary(stats: dict) -> str:
    n_reg = stats["frames_registered"]
    n_iqa = max(stats["frames_after_iqa"], 1)
    rows = [
        ("Frames extracted",        f"{stats['frames_extracted']:,}"),
        ("Frames after IQA filter", f"{stats['frames_after_iqa']:,}"),
        ("Frames registered",       f"{n_reg:,}  ({n_reg / n_iqa:.1%})"),
        ("Sparse points (SfM)",     f"{stats['sparse_points']:,}"),
        ("Densification branch",    stats["densification_branch"]),
        ("Densified points",        f"{stats['densified_points']:,}"),
        ("Mean reprojection error", f"{stats['mean_reproj']:.2f} px"),
    ]
    trs = "\n".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in rows
    )
    return _card("Pipeline Summary", f"<table class='summary'>{trs}</table>")


def _section_cameras(images: dict) -> str:
    if not images:
        return ""

    centers = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t = np.array(img.tvec)
        centers.append(-R.T @ t)
    centers = np.array(centers)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Top-down (X-Z)
    ax = axes[0]
    sc = ax.scatter(centers[:, 0], centers[:, 2],
                    c=np.arange(len(centers)), cmap="plasma", s=8, alpha=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Camera trajectory — top-down (X-Z)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="frame index")

    # Side view (X-Y)
    ax = axes[1]
    sc = ax.scatter(centers[:, 0], centers[:, 1],
                    c=np.arange(len(centers)), cmap="plasma", s=8, alpha=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Camera trajectory — side view (X-Y)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="frame index")

    fig.tight_layout()
    img_b64 = _fig_to_b64(fig)
    plt.close(fig)

    n = len(centers)
    span = centers.max(axis=0) - centers.min(axis=0)
    info = (f"<p>{n} registered cameras &nbsp;|&nbsp; "
            f"scene span: {span[0]:.1f} × {span[1]:.1f} × {span[2]:.1f} m</p>")
    return _card("Camera Trajectory", info + _img(img_b64))


def _section_reprojection(points3d: dict) -> str:
    if not points3d:
        return ""
    errors = [p.error for p in points3d.values() if p.error > 0]
    if not errors:
        return ""

    errors = np.array(errors)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=100, color="#4c72b0", edgecolor="none", range=(0, min(errors.max(), 5)))
    ax.axvline(errors.mean(), color="crimson", lw=1.5, label=f"mean {errors.mean():.2f} px")
    ax.axvline(np.median(errors), color="orange", lw=1.5,
               linestyle="--", label=f"median {np.median(errors):.2f} px")
    ax.set_xlabel("Reprojection error (px)")
    ax.set_ylabel("# points")
    ax.set_title("Reprojection error distribution")
    ax.legend()
    fig.tight_layout()
    img_b64 = _fig_to_b64(fig)
    plt.close(fig)

    p95 = np.percentile(errors, 95)
    info = (f"<p>mean {errors.mean():.3f} px &nbsp;|&nbsp; "
            f"median {np.median(errors):.3f} px &nbsp;|&nbsp; "
            f"p95 {p95:.3f} px</p>")
    return _card("Reprojection Error", info + _img(img_b64))


def _section_tracks(points3d: dict) -> str:
    if not points3d:
        return ""
    # Densified points from depth unprojection have track_length=0 — exclude them
    lengths = [len(p.point2D_idxs) for p in points3d.values() if len(p.point2D_idxs) > 0]
    if not lengths:
        return ""

    lengths = np.array(lengths)
    n_bins = max(1, min(60, int(lengths.max())))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=n_bins,
            color="#55a868", edgecolor="none",
            range=(1, min(lengths.max() + 1, 50)))
    ax.axvline(lengths.mean(), color="crimson", lw=1.5, label=f"mean {lengths.mean():.1f}")
    ax.set_xlabel("Track length (# images)")
    ax.set_ylabel("# points")
    ax.set_title("Track length distribution")
    ax.legend()
    fig.tight_layout()
    img_b64 = _fig_to_b64(fig)
    plt.close(fig)

    info = (f"<p>mean track length {lengths.mean():.1f} &nbsp;|&nbsp; "
            f"max {lengths.max()}</p>")
    return _card("Track Length Distribution", info + _img(img_b64))


def _section_observations(images: dict) -> str:
    if not images:
        return ""
    obs = sorted(
        [(img.name, int((img.point3D_ids >= 0).sum())) for img in images.values()],
        key=lambda x: x[1]
    )
    counts = np.array([o[1] for o in obs])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(counts)), counts, width=1.0, color="#c44e52", linewidth=0)
    ax.set_xlabel("Registered image index (sorted by observation count)")
    ax.set_ylabel("# 3D observations")
    ax.set_title("3D observations per registered image")
    ax.axhline(counts.mean(), color="navy", lw=1.5, linestyle="--",
               label=f"mean {counts.mean():.0f}")
    ax.legend()
    fig.tight_layout()
    img_b64 = _fig_to_b64(fig)
    plt.close(fig)

    info = (f"<p>mean {counts.mean():.0f} obs/image &nbsp;|&nbsp; "
            f"min {counts.min()} &nbsp;|&nbsp; max {counts.max()}</p>")
    return _card("Observations per Image", info + _img(img_b64))


def _section_sample_frames(output_dir: Path) -> str:
    images_dir = output_dir / "images"
    frames = sorted(images_dir.glob("*.png"))
    if not frames:
        return ""

    # Pick 9 evenly-spaced frames
    idxs = np.linspace(0, len(frames) - 1, 9, dtype=int)
    chosen = [frames[i] for i in idxs]

    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for ax, fp in zip(axes.flat, chosen):
        img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
        # Thumbnail
        h, w = img.shape[:2]
        scale = 400 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ax.imshow(img)
        ax.set_title(fp.name, fontsize=7)
        ax.axis("off")
    fig.suptitle("Sample extracted frames", fontsize=11)
    fig.tight_layout()
    img_b64 = _fig_to_b64(fig)
    plt.close(fig)

    return _card(f"Sample Frames ({len(frames)} total)", _img(img_b64))


def _section_depth_maps(output_dir: Path) -> str:
    depth_dir = output_dir / "depth"
    if not depth_dir.exists():
        return ""
    maps = sorted(depth_dir.glob("*.npy"))
    if not maps:
        return ""

    idxs = np.linspace(0, len(maps) - 1, 9, dtype=int)
    chosen = [maps[i] for i in idxs]

    fig, axes = plt.subplots(3, 3, figsize=(12, 7))
    for ax, fp in zip(axes.flat, chosen):
        dm = np.load(str(fp)).astype(np.float32)
        valid = dm[dm > 0]
        vmin = float(np.percentile(valid, 2)) if len(valid) else 0
        vmax = float(np.percentile(valid, 98)) if len(valid) else 1
        im = ax.imshow(dm, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"{fp.stem}  [{vmin:.1f}–{vmax:.1f} m]", fontsize=7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Sample depth maps (metric, m)", fontsize=11)
    fig.tight_layout()
    img_b64 = _fig_to_b64(fig)
    plt.close(fig)

    return _card(f"Depth Maps ({len(maps)} frames)", _img(img_b64))


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:4px">'


def _card(title: str, body: str) -> str:
    return (
        f'<div class="card">'
        f'<h2>{title}</h2>'
        f'{body}'
        f'</div>'
    )


def _page(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GS Init Pipeline — Report</title>
<style>
  body {{
    font-family: system-ui, sans-serif;
    background: #f4f6f9;
    color: #222;
    margin: 0;
    padding: 24px;
  }}
  h1 {{ text-align: center; margin-bottom: 32px; color: #1a1a2e; }}
  .card {{
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12);
    padding: 20px 24px;
    margin-bottom: 24px;
  }}
  h2 {{ margin-top: 0; font-size: 1.1rem; color: #333; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
  table.summary {{ border-collapse: collapse; width: 100%; max-width: 540px; }}
  table.summary td {{ padding: 6px 12px; border-bottom: 1px solid #f0f0f0; }}
  table.summary tr:last-child td {{ border-bottom: none; }}
  p {{ margin: 6px 0 12px; font-size: .9rem; color: #555; }}
</style>
</head>
<body>
<h1>GS Initialization Pipeline — Report</h1>
{body}
</body>
</html>"""
