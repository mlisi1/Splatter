# Splatter — GS Initialization Pipeline

Converts a video file (plus optional LiDAR point cloud) into a **COLMAP sparse reconstruction** ready to train any Gaussian Splatting model (2DGS, 3DGS, gsplat, OpenSplat, …).

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Quick start (Docker)](#quick-start-docker)
- [Bare-metal setup](#bare-metal-setup)
- [Usage examples](#usage-examples)
- [All flags](#all-flags)
- [Extrinsics file format](#extrinsics-file-format)
- [Calibration file format](#calibration-file-format)
- [Telemetry georegistration](#telemetry-georegistration)
- [Output structure](#output-structure)
- [Training GS models](#training-gs-models)
- [Project layout](#project-layout)

---

## Pipeline overview

| # | Stage | Tool |
|---|-------|------|
| 1 | Frame extraction + blur/dedup filter | FFmpeg + Laplacian IQA + sequential near-duplicate filter |
| 1.5 | Known-calibration undistortion (optional) | OpenCV, if `--calibration` is supplied |
| 2 | Feature extraction + matching | hloc (DISK/ALIKED + LightGlue + NetVLAD) |
| 3 | Structure-from-Motion | COLMAP (global → hierarchical → incremental fallback), single shared camera per video |
| 3.5 | Undistortion | COLMAP `image_undistorter` (self-calibrated model → PINHOLE) |
| 3.6 | Telemetry georegistration (optional) | COLMAP `model_aligner`, from DJI SRT sidecar or exiftool-decoded embedded GPS |
| 4A | Densification — images only | Depth Anything V2 anchored to SfM sparse cloud |
| 4B | Densification — LiDAR fusion | LiDAR metric scale + Depth Anything V2 infill |

See `docs/PIPELINE.md` for a full walkthrough of what each stage does and why.

The final output is written to `<output_dir_name>_dataset/` — a text-format COLMAP dataset (real
files, not symlinks, so it's safe to copy elsewhere — e.g. to a Jetson — as a self-contained
folder) that all major GS trainers accept without conversion.

---

## Quick start (Docker)

The Docker image bundles CUDA 12.8, COLMAP 4 compiled for `sm_120` (RTX 5070/5080/5090), and all Python dependencies.

**One-time: install NVIDIA Container Toolkit** (runtime shim — no driver changes):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Build the image** (once, ~20–30 min — COLMAP compiles from source):

```bash
docker compose build
```

**Start the container** (stays alive in the background):

```bash
docker compose up -d
```

**Run the pipeline** — drop input files into `data/input/`, then:

```bash
# Images only
docker compose exec splatter entry_script.sh \
  --video /data/input/mission.mp4 \
  --output /data/output

# With LiDAR and extrinsics
docker compose exec splatter entry_script.sh \
  --video /data/input/mission.mp4 \
  --lidar /data/input/scan.ply \
  --extrinsics /data/input/cam_to_lidar.txt \
  --output /data/output

# DJI/GoPro footage with embedded GPS telemetry — auto-detected, georegisters
# the reconstruction to real-world metric scale without needing LiDAR.
# For DJI, drop the matching DJI_0001.SRT sidecar into data/input/ alongside the video.
docker compose exec splatter entry_script.sh \
  --video /data/input/DJI_0001.MP4 \
  --output /data/output
```

Output lands in `data/output/` on your host. Every model checkpoint this pipeline downloads —
Depth Anything V2 (~0.3–1.3 GB), hloc's NetVLAD/DISK/ALIKED/LightGlue weights, and the SegFormer
sky-mask model (`--sky-mask`) — is cached in a single named Docker volume (`model_cache`, mounted at
`/cache/.cache`) and reused across container restarts: only the very first run of each downloads
anything.

```bash
docker compose down   # stop the container when done
```

---

## Bare-metal setup

```bash
# System dependencies
sudo apt install ffmpeg colmap   # COLMAP ≥ 4.0 required
sudo apt install perl   # Ubuntu's exiftool package is too old for DJI GPS support — install exiftool ≥ 13.0 manually (see CLAUDE.md), optional, only needed for embedded-track telemetry georegistration

# Python dependencies
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch, Open3D, OpenCV, hloc, Depth Anything V2, and SegFormer (for `--sky-mask`). DAv2 model weights are downloaded automatically on first use.

> **GPU:** CUDA is strongly recommended. Depth estimation on CPU is viable but slow (~10× slower).

---

## Usage examples

```bash
# Minimal — images only, no LiDAR
python pipeline.py --video mission.mp4 --output ./output

# Outdoor scene with sky + camera rig visible at the bottom of the frame
# (lower IQA threshold to avoid discarding slightly-blurry frames)
python pipeline.py \
  --video mission.mp4 \
  --iqa-threshold 0.2 \
  --sky-mask \
  --min-depth 1.0 \
  --mask-bottom 0.15 \
  --output ./output

# With LiDAR and known extrinsics
python pipeline.py \
  --video mission.mp4 \
  --lidar scan.ply \
  --extrinsics cam_to_lidar.txt \
  --output ./output

# With a known camera calibration — skips COLMAP self-calibration, undistorts
# right after extraction instead of only before densification
python pipeline.py \
  --video mission.mp4 \
  --calibration cam_calib.txt \
  --output ./output

# DJI drone footage with a DJI_0001.SRT sidecar next to the video — auto-detected,
# georegisters the reconstruction to real-world metric scale without any LiDAR
python pipeline.py \
  --video DJI_0001.MP4 \
  --output ./output

# GoPro (or newer DJI Osmo Action/drone) footage — force the embedded-track path explicitly
python pipeline.py \
  --video GX010001.MP4 \
  --telemetry embedded \
  --output ./output

# ALIKED features — better for low-texture or low-overlap scenes
python pipeline.py --video mission.mp4 --matcher aliked --output ./output

# Skip densification — use raw SfM sparse cloud as-is
python pipeline.py --video mission.mp4 --no-densify --output ./output

# Resize 4K footage to speed up features + depth estimation
python pipeline.py --video mission.mp4 --resize 1280 --output ./output

# Large capture — hierarchical mapper, lighter depth backbone
python pipeline.py \
  --video long_mission.mp4 \
  --sfm hierarchical \
  --depth-model dav2_vitb \
  --max-frames 3000 \
  --output ./output

# Skip SfM if already done (resume after a crash or flag change)
python pipeline.py \
  --video mission.mp4 \
  --skip-sfm \
  --sky-mask \
  --output ./output

# Re-tune point cloud density without re-running depth estimation
# (depth/*.npy already exist → only merge/downsample runs, ~2 min)
python pipeline.py \
  --video mission.mp4 \
  --skip-sfm \
  --voxel-size 0.03 \
  --max-points 3000000 \
  --outlier-std 2.5 \
  --output ./output
```

---

## All flags

### Inputs

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | *(required)* | Input video (MP4, MOV, MKV, AVI) |
| `--lidar` | — | LiDAR point cloud in PLY format (XYZ or XYZI). Enables LiDAR-assisted densification |
| `--extrinsics` | — | 4×4 `T_cam_to_lidar` matrix (plain text, row-major, metres). If omitted with `--lidar`, identity is assumed and a warning is printed |
| `--calibration` | — | Known camera calibration file (model, WxH, params). If supplied, frames are undistorted right after extraction (Stage 1.5) instead of relying on COLMAP self-calibration |
| `--telemetry` | `auto` | `auto`/`dji`/`embedded`/`off` — georegister the reconstruction to real-world metric scale using GPS telemetry embedded in the video (DJI SRT sidecar, or an embedded GPS track decoded via exiftool — GoPro GPMF, DJI's newer protobuf `djmd` track, Garmin VIRB, etc.). Ignored when `--lidar` is supplied |
| `--output` | `./output` | Output directory |
| `--device` | `cuda` | Compute device: `cuda` or `cpu` |

### Frame extraction

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | 5 | Frame extraction rate |
| `--auto-fps` | off | Probe 5 / 7 / 10 fps on a 30 s clip and pick the minimum rate that achieves full registration |
| `--max-frames` | 2000 | Hard cap on extracted frames |
| `--resize` | 0 (full res) | Resize frames so the longer edge is this many pixels before any processing |
| `--iqa-threshold` | 0.5 | Blur rejection threshold: 0 = keep all frames, 1 = keep only the sharpest. **Warning:** 0.5 is very aggressive and can discard 80%+ of frames on outdoor or robot captures. Use 0.2 or lower, or 0 to let SfM reject unregisterable frames naturally |
| `--dedup-threshold` | 0 (off) | Sequential near-duplicate rejection, ~0–1. Collapses a run of near-identical frames (e.g. a robot/camera sitting still) to a handful by comparing each frame only to the last one kept — a later revisit of an earlier viewpoint is unaffected. No universally safe non-zero default: start around 0.02 |

### Feature matching & SfM

| Flag | Default | Description |
|------|---------|-------------|
| `--matcher` | `disk` | Local feature extractor: `disk` or `aliked` (ALIKED is better for low-texture / low-overlap scenes) |
| `--sfm` | `global` | COLMAP mapper: `global`, `hierarchical`, `incremental`. Global is fastest; if it achieves < 80% registration the pipeline automatically retries with incremental |
| `--skip-sfm` | off | Skip SfM if `sparse/0/` already exists (e.g. when resuming after a crash) |

### Densification

| Flag | Default | Description |
|------|---------|-------------|
| `--depth-model` | `dav2_vitl` | Depth Anything V2 backbone: `dav2_vitl` (best), `dav2_vitb` (balanced), `dav2_vits` (fastest) |
| `--min-depth` | 0 (off) | Zero depth-map pixels closer than this many metres. Useful to exclude near-field objects, though `--mask-bottom` is more reliable for a camera rig that always occupies the bottom of the frame |
| `--sky-mask` | off | Run SegFormer-B0 semantic sky segmentation and zero sky pixels before unprojection. Downloads ~14 MB on first use. Recommended for outdoor scenes |
| `--mask-bottom` | 0 (off) | Fraction of image height (0.0–1.0) to zero at the bottom of every depth map. Use to exclude a camera rig or robot body in a fixed bottom strip (e.g. `--mask-bottom 0.15`) |
| `--voxel-size` | 0.02 | Initial voxel size (m) for point cloud downsampling. Larger = fewer points, faster. Tries 1.5× and 2.5× automatically if result still exceeds 10 M points |
| `--max-points` | 5000000 | Target maximum points in the final cloud. A secondary voxel pass runs if still above this after outlier removal |
| `--outlier-std` | 3.0 | `std_ratio` for statistical outlier removal. Higher = keep more points (less aggressive) |
| `--outlier-nb` | 20 | Number of neighbours for statistical outlier removal |
| `--no-densify` | off | Skip densification entirely; use the raw SfM sparse cloud |

---

## Extrinsics file format

Plain text, 4×4 homogeneous matrix `T_cam_to_lidar`, row-major, space-separated. Units: metres.

```
r00 r01 r02 tx
r10 r11 r12 ty
r20 r21 r22 tz
0   0   0   1
```

Identity example (camera and LiDAR co-located):

```
1.0 0.0 0.0 0.0
0.0 1.0 0.0 0.0
0.0 0.0 1.0 0.0
0.0 0.0 0.0 1.0
```

---

## Calibration file format

Plain text, 3 lines: COLMAP camera model (`OPENCV` or `FULL_OPENCV`), the resolution the
calibration was measured at, then space-separated params in COLMAP order — identical to
OpenCV's own `cameraMatrix`/`distCoeffs` convention, so no reordering is needed for calibration
data from `cv2.calibrateCamera`.

```
OPENCV
1920 1080
1200.0 1200.0 960.0 540.0 -0.18 0.05 0.0002 -0.0001
```

`fx, fy, cx, cy` are scaled automatically to match the actual extracted frame size (so this
works unchanged with `--resize`); `k1, k2, p1, p2[, k3..]` are resolution-invariant. See
`CLAUDE.md` for the full param-order reference.

---

## Telemetry georegistration

`--telemetry {auto,dji,embedded,off}` rescales/reorients the reconstruction into real-world ENU
metres using GPS already embedded in the video — a DJI drone's `<video>.srt` sidecar, or an
embedded GPS track read via `exiftool` (GoPro GPMF, DJI's newer protobuf `djmd` track used by
drones/Osmo Action, Garmin VIRB, Insta360, etc.) — instead of relying on `--lidar` to ground the
otherwise scale-ambiguous monocular reconstruction. `auto` (default) detects whichever is
present; the stage is always best-effort (a warning and no change to `sparse/0` if telemetry
can't be found or matched, never a hard failure) and is skipped automatically when `--lidar` is
supplied. **Requires exiftool ≥ 13.0** for the embedded-track path — see `CLAUDE.md`'s Docker
deployment notes; Ubuntu's packaged exiftool (12.76) predates DJI protobuf GPS support and will
silently find nothing on a DJI `djmd` track. See `docs/PIPELINE.md` (Stage 3.6) and `CLAUDE.md`
for the full mechanics.

---

## Output structure

```
<output>/
├── images -> <name>_dataset/images  # Relative symlink; real files live in <name>_dataset/ (see below)
├── images_distorted/           # Original distorted frames, kept as a backup by Stage 3.5
├── sparse/
│   ├── 0/                      # COLMAP sparse reconstruction (binary), PINHOLE after Stage 3.5
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin        # Replaced with densified cloud after Stage 4
│   ├── _distorted_backup/      # Pre-undistortion model (OPENCV/RADIAL), kept by Stage 3.5
│   └── _pre_georegister_backup/ # Pre-alignment model, kept by Stage 3.6 if telemetry georegistration ran
├── telemetry_ref_images.txt    # image_name lat lon alt used to fit the Stage 3.6 alignment, if it ran
├── telemetry_transform.txt     # Similarity transform applied by Stage 3.6, if it ran
├── sparse_txt/
│   └── 0/                      # Human-readable text copy of the above
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── <name>_dataset/              # Ready-to-train COLMAP dataset, named after the output dir itself
│   │                             # (e.g. --output ./output/crosslab1 → crosslab1_dataset/) — point
│   │                             # trainers here; every file inside is real (no symlinks), so this
│   │                             # folder alone is safe to copy elsewhere (e.g. to a Jetson)
│   ├── images/                  # Real directory, moved (not copied) from <output>/images/
│   └── sparse/
│       └── 0/
│           ├── cameras.txt
│           ├── images.txt
│           └── points3D.txt
├── depth/                      # Per-frame metric depth maps (float32 .npy), if densification ran
│   └── 000001.npy, …
├── points3D.ply                # Full densified cloud with RGB (CloudCompare / MeshLab)
├── report.html                 # Self-contained HTML run report
├── config.json                 # CLI args used for this run + frames_extracted/after_iqa/after_dedup counts
├── database.db                 # COLMAP feature database
└── pipeline.log                # Full run log with parameters and timing (ends with total elapsed time)
```

---

## Training GS models

Point trainers at `<output>/<name>_dataset/` (e.g. `./output/crosslab1/crosslab1_dataset/`) — it
contains the text-format COLMAP files all major frameworks accept, with real (non-symlinked) image
files, so the folder is also what you'd `scp`/`rsync` to a training box (e.g. a Jetson):

```bash
# 2DGS / original 3DGS (INRIA)
python train.py -s ./output/crosslab1/crosslab1_dataset

# gsplat / Nerfstudio
ns-train splatfacto --data ./output/crosslab1/crosslab1_dataset

# OpenSplat
opensplat ./output/crosslab1/crosslab1_dataset -n 30000
```

---

## Project layout

```
splatter/
├── io/
│   └── colmap.py          # COLMAP binary/text readers & writers; Camera/Image/Point3D types
├── utils/
│   ├── geometry.py        # qvec→rotmat, intrinsics, depth unprojection, LO-RANSAC
│   └── logging.py         # Console + file logger setup
└── stages/
    ├── extraction.py      # Stage 1: FFmpeg extraction, auto-FPS, IQA blur filter
    ├── features.py        # Stage 2: hloc (DISK/ALIKED + NetVLAD + LightGlue)
    ├── sfm.py             # Stage 3: COLMAP mapper with fallback chain
    ├── undistort.py       # Stage 1.5 (known calibration) + Stage 3.5 (self-calibrated) undistortion
    ├── telemetry.py       # Stage 3.6: GPS georegistration from DJI SRT sidecar or exiftool-decoded embedded track
    └── densification.py   # Stage 4: Depth Anything V2, LiDAR fusion, cloud merge
pipeline.py                # CLI entry point — arg parsing, orchestration, validation, summary
entry_script.sh            # Docker exec entrypoint wrapper
```

To add a stage: create a file under `splatter/stages/` and wire it into `pipeline.py`.
