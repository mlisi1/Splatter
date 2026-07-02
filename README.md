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
- [Output structure](#output-structure)
- [Training GS models](#training-gs-models)
- [Project layout](#project-layout)

---

## Pipeline overview

| # | Stage | Tool |
|---|-------|------|
| 1 | Frame extraction + blur filter | FFmpeg + Laplacian IQA |
| 2 | Feature extraction + matching | hloc (DISK/ALIKED + LightGlue + NetVLAD) |
| 3 | Structure-from-Motion | COLMAP (global → hierarchical → incremental fallback) |
| 4A | Densification — images only | Depth Anything V2 anchored to SfM sparse cloud |
| 4B | Densification — LiDAR fusion | LiDAR metric scale + Depth Anything V2 infill |

The final output is written to `gs_dataset/` — a text-format COLMAP dataset that all major GS trainers accept without conversion.

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
```

Output lands in `data/output/` on your host. Depth Anything V2 weights (~0.3–1.3 GB) are stored in a named Docker volume and reused across runs.

```bash
docker compose down   # stop the container when done
```

---

## Bare-metal setup

```bash
# System dependencies
sudo apt install ffmpeg colmap   # COLMAP ≥ 4.0 required

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

## Output structure

```
<output>/
├── images/                     # Extracted and filtered frames (000001.png, …)
├── sparse/
│   └── 0/                      # COLMAP sparse reconstruction (binary)
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin        # Replaced with densified cloud after Stage 4
├── sparse_txt/
│   └── 0/                      # Human-readable text copy of the above
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── gs_dataset/                 # Ready-to-train COLMAP dataset — point trainers here
│   ├── images -> ../images     # Symlink (no frame duplication)
│   └── sparse/
│       └── 0/
│           ├── cameras.txt
│           ├── images.txt
│           └── points3D.txt
├── depth/                      # Per-frame metric depth maps (float32 .npy), if densification ran
│   └── 000001.npy, …
├── points3D.ply                # Full densified cloud with RGB (CloudCompare / MeshLab)
├── report.html                 # Self-contained HTML run report
├── database.db                 # COLMAP feature database
└── pipeline.log                # Full run log with parameters and timing
```

---

## Training GS models

Point trainers at `gs_dataset/` — it contains the text-format COLMAP files all major frameworks accept:

```bash
# 2DGS / original 3DGS (INRIA)
python train.py -s ./output/gs_dataset

# gsplat / Nerfstudio
ns-train splatfacto --data ./output/gs_dataset

# OpenSplat
opensplat ./output/gs_dataset -n 30000
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
    └── densification.py   # Stage 4: Depth Anything V2, LiDAR fusion, cloud merge
pipeline.py                # CLI entry point — arg parsing, orchestration, validation, summary
entry_script.sh            # Docker exec entrypoint wrapper
```

To add a stage: create a file under `splatter/stages/` and wire it into `pipeline.py`.
