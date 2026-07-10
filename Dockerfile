FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System dependencies ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential pkg-config wget \
    python3 python3-pip python3-dev python3-venv \
    # COLMAP build deps
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev \
    libeigen3-dev libflann-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev libgflags-dev \
    libsqlite3-dev libglew-dev libopenblas-dev \
    qtbase5-dev libqt5opengl5-dev libqt5svg5-dev \
    libceres-dev libopenimageio-dev openimageio-tools libopencv-dev \
    # Pipeline runtime deps
    ffmpeg \
    perl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ExifTool — installed from upstream instead of apt's libimage-exiftool-perl,
# which on Ubuntu 24.04 is pinned to 12.76. Stage 3.6's telemetry georegistration
# needs 13.0+: that's when DJI's protobuf-encoded "djmd" video metadata track
# (used by current-gen drones and Osmo Action cameras) was added to exiftool's
# DJI module — an older exiftool runs without error but silently decodes zero
# GPS samples from that track. Pure Perl, no build step; only needs `perl` above.
RUN wget -q -O /tmp/exiftool.tar.gz \
        "https://sourceforge.net/projects/exiftool/files/Image-ExifTool-13.59.tar.gz/download" \
    && tar xzf /tmp/exiftool.tar.gz -C /opt \
    && ln -s /opt/Image-ExifTool-13.59/exiftool /usr/local/bin/exiftool \
    && rm /tmp/exiftool.tar.gz

# ---- Build COLMAP from source (CUDA 12.8, sm_120 for RTX 5070) ------------
RUN git clone --recursive --depth 1 https://github.com/colmap/colmap.git /opt/colmap-src

RUN cmake -S /opt/colmap-src -B /opt/colmap-src/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="120" \
        -DCGAL_ENABLED=OFF \
    && ninja -C /opt/colmap-src/build -j4 \
    && ninja -C /opt/colmap-src/build install \
    && rm -rf /opt/colmap-src

# ---- Python virtual environment --------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch with CUDA 12.8 — install first so later packages reuse the cache
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Core Python deps
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    Pillow \
    tqdm \
    open3d

# hloc (includes DISK, ALIKED, LightGlue, NetVLAD)
RUN pip install --no-cache-dir \
    git+https://github.com/cvg/Hierarchical-Localization.git

# Depth Anything V2 — not a pip package, clone and add to path
RUN git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git /opt/depth-anything-v2
ENV PYTHONPATH="/opt/depth-anything-v2:${PYTHONPATH}"

# ---- Pipeline source -------------------------------------------------------
# Copied last so code changes don't invalidate the COLMAP/Python layers above
WORKDIR /workspace
COPY pipeline.py .
COPY splatter/ splatter/
COPY entry_script.sh /usr/local/bin/entry_script.sh
RUN chmod +x /usr/local/bin/entry_script.sh

# Model weights cache dir — /cache/.cache is bind-mounted at runtime (see
# docker-compose.yml) so every downloaded checkpoint persists across container
# restarts/recreation and never re-downloads after the first run: DAv2 (its own
# manual cache_dir under ~/.cache/depth_anything_v2), hloc's NetVLAD/DISK/ALIKED/
# LightGlue (all resolve through torch.hub, hence TORCH_HOME below), and the
# SegFormer sky-mask model (resolves through HuggingFace's cache, hence HF_HOME).
ENV HOME=/cache
ENV TORCH_HOME=/cache/.cache/torch
ENV HF_HOME=/cache/.cache/huggingface
RUN mkdir -p /cache/.cache/depth_anything_v2 /cache/.cache/torch /cache/.cache/huggingface

# Run as a non-root user so files written to bind-mounted host directories
# (./data/output, and ./pipeline.py/splatter/ during development) come out
# owned by the host user instead of root. Ubuntu 24.04 already ships a
# preconfigured "ubuntu" user at UID/GID 1000:1000 — the same UID/GID as the
# first non-system user on virtually any single-user Linux install (including
# this project's host machine), so no build-arg parameterization is needed.
# Everything this user needs at runtime is already in place: COLMAP, the
# venv, and exiftool are all installed world-readable/executable (default
# umask from apt/pip/ninja install), and the CUDA base image's own
# entrypoint (/opt/nvidia/nvidia_entrypoint.sh) only prints banner text —
# no privileged setup — so GPU access needs no root step here either.
RUN chown -R ubuntu:ubuntu /cache/.cache /workspace
USER ubuntu

# Keep the container alive so docker compose exec can be used
CMD ["tail", "-f", "/dev/null"]
