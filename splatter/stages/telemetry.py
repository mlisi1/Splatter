"""Stage 3.6 — Georegistration from embedded video telemetry.

Monocular SfM (Stage 3) has no absolute scale reference: COLMAP's mapper
recovers camera poses and structure only up to an arbitrary similarity
transform. Without --lidar, that arbitrary scale propagates straight through
densification's images-only branch (run_images_only fits DAv2 disparity to
the SfM cloud's own 1/z, so its "metric" depth is only as metric as the SfM
scale it was fit to) — self-consistent, not real-world metric.

This stage extracts per-frame GPS positions already embedded in many
cameras' video files — DJI drone SRT sidecars, or an embedded GPS track
decoded via exiftool (GoPro GPMF, DJI's newer protobuf-encoded djmd track
used by drones and Osmo Action cameras, Garmin VIRB, Insta360, etc.) —
matches them to registered frames by capture timestamp, and uses COLMAP's
model_aligner to fit a similarity transform (scale + rotation + translation)
that rewrites sparse/0 in real-world ENU metres. It plays the same role
--lidar does for scale, just sourced from telemetry instead of a physical
scan, and is skipped automatically when --lidar is supplied — LiDAR fusion's
T_cam_to_lidar extrinsic is a fixed physical measurement calibrated against
the SfM reconstruction's own scale, and rescaling on top of it would
silently double-scale the fused cloud.

Best-effort throughout: any failure (no telemetry found, too few frames
matched, model_aligner rejects the correspondences) logs a warning and
leaves sparse/0 untouched rather than aborting the pipeline.
"""

import logging
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from splatter.io.colmap import read_images_binary

logger = logging.getLogger("gs_init")

_GPX_FMT = Path(__file__).with_name("_telemetry_gpx.fmt")
_GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/0"}


@dataclass
class TelemetrySample:
    time_sec: float
    lat: float
    lon: float
    alt: float


def _describe_samples(samples: list) -> str:
    lats = [s.lat for s in samples]
    lons = [s.lon for s in samples]
    alts = [s.alt for s in samples]
    times = [s.time_sec for s in samples]
    return (
        f"{len(samples)} samples spanning {times[0]:.1f}s–{times[-1]:.1f}s, "
        f"lat [{min(lats):.6f}, {max(lats):.6f}], lon [{min(lons):.6f}, {max(lons):.6f}], "
        f"alt [{min(alts):.1f}, {max(alts):.1f}] m"
    )


# ---------------------------------------------------------------------------
# DJI drone SRT sidecar
# ---------------------------------------------------------------------------

_SRT_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->")
# Newer DJI firmware: explicit keyed fields, e.g. "[latitude: 22.542033] [longitude : 114.062996]"
_KEYED_LATLON_RE = re.compile(
    r"latitude\s*:\s*(-?\d+\.?\d*).*?longitude\s*:\s*(-?\d+\.?\d*)", re.IGNORECASE | re.DOTALL
)
_ABS_ALT_RE = re.compile(r"abs_alt\s*:\s*(-?\d+\.?\d*)", re.IGNORECASE)
_REL_ALT_RE = re.compile(r"rel_alt\s*:\s*(-?\d+\.?\d*)", re.IGNORECASE)
# Older DJI firmware (Phantom/Mavic/Air): "GPS (lon, lat, alt)" — note order.
_PAREN_GPS_RE = re.compile(
    r"GPS\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)", re.IGNORECASE
)


def find_dji_srt(video_path: Path) -> Path | None:
    for suffix in (".srt", ".SRT", ".Srt"):
        candidate = video_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def parse_dji_srt(srt_path: Path) -> list[TelemetrySample]:
    """
    Parse a DJI telemetry-overlay SRT sidecar. Field layout varies across
    drone generations/firmware, so both known flavours are tried per block:
    keyed `latitude:`/`longitude:` fields (newer), or a parenthetical
    `GPS (lon, lat, alt)` triple (older) — see module regexes above.
    """
    text = srt_path.read_text(errors="replace")
    samples = []
    for block in re.split(r"\n\s*\n", text.strip()):
        time_m = _SRT_TIME_RE.search(block)
        if not time_m:
            continue
        h, m, s, ms = (int(x) for x in time_m.groups())
        t = h * 3600.0 + m * 60.0 + s + ms / 1000.0

        keyed_m = _KEYED_LATLON_RE.search(block)
        if keyed_m:
            lat, lon = float(keyed_m.group(1)), float(keyed_m.group(2))
            alt_m = _ABS_ALT_RE.search(block) or _REL_ALT_RE.search(block)
            alt = float(alt_m.group(1)) if alt_m else 0.0
        else:
            paren_m = _PAREN_GPS_RE.search(block)
            if not paren_m:
                continue
            lon, lat, alt = (float(x) for x in paren_m.groups())

        samples.append(TelemetrySample(t, lat, lon, alt))

    return samples


# ---------------------------------------------------------------------------
# Embedded GPS track, decoded generically via exiftool. This covers GoPro's
# GPMF stream, Garmin VIRB, Insta360, and — since exiftool 13.00 (Dec 2024) —
# DJI's newer protobuf-encoded "djmd" metadata track, used by current-gen
# drones and Osmo Action cameras instead of (or alongside) the classic SRT
# sidecar. All of these decode down to exiftool's standard GPSLatitude /
# GPSLongitude / GPSAltitude / GPSDateTime tags, so one extraction path
# handles them all.
# ---------------------------------------------------------------------------

_MIN_EXIFTOOL_VERSION = (13, 0)


def _exiftool_version() -> tuple[int, int] | None:
    try:
        r = subprocess.run(["exiftool", "-ver"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    m = re.match(r"(\d+)\.(\d+)", r.stdout.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_embedded_gps_track(video_path: Path) -> list[TelemetrySample]:
    """
    Extract the embedded GPS track via exiftool. -ee3 walks every embedded
    sample (not just the single-point summary tag a plain -GPSLatitude query
    would return), formatted as GPX by _telemetry_gpx.fmt.

    An exiftool older than 13.0 will run without error but silently decode
    zero GPS samples from a DJI djmd track (it recognises the protobuf
    stream exists but doesn't yet know its schema) — so an empty result from
    an old exiftool looks identical to "no GPS in this video" unless we
    check the version explicitly and say so.
    """
    version = _exiftool_version()
    if version is not None and version < _MIN_EXIFTOOL_VERSION:
        logger.warning(
            f"exiftool {version[0]}.{version[1]} found (< {_MIN_EXIFTOOL_VERSION[0]}.{_MIN_EXIFTOOL_VERSION[1]}) "
            "— DJI's protobuf-encoded djmd track (newer drones, Osmo Action) was only added to "
            "exiftool's DJI module in Dec 2024, so an old exiftool will silently find no GPS on "
            "that source even if it's present. GoPro GPMF is unaffected. Upgrade exiftool if a "
            "DJI video's GPS isn't being found."
        )

    r = subprocess.run(
        ["exiftool", "-ee3", "-p", str(_GPX_FMT), str(video_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not r.stdout.strip():
        logger.debug(f"exiftool GPX extraction produced no output: {r.stderr.strip()[:300]}")
        return []

    try:
        root = ET.fromstring(r.stdout)
    except ET.ParseError as e:
        logger.debug(f"Failed to parse exiftool GPX output: {e}")
        return []

    samples = []
    t0 = None
    for trkpt in root.findall(".//gpx:trkpt", _GPX_NS):
        lat, lon = trkpt.get("lat"), trkpt.get("lon")
        if lat is None or lon is None:
            continue
        ele_el = trkpt.find("gpx:ele", _GPX_NS)
        alt = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
        time_el = trkpt.find("gpx:time", _GPX_NS)
        if time_el is None or not time_el.text:
            continue

        time_str = time_el.text.strip().rstrip("Z")
        try:
            ts_dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            ts_dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")
        ts = ts_dt.replace(tzinfo=timezone.utc).timestamp()

        if t0 is None:
            t0 = ts
        samples.append(TelemetrySample(ts - t0, float(lat), float(lon), alt))

    return samples


# ---------------------------------------------------------------------------
# Detection + orchestration
# ---------------------------------------------------------------------------

def detect_and_extract(video_path: Path, mode: str) -> tuple[list, str] | tuple[None, None]:
    logger.info(f"Telemetry detection: mode={mode}, video={video_path.name}")

    if mode in ("auto", "dji"):
        srt_path = find_dji_srt(video_path)
        if srt_path is not None:
            logger.info(f"Found DJI sidecar: {srt_path} — parsing")
            samples = parse_dji_srt(srt_path)
            if samples:
                logger.info(f"DJI SRT telemetry found: {_describe_samples(samples)}")
                return samples, "DJI SRT telemetry"
            logger.warning(
                f"Found DJI sidecar {srt_path.name} but couldn't parse any GPS samples from it "
                "(unrecognised field layout — see parse_dji_srt regexes)"
            )
        else:
            logger.info(f"No DJI .srt sidecar found next to {video_path.name}")
        if mode == "dji":
            return None, None

    if mode in ("auto", "embedded"):
        logger.info("Probing video for an embedded GPS track via exiftool (GoPro GPMF, DJI djmd/protobuf, etc.)")
        samples = parse_embedded_gps_track(video_path)
        if samples:
            logger.info(f"Embedded GPS track found: {_describe_samples(samples)}")
            return samples, "Embedded GPS track (exiftool)"
        logger.info("No embedded GPS track found in the video")
        if mode == "embedded":
            logger.warning("--telemetry embedded requested but no embedded GPS track found in the video")

    return None, None


def _match_frame_times(image_names: list[str], fps: float, samples: list,
                        max_gap: float) -> dict[str, "TelemetrySample"]:
    sample_times = np.array([s.time_sec for s in samples])
    matches = {}
    for name in image_names:
        try:
            idx = int(Path(name).stem)
        except ValueError:
            continue
        t = (idx - 1) / fps
        j = int(np.argmin(np.abs(sample_times - t)))
        if abs(sample_times[j] - t) <= max_gap:
            matches[name] = samples[j]
    return matches


def _write_ref_images(matches: dict[str, "TelemetrySample"], path: Path) -> None:
    with open(path, "w") as f:
        for name, s in matches.items():
            f.write(f"{name} {s.lat:.8f} {s.lon:.8f} {s.alt:.3f}\n")


def run(
    output_dir: Path,
    sparse_0: Path,
    video_path: Path,
    fps: float,
    mode: str = "auto",
    max_error: float = 10.0,
    min_matches: int = 10,
) -> str | None:
    """
    Georegister sparse_0 to real-world ENU metres using telemetry GPS.

    Returns the telemetry source string if a rescale was applied, None if
    skipped for any reason (no telemetry found, too few matched frames,
    model_aligner failed) — always non-fatal; the pipeline continues with
    the un-rescaled model.
    """
    if mode == "off":
        logger.info("Telemetry georegistration disabled (--telemetry off)")
        return None

    samples, source = detect_and_extract(video_path, mode)
    if not samples:
        logger.info("No usable telemetry found — skipping georegistration")
        return None

    images = read_images_binary(sparse_0 / "images.bin")
    image_names = [img.name for img in images.values()]
    logger.info(f"{len(image_names)} registered frames available to match against {source}")

    sorted_times = sorted(s.time_sec for s in samples)
    median_dt = float(np.median(np.diff(sorted_times))) if len(sorted_times) > 1 else 1.0
    max_gap = max(2.0 / fps, 2.0 * median_dt, 1.0)
    logger.info(
        f"Matching frames to telemetry by timestamp: fps={fps}, "
        f"telemetry sample interval~{median_dt:.3f}s, max time gap allowed={max_gap:.3f}s"
    )

    matches = _match_frame_times(image_names, fps, samples, max_gap)
    logger.info(f"Matched {len(matches)}/{len(image_names)} registered frames to a telemetry sample")
    if len(matches) < min_matches:
        unmatched = len(image_names) - len(matches)
        logger.warning(
            f"Only {len(matches)} frames matched to telemetry samples (need >= {min_matches}); "
            f"{unmatched} frames had no telemetry sample within {max_gap:.3f}s "
            "— skipping georegistration"
        )
        return None

    ref_path = output_dir / "telemetry_ref_images.txt"
    _write_ref_images(matches, ref_path)
    logger.info(
        f"Wrote {len(matches)} GPS reference positions to {ref_path.name} — "
        f"running colmap model_aligner (source={source})"
    )

    tmp_dir = output_dir / "_georegister_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    transform_path = output_dir / "telemetry_transform.txt"
    r = subprocess.run(
        ["colmap", "model_aligner",
         "--input_path", str(sparse_0),
         "--output_path", str(tmp_dir),
         "--ref_images_path", str(ref_path),
         "--ref_is_gps", "1",
         "--alignment_type", "enu",
         "--alignment_max_error", str(max_error),
         "--min_common_images", str(min_matches),
         "--transform_path", str(transform_path)],
        capture_output=True,
    )
    if r.returncode != 0 or not (tmp_dir / "cameras.bin").exists():
        stderr = r.stderr.decode(errors="replace").strip()
        logger.warning(f"colmap model_aligner failed — skipping georegistration. {stderr[:500]}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    backup_dir = sparse_0.parent / "_pre_georegister_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(sparse_0, backup_dir)

    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        (tmp_dir / fname).replace(sparse_0 / fname)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        "Georegistration complete — sparse/0 rescaled to real-world ENU metres "
        f"(transform saved to {transform_path.name}; pre-alignment backup at {backup_dir.name})"
    )
    return source
