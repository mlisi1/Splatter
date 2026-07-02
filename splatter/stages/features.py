"""Stage 2 — Feature extraction and matching via hloc."""

import logging
from pathlib import Path

logger = logging.getLogger("gs_init")

_EXTRACTOR_KEYS = {"disk": "disk", "aliked": "aliked-n16"}
_MATCHER_KEYS = {"disk": "disk+lightglue", "aliked": "aliked+lightglue"}


def run_hloc(output_dir: Path, matcher: str) -> tuple[Path, Path, Path]:
    """
    Run the full hloc pipeline:
      1. Local feature extraction (DISK or ALIKED)
      2. Global descriptor extraction (NetVLAD)
      3. Retrieval-based pair selection (top-20)
      4. Temporal pair augmentation (±5 frames)
      5. LightGlue feature matching

    Returns:
        feature_path : HDF5 file with local features
        match_path   : HDF5 file with matches
        pairs_path   : text file listing all image pairs
    """
    try:
        from hloc import extract_features, match_features, pairs_from_retrieval
    except ImportError:
        raise ImportError(
            "hloc not found. Install:\n"
            "  pip install git+https://github.com/cvg/Hierarchical-Localization.git"
        )

    images_dir = output_dir / "images"
    feature_path = output_dir / f"feats-{matcher}.h5"
    global_feat_path = output_dir / "global-feats-netvlad.h5"
    pairs_retrieval = output_dir / "pairs-netvlad.txt"
    pairs_all = output_dir / "pairs-all.txt"
    match_path = output_dir / f"matches-{matcher}-lightglue.h5"

    if feature_path.exists():
        logger.info(f"Found existing features {feature_path.name} — skipping extraction")
    else:
        logger.info(f"Extracting local features ({matcher.upper()})")
        extract_features.main(
            extract_features.confs[_EXTRACTOR_KEYS[matcher]],
            images_dir,
            feature_path=feature_path,
        )

    if global_feat_path.exists():
        logger.info("Found existing global descriptors — skipping NetVLAD")
    else:
        logger.info("Extracting global descriptors (NetVLAD)")
        extract_features.main(
            extract_features.confs["netvlad"],
            images_dir,
            feature_path=global_feat_path,
        )

    if pairs_all.exists():
        logger.info("Found existing pairs file — skipping retrieval")
    else:
        logger.info("Retrieving top-20 pairs")
        pairs_from_retrieval.main(global_feat_path, pairs_retrieval, num_matched=20)
        pairs_all.write_text(_merge_pairs(pairs_retrieval, images_dir))

    if match_path.exists():
        logger.info(f"Found existing matches {match_path.name} — skipping matching")
    else:
        logger.info("Matching features (LightGlue)")
        match_features.main(
            match_features.confs[_MATCHER_KEYS[matcher]],
            pairs_all,
            feature_path,
            matches=match_path,
        )

    return feature_path, match_path, pairs_all


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_pairs(retrieval_file: Path, images_dir: Path) -> str:
    """Merge retrieval pairs with ±5-frame temporal pairs and return as text."""
    image_names = sorted(p.name for p in images_dir.glob("*.png"))

    temporal: set[tuple[str, str]] = set()
    for i, a in enumerate(image_names):
        for j in range(max(0, i - 10), min(len(image_names), i + 11)):
            if i != j:
                temporal.add(tuple(sorted([a, image_names[j]])))  # type: ignore[arg-type]

    retrieval: set[tuple[str, str]] = set()
    if retrieval_file.exists():
        for line in retrieval_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                retrieval.add(tuple(sorted(parts)))  # type: ignore[arg-type]

    all_pairs = retrieval | temporal
    logger.info(
        f"Pairs: {len(all_pairs)} total "
        f"(retrieval {len(retrieval)}, temporal {len(temporal)})"
    )
    return "\n".join(f"{a} {b}" for a, b in sorted(all_pairs)) + "\n"
