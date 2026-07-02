"""Pipeline logger: console + file handler."""

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("gs_init")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(output_dir / "pipeline.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
