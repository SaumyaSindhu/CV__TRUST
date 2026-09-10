"""
utils.py
--------
Small, dependency-light helper functions shared across the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent, simple format for the whole package."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, e.g. 2026-09-01T12:34:56.789012+00:00."""
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: str, chunk_size: int = 65536) -> str:
    """Compute a SHA-256 hex digest of a file's contents.

    Used as part of model/image metadata so later audit phases can verify
    that a file has not changed between pipeline runs.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_images(input_path: str) -> list:
    """Resolve a path (file or directory) into a sorted list of image file paths."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if p.is_file():
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"File does not look like a supported image: {input_path}")
        return [str(p)]

    images = sorted(
        str(f) for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in directory: {input_path}")
    return images


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
