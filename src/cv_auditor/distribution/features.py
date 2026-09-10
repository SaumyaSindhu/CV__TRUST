"""
features.py (distribution)
--------------------------
Per-image feature vectors for distribution-shift analysis.

Reuses Phase 2's ``EmbeddingExtractor`` (pretrained CNN when available,
handcrafted color/texture descriptor offline -- backend is recorded) and
concatenates a small block of *global acquisition statistics* computed
directly from the image: mean/std brightness, per-channel means,
gradient energy, and a grayscale-entropy signal.

Why the extra block: terrain / season / sensor / illumination changes --
the drivers Phase 5 must detect -- manifest exactly as global color,
brightness and texture changes, and Phase 2's foreground-oriented
descriptor deliberately suppresses background, so a whole-image
statistics block is added here (pure PIL + numpy, offline, deterministic,
fully explainable).

The combined backend identifier is stored in every baseline and every
report; baselines built with one backend refuse comparison against
batches featurized with another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image

from ..dataset.embeddings import EmbeddingExtractor
from ..utils import get_logger

logger = get_logger(__name__)

GLOBAL_STATS_VERSION = "globalstats_v1"
GLOBAL_STATS_DIM = 8  # brightness mean/std, R/G/B means, gradient energy, entropy, saturation


@dataclass
class FeaturizedBatch:
    """Features for one batch of images, plus per-image bookkeeping."""

    image_paths: List[str]           # images successfully featurized (order matches rows)
    features: np.ndarray             # [n, d] float64
    corrupt_images: List[str] = field(default_factory=list)
    corrupt_errors: List[str] = field(default_factory=list)

    @property
    def num_ok(self) -> int:
        return len(self.image_paths)

    @property
    def num_corrupt(self) -> int:
        return len(self.corrupt_images)


def _global_stats(image_path: str, size: int = 96) -> np.ndarray:
    """Whole-image acquisition statistics in [0, 1]-ish ranges."""
    with Image.open(image_path) as img:
        rgb = np.asarray(
            img.convert("RGB").resize((size, size), Image.BILINEAR),
            dtype=np.float64,
        ) / 255.0

    gray = rgb.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()

    hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 5.0)  # /log2(32) -> [0,1]

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = float(np.mean(np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-9), 0.0)))

    return np.array(
        [
            float(gray.mean()),          # overall brightness
            float(gray.std()),           # global contrast
            float(rgb[..., 0].mean()),   # R
            float(rgb[..., 1].mean()),   # G
            float(rgb[..., 2].mean()),   # B
            float((gx + gy) / 2.0),      # gradient energy (texture/sharpness)
            entropy,                     # tonal entropy
            saturation,                  # colorfulness
        ],
        dtype=np.float64,
    )


class DistributionFeatureExtractor:
    """Phase 2 embedding + global acquisition statistics, per image."""

    def __init__(self, device: str = "cpu", cache_dir: Optional[str] = None,
                 embedding_extractor: Optional[EmbeddingExtractor] = None):
        self._embedder = embedding_extractor or EmbeddingExtractor(
            device=device, cache_dir=cache_dir,
        )
        self.backend = f"{self._embedder.backend}+{GLOBAL_STATS_VERSION}"
        self.pretrained_embeddings = self._embedder.pretrained

    def extract_one(self, image_path: str) -> np.ndarray:
        emb = np.asarray(self._embedder.extract(image_path), dtype=np.float64)
        return np.concatenate([emb, _global_stats(image_path)])

    def extract_batch(self, image_paths: List[str]) -> FeaturizedBatch:
        """Featurize a list of images, tolerating unreadable files.

        Corrupt/unreadable images are excluded from the feature matrix
        but recorded so the auditor can surface them as findings -- a
        batch full of undecodable files is itself evidence.
        """
        rows, ok_paths, corrupt, errors = [], [], [], []
        for path in image_paths:
            try:
                rows.append(self.extract_one(path))
                ok_paths.append(path)
            except Exception as exc:  # noqa: BLE001 - per-image robustness
                corrupt.append(path)
                errors.append(f"{type(exc).__name__}: {exc}")
                logger.warning("Could not featurize '%s': %s", path, exc)

        features = (
            np.vstack(rows) if rows
            else np.empty((0, 0), dtype=np.float64)
        )
        return FeaturizedBatch(
            image_paths=ok_paths, features=features,
            corrupt_images=corrupt, corrupt_errors=errors,
        )
