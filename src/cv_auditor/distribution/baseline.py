"""
baseline.py (distribution)
--------------------------
The trusted reference distribution: robust statistics over a set of
reference images, stored as a single JSON file so future batches can be
compared against it fully offline.

What is stored (and why):
- centroid + per-dimension mean/std ................ cheap location/scale checks
- regularized covariance + its (pseudo-)inverse .... Mahalanobis distances
- the baseline's OWN per-sample Mahalanobis
  distances (mean/std/percentiles) ................. the calibration scale:
  every Phase 5 threshold is expressed relative to how far the baseline's
  own images sit from their own centroid, which is what makes thresholds
  portable across feature backends and image domains
- capped raw feature vectors ....................... MMD and later re-calibration
- feature backend id ............................... refuse apples-vs-oranges
- optional acquisition metadata .................... drift-vs-anomaly context
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils import find_images, get_logger, utc_timestamp
from .config import DistributionConfig
from .features import DistributionFeatureExtractor

logger = get_logger(__name__)

BASELINE_SCHEMA_VERSION = "1.0"


def _regularized_covariance(features: np.ndarray) -> np.ndarray:
    """Shrinkage covariance: blend the sample covariance toward its own
    diagonal (lambda grows as dimension approaches sample count) plus a
    small ridge. Keeps Mahalanobis distances sane when n is not much
    larger than d -- exactly the small-baseline regime Phase 5 runs in."""
    n, d = features.shape
    cov = np.atleast_2d(np.cov(features, rowvar=False))
    lam = d / (d + max(n, 1))
    cov = (1.0 - lam) * cov + lam * np.diag(np.diag(cov))
    ridge = 1e-6 + 1e-3 * float(np.trace(cov)) / cov.shape[0]
    return cov + ridge * np.eye(cov.shape[0])


def mahalanobis_distances(features: np.ndarray, centroid: np.ndarray,
                          inv_cov: np.ndarray) -> np.ndarray:
    delta = features - centroid
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inv_cov, delta), 0.0))


def _loo_self_distances(features: np.ndarray) -> np.ndarray:
    """Leave-one-out Mahalanobis distances of the baseline's own samples.

    In-sample distances systematically UNDERestimate how far genuinely
    new points land (each point pulls the centroid/covariance toward
    itself, badly so when n is close to d), which would make every
    future batch look anomalous. Scoring each baseline image against
    statistics built from the OTHER images mimics the held-out setting
    future batches are actually in, which is what makes the stored
    calibration (mean/std/percentiles) honest.
    """
    n = features.shape[0]
    out = np.empty(n, dtype=np.float64)
    idx = np.arange(n)
    for i in range(n):
        rest = features[idx != i]
        centroid = rest.mean(axis=0)
        inv_cov = np.linalg.pinv(_regularized_covariance(rest))
        out[i] = mahalanobis_distances(features[i:i + 1], centroid, inv_cov)[0]
    return out


@dataclass
class DistributionBaseline:
    """A stored, trusted reference distribution."""

    schema_version: str
    created_at: str
    feature_backend: str
    pretrained_embeddings: bool
    source_dir: str
    num_samples: int
    feature_dim: int
    centroid: np.ndarray
    feature_std: np.ndarray
    covariance: np.ndarray
    inv_covariance: np.ndarray
    self_distance_mean: float
    self_distance_std: float
    self_distance_p95: float
    self_distance_p99: float
    self_distance_max: float
    stored_embeddings: np.ndarray           # [m, d], m <= max_stored_embeddings
    metadata: Optional[Dict] = None         # declared acquisition conditions
    corrupt_images: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def summary_dict(self) -> dict:
        """Compact description for embedding into reports."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "feature_backend": self.feature_backend,
            "pretrained_embeddings": self.pretrained_embeddings,
            "source_dir": self.source_dir,
            "num_samples": self.num_samples,
            "feature_dim": self.feature_dim,
            "self_distance_mean": round(self.self_distance_mean, 4),
            "self_distance_std": round(self.self_distance_std, 4),
            "self_distance_p99": round(self.self_distance_p99, 4),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict:
        d = self.summary_dict()
        d.update(
            {
                "centroid": self.centroid.tolist(),
                "feature_std": self.feature_std.tolist(),
                "covariance": self.covariance.tolist(),
                "inv_covariance": self.inv_covariance.tolist(),
                "self_distance_p95": round(self.self_distance_p95, 6),
                "self_distance_max": round(self.self_distance_max, 6),
                "stored_embeddings": self.stored_embeddings.tolist(),
                "corrupt_images": self.corrupt_images,
            }
        )
        return d

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        logger.info("Distribution baseline saved to %s", path)

    @classmethod
    def from_dict(cls, d: dict) -> "DistributionBaseline":
        return cls(
            schema_version=d["schema_version"],
            created_at=d["created_at"],
            feature_backend=d["feature_backend"],
            pretrained_embeddings=bool(d["pretrained_embeddings"]),
            source_dir=d["source_dir"],
            num_samples=int(d["num_samples"]),
            feature_dim=int(d["feature_dim"]),
            centroid=np.asarray(d["centroid"], dtype=np.float64),
            feature_std=np.asarray(d["feature_std"], dtype=np.float64),
            covariance=np.asarray(d["covariance"], dtype=np.float64),
            inv_covariance=np.asarray(d["inv_covariance"], dtype=np.float64),
            self_distance_mean=float(d["self_distance_mean"]),
            self_distance_std=float(d["self_distance_std"]),
            self_distance_p95=float(d["self_distance_p95"]),
            self_distance_p99=float(d["self_distance_p99"]),
            self_distance_max=float(d["self_distance_max"]),
            stored_embeddings=np.asarray(d["stored_embeddings"], dtype=np.float64),
            metadata=d.get("metadata"),
            corrupt_images=list(d.get("corrupt_images", [])),
        )

    @classmethod
    def load(cls, path: str) -> "DistributionBaseline":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def build_baseline(
    reference_dir: str,
    config: Optional[DistributionConfig] = None,
    metadata: Optional[Dict] = None,
    extractor: Optional[DistributionFeatureExtractor] = None,
) -> DistributionBaseline:
    """Build a trusted baseline from a directory (or single file) of
    reference images."""
    cfg = config or DistributionConfig()
    extractor = extractor or DistributionFeatureExtractor(
        device=cfg.device, cache_dir=cfg.embedding_cache_dir,
    )

    image_paths = find_images(reference_dir)
    batch = extractor.extract_batch(image_paths)

    if batch.num_ok < cfg.min_baseline_samples:
        raise ValueError(
            f"Baseline needs at least {cfg.min_baseline_samples} readable "
            f"reference images; got {batch.num_ok} usable out of "
            f"{len(image_paths)} found under '{reference_dir}'."
        )

    features = batch.features
    centroid = features.mean(axis=0)
    feature_std = features.std(axis=0, ddof=1)
    cov = _regularized_covariance(features)
    inv_cov = np.linalg.pinv(cov)

    self_d = _loo_self_distances(features)
    std = float(self_d.std(ddof=1)) if len(self_d) > 1 else 0.0

    stored = features[: cfg.max_stored_embeddings]

    baseline = DistributionBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        created_at=utc_timestamp(),
        feature_backend=extractor.backend,
        pretrained_embeddings=extractor.pretrained_embeddings,
        source_dir=str(reference_dir),
        num_samples=batch.num_ok,
        feature_dim=int(features.shape[1]),
        centroid=centroid,
        feature_std=feature_std,
        covariance=cov,
        inv_covariance=inv_cov,
        self_distance_mean=float(self_d.mean()),
        self_distance_std=max(std, 1e-6),
        self_distance_p95=float(np.percentile(self_d, 95)),
        self_distance_p99=float(np.percentile(self_d, 99)),
        self_distance_max=float(self_d.max()),
        stored_embeddings=stored,
        metadata=metadata,
        corrupt_images=batch.corrupt_images,
    )
    logger.info(
        "Baseline built from %d image(s) (dim=%d, backend=%s, %d corrupt skipped)",
        batch.num_ok, features.shape[1], extractor.backend, batch.num_corrupt,
    )
    return baseline
