"""
config.py (distribution)
------------------------
Configuration for Phase 5 -- Distribution Shift & Anomaly Detection.
Follows the flat-dataclass style of Phases 1-3 configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class DistributionConfig:
    """Tunables for baseline building and shift auditing.

    All thresholds operate on *baseline-relative* scales (z-scores against
    the baseline's own internal variability), which is what makes them
    portable across feature backends and image domains.

    Attributes
    ----------
    baseline_path:
        Where the trusted baseline JSON is stored / loaded from.
    min_baseline_samples:
        Building a baseline from fewer images than this is refused --
        covariance statistics below this size are meaningless.
    max_stored_embeddings:
        Cap on raw per-image feature vectors kept inside the baseline
        file (needed for MMD and honest re-calibration later).
    sample_anomaly_z:
        A current image whose Mahalanobis distance to the baseline lies
        this many baseline standard deviations above the baseline's own
        mean self-distance is flagged as an anomalous sample.
    batch_shift_z_medium / batch_shift_z_high:
        Batch-level displacement thresholds (in the same per-sample
        z-units) separating LOW / MEDIUM / HIGH risk.
    anomalous_fraction_medium / anomalous_fraction_high:
        Alternatively, risk escalates when this fraction of the incoming
        batch is individually anomalous (with at least
        ``min_anomalous_count`` images).
    """

    baseline_path: str = "results/distribution/baseline.json"

    # baseline building
    min_baseline_samples: int = 4
    max_stored_embeddings: int = 500

    # per-sample anomaly detection
    sample_anomaly_z: float = 3.0

    # batch-level risk thresholds (per-sample z units)
    batch_shift_z_medium: float = 1.5
    batch_shift_z_high: float = 3.5
    anomalous_fraction_medium: float = 0.10
    anomalous_fraction_high: float = 0.30
    min_anomalous_count: int = 2

    # dispersion-change evidence threshold (ratio of spreads)
    dispersion_ratio_threshold: float = 2.0

    device: str = "cpu"
    embedding_cache_dir: Optional[str] = None  # None = no on-disk cache

    output_path: str = "results/distribution/distribution_report.json"

    def __post_init__(self):
        if self.min_baseline_samples < 2:
            raise ValueError("min_baseline_samples must be >= 2")
        if self.sample_anomaly_z <= 0:
            raise ValueError("sample_anomaly_z must be positive")
        if not (self.batch_shift_z_medium < self.batch_shift_z_high):
            raise ValueError("batch_shift_z_medium must be < batch_shift_z_high")
        if not (0 < self.anomalous_fraction_medium < self.anomalous_fraction_high <= 1):
            raise ValueError("anomalous fraction thresholds must satisfy 0 < medium < high <= 1")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DistributionConfig":
        with open(path) as f:
            return cls(**json.load(f))
