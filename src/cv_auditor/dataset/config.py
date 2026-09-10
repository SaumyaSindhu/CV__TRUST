"""
config.py (dataset)
--------------------
Central configuration for a Dataset Integrity Auditor run. Mirrors the
style of ``cv_auditor.config.PipelineConfig`` from Phase 1: one flat
dataclass, JSON-serializable, every tunable in one place so a saved
audit report can show exactly which thresholds produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class DatasetAuditConfig:
    """Configuration for a single dataset-integrity audit run.

    Attributes
    ----------
    dataset_dir:
        Root directory of the dataset (expects an ``images/`` subfolder,
        and optionally a ``labels/`` subfolder in YOLO format).
    dataset_format:
        Which registered ``DatasetFormat`` to use. Only "yolo_folder" is
        implemented in Phase 2.
    class_names:
        Optional list mapping YOLO class_id -> human-readable name.
    output_path:
        Where the structured JSON audit report is written.
    near_duplicate_similarity_threshold:
        Perceptual-hash similarity (0-1) at/above which two images are
        reported as near-duplicates.
    phash_size:
        Side length of the perceptual hash grid (produces
        phash_size^2 bits). 8 is the well-established default for dHash.
    embedding_model:
        Which TorchVision backbone to use for embeddings.
    embedding_cache_dir:
        Where computed embeddings are cached (keyed by image content
        hash). Set to None to disable caching.
    device:
        "cpu" or "cuda" for embedding extraction.
    ood_k:
        Number of nearest neighbours used for the OOD/anomaly score.
    ood_threshold:
        Cosine-distance score at/above which a sample is flagged as a
        potential OOD/anomaly finding.
    ood_high_severity_threshold:
        Score at/above which an OOD finding is marked "high" severity
        instead of "medium".
    label_cluster_count:
        Number of k-means clusters used for label-anomaly detection. If
        None (the default), it is chosen automatically as the number of
        distinct class ids observed in the dataset's labels (clamped to
        at least 2) -- a reasonable data-driven default, since visually
        grouping into roughly "one cluster per known class" is what the
        majority-label consistency check assumes. Set explicitly to
        override.
    label_min_cluster_size:
        Minimum cluster size considered for label-anomaly detection.
    label_minority_ratio_threshold:
        A class occurring in less than this fraction of a cluster's
        members is treated as a minority (and thus a candidate anomaly).
    """

    dataset_dir: str = "data/sample_dataset"
    dataset_format: str = "yolo_folder"
    class_names: Optional[List[str]] = None
    output_path: str = "results/dataset_audit_report.json"

    near_duplicate_similarity_threshold: float = 0.90
    phash_size: int = 8

    embedding_model: str = "resnet18"
    embedding_cache_dir: Optional[str] = "results/embedding_cache"
    device: str = "cpu"

    ood_k: int = 5
    ood_threshold: float = 2.0
    ood_high_severity_threshold: float = 3.0

    label_cluster_count: Optional[int] = None
    label_min_cluster_size: int = 3
    label_minority_ratio_threshold: float = 0.34

    def __post_init__(self):
        if not (0.0 <= self.near_duplicate_similarity_threshold <= 1.0):
            raise ValueError("near_duplicate_similarity_threshold must be in [0, 1]")
        if self.phash_size < 2:
            raise ValueError("phash_size must be >= 2")
        if self.ood_k < 1:
            raise ValueError("ood_k must be >= 1")
        if not (0.0 < self.label_minority_ratio_threshold <= 1.0):
            raise ValueError("label_minority_ratio_threshold must be in (0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DatasetAuditConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
