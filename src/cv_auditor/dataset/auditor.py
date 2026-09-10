"""
auditor.py
----------
``DatasetAuditor`` ties every Phase 2 detector together into a single
end-to-end run and produces one structured, saveable report:

    dataset directory
        -> load (dataset_loader.py)
        -> validate (validation.py)
        -> exact duplicates (duplicates.py)
        -> near duplicates (near_duplicates.py)
        -> embeddings (embeddings.py)
        -> label anomalies (label_anomaly.py)
        -> OOD/anomaly (ood.py)
        -> findings + dataset-level summary (this file)

This module intentionally stops at a *dataset-level* finding/severity
summary (see ``_overall_level`` below). It does NOT compute a
project-wide risk score -- that is explicitly out of scope for Phase 2
(see README / project instructions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from ..utils import ensure_parent_dir, get_logger, utc_timestamp
from .config import DatasetAuditConfig
from .dataset_loader import DatasetIndex, DatasetLoader
from .duplicates import exact_duplicate_findings, find_exact_duplicate_groups
from .embeddings import EmbeddingExtractor
from .findings import FindingsCollection
from .label_anomaly import label_anomaly_findings
from .near_duplicates import near_duplicate_findings
from .ood import ood_findings
from .perceptual_hash import find_near_duplicate_groups
from .validation import validate_dataset

logger = get_logger(__name__)

# Dataset-level severity thresholds. These are simple, documented,
# tunable-in-code heuristics -- not a learned or "official" scoring
# model. See README "What the scores mean".
_HIGH_INVALID_FRACTION = 0.05     # >5% unreadable/invalid samples -> HIGH
_HIGH_FINDING_COUNT = 5           # >5 high-severity findings -> HIGH
_MEDIUM_FINDING_COUNT = 1         # any medium-severity findings -> MEDIUM


@dataclass
class DatasetAuditReport:
    """The full, structured result of a dataset-integrity audit run."""

    config: dict
    dataset_dir: str
    generated_at: str
    total_samples: int
    valid_samples: int
    invalid_samples: int
    exact_duplicate_groups: int
    near_duplicate_groups: int
    label_anomalies: int
    ood_anomalies: int
    overall_finding_level: str
    embedding_model: str
    embedding_pretrained: bool
    embedding_backend: str
    findings: list

    def to_dict(self) -> dict:
        return {
            "run_metadata": {
                "config": self.config,
                "dataset_dir": self.dataset_dir,
                "generated_at": self.generated_at,
                "embedding_model": self.embedding_model,
                "embedding_pretrained": self.embedding_pretrained,
                "embedding_backend": self.embedding_backend,
            },
            "summary": {
                "total_samples": self.total_samples,
                "valid_samples": self.valid_samples,
                "invalid_samples": self.invalid_samples,
                "exact_duplicate_groups": self.exact_duplicate_groups,
                "near_duplicate_groups": self.near_duplicate_groups,
                "label_anomalies": self.label_anomalies,
                "potential_ood_anomalous_samples": self.ood_anomalies,
                "overall_dataset_finding_level": self.overall_finding_level,
            },
            "findings": self.findings,
        }

    def save(self, path: str) -> str:
        ensure_parent_dir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved dataset audit report (%d findings) to %s", len(self.findings), path)
        return path

    def print_summary(self) -> None:
        s = self.to_dict()["summary"]
        print("\nDataset Integrity Summary\n")
        print(f"Total samples: {s['total_samples']}")
        print(f"Valid samples: {s['valid_samples']}")
        print(f"Invalid samples: {s['invalid_samples']}")
        print()
        print(f"Exact duplicate groups: {s['exact_duplicate_groups']}")
        print(f"Near-duplicate groups: {s['near_duplicate_groups']}")
        print(f"Label anomalies: {s['label_anomalies']}")
        print(f"Potential OOD/anomalous samples: {s['potential_ood_anomalous_samples']}")
        print()
        print(f"Overall dataset finding level: {s['overall_dataset_finding_level']}")

    @staticmethod
    def load(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Report file not found: {path}")
        with open(p, "r") as f:
            return json.load(f)


class DatasetAuditor:
    """Runs the full Phase 2 dataset-integrity audit pipeline."""

    def __init__(self, config: Optional[DatasetAuditConfig] = None):
        self.config = config or DatasetAuditConfig()

    def run(self) -> DatasetAuditReport:
        cfg = self.config
        collection = FindingsCollection()

        # 1-2. Ingestion + validation
        index = DatasetLoader(
            cfg.dataset_dir, format_name=cfg.dataset_format, class_names=cfg.class_names,
        ).load()
        collection.extend(validate_dataset(index))

        # 3. Exact duplicates
        exact_groups = find_exact_duplicate_groups(index)
        collection.extend(exact_duplicate_findings(index))

        # 4. Near duplicates
        near_groups = find_near_duplicate_groups(
            {s.image_id: s.image_path for s in index.samples if s.readable},
            similarity_threshold=cfg.near_duplicate_similarity_threshold,
            hash_size=cfg.phash_size,
        )
        collection.extend(
            near_duplicate_findings(
                index,
                similarity_threshold=cfg.near_duplicate_similarity_threshold,
                hash_size=cfg.phash_size,
            )
        )

        # 5. Embeddings (needed for label-anomaly + OOD detection)
        extractor = EmbeddingExtractor(
            model_name=cfg.embedding_model,
            device=cfg.device,
            cache_dir=cfg.embedding_cache_dir,
        )
        embeddings: Dict = extractor.extract_batch(index.samples)

        # 6. Label anomaly detection (only meaningful for labeled datasets)
        n_label_anomalies = 0
        if index.labels_expected and embeddings:
            n_clusters = cfg.label_cluster_count
            if n_clusters is None:
                distinct_classes = {c for s in index.samples for c in s.class_ids}
                n_clusters = max(2, len(distinct_classes))
                logger.info(
                    "label_cluster_count not set; using %d clusters "
                    "(= number of distinct class ids observed).", n_clusters,
                )
            label_findings = label_anomaly_findings(
                index, embeddings,
                n_clusters=n_clusters,
                min_cluster_size=cfg.label_min_cluster_size,
                minority_ratio_threshold=cfg.label_minority_ratio_threshold,
            )
            collection.extend(label_findings)
            n_label_anomalies = len(label_findings)
        elif not index.labels_expected:
            logger.info("Dataset has no labels; skipping label-anomaly detection.")

        # 7. OOD / anomaly detection
        n_ood = 0
        if embeddings:
            ood_finds = ood_findings(
                embeddings,
                k=cfg.ood_k,
                threshold=cfg.ood_threshold,
                high_severity_threshold=cfg.ood_high_severity_threshold,
            )
            collection.extend(ood_finds)
            n_ood = len(ood_finds)

        # 8-9. Summary
        valid_samples = sum(1 for s in index.samples if s.readable)
        invalid_samples = index.num_samples - valid_samples
        overall_level = self._overall_level(
            total_samples=index.num_samples,
            invalid_samples=invalid_samples,
            severity_counts=collection.counts_by_severity(),
        )

        report = DatasetAuditReport(
            config=cfg.to_dict(),
            dataset_dir=cfg.dataset_dir,
            generated_at=utc_timestamp(),
            total_samples=index.num_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            exact_duplicate_groups=len(exact_groups),
            near_duplicate_groups=len(near_groups),
            label_anomalies=n_label_anomalies,
            ood_anomalies=n_ood,
            overall_finding_level=overall_level,
            embedding_model=cfg.embedding_model,
            embedding_pretrained=extractor.pretrained,
            embedding_backend=extractor.backend,
            findings=collection.to_list(),
        )
        return report

    @staticmethod
    def _overall_level(total_samples: int, invalid_samples: int, severity_counts: Dict[str, int]) -> str:
        """A simple, documented heuristic -- NOT a calibrated risk score.

        This intentionally stays coarse (LOW/MEDIUM/HIGH) and dataset-
        scoped. Combining this with model- and inference-level findings
        into one overall project risk score is explicitly deferred to a
        later phase.
        """
        invalid_fraction = (invalid_samples / total_samples) if total_samples else 0.0

        if invalid_fraction > _HIGH_INVALID_FRACTION or severity_counts.get("high", 0) > _HIGH_FINDING_COUNT:
            return "HIGH"
        if severity_counts.get("medium", 0) >= _MEDIUM_FINDING_COUNT or severity_counts.get("high", 0) > 0:
            return "MEDIUM"
        if severity_counts.get("low", 0) > 0:
            return "LOW"
        return "LOW"
