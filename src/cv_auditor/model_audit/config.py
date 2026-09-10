"""
config.py (model_audit)
-------------------------
Central configuration for a single Model Integrity Audit run. Mirrors
the flat-dataclass style of Phase 1's ``PipelineConfig`` and Phase 2's
``DatasetAuditConfig``: every tunable in one place, JSON-serializable,
so a saved report can show exactly which settings produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from ..config import COCO_INSTANCE_CATEGORY_NAMES, SUPPORTED_ARCHITECTURES


@dataclass
class ModelAuditConfig:
    """Configuration for a single model-integrity audit run.

    Attributes
    ----------
    candidate_model_path:
        Path to the model file being audited (required).
    reference_model_path:
        Path to a trusted reference model file, used to (re)generate a
        behavioral baseline by running it on ``test_images_dir``. May be
        omitted if ``reference_behavior_path`` already points at a
        previously saved baseline -- the reference weights file itself
        does not need to be kept around indefinitely once its baseline
        has been captured.
    reference_sha256:
        A previously recorded, trusted SHA-256 hash to compare the
        candidate against. If omitted but ``reference_model_path`` is
        given, it is computed from that file instead.
    reference_behavior_path:
        Where to save (if generated fresh) or load (if it already
        exists and ``regenerate_reference_behavior`` is False) the
        reference model's behavioral baseline.
    regenerate_reference_behavior:
        If True, always re-run the reference model even if a baseline
        already exists at ``reference_behavior_path``. If False (the
        default) and a baseline already exists there, it is reused
        without re-running the reference model.
    architecture / num_classes / class_names:
        Passed through to Phase 1's ``ModelLoader`` for both the
        reference and candidate models. Both models must share the same
        architecture/label space for a meaningful comparison.
    test_images_dir:
        The FIXED set of test images both models are run on. Must be
        the same directory for the reference baseline and the candidate
        run, or the comparison is not meaningful (see
        ``comparison.py``'s ``skipped_images``).
    confidence_threshold:
        Predictions below this score are dropped from each model's
        behavioral profile before comparison.
    detector_score_thresh:
        Forwarded to Phase 1's ``PipelineConfig.detector_score_thresh``.
        See that field's docstring -- relevant when auditing
        untrained/randomly-initialized models (as in this sandboxed
        environment), where torchvision's internal filtering can
        otherwise suppress every detection regardless of
        ``confidence_threshold``.
    iou_threshold:
        IoU at/above which a reference and candidate detection are
        considered the "same" box for matching purposes.
    device:
        "cpu" or "cuda".
    output_path:
        Where the structured JSON audit report is written.
    """

    candidate_model_path: str = ""
    reference_model_path: Optional[str] = None
    reference_sha256: Optional[str] = None
    reference_behavior_path: Optional[str] = "results/model_audit/reference_behavior.json"
    regenerate_reference_behavior: bool = False

    architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    num_classes: int = 91
    class_names: list = field(default_factory=lambda: list(COCO_INSTANCE_CATEGORY_NAMES))

    test_images_dir: str = "data/sample_images"
    confidence_threshold: float = 0.5
    detector_score_thresh: Optional[float] = None
    iou_threshold: float = 0.5
    device: str = "cpu"

    candidate_model_name: str = "candidate_model"
    reference_model_name: str = "reference_model"

    unusual_agreement_threshold: float = 0.90
    unusual_confidence_diff_threshold: float = 0.25
    unusual_class_flip_rate_threshold: float = 0.10
    trojan_min_added_detections: int = 5
    trojan_dominant_class_fraction: float = 0.60
    trojan_dominant_class_min_confidence: float = 0.85

    output_path: str = "results/model_audit_report.json"

    def __post_init__(self):
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{self.architecture}'. Supported: {SUPPORTED_ARCHITECTURES}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0, 1]")
        if not self.candidate_model_path:
            raise ValueError("candidate_model_path is required")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ModelAuditConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
