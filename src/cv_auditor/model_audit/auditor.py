"""
auditor.py
----------
``ModelAuditor`` ties fingerprinting, behavioral-baseline generation,
comparison, and suspicion classification together into a single
end-to-end run:

    reference model  --> behavioral baseline (generate or load cached)
    candidate model  --> fingerprint (vs. trusted reference hash)
                      --> behavioral profile (same fixed test images)
                      --> comparison (vs. reference baseline)
                      --> classification --> findings + summary

This module intentionally stops at reporting findings about ONE
candidate model. It does NOT compute a project-wide risk score, does
NOT implement TrojAI-style trigger validation, and does NOT implement
inference-provenance/tamper detection -- all explicitly out of scope for
this phase (see README).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import PipelineConfig
from ..model_loader import ModelLoader
from ..utils import ensure_parent_dir, get_logger, utc_timestamp
from .behavior import BehaviorProfile, generate_behavior_profile, load_test_images
from .comparison import ComparisonResult, compare_profiles
from .config import ModelAuditConfig
from .findings import ModelFindingsCollection
from .fingerprint import FingerprintResult, compare_fingerprint
from .suspicion import SuspicionThresholds, classify

logger = get_logger(__name__)


@dataclass
class ModelAuditReport:
    """The full, structured result of a model-integrity audit run."""

    config: dict
    candidate_model_path: str
    generated_at: str
    fingerprint: dict
    comparison: Optional[dict]
    overall_finding_level: str
    category: str  # "clean" | "file_changed_only" | "unusual_behavior" | "possible_trojan_indicator"
    findings: list

    def to_dict(self) -> dict:
        return {
            "run_metadata": {
                "config": self.config,
                "candidate_model_path": self.candidate_model_path,
                "generated_at": self.generated_at,
            },
            "fingerprint": self.fingerprint,
            "comparison": self.comparison,
            "summary": {
                "overall_finding_level": self.overall_finding_level,
                "category": self.category,
                "num_findings": len(self.findings),
            },
            "findings": self.findings,
        }

    def save(self, path: str) -> str:
        ensure_parent_dir(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved model audit report (%d findings) to %s", len(self.findings), path)
        return path

    def print_summary(self) -> None:
        print("\nModel Integrity Summary\n")
        print(f"Candidate model: {self.candidate_model_path}")
        print(f"Fingerprint changed: {self.fingerprint.get('changed')}")
        if self.comparison:
            print(f"Agreement rate:  {self.comparison.get('agreement_rate')}")
            print(f"Mean confidence diff: {self.comparison.get('mean_confidence_diff')}")
            print(f"Class flip rate: {self.comparison.get('class_flip_rate')}")
        print(f"\nCategory: {self.category}")
        print(f"Overall model finding level: {self.overall_finding_level}")
        print(f"Findings: {len(self.findings)}")
        for f in self.findings:
            print(f"  - [{f['severity'].upper()}] {f['type']}: {f['explanation']}")

    @staticmethod
    def load(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Report file not found: {path}")
        with open(p, "r") as f:
            return json.load(f)


class ModelAuditor:
    """Runs the full Phase 3 model-integrity audit pipeline for one candidate model."""

    def __init__(self, config: Optional[ModelAuditConfig] = None):
        self.config = config or ModelAuditConfig()

    def _get_reference_behavior(self) -> Optional[BehaviorProfile]:
        cfg = self.config

        if not cfg.regenerate_reference_behavior and cfg.reference_behavior_path:
            if Path(cfg.reference_behavior_path).exists():
                logger.info(
                    "Reusing cached reference behavioral baseline from %s",
                    cfg.reference_behavior_path,
                )
                return BehaviorProfile.load(cfg.reference_behavior_path)

        if not cfg.reference_model_path:
            logger.warning(
                "No reference_model_path given and no cached reference behavioral "
                "baseline found; behavior comparison will be skipped."
            )
            return None

        pipeline_cfg = PipelineConfig(
            model_path=cfg.reference_model_path,
            architecture=cfg.architecture,
            model_name=cfg.reference_model_name,
            num_classes=cfg.num_classes,
            device=cfg.device,
            confidence_threshold=cfg.confidence_threshold,
            detector_score_thresh=cfg.detector_score_thresh,
            class_names=list(cfg.class_names),
        )
        model, metadata = ModelLoader(pipeline_cfg).load()
        images = load_test_images(cfg.test_images_dir)
        resolved_device = next(model.parameters()).device
        profile = generate_behavior_profile(
            # Architectures with their own label space (YOLOv5) attach it
            # to the loaded model; torchvision models use the configured list.
            model, metadata, images,
            getattr(model, "class_names", None) or cfg.class_names,
            device=resolved_device,
            confidence_threshold=cfg.confidence_threshold,
            test_images_dir=cfg.test_images_dir,
        )
        if cfg.reference_behavior_path:
            profile.save(cfg.reference_behavior_path)
        return profile

    def _get_reference_hash(self, reference_profile: Optional[BehaviorProfile]) -> Optional[str]:
        cfg = self.config
        if cfg.reference_sha256:
            return cfg.reference_sha256
        if cfg.reference_model_path:
            from ..utils import sha256_of_file
            return sha256_of_file(cfg.reference_model_path)
        if reference_profile is not None:
            return reference_profile.weights_sha256
        return None

    def run(self) -> ModelAuditReport:
        cfg = self.config

        reference_profile = self._get_reference_behavior()
        reference_hash = self._get_reference_hash(reference_profile)

        fingerprint: FingerprintResult = compare_fingerprint(cfg.candidate_model_path, reference_hash)

        pipeline_cfg = PipelineConfig(
            model_path=cfg.candidate_model_path,
            architecture=cfg.architecture,
            model_name=cfg.candidate_model_name,
            num_classes=cfg.num_classes,
            device=cfg.device,
            confidence_threshold=cfg.confidence_threshold,
            detector_score_thresh=cfg.detector_score_thresh,
            class_names=list(cfg.class_names),
        )
        model, metadata = ModelLoader(pipeline_cfg).load()
        images = load_test_images(cfg.test_images_dir)
        resolved_device = next(model.parameters()).device
        candidate_profile = generate_behavior_profile(
            model, metadata, images,
            getattr(model, "class_names", None) or cfg.class_names,
            device=resolved_device,
            confidence_threshold=cfg.confidence_threshold,
            test_images_dir=cfg.test_images_dir,
        )

        comparison: Optional[ComparisonResult] = None
        if reference_profile is not None:
            comparison = compare_profiles(reference_profile, candidate_profile, iou_threshold=cfg.iou_threshold)

        thresholds = SuspicionThresholds(
            unusual_agreement_threshold=cfg.unusual_agreement_threshold,
            unusual_confidence_diff_threshold=cfg.unusual_confidence_diff_threshold,
            unusual_class_flip_rate_threshold=cfg.unusual_class_flip_rate_threshold,
            trojan_min_added_detections=cfg.trojan_min_added_detections,
            trojan_dominant_class_fraction=cfg.trojan_dominant_class_fraction,
            trojan_dominant_class_min_confidence=cfg.trojan_dominant_class_min_confidence,
        )
        findings = classify(cfg.candidate_model_name, fingerprint, comparison, thresholds)

        collection = ModelFindingsCollection()
        collection.extend(findings)

        category, overall_level = self._summarize(fingerprint, collection)

        report = ModelAuditReport(
            config=cfg.to_dict(),
            candidate_model_path=cfg.candidate_model_path,
            generated_at=utc_timestamp(),
            fingerprint=fingerprint.to_dict(),
            comparison=comparison.to_dict() if comparison else None,
            overall_finding_level=overall_level,
            category=category,
            findings=collection.to_list(),
        )
        return report

    @staticmethod
    def _summarize(fingerprint: FingerprintResult, findings: ModelFindingsCollection):
        """A simple, documented heuristic -- NOT a calibrated risk score.

        Mirrors Phase 2's ``DatasetAuditor._overall_level`` in spirit:
        coarse LOW/MEDIUM/HIGH plus a human-readable category label.
        Combining this with dataset- and inference-level findings into a
        project-wide risk score is explicitly deferred to a later phase.
        """
        types = {f.type for f in findings}
        if "possible_trojan_indicator" in types:
            return "possible_trojan_indicator", "HIGH"
        if "unusual_behavior" in types:
            severities = [f.severity for f in findings if f.type == "unusual_behavior"]
            level = "HIGH" if "high" in severities else "MEDIUM"
            return "unusual_behavior", level
        if "model_file_changed" in types:
            return "file_changed_only", "LOW"
        if fingerprint.changed:
            return "file_changed_only", "LOW"
        return "clean", "LOW"
