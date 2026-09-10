"""
auditor.py (distribution)
-------------------------
Runs the full Phase 5 analysis for one incoming batch against a stored
baseline and produces a ``DistributionShiftReport`` in the project's
run_metadata / summary / findings shape, extended with the Phase 5
schema fields (shift_score, anomalous_fraction, confidence, risk,
interpretation, recommendation, limitations).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .. import __version__ as project_version
from ..utils import find_images, get_logger, utc_timestamp
from .baseline import DistributionBaseline, build_baseline  # noqa: F401 (re-export)
from .comparison import (
    PATTERN_CONCENTRATED,
    PATTERN_MIXED,
    PATTERN_NONE,
    PATTERN_UNIFORM,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    ShiftComparison,
    compare_batch,
)
from .config import DistributionConfig
from .features import DistributionFeatureExtractor
from .findings import DistributionFinding

logger = get_logger(__name__)

REPORT_SCHEMA_VERSION = "1.0"

RECOMMEND_ACCEPT, RECOMMEND_REVIEW, RECOMMEND_QUARANTINE = (
    "ACCEPT", "REVIEW", "QUARANTINE",
)

STANDARD_LIMITATIONS = [
    "Distribution shift is statistical evidence only: it does not establish "
    "manipulation, malicious intent, or an attack. Terrain, season, sensor, "
    "illumination, and other legitimate acquisition changes also shift "
    "distributions.",
    "Features summarize coarse color/texture/acquisition structure; a shift "
    "invisible to these features (or an attack crafted to preserve them) "
    "will not be detected.",
    "Thresholds are calibrated against the baseline's own internal "
    "variability; a baseline that is too small, too uniform, or itself "
    "compromised weakens every downstream conclusion.",
    "Per-sample anomaly detection flags statistical outliers, which can be "
    "rare-but-legitimate content just as easily as injected content.",
]


def _interpretation(cmp: ShiftComparison) -> str:
    declared = ", ".join(cmp.declared_changes) if cmp.declared_changes else ""
    if cmp.num_current == 0:
        return "No usable images in the incoming batch; nothing could be assessed."
    if cmp.shift_pattern == PATTERN_NONE:
        return (
            "The incoming batch is statistically consistent with the trusted "
            "baseline distribution (no significant batch-level displacement, "
            "few or no individually anomalous samples)."
        )
    if cmp.shift_pattern == PATTERN_UNIFORM:
        base = (
            "The batch has shifted as a whole while staying internally "
            "coherent -- the signature of a domain/environment change "
            "(e.g. terrain, season, sensor, illumination) rather than of a "
            "few injected samples."
        )
        if declared:
            base += (
                f" Declared acquisition changes ({declared}) make this drift "
                f"expected; it is NOT automatically suspicious."
            )
        else:
            base += (
                " No acquisition change was declared, so the cause should be "
                "confirmed with the data provider; drift like this is common "
                "and not by itself evidence of manipulation."
            )
        return base
    if cmp.shift_pattern == PATTERN_CONCENTRATED:
        return (
            "Most of the batch matches the baseline, but a small subset of "
            "images is individually far outside it. Concentrated outliers "
            "are the pattern that most warrants inspection of the specific "
            "flagged samples -- though rare legitimate content produces the "
            "same signature; this is evidence, not proof of injection."
        )
    return (
        "The batch shows both overall displacement and heterogeneous "
        "per-sample anomalies (mixed pattern). Manual review of the flagged "
        "samples and of the acquisition conditions is recommended before "
        "drawing any conclusion."
    )


def _recommendation(cmp: ShiftComparison) -> str:
    if cmp.num_current == 0:
        return RECOMMEND_REVIEW
    if cmp.risk == RISK_LOW:
        return RECOMMEND_ACCEPT
    if cmp.risk == RISK_MEDIUM:
        return RECOMMEND_REVIEW
    # HIGH: uniform drift with a declared acquisition change is expected
    # drift -> a human should confirm rather than the pipeline quarantining.
    if cmp.shift_pattern == PATTERN_UNIFORM and cmp.declared_changes:
        return RECOMMEND_REVIEW
    return RECOMMEND_QUARANTINE


def _headline_finding(cmp: ShiftComparison) -> str:
    if cmp.num_current == 0:
        return "empty_or_unreadable_batch"
    if cmp.risk == RISK_LOW:
        return "no_significant_distribution_shift"
    return f"{cmp.shift_pattern} ({cmp.risk.lower()} risk)"


@dataclass
class DistributionShiftReport:
    baseline: DistributionBaseline
    comparison: ShiftComparison
    findings: List[DistributionFinding]
    batch_dir: str
    corrupt_images: List[str]
    current_metadata: Optional[Dict]
    generated_at: str

    # ------------------------------------------------------------------ #
    @property
    def risk(self) -> str:
        return self.comparison.risk

    @property
    def recommendation(self) -> str:
        return _recommendation(self.comparison)

    def to_dict(self) -> dict:
        cmp = self.comparison
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_metadata": {
                "tool": "cv_auditor.distribution",
                "tool_version": project_version,
                "generated_at": self.generated_at,
                "batch_dir": self.batch_dir,
            },
            "baseline": self.baseline.summary_dict(),
            "current_batch": {
                "num_images": cmp.num_current,
                "num_corrupt": len(self.corrupt_images),
                "corrupt_images": self.corrupt_images,
                "metadata": self.current_metadata,
            },
            "shift_score": cmp.shift_score,
            "anomalous_fraction": cmp.anomalous_fraction,
            "confidence": cmp.confidence,
            "risk": cmp.risk,
            "finding": _headline_finding(cmp),
            "evidence": cmp.evidence_dict(),
            "interpretation": _interpretation(cmp),
            "recommendation": self.recommendation,
            "limitations": list(STANDARD_LIMITATIONS),
            "summary": {
                "risk": cmp.risk,
                "shift_score": cmp.shift_score,
                "anomalous_fraction": cmp.anomalous_fraction,
                "confidence": cmp.confidence,
                "shift_pattern": cmp.shift_pattern,
                "recommendation": self.recommendation,
                "total_findings": len(self.findings),
            },
            "findings": [f.to_dict() for f in self.findings],
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Distribution shift report saved to %s", path)

    def print_summary(self) -> None:
        cmp = self.comparison
        print("\nDistribution Shift Summary\n")
        print(f"Baseline:            {self.baseline.source_dir} "
              f"({self.baseline.num_samples} images)")
        print(f"Incoming batch:      {self.batch_dir} ({cmp.num_current} images, "
              f"{len(self.corrupt_images)} corrupt)")
        print(f"Shift score:         {cmp.shift_score}")
        print(f"Anomalous fraction:  {cmp.anomalous_fraction}")
        print(f"Pattern:             {cmp.shift_pattern}")
        print(f"Risk:                {cmp.risk}   (confidence {cmp.confidence})")
        print(f"Recommendation:      {self.recommendation}")
        print(f"\n{_interpretation(cmp)}")


class DistributionAuditor:
    """Phase 5 orchestrator: featurize -> compare -> findings -> report."""

    def __init__(self, config: Optional[DistributionConfig] = None,
                 extractor: Optional[DistributionFeatureExtractor] = None):
        self.config = config or DistributionConfig()
        self.extractor = extractor or DistributionFeatureExtractor(
            device=self.config.device,
            cache_dir=self.config.embedding_cache_dir,
        )

    # ------------------------------------------------------------------ #
    def _findings(self, cmp: ShiftComparison, corrupt: List[str],
                  backend_mismatch: Optional[str]) -> List[DistributionFinding]:
        findings: List[DistributionFinding] = []

        if backend_mismatch:
            findings.append(DistributionFinding(
                type="feature_backend_mismatch", severity="high", sample="<batch>",
                reason="Baseline and incoming batch were featurized with "
                       "different backends; distances are not comparable.",
                evidence=backend_mismatch,
            ))
            return findings

        for path, err in corrupt:
            findings.append(DistributionFinding(
                type="corrupt_image", severity="medium", sample=Path(path).name,
                reason="Image could not be decoded/featurized and was "
                       "excluded from the statistical analysis.",
                evidence=err,
            ))

        if cmp.num_current == 0:
            findings.append(DistributionFinding(
                type="batch_too_small", severity="medium", sample="<batch>",
                reason="No usable images in the incoming batch; distribution "
                       "shift could not be assessed.",
                evidence=f"{len(corrupt)} corrupt/unreadable file(s)",
            ))
            return findings

        for s in cmp.sample_scores:
            if s.anomalous:
                findings.append(DistributionFinding(
                    type="anomalous_sample", severity="medium",
                    sample=Path(s.image).name, score=s.z,
                    reason="Image lies far outside the trusted baseline "
                           "distribution.",
                    evidence=(f"Mahalanobis {s.mahalanobis:.3f} is z={s.z:.2f} "
                              f"baseline std-devs above the baseline's own "
                              f"mean self-distance "
                              f"(threshold z>={self.config.sample_anomaly_z})"),
                ))

        anomalous_count = sum(s.anomalous for s in cmp.sample_scores)
        if (cmp.anomalous_fraction >= self.config.anomalous_fraction_medium
                and anomalous_count >= self.config.min_anomalous_count):
            severity = ("high" if cmp.anomalous_fraction
                        >= self.config.anomalous_fraction_high else "medium")
            findings.append(DistributionFinding(
                type="high_anomalous_fraction", severity=severity,
                sample="<batch>", score=cmp.anomalous_fraction,
                reason="An unusually large fraction of the batch is "
                       "individually anomalous vs the baseline.",
                evidence=f"{anomalous_count}/{cmp.num_current} images flagged "
                         f"(fraction {cmp.anomalous_fraction})",
                related_samples=[Path(s.image).name for s in cmp.sample_scores
                                 if s.anomalous],
            ))

        if cmp.batch_shift_z >= self.config.batch_shift_z_medium:
            severity = ("high" if cmp.batch_shift_z
                        >= self.config.batch_shift_z_high else "medium")
            findings.append(DistributionFinding(
                type="batch_distribution_shift", severity=severity,
                sample="<batch>", score=cmp.batch_shift_z,
                reason="The incoming batch as a whole is displaced from the "
                       "trusted baseline distribution.",
                evidence=(f"mean per-sample z={cmp.batch_shift_z:.2f} "
                          f"(medium>={self.config.batch_shift_z_medium}, "
                          f"high>={self.config.batch_shift_z_high}); "
                          f"centroid Mahalanobis {cmp.centroid_mahalanobis:.3f}; "
                          f"MMD ratio vs within-baseline null: {cmp.mmd_ratio}"),
                extra={"shift_pattern": cmp.shift_pattern},
            ))

        if (cmp.num_current > 1 and
                (cmp.dispersion_ratio >= self.config.dispersion_ratio_threshold
                 or cmp.dispersion_ratio <= 1.0 / self.config.dispersion_ratio_threshold)):
            findings.append(DistributionFinding(
                type="dispersion_change", severity="low", sample="<batch>",
                score=cmp.dispersion_ratio,
                reason="The batch's spread differs strongly from the "
                       "baseline's own spread.",
                evidence=f"dispersion ratio {cmp.dispersion_ratio:.2f} "
                         f"(threshold {self.config.dispersion_ratio_threshold}x)",
            ))
        return findings

    # ------------------------------------------------------------------ #
    def run(
        self,
        batch_dir: str,
        baseline: Optional[DistributionBaseline] = None,
        current_metadata: Optional[Dict] = None,
    ) -> DistributionShiftReport:
        cfg = self.config
        baseline = baseline or DistributionBaseline.load(cfg.baseline_path)

        try:
            image_paths = find_images(batch_dir)
        except FileNotFoundError:
            if not Path(batch_dir).exists():
                raise  # a missing path is a caller error, not an empty batch
            image_paths = []  # existing directory with no images: assess as empty
        batch = self.extractor.extract_batch(image_paths)

        backend_mismatch = None
        if baseline.feature_backend != self.extractor.backend:
            backend_mismatch = (
                f"baseline backend '{baseline.feature_backend}' != "
                f"batch backend '{self.extractor.backend}'"
            )

        corrupt_fraction = (
            batch.num_corrupt / max(len(image_paths), 1)
        )
        if backend_mismatch:
            cmp = compare_batch(
                baseline, batch.features[:0].reshape(0, 0), [], cfg,
                pretrained=self.extractor.pretrained_embeddings,
                corrupt_fraction=corrupt_fraction,
                current_metadata=current_metadata,
            )
        else:
            cmp = compare_batch(
                baseline, batch.features, batch.image_paths, cfg,
                pretrained=self.extractor.pretrained_embeddings,
                corrupt_fraction=corrupt_fraction,
                current_metadata=current_metadata,
            )

        findings = self._findings(
            cmp, list(zip(batch.corrupt_images, batch.corrupt_errors)),
            backend_mismatch,
        )
        report = DistributionShiftReport(
            baseline=baseline,
            comparison=cmp,
            findings=findings,
            batch_dir=str(batch_dir),
            corrupt_images=[Path(p).name for p in batch.corrupt_images],
            current_metadata=current_metadata,
            generated_at=utc_timestamp(),
        )
        logger.info(
            "Distribution audit: risk=%s shift_score=%.3f anomalous=%.2f "
            "pattern=%s findings=%d",
            cmp.risk, cmp.shift_score, cmp.anomalous_fraction,
            cmp.shift_pattern, len(findings),
        )
        return report
