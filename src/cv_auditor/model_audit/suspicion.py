"""
suspicion.py
-------------
Combines a fingerprint result and a behavior comparison into a list of
structured ``ModelFinding`` objects, using a small, explainable,
tunable rule set.

The rules distinguish three, and only three, kinds of finding:

1. ``model_file_changed`` -- the file hash differs from the trusted
   reference, but behavior on the fixed test set stayed close to the
   reference. This covers legitimate re-exports, metadata changes,
   checkpoint re-saves, minor fine-tunes, etc.
2. ``unusual_behavior`` -- predictions diverge from the reference by
   more than the configured tolerance, broadly (no single class
   dominates the divergence). Could be a legitimate retrain, a version
   mismatch, a different training run -- or something worth a closer
   look. This finding does not say which.
3. ``possible_trojan_indicator`` -- the divergence follows a SPECIFIC,
   targeted pattern: one class accounts for most of the newly-added
   detections, at unusually high and unusually uniform confidence. This
   is the classic *symptom* of a classifier-level backdoor (a "trigger"
   that makes the model confidently predict one target class), but it
   is reported as an indicator to investigate further, NEVER as a
   confirmed trojan/backdoor.

IMPORTANT: this module explicitly does NOT try to detect every possible
backdoor. A sophisticated, narrowly-triggered backdoor (one that only
activates on a specific, rare visual pattern not present in the fixed
test image set) can produce behavior that looks identical to the
reference model on every image checked here, and this baseline WILL NOT
catch it. Confirming or ruling out a backdoor requires targeted
trigger-search techniques (e.g. TrojAI-style validation) that are
explicitly out of scope for this phase -- see README "Known
Limitations".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .comparison import ComparisonResult
from .findings import ModelFinding
from .fingerprint import FingerprintResult


@dataclass
class SuspicionThresholds:
    """Tunable thresholds for the classification rules below.

    All defaults are documented, reasonable starting points -- not a
    calibrated/validated risk model. See README "What the Scores Mean".
    """

    # Below this agreement rate, behavior is considered to have
    # meaningfully diverged (rather than "close enough to call unchanged").
    unusual_agreement_threshold: float = 0.90
    # Above this mean confidence-difference on matched detections,
    # behavior is considered to have meaningfully diverged.
    unusual_confidence_diff_threshold: float = 0.25
    # Above this class-flip rate among matched detections, behavior is
    # considered to have meaningfully diverged.
    unusual_class_flip_rate_threshold: float = 0.10

    # A finding escalates from "unusual_behavior" to
    # "possible_trojan_indicator" only when ALL of the following hold on
    # the "added" (candidate-only) detections:
    trojan_min_added_detections: int = 5           # enough evidence to matter
    trojan_dominant_class_fraction: float = 0.60    # one class dominates
    trojan_dominant_class_min_confidence: float = 0.85  # ...at high confidence


def _behavior_is_unusual(comparison: ComparisonResult, t: SuspicionThresholds) -> bool:
    if comparison.agreement_rate < t.unusual_agreement_threshold:
        return True
    if comparison.mean_confidence_diff is not None and comparison.mean_confidence_diff > t.unusual_confidence_diff_threshold:
        return True
    if comparison.class_flip_rate is not None and comparison.class_flip_rate > t.unusual_class_flip_rate_threshold:
        return True
    return False


def _behavior_looks_targeted(comparison: ComparisonResult, t: SuspicionThresholds) -> bool:
    if comparison.total_added < t.trojan_min_added_detections:
        return False
    if comparison.dominant_added_class_fraction is None:
        return False
    if comparison.dominant_added_class_fraction < t.trojan_dominant_class_fraction:
        return False
    if comparison.dominant_added_class_mean_confidence is None:
        return False
    return comparison.dominant_added_class_mean_confidence >= t.trojan_dominant_class_min_confidence


def classify(
    model_name: str,
    fingerprint: FingerprintResult,
    comparison: Optional[ComparisonResult],
    thresholds: Optional[SuspicionThresholds] = None,
) -> List[ModelFinding]:
    """Produce the finding list for one candidate model.

    ``comparison`` may be None if no reference model/behavior baseline
    was available to compare against (in that case, only a fingerprint
    finding -- if any -- can be produced).
    """
    t = thresholds or SuspicionThresholds()
    findings: List[ModelFinding] = []

    file_changed = bool(fingerprint.changed)

    if comparison is None:
        if file_changed:
            findings.append(
                ModelFinding(
                    type="model_file_changed",
                    severity="low",
                    model=model_name,
                    evidence=(
                        f"candidate sha256={fingerprint.candidate_sha256} != "
                        f"reference sha256={fingerprint.reference_sha256}"
                    ),
                    explanation=(
                        "The candidate model file's hash differs from the trusted "
                        "reference hash. No reference model/behavioral baseline was "
                        "available to check whether this change affected behavior, "
                        "so this is reported as a file-level change only -- not "
                        "evidence of malicious intent."
                    ),
                    comparison_scores={},
                )
            )
        return findings

    unusual = _behavior_is_unusual(comparison, t)
    targeted = unusual and _behavior_looks_targeted(comparison, t)

    scores = {
        "agreement_rate": comparison.agreement_rate,
        "mean_confidence_diff": comparison.mean_confidence_diff,
        "class_flip_rate": comparison.class_flip_rate,
        "missing_rate": comparison.missing_rate,
        "added_rate": comparison.added_rate,
        "dominant_added_class_id": comparison.dominant_added_class_id,
        "dominant_added_class_fraction": comparison.dominant_added_class_fraction,
        "dominant_added_class_mean_confidence": comparison.dominant_added_class_mean_confidence,
    }

    if targeted:
        findings.append(
            ModelFinding(
                type="possible_trojan_indicator",
                severity="high",
                model=model_name,
                evidence=(
                    f"{comparison.total_added} candidate-only detection(s) on the fixed "
                    f"test set; class_id={comparison.dominant_added_class_id} accounts for "
                    f"{comparison.dominant_added_class_fraction:.0%} of them at a mean "
                    f"confidence of {comparison.dominant_added_class_mean_confidence:.3f}"
                ),
                explanation=(
                    "The candidate model produces a large number of new detections "
                    "not present in the reference model's output, heavily concentrated "
                    "on a single class at unusually high, unusually uniform confidence. "
                    "This pattern is CONSISTENT WITH a targeted backdoor/trojan-style "
                    "change (a 'trigger' that makes the model confidently predict one "
                    "target class), but it is NOT confirmed as one -- the same pattern "
                    "can arise from a biased retrain, a corrupted checkpoint, or a class-"
                    "imbalance issue. Confirming an actual backdoor requires targeted "
                    "trigger-search techniques (e.g. TrojAI-style validation) that are "
                    "explicitly out of scope for this phase."
                ),
                comparison_scores=scores,
            )
        )
    elif unusual:
        findings.append(
            ModelFinding(
                type="unusual_behavior",
                severity="medium" if comparison.agreement_rate >= 0.5 else "high",
                model=model_name,
                evidence=(
                    f"agreement_rate={comparison.agreement_rate:.3f}, "
                    f"mean_confidence_diff={comparison.mean_confidence_diff}, "
                    f"class_flip_rate={comparison.class_flip_rate}"
                ),
                explanation=(
                    "The candidate model's predictions on the fixed test set diverge "
                    "from the reference model's by more than the configured tolerance. "
                    "This may be caused by a legitimate retrain/fine-tune, a version "
                    "mismatch, a different random seed, or something else -- this "
                    "finding alone does not indicate malicious intent."
                ),
                comparison_scores=scores,
            )
        )

    if file_changed and not findings:
        findings.append(
            ModelFinding(
                type="model_file_changed",
                severity="low",
                model=model_name,
                evidence=(
                    f"candidate sha256={fingerprint.candidate_sha256} != "
                    f"reference sha256={fingerprint.reference_sha256}"
                ),
                explanation=(
                    "The candidate model file's hash differs from the trusted "
                    "reference hash, but its predictions on the fixed test set stayed "
                    "close to the reference model's. This looks like a low-impact "
                    "change (e.g. re-serialization, metadata update, minor fine-tune) "
                    "rather than a functionally significant one."
                ),
                comparison_scores=scores,
            )
        )
    elif file_changed:
        # Behavior already produced a finding above; note the file change
        # as additional context on that same finding rather than a
        # separate, redundant one.
        findings[-1].extra["model_file_also_changed"] = True
        findings[-1].evidence += (
            f"; also, file hash changed (candidate={fingerprint.candidate_sha256}, "
            f"reference={fingerprint.reference_sha256})"
        )

    return findings
