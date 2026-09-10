"""
cv_auditor.model_audit - Phase 3: Model Integrity Auditor
============================================================

Checks whether a candidate computer-vision object-detection model is
consistent with a trusted/reference model:

  * Fingerprint -- has the model FILE changed (SHA-256 vs. a trusted hash)?
  * Behavior -- does the model BEHAVE differently on a fixed set of test
    images (predictions, confidences) compared to the reference model?
  * Suspicion classification -- combines both signals into one of three
    finding types: a changed file, unusual behavior, or a targeted
    pattern consistent with (but not confirming) a backdoor/trojan.

This package produces *findings* (evidence), not verdicts -- see
README.md, "Changed vs. Unusual vs. Possible Trojan".
"""

from .config import ModelAuditConfig
from .fingerprint import FingerprintResult, compare_fingerprint, compute_fingerprint
from .behavior import BehaviorProfile, generate_behavior_profile, load_test_images
from .comparison import ComparisonResult, compare_profiles
from .findings import ModelFinding, ModelFindingsCollection
from .suspicion import SuspicionThresholds, classify
from .auditor import ModelAuditor, ModelAuditReport

__all__ = [
    "ModelAuditConfig",
    "FingerprintResult",
    "compare_fingerprint",
    "compute_fingerprint",
    "BehaviorProfile",
    "generate_behavior_profile",
    "load_test_images",
    "ComparisonResult",
    "compare_profiles",
    "ModelFinding",
    "ModelFindingsCollection",
    "SuspicionThresholds",
    "classify",
    "ModelAuditor",
    "ModelAuditReport",
]
