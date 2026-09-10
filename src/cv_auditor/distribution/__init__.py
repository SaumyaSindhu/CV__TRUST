"""Phase 5 -- Distribution Shift & Anomaly Detection."""

from .baseline import DistributionBaseline, build_baseline
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
from .auditor import (
    DistributionAuditor,
    DistributionShiftReport,
    RECOMMEND_ACCEPT,
    RECOMMEND_QUARANTINE,
    RECOMMEND_REVIEW,
    STANDARD_LIMITATIONS,
)
from .features import DistributionFeatureExtractor, FeaturizedBatch
from .findings import DIST_FINDING_TYPES, DistributionFinding

__all__ = [
    "DistributionAuditor",
    "DistributionBaseline",
    "DistributionConfig",
    "DistributionFeatureExtractor",
    "DistributionFinding",
    "DistributionShiftReport",
    "DIST_FINDING_TYPES",
    "FeaturizedBatch",
    "PATTERN_CONCENTRATED",
    "PATTERN_MIXED",
    "PATTERN_NONE",
    "PATTERN_UNIFORM",
    "RECOMMEND_ACCEPT",
    "RECOMMEND_QUARANTINE",
    "RECOMMEND_REVIEW",
    "RISK_HIGH",
    "RISK_LOW",
    "RISK_MEDIUM",
    "STANDARD_LIMITATIONS",
    "ShiftComparison",
    "build_baseline",
    "compare_batch",
]
