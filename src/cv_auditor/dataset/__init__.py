"""
cv_auditor.dataset - Phase 2: Dataset Integrity Auditor
=========================================================

Analyzes a computer-vision dataset for evidence worth investigating:
exact/near duplicates, suspicious label/image relationships, and
visually or statistically unusual samples.

This package produces *findings* (evidence), not verdicts. See
README.md, section "Findings vs. Conclusions".
"""

from .config import DatasetAuditConfig
from .dataset_loader import DatasetLoader, DatasetIndex, Sample
from .findings import Finding, FindingsCollection
from .auditor import DatasetAuditor, DatasetAuditReport

__all__ = [
    "DatasetAuditConfig",
    "DatasetLoader",
    "DatasetIndex",
    "Sample",
    "Finding",
    "FindingsCollection",
    "DatasetAuditor",
    "DatasetAuditReport",
]
