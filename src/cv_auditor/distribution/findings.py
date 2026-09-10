"""
findings.py (distribution)
--------------------------
Distribution-shift findings, following exactly the evidence-not-verdict
pattern of Phase 2's dataset ``Finding`` and Phase 3's ``ModelFinding``:
same field names, own type vocabulary.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

DIST_FINDING_TYPES = (
    "batch_distribution_shift",   # the batch as a whole moved off-baseline
    "anomalous_sample",           # one image is individually far off-baseline
    "high_anomalous_fraction",    # unusually many individually-anomalous images
    "dispersion_change",          # spread changed drastically vs baseline
    "corrupt_image",              # image could not be decoded/featurized
    "batch_too_small",            # not enough usable images to assess
    "feature_backend_mismatch",   # baseline and batch featurized differently
)

SEVERITIES = ("low", "medium", "high")

_counter = itertools.count(1)


def _next_id(prefix: str) -> str:
    return f"DIST-{prefix.upper()[:4]}-{next(_counter):04d}"


@dataclass
class DistributionFinding:
    """One piece of distribution-shift evidence. NEVER a verdict: a
    statistical anomaly does not establish manipulation or intent."""

    type: str
    severity: str
    sample: str                      # image filename or "<batch>"
    reason: str
    evidence: str = ""
    related_samples: List[str] = field(default_factory=list)
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    finding_id: str = ""

    def __post_init__(self):
        if self.type not in DIST_FINDING_TYPES:
            raise ValueError(f"Unknown finding type '{self.type}'. Known: {DIST_FINDING_TYPES}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity '{self.severity}'. Known: {SEVERITIES}")
        if not self.finding_id:
            self.finding_id = _next_id(self.type)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["score"] is not None:
            d["score"] = round(float(d["score"]), 4)
        return d
