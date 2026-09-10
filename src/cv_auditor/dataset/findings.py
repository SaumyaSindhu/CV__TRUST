"""
findings.py
-----------
Shared data model for every Phase 2 detector.

Every detector (validation, exact-duplicate, near-duplicate, label-anomaly,
OOD/anomaly) produces a stream of ``Finding`` objects using this single
schema, so later phases (and the summary step at the end of this phase)
can consume them uniformly without knowing which detector produced them.

Design note
-----------
A ``Finding`` is deliberately *evidence*, not a verdict. Nothing in this
module (or anywhere else in the dataset auditor) labels a finding as
"malicious" or "an attack" -- see README.md, section "Findings vs.
Conclusions" for the reasoning.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allowed severities, kept as a plain tuple (like SUPPORTED_ARCHITECTURES in
# config.py) so new severities can be added without touching calling code.
SEVERITIES = ("low", "medium", "high")

# Allowed finding types. Extend this as new detectors are added in later
# phases (e.g. "backdoor_indicator" would be added in Phase 3+, NOT here).
FINDING_TYPES = (
    "invalid_sample",
    "exact_duplicate",
    "near_duplicate",
    "label_anomaly",
    "ood_anomaly",
)

_counter = itertools.count(1)


def _next_finding_id(prefix: str) -> str:
    """Generate a short, human-scannable, unique finding id.

    Not cryptographically unique -- just unique within a single process
    run, which is all that is required for a finding list that gets
    serialized once per audit run.
    """
    return f"{prefix}-{next(_counter):06d}"


@dataclass
class Finding:
    """A single, self-contained piece of dataset-integrity evidence.

    Attributes
    ----------
    finding_id:
        Unique identifier for this finding within the audit run.
    type:
        One of ``FINDING_TYPES``.
    severity:
        One of ``SEVERITIES``. Reflects how noteworthy the evidence is,
        NOT how confident we are that something malicious happened.
    sample:
        The primary sample (image filename) this finding is about.
    related_samples:
        Other samples involved (e.g. the rest of a duplicate group).
    score:
        Optional numeric score backing the finding (similarity, distance,
        anomaly score, etc.). Meaning is detector-specific; see README.
    reason:
        Short, human-readable explanation of what was found.
    evidence:
        Short human-readable description of the concrete evidence backing
        `reason` (e.g. which two hashes matched, which cluster, etc.).
    extra:
        Free-form dict for detector-specific structured details (cluster
        id, thresholds used, bbox, etc.) that don't fit the fixed fields
        above but are useful for a human or downstream phase to inspect.
    """

    type: str
    severity: str
    sample: str
    reason: str
    evidence: str = ""
    related_samples: List[str] = field(default_factory=list)
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    finding_id: str = ""

    def __post_init__(self):
        if self.type not in FINDING_TYPES:
            raise ValueError(f"Unknown finding type '{self.type}'. Known: {FINDING_TYPES}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity '{self.severity}'. Known: {SEVERITIES}")
        if not self.finding_id:
            self.finding_id = _next_finding_id(self.type)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["score"] is not None:
            d["score"] = round(float(d["score"]), 4)
        return d


class FindingsCollection:
    """A simple ordered container of ``Finding`` objects with helpers.

    Kept intentionally thin -- this is not a database, just a convenience
    wrapper used while an audit run is assembling its results.
    """

    def __init__(self) -> None:
        self._findings: List[Finding] = []

    def add(self, finding: Finding) -> None:
        self._findings.append(finding)

    def extend(self, findings: List[Finding]) -> None:
        self._findings.extend(findings)

    def __len__(self) -> int:
        return len(self._findings)

    def __iter__(self):
        return iter(self._findings)

    def by_type(self, finding_type: str) -> List[Finding]:
        return [f for f in self._findings if f.type == finding_type]

    def by_severity(self, severity: str) -> List[Finding]:
        return [f for f in self._findings if f.severity == severity]

    def counts_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {t: 0 for t in FINDING_TYPES}
        for f in self._findings:
            counts[f.type] += 1
        return counts

    def counts_by_severity(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s: 0 for s in SEVERITIES}
        for f in self._findings:
            counts[f.severity] += 1
        return counts

    def to_list(self) -> List[dict]:
        return [f.to_dict() for f in self._findings]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_list(), fh, indent=2)
