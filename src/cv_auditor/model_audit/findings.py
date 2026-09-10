"""
findings.py (model_audit)
--------------------------
Shared finding schema for the Phase 3 Model Integrity Auditor.

Deliberately a separate, smaller schema from Phase 2's
``cv_auditor.dataset.findings.Finding`` -- the requested fields for a
model-audit finding (``model``, ``comparison_scores``, ``explanation``)
don't map cleanly onto the dataset schema (``sample``, ``related_samples``,
``score``), and forcing them into one shared shape would make both harder
to read. The *philosophy* is identical though: a finding is evidence, not
a verdict. See README, "Findings vs. Conclusions" (Phase 2) and
"Changed vs. Unusual vs. Possible Trojan" (Phase 3) for why.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SEVERITIES = ("low", "medium", "high")

# "model_file_changed"        -- the candidate's file hash differs from the
#                                 trusted/reference hash. Not itself good or
#                                 bad -- see README.
# "unusual_behavior"          -- the candidate's predictions on the fixed
#                                 test set diverge from the reference
#                                 model's by more than the configured
#                                 tolerance, without a targeted pattern.
# "possible_trojan_indicator" -- the divergence follows a specific,
#                                 targeted pattern (a single class
#                                 dominating newly-added, high-confidence
#                                 detections) that is CONSISTENT WITH but
#                                 does not confirm a backdoor/trojan-style
#                                 change. Never reported as "confirmed".
FINDING_TYPES = (
    "model_file_changed",
    "unusual_behavior",
    "possible_trojan_indicator",
)

_counter = itertools.count(1)


def _next_finding_id(prefix: str) -> str:
    return f"{prefix}-{next(_counter):06d}"


@dataclass
class ModelFinding:
    """A single piece of model-integrity evidence.

    Attributes
    ----------
    type:
        One of ``FINDING_TYPES``.
    severity:
        One of ``SEVERITIES``. Reflects how noteworthy the evidence is,
        NOT confidence that something malicious happened.
    model:
        Identifier for the candidate model this finding is about
        (typically its ``model_name`` from ``ModelMetadata``).
    evidence:
        Concrete, checkable evidence backing this finding (hashes,
        specific counts, specific classes involved).
    comparison_scores:
        The relevant subset of the reference-vs-candidate comparison
        metrics (see ``comparison.py``) that led to this finding.
    explanation:
        Plain-language explanation of what was found and, importantly,
        what it does and doesn't mean.
    """

    type: str
    severity: str
    model: str
    evidence: str
    explanation: str
    comparison_scores: Dict[str, Any] = field(default_factory=dict)
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
        return asdict(self)


class ModelFindingsCollection:
    """Thin ordered container of ``ModelFinding`` objects, mirroring
    Phase 2's ``FindingsCollection`` for consistency."""

    def __init__(self) -> None:
        self._findings: List[ModelFinding] = []

    def add(self, finding: ModelFinding) -> None:
        self._findings.append(finding)

    def extend(self, findings: List[ModelFinding]) -> None:
        self._findings.extend(findings)

    def __len__(self) -> int:
        return len(self._findings)

    def __iter__(self):
        return iter(self._findings)

    def by_type(self, finding_type: str) -> List[ModelFinding]:
        return [f for f in self._findings if f.type == finding_type]

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
