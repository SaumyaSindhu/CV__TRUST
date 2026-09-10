"""
finding_model.py (governance)
------------------------------
Phase 6's normalized finding: one uniform, analyst-facing shape for
evidence produced by ANY earlier phase.

Each ``GovernanceFinding`` carries: finding ID, timestamp, affected
asset, source phase, category, severity, confidence, evidence,
explanation, recommendation, and limitations -- and the normalizers here
translate every phase's native finding format into it WITHOUT rewording
the underlying evidence into stronger claims than the source phase made.

Wording discipline (deliberate, enforced here and tested):
- ``possible_trojan_indicator`` is presented as "behavioral indicators
  consistent with a targeted backdoor/trojan-style modification" --
  never as a confirmed backdoor.
- Statistical anomalies (dataset OOD, distribution shift) are presented
  as anomalies, with an explicit note that they do not establish
  malicious intent.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from ..utils import utc_timestamp

SEVERITIES = ("low", "medium", "high")
RECOMMENDATIONS = ("ACCEPT", "REVIEW", "QUARANTINE")

PHASE_DATASET = "phase2_dataset_integrity"
PHASE_MODEL = "phase3_model_integrity"
PHASE_PROVENANCE = "phase4_inference_provenance"
PHASE_DISTRIBUTION = "phase5_distribution_shift"

_counter = itertools.count(1)


def _next_id(phase: str) -> str:
    short = {"phase2_dataset_integrity": "DS", "phase3_model_integrity": "MD",
             "phase4_inference_provenance": "PR",
             "phase5_distribution_shift": "DI"}.get(phase, "GN")
    return f"F-{short}-{next(_counter):04d}"


# base confidence that a finding of this severity, from this kind of
# detector, reflects a real (not spurious) technical observation --
# NOT the probability of malice, which no phase can estimate.
_SEVERITY_CONFIDENCE = {"low": 0.5, "medium": 0.65, "high": 0.8}

STATISTICAL_DISCLAIMER = (
    "This is a statistical anomaly: it does not by itself establish "
    "manipulation or malicious intent."
)


@dataclass
class GovernanceFinding:
    finding_id: str
    timestamp: str
    affected_asset: str
    source_phase: str
    category: str
    severity: str
    confidence: float
    evidence: Dict[str, Any]
    explanation: str
    recommendation: str
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(float(d["confidence"]), 3)
        return d


def make_finding(
    *, affected_asset: str, source_phase: str, category: str, severity: str,
    evidence: Dict[str, Any], explanation: str, recommendation: str,
    confidence: Optional[float] = None, limitations: Optional[List[str]] = None,
) -> GovernanceFinding:
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity '{severity}'")
    if recommendation not in RECOMMENDATIONS:
        raise ValueError(f"Unknown recommendation '{recommendation}'")
    return GovernanceFinding(
        finding_id=_next_id(source_phase),
        timestamp=utc_timestamp(),
        affected_asset=affected_asset,
        source_phase=source_phase,
        category=category,
        severity=severity,
        confidence=(confidence if confidence is not None
                    else _SEVERITY_CONFIDENCE[severity]),
        evidence=evidence,
        explanation=explanation,
        recommendation=recommendation,
        limitations=list(limitations or []),
    )


# --------------------------------------------------------------------- #
# per-phase normalizers (dicts in -> GovernanceFindings out)
# --------------------------------------------------------------------- #
def from_dataset_report(report: dict, asset: str) -> List[GovernanceFinding]:
    from .policy import recommend_for

    out = []
    for f in report.get("findings", []):
        category = f.get("type", "dataset_finding")
        severity = f.get("severity", "low")
        statistical = category in ("potential_ood_anomaly", "label_anomaly",
                                   "near_duplicate")
        explanation = f.get("reason", "")
        if statistical:
            explanation = f"{explanation} {STATISTICAL_DISCLAIMER}"
        out.append(make_finding(
            affected_asset=f"{asset}::{f.get('sample', '')}",
            source_phase=PHASE_DATASET,
            category=category, severity=severity,
            evidence={"detail": f.get("evidence", ""),
                      "score": f.get("score"),
                      "related_samples": f.get("related_samples", []),
                      "native_finding_id": f.get("finding_id")},
            explanation=explanation.strip(),
            recommendation=recommend_for(severity, category),
            limitations=[
                "Dataset findings are evidence for review; a dataset anomaly "
                "does not automatically mean poisoning."
            ],
        ))
    return out


def from_model_report(report: dict, asset: str) -> List[GovernanceFinding]:
    from .policy import recommend_for, safe_model_explanation

    out = []
    agreement = (report.get("comparison") or {}).get("agreement_rate")
    for f in report.get("findings", []):
        category = f.get("type", "model_finding")
        severity = f.get("severity", "low")
        out.append(make_finding(
            affected_asset=asset,
            source_phase=PHASE_MODEL,
            category=category, severity=severity,
            evidence={"detail": f.get("evidence", f.get("description", "")),
                      "score": f.get("score"),
                      "agreement_rate": agreement,
                      "native_finding_id": f.get("finding_id")},
            explanation=safe_model_explanation(category, f, agreement),
            recommendation=recommend_for(severity, category,
                                         agreement_rate=agreement),
            limitations=[
                "Behavioral comparison covers only the fixed test-image set; "
                "triggers outside it are invisible to this analysis.",
            ],
        ))
    return out


def from_provenance_report(report: dict, asset: str) -> List[GovernanceFinding]:
    from .policy import recommend_for

    out = []
    summary = report.get("summary", {})
    for f in report.get("findings", []):
        code = f.get("code", "provenance_finding")
        severity = "high" if code != "VALID" else "low"
        out.append(make_finding(
            affected_asset=asset,
            source_phase=PHASE_PROVENANCE,
            category=code, severity=severity,
            evidence={"component": f.get("component"),
                      "detail": f.get("evidence", ""),
                      "hash_integrity": summary.get("hash_integrity"),
                      "signature_status": summary.get("signature_status"),
                      "replay_status": summary.get("replay_status")},
            explanation=f.get("explanation", ""),
            recommendation=recommend_for(severity, code),
            limitations=[
                "Hashing proves content identity, not authorship; the HMAC "
                "signature is symmetric (any verifier could also sign).",
            ],
        ))
    return out


def from_distribution_report(report: dict, asset: str) -> List[GovernanceFinding]:
    from .policy import recommend_for

    out = []
    interpretation = report.get("interpretation", "")
    for f in report.get("findings", []):
        category = f.get("type", "distribution_finding")
        severity = f.get("severity", "low")
        explanation = f.get("reason", "")
        if category in ("batch_distribution_shift", "anomalous_sample",
                        "high_anomalous_fraction", "dispersion_change"):
            explanation = f"{explanation} {STATISTICAL_DISCLAIMER}"
        out.append(make_finding(
            affected_asset=f"{asset}::{f.get('sample', '')}",
            source_phase=PHASE_DISTRIBUTION,
            category=category, severity=severity,
            confidence=min(_SEVERITY_CONFIDENCE[severity],
                           float(report.get("confidence", 1.0)) or 0.0)
            if report.get("confidence") is not None else None,
            evidence={"detail": f.get("evidence", ""),
                      "score": f.get("score"),
                      "shift_pattern": (report.get("evidence") or {}).get("shift_pattern"),
                      "native_finding_id": f.get("finding_id")},
            explanation=explanation.strip(),
            recommendation=recommend_for(
                severity, category,
                distribution_recommendation=report.get("recommendation"),
            ),
            limitations=[
                "Distribution shift can be legitimate (terrain, season, "
                "sensor, illumination); see the phase interpretation: "
                + interpretation[:200],
            ],
        ))
    return out
