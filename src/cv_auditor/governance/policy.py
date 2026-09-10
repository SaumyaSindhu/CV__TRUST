"""
policy.py (governance)
-----------------------
The severity -> recommendation policy and the wording rules that keep
Phase 6 honest. Every mapping decision here returns not just a label but
is backed by an explainable rule; ``decision_reasons`` in the unified
report records which rules fired.

Core mapping (with documented exceptions):
    low    -> ACCEPT   (exception: a changed model file alone -> REVIEW,
                        because a human should confirm the change was an
                        expected update, even when behavior is consistent)
    medium -> REVIEW
    high   -> QUARANTINE (exception: Phase 5 may itself soften a HIGH
                        uniform drift with a DECLARED acquisition change
                        to REVIEW; governance respects that judgment)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

TROJAN_WORDING = (
    "Behavioral indicators are consistent with a targeted backdoor/"
    "trojan-style modification. This is an indicator, NOT a confirmed "
    "backdoor: a biased retrain, a corrupted checkpoint, or a legitimate "
    "class-distribution fix can produce the same statistical pattern."
)

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
SEVERITY_TO_RISK = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


def recommend_for(
    severity: str,
    category: str,
    agreement_rate: Optional[float] = None,
    distribution_recommendation: Optional[str] = None,
) -> str:
    """Recommendation for a single finding."""
    # Phase 5 already reasons about declared drift; respect its judgment
    # for distribution findings when it softened the outcome.
    if distribution_recommendation in ("ACCEPT", "REVIEW", "QUARANTINE"):
        if severity == "high" and distribution_recommendation == "REVIEW":
            return "REVIEW"

    if severity == "high":
        return "QUARANTINE"
    if severity == "medium":
        return "REVIEW"
    # low:
    if category == "model_file_changed":
        # changed file, consistent behavior -> human confirms it was an
        # expected update rather than auto-accepting a modified artifact
        if agreement_rate is not None and agreement_rate >= 0.99:
            return "ACCEPT"
        return "REVIEW"
    return "ACCEPT"


def safe_model_explanation(category: str, finding: dict,
                           agreement_rate: Optional[float]) -> str:
    """Analyst wording for model findings, with the trojan-indicator
    wording rule enforced centrally."""
    base = finding.get("description") or finding.get("reason") or ""
    if category == "possible_trojan_indicator":
        return f"{TROJAN_WORDING} Underlying evidence: {base}".strip()
    if category == "model_file_changed":
        note = (
            " Behavioral comparison remained consistent "
            f"(agreement rate {agreement_rate})." if (agreement_rate or 0) >= 0.99
            else ""
        )
        return (f"The model file's fingerprint differs from the trusted "
                f"reference. A changed file is not by itself malicious "
                f"(re-serialization, retraining, or updates also change "
                f"it).{note} {base}").strip()
    if category == "unusual_behavior":
        return (f"The candidate's behavior deviates from the reference "
                f"beyond configured thresholds. This is anomalous, not "
                f"proven-malicious. {base}").strip()
    return base


def overall_decision(
    findings: List[dict],
    section_statuses: Dict[str, Optional[dict]],
) -> Tuple[str, float, str, List[str]]:
    """Aggregate finding-level evidence into (overall_risk,
    overall_confidence, recommendation, decision_reasons)."""
    reasons: List[str] = []

    active_sections = {k: v for k, v in section_statuses.items() if v is not None}
    if not findings:
        if active_sections:
            reasons.append(
                "No findings were produced by any executed phase "
                f"({', '.join(sorted(active_sections))})."
            )
            conf = 0.85
            if len(active_sections) < 4:
                conf *= 0.75 + 0.0625 * len(active_sections)
                reasons.append(
                    f"Only {len(active_sections)} of 4 evidence layers were "
                    f"executed; overall confidence is reduced accordingly."
                )
            return "LOW", round(conf, 3), "ACCEPT", reasons
        reasons.append("No phases were executed; nothing was assessed.")
        return "LOW", 0.0, "REVIEW", reasons

    overall = "LOW"
    for f in findings:
        risk = SEVERITY_TO_RISK.get(f["severity"], "LOW")
        if RISK_ORDER[risk] > RISK_ORDER[overall]:
            overall = risk

    # recommendation = strictest individual recommendation
    rec_order = {"ACCEPT": 0, "REVIEW": 1, "QUARANTINE": 2}
    recommendation = max(
        (f["recommendation"] for f in findings), key=lambda r: rec_order[r],
    )

    high = [f for f in findings if f["severity"] == "high"]
    medium = [f for f in findings if f["severity"] == "medium"]
    if high:
        cats = sorted({f["category"] for f in high})
        reasons.append(
            f"{len(high)} high-severity finding(s) ({', '.join(cats)}) "
            f"drive the overall risk to HIGH."
        )
        if any(f["category"] == "possible_trojan_indicator" for f in high):
            reasons.append(TROJAN_WORDING)
        if any(f["source_phase"].startswith("phase4") for f in high):
            reasons.append(
                "Provenance verification failures mean the inference "
                "artifacts can no longer be trusted as recorded."
            )
    elif medium:
        cats = sorted({f["category"] for f in medium})
        reasons.append(
            f"{len(medium)} medium-severity finding(s) ({', '.join(cats)}) "
            f"warrant analyst review; no high-severity evidence was found."
        )
    else:
        reasons.append(
            "Only low-severity evidence was found; assets appear consistent "
            "with their references."
        )

    # confidence: mean of finding confidences, damped when phases are missing
    conf = sum(f["confidence"] for f in findings) / len(findings)
    executed = len(active_sections)
    if executed < 4:
        conf *= 0.75 + 0.0625 * executed  # fewer evidence layers -> less certainty
        reasons.append(
            f"Only {executed} of 4 evidence layers were executed; overall "
            f"confidence is reduced accordingly."
        )
    return overall, round(conf, 3), recommendation, reasons
