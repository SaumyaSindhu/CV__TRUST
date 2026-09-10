import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cv_auditor.model_audit.comparison import compare_profiles
from cv_auditor.model_audit.fingerprint import compare_fingerprint
from cv_auditor.model_audit.suspicion import SuspicionThresholds, classify

from .model_audit_helpers import make_prediction, make_profile


def _fingerprint(changed: bool):
    class _FP:
        pass
    fp = _FP()
    fp.changed = changed
    fp.candidate_sha256 = "cand_hash"
    fp.reference_sha256 = "ref_hash" if changed or changed is False else None
    return fp


def test_no_findings_when_unchanged_and_behavior_matches():
    ref = make_profile("ref", "hash_a", {"a.jpg": [make_prediction(3, 0.9)]})
    cand = make_profile("cand", "hash_a", {"a.jpg": [make_prediction(3, 0.9)]})
    comparison = compare_profiles(ref, cand)
    findings = classify("cand", _fingerprint(False), comparison)
    assert findings == []


def test_file_changed_but_behavior_close_gives_low_severity_file_changed_finding():
    ref = make_profile("ref", "hash_a", {"a.jpg": [make_prediction(3, 0.9)]})
    cand = make_profile("cand", "hash_b", {"a.jpg": [make_prediction(3, 0.91)]})  # tiny diff
    comparison = compare_profiles(ref, cand)

    findings = classify("cand", _fingerprint(True), comparison)
    assert len(findings) == 1
    assert findings[0].type == "model_file_changed"
    assert findings[0].severity == "low"


def test_broadly_divergent_behavior_gives_unusual_behavior_finding():
    ref = make_profile("ref", "hash_a", {
        "a.jpg": [make_prediction(3, 0.9, (0, 0, 10, 10))],
        "b.jpg": [make_prediction(4, 0.9, (0, 0, 10, 10))],
        "c.jpg": [make_prediction(5, 0.9, (0, 0, 10, 10))],
    })
    # Candidate detects nothing at all -- large, broad divergence, no
    # single dominant "added" class (there are no added detections).
    cand = make_profile("cand", "hash_b", {"a.jpg": [], "b.jpg": [], "c.jpg": []})
    comparison = compare_profiles(ref, cand)

    findings = classify("cand", _fingerprint(True), comparison)
    types = {f.type for f in findings}
    assert "unusual_behavior" in types
    assert "possible_trojan_indicator" not in types


def test_targeted_single_class_divergence_gives_trojan_indicator():
    ref = make_profile("ref", "hash_a", {"a.jpg": [], "b.jpg": [], "c.jpg": []})
    # Candidate suddenly and confidently detects class 5 everywhere --
    # the classic "backdoor target class" symptom.
    cand = make_profile("cand", "hash_b", {
        "a.jpg": [make_prediction(5, 0.95, (i, i, i + 10, i + 10)) for i in range(0, 30, 10)],
        "b.jpg": [make_prediction(5, 0.96, (i, i, i + 10, i + 10)) for i in range(0, 30, 10)],
        "c.jpg": [make_prediction(5, 0.94, (i, i, i + 10, i + 10)) for i in range(0, 30, 10)],
    })
    comparison = compare_profiles(ref, cand)

    findings = classify("cand", _fingerprint(True), comparison)
    types = {f.type for f in findings}
    assert "possible_trojan_indicator" in types
    trojan_finding = next(f for f in findings if f.type == "possible_trojan_indicator")
    assert trojan_finding.severity == "high"


def test_trojan_indicator_wording_never_confirms_malice():
    ref = make_profile("ref", "hash_a", {"a.jpg": []})
    cand = make_profile("cand", "hash_b", {
        "a.jpg": [make_prediction(5, 0.95, (i, i, i + 10, i + 10)) for i in range(0, 60, 10)],
    })
    comparison = compare_profiles(ref, cand)
    findings = classify("cand", _fingerprint(True), comparison)

    trojan_finding = next(f for f in findings if f.type == "possible_trojan_indicator")
    text = (trojan_finding.explanation + trojan_finding.evidence).lower()
    assert "not confirmed" in trojan_finding.explanation.lower() or "consistent with" in trojan_finding.explanation.lower()
    assert "confirmed backdoor" not in text
    assert "is a trojan" not in text


def test_small_number_of_added_detections_does_not_trigger_trojan_indicator():
    """A handful of added detections for one class shouldn't be enough
    evidence to call it a targeted pattern -- avoids false alarms on tiny
    samples."""
    ref = make_profile("ref", "hash_a", {"a.jpg": []})
    cand = make_profile("cand", "hash_b", {
        "a.jpg": [make_prediction(5, 0.95, (0, 0, 10, 10))],  # only 1 added detection
    })
    comparison = compare_profiles(ref, cand)
    thresholds = SuspicionThresholds(trojan_min_added_detections=5)
    findings = classify("cand", _fingerprint(True), comparison, thresholds)
    types = {f.type for f in findings}
    assert "possible_trojan_indicator" not in types


def test_thresholds_are_configurable():
    ref = make_profile("ref", "hash_a", {"a.jpg": [make_prediction(3, 0.9)]})
    cand = make_profile("cand", "hash_a", {"a.jpg": [make_prediction(3, 0.7)]})  # 0.2 diff
    comparison = compare_profiles(ref, cand)

    lenient = classify("cand", _fingerprint(False), comparison,
                        SuspicionThresholds(unusual_confidence_diff_threshold=0.5))
    strict = classify("cand", _fingerprint(False), comparison,
                       SuspicionThresholds(unusual_confidence_diff_threshold=0.1))

    assert lenient == []
    assert any(f.type == "unusual_behavior" for f in strict)


def test_no_comparison_available_only_reports_fingerprint():
    findings = classify("cand", _fingerprint(True), comparison=None)
    assert len(findings) == 1
    assert findings[0].type == "model_file_changed"


def test_no_comparison_and_unchanged_reports_nothing():
    findings = classify("cand", _fingerprint(False), comparison=None)
    assert findings == []
