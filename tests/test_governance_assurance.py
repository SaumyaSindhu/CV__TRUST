"""
test_governance_assurance.py
-----------------------------
Phase 6 tests: finding normalization, wording discipline, policy
mapping, the full orchestrated assurance engine over all four evidence
layers, hash-chain tamper evidence for assessment entries, and analyst
decision recording.
"""

import json
from pathlib import Path

import pytest

from cv_auditor.governance import (
    AssuranceConfig,
    AssuranceEngine,
    GLOBAL_LIMITATIONS,
    TROJAN_WORDING,
    UNSUPPORTED_ATTACK_CLASSES,
    from_distribution_report,
    from_model_report,
    overall_decision,
    recommend_for,
)
from cv_auditor.provenance import AuditLog

from .distribution_helpers import BASELINE_STYLE, WINTER_STYLE, make_batch
from .provenance_helpers import make_record

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASET = ROOT / "data" / "sample_dataset"
SAMPLE_MODEL = ROOT / "data" / "sample_model" / "sample_model.pth"

REQUIRED_FINDING_FIELDS = {
    "finding_id", "timestamp", "affected_asset", "source_phase", "category",
    "severity", "confidence", "evidence", "explanation", "recommendation",
    "limitations",
}


class TestPolicyMapping:
    def test_severity_mapping(self):
        assert recommend_for("high", "possible_trojan_indicator") == "QUARANTINE"
        assert recommend_for("medium", "batch_distribution_shift") == "REVIEW"
        assert recommend_for("low", "potential_ood_anomaly") == "ACCEPT"

    def test_changed_model_with_consistent_behavior(self):
        # LOW + consistent behavior -> ACCEPT; without proof of consistency -> REVIEW
        assert recommend_for("low", "model_file_changed", agreement_rate=1.0) == "ACCEPT"
        assert recommend_for("low", "model_file_changed", agreement_rate=0.8) == "REVIEW"
        assert recommend_for("low", "model_file_changed") == "REVIEW"

    def test_phase5_softening_respected(self):
        assert recommend_for(
            "high", "batch_distribution_shift",
            distribution_recommendation="REVIEW",
        ) == "REVIEW"

    def test_overall_decision_reasons_and_confidence_damping(self):
        findings = [{
            "severity": "medium", "category": "batch_distribution_shift",
            "recommendation": "REVIEW", "confidence": 0.6,
            "source_phase": "phase5_distribution_shift",
        }]
        risk, conf, rec, reasons = overall_decision(
            findings, {"dataset": None, "model": None,
                       "provenance": None, "distribution": {"risk": "MEDIUM"}},
        )
        assert (risk, rec) == ("MEDIUM", "REVIEW")
        assert conf < 0.6  # damped: only 1 of 4 evidence layers ran
        assert any("evidence layers" in r for r in reasons)
        assert reasons  # every risk classification has a reason

    def test_no_findings_no_phases(self):
        risk, conf, rec, _ = overall_decision([], {"dataset": None, "model": None,
                                                   "provenance": None,
                                                   "distribution": None})
        assert (risk, conf, rec) == ("LOW", 0.0, "REVIEW")


class TestWordingDiscipline:
    def test_trojan_indicator_never_claims_confirmation(self):
        report = {
            "comparison": {"agreement_rate": 0.1},
            "findings": [{
                "type": "possible_trojan_indicator", "severity": "high",
                "description": "300 added detections of a single class",
                "finding_id": "MODEL-0001",
            }],
        }
        [f] = from_model_report(report, "candidate.pth")
        text = f.explanation
        assert "consistent with" in text
        assert "NOT a confirmed" in text
        assert "confirmed backdoor" not in text.replace("NOT a confirmed backdoor", "")
        assert f.recommendation == "QUARANTINE"

    def test_statistical_findings_carry_disclaimer(self):
        report = {
            "confidence": 0.7, "recommendation": "REVIEW",
            "interpretation": "seasonal drift",
            "evidence": {"shift_pattern": "uniform_shift"},
            "findings": [{
                "type": "batch_distribution_shift", "severity": "medium",
                "reason": "Batch displaced from baseline.",
                "evidence": "z=2.2", "finding_id": "DIST-0001",
            }],
        }
        [f] = from_distribution_report(report, "incoming")
        assert "does not by itself establish" in f.explanation
        assert f.confidence <= 0.7  # capped by the phase's own confidence


@pytest.fixture(scope="module")
def engine_run(tmp_path_factory):
    """One full four-layer assessment shared by the engine tests."""
    tmp = tmp_path_factory.mktemp("assurance")

    # phase 5 assets: baseline + shifted incoming batch
    make_batch(tmp / "reference", n=12, seed=3, style=BASELINE_STYLE)
    make_batch(tmp / "incoming", n=8, seed=4, style=WINTER_STYLE)
    from cv_auditor.distribution import DistributionConfig, build_baseline
    baseline = build_baseline(str(tmp / "reference"), DistributionConfig())
    baseline_path = tmp / "baseline.json"
    baseline.save(str(baseline_path))

    # phase 4 asset: a valid protected record bound to tiny fixture files
    record = make_record(tmp)
    record_path = tmp / "record.json"
    record_path.write_text(json.dumps(record))

    config = AssuranceConfig(
        dataset_dir=str(SAMPLE_DATASET),
        candidate_model_path=str(SAMPLE_MODEL),
        reference_model_path=str(SAMPLE_MODEL),
        test_images_dir="data/sample_images",
        detector_score_thresh=0.0,
        inference_record_path=str(record_path),
        incoming_dir=str(tmp / "incoming"),
        distribution_baseline_path=str(baseline_path),
        workdir=str(tmp / "work"),
        audit_log_path=str(tmp / "assurance_audit_log.jsonl"),
    )
    report = AssuranceEngine(config).run()
    report_path = tmp / "assurance_report.json"
    report.save(str(report_path))
    return config, report, report_path


@pytest.mark.skipif(not SAMPLE_DATASET.is_dir() or not SAMPLE_MODEL.exists(),
                    reason="sample assets missing")
class TestAssuranceEngineEndToEnd:
    def test_unified_report_schema(self, engine_run):
        _, report, _ = engine_run
        d = report.to_dict()
        for key in ("schema_version", "assessment_id", "run_metadata", "assets",
                    "sections", "findings", "governance", "limitations",
                    "unsupported_attack_classes", "audit"):
            assert key in d, key
        assert set(d["sections"]) == {"dataset", "model", "provenance",
                                      "distribution"}
        for section in d["sections"].values():
            assert section is not None and section["status"] is not None
        assert d["limitations"] == GLOBAL_LIMITATIONS
        assert d["unsupported_attack_classes"] == UNSUPPORTED_ATTACK_CLASSES
        assert d["assets"]["dataset_hash"] and d["assets"]["model_hash"]
        assert d["assets"]["configuration_hash"]

    def test_identical_model_and_valid_record_sections(self, engine_run):
        _, report, _ = engine_run
        d = report.to_dict()
        assert d["sections"]["model"]["status"] == "clean"
        assert d["sections"]["provenance"]["status"] == "VALID"
        assert d["sections"]["distribution"]["status"] in ("MEDIUM", "HIGH")

    def test_findings_are_fully_normalized(self, engine_run):
        _, report, _ = engine_run
        findings = report.to_dict()["findings"]
        assert findings  # the shifted batch guarantees at least one
        phases = {f["source_phase"] for f in findings}
        assert "phase5_distribution_shift" in phases
        for f in findings:
            assert REQUIRED_FINDING_FIELDS <= set(f), f
            assert f["evidence"], "every finding must carry evidence"
            assert 0.0 <= f["confidence"] <= 1.0

    def test_governance_outcome_has_reasons(self, engine_run):
        _, report, _ = engine_run
        g = report.to_dict()["governance"]
        assert g["overall_risk"] in ("LOW", "MEDIUM", "HIGH")
        assert g["recommendation"] in ("ACCEPT", "REVIEW", "QUARANTINE")
        assert g["decision_reasons"]
        assert g["analyst_decision"] is None

    def test_assessment_is_audit_logged_and_chain_verifies(self, engine_run):
        config, report, _ = engine_run
        log = AuditLog(config.audit_log_path)
        entries = log.read_entries()
        assert entries[-1]["record_id"] == report.assessment_id
        assert entries[-1]["record_digest"] == report.data["audit"]["record_digest"]
        assert log.verify_chain().valid

    def test_tampering_with_earlier_audit_entry_breaks_chain(self, engine_run, tmp_path):
        config, _, _ = engine_run
        tampered = tmp_path / "tampered.jsonl"
        lines = Path(config.audit_log_path).read_text().splitlines()
        first = json.loads(lines[0])
        first["record_digest"] = "0" * 64  # rewrite history
        lines[0] = json.dumps(first, sort_keys=True)
        tampered.write_text("\n".join(lines) + "\n")
        assert not AuditLog(str(tampered)).verify_chain().valid

    def test_analyst_decision_recorded_and_audit_logged(self, engine_run):
        config, report, report_path = engine_run
        before = len(AuditLog(config.audit_log_path).read_entries())
        entry = AssuranceEngine.record_analyst_decision(
            str(report_path), "REVIEW", analyst="test.analyst",
            audit_log_path=config.audit_log_path, note="unit test",
        )
        assert entry["decision"] == "REVIEW"
        stored = json.loads(report_path.read_text())
        assert stored["governance"]["analyst_decision"]["analyst"] == "test.analyst"
        log = AuditLog(config.audit_log_path)
        entries = log.read_entries()
        assert len(entries) == before + 1
        assert entries[-1]["record_id"].endswith("-decision")
        assert log.verify_chain().valid

    def test_invalid_analyst_decision_rejected(self, engine_run):
        config, _, report_path = engine_run
        with pytest.raises(ValueError):
            AssuranceEngine.record_analyst_decision(
                str(report_path), "YOLO", analyst="x",
                audit_log_path=config.audit_log_path,
            )


class TestPartialAssessments:
    def test_distribution_only_assessment(self, tmp_path):
        make_batch(tmp_path / "ref", n=16, seed=1, style=BASELINE_STYLE)
        make_batch(tmp_path / "cur", n=10, seed=77, style=BASELINE_STYLE)
        from cv_auditor.distribution import DistributionConfig, build_baseline
        baseline = build_baseline(str(tmp_path / "ref"), DistributionConfig())
        baseline.save(str(tmp_path / "baseline.json"))

        report = AssuranceEngine(AssuranceConfig(
            incoming_dir=str(tmp_path / "cur"),
            distribution_baseline_path=str(tmp_path / "baseline.json"),
            workdir=str(tmp_path / "work"),
            audit_log_path=str(tmp_path / "log.jsonl"),
        )).run()
        d = report.to_dict()
        assert d["sections"]["dataset"] is None
        assert d["sections"]["model"] is None
        assert d["sections"]["provenance"] is None
        assert d["sections"]["distribution"] is not None
        # matching distribution, single layer -> LOW but damped confidence
        assert d["governance"]["overall_risk"] == "LOW"
        assert any("evidence layers" in r
                   for r in d["governance"]["decision_reasons"])
