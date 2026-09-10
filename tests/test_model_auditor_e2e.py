"""
End-to-end test for the Phase 3 Model Integrity Auditor.

Runs the REAL auditor (no mocking) against Phase 1's shipped sample
checkpoint as the reference model, and against candidate checkpoints
constructed on the fly (not committed to the repo -- see
``tests/model_audit_helpers.py``) with controlled, known differences:

  * an exact byte-for-byte copy       -> expect no findings
  * tiny Gaussian noise on every param -> expect a low-severity
    "model_file_changed" finding only
  * a single boosted classification bias (the "backdoor-like" case)
    -> expect a high-severity "possible_trojan_indicator" finding

This is the "small controlled test case where a candidate model behaves
differently from the reference model" required by the Phase 3 spec.
"""

import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from cv_auditor.model_audit import ModelAuditConfig, ModelAuditor

from .model_audit_helpers import add_noise_checkpoint, perturb_bias_checkpoint

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MODEL = ROOT / "data" / "sample_model" / "sample_model.pth"
SAMPLE_IMAGES = ROOT / "data" / "sample_images"

pytestmark = pytest.mark.skipif(
    not SAMPLE_MODEL.exists() or not SAMPLE_IMAGES.is_dir(),
    reason="Phase 1 sample assets not found; run scripts/generate_sample_assets.py first.",
)


def _base_config(candidate_path: str, tmp_path: Path, **overrides) -> ModelAuditConfig:
    defaults = dict(
        candidate_model_path=candidate_path,
        reference_model_path=str(SAMPLE_MODEL),
        reference_behavior_path=str(tmp_path / "reference_behavior.json"),
        test_images_dir=str(SAMPLE_IMAGES),
        confidence_threshold=0.5,
        detector_score_thresh=0.0,  # needed for untrained weights, see PipelineConfig
        output_path=str(tmp_path / "report.json"),
    )
    defaults.update(overrides)
    return ModelAuditConfig(**defaults)


def test_identical_candidate_produces_no_findings(tmp_path):
    identical_path = tmp_path / "candidate_identical.pth"
    shutil.copy(str(SAMPLE_MODEL), str(identical_path))

    report = ModelAuditor(_base_config(str(identical_path), tmp_path)).run()

    assert report.fingerprint["changed"] is False
    assert report.findings == []
    assert report.category == "clean"
    assert report.overall_finding_level == "LOW"


def test_minor_noise_candidate_produces_only_low_severity_file_changed(tmp_path):
    noisy_path = tmp_path / "candidate_minor_noise.pth"
    add_noise_checkpoint(str(SAMPLE_MODEL), str(noisy_path), std=1e-4)

    report = ModelAuditor(_base_config(str(noisy_path), tmp_path)).run()

    assert report.fingerprint["changed"] is True
    types = {f["type"] for f in report.findings}
    assert types == {"model_file_changed"}
    assert all(f["severity"] == "low" for f in report.findings)
    assert report.overall_finding_level == "LOW"


def test_backdoor_like_candidate_produces_trojan_indicator(tmp_path):
    backdoor_path = tmp_path / "candidate_backdoor_like.pth"
    perturb_bias_checkpoint(str(SAMPLE_MODEL), str(backdoor_path), class_id=5, boost=40.0)

    report = ModelAuditor(_base_config(str(backdoor_path), tmp_path)).run()

    assert report.fingerprint["changed"] is True
    types = {f["type"] for f in report.findings}
    assert "possible_trojan_indicator" in types
    assert report.overall_finding_level == "HIGH"
    assert report.category == "possible_trojan_indicator"

    trojan_finding = next(f for f in report.findings if f["type"] == "possible_trojan_indicator")
    text = trojan_finding["explanation"].lower()
    assert "not confirmed" in text or "consistent with" in text
    assert "confirmed" not in text.replace("not confirmed", "")


def test_report_structure_matches_required_schema(tmp_path):
    identical_path = tmp_path / "candidate_identical.pth"
    shutil.copy(str(SAMPLE_MODEL), str(identical_path))
    report = ModelAuditor(_base_config(str(identical_path), tmp_path)).run()
    d = report.to_dict()

    assert set(d.keys()) == {"run_metadata", "fingerprint", "comparison", "summary", "findings"}
    for key in ("candidate_sha256", "reference_sha256", "changed"):
        assert key in d["fingerprint"]
    assert "overall_finding_level" in d["summary"]
    assert "category" in d["summary"]


def test_finding_json_schema_has_all_required_fields(tmp_path):
    backdoor_path = tmp_path / "candidate_backdoor_like.pth"
    perturb_bias_checkpoint(str(SAMPLE_MODEL), str(backdoor_path), class_id=5, boost=40.0)
    report = ModelAuditor(_base_config(str(backdoor_path), tmp_path)).run()

    assert len(report.findings) >= 1
    for f in report.findings:
        for key in ("finding_id", "type", "severity", "model", "evidence", "comparison_scores", "explanation"):
            assert key in f
        assert f["severity"] in ("low", "medium", "high")


def test_report_can_be_saved_and_reloaded(tmp_path):
    identical_path = tmp_path / "candidate_identical.pth"
    shutil.copy(str(SAMPLE_MODEL), str(identical_path))
    report = ModelAuditor(_base_config(str(identical_path), tmp_path)).run()

    out_path = report.save(str(tmp_path / "saved_report.json"))
    reloaded = type(report).load(out_path)
    assert reloaded["summary"]["category"] == report.category


def test_reference_behavior_is_cached_and_reused(tmp_path):
    """Second run with the same reference_behavior_path should reuse the
    cached baseline instead of re-running the reference model."""
    identical_path = tmp_path / "candidate_identical.pth"
    shutil.copy(str(SAMPLE_MODEL), str(identical_path))
    cfg = _base_config(str(identical_path), tmp_path)

    ModelAuditor(cfg).run()
    assert Path(cfg.reference_behavior_path).exists()
    cached_mtime = Path(cfg.reference_behavior_path).stat().st_mtime

    # Run again -- reference behavior file should not be rewritten.
    ModelAuditor(cfg).run()
    assert Path(cfg.reference_behavior_path).stat().st_mtime == cached_mtime


def test_missing_reference_still_reports_fingerprint_only(tmp_path):
    """Without any reference model/hash/behavior, the auditor should
    still run (just with no behavior comparison), not crash."""
    identical_path = tmp_path / "candidate_identical.pth"
    shutil.copy(str(SAMPLE_MODEL), str(identical_path))
    cfg = ModelAuditConfig(
        candidate_model_path=str(identical_path),
        reference_model_path=None,
        reference_behavior_path=str(tmp_path / "does_not_exist.json"),
        test_images_dir=str(SAMPLE_IMAGES),
        confidence_threshold=0.5,
        detector_score_thresh=0.0,
        output_path=str(tmp_path / "report.json"),
    )
    report = ModelAuditor(cfg).run()
    assert report.comparison is None
    assert report.fingerprint["changed"] is None
