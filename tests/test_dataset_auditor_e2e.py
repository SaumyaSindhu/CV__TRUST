"""
End-to-end test for the Phase 2 Dataset Integrity Auditor.

Runs the REAL auditor (no mocking, no hard-coded results) against the
synthetic dataset shipped at data/sample_dataset/ (see
scripts/generate_sample_dataset.py) and checks that its actual output
matches the controlled cases that dataset was built to contain. If the
synthetic dataset is regenerated with a different seed/layout, these
assertions may need to be revisited -- they check for the specific
planted cases, not just "detector runs without crashing".
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from cv_auditor.dataset import DatasetAuditConfig, DatasetAuditor

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATASET_DIR = ROOT / "data" / "sample_dataset"

pytestmark = pytest.mark.skipif(
    not (SAMPLE_DATASET_DIR / "images").is_dir(),
    reason="Synthetic sample dataset not found; run scripts/generate_sample_dataset.py first.",
)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("embedding_cache")
    config = DatasetAuditConfig(
        dataset_dir=str(SAMPLE_DATASET_DIR),
        embedding_cache_dir=str(cache_dir),
    )
    return DatasetAuditor(config).run()


def test_report_has_expected_top_level_structure(report):
    d = report.to_dict()
    assert set(d.keys()) == {"run_metadata", "summary", "findings"}
    for key in (
        "total_samples", "valid_samples", "invalid_samples",
        "exact_duplicate_groups", "near_duplicate_groups",
        "label_anomalies", "potential_ood_anomalous_samples",
        "overall_dataset_finding_level",
    ):
        assert key in d["summary"]


def test_overall_finding_level_is_a_known_value(report):
    assert report.overall_finding_level in ("LOW", "MEDIUM", "HIGH")


def test_all_samples_accounted_for(report):
    assert report.valid_samples + report.invalid_samples == report.total_samples
    assert report.total_samples > 0


def test_planted_corrupted_image_is_detected(report):
    corrupted = [
        f for f in report.findings
        if f["type"] == "invalid_sample" and "unreadable" in f["reason"].lower()
    ]
    assert len(corrupted) >= 1
    assert corrupted[0]["severity"] == "high"


def test_planted_exact_duplicate_is_detected(report):
    assert report.exact_duplicate_groups >= 1
    exact_dupes = [f for f in report.findings if f["type"] == "exact_duplicate"]
    assert len(exact_dupes) >= 2  # both members of at least one group


def test_planted_near_duplicate_is_detected(report):
    assert report.near_duplicate_groups >= 1


def test_planted_mislabeled_sample_is_flagged(report):
    label_anomalies = [f for f in report.findings if f["type"] == "label_anomaly"]
    assert len(label_anomalies) >= 1
    for f in label_anomalies:
        assert "anomaly indicator" in f["reason"] or "not proof" in f["reason"]


def test_planted_visually_unusual_sample_is_flagged(report):
    ood = [f for f in report.findings if f["type"] == "ood_anomaly"]
    assert len(ood) >= 1
    for f in ood:
        assert "attack" not in f["reason"].lower()
        assert "poison" not in f["reason"].lower()


def test_findings_are_valid_json_and_within_schema(report):
    for f in report.findings:
        assert f["type"] in (
            "invalid_sample", "exact_duplicate", "near_duplicate",
            "label_anomaly", "ood_anomaly",
        )
        assert f["severity"] in ("low", "medium", "high")
        assert isinstance(f["sample"], str) and f["sample"]


def test_report_can_be_saved_and_reloaded(report, tmp_path):
    out_path = report.save(str(tmp_path / "report.json"))
    reloaded = type(report).load(out_path)
    assert reloaded["summary"]["total_samples"] == report.total_samples
