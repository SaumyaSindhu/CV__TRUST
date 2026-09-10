"""
test_distribution_shift.py
---------------------------
Phase 5 tests -- the twelve required scenarios (identical distribution,
normal drift, strong shift, individual anomaly, mixed batch, empty
dataset, corrupt images, metadata absent/present, threshold behavior,
confidence/risk calculation; Phases 1-4 regression is the full suite).

All fixtures are tiny, deterministic, and generated on the fly.
"""

import json

import pytest

from cv_auditor.distribution import (
    DistributionAuditor,
    DistributionBaseline,
    DistributionConfig,
    DistributionFeatureExtractor,
    PATTERN_UNIFORM,
    build_baseline,
)

from .distribution_helpers import (
    ANOMALY_STYLE,
    BASELINE_STYLE,
    SMALL_DRIFT_STYLE,
    WINTER_STYLE,
    make_batch,
    make_corrupt_file,
)


@pytest.fixture(scope="module")
def extractor():
    """One shared feature extractor (avoids re-probing for pretrained
    weights in every test)."""
    return DistributionFeatureExtractor(cache_dir=None)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory, extractor):
    ref_dir = tmp_path_factory.mktemp("reference")
    make_batch(ref_dir, n=16, seed=1, style=BASELINE_STYLE)
    return build_baseline(str(ref_dir), DistributionConfig(), extractor=extractor)


def run_audit(baseline, batch_dir, extractor, cfg=None, metadata=None):
    auditor = DistributionAuditor(cfg or DistributionConfig(), extractor=extractor)
    return auditor.run(str(batch_dir), baseline=baseline, current_metadata=metadata)


class TestBaseline:
    def test_baseline_statistics_and_roundtrip(self, baseline, tmp_path):
        assert baseline.num_samples == 16
        assert baseline.feature_dim == baseline.centroid.shape[0]
        assert baseline.self_distance_std > 0
        path = tmp_path / "baseline.json"
        baseline.save(str(path))
        loaded = DistributionBaseline.load(str(path))
        assert loaded.num_samples == baseline.num_samples
        assert loaded.feature_backend == baseline.feature_backend
        assert loaded.centroid.tolist() == pytest.approx(baseline.centroid.tolist())

    def test_baseline_refuses_too_few_images(self, tmp_path, extractor):
        make_batch(tmp_path / "tiny", n=2, seed=9)
        with pytest.raises(ValueError, match="at least"):
            build_baseline(str(tmp_path / "tiny"), DistributionConfig(),
                           extractor=extractor)


class TestScenario1IdenticalDistribution:
    def test_same_distribution_is_low(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "same", n=10, seed=77, style=BASELINE_STYLE)
        report = run_audit(baseline, tmp_path / "same", extractor)
        assert report.risk == "LOW"
        assert report.recommendation == "ACCEPT"
        assert report.comparison.anomalous_fraction <= 0.1
        assert report.to_dict()["finding"] == "no_significant_distribution_shift"


class TestScenario2NormalDrift:
    def test_small_drift_does_not_become_high(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "drift", n=10, seed=42, style=SMALL_DRIFT_STYLE)
        report = run_audit(baseline, tmp_path / "drift", extractor)
        assert report.risk in ("LOW", "MEDIUM")  # never HIGH for mild drift
        assert report.recommendation in ("ACCEPT", "REVIEW")


class TestScenario3StrongShift:
    def test_domain_change_is_flagged(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "winter", n=10, seed=5, style=WINTER_STYLE)
        report = run_audit(baseline, tmp_path / "winter", extractor)
        assert report.risk in ("MEDIUM", "HIGH")
        d = report.to_dict()
        assert d["shift_score"] > 0.3
        types = {f.type for f in report.findings}
        assert "batch_distribution_shift" in types
        # a whole-batch coherent move must be described as drift-like,
        # never asserted as malicious
        assert report.comparison.shift_pattern == PATTERN_UNIFORM
        text = d["interpretation"].lower()
        assert "not" in text and ("manipulat" in text or "suspicious" in text)


class TestScenario4IndividualAnomaly:
    def test_single_wild_image_is_flagged(self, baseline, tmp_path, extractor):
        batch = tmp_path / "one_bad"
        make_batch(batch, n=9, seed=11, style=BASELINE_STYLE)
        make_batch(batch, n=1, seed=999, style=ANOMALY_STYLE, prefix="odd")
        report = run_audit(baseline, batch, extractor)
        flagged = [f for f in report.findings if f.type == "anomalous_sample"]
        assert len(flagged) >= 1
        assert any(f.sample.startswith("odd") for f in flagged)
        # the rest of the batch keeps the overall picture from exploding
        assert report.comparison.anomalous_fraction <= 0.2


class TestScenario5MixedBatch:
    def test_mostly_normal_plus_anomalies(self, baseline, tmp_path, extractor):
        batch = tmp_path / "mixed"
        make_batch(batch, n=8, seed=21, style=BASELINE_STYLE)
        make_batch(batch, n=3, seed=888, style=ANOMALY_STYLE, prefix="odd")
        report = run_audit(baseline, batch, extractor)
        assert report.risk in ("MEDIUM", "HIGH")
        types = {f.type for f in report.findings}
        assert "anomalous_sample" in types
        assert "high_anomalous_fraction" in types
        odd = [f for f in report.findings if f.type == "anomalous_sample"]
        assert sum(f.sample.startswith("odd") for f in odd) >= 2


class TestScenario6EmptyDataset:
    def test_empty_batch_handled_gracefully(self, baseline, tmp_path, extractor):
        empty = tmp_path / "empty"
        empty.mkdir()
        report = run_audit(baseline, empty, extractor)
        assert report.comparison.num_current == 0
        assert report.comparison.confidence == 0.0
        assert report.recommendation == "REVIEW"
        assert any(f.type == "batch_too_small" for f in report.findings)

    def test_missing_directory_raises(self, baseline, tmp_path, extractor):
        with pytest.raises(FileNotFoundError):
            run_audit(baseline, tmp_path / "does_not_exist", extractor)


class TestScenario7CorruptImages:
    def test_corrupt_files_excluded_and_reported(self, baseline, tmp_path, extractor):
        batch = tmp_path / "with_corrupt"
        make_batch(batch, n=8, seed=31, style=BASELINE_STYLE)
        make_corrupt_file(batch, "broken.jpg")
        report = run_audit(baseline, batch, extractor)
        assert report.comparison.num_current == 8
        assert "broken.jpg" in report.corrupt_images
        assert any(f.type == "corrupt_image" and f.sample == "broken.jpg"
                   for f in report.findings)
        # corrupt files reduce confidence but don't fake a shift
        assert report.risk == "LOW"

    def test_all_corrupt_behaves_like_empty(self, baseline, tmp_path, extractor):
        batch = tmp_path / "all_corrupt"
        make_corrupt_file(batch, "a.jpg")
        make_corrupt_file(batch, "b.jpg")
        report = run_audit(baseline, batch, extractor)
        assert report.comparison.num_current == 0
        assert report.comparison.confidence == 0.0
        assert any(f.type == "batch_too_small" for f in report.findings)


class TestScenario8And9Metadata:
    def test_no_metadata_strong_uniform_shift_quarantines(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "w1", n=10, seed=6, style=WINTER_STYLE)
        report = run_audit(baseline, tmp_path / "w1", extractor)
        assert report.to_dict()["evidence"]["declared_metadata_changes"] == {}
        if report.risk == "HIGH":
            assert report.recommendation == "QUARANTINE"

    def test_declared_acquisition_change_softens_to_review(self, tmp_path_factory, tmp_path, extractor):
        ref_dir = tmp_path_factory.mktemp("ref_meta")
        make_batch(ref_dir, n=16, seed=2, style=BASELINE_STYLE)
        base = build_baseline(str(ref_dir), DistributionConfig(),
                              metadata={"season": "summer", "sensor": "cam_A"},
                              extractor=extractor)
        make_batch(tmp_path / "w2", n=10, seed=7, style=WINTER_STYLE)
        report = run_audit(base, tmp_path / "w2", extractor,
                           metadata={"season": "winter", "sensor": "cam_A"})
        changes = report.to_dict()["evidence"]["declared_metadata_changes"]
        assert "season" in changes and "sensor" not in changes
        if (report.risk == "HIGH"
                and report.comparison.shift_pattern == PATTERN_UNIFORM):
            assert report.recommendation == "REVIEW"  # declared drift, human confirms
        assert "declared" in report.to_dict()["interpretation"].lower()


class TestScenario10Thresholds:
    def test_stricter_thresholds_escalate_risk(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "drift2", n=10, seed=13, style=SMALL_DRIFT_STYLE)
        default = run_audit(baseline, tmp_path / "drift2", extractor)
        strict = run_audit(
            baseline, tmp_path / "drift2", extractor,
            cfg=DistributionConfig(batch_shift_z_medium=0.01,
                                   batch_shift_z_high=0.02),
        )
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        assert order[strict.risk] >= order[default.risk]
        assert strict.risk == "HIGH"

    def test_invalid_threshold_config_rejected(self):
        with pytest.raises(ValueError):
            DistributionConfig(batch_shift_z_medium=5.0, batch_shift_z_high=1.0)
        with pytest.raises(ValueError):
            DistributionConfig(anomalous_fraction_medium=0.5,
                               anomalous_fraction_high=0.2)


class TestScenario11ConfidenceAndRisk:
    def test_small_batches_get_lower_confidence(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "big", n=12, seed=51, style=BASELINE_STYLE)
        make_batch(tmp_path / "small", n=3, seed=52, style=BASELINE_STYLE)
        big = run_audit(baseline, tmp_path / "big", extractor)
        small = run_audit(baseline, tmp_path / "small", extractor)
        assert small.comparison.confidence < big.comparison.confidence

    def test_every_risk_has_reasoned_evidence(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "w3", n=10, seed=8, style=WINTER_STYLE)
        report = run_audit(baseline, tmp_path / "w3", extractor)
        d = report.to_dict()
        assert d["risk"] in ("LOW", "MEDIUM", "HIGH")
        assert 0.0 <= d["confidence"] <= 1.0
        assert 0.0 <= d["shift_score"] <= 1.0
        assert d["interpretation"]
        assert d["limitations"]
        for f in d["findings"]:
            assert f["evidence"]  # every finding carries concrete evidence

    def test_report_schema_and_save(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "s", n=6, seed=61, style=BASELINE_STYLE)
        report = run_audit(baseline, tmp_path / "s", extractor)
        out = tmp_path / "report.json"
        report.save(str(out))
        d = json.loads(out.read_text())
        for key in ("schema_version", "baseline", "current_batch", "shift_score",
                    "anomalous_fraction", "confidence", "risk", "finding",
                    "evidence", "interpretation", "recommendation",
                    "limitations", "summary", "findings", "run_metadata"):
            assert key in d, key
        assert d["baseline"]["num_samples"] == 16

    def test_backend_mismatch_refused_with_finding(self, baseline, tmp_path, extractor):
        make_batch(tmp_path / "bm", n=6, seed=71, style=BASELINE_STYLE)
        import copy
        other = copy.copy(baseline)
        other.feature_backend = "some_other_backend_v9"
        report = run_audit(other, tmp_path / "bm", extractor)
        assert [f.type for f in report.findings] == ["feature_backend_mismatch"]
        assert report.findings[0].severity == "high"
