import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from cv_auditor.dataset.ood import compute_ood_scores, ood_findings


def _clustered_embeddings_with_one_outlier(n_normal: int = 12, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=1.0, scale=0.05, size=(n_normal, dim))
    outlier = rng.normal(loc=-5.0, scale=0.05, size=(1, dim))
    vectors = np.vstack([normal, outlier])
    ids = [f"normal_{i}" for i in range(n_normal)] + ["outlier"]
    return {image_id: vec.astype(np.float32) for image_id, vec in zip(ids, vectors)}


def test_outlier_gets_highest_score():
    embeddings = _clustered_embeddings_with_one_outlier()
    scores = compute_ood_scores(embeddings, k=5, threshold=2.0)
    top = max(scores, key=lambda s: s.score)
    assert top.image_id == "outlier"
    assert top.is_anomalous


def test_tight_cluster_members_are_not_flagged():
    embeddings = _clustered_embeddings_with_one_outlier()
    scores = compute_ood_scores(embeddings, k=5, threshold=2.0)
    normal_scores = [s for s in scores if s.image_id != "outlier"]
    flagged_normal = [s for s in normal_scores if s.is_anomalous]
    assert flagged_normal == []


def test_threshold_is_configurable():
    embeddings = _clustered_embeddings_with_one_outlier()
    lenient = compute_ood_scores(embeddings, k=5, threshold=100.0)  # nothing should pass
    strict = compute_ood_scores(embeddings, k=5, threshold=0.01)  # nearly everything should pass

    assert not any(s.is_anomalous for s in lenient)
    assert any(s.is_anomalous for s in strict)


def test_too_few_samples_returns_non_anomalous_without_crashing():
    embeddings = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
    }
    scores = compute_ood_scores(embeddings, k=5, threshold=2.0)
    assert len(scores) == 2
    assert all(not s.is_anomalous for s in scores)


def test_identical_embeddings_produce_zero_score_not_nan():
    embeddings = {f"s{i}": np.array([1.0, 1.0, 1.0], dtype=np.float32) for i in range(6)}
    scores = compute_ood_scores(embeddings, k=3, threshold=2.0)
    assert all(np.isfinite(s.score) for s in scores)
    assert all(not s.is_anomalous for s in scores)


def test_ood_findings_are_never_worded_as_an_attack():
    embeddings = _clustered_embeddings_with_one_outlier()
    findings = ood_findings(embeddings, k=5, threshold=2.0)

    assert len(findings) >= 1
    for f in findings:
        assert f.type == "ood_anomaly"
        assert f.severity in ("medium", "high")
        forbidden_words = ("attack", "malicious", "poison", "adversarial")
        combined_text = (f.reason + f.evidence).lower()
        assert not any(word in combined_text for word in forbidden_words)


def test_high_severity_threshold_is_respected():
    embeddings = _clustered_embeddings_with_one_outlier()
    findings = ood_findings(embeddings, k=5, threshold=2.0, high_severity_threshold=1000.0)
    # With an unreachable high-severity threshold, any flagged finding
    # should fall back to medium.
    assert all(f.severity == "medium" for f in findings)
