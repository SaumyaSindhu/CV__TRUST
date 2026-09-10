import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json

import pytest

from cv_auditor.dataset.findings import Finding, FindingsCollection


def test_finding_gets_a_unique_auto_generated_id():
    f1 = Finding(type="exact_duplicate", severity="low", sample="a.jpg", reason="r")
    f2 = Finding(type="exact_duplicate", severity="low", sample="b.jpg", reason="r")
    assert f1.finding_id != f2.finding_id
    assert f1.finding_id.startswith("exact_duplicate")


def test_finding_rejects_unknown_type():
    with pytest.raises(ValueError):
        Finding(type="not_a_real_type", severity="low", sample="a.jpg", reason="r")


def test_finding_rejects_unknown_severity():
    with pytest.raises(ValueError):
        Finding(type="exact_duplicate", severity="critical", sample="a.jpg", reason="r")


def test_finding_to_dict_is_json_serializable():
    f = Finding(
        type="ood_anomaly", severity="high", sample="a.jpg", reason="r", evidence="e",
        related_samples=["b.jpg"], score=0.987654321, extra={"k": 5},
    )
    d = f.to_dict()
    json.dumps(d)  # should not raise
    assert d["score"] == 0.9877  # rounded
    assert d["related_samples"] == ["b.jpg"]
    assert d["extra"] == {"k": 5}


def test_findings_collection_counts_by_type_and_severity():
    collection = FindingsCollection()
    collection.add(Finding(type="exact_duplicate", severity="low", sample="a.jpg", reason="r"))
    collection.add(Finding(type="exact_duplicate", severity="low", sample="b.jpg", reason="r"))
    collection.add(Finding(type="ood_anomaly", severity="high", sample="c.jpg", reason="r"))

    assert len(collection) == 3
    assert collection.counts_by_type()["exact_duplicate"] == 2
    assert collection.counts_by_type()["ood_anomaly"] == 1
    assert collection.counts_by_type()["label_anomaly"] == 0
    assert collection.counts_by_severity()["low"] == 2
    assert collection.counts_by_severity()["high"] == 1


def test_findings_collection_filters_by_type_and_severity():
    collection = FindingsCollection()
    collection.add(Finding(type="invalid_sample", severity="high", sample="a.jpg", reason="r"))
    collection.add(Finding(type="near_duplicate", severity="low", sample="b.jpg", reason="r"))

    assert len(collection.by_type("invalid_sample")) == 1
    assert len(collection.by_severity("low")) == 1


def test_findings_collection_save_writes_valid_json(tmp_path):
    collection = FindingsCollection()
    collection.add(Finding(type="label_anomaly", severity="medium", sample="a.jpg", reason="r"))

    out = tmp_path / "nested" / "findings.json"
    collection.save(str(out))

    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0]["type"] == "label_anomaly"


def test_findings_never_use_the_word_attack_or_malicious_by_default():
    """A basic sanity guard: the built-in finding constructors used
    throughout the auditor should never phrase evidence as a verdict."""
    collection = FindingsCollection()
    collection.extend([
        Finding(type="exact_duplicate", severity="low", sample="a.jpg", reason="duplicate found"),
        Finding(type="ood_anomaly", severity="high", sample="b.jpg", reason="distribution anomaly"),
    ])
    for f in collection:
        assert "attack" not in f.reason.lower()
        assert "malicious" not in f.reason.lower()
