import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import json

import pytest

from cv_auditor.model_audit.findings import ModelFinding, ModelFindingsCollection


def test_finding_gets_a_unique_auto_generated_id():
    f1 = ModelFinding(type="model_file_changed", severity="low", model="m1", evidence="e", explanation="x")
    f2 = ModelFinding(type="model_file_changed", severity="low", model="m2", evidence="e", explanation="x")
    assert f1.finding_id != f2.finding_id


def test_finding_rejects_unknown_type():
    with pytest.raises(ValueError):
        ModelFinding(type="not_real", severity="low", model="m", evidence="e", explanation="x")


def test_finding_rejects_unknown_severity():
    with pytest.raises(ValueError):
        ModelFinding(type="model_file_changed", severity="critical", model="m", evidence="e", explanation="x")


def test_finding_to_dict_has_required_fields():
    f = ModelFinding(
        type="possible_trojan_indicator", severity="high", model="m", evidence="e", explanation="x",
        comparison_scores={"agreement_rate": 0.5},
    )
    d = f.to_dict()
    for key in ("finding_id", "type", "severity", "model", "evidence", "comparison_scores", "explanation"):
        assert key in d
    json.dumps(d)


def test_findings_collection_counts_by_type_and_severity():
    collection = ModelFindingsCollection()
    collection.add(ModelFinding(type="model_file_changed", severity="low", model="m", evidence="e", explanation="x"))
    collection.add(ModelFinding(type="unusual_behavior", severity="medium", model="m", evidence="e", explanation="x"))
    assert collection.counts_by_type()["model_file_changed"] == 1
    assert collection.counts_by_severity()["medium"] == 1


def test_findings_collection_save_writes_valid_json(tmp_path):
    collection = ModelFindingsCollection()
    collection.add(ModelFinding(type="unusual_behavior", severity="medium", model="m", evidence="e", explanation="x"))
    out = tmp_path / "nested" / "findings.json"
    collection.save(str(out))
    data = json.loads(out.read_text())
    assert len(data) == 1
