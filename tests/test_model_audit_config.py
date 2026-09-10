import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from cv_auditor.model_audit.config import ModelAuditConfig


def test_requires_candidate_model_path():
    with pytest.raises(ValueError):
        ModelAuditConfig(candidate_model_path="")


def test_valid_config_ok():
    cfg = ModelAuditConfig(candidate_model_path="some/model.pth")
    assert cfg.candidate_model_path == "some/model.pth"


def test_rejects_bad_architecture():
    with pytest.raises(ValueError):
        ModelAuditConfig(candidate_model_path="m.pth", architecture="not_real")


def test_rejects_out_of_range_confidence_threshold():
    with pytest.raises(ValueError):
        ModelAuditConfig(candidate_model_path="m.pth", confidence_threshold=1.5)


def test_save_and_load_round_trip(tmp_path):
    cfg = ModelAuditConfig(candidate_model_path="m.pth", iou_threshold=0.7)
    out = tmp_path / "config.json"
    cfg.save(str(out))
    loaded = ModelAuditConfig.from_json(str(out))
    assert loaded.candidate_model_path == "m.pth"
    assert loaded.iou_threshold == 0.7
