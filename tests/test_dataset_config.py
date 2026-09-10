import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from cv_auditor.dataset.config import DatasetAuditConfig


def test_default_config_is_valid():
    cfg = DatasetAuditConfig()
    assert 0.0 <= cfg.near_duplicate_similarity_threshold <= 1.0


def test_rejects_out_of_range_similarity_threshold():
    with pytest.raises(ValueError):
        DatasetAuditConfig(near_duplicate_similarity_threshold=1.5)


def test_rejects_invalid_phash_size():
    with pytest.raises(ValueError):
        DatasetAuditConfig(phash_size=1)


def test_rejects_invalid_minority_ratio():
    with pytest.raises(ValueError):
        DatasetAuditConfig(label_minority_ratio_threshold=0.0)


def test_save_and_load_round_trip(tmp_path):
    cfg = DatasetAuditConfig(dataset_dir="some/dataset", near_duplicate_similarity_threshold=0.8)
    out = tmp_path / "config.json"
    cfg.save(str(out))

    loaded = DatasetAuditConfig.from_json(str(out))
    assert loaded.dataset_dir == "some/dataset"
    assert loaded.near_duplicate_similarity_threshold == 0.8
