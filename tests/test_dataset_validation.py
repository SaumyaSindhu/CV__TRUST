import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cv_auditor.dataset.dataset_loader import DatasetLoader
from cv_auditor.dataset.validation import validate_dataset

from .dataset_helpers import build_basic_dataset, make_image


def test_valid_dataset_produces_no_findings(tmp_path):
    build_basic_dataset(tmp_path, n=4)
    index = DatasetLoader(str(tmp_path)).load()
    findings = validate_dataset(index)
    assert findings == []


def test_corrupted_image_produces_high_severity_finding(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "images" / "bad.jpg").write_bytes(b"garbage, not an image")

    index = DatasetLoader(str(tmp_path)).load()
    findings = validate_dataset(index)

    assert len(findings) == 1
    f = findings[0]
    assert f.type == "invalid_sample"
    assert f.severity == "high"
    assert f.sample == "bad.jpg"


def test_missing_label_produces_medium_finding(tmp_path):
    build_basic_dataset(tmp_path, n=2)
    (tmp_path / "labels" / "img_001.txt").unlink()

    index = DatasetLoader(str(tmp_path)).load()
    findings = validate_dataset(index)

    missing = [f for f in findings if "Missing label" in f.reason]
    assert len(missing) == 1
    assert missing[0].severity == "medium"
    assert missing[0].sample == "img_001.jpg"


def test_malformed_label_line_produces_low_severity_finding(tmp_path):
    build_basic_dataset(tmp_path, n=1)
    (tmp_path / "labels" / "img_001.txt").write_text("0 0.5 0.5\n")  # missing w/h

    index = DatasetLoader(str(tmp_path)).load()
    findings = validate_dataset(index)

    assert len(findings) == 1
    assert findings[0].severity == "low"
    assert "Malformed" in findings[0].reason


def test_invalid_bbox_produces_medium_severity_finding(tmp_path):
    build_basic_dataset(tmp_path, n=1)
    (tmp_path / "labels" / "img_001.txt").write_text("0 1.4 0.5 0.2 0.2\n")

    index = DatasetLoader(str(tmp_path)).load()
    findings = validate_dataset(index)

    assert len(findings) == 1
    assert findings[0].severity == "medium"
    assert "Invalid bounding box" in findings[0].reason


def test_unreadable_image_skips_label_validation(tmp_path):
    """An unreadable image shouldn't also spam label-related findings."""
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    (tmp_path / "images" / "bad.jpg").write_bytes(b"garbage")
    (tmp_path / "labels" / "bad.txt").write_text("not a valid label line\n")

    index = DatasetLoader(str(tmp_path)).load()
    findings = validate_dataset(index)

    assert len(findings) == 1
    assert findings[0].reason == "Image is unreadable or corrupted"


def test_one_bad_sample_does_not_prevent_processing_the_rest(tmp_path):
    build_basic_dataset(tmp_path, n=3)
    (tmp_path / "images" / "corrupt.jpg").write_bytes(b"not an image")

    index = DatasetLoader(str(tmp_path)).load()
    assert index.num_samples == 4  # 3 good + 1 corrupt, no crash

    findings = validate_dataset(index)
    assert len(findings) == 1
    assert findings[0].sample == "corrupt.jpg"
