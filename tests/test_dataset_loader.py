import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from cv_auditor.dataset.dataset_loader import DatasetLoader
from cv_auditor.dataset.labels import parse_yolo_label_file
from .dataset_helpers import build_basic_dataset, make_image, write_label


def test_loads_basic_labeled_dataset(tmp_path):
    build_basic_dataset(tmp_path, n=5)
    index = DatasetLoader(str(tmp_path)).load()

    assert index.num_samples == 5
    assert index.labels_expected is True
    for sample in index.samples:
        assert sample.readable
        assert sample.width == 64 and sample.height == 64
        assert sample.sha256
        assert len(sample.boxes) == 1


def test_unlabeled_dataset_does_not_produce_missing_label_errors(tmp_path):
    build_basic_dataset(tmp_path, n=3, with_labels=False)
    index = DatasetLoader(str(tmp_path)).load()

    assert index.labels_expected is False
    for sample in index.samples:
        assert sample.label_path is None
        assert sample.label_errors == []


def test_missing_single_label_file_is_recorded_when_dataset_has_labels(tmp_path):
    build_basic_dataset(tmp_path, n=3, with_labels=True)
    (tmp_path / "labels" / "img_002.txt").unlink()

    index = DatasetLoader(str(tmp_path)).load()
    sample = next(s for s in index.samples if s.image_id == "img_002.jpg")
    assert sample.label_path is None
    assert any("Missing label file" in e for e in sample.label_errors)


def test_missing_dataset_directory_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetLoader(str(tmp_path / "does_not_exist")).load()


def test_empty_images_directory_raises(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError):
        DatasetLoader(str(tmp_path)).load()


def test_class_names_are_resolved_when_provided(tmp_path):
    build_basic_dataset(tmp_path, n=1)
    index = DatasetLoader(str(tmp_path), class_names=["square", "circle", "triangle"]).load()
    box = index.samples[0].boxes[0]
    assert box.class_name in ("square", "circle", "triangle")


def test_unreadable_image_is_marked_not_readable(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "images" / "bad.jpg").write_bytes(b"definitely not a jpeg")

    index = DatasetLoader(str(tmp_path)).load()
    assert index.num_samples == 1
    sample = index.samples[0]
    assert sample.readable is False
    assert sample.read_error is not None
    assert sample.sha256 is None


# --- label parsing (labels.py) ---------------------------------------------

def test_parse_yolo_label_file_valid(tmp_path):
    label_path = tmp_path / "a.txt"
    label_path.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.1 0.05 0.05\n")
    result = parse_yolo_label_file(str(label_path))
    assert len(result.boxes) == 2
    assert result.errors == []


def test_parse_yolo_label_file_malformed_line_is_skipped_not_raised(tmp_path):
    label_path = tmp_path / "a.txt"
    label_path.write_text("0 0.5 0.5 0.2\nnot numbers at all\n1 0.4 0.4 0.1 0.1\n")
    result = parse_yolo_label_file(str(label_path))
    assert len(result.boxes) == 1
    assert len(result.errors) == 2


def test_parse_yolo_label_file_invalid_bbox_geometry(tmp_path):
    label_path = tmp_path / "a.txt"
    # x_center out of [0,1] range
    label_path.write_text("0 1.5 0.5 0.2 0.2\n")
    result = parse_yolo_label_file(str(label_path))
    assert result.boxes == []
    assert "invalid bounding box geometry" in result.errors[0]


def test_parse_yolo_label_file_missing_file_reports_error_not_exception(tmp_path):
    result = parse_yolo_label_file(str(tmp_path / "missing.txt"))
    assert result.boxes == []
    assert "not found" in result.errors[0]


def test_parse_yolo_label_file_class_id_out_of_range(tmp_path):
    label_path = tmp_path / "a.txt"
    label_path.write_text("5 0.5 0.5 0.2 0.2\n")
    result = parse_yolo_label_file(str(label_path), class_names=["a", "b"])
    assert any("out of range" in e for e in result.errors)
