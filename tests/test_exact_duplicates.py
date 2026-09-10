import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cv_auditor.dataset.dataset_loader import DatasetLoader
from cv_auditor.dataset.duplicates import exact_duplicate_findings, find_exact_duplicate_groups

from .dataset_helpers import build_basic_dataset, make_image


def test_no_duplicates_in_distinct_dataset(tmp_path):
    build_basic_dataset(tmp_path, n=4)
    index = DatasetLoader(str(tmp_path)).load()
    assert find_exact_duplicate_groups(index) == {}
    assert exact_duplicate_findings(index) == []


def test_byte_identical_copy_is_detected(tmp_path):
    build_basic_dataset(tmp_path, n=2)
    # Make an exact byte-for-byte copy of img_001 as img_003.
    src = tmp_path / "images" / "img_001.jpg"
    dst = tmp_path / "images" / "img_003.jpg"
    dst.write_bytes(src.read_bytes())

    index = DatasetLoader(str(tmp_path)).load()
    groups = find_exact_duplicate_groups(index)

    assert len(groups) == 1
    (members,) = groups.values()
    assert set(members) == {"img_001.jpg", "img_003.jpg"}


def test_findings_reference_each_other_as_related_samples(tmp_path):
    build_basic_dataset(tmp_path, n=2)
    src = tmp_path / "images" / "img_001.jpg"
    (tmp_path / "images" / "img_003.jpg").write_bytes(src.read_bytes())

    index = DatasetLoader(str(tmp_path)).load()
    findings = exact_duplicate_findings(index)

    assert len(findings) == 2
    by_sample = {f.sample: f for f in findings}
    assert by_sample["img_001.jpg"].related_samples == ["img_003.jpg"]
    assert by_sample["img_003.jpg"].related_samples == ["img_001.jpg"]
    for f in findings:
        assert f.type == "exact_duplicate"
        assert f.severity == "low"
        assert f.score == 1.0


def test_visually_similar_but_not_byte_identical_is_not_an_exact_duplicate(tmp_path):
    """Two independently-generated images with the same shape/color are
    NOT byte-identical (PIL/JPEG encoding differs), so this must not be
    reported as an exact duplicate -- that's near_duplicates.py's job."""
    make_image(tmp_path / "images" / "a.jpg", color=(200, 60, 60), size=64)
    make_image(tmp_path / "images" / "b.jpg", color=(202, 58, 62), size=64)

    index = DatasetLoader(str(tmp_path)).load()
    assert find_exact_duplicate_groups(index) == {}


def test_unreadable_images_are_excluded_from_duplicate_grouping(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "images" / "bad1.jpg").write_bytes(b"same garbage bytes")
    (tmp_path / "images" / "bad2.jpg").write_bytes(b"same garbage bytes")

    index = DatasetLoader(str(tmp_path)).load()
    # Even though the raw bytes are identical, neither is a *readable*
    # image, so they should not appear in duplicate-group findings.
    assert find_exact_duplicate_groups(index) == {}
