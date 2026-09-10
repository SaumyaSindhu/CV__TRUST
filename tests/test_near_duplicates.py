import io
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image

from cv_auditor.dataset.dataset_loader import DatasetLoader
from cv_auditor.dataset.near_duplicates import near_duplicate_findings
from cv_auditor.dataset.perceptual_hash import (
    compute_dhash,
    find_near_duplicate_groups,
    hamming_distance,
    similarity_from_hashes,
)

from .dataset_helpers import make_image


def test_identical_image_has_zero_hamming_distance(tmp_path):
    path = tmp_path / "a.jpg"
    make_image(path, color=(200, 60, 60))
    h1 = compute_dhash(str(path))
    h2 = compute_dhash(str(path))
    assert hamming_distance(h1, h2) == 0
    assert similarity_from_hashes(h1, h2) == 1.0


def test_recompressed_copy_is_a_near_duplicate(tmp_path):
    original_path = tmp_path / "images" / "orig.jpg"
    make_image(original_path, color=(60, 120, 200), size=200)

    # Simulate a recompression pass at lower quality -- same visual
    # content, different bytes.
    with Image.open(original_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=40)
    recompressed_path = tmp_path / "images" / "recompressed.jpg"
    recompressed_path.write_bytes(buf.getvalue())

    image_paths = {"orig.jpg": str(original_path), "recompressed.jpg": str(recompressed_path)}
    groups = find_near_duplicate_groups(image_paths, similarity_threshold=0.90)

    assert len(groups) == 1
    assert set(groups[0]["members"]) == {"orig.jpg", "recompressed.jpg"}


def test_visually_different_images_are_not_grouped(tmp_path):
    a = tmp_path / "images" / "a.jpg"
    b = tmp_path / "images" / "b.jpg"
    make_image(a, color=(220, 30, 30), size=128)  # solid-ish red circle
    # A very different-looking image: full noise.
    noise = Image.effect_noise((128, 128), 80).convert("RGB")
    noise.save(b)

    groups = find_near_duplicate_groups(
        {"a.jpg": str(a), "b.jpg": str(b)}, similarity_threshold=0.90,
    )
    assert groups == []


def test_similarity_threshold_is_configurable(tmp_path):
    """A moderately-degraded near-duplicate should pass a loose threshold
    but fail a very strict one."""
    a = tmp_path / "images" / "a.jpg"
    make_image(a, color=(200, 60, 60), size=128)
    with Image.open(a) as img:
        buf = io.BytesIO()
        # Crop + heavy recompression -> visually the same, but far from
        # byte-identical and not a perfect hash match either.
        img.crop((3, 3, 125, 125)).resize((128, 128)).save(buf, format="JPEG", quality=20)
    b = tmp_path / "images" / "b.jpg"
    b.write_bytes(buf.getvalue())

    paths = {"a.jpg": str(a), "b.jpg": str(b)}
    strict = find_near_duplicate_groups(paths, similarity_threshold=0.99)
    loose = find_near_duplicate_groups(paths, similarity_threshold=0.85)

    assert strict == []  # degraded copy doesn't pass a near-perfect threshold
    assert len(loose) == 1  # but does pass a more permissive one


def test_near_duplicate_findings_reference_group_members(tmp_path):
    (tmp_path / "images").mkdir(parents=True)
    a = tmp_path / "images" / "img_001.jpg"
    b = tmp_path / "images" / "img_002.jpg"
    make_image(a, color=(200, 60, 60), size=128)
    with Image.open(a) as img:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
    b.write_bytes(buf.getvalue())

    index = DatasetLoader(str(tmp_path)).load()
    findings = near_duplicate_findings(index, similarity_threshold=0.85)

    assert len(findings) == 2
    for f in findings:
        assert f.type == "near_duplicate"
        assert f.severity == "low"
        assert 0.0 <= f.score <= 1.0
        assert len(f.related_samples) == 1
