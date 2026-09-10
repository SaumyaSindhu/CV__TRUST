import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pytest

from cv_auditor.dataset.embeddings import EmbeddingExtractor, _handcrafted_embedding

from .dataset_helpers import make_image


@pytest.fixture
def extractor(tmp_path):
    # cache_dir under tmp_path keeps every test's cache isolated.
    return EmbeddingExtractor(cache_dir=str(tmp_path / "cache"))


def test_extract_returns_consistent_dimensionality(tmp_path, extractor):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    make_image(a, color=(200, 60, 60))
    make_image(b, color=(60, 120, 200))

    va = extractor.extract(str(a))
    vb = extractor.extract(str(b))

    assert va.shape == vb.shape
    assert va.dtype == np.float32
    assert np.isfinite(va).all()


def test_records_which_backend_produced_the_embedding(extractor):
    # In this sandboxed environment there's no internet access to
    # TorchVision's weights server, so the handcrafted fallback is
    # expected -- but either outcome is valid depending on environment,
    # this test just checks the flag is set consistently.
    assert extractor.backend in ("resnet18_pretrained", "handcrafted_color_texture")
    assert isinstance(extractor.pretrained, bool)


def test_embedding_cache_is_reused_for_identical_content(tmp_path, extractor):
    a = tmp_path / "a.jpg"
    make_image(a, color=(200, 60, 60))

    v1 = extractor.extract(str(a))
    cache_files_after_first = list((tmp_path / "cache").glob("*.json"))
    assert len(cache_files_after_first) == 1

    # Extracting the exact same file again should hit the cache, not
    # create a second cache entry.
    v2 = extractor.extract(str(a))
    cache_files_after_second = list((tmp_path / "cache").glob("*.json"))

    assert len(cache_files_after_second) == 1
    assert np.allclose(v1, v2)


def test_cache_is_keyed_by_content_not_path(tmp_path, extractor):
    """Renaming/copying a file to a new path should still hit the cache
    entry for that image's content hash."""
    a = tmp_path / "a.jpg"
    make_image(a, color=(60, 180, 90))
    extractor.extract(str(a))

    copy_path = tmp_path / "a_renamed.jpg"
    copy_path.write_bytes(a.read_bytes())

    cache_files_before = list((tmp_path / "cache").glob("*.json"))
    extractor.extract(str(copy_path))
    cache_files_after = list((tmp_path / "cache").glob("*.json"))

    assert len(cache_files_before) == len(cache_files_after) == 1


def test_extract_batch_skips_unreadable_samples(tmp_path, extractor):
    from cv_auditor.dataset.dataset_loader import DatasetLoader

    (tmp_path / "images").mkdir(parents=True)
    make_image(tmp_path / "images" / "good.jpg", color=(200, 60, 60))
    (tmp_path / "images" / "bad.jpg").write_bytes(b"not an image")

    index = DatasetLoader(str(tmp_path)).load()
    vectors = extractor.extract_batch(index.samples)

    assert "good.jpg" in vectors
    assert "bad.jpg" not in vectors


def test_caching_disabled_when_cache_dir_is_none(tmp_path):
    extractor_no_cache = EmbeddingExtractor(cache_dir=None)
    a = tmp_path / "a.jpg"
    make_image(a, color=(200, 60, 60))
    v1 = extractor_no_cache.extract(str(a))
    v2 = extractor_no_cache.extract(str(a))
    assert np.allclose(v1, v2)  # still deterministic, just recomputed


# --- handcrafted fallback embedding, tested directly (deterministic, no torch needed) ---

def test_handcrafted_embedding_is_l2_normalized(tmp_path):
    a = tmp_path / "a.jpg"
    make_image(a, color=(200, 60, 60))
    vector = _handcrafted_embedding(str(a))
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-5


def test_handcrafted_embedding_distinguishes_different_colors(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    make_image(a, color=(220, 30, 30), size=128)
    make_image(b, color=(30, 30, 220), size=128)

    va = _handcrafted_embedding(str(a))
    vb = _handcrafted_embedding(str(b))
    same = _handcrafted_embedding(str(a))

    cos_same = float(va @ same)
    cos_diff = float(va @ vb)
    assert cos_same > cos_diff
