"""
Tests for behavior.py -- uses Phase 1's real, shipped sample checkpoint
and sample images so this exercises genuine inference (via Phase 1's
InferenceEngine/ModelLoader), not mocked output.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from cv_auditor.config import PipelineConfig, COCO_INSTANCE_CATEGORY_NAMES
from cv_auditor.model_loader import ModelLoader
from cv_auditor.model_audit.behavior import BehaviorProfile, generate_behavior_profile, load_test_images

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MODEL = ROOT / "data" / "sample_model" / "sample_model.pth"
SAMPLE_IMAGES = ROOT / "data" / "sample_images"

pytestmark = pytest.mark.skipif(
    not SAMPLE_MODEL.exists() or not SAMPLE_IMAGES.is_dir(),
    reason="Phase 1 sample assets not found; run scripts/generate_sample_assets.py first.",
)


@pytest.fixture(scope="module")
def loaded_model():
    cfg = PipelineConfig(
        model_path=str(SAMPLE_MODEL),
        confidence_threshold=0.0,
        detector_score_thresh=0.0,  # see PipelineConfig docstring: needed for untrained weights
        model_name="test_reference",
    )
    return ModelLoader(cfg).load(), cfg


def test_generate_behavior_profile_runs_real_inference(loaded_model):
    (model, metadata), cfg = loaded_model
    images = load_test_images(str(SAMPLE_IMAGES))

    profile = generate_behavior_profile(
        model, metadata, images, COCO_INSTANCE_CATEGORY_NAMES,
        device=next(model.parameters()).device,
        confidence_threshold=cfg.confidence_threshold,
        test_images_dir=str(SAMPLE_IMAGES),
    )

    assert profile.model_name == "test_reference"
    assert profile.weights_sha256 == metadata.weights_sha256
    assert len(profile.image_results) == len(images)
    for image_result in profile.image_results:
        assert "predictions" in image_result
        assert isinstance(image_result["predictions"], list)


def test_behavior_profile_save_and_load_round_trip(loaded_model, tmp_path):
    (model, metadata), cfg = loaded_model
    images = load_test_images(str(SAMPLE_IMAGES))
    profile = generate_behavior_profile(
        model, metadata, images, COCO_INSTANCE_CATEGORY_NAMES,
        device=next(model.parameters()).device,
        confidence_threshold=cfg.confidence_threshold,
        test_images_dir=str(SAMPLE_IMAGES),
    )

    out = tmp_path / "baseline.json"
    profile.save(str(out))
    assert out.exists()

    loaded = BehaviorProfile.load(str(out))
    assert loaded.model_name == profile.model_name
    assert loaded.weights_sha256 == profile.weights_sha256
    assert len(loaded.image_results) == len(profile.image_results)


def test_load_test_images_reuses_phase1_image_loader():
    images = load_test_images(str(SAMPLE_IMAGES))
    assert len(images) >= 1
    for img in images:
        assert img.sha256  # Phase 1's LoadedImage always computes this


def test_two_independent_runs_of_the_same_model_produce_identical_profiles(loaded_model):
    """Determinism check: a model in eval() mode with no dropout/BN
    randomness should produce identical output run-to-run."""
    (model, metadata), cfg = loaded_model
    images = load_test_images(str(SAMPLE_IMAGES))

    profile_a = generate_behavior_profile(
        model, metadata, images, COCO_INSTANCE_CATEGORY_NAMES,
        device=next(model.parameters()).device,
        confidence_threshold=cfg.confidence_threshold,
    )
    profile_b = generate_behavior_profile(
        model, metadata, images, COCO_INSTANCE_CATEGORY_NAMES,
        device=next(model.parameters()).device,
        confidence_threshold=cfg.confidence_threshold,
    )
    def strip_timestamp(image_results):
        return [{k: v for k, v in r.items() if k != "inference_timestamp"} for r in image_results]

    assert strip_timestamp(profile_a.image_results) == strip_timestamp(profile_b.image_results)
