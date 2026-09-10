import torch

from cv_auditor.config import PipelineConfig
from cv_auditor.model_loader import ModelLoader
from cv_auditor.image_loader import ImageLoader
from cv_auditor.inference import InferenceEngine
from cv_auditor.result_formatter import ImageResult, DetectionResult

from .fixtures import SAMPLE_MODEL_PATH, SAMPLE_IMAGES_DIR, SAMPLE_ARCHITECTURE


def _build_engine(confidence_threshold: float = 0.0):
    config = PipelineConfig(
        model_path=str(SAMPLE_MODEL_PATH),
        architecture=SAMPLE_ARCHITECTURE,
        confidence_threshold=confidence_threshold,
    )
    model, metadata = ModelLoader(config).load()
    engine = InferenceEngine(
        model=model,
        model_name=config.model_name,
        class_names=config.class_names,
        device=torch.device("cpu"),
        confidence_threshold=confidence_threshold,
    )
    return engine, config


def test_inference_runs_on_single_image_and_returns_image_result():
    engine, _ = _build_engine()
    image = ImageLoader(str(SAMPLE_IMAGES_DIR)).load_all()[0]

    result = engine.run_one(image)

    assert isinstance(result, ImageResult)
    assert result.image == image.filename
    assert result.model == "reference_model"
    assert isinstance(result.predictions, list)
    for pred in result.predictions:
        assert isinstance(pred, DetectionResult)
        assert 0.0 <= pred.confidence <= 1.0
        assert len(pred.bbox) == 4


def test_inference_runs_on_batch_of_images():
    engine, _ = _build_engine()
    images = ImageLoader(str(SAMPLE_IMAGES_DIR)).load_all()

    results = engine.run_batch(images)

    assert len(results) == len(images)
    assert all(isinstance(r, ImageResult) for r in results)


def test_confidence_threshold_filters_low_score_predictions():
    # A threshold of exactly 1.0 keeps only perfect-confidence predictions,
    # which real detectors essentially never produce, so this should
    # filter out everything regardless of model output.
    engine, _ = _build_engine(confidence_threshold=1.0)
    image = ImageLoader(str(SAMPLE_IMAGES_DIR)).load_all()[0]

    result = engine.run_one(image)

    assert result.predictions == []
