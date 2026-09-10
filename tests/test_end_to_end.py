"""
test_end_to_end.py
-------------------
Exercises the full Phase 1 pipeline exactly as ``demo.py`` does:

    load model -> load images -> run inference -> format -> save -> reload

This is the single most important test in the suite: if this passes, the
Phase 1 deliverable ("model + image -> structured, saved result") works.
"""

import torch

from cv_auditor import (
    PipelineConfig, ModelLoader, ImageLoader, InferenceEngine, ResultFormatter,
)

from .fixtures import SAMPLE_MODEL_PATH, SAMPLE_IMAGES_DIR, SAMPLE_ARCHITECTURE


def test_full_pipeline_end_to_end(tmp_path):
    output_path = tmp_path / "e2e_results.json"

    config = PipelineConfig(
        model_path=str(SAMPLE_MODEL_PATH),
        architecture=SAMPLE_ARCHITECTURE,
        model_name="e2e_test_model",
        confidence_threshold=0.0,
        input_dir=str(SAMPLE_IMAGES_DIR),
        output_path=str(output_path),
    )

    # 1. Model loads successfully.
    model, model_metadata = ModelLoader(config).load()
    assert model is not None

    # 2. Images load successfully.
    images = ImageLoader(config.input_dir).load_all()
    assert len(images) > 0

    # 3. Inference runs successfully and returns the expected structure.
    engine = InferenceEngine(
        model=model,
        model_name=config.model_name,
        class_names=config.class_names,
        device=torch.device("cpu"),
        confidence_threshold=config.confidence_threshold,
    )
    image_results = engine.run_batch(images)
    assert len(image_results) == len(images)

    # 4. Results save and reload correctly.
    formatter = ResultFormatter(
        model_metadata=model_metadata.to_dict(), config=config.to_dict(),
    )
    for r in image_results:
        formatter.add_result(r)
    formatter.save(str(output_path))

    assert output_path.exists()
    reloaded = ResultFormatter.load(str(output_path))
    assert reloaded["run_metadata"]["model"]["model_name"] == "e2e_test_model"
    assert len(reloaded["results"]) == len(images)
    for image_result in reloaded["results"]:
        assert "predictions" in image_result
        for pred in image_result["predictions"]:
            assert set(["class", "class_id", "confidence", "bbox"]).issubset(pred.keys())
