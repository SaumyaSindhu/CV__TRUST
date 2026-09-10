import json

from cv_auditor.result_formatter import (
    ResultFormatter, ImageResult, DetectionResult,
)


def _sample_image_result() -> ImageResult:
    return ImageResult(
        image="image_001.jpg",
        image_path="data/sample_images/image_001.jpg",
        image_sha256="a" * 64,
        width=640,
        height=480,
        model="reference_model",
        inference_timestamp="2026-09-01T00:00:00+00:00",
        predictions=[
            DetectionResult(class_name="car", class_id=3, confidence=0.94,
                             bbox=[120.0, 80.0, 450.0, 300.0]),
        ],
    )


def test_result_expected_structure():
    result = _sample_image_result()
    d = result.to_dict()

    assert d["image"] == "image_001.jpg"
    assert "predictions" in d
    pred = d["predictions"][0]
    assert set(["class", "class_id", "confidence", "bbox"]).issubset(pred.keys())
    assert pred["class"] == "car"
    assert len(pred["bbox"]) == 4


def test_save_and_reload_results_round_trip(tmp_path):
    formatter = ResultFormatter(
        model_metadata={"model_name": "reference_model"},
        config={"architecture": "fasterrcnn_mobilenet_v3_large_320_fpn"},
    )
    formatter.add_result(_sample_image_result())

    out_path = tmp_path / "results.json"
    saved_path = formatter.save(str(out_path))

    assert out_path.exists()

    reloaded = ResultFormatter.load(saved_path)
    assert "run_metadata" in reloaded
    assert "results" in reloaded
    assert len(reloaded["results"]) == 1
    assert reloaded["results"][0]["image"] == "image_001.jpg"
    assert reloaded["results"][0]["predictions"][0]["class"] == "car"

    # Sanity: file on disk is valid, well-formed JSON.
    with open(saved_path) as f:
        json.load(f)


def test_load_missing_file_raises_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    try:
        ResultFormatter.load(str(missing))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
