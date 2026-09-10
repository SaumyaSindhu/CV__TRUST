"""Shared helpers/fixtures for the Phase 4 (provenance) tests.

Everything here is small and synthetic: tiny byte "files" standing in for
an image and a model, plus a fully deterministic record payload. No torch,
no network -- real inference is exercised only in
``test_provenance_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cv_auditor.provenance import build_payload, protect_payload
from cv_auditor.utils import sha256_of_file

FAKE_IMAGE_BYTES = b"\xff\xd8\xff synthetic-not-a-real-jpeg image bytes \x00\x01"
FAKE_MODEL_BYTES = b"PK synthetic-not-a-real-checkpoint model bytes \x00\x02"


def write_fixture_files(tmp_path: Path) -> Tuple[Path, Path]:
    """Write a tiny fake 'image' and 'model' file; return their paths."""
    image_path = tmp_path / "image_001.jpg"
    model_path = tmp_path / "model.pth"
    image_path.write_bytes(FAKE_IMAGE_BYTES)
    model_path.write_bytes(FAKE_MODEL_BYTES)
    return image_path, model_path


def make_payload(
    tmp_path: Path,
    *,
    prediction_class: str = "airplane",
    sequence_number: int = 0,
    record_id: str = "record-0001",
    nonce: str = "aa" * 16,
    files: Optional[Tuple[Path, Path]] = None,
) -> Dict[str, Any]:
    """Deterministic synthetic payload bound to real (tiny) fixture files."""
    image_path, model_path = files if files is not None else write_fixture_files(tmp_path)
    return build_payload(
        input_info={
            "image": image_path.name,
            "image_path": str(image_path),
            "image_sha256": sha256_of_file(str(image_path)),
            "width": 32,
            "height": 24,
        },
        model_info={
            "model_name": "synthetic_model",
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "source": "checkpoint",
            "weights_path": str(model_path),
            "model_sha256": sha256_of_file(str(model_path)),
            "num_parameters": 12345,
            "num_classes": 91,
            "framework": "pytorch",
            "framework_version": "0.0-test",
        },
        preprocessing_config={
            "color_space": "RGB",
            "tensor_dtype": "float32",
            "value_range": [0.0, 1.0],
            "resize": "model-internal",
        },
        inference_config={
            "confidence_threshold": 0.5,
            "detector_score_thresh": None,
            "device": "cpu",
        },
        output={
            "predictions": [
                {
                    "class": prediction_class,
                    "class_id": 5,
                    "confidence": 0.97,
                    "bbox": [10.0, 20.0, 110.0, 220.0],
                }
            ]
        },
        sequence_number=sequence_number,
        record_id=record_id,
        created_at="2026-09-01T00:00:00+00:00",
        nonce=nonce,
        inference_timestamp="2026-09-01T00:00:01+00:00",
        software={"cv_auditor": "0.1.0-test", "python": "3.x-test",
                  "torch": None, "torchvision": None},
    )


def make_record(tmp_path: Path, **kwargs) -> Dict[str, Any]:
    """Protected synthetic record (see ``make_payload`` for kwargs)."""
    return protect_payload(make_payload(tmp_path, **kwargs))


def finding_codes(report) -> set:
    """The set of finding codes present in a VerificationReport."""
    return {f.code for f in report.findings}
