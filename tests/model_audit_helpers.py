"""Shared helpers for Phase 3 (model audit) tests.

Keeps unit tests for comparison/suspicion logic fast and independent of
real (large, ~75MB) model checkpoints by constructing small synthetic
``BehaviorProfile``-shaped data directly. Real-checkpoint, real-inference
coverage lives in ``test_model_auditor_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch

from cv_auditor.model_audit.behavior import BehaviorProfile


def make_prediction(class_id: int, confidence: float, bbox=(10.0, 10.0, 50.0, 50.0)) -> dict:
    return {
        "class": f"class_{class_id}",
        "class_id": class_id,
        "confidence": confidence,
        "bbox": list(bbox),
    }


def make_image_result(image: str, predictions: List[dict]) -> dict:
    return {
        "image": image,
        "image_path": f"/fake/{image}",
        "image_sha256": "0" * 64,
        "width": 300,
        "height": 300,
        "model": "fake_model",
        "inference_timestamp": "2026-01-01T00:00:00+00:00",
        "predictions": predictions,
    }


def make_profile(model_name: str, weights_sha256: str, images_and_predictions: dict) -> BehaviorProfile:
    """``images_and_predictions``: {image_filename: [prediction_dict, ...]}"""
    return BehaviorProfile(
        model_name=model_name,
        architecture="fasterrcnn_mobilenet_v3_large_320_fpn",
        weights_sha256=weights_sha256,
        test_images_dir="/fake/test_images",
        confidence_threshold=0.5,
        generated_at="2026-01-01T00:00:00+00:00",
        image_results=[make_image_result(name, preds) for name, preds in images_and_predictions.items()],
    )


def perturb_bias_checkpoint(
    source_checkpoint: str,
    dest_checkpoint: str,
    bias_key: str = "roi_heads.box_predictor.cls_score.bias",
    class_id: int = 5,
    boost: float = 40.0,
) -> None:
    """Copy a checkpoint and boost one classification bias, in-place on disk.

    Mirrors ``scripts/generate_model_audit_assets.py``'s
    ``candidate_backdoor_like`` construction, but writes to an arbitrary
    (typically tmp_path) destination so tests don't depend on, or need
    to commit, large pre-generated checkpoint files.
    """
    state = torch.load(source_checkpoint, map_location="cpu")
    state = {k: v.clone() for k, v in state.items()}
    state[bias_key][class_id] += boost
    Path(dest_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dest_checkpoint)


def add_noise_checkpoint(source_checkpoint: str, dest_checkpoint: str, std: float = 1e-4, seed: int = 1234) -> None:
    """Copy a checkpoint with tiny Gaussian noise added to every float tensor."""
    torch.manual_seed(seed)
    state = torch.load(source_checkpoint, map_location="cpu")
    noisy = {}
    for k, v in state.items():
        noisy[k] = v + torch.randn_like(v) * std if torch.is_floating_point(v) else v.clone()
    Path(dest_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(noisy, dest_checkpoint)
